#!/usr/bin/env python

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Create a balanced stratified sample from RoboSpatial or RefSpatial datasets.

Supports:
- RoboSpatial: Uses VeRL's task type detection from question text
- RefSpatial: Uses qa_type field from extra_info (depth, spatial, object, vacant)
- Merging multiple input files before sampling
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys
import os
import ast
from typing import List, Tuple

# Add VeRL to path for imports
import verl
verl_root = os.path.dirname(os.path.dirname(verl.__file__))
sys.path.insert(0, verl_root)

# Import VeRL's task type detection to properly classify questions
from verl.utils.reward_score.robos_vision import _detect_task_type


def _parse_points(text: str) -> List[Tuple[float, float]]:
    """Extract a list of (x, y) floats from text by evaluating it as a Python literal."""
    try:
        parsed = ast.literal_eval(text)
        # Handle both list of points and nested list of points
        if isinstance(parsed, list):
            if len(parsed) > 0 and isinstance(parsed[0], list):
                # Flatten nested list [[points]] -> [points]
                parsed = parsed[0]
            # Convert to list of tuples
            return [tuple(p) if isinstance(p, (list, tuple)) else p for p in parsed]
        return []
    except (ValueError, SyntaxError):
        return []


def _cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """2-D cross product (OA × OB). Positive if OAB makes a counter-clockwise turn."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return the vertices of the convex hull in counter-clockwise order.
    
    Uses Andrew's monotone chain algorithm (O(n log n)).
    """
    if len(points) <= 1:
        return list(points)
    
    points_sorted = sorted(points)
    
    # Build lower hull
    lower = []
    for p in points_sorted:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points_sorted):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Remove last point of each half because it's repeated at the beginning of the other half
    return lower[:-1] + upper[:-1]


def _polygon_area_shoelace(points: List[Tuple[float, float]]) -> float:
    """Calculate the area of a polygon using the shoelace formula."""
    if len(points) < 3:
        return 0.0
    
    area = 0.0
    n = len(points)
    
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
    
    return abs(area) / 2.0


def compute_convex_hull_area(ground_truth: str) -> float:
    """Compute the area of the convex hull of points in the ground truth string.
    
    Returns 0.0 if no points are found or if there are fewer than 3 points.
    """
    points = _parse_points(ground_truth)
    if len(points) < 3:
        return 0.0
    
    hull = _convex_hull(points)
    return _polygon_area_shoelace(hull)


def create_balanced_subset(input_path, output_path, n_samples=4000, seed=42, use_qa_type_field=False,
                          exclude_worlds_perspective=False, min_point_area=None, include_question_types=None,
                          interactive=False, yesno_yes_fraction=0.5, pairwise_compatibility_yes_fraction=None,
                          pairwise_positional_yes_fraction=None, add_to=None):
    """
    Create a balanced subset with equal numbers of each question type.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        n_samples: Total number of samples (will be split equally among question types)
        seed: Random seed for reproducibility
        use_qa_type_field: If True, use qa_type from extra_info instead of detecting from question text
        exclude_worlds_perspective: If True, exclude questions containing "world's perspective"
        min_point_area: If set, exclude pointing questions where convex hull area < this value
        include_question_types: If set, only include specified question types (e.g., ['point', 'yesno'])
        interactive: If True, prompt user for custom weights for each question type
        yesno_yes_fraction: Fraction of "yes" answers for 'yesno' type questions (default: 0.5)
        pairwise_compatibility_yes_fraction: Fraction of "yes" answers for 'pairwise_compatibility' type (default: same as yesno_yes_fraction)
        pairwise_positional_yes_fraction: Fraction of "yes" answers for 'pairwise_positional' type (default: same as yesno_yes_fraction)
        add_to: If set, path to existing parquet file; will exclude examples already in that dataset
    """
    np.random.seed(seed)
    
    print(f"Loading dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  Total rows: {len(df):,}")
    
    # Load existing dataset if provided and exclude overlapping examples
    if add_to is not None:
        print(f"Loading existing dataset from {add_to}...")
        existing_df = pd.read_parquet(add_to)
        print(f"  Existing dataset rows: {len(existing_df):,}")
        
        # Create unique identifiers for both datasets
        # Use question text + image path as the unique identifier
        def create_identifier(row):
            question = row['extra_info'].get('question', '')
            image_path = row['extra_info'].get('image_path', '')
            return f"{question}|||{image_path}"
        
        existing_ids = set(existing_df.apply(create_identifier, axis=1))
        current_ids = df.apply(create_identifier, axis=1)
        
        # Filter out examples that exist in the existing dataset
        mask = [cid not in existing_ids for cid in current_ids]
        df = df[mask].reset_index(drop=True)
        
        excluded = len(current_ids) - len(df)
        print(f"  Excluded {excluded:,} examples already in existing dataset ({len(df):,} remaining)")
    
    # Apply filters
    initial_count = len(df)
    
    # Filter 0: Include only specific question types
    if include_question_types is not None:
        questions = df['extra_info'].apply(lambda x: x.get('question', '')).tolist()
        question_types = [_detect_task_type(q) for q in questions]
        mask = [qt in include_question_types for qt in question_types]
        df = df[mask].reset_index(drop=True)
        excluded = initial_count - len(df)
        print(f"  Filtered to question types {include_question_types}: excluded {excluded:,} ({len(df):,} remaining)")
        initial_count = len(df)
    
    # Filter 1: Exclude questions with "world's perspective"
    if exclude_worlds_perspective:
        questions = df['extra_info'].apply(lambda x: x.get('question', '')).tolist()
        mask = [("world's perspective" not in q.lower()) for q in questions]
        df = df[mask].reset_index(drop=True)
        excluded = initial_count - len(df)
        print(f"  Excluded {excluded:,} questions containing \"world's perspective\" ({len(df):,} remaining)")
        initial_count = len(df)
    
    # Filter 2: Exclude pointing questions with small convex hull area
    if min_point_area is not None:
        print(f"  Filtering pointing questions with convex hull area < {min_point_area}")
        questions = df['extra_info'].apply(lambda x: x.get('question', '')).tolist()
        ground_truths = df['reward_model'].apply(lambda x: x.get('ground_truth', '')).tolist()
        
        mask = []
        for q, gt in zip(questions, ground_truths):
            # Detect if this is a pointing question
            task_type = _detect_task_type(q)
            if task_type == 'point':
                # Compute convex hull area
                area = compute_convex_hull_area(gt)
                # Keep if area >= threshold
                mask.append(area >= min_point_area)
            else:
                # Keep non-pointing questions
                mask.append(True)
        
        df = df[mask].reset_index(drop=True)
        excluded = initial_count - len(df)
        print(f"  Excluded {excluded:,} pointing questions with small area ({len(df):,} remaining)")
    
    if use_qa_type_field:
        # Use qa_type field from extra_info
        print("  Using qa_type field from extra_info")
        question_types = df['extra_info'].apply(lambda x: x.get('qa_type', 'unknown')).tolist()
        
        # Group indices by qa_type
        type_to_indices = {}
        for i, qt in enumerate(question_types):
            if qt not in type_to_indices:
                type_to_indices[qt] = []
            type_to_indices[qt].append(i)
        
        # Print counts for each type
        for qtype, indices in sorted(type_to_indices.items()):
            print(f"  {qtype}: {len(indices):,}")
    else:
        # Identify question types using VeRL's detection (based on question text)
        print("  Detecting question types from question text")
        questions = df['extra_info'].apply(lambda x: x.get('question', '')).tolist()
        question_types = [_detect_task_type(q) for q in questions]
        
        # Group indices by question type
        yesno_indices = [i for i, qt in enumerate(question_types) if qt == 'yesno']
        point_indices = [i for i, qt in enumerate(question_types) if qt == 'point']
        bbox_indices = [i for i, qt in enumerate(question_types) if qt == 'bounding_box']
        
        type_to_indices = {}
        if yesno_indices:
            type_to_indices['yesno'] = yesno_indices
        if point_indices:
            type_to_indices['point'] = point_indices
        if bbox_indices:
            type_to_indices['bounding_box'] = bbox_indices
        
        print(f"  Yes/No questions: {len(yesno_indices):,}")
        print(f"  Point questions: {len(point_indices):,}")
        print(f"  Bounding Box questions: {len(bbox_indices):,}")
    
    # Filter out empty types
    if not type_to_indices:
        raise ValueError("No questions found in dataset!")
    
    # Determine weights for each question type
    available_types = sorted(type_to_indices.keys())
    
    if interactive:
        # Interactive mode: prompt user for weights
        print("\n" + "="*60)
        print("AVAILABLE QUESTION TYPES:")
        print("="*60)
        for qtype in available_types:
            count = len(type_to_indices[qtype])
            print(f"  {qtype}: {count:,} questions available")
        
        print("\n" + "="*60)
        print("Please enter weights for each question type.")
        print("Weights do not need to sum to 1 (they will be normalized).")
        print("="*60)
        
        weights = {}
        for qtype in available_types:
            while True:
                try:
                    weight = float(input(f"Weight for '{qtype}': "))
                    if weight < 0:
                        print("  Weight must be non-negative. Try again.")
                        continue
                    weights[qtype] = weight
                    break
                except ValueError:
                    print("  Invalid input. Please enter a number.")
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("All weights cannot be zero!")
        
        normalized_weights = {qtype: w / total_weight for qtype, w in weights.items()}
        
        print("\n" + "="*60)
        print("NORMALIZED WEIGHTS:")
        for qtype in available_types:
            print(f"  {qtype}: {normalized_weights[qtype]:.4f} ({weights[qtype]} → {normalized_weights[qtype]:.4f})")
        print("="*60 + "\n")
        
        # Prompt for yes/no ratio for each yes/no question type
        yesno_types = {'yesno', 'pairwise_compatibility', 'pairwise_positional'}
        yesno_types_present = [qt for qt in available_types if qt in yesno_types]
        
        yesno_yes_fractions = {}
        if yesno_types_present:
            print("\n" + "="*60)
            print("YES/NO ANSWER RATIOS")
            print("="*60)
            
            for qtype in yesno_types_present:
                print(f"\nFor question type: {qtype}")
                while True:
                    try:
                        fraction = float(input(f"  Fraction of 'yes' answers (0.0 to 1.0): "))
                        if 0 <= fraction <= 1:
                            yesno_yes_fractions[qtype] = fraction
                            print(f"    'yes': {fraction:.2%}, 'no': {1 - fraction:.2%}")
                            break
                        else:
                            print("    Fraction must be between 0 and 1. Try again.")
                    except ValueError:
                        print("    Invalid input. Please enter a number between 0 and 1.")
            print("="*60 + "\n")
        
        # For non-yesno types or if not set, use default
        if not yesno_yes_fractions:
            # Create dict with default value for any yesno type
            yesno_yes_fractions = {qt: yesno_yes_fraction for qt in yesno_types}
    else:
        # Non-interactive mode: equal weights
        normalized_weights = {qtype: 1.0 / len(available_types) for qtype in available_types}
        # Create dict with specific or default values for each yesno type
        yesno_types = {'yesno', 'pairwise_compatibility', 'pairwise_positional'}
        yesno_yes_fractions = {
            'yesno': yesno_yes_fraction,
            'pairwise_compatibility': pairwise_compatibility_yes_fraction if pairwise_compatibility_yes_fraction is not None else yesno_yes_fraction,
            'pairwise_positional': pairwise_positional_yes_fraction if pairwise_positional_yes_fraction is not None else yesno_yes_fraction
        }
    
    # Sample according to weights
    sampled_indices = []
    actual_counts = {}
    
    # Define yes/no question types
    yesno_types = {'yesno', 'pairwise_compatibility', 'pairwise_positional'}
    
    for qtype in available_types:
        indices = type_to_indices[qtype]
        n_to_sample = int(n_samples * normalized_weights[qtype])
        n_to_sample = min(n_to_sample, len(indices))
        
        if n_to_sample > 0:
            # Special handling for yes/no questions to balance yes/no answers
            if qtype in yesno_types:
                # Split indices by ground truth
                ground_truths = df.iloc[indices]['reward_model'].apply(
                    lambda x: x.get('ground_truth', '').strip().lower()
                ).tolist()
                
                yes_indices = [idx for idx, gt in zip(indices, ground_truths) if gt == 'yes']
                no_indices = [idx for idx, gt in zip(indices, ground_truths) if gt == 'no']
                
                print(f"  {qtype} breakdown: {len(yes_indices):,} 'yes', {len(no_indices):,} 'no'")
                
                # Calculate how many of each to sample using the specific fraction for this qtype
                yes_fraction = yesno_yes_fractions.get(qtype, 0.5)
                n_yes = int(n_to_sample * yes_fraction)
                n_no = n_to_sample - n_yes
                
                # Adjust if we don't have enough of either type
                n_yes = min(n_yes, len(yes_indices))
                n_no = min(n_no, len(no_indices))
                
                # Sample from each
                sampled_yes = np.random.choice(yes_indices, n_yes, replace=False) if n_yes > 0 else []
                sampled_no = np.random.choice(no_indices, n_no, replace=False) if n_no > 0 else []
                
                sampled = list(sampled_yes) + list(sampled_no)
                sampled_indices.extend(sampled)
                actual_counts[qtype] = len(sampled)
                actual_counts[f'{qtype}_yes'] = n_yes
                actual_counts[f'{qtype}_no'] = n_no
            else:
                sampled = np.random.choice(indices, n_to_sample, replace=False)
                sampled_indices.extend(sampled)
                actual_counts[qtype] = n_to_sample
    
    # Shuffle combined indices
    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)
    
    # Create subset
    subset_df = df.iloc[sampled_indices].reset_index(drop=True)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_parquet(output_path)
    
    print(f"\n✅ Created balanced subset:")
    print(f"  Output: {output_path}")
    print(f"  Total samples: {len(subset_df):,}")
    for qtype, count in actual_counts.items():
        if qtype.endswith('_yes') or qtype.endswith('_no'):
            # Show yes/no breakdown with indentation
            answer = qtype.split('_')[-1]  # Extract 'yes' or 'no'
            print(f"    {answer}: {count:,}")
        else:
            print(f"  {qtype}: {count:,}")
    print(f"  Seed: {seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create balanced subset from RoboSpatial dataset")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--n-samples", type=int, default=4000, help="Total number of samples (default: 4000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--use-qa-type-field", action="store_true", 
                        help="Use qa_type field from extra_info instead of detecting from question text")
    parser.add_argument("--exclude-worlds-perspective", action="store_true",
                        help="Exclude questions containing \"world's perspective\"")
    parser.add_argument("--min-point-area", type=float, default=0.004,
                        help="Minimum convex hull area for pointing questions (default: 0.004)")
    parser.add_argument("--include-question-types", nargs='+', default=None,
                        help="Only include specific question types (e.g., point yesno bounding_box)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactively prompt for custom weights for each question type")
    parser.add_argument("--yesno-yes-fraction", type=float, default=0.5,
                        help="Fraction of 'yes' answers for 'yesno' type questions (default: 0.5)")
    parser.add_argument("--pairwise-compatibility-yes-fraction", type=float, default=None,
                        help="Fraction of 'yes' answers for 'pairwise_compatibility' type (default: same as --yesno-yes-fraction)")
    parser.add_argument("--pairwise-positional-yes-fraction", type=float, default=None,
                        help="Fraction of 'yes' answers for 'pairwise_positional' type (default: same as --yesno-yes-fraction)")
    parser.add_argument("--add-to", type=str, default=None,
                        help="Path to existing parquet file; will exclude examples already in that dataset")
    
    args = parser.parse_args()
    create_balanced_subset(args.input, args.output, args.n_samples, args.seed, args.use_qa_type_field,
                          args.exclude_worlds_perspective, args.min_point_area, args.include_question_types,
                          args.interactive, args.yesno_yes_fraction, args.pairwise_compatibility_yes_fraction,
                          args.pairwise_positional_yes_fraction, args.add_to)
