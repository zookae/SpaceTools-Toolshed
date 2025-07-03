# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Regenerate training data from an existing checkpoint file.

This script takes a checkpoint from a previous evaluation run and regenerates
the training data in ShareGPT format. This is useful when you:
- Want to change training data filtering criteria
- Need to regenerate training data after a failed run
- Want to extract training data from multiple checkpoints

The script requires the original dataset file to load images, as images are
not stored in the checkpoint to keep file sizes manageable.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add toolshed to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evals.robospatial import RoboSpatialEvaluator
from evals.refspatial import RefSpatialEvaluator
from toolshed.formats.sharegpt_simple import build_training_example

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate training data from checkpoint file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate training data from checkpoint (RoboSpatial)
  python -m evals.regenerate_training_data \\
    --checkpoint ./results/checkpoint.json \\
    --dataset data/robospatial_home/test.parquet \\
    --dataset-type robospatial \\
    --output ./regenerated_training_data

  # Regenerate training data from checkpoint (RefSpatial)
  python -m evals.regenerate_training_data \\
    --checkpoint ./results/checkpoint.json \\
    --dataset data/refspatial_multiturn/depth.parquet \\
    --dataset-type refspatial \\
    --output ./regenerated_training_data

  # Regenerate with different filtering criteria
  python -m evals.regenerate_training_data \\
    --checkpoint ./results/checkpoint.json \\
    --dataset data/robospatial_home/test.parquet \\
    --dataset-type robospatial \\
    --output ./training_data_filtered \\
    --min-score 0.8
        """
    )
    
    parser.add_argument('--checkpoint', required=True,
                       help='Path to checkpoint.json file')
    parser.add_argument('--dataset', required=True,
                       help='Path to original dataset file (needed for images)')
    parser.add_argument('--dataset-type', required=True, choices=['robospatial', 'refspatial'],
                       help='Type of dataset')
    parser.add_argument('--output', required=True,
                       help='Output directory for training data')
    parser.add_argument('--min-score', type=float,
                       help='Minimum score threshold (overrides dataset-specific thresholds)')
    parser.add_argument('--force', action='store_true',
                       help='Regenerate all training data, ignoring tracking state')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    
    if 'detailed_results' not in data:
        raise ValueError(f"Invalid checkpoint format: missing 'detailed_results' field")
    
    logger.info(f"Loaded checkpoint with {len(data['detailed_results'])} results")
    return data


def get_example_index_robospatial(row_dict: Dict, pandas_idx: int) -> int:
    """
    Extract index from RoboSpatial dataset row.
    
    RoboSpatial stores index in row['extra_info']['index'].
    """
    if 'extra_info' not in row_dict:
        raise ValueError(f"RoboSpatial row missing 'extra_info' field at pandas index {pandas_idx}")
    
    extra_info = row_dict['extra_info']
    if not isinstance(extra_info, dict):
        raise ValueError(f"RoboSpatial row['extra_info'] is not a dict at pandas index {pandas_idx}")
    
    if 'index' not in extra_info:
        raise ValueError(f"RoboSpatial row['extra_info'] missing 'index' field at pandas index {pandas_idx}")
    
    return extra_info['index']


def get_example_index_refspatial(row_dict: Dict, pandas_idx: int) -> int:
    """
    Extract index from RefSpatial dataset row.
    
    RefSpatial stores index in row['extra_info']['index'], same as RoboSpatial.
    """
    if 'extra_info' not in row_dict:
        raise ValueError(f"RefSpatial row missing 'extra_info' field at pandas index {pandas_idx}")
    
    extra_info = row_dict['extra_info']
    if not isinstance(extra_info, dict):
        raise ValueError(f"RefSpatial row['extra_info'] is not a dict at pandas index {pandas_idx}")
    
    if 'index' not in extra_info:
        raise ValueError(f"RefSpatial row['extra_info'] missing 'index' field at pandas index {pandas_idx}")
    
    return extra_info['index']


def load_dataset(dataset_path: str, dataset_type: str) -> Dict[int, Dict]:
    """
    Load dataset and create index mapping.
    
    Args:
        dataset_path: Path to dataset file
        dataset_type: Type of dataset (determines index extraction logic)
    
    Returns:
        Dict mapping dataset index to row data
    """
    import pandas as pd
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Load based on file extension
    if dataset_path.suffix == '.parquet':
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == '.json':
        df = pd.read_json(dataset_path)
    elif dataset_path.suffix == '.jsonl':
        df = pd.read_json(dataset_path, lines=True)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
    
    # Create index mapping using dataset-specific logic
    dataset_index = {}
    
    if dataset_type == 'robospatial':
        for pandas_idx, row in df.iterrows():
            row_dict = row.to_dict()
            key = get_example_index_robospatial(row_dict, pandas_idx)
            dataset_index[key] = row_dict
    elif dataset_type == 'refspatial':
        for pandas_idx, row in df.iterrows():
            row_dict = row.to_dict()
            key = get_example_index_refspatial(row_dict, pandas_idx)
            dataset_index[key] = row_dict
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Add index extraction logic for this dataset.")
    
    logger.info(f"Loaded dataset with {len(dataset_index)} examples")
    return dataset_index


def create_evaluator_for_filtering(dataset_type: str, min_score: Optional[float] = None):
    """
    Create a minimal evaluator instance for filtering logic.
    
    We don't need a full model, just the evaluator's filtering methods.
    """
    # Create a dummy model for the evaluator
    class DummyModel:
        pass
    
    if dataset_type == 'robospatial':
        evaluator = RoboSpatialEvaluator(model=DummyModel())
    elif dataset_type == 'refspatial':
        evaluator = RefSpatialEvaluator(model=DummyModel())
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Override filtering if custom min_score provided
    if min_score is not None:
        original_should_include = evaluator.should_include_in_training_data
        def custom_filter(result: Dict[str, Any]) -> bool:
            # Use custom threshold instead of dataset-specific
            return result.get('score', 0.0) >= min_score
        evaluator.should_include_in_training_data = custom_filter
    
    return evaluator


def regenerate_training_data(checkpoint_data: Dict, dataset_index: Dict, 
                             evaluator, output_dir: str, force: bool = False):
    """
    Regenerate training data from checkpoint and dataset.
    
    Args:
        checkpoint_data: Loaded checkpoint data
        dataset_index: Mapping from example index to dataset row
        evaluator: Evaluator instance for filtering logic
        output_dir: Output directory for training data
        force: If True, ignore tracking state and regenerate all
    """
    from tqdm import tqdm
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / "train_0.jsonl"
    images_dir = output_dir / "images_0"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Get tracking state if not forcing
    saved_indices = set()
    if not force and 'tracking_state' in checkpoint_data:
        saved_indices = set(checkpoint_data['tracking_state'].get('saved_training_indices', []))
        logger.info(f"Tracking state: {len(saved_indices)} examples were previously saved")
    
    # Get metadata for tool schemas
    metadata = checkpoint_data.get('metadata', {})
    
    # Process each result
    num_included = 0
    num_skipped = 0
    num_errors = 0
    
    detailed_results = checkpoint_data['detailed_results']
    pbar = tqdm(detailed_results, desc="Processing results", unit="example")
    
    for result in pbar:
        result_idx = result.get('index', -1)
        
        # Skip if already saved (unless force)
        if not force and result_idx in saved_indices:
            num_skipped += 1
            pbar.set_postfix({"included": num_included, "skipped": num_skipped, "errors": num_errors})
            continue
        
        # Skip if has errors or no score
        if 'error' in result or 'scoring_error' in result or result.get('score') is None:
            num_skipped += 1
            pbar.set_postfix({"included": num_included, "skipped": num_skipped, "errors": num_errors})
            continue
        
        # Check if should be included based on filtering criteria
        if not evaluator.should_include_in_training_data(result):
            num_skipped += 1
            pbar.set_postfix({"included": num_included, "skipped": num_skipped, "errors": num_errors})
            continue
        
        # Check if we have messages
        if 'messages' not in result or not result['messages']:
            logger.warning(f"No messages found for example {result_idx}, skipping")
            num_skipped += 1
            pbar.set_postfix({"included": num_included, "skipped": num_skipped, "errors": num_errors})
            continue
        
        # Get image from dataset
        image = None
        if result_idx in dataset_index:
            dataset_row = dataset_index[result_idx]
            try:
                image = evaluator.load_image_from_row(dataset_row)
            except Exception as e:
                logger.warning(f"Could not load image for example {result_idx}: {e}")
        else:
            logger.warning(f"Example {result_idx} not found in dataset index")
        
        # Save image and get path
        image_paths = []
        if image is not None:
            try:
                img_filename = f"s{num_included:03d}_img0000.png"
                img_full_path = images_dir / img_filename
                image.save(img_full_path)
                image_paths.append(f"images_0/{img_filename}")
            except Exception as e:
                logger.warning(f"Could not save image for example {result_idx}: {e}")
        
        # Build training example
        try:
            example = build_training_example(
                messages=result['messages'],
                task_id=f"{evaluator._get_dataset_name()}_0_{num_included:05d}",
                dataset_index=result_idx,
                score=result.get('score', 0.0),
                image_paths=image_paths,
                tool_schemas=None,  # Could extract from metadata if needed
                system_prompt_override=result.get('system_prompt_original'),
                ground_truth=result.get('ground_truth'),
                question_type=result.get('question_type')
            )
        except Exception as e:
            logger.error(f"Failed to build training example for {result_idx}: {e}")
            num_errors += 1
            pbar.set_postfix({"included": num_included, "skipped": num_skipped, "errors": num_errors})
            continue
        
        # Append to JSONL
        with open(train_file, 'a') as f:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        num_included += 1
        pbar.set_postfix({"included": num_included, "skipped": num_skipped, "errors": num_errors})
    
    # Convert JSONL to JSON format
    json_file = output_dir / "train_0.json"
    if train_file.exists():
        logger.info("Converting JSONL to JSON format...")
        try:
            examples = []
            with open(train_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        examples.append(json.loads(line))
            
            # Write as nicely indented JSON array
            with open(json_file, 'w') as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Converted to JSON format: {json_file} ({len(examples)} examples)")
        except Exception as e:
            logger.warning(f"Failed to convert training JSONL to JSON: {e}")
    
    logger.info(f"\nTraining data regeneration complete:")
    logger.info(f"  Included: {num_included}")
    logger.info(f"  Skipped: {num_skipped}")
    logger.info(f"  Errors: {num_errors}")
    logger.info(f"  JSONL: {train_file}")
    logger.info(f"  JSON: {json_file}")
    logger.info(f"  Images: {images_dir}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('toolshed').setLevel(logging.DEBUG)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint_data = load_checkpoint(args.checkpoint)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    dataset_index = load_dataset(args.dataset, args.dataset_type)
    
    # Create evaluator for filtering
    logger.info(f"Creating evaluator for {args.dataset_type}")
    evaluator = create_evaluator_for_filtering(args.dataset_type, args.min_score)
    
    # Regenerate training data
    logger.info(f"Regenerating training data to {args.output}")
    regenerate_training_data(
        checkpoint_data=checkpoint_data,
        dataset_index=dataset_index,
        evaluator=evaluator,
        output_dir=args.output,
        force=args.force
    )


if __name__ == '__main__':
    main()

