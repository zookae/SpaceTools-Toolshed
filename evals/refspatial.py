# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
RefSpatial dataset evaluator.

This module provides evaluation capabilities for the RefSpatial dataset,
which includes five subsets:
- depth: A/B depth comparison questions
- spatial: A/B spatial relationship questions  
- object: Point picking for object locations
- vacant: Point picking for vacant space locations
- bench_point: Benchmark pointing questions with mask-based ground truth

The evaluator uses the system prompt present in the dataset by default,
and only overrides it if explicitly requested via the system_prompt parameter.
"""

import logging
import sys
import os
from typing import Dict, Any, List, Optional

# Import the exact scoring function from VeRL
from verl.utils.reward_score.refspatial import compute_score
from verl.utils.reward_score.refspatial_bench import compute_score as compute_score_bench

# Import VeRL system prompts for point-picking tasks
# Add VeRL to path for imports
import verl
verl_root = os.path.dirname(os.path.dirname(verl.__file__))
sys.path.insert(0, verl_root)
from examples.data_preprocess.robospatial_multiturn_vision_v2 import (
    DEFAULT_SYSTEM_CONTENT_POINT,
)
from toolshed.prompts.spatial_reasoning import RS_SYSTEM_PROMPT_APPENDIX

from .common import BaseEvaluator
from .models import BaseModel

logger = logging.getLogger(__name__)


# System prompt for A/B answer questions (depth and spatial)
DEFAULT_SYSTEM_CONTENT_AB = (
    "You are an expert in 3D spatial reasoning for robotics. "
    "Given an image and a spatial reasoning question, follow this process:\n\n"
    "1. First, think about the reasoning process as an internal monologue.\n"
    "Your reasoning process MUST be enclosed within <think> </think> tags.\n"
    "2. After thinking, if you need to locate specific objects in the image, call the available vision detection tool exactly once for each type of object.\n"
    "3. When you receive a tool response that contains reasonable positions of the objects, use that information to continue your analysis.\n"
    "4. Once no further visual analysis or tool calls are needed, you MUST provide your final answer inside "
    "<answer> and </answer> tags without detailed explanations.\n\n"
    "Example answer format: <answer>A</answer>, or <answer>B</answer>."
)

# No-tools version for A/B questions
DEFAULT_SYSTEM_CONTENT_AB_NO_TOOLS = (
    "You are an expert in 3D spatial reasoning for robotics. "
    "Given an image and a spatial reasoning question, follow this process:\n\n"
    "1. First, think about the reasoning process as an internal monologue.\n"
    "Your reasoning process MUST be enclosed within <think> </think> tags.\n"
    "2. Once no further visual analysis is needed, you MUST provide your final answer inside "
    "<answer> and </answer> tags without detailed explanations.\n\n"
    "Example answer format: <answer>A</answer>, or <answer>B</answer>."
)

# No-tools version for point picking questions
DEFAULT_SYSTEM_CONTENT_POINT_NO_TOOLS = (
    "You are an expert in 3D spatial reasoning for robotics. "
    "Given an image and a spatial reasoning question asking you to detect specific objects or point to vacant space, follow this process:\n\n"
    "1. First, think about the reasoning process as an internal monologue.\n"
    "Your reasoning process MUST be enclosed within <think> </think> tags.\n"
    "2. Once no further visual analysis is needed, you MUST provide your final answer inside "
    "<answer> and </answer> tags without detailed explanations.\n\n"
    "Example answer format: <answer>[(0.006, 0.311)]</answer>"
)


class RefSpatialEvaluator(BaseEvaluator):
    """Evaluates models on the RefSpatial dataset."""
    
    def __init__(self, model: BaseModel,
                 save_conversations_dir: Optional[str] = None,
                 save_training_dir: Optional[str] = None,
                 think_output_mode: Optional[str] = None,
                 system_prompt: Optional[str] = None):
        """
        Initialize the evaluator with a model.
        
        Args:
            model: Model instance that implements the BaseModel interface
            save_conversations_dir: Optional directory to save conversations in VeRL format
            save_training_dir: Optional directory to save training data in ShareGPT format
            think_output_mode: None, "suppress", "force"
            system_prompt: "default", "expanded"
        """
        super().__init__(model, save_conversations_dir, save_training_dir)
        self.think_output_mode = think_output_mode
        self.system_prompt = system_prompt
        self.num_scoring_failures = 0  # Track cumulative scoring failures

    def parse_example(self, row: Dict) -> Dict:
        """
        Parse a RefSpatial dataset row into standardized format.
        
        Extracts question, ground truth, qa_type, and image from the RefSpatial schema.
        Uses system prompt from data by default, or overrides if requested.
        
        Args:
            row: Raw dataset row from parquet file
            
        Returns:
            Standardized example dict with question, ground_truth, image, system_prompt, metadata
        """
        # Extract RefSpatial-specific fields
        question = row['extra_info']['question']
        ground_truth = row['reward_model']['ground_truth']
        qa_type = row['extra_info'].get('qa_type', 'unknown')
        
        # Detect bench_point type if ground truth is a base64-encoded mask
        # Base64 strings are typically long and contain only alphanumeric chars + / + =
        if qa_type == 'unknown' and isinstance(ground_truth, str) and len(ground_truth) > 100:
            # Simple heuristic: if it looks like base64 (long string with base64 chars), treat as bench_point
            import string
            base64_chars = set(string.ascii_letters + string.digits + '+/=')
            if all(c in base64_chars for c in ground_truth.strip()):
                qa_type = 'bench_point'
        
        # Handle no system prompt case
        if self.system_prompt is None:
            system_prompt = None
            mod_system_prompt = None
        else:
            # Extract system prompt from data if present
            # Note: prompt is a list/array from parquet, avoid boolean check that causes ambiguous truth error
            system_prompt = None
            prompt_list = row.get('prompt')
            if prompt_list is not None:
                for msg in prompt_list:
                    if isinstance(msg, dict) and msg.get('role') == 'system':
                        system_prompt = msg.get('content', '')
                        break
            
            no_tools = (self.model.toolkit is None)

            # Fallback to default prompts if no system prompt in data
            if (not system_prompt) or no_tools:
                if qa_type in ['depth', 'spatial']:
                    print(f"WARNING:Using default system prompt for {qa_type} question")
                    # Use no-tools version if model has no toolkit
                    if no_tools:
                        print(f"Using no-tools version for {qa_type} question")
                        system_prompt = DEFAULT_SYSTEM_CONTENT_AB_NO_TOOLS
                    else:
                        print(f"Using tools version for {qa_type} question")
                        system_prompt = DEFAULT_SYSTEM_CONTENT_AB
                else:  # object, vacant, bench_point
                    print(f"WARNING:Using default system prompt for {qa_type} question")
                    # Use no-tools version if model has no toolkit
                    if no_tools:
                        print(f"Using no-tools version for {qa_type} question")
                        system_prompt = DEFAULT_SYSTEM_CONTENT_POINT_NO_TOOLS
                    else:
                        print(f"Using tools version for {qa_type} question")
                        system_prompt = DEFAULT_SYSTEM_CONTENT_POINT


            # Add expanded prompt appendix if requested
            if self.system_prompt == "expanded":
                system_prompt += RS_SYSTEM_PROMPT_APPENDIX

            # TODO: This was used for data gen, standardize.
            #system_prompt += """
            #You must ALWAYS use at least some of the available tools. Reason about the question and think of a strategy to answer it using the available tools.
            #Do NOT try to write code unless a tool is specifically provided to do so, the environment doesn't support it.
            #Note: the index_at and bounding_box tools have different arguments than the other tools, e.g. they don't take image_index.
            #Your <answer> and </answer> tags must ONLY include the final numerical answer or answer choice, without ANY additional text or explanation."""

            system_prompt.replace("tuple, i.e. [(x, y)]", "list containing a single tuple, i.e. [(x, y)]")

            # Apply think output mode modifications
            if self.think_output_mode == "suppress":
                mod_system_prompt = system_prompt.replace(
                    "Your reasoning process MUST be enclosed within <think> </think> tags.", 
                    "You should NEVER say ANYTHING to the user other than the final answer in <answer></answer> tags.")
            elif self.think_output_mode == "force":
                mod_system_prompt = system_prompt.replace(
                    "Your reasoning process MUST be enclosed within <think> </think> tags.", 
                    "You must ALWAYS say your full reasoning process in detail, documenting every step of how you are approaching the problem. The reasoning process MUST be enclosed within <think> </think> tags in every message to the user, including before any tool calls, and before the final answer.")
            else:
                mod_system_prompt = system_prompt
        
        # Load image using parent's utility method
        image = self.load_image_from_row(row)
        
        return {
            'question': question,
            'ground_truth': ground_truth,
            'image': image,
            'system_prompt': mod_system_prompt,
            'system_prompt_original': system_prompt,
            'metadata': {
                'original_question': question,
                'index': row['extra_info'].get('index', -1),
                'qa_type': qa_type
            }
        }
    
    def score_prediction(self, prediction: str, parsed_example: Dict, messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Score a model prediction using VeRL's RefSpatial compute_score function.
        
        Args:
            prediction: Model's output string
            parsed_example: Output from parse_example()
            messages: Optional conversation history (for parallel execution)
            
        Returns:
            Dict with 'score' (float between 0.0 and 1.0), and optionally 'scoring_error' if scoring failed
        """
        try:
            if self.system_prompt is None and "<answer>" not in prediction:
                print("No <answer> tag found in prediction, adding it")
                prediction = "<answer>" + prediction + "</answer>"

            qa_type = parsed_example['metadata']['qa_type']
            
            # Use bench scoring function for bench_point type
            if qa_type == 'bench_point':
                score_result = compute_score_bench(
                    solution_str=prediction,
                    ground_truth=parsed_example['ground_truth']
                )
                # Bench scorer returns a float directly
                return {'score': score_result}
            else:
                # Use standard scoring function for other types
                score_result = compute_score(
                    predict_str=prediction,
                    ground_truth=parsed_example['ground_truth'],
                    qa_type=qa_type if qa_type != 'unknown' else None
                )
                
                # Handle dict or float return value
                if isinstance(score_result, dict):
                    return {
                        'score': score_result.get('score', 0.0),
                        'acc': score_result.get('acc', 0.0),
                        'fmt': score_result.get('fmt', 0.0)
                    }
                else:
                    return {'score': score_result}
                
        except Exception as e:
            # Catch any errors during scoring
            self.num_scoring_failures += 1
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Scoring failed for example {parsed_example['metadata'].get('index', '?')}: {error_msg}")
            return {
                'score': 0.0,
                'scoring_error': error_msg
            }
    
    def get_result_metadata(self, parsed_example: Dict, score: float) -> Dict:
        """
        Extract RefSpatial-specific metadata for results.
        
        Args:
            parsed_example: Output from parse_example()
            score: Score from score_prediction()
            
        Returns:
            Dict with question_type, ground_truth, original question, and original system prompt
        """
        return {
            'question_type': parsed_example['metadata']['qa_type'],
            'ground_truth': parsed_example['ground_truth'],
            'question': parsed_example['metadata']['original_question'],
            'system_prompt_original': parsed_example['system_prompt_original']
        }
    
    def compute_custom_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Compute per-qa-type metrics for RefSpatial.
        
        Args:
            results: List of valid result dictionaries
            
        Returns:
            Dict with metrics_by_type breakdown
        """
        type_metrics = {}
        
        for qa_type in ['depth', 'spatial', 'object', 'vacant', 'bench_point']:
            type_results = [r for r in results if r.get('question_type') == qa_type]
            if type_results:
                type_scores = [r['score'] for r in type_results]
                type_metrics[qa_type] = {
                    'count': len(type_results),
                    'average_score': sum(type_scores) / len(type_scores),
                    'accuracy': sum(1 for s in type_scores if s > 0.5) / len(type_scores)
                }
        
        return {'metrics_by_type': type_metrics}
    
    def should_include_in_training_data(self, result: Dict[str, Any]) -> bool:
        """
        Determine whether a RefSpatial result should be included in training data.
        
        Applies dataset-specific score thresholds based on question_type:
        - A/B questions (depth, spatial): score > 0.5
        - Point questions (object, vacant): score > 0.5
        - Benchmark point questions (bench_point): score > 0.5
        
        Args:
            result: Evaluation result dictionary
            
        Returns:
            True if result should be included in training data, False otherwise
        """
        score = result.get('score', 0.0)
        question_type = result.get('question_type', 'unknown')
        
        # Apply type-specific thresholds
        if question_type in ['depth', 'spatial']:
            # Binary A/B questions - strict threshold
            return score > 0.5
        elif question_type in ['object', 'vacant', 'bench_point']:
            # Point picking questions - use same threshold as robospatial
            return score > 0.5
        else:
            # Unknown type - use default threshold
            logger.warning(f"Unknown question_type '{question_type}' for result {result.get('index')}, using default threshold")
            return score > 0.5
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of RefSpatial evaluation results."""
        print("\n" + "="*60)
        print("REFSPATIAL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nTotal examples: {results['num_examples']}")
        print(f"Valid examples: {results['num_valid']}")
        error_rate_str = f"{results['error_rate']:.1%}" if results['error_rate'] is not None else "N/A"
        print(f"Errors: {results['num_errors']} ({error_rate_str})")
        
        # Show scoring errors if any
        num_scoring_errors = results.get('num_scoring_errors', 0)
        if num_scoring_errors > 0:
            print(f"  └─ Scoring errors: {num_scoring_errors}")
        
        print(f"\nOverall Performance:")
        acc_str = f"{results['accuracy']:.2%}" if results['accuracy'] is not None else "N/A"
        avg_str = f"{results['average_score']:.3f}" if results['average_score'] is not None else "N/A"
        print(f"  Accuracy: {acc_str}")
        print(f"  Average Score: {avg_str}")
        print(f"  Correct: {results['num_correct']}/{results['num_valid']}")
        
        if results.get('metrics_by_type'):
            print(f"\nPerformance by QA Type:")
            for qa_type, metrics in results['metrics_by_type'].items():
                print(f"  {qa_type}:")
                print(f"    Count: {metrics['count']}")
                print(f"    Accuracy: {metrics['accuracy']:.2%}")
                print(f"    Avg Score: {metrics['average_score']:.3f}")
        
        print("="*60 + "\n")

