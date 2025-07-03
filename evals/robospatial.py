# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
RoboSpatial-Home dataset evaluator.

This module provides evaluation capabilities for the RoboSpatial-Home dataset,
which includes yes/no questions, point picking, and bounding box detection tasks.
"""

import logging
import sys
import os
from typing import Dict, Any, List, Optional

# Import the exact scoring function and task detection used by VeRL
from verl.utils.reward_score.robos_vision import compute_score, _detect_task_type

# Import VeRL Vision Tool components
# Add VeRL to path for imports
import verl
verl_root = os.path.dirname(os.path.dirname(verl.__file__))
sys.path.insert(0, verl_root)
from examples.data_preprocess.robospatial_multiturn_vision_v2 import (
    DEFAULT_SYSTEM_CONTENT_POINT,
    DEFAULT_SYSTEM_CONTENT_YN,
    modify_question_text
)
from toolshed.prompts.spatial_reasoning import RS_SYSTEM_PROMPT_APPENDIX

from .common import BaseEvaluator
from .models import BaseModel

logger = logging.getLogger(__name__)

# No-tools versions of system prompts for RoboSpatial
DEFAULT_SYSTEM_CONTENT_POINT_NO_TOOLS = (
    "You are an expert in 3D spatial reasoning for robotics. "
    "Given an image and a spatial reasoning question asking you to detect specific objects or point to vacant space on a flat surface where an object could be placed, follow this process:\n\n"
    "1. First, think about the reasoning process as an internal monologue.\n"
    "Your reasoning process MUST be enclosed within <think> </think> tags.\n"
    "2. Once no further visual analysis is needed, you MUST provide your final answer inside "
    "<answer> and </answer> tags without detailed explanations.\n\n"
    "Example answer format: <answer>[[(0.00, 0.00), (0.55, 0.00), (0.55, 1.00), (0.00, 1.00)]]</answer>, or <answer>[(0.006, 0.311)]</answer>"
)

DEFAULT_SYSTEM_CONTENT_YN_NO_TOOLS = (
    "You are an expert in 3D spatial reasoning for robotics. "
    "Given an image and a spatial reasoning question asking for spatial relationship between two objects, follow this process:\n\n"
    "1. First, think about the reasoning process as an internal monologue.\n"
    "Your reasoning process MUST be enclosed within <think> </think> tags.\n"
    "2. Once no further visual analysis is needed, you MUST provide your final answer inside "
    "<answer> and </answer> tags without detailed illustrations.\n\n"
    "Example answer format: <answer>Yes</answer>, or <answer>No</answer>."
)


class RoboSpatialEvaluator(BaseEvaluator):
    """Evaluates models on the RoboSpatial-Home dataset."""
    
    def __init__(self, model: BaseModel, point_evaluation_method: str = 'convex_hull',
                 save_conversations_dir: Optional[str] = None,
                 save_training_dir: Optional[str] = None,
                 think_output_mode: Optional[str] = None,
                 system_prompt: Optional[str] = None):
        """
        Initialize the evaluator with a model.
        
        Args:
            model: Model instance that implements the BaseModel interface
            point_evaluation_method: Method for point evaluation (default: 'convex_hull')
            save_conversations_dir: Optional directory to save conversations in VeRL format
            save_training_dir: Optional directory to save training data in ShareGPT format
            think_output_mode: None, "suppress", "force"
            system_prompt: "default", "expanded"
        """
        super().__init__(model, save_conversations_dir, save_training_dir)
        self.point_evaluation_method = point_evaluation_method
        self.think_output_mode = think_output_mode
        self.system_prompt = system_prompt
        self.num_scoring_failures = 0  # Track cumulative scoring failures

    def parse_example(self, row: Dict) -> Dict:
        """
        Parse a RoboSpatial dataset row into standardized format.
        
        Extracts question, ground truth, and image from the RoboSpatial schema.
        Applies VeRL-specific question modifications and selects appropriate system prompt.
        
        Args:
            row: Raw dataset row from parquet file
            
        Returns:
            Standardized example dict with question, ground_truth, image, system_prompt, metadata
        """
        # Extract RoboSpatial-specific fields
        question = row['extra_info']['question']
        ground_truth = row['reward_model']['ground_truth']
        
        # Apply VeRL question modifications (several points → one point, etc.)
        modified_question = modify_question_text(question)
        
        # Handle no system prompt case
        if self.system_prompt is None:
            system_prompt = None
            mod_system_prompt = None
        else:
            # Select system prompt based on task type (determined by ground truth format)
            # Use no-tools version if model has no toolkit
            if ground_truth in ["Yes", "No"]:
                if self.model.toolkit is None:
                    system_prompt = DEFAULT_SYSTEM_CONTENT_YN_NO_TOOLS
                else:
                    system_prompt = DEFAULT_SYSTEM_CONTENT_YN
            else:
                if self.model.toolkit is None:
                    system_prompt = DEFAULT_SYSTEM_CONTENT_POINT_NO_TOOLS
                else:
                    system_prompt = DEFAULT_SYSTEM_CONTENT_POINT

            if self.system_prompt == "expanded":
                system_prompt += RS_SYSTEM_PROMPT_APPENDIX

            # Remove the <think> requirement
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
            'question': modified_question,
            'ground_truth': ground_truth,
            'image': image,
            'system_prompt': mod_system_prompt,
            'system_prompt_original': system_prompt,
            'metadata': {
                'original_question': question,
                'index': row['extra_info'].get('index', -1)
            }
        }
    
    def score_prediction(self, prediction: str, parsed_example: Dict, messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Score a model prediction using VeRL's compute_score function.
        
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

            score_result = compute_score(
                predict_str=prediction,
                ground_truth=parsed_example['ground_truth'],
                format_score=0.0,
                question=parsed_example['metadata']['original_question'],
                point_evaluation_method=self.point_evaluation_method
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
        Extract RoboSpatial-specific metadata for results.
        
        Args:
            parsed_example: Output from parse_example()
            score: Score from score_prediction()
            
        Returns:
            Dict with question_type, ground_truth, original question, and original system prompt
        """
        original_question = parsed_example['metadata']['original_question']
        return {
            'question_type': _detect_task_type(original_question),
            'ground_truth': parsed_example['ground_truth'],
            'question': original_question,
            'system_prompt_original': parsed_example['system_prompt_original']
        }
    
    def compute_custom_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Compute per-question-type metrics for RoboSpatial.
        
        Args:
            results: List of valid result dictionaries
            
        Returns:
            Dict with metrics_by_type breakdown
        """
        type_metrics = {}
        
        for qtype in ['yesno', 'point', 'bounding_box']:
            type_results = [r for r in results if r.get('question_type') == qtype]
            if type_results:
                type_scores = [r['score'] for r in type_results]
                type_metrics[qtype] = {
                    'count': len(type_results),
                    'average_score': sum(type_scores) / len(type_scores),
                    'accuracy': sum(1 for s in type_scores if s > 0.5) / len(type_scores)
                }
        
        return {'metrics_by_type': type_metrics}
    
    def should_include_in_training_data(self, result: Dict[str, Any]) -> bool:
        """
        Determine whether a RoboSpatial result should be included in training data.
        
        Applies dataset-specific score thresholds based on question type:
        - Yes/No questions: score > 0.5
        - Point picking questions: score > 0.5
        - Bounding box questions: score > 0.3 (more lenient due to task difficulty)
        
        Args:
            result: Evaluation result dictionary
            
        Returns:
            True if result should be included in training data, False otherwise
        """
        score = result.get('score', 0.0)
        question_type = result.get('question_type', 'unknown')
        
        # Apply type-specific thresholds
        if question_type == 'yesno':
            return score > 0.5
        elif question_type == 'point':
            return score > 0.5
        elif question_type == 'bounding_box':
            return score > 0.3
        else:
            # Unknown type - use default threshold
            logger.warning(f"Unknown question type '{question_type}' for result {result.get('index')}, using default threshold")
            return score > 0.5
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of RoboSpatial evaluation results."""
        print("\n" + "="*60)
        print("ROBOSPATIAL EVALUATION RESULTS")
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
            print(f"\nPerformance by Question Type:")
            for qtype, metrics in results['metrics_by_type'].items():
                print(f"  {qtype}:")
                print(f"    Count: {metrics['count']}")
                print(f"    Accuracy: {metrics['accuracy']:.2%}")
                print(f"    Avg Score: {metrics['average_score']:.3f}")
        
        print("="*60 + "\n")
