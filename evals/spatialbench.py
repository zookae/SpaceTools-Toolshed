# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
SpatialBench Positional dataset evaluator.

This module provides evaluation capabilities for the SpatialBench positional
multiple-choice task, which tests spatial reasoning with A/B/C/D questions.
"""

import logging
import sys
import os
from typing import Dict, Any, List, Optional

# Import the exact scoring function used by VeRL
from verl.utils.reward_score.spatialbench_positional import compute_score

# Import VeRL Vision Tool components for extract_solution
from verl.utils.reward_score.robos_vision import extract_solution

from .common import BaseEvaluator
from .models import BaseModel

logger = logging.getLogger(__name__)

# Default system prompt for SpatialBench positional tasks
DEFAULT_SYSTEM_PROMPT = """You are an expert in 3D spatial reasoning for robotics. Given an image and a spatial reasoning question, follow this process:

1. First, think about the reasoning process as an internal monologue the first time you receive the question, and every time you receive new information.
Your reasoning process MUST be enclosed within <think> </think> tags.
2. After thinking, if you need additional information to answer the question, such as specific object location in the image or information about depth or geometry, call the appropriate vision tools.
3. When you receive a tool response that contains reasonable information, use that information to continue your analysis on the question.
4. Once no further visual analysis or tool calls are needed, you MUST provide your final answer inside <answer> and </answer> tags without detailed illustrations.

Example answer format: <answer> <Your final answer here> </answer>."""

DEFAULT_SYSTEM_PROMPT_NO_TOOLS = """You are an expert in 3D spatial reasoning for robotics. Given an image and a spatial reasoning question, follow this process:

1. First, think about the reasoning process as an internal monologue.
Your reasoning process MUST be enclosed within <think> </think> tags.
2. Once no further visual analysis is needed, you MUST provide your final answer inside <answer> and </answer> tags without detailed illustrations.

Example answer format: <answer> <Your final answer here> </answer>."""


class SpatialBenchEvaluator(BaseEvaluator):
    """Evaluates models on the SpatialBench positional dataset."""
    
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
            system_prompt: "default" or custom system prompt string
        """
        super().__init__(model, save_conversations_dir, save_training_dir)
        self.think_output_mode = think_output_mode
        self.system_prompt = system_prompt
        self.num_scoring_failures = 0  # Track cumulative scoring failures

    def parse_example(self, row: Dict) -> Dict:
        """
        Parse a SpatialBench dataset row into standardized format.
        
        Extracts question, ground truth, and image from the SpatialBench schema.
        
        Args:
            row: Raw dataset row from parquet file
            
        Returns:
            Standardized example dict with question, ground_truth, image, system_prompt, metadata
        """
        # Extract SpatialBench-specific fields
        question = row['extra_info']['question']
        ground_truth = row['reward_model']['ground_truth']
        
        # Handle no system prompt case
        if self.system_prompt is None:
            system_prompt = None
            mod_system_prompt = None
        else:
            # Use default system prompt (no-tools version if model has no toolkit)
            if self.model.toolkit is None:
                system_prompt = DEFAULT_SYSTEM_PROMPT_NO_TOOLS
            else:
                system_prompt = DEFAULT_SYSTEM_PROMPT
            
            # Modify system prompt based on think_output_mode
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
                'answer': row['extra_info'].get('answer', ground_truth)
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
            score_result = compute_score(
                solution_str=prediction,
                ground_truth=parsed_example['ground_truth']
            )
            
            # Handle dict or float return value
            if isinstance(score_result, dict):
                return {
                    'score': score_result.get('score', 0.0),
                    'acc': score_result.get('acc', 0.0)
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
        Extract SpatialBench-specific metadata for results.
        
        Args:
            parsed_example: Output from parse_example()
            score: Score from score_prediction()
            
        Returns:
            Dict with ground_truth, original question, and original system prompt
        """
        return {
            'ground_truth': parsed_example['ground_truth'],
            'question': parsed_example['metadata']['original_question'],
            'system_prompt_original': parsed_example['system_prompt_original']
        }
    
    def compute_custom_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Compute custom metrics for SpatialBench.
        
        Args:
            results: List of valid result dictionaries
            
        Returns:
            Dict with additional metrics (currently empty, can be extended)
        """
        # Could add per-choice statistics or other metrics here if needed
        return {}
    
    def should_include_in_training_data(self, result: Dict[str, Any]) -> bool:
        """
        Determine whether a SpatialBench result should be included in training data.
        
        For multiple-choice questions, we use a simple threshold:
        - Correct answers (score == 1.0) are included
        - Incorrect answers (score == 0.0) are excluded
        
        Args:
            result: Evaluation result dictionary
            
        Returns:
            True if result should be included in training data, False otherwise
        """
        score = result.get('score', 0.0)
        # For multiple choice, it's binary: either correct (1.0) or incorrect (0.0)
        return score > 0.5
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of SpatialBench evaluation results."""
        print("\n" + "="*60)
        print("SPATIALBENCH POSITIONAL EVALUATION RESULTS")
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
        
        print("="*60 + "\n")

