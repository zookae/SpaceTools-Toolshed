# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
BLINK Benchmark evaluator.

This module provides evaluation capabilities for the BLINK benchmark,
which includes Relative_Depth and Spatial_Relation multiple-choice questions.
"""

import logging
import sys
import os
from typing import Dict, Any, List, Optional

# Import the scoring function for Relative_Depth from VeRL
from verl.utils.reward_score.blink_relative_depth import compute_score as compute_relative_depth_score

from .common import BaseEvaluator
from .models import BaseModel

logger = logging.getLogger(__name__)

# Default system prompt for BLINK tasks
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

class BlinkEvaluator(BaseEvaluator):
    """Evaluates models on the BLINK benchmark (Relative_Depth and Spatial_Relation)."""
    
    def __init__(self, model: BaseModel,
                 save_conversations_dir: Optional[str] = None,
                 save_training_dir: Optional[str] = None,
                 think_output_mode: Optional[str] = None,
                 system_prompt: Optional[str] = "default"):
        """
        Initialize the evaluator with a model.
        
        Args:
            model: Model instance that implements the BaseModel interface
            save_conversations_dir: Optional directory to save conversations in VeRL format
            save_training_dir: Optional directory to save training data in ShareGPT format
            think_output_mode: None, "suppress", "force"
            system_prompt: "default" to use DEFAULT_SYSTEM_PROMPT, or custom system prompt string
        """
        super().__init__(model, save_conversations_dir, save_training_dir)
        self.think_output_mode = think_output_mode
        self.system_prompt = system_prompt
        self.num_scoring_failures = 0  # Track cumulative scoring failures

    def parse_example(self, row: Dict) -> Dict:
        """
        Parse a BLINK dataset row into standardized format.
        
        Extracts question, ground truth, and image from the BLINK schema.
        
        Args:
            row: Raw dataset row from parquet file
            
        Returns:
            Standardized example dict with question, ground_truth, image, system_prompt, metadata
        """
        # Extract BLINK-specific fields
        question = row['extra_info']['question']
        ground_truth = row['reward_model']['ground_truth']
        data_source = row['data_source']
        
        # Determine question type
        if "Relative_Depth" in data_source:
            question_type = "relative_depth"
        elif "Spatial_Relation" in data_source:
            question_type = "spatial_relation"
        else:
            question_type = "unknown"
        
        # Handle no system prompt case
        if self.system_prompt is None:
            system_prompt = None
            mod_system_prompt = None
        else:
            # Use default system prompt (override by default as per design decision)
            if self.system_prompt == "default":
                # Use no-tools version if model has no toolkit
                if self.model.toolkit is None:
                    system_prompt = DEFAULT_SYSTEM_PROMPT_NO_TOOLS
                else:
                    system_prompt = DEFAULT_SYSTEM_PROMPT
            else:
                system_prompt = self.system_prompt
            
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
            'question_type': question_type,
            'metadata': {
                'original_question': question,
                'index': row['extra_info'].get('index', -1),
                'answer': row['extra_info'].get('answer', ground_truth),
                'question_type': question_type,
                'data_source': data_source
            }
        }
    
    def score_prediction(self, prediction: str, parsed_example: Dict, messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Score a model prediction using question-type-specific scoring functions.
        
        Args:
            prediction: Model's output string
            parsed_example: Output from parse_example()
            messages: Optional conversation history (for parallel execution)
            
        Returns:
            Dict with 'score' (float between 0.0 and 1.0), and optionally 'scoring_error' if scoring failed
        """
        question_type = parsed_example['question_type']
        
        try:
            if question_type == "relative_depth":
                score = compute_relative_depth_score(
                    solution_str=prediction,
                    ground_truth=parsed_example['ground_truth']
                )
                return {
                    'score': score,
                    'acc': score
                }
            elif question_type == "spatial_relation":
                raise NotImplementedError(
                    "Spatial_Relation scoring is not yet implemented. "
                    "Please implement the scoring function in verl.utils.reward_score.blink_spatial_relation"
                )
            else:
                raise ValueError(f"Unknown question type: {question_type}")
                
        except NotImplementedError:
            # Re-raise NotImplementedError without catching
            raise
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
        Extract BLINK-specific metadata for results.
        
        Args:
            parsed_example: Output from parse_example()
            score: Score from score_prediction()
            
        Returns:
            Dict with ground_truth, question, question_type, and original system prompt
        """
        return {
            'ground_truth': parsed_example['ground_truth'],
            'question': parsed_example['metadata']['original_question'],
            'question_type': parsed_example['question_type'],
            'system_prompt_original': parsed_example['system_prompt_original']
        }
    
    def compute_custom_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Compute custom metrics for BLINK, including per-question-type breakdown.
        
        Args:
            results: List of valid result dictionaries
            
        Returns:
            Dict with additional metrics broken down by question type
        """
        # Compute per-question-type metrics
        metrics = {}
        
        # Group by question type
        by_type = {}
        for result in results:
            qtype = result.get('question_type', 'unknown')
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(result)
        
        # Compute metrics for each type
        for qtype, type_results in by_type.items():
            scores = [r['score'] for r in type_results]
            metrics[f'{qtype}_count'] = len(type_results)
            metrics[f'{qtype}_accuracy'] = sum(1 for s in scores if s > 0.5) / len(scores) if scores else 0.0
            metrics[f'{qtype}_avg_score'] = sum(scores) / len(scores) if scores else 0.0
        
        return metrics
    
    def should_include_in_training_data(self, result: Dict[str, Any]) -> bool:
        """
        Determine whether a BLINK result should be included in training data.
        
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
        """Print a formatted summary of BLINK evaluation results."""
        print("\n" + "="*60)
        print("BLINK BENCHMARK EVALUATION RESULTS")
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
        
        # Show per-question-type breakdown
        print(f"\nBreakdown by Question Type:")
        for key in sorted(results.keys()):
            if key.endswith('_count'):
                qtype = key.replace('_count', '')
                count = results[key]
                acc = results.get(f'{qtype}_accuracy', 0.0)
                avg = results.get(f'{qtype}_avg_score', 0.0)
                print(f"  {qtype}:")
                print(f"    Count: {count}")
                print(f"    Accuracy: {acc:.2%}")
                print(f"    Avg Score: {avg:.3f}")
        
        print("="*60 + "\n")

