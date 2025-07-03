# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
BOP-ASK Bench dataset evaluator.

This module provides evaluation capabilities for the BOP-ASK Bench dataset,
which includes pose estimation (8-corner point prediction) and spatial 
reasoning (yes/no) tasks.

The evaluator supports two schema formats:
- Format 1 (legacy): Question in prompt[0]['content'], no extra_info
- Format 2 (current): Question in extra_info['question'], includes index and system_prompt
"""

import logging
import sys
import os
from typing import Dict, Any, List, Optional
import re

# Import exact scoring function from VeRL
from verl.utils.reward_score.bop_ask_bench import compute_score

# Import VeRL data preprocessing utilities for prompt/question handling
import verl
verl_root = os.path.dirname(os.path.dirname(verl.__file__))
sys.path.insert(0, verl_root)
from examples.data_preprocess.bop_ask_bench import (
    _normalise_question,
    _ANSWER_FORMAT,
    _DEFAULT_SYSTEM_PROMPT
)

from .common import BaseEvaluator
from .models import BaseModel

logger = logging.getLogger(__name__)

# No-tools version of system prompt for BOP-ASK
_DEFAULT_SYSTEM_PROMPT_NO_TOOLS = (
    "You are an expert in 3D spatial reasoning for robotics. "
    "Given an image and a spatial reasoning question, follow this process:\n\n"
    "1. First, think about the reasoning process as an internal monologue.\n"
    "Your reasoning process MUST be enclosed within <think> </think> tags.\n"
    "2. Once no further visual analysis is needed, you MUST provide your final answer inside "
    "<answer> and </answer> tags without detailed explanations.\n\n"
    "Example answer format depends on the question type."
)


def _detect_question_type_from_ground_truth(ground_truth: str) -> str:
    """
    Detect BOP-ASK question type from ground truth format.
    
    This is a fallback when data_source field is missing or malformed.
    
    Args:
        ground_truth: Ground truth string
        
    Returns:
        Detected question type: 'pose', 'spatial_reasoning', or 'unknown'
    """
    gt_str = str(ground_truth).strip()
    
    # Spatial reasoning: Yes/No answers
    if gt_str.lower() in ['yes', 'no']:
        return 'spatial_reasoning'
    
    # Pose: 8 points (list of 8 tuples)
    try:
        if gt_str.startswith('[') and gt_str.endswith(']'):
            import ast
            parsed = ast.literal_eval(gt_str)
            if isinstance(parsed, list) and len(parsed) == 8:
                if all(isinstance(p, (tuple, list)) and len(p) == 2 for p in parsed):
                    return 'pose'
    except (ValueError, SyntaxError):
        pass
    
    return 'unknown'


class BopAskEvaluator(BaseEvaluator):
    """Evaluates models on the BOP-ASK Bench dataset."""
    
    def __init__(self, model: BaseModel, 
                 save_conversations_dir: Optional[str] = None,
                 save_training_dir: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 think_output_mode: Optional[str] = None,
                 mock_robot_mode: bool = False,
                 depth_toolkit = None):
        """
        Initialize the evaluator with a model.
        
        Args:
            model: Model instance implementing BaseModel interface
            save_conversations_dir: Optional directory to save conversations
            save_training_dir: Optional directory to save training data
            system_prompt: System prompt mode ("default" or None for default)
            think_output_mode: None, "suppress", "force" (for model-specific reasoning handling)
            mock_robot_mode: If True, enables mock robot mode with depth preprocessing
            depth_toolkit: Toolkit instance with depth_estimator (required for mock_robot_mode)
        """
        super().__init__(model, save_conversations_dir, save_training_dir)
        self.system_prompt_mode = system_prompt# or "default"
        self.think_output_mode = think_output_mode
        self.mock_robot_mode = mock_robot_mode
        self.depth_toolkit = depth_toolkit
        self.num_scoring_failures = 0

    def convert_grasp_prompt_to_manipulation_instruction(self, question: str) -> str:
        obj_name = question.split("grasp plane for the ")[1].split("? Your")[0]
        print(f"Extracted object name: {obj_name}")
        new_user_question = f"Pick up the {obj_name}."
        return new_user_question

    def parse_example(self, row: Dict) -> Dict:
        """
        Parse a BOP-ASK dataset row into standardized format.
        
        Supports two schema formats:
        - Format 1: Question in prompt[0]['content'], no extra_info
        - Format 2: Question in extra_info['question'], includes index
        
        Extracts question, ground truth, image, and question type from parquet schema.
        Applies VeRL-specific question normalization (pixel → normalized coords).
        
        Args:
            row: Raw dataset row from parquet file
            
        Returns:
            Standardized example dict with question, ground_truth, image, 
            system_prompt, metadata
        """
        # Extract data_source to determine question type
        data_source = row.get('data_source', 'unknown')
        try:
            question_type = data_source.split('-')[-1]  # "pose", "spatial_reasoning", etc.
        except (AttributeError, IndexError):
            question_type = "unknown"
        
        # Filter question types in mock_robot mode
        if self.mock_robot_mode and question_type != 'grasp':
            raise ValueError(
                f"Mock robot mode only supports 'grasp' questions, got '{question_type}'. "
                f"Filter your dataset to only include grasp examples."
            )
        
        # Handle two schema formats
        if 'extra_info' in row and 'question' in row['extra_info']:
            # Format 2 (current): Direct access to question field
            question = row['extra_info']['question']
            index = row['extra_info'].get('index', -1)
        else:
            # Format 1 (legacy): Extract from prompt
            prompt_list = row.get('prompt', [])
            if len(prompt_list) > 1 and prompt_list[1]['role'] == 'user':
                question = prompt_list[1]['content']
            else:
                # Cannot extract question - set to unknown and log warning
                logger.warning(f"Cannot extract question from row with keys: {row.keys()}. Setting question_type to 'unknown'")
                question_type = "unknown"
                question = "Unknown question format"
            index = -1
        
        # Extract ground truth
        ground_truth = row['reward_model']['ground_truth']
        
        # If question_type is still unknown, try to detect it from ground truth format
        if question_type == "unknown":
            question_type = _detect_question_type_from_ground_truth(ground_truth)
        
        # Apply VeRL question normalization (pixel coords → normalized)
        normalized_question = _normalise_question(question)

        # Add answer format instruction if available for this question type
        answer_format = _ANSWER_FORMAT.get(question_type, "")
        if answer_format:
            normalized_question = f"{normalized_question} {answer_format}".strip()
        
        # Handle no system prompt case
        if self.system_prompt_mode is None:
            system_prompt = None
            mod_system_prompt = None
        else:
            # Select system prompt
            # Format 2 may include a system prompt, but we use default for consistency
            # Use no-tools version if model has no toolkit
            if self.system_prompt_mode == "default":
                if self.model.toolkit is None:
                    system_prompt = _DEFAULT_SYSTEM_PROMPT_NO_TOOLS
                else:
                    system_prompt = _DEFAULT_SYSTEM_PROMPT
            else:
                if self.model.toolkit is None:
                    system_prompt = _DEFAULT_SYSTEM_PROMPT_NO_TOOLS
                else:
                    system_prompt = _DEFAULT_SYSTEM_PROMPT
        
        # Load image using parent's utility method
        image = self.load_image_from_row(row)
        
        # Prepare mock data in mock_robot mode
        mock_data = None
        image_for_prompt = image
        if self.mock_robot_mode:
            from .mock_depth_prep import prepare_mock_data_from_image
            mock_data = prepare_mock_data_from_image(image, self.depth_toolkit)
            # Don't include image in prompt - LLM must retrieve it via tool
            image_for_prompt = None
            
            if system_prompt is not None:
                # Add robot control instructions to system prompt
                robot_appendix = (
                    "\n\nYou are controlling a robot with sensors and actuators. "
                    "Use the available tools to capture images, get depth data, and execute manipulation tasks."
                    "Note that NONE of the robot tools accept image arguments, pay attention to the tool descriptions."
                )
                system_prompt = system_prompt + robot_appendix

            normalized_question = self.convert_grasp_prompt_to_manipulation_instruction(normalized_question)
        
        if system_prompt is not None:
            print(f"System prompt: {system_prompt}")
            # Apply think output mode modifications (same pattern as RoboSpatial)
            # Note: BOP-ASK default prompt doesn't have <think> instructions, 
            # but we support adding/suppressing them for model compatibility
            if self.think_output_mode == "suppress":
                # Suppress any thinking output - instruct model to only give answer
                mod_system_prompt = system_prompt + " You should NEVER say ANYTHING to the user other than the final answer in <answer></answer> tags."
                mod_system_prompt = mod_system_prompt.replace("Your reasoning process MUST be enclosed within <think> </think> tags.", "")
            elif self.think_output_mode == "force":
                # Force thinking output - instruct model to show full reasoning
                mod_system_prompt = system_prompt + " You must ALWAYS say your full reasoning process in detail, documenting every step of how you are approaching the problem. The reasoning process MUST be enclosed within <think> </think> tags in every message to the user, including before any tool calls, and before the final answer."
            else:
                mod_system_prompt = system_prompt

            if question_type == "grasp":
                mod_system_prompt = mod_system_prompt + "\nDo not attempt to call the grasp tool more than once. "
                if self.mock_robot_mode:
                    mod_system_prompt = mod_system_prompt + "If it fails to find grasps, simply answer 'no grasp found' in the <answer></answer> tags."
                else:
                    mod_system_prompt = mod_system_prompt + "If it fails to find grasps, try to look at the image and the object coordinate, and estimate the grasp pose yourself."
        
        return {
            'question': normalized_question,
            'ground_truth': ground_truth,
            'image': image_for_prompt,  # None in mock_robot mode
            'mock_data': mock_data,  # Will be passed to model.predict()
            'system_prompt': mod_system_prompt,
            'system_prompt_original': system_prompt,
            '_training_image': image if self.mock_robot_mode else None,  # Store for training data in mock_robot mode
            'metadata': {
                'original_question': question,
                'question_type': question_type,
                'index': index,
                'data_source': data_source
            }
        }
    
    def score_prediction(self, prediction: str, parsed_example: Dict, messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Score a model prediction using VeRL's compute_score function.
        
        In mock_robot mode, also verifies that grasp execution was attempted.
        
        Args:
            prediction: Model's output string
            parsed_example: Output from parse_example()
            messages: Conversation history (for parallel execution)
            
        Returns:
            Dict with 'score', 'acc', 'fmt', and optionally 'scoring_error', 'execution_verified'
        """
        try:
            # Handle None prediction gracefully
            if prediction is None:
                logger.warning("Model returned None prediction - treating as empty string")
                prediction = ""

            if self.system_prompt_mode is None and "<answer>" not in prediction:
                print("No <answer> tag found in prediction, adding it")
                prediction = "<answer>" + prediction + "</answer>"

            score_result = compute_score(
                solution_str=prediction,
                ground_truth=parsed_example['ground_truth'],
                question_type=parsed_example['metadata']['question_type'],
                collapse_penalty=False  # Always False - collapse penalty is for RL training only
            )
            
            # Fix acc field for grasp questions - VeRL returns raw NCE, not binary
            question_type = parsed_example['metadata']['question_type']
            if question_type == 'grasp':
                # Grasp accuracy: score < 2.0 means success
                score_result['acc'] = 1.0 if score_result['score'] < 2.0 else 0.0
            
            # In mock_robot mode, verify grasp execution was attempted
            if self.mock_robot_mode:
                execution_verified = self._verify_grasp_execution(messages)
                
                if not execution_verified:
                    # Execution not attempted - fail regardless of grasp quality
                    # Both execution AND grasp quality must pass
                    score_result['score'] = 10.0
                    score_result['acc'] = 0.0
                    score_result['execution_verified'] = False
                    logger.info(f"Grasp execution not verified for example {parsed_example['metadata'].get('index', '?')}")
                else:
                    score_result['score'] = 0.0
                    score_result['acc'] = 1.0
                    score_result['execution_verified'] = True
            
            # compute_score returns dict with 'score', 'acc', 'fmt'
            return score_result
            
        except Exception as e:
            # Print error immediately to console to avoid hiding it
            import traceback
            example_idx = parsed_example['metadata'].get('index', '?')
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            print(f"\n{'='*80}")
            print(f"SCORING ERROR for example {example_idx}")
            print(f"{'='*80}")
            print(f"Error: {error_msg}")
            print(f"\nQuestion: {parsed_example['metadata'].get('original_question', 'N/A')}")
            print(f"Question type: {parsed_example['metadata'].get('question_type', 'N/A')}")
            print(f"\nGround truth: {str(parsed_example.get('ground_truth', 'N/A'))}")
            print(f"\nFULL Model output (not truncated):")
            print(f"{'-'*80}")
            print(prediction if prediction else 'N/A')
            print(f"{'-'*80}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            print(f"{'='*80}\n")
            
            # Also log it
            logger.error(f"Scoring failed for example {example_idx}: {error_msg}")
            
            # Return error result to allow evaluation to continue
            return {
                'score': 0.0,
                'acc': 0.0,
                'fmt': 0.0,
                'scoring_error': error_msg
            }
    
    def _verify_grasp_execution(self, messages: Optional[List[Dict]] = None) -> bool:
        """Check if grasp execution was attempted by examining conversation history.
        
        This looks for the success message from mock_robot.execute_grasp() in the
        conversation history to verify the agent actually called the grasp tool.
        
        Args:
            messages: Conversation history (passed from score_prediction for parallel execution)
        
        Returns:
            True if "Grasp execution succeeded." found in conversation, False otherwise
        """
        # Use provided messages or fall back to model's conversation history (for sequential mode)
        if messages is None:
            if hasattr(self.model, 'get_conversation_history'):
                messages = self.model.get_conversation_history()
            else:
                messages = None
        
        if not messages:
            logger.debug("No conversation history available for execution verification")
            return False
        
        # Search for the success string from mock_robot.py execute_grasp
        success_marker = "Grasp execution succeeded."
        return success_marker in str(messages)
    
    
    def get_result_metadata(self, parsed_example: Dict, score: float) -> Dict:
        """
        Extract BOP-ASK-specific metadata for results.
        
        Args:
            parsed_example: Output from parse_example()
            score: Score from score_prediction()
            
        Returns:
            Dict with question_type, ground_truth, question, data_source, system_prompt_original
        """
        return {
            'question_type': parsed_example['metadata']['question_type'],
            'ground_truth': parsed_example['ground_truth'],
            'question': parsed_example['metadata']['original_question'],
            'data_source': parsed_example['metadata']['data_source'],
            'system_prompt_original': parsed_example['system_prompt_original']
        }
    
    def compute_custom_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Compute per-question-type metrics for BOP-ASK.
        
        Args:
            results: List of valid result dictionaries
            
        Returns:
            Dict with metrics_by_type breakdown and execution_stats (if mock_robot mode)
        """
        metrics = {}
        
        # Per-type metrics
        type_metrics = {}
        
        # Get unique question types from results
        question_types = set(r.get('question_type') for r in results if 'question_type' in r)
        
        for qtype in question_types:
            type_results = [r for r in results if r.get('question_type') == qtype]
            if type_results:
                type_scores = [r['score'] for r in type_results]
                type_accs = [r.get('acc', r['score']) for r in type_results]
                
                type_metrics[qtype] = {
                    'count': len(type_results),
                    'average_score': sum(type_scores) / len(type_scores),
                    'average_acc': sum(type_accs) / len(type_accs),
                    'accuracy': sum(1 for r in type_results if r.get('acc', r['score']) > 0.5) / len(type_results)
                }
        
        metrics['metrics_by_type'] = type_metrics
        
        # Add execution stats for mock_robot mode
        if self.mock_robot_mode:
            num_executed = sum(1 for r in results if r.get('execution_verified', False))
            metrics['execution_stats'] = {
                'num_executed': num_executed,
                'execution_rate': num_executed / len(results) if results else 0.0
            }
        
        return metrics
    
    def should_include_in_training_data(self, result: Dict[str, Any]) -> bool:
        """
        Determine whether a BOP-ASK result should be included in training data.
        
        Uses the 'acc' field which already encodes question-type-specific correctness.
        
        Args:
            result: Evaluation result dictionary
            
        Returns:
            True if result should be included in training data, False otherwise
        """
        # Use 'acc' field which already has question-type-specific semantics baked in
        acc = result.get('acc', result.get('score', 0.0))
        return acc > 0.5
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of BOP-ASK evaluation results."""
        print("\n" + "="*60)
        print("BOP-ASK BENCH EVALUATION RESULTS")
        if self.mock_robot_mode:
            print("(Mock Robot Mode)")
        print("="*60)
        
        print(f"\nTotal examples: {results['num_examples']}")
        print(f"Valid examples: {results['num_valid']}")
        error_rate_str = f"{results['error_rate']:.1%}" if results['error_rate'] is not None else "N/A"
        print(f"Errors: {results['num_errors']} ({error_rate_str})")
        
        # Show scoring errors if any
        num_scoring_errors = results.get('num_scoring_errors', 0)
        if num_scoring_errors > 0:
            print(f"  └─ Scoring errors: {num_scoring_errors}")
            
            # Print actual error messages to avoid hiding errors
            if 'detailed_results' in results:
                error_examples = [r for r in results['detailed_results'] if 'scoring_error' in r]
                if error_examples:
                    print(f"\n  Scoring Error Details (showing up to 5):")
                    for i, err_result in enumerate(error_examples[:5]):
                        idx = err_result.get('index', '?')
                        error_msg = err_result.get('scoring_error', 'Unknown error')
                        print(f"    Example {idx}: {error_msg}")
                        # Also print the model output to help debug
                        model_output = err_result.get('model_output', '') or ''
                        output_preview = model_output[:100] if model_output else 'None'
                        print(f"      Model output: {output_preview}...")
                    if len(error_examples) > 5:
                        print(f"    ... and {len(error_examples) - 5} more scoring errors")
        
        print(f"\nOverall Performance:")
        acc_str = f"{results['accuracy']:.2%}" if results['accuracy'] is not None else "N/A"
        avg_str = f"{results['average_score']:.3f}" if results['average_score'] is not None else "N/A"
        print(f"  Accuracy: {acc_str}")
        print(f"  Average Score: {avg_str}")
        print(f"  Correct: {results['num_correct']}/{results['num_valid']}")
        
        # Show mock robot execution stats
        if self.mock_robot_mode and results.get('execution_stats'):
            print(f"\nMock Robot Execution Stats:")
            exec_stats = results['execution_stats']
            print(f"  Execution attempts: {exec_stats['num_executed']}/{results['num_valid']}")
            print(f"  Execution rate: {exec_stats['execution_rate']:.2%}")
        
        if results.get('metrics_by_type'):
            print(f"\nPerformance by Question Type:")
            for qtype, metrics in results['metrics_by_type'].items():
                print(f"  {qtype}:")
                print(f"    Count: {metrics['count']}")
                print(f"    Accuracy: {metrics['accuracy']:.2%}")
                print(f"    Avg Score: {metrics['average_score']:.3f}")
                if 'average_acc' in metrics and metrics['average_acc'] != metrics['average_score']:
                    print(f"    Avg Acc (raw): {metrics['average_acc']:.3f}")
        
        print("="*60 + "\n")

