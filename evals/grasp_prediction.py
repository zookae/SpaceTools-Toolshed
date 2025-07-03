# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Grasp prediction evaluator for robot grasping tasks with mock robot.

This evaluator generates conversation data for grasp prediction by:
1. Using a depth estimator to preprocess dataset images (extract depth, pointcloud)
2. Passing the preprocessed data as mock_data to the model
3. LLM uses mock_robot tool which receives the injected mock data
4. Collecting conversation traces for training data
"""

import logging
from typing import Dict, Any, Optional
from PIL import Image

from .common import BaseEvaluator
from .models import BaseModel

logger = logging.getLogger(__name__)


class GraspPredictionEvaluator(BaseEvaluator):
    """Evaluator for grasp prediction tasks with mock robot simulation.
    
    This evaluator uses two toolkits:
    1. depth_toolkit: For preprocessing images to extract depth/pointcloud (runs once per image)
    2. main toolkit (via model): Contains mock_robot and other tools for LLM to use
    
    The depth data is prepared beforehand and injected as mock_data into the LLM session,
    where it's automatically passed to mock_robot tool calls.
    """
    
    def __init__(
        self,
        model: BaseModel,
        depth_estimator_toolkit=None,
        save_conversations_dir: Optional[str] = None,
        save_training_dir: Optional[str] = None,
        system_prompt_mode: str = "default",
    ):
        """Initialize grasp prediction evaluator.
        
        Args:
            model: Model instance with main toolkit (containing mock_robot)
            depth_estimator_toolkit: Separate toolkit instance with depth_estimator for preprocessing
            save_conversations_dir: Directory to save conversation JSON files
            save_training_dir: Directory to save training data in ShareGPT format
            system_prompt_mode: System prompt mode ('default')
        """
        super().__init__(
            model,
            save_conversations_dir=save_conversations_dir,
            save_training_dir=save_training_dir
        )
        
        self.depth_router, self.depth_toolkit = depth_estimator_toolkit
        self.system_prompt_mode = system_prompt_mode
        
        if self.depth_toolkit is None:
            logger.warning(
                "No depth_estimator_toolkit provided. Mock data will not include "
                "depth/pointcloud information. This may cause mock_robot tools to fail."
            )
    
    def _prepare_mock_data(self, image: Image.Image) -> Dict[str, Any]:
        """Generate mock robot data from image using depth estimator.
        
        This runs the depth estimator on the input image to extract:
        - depth_map: 2D array of depth values
        - point_cloud: Nx3 array of 3D points
        - focal_length_px: Estimated camera focal length
        
        Args:
            image: PIL Image to process
            
        Returns:
            Dict with mock data for injection into robot tool calls
        """
        # Use depth estimator to get depth and pointcloud
        result = self.depth_toolkit.depth_estimator.estimate_depth_with_pointcloud(image)
        
        # Use Ray object references for large data to avoid serialization overhead
        mock_data = {
            "mock_image": ray.put(image),
            "mock_depth_map": ray.put(result.value["depth_map"]),
            "mock_point_cloud": ray.put(result.value["point_cloud"]),
            "mock_focal_length_px": result.value["focal_length_px"],
            "image_width": result.value["width"],
            "image_height": result.value["height"],
        }
        
        logger.debug(
            f"Prepared mock data: depth_map={mock_data['mock_depth_map']}, "
            f"point_cloud={mock_data['mock_point_cloud']}, "
            f"focal_length={mock_data['mock_focal_length_px']:.1f}px"
        )
        
        return mock_data

    
    def _get_system_prompt(self) -> Optional[str]:
        """Get system prompt based on mode."""
        if self.system_prompt_mode is None:
            return None
        elif self.system_prompt_mode == "default":
            return (
                "You are a helpful robot assistant with the ability to control a robot to grasp and manipulate objects. "
                "You can capture images, get depth information, compute grasp poses, and execute grasps. "
                "Use the available tools to perceive the environment and plan your actions carefully."
            )
        else:
            raise ValueError(f"Unknown system_prompt_mode: {self.system_prompt_mode}")
    
    def parse_example(self, row: Dict) -> Dict:
        """Parse a dataset row into standardized format.
        
        Expected row format:
            - image or image_path: Image data or path to image file
            - question or instruction: Task description
            - ground_truth (optional): Expected answer/output for scoring
            
        Returns:
            Dict with:
                - question: The task/instruction
                - ground_truth: Expected output (if available)
                - image: PIL Image
                - system_prompt: System prompt for the model
                - mock_data: Preprocessed depth/pointcloud data
                - metadata: Additional info for tracking
        """
        # Load image
        image = self.load_image_from_row(row)
        
        # Get question/instruction
        question = row.get('question') or row.get('instruction') or row.get('task')
        if not question:
            raise ValueError(f"Row must contain 'question', 'instruction', or 'task' field: {row.keys()}")
        
        # Prepare mock data from image using depth estimator
        mock_data = self._prepare_mock_data(image)
        
        return {
            'question': question,
            'ground_truth': row.get('ground_truth'),
            'image': image,
            'system_prompt': self._get_system_prompt(),
            'mock_data': mock_data,  # This will be passed to model.predict()
            'metadata': {
                'index': row.get('index', -1),
                'scene_id': row.get('scene_id', 'unknown'),
            }
        }
    
    def score_prediction(self, prediction: str, parsed_example: Dict) -> float:
        """Score the model's prediction.
        
        For grasp prediction, we're primarily interested in collecting conversation data
        rather than strict accuracy scoring. We can do basic checks like:
        - Did the model attempt to use tools?
        - Did it execute a grasp?
        
        Args:
            prediction: Model's text response
            parsed_example: Parsed example dict
            
        Returns:
            Score between 0.0 and 1.0 (1.0 if model attempted grasp-related actions)
        """
        # Simple heuristic: check if prediction mentions grasp-related keywords
        prediction_lower = prediction.lower()
        
        grasp_keywords = ['grasp', 'execute_grasp', 'pick', 'grip']
        tool_keywords = ['capture_image', 'get_depth', 'code_executor']
        
        used_grasp = any(keyword in prediction_lower for keyword in grasp_keywords)
        used_tools = any(keyword in prediction_lower for keyword in tool_keywords)
        
        # Score based on whether model engaged with the task
        if used_grasp:
            score = 1.0
        elif used_tools:
            score = 0.5  # Used tools but didn't execute grasp
        else:
            score = 0.0
        
        return score
    
    def get_result_metadata(self, parsed_example: Dict, score: float) -> Dict:
        """Extract metadata to include in results.
        
        Args:
            parsed_example: Parsed example dict
            score: Computed score
            
        Returns:
            Dict of metadata fields
        """
        return {
            'scene_id': parsed_example['metadata'].get('scene_id', 'unknown'),
            'has_mock_data': bool(parsed_example.get('mock_data')),
            'question': parsed_example['question'][:100],  # Truncated for readability
        }
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print human-readable results summary.
        
        Args:
            results: Results dictionary from evaluate()
        """
        print("\n" + "=" * 80)
        print("GRASP PREDICTION EVALUATION RESULTS")
        print("=" * 80)
        print(f"Total examples:    {results['total_examples']}")
        print(f"Average score:     {results['average_score']:.3f}")
        print(f"Accuracy:          {results['accuracy']:.2%}")
        
        if 'detailed_results' in results:
            # Count how many examples used tools
            examples_with_tools = sum(
                1 for r in results['detailed_results']
                if r.get('score', 0) > 0
            )
            print(f"Examples with tool usage: {examples_with_tools}/{results['total_examples']}")
        
        print("=" * 80)
        
        if results['total_examples'] > 0:
            print("\nExample outputs:")
            for i, result in enumerate(results.get('detailed_results', [])[:3]):
                print(f"\n  Example {i+1} (score={result.get('score', 0):.2f}):")
                output = result.get('model_output', 'N/A')
                print(f"    {output[:150]}{'...' if len(output) > 150 else ''}")

