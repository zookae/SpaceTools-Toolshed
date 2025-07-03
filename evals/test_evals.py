# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Simple tests for the evaluation framework.

Run with: python -m pytest evals/test_evals.py
"""

import unittest
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np

from .models import BaseModel, UnifiedModel
from .robospatial import RoboSpatialEvaluator
from .bop_ask import BopAskEvaluator


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = 0
    
    def predict(self, text: str, images=None, system_prompt=None) -> str:
        self.call_count += 1
        
        # Return predefined responses based on question type
        if "yes or no" in text.lower():
            return self.responses.get('yes_no', '#### yes')
        elif "pinpoint" in text.lower():
            return self.responses.get('point', '#### [(0.5, 0.5)]')
        elif "find all instances" in text.lower():
            return self.responses.get('bbox', '#### [[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]]')
        else:
            return self.responses.get('default', '#### unknown')


class TestRoboSpatialEvaluator(unittest.TestCase):
    """Test the RoboSpatial evaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MockModel()
        self.evaluator = RoboSpatialEvaluator(self.mock_model)
    
    def test_detect_question_type(self):
        """Test question type detection."""
        test_cases = [
            ("Is the object above the table? Answer yes or no.", "yes_no"),
            ("Pinpoint one point on the object.", "point"),
            ("Find all instances of chairs in the image.", "bounding_box"),
            ("What color is the sky?", "yes_no"),  # Default to yes_no
        ]
        
        for question, expected_type in test_cases:
            result = self.evaluator._detect_question_type(question)
            self.assertEqual(result, expected_type, 
                           f"Failed for question: {question}")
    
    def test_simple_score_yes_no(self):
        """Test simple scoring for yes/no questions."""
        test_cases = [
            ("#### yes", "yes", "Is it red? Answer yes or no.", 1.0),
            ("#### no", "no", "Is it red? Answer yes or no.", 1.0),
            ("#### yes", "no", "Is it red? Answer yes or no.", 0.0),
            ("The answer is yes", "yes", "Is it red? Answer yes or no.", 1.0),
            ("<answer>yes</answer>", "yes", "Is it red? Answer yes or no.", 1.0),
        ]
        
        for prediction, ground_truth, question, expected_score in test_cases:
            score = self.evaluator._simple_score(prediction, ground_truth, question)
            self.assertEqual(score, expected_score,
                           f"Failed for prediction='{prediction}', gt='{ground_truth}'")
    
    def test_simple_score_points(self):
        """Test simple scoring for point questions."""
        test_cases = [
            ("#### [(0.5, 0.5)]", "[(0.4, 0.4)]", "Pinpoint one point", 1.0),  # Any point counts
            ("#### (0.5, 0.5)", "[(0.4, 0.4)]", "Pinpoint one point", 1.0),
            ("#### no points found", "[(0.4, 0.4)]", "Pinpoint one point", 0.0),
        ]
        
        for prediction, ground_truth, question, expected_score in test_cases:
            score = self.evaluator._simple_score(prediction, ground_truth, question)
            self.assertEqual(score, expected_score,
                           f"Failed for prediction='{prediction}', gt='{ground_truth}'")
    
    def test_evaluate_example(self):
        """Test evaluating a single example."""
        # Create a mock example
        example = {
            'prompt': [{'content': 'Is the object red? Answer yes or no.'}],
            'images': [{'image': 'data:image/jpeg;base64,/9j/4AAQ...'}],  # Dummy base64
            'reward_model': {'ground_truth': 'yes'},
            'extra_info': {'question': 'Is the object red? Answer yes or no.', 'index': 0}
        }
        
        # Mock image loading
        with patch.object(self.evaluator, '_load_image_from_data_uri') as mock_load:
            mock_load.return_value = Image.new('RGB', (100, 100))
            
            result = self.evaluator._evaluate_example(example)
        
        self.assertEqual(result['index'], 0)
        self.assertEqual(result['ground_truth'], 'yes')
        self.assertEqual(result['question_type'], 'yes_no')
        self.assertIn('score', result)
        self.assertIn('prediction', result)
    
    def test_aggregate_results(self):
        """Test result aggregation."""
        results = [
            {'index': 0, 'score': 1.0, 'question_type': 'yes_no'},
            {'index': 1, 'score': 0.0, 'question_type': 'yes_no'},
            {'index': 2, 'score': 1.0, 'question_type': 'point'},
            {'index': 3, 'score': 0.5, 'question_type': 'point'},
            {'index': 4, 'error': 'Failed to process'},
        ]
        
        aggregated = self.evaluator._aggregate_results(results)
        
        self.assertEqual(aggregated['num_examples'], 5)
        self.assertEqual(aggregated['num_valid'], 4)
        self.assertEqual(aggregated['num_errors'], 1)
        self.assertEqual(aggregated['error_rate'], 0.2)
        self.assertEqual(aggregated['num_correct'], 2)  # score > 0.5
        self.assertEqual(aggregated['accuracy'], 0.5)   # 2/4
        
        # Check metrics by type
        self.assertIn('yes_no', aggregated['metrics_by_type'])
        self.assertEqual(aggregated['metrics_by_type']['yes_no']['count'], 2)
        self.assertEqual(aggregated['metrics_by_type']['yes_no']['accuracy'], 0.5)


class TestBopAskEvaluator(unittest.TestCase):
    """Test the BOP-ASK evaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MockModel({
            'spatial_reasoning': '<answer>Yes</answer>',
            'pose': '<answer>[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]</answer>',
        })
        self.evaluator = BopAskEvaluator(self.mock_model)
    
    def test_parse_example_format_1(self):
        """Test parsing Format 1 (legacy) schema."""
        example = {
            'data_source': 'BOP-ASK-Bench-pose',
            'prompt': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Predict the 8 corner points.'}
            ],
            'images': [{'image': 'data:image/jpeg;base64,/9j/4AAQ...'}],
            'reward_model': {'ground_truth': '[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]'}
        }
        
        with patch.object(self.evaluator, 'load_image_from_row') as mock_load:
            mock_load.return_value = Image.new('RGB', (100, 100))
            parsed = self.evaluator.parse_example(example)
        
        self.assertIn('Predict the 8 corner points', parsed['question'])
        self.assertEqual(parsed['metadata']['question_type'], 'pose')
        self.assertEqual(parsed['metadata']['index'], -1)  # Format 1 has no index
    
    def test_parse_example_format_2(self):
        """Test parsing Format 2 (current) schema."""
        example = {
            'data_source': 'BOP-ASK-Bench-spatial_reasoning',
            'prompt': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Is the object above the table?'}
            ],
            'images': [{'image': 'data:image/jpeg;base64,/9j/4AAQ...'}],
            'reward_model': {'ground_truth': 'Yes'},
            'extra_info': {
                'question': 'Is the object above the table?',
                'index': 42,
                'answer': 'Yes'
            }
        }
        
        with patch.object(self.evaluator, 'load_image_from_row') as mock_load:
            mock_load.return_value = Image.new('RGB', (100, 100))
            parsed = self.evaluator.parse_example(example)
        
        self.assertEqual(parsed['metadata']['original_question'], 'Is the object above the table?')
        self.assertEqual(parsed['metadata']['question_type'], 'spatial_reasoning')
        self.assertEqual(parsed['metadata']['index'], 42)
    
    def test_think_output_mode_suppress(self):
        """Test think output mode suppression."""
        evaluator = BopAskEvaluator(self.mock_model, think_output_mode='suppress')
        
        example = {
            'data_source': 'BOP-ASK-Bench-pose',
            'prompt': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Predict points.'}
            ],
            'images': [{'image': 'data:image/jpeg;base64,/9j/4AAQ...'}],
            'reward_model': {'ground_truth': '[(0.1, 0.1)]'},
            'extra_info': {'question': 'Predict points.', 'index': 0}
        }
        
        with patch.object(evaluator, 'load_image_from_row') as mock_load:
            mock_load.return_value = Image.new('RGB', (100, 100))
            parsed = evaluator.parse_example(example)
        
        self.assertIn('NEVER say ANYTHING', parsed['system_prompt'])
    
    def test_think_output_mode_force(self):
        """Test think output mode forcing."""
        evaluator = BopAskEvaluator(self.mock_model, think_output_mode='force')
        
        example = {
            'data_source': 'BOP-ASK-Bench-pose',
            'prompt': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Predict points.'}
            ],
            'images': [{'image': 'data:image/jpeg;base64,/9j/4AAQ...'}],
            'reward_model': {'ground_truth': '[(0.1, 0.1)]'},
            'extra_info': {'question': 'Predict points.', 'index': 0}
        }
        
        with patch.object(evaluator, 'load_image_from_row') as mock_load:
            mock_load.return_value = Image.new('RGB', (100, 100))
            parsed = evaluator.parse_example(example)
        
        self.assertIn('ALWAYS say your full reasoning', parsed['system_prompt'])
        self.assertIn('<think>', parsed['system_prompt'])
    
    def test_score_prediction_pose(self):
        """Test scoring pose estimation predictions."""
        parsed_example = {
            'ground_truth': '[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]',
            'metadata': {'question_type': 'pose', 'index': 0}
        }
        
        # Exact match
        prediction = '[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]'
        result = self.evaluator.score_prediction(prediction, parsed_example)
        
        self.assertIn('score', result)
        self.assertIn('acc', result)
        self.assertIn('fmt', result)
        self.assertGreater(result['score'], 0.9)  # Should be very high IoU
    
    def test_score_prediction_spatial_reasoning(self):
        """Test scoring spatial reasoning predictions."""
        parsed_example = {
            'ground_truth': 'Yes',
            'metadata': {'question_type': 'spatial_reasoning', 'index': 0}
        }
        
        # Correct answer
        prediction = '<answer>Yes</answer>'
        result = self.evaluator.score_prediction(prediction, parsed_example)
        
        self.assertEqual(result['score'], 1.0)
        self.assertEqual(result['acc'], 1.0)
        
        # Wrong answer
        prediction = '<answer>No</answer>'
        result = self.evaluator.score_prediction(prediction, parsed_example)
        
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['acc'], 0.0)
    
    def test_compute_custom_metrics(self):
        """Test computing per-type metrics."""
        results = [
            {'score': 1.0, 'acc': 1.0, 'question_type': 'spatial_reasoning'},
            {'score': 0.0, 'acc': 0.0, 'question_type': 'spatial_reasoning'},
            {'score': 0.8, 'acc': 0.8, 'question_type': 'pose'},
            {'score': 0.3, 'acc': 0.3, 'question_type': 'pose'},
        ]
        
        metrics = self.evaluator.compute_custom_metrics(results)
        
        self.assertIn('metrics_by_type', metrics)
        self.assertIn('spatial_reasoning', metrics['metrics_by_type'])
        self.assertIn('pose', metrics['metrics_by_type'])
        
        sr_metrics = metrics['metrics_by_type']['spatial_reasoning']
        self.assertEqual(sr_metrics['count'], 2)
        self.assertEqual(sr_metrics['average_score'], 0.5)
        
        pose_metrics = metrics['metrics_by_type']['pose']
        self.assertEqual(pose_metrics['count'], 2)
        self.assertEqual(pose_metrics['average_score'], 0.55)
    
    def test_should_include_in_training_data(self):
        """Test training data inclusion thresholds."""
        # Spatial reasoning: threshold > 0.5
        result_sr_pass = {'score': 0.6, 'question_type': 'spatial_reasoning'}
        result_sr_fail = {'score': 0.4, 'question_type': 'spatial_reasoning'}
        
        self.assertTrue(self.evaluator.should_include_in_training_data(result_sr_pass))
        self.assertFalse(self.evaluator.should_include_in_training_data(result_sr_fail))
        
        # Pose: threshold > 0.3 (more lenient)
        result_pose_pass = {'score': 0.4, 'question_type': 'pose'}
        result_pose_fail = {'score': 0.2, 'question_type': 'pose'}
        
        self.assertTrue(self.evaluator.should_include_in_training_data(result_pose_pass))
        self.assertFalse(self.evaluator.should_include_in_training_data(result_pose_fail))


class TestModels(unittest.TestCase):
    """Test model implementations."""
    
    def test_base_model_interface(self):
        """Test that BaseModel enforces the interface."""
        with self.assertRaises(TypeError):
            # Should not be able to instantiate abstract class
            BaseModel()
    
    def test_mock_model(self):
        """Test the mock model works correctly."""
        responses = {
            'yes_no': '#### no',
            'point': '#### [(0.7, 0.3)]',
            'bbox': '#### [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]'
        }
        
        model = MockModel(responses)
        
        # Test yes/no
        result = model.predict("Is it blue? Answer yes or no.")
        self.assertEqual(result, '#### no')
        
        # Test point
        result = model.predict("Pinpoint the center of the object.")
        self.assertEqual(result, '#### [(0.7, 0.3)]')
        
        # Test bounding box
        result = model.predict("Find all instances of tables.")
        self.assertEqual(result, '#### [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]')
        
        self.assertEqual(model.call_count, 3)
    
    def test_unified_model_interface(self):
        """Test that UnifiedModel can be instantiated with mock toolkit."""
        # Mock the LLM integration to avoid actual API calls
        with patch('evals.models.create_tool_agent') as mock_create:
            mock_integration = Mock()
            mock_integration.sessions = {}
            # Mock session object
            mock_session = Mock()
            mock_session.add_system_message = Mock()
            mock_session.add_user_message = Mock()
            mock_integration.create_session = Mock(return_value=mock_session)
            mock_integration.get_response = Mock(return_value={"response": "test response"})
            mock_integration.get_session_history = Mock(return_value={"messages": []})
            mock_integration.clear_session = Mock()
            mock_create.return_value = mock_integration
            
            # Test initialization
            model = UnifiedModel(
                toolkit=None,
                provider="openai",
                model="gpt-4o",
                enable_variables=False,
                max_iterations=1
            )
            
            self.assertEqual(model.provider, "openai")
            self.assertEqual(model.model_name, "gpt-4o")
            self.assertEqual(model.max_iterations, 1)
            
            # Test predict method
            result = model.predict("Test question")
            self.assertEqual(result, "test response")
            
            # Verify integration methods were called
            mock_integration.create_session.assert_called_once()
            mock_session.add_user_message.assert_called_once()
            mock_integration.get_response.assert_called_once()


if __name__ == '__main__':
    unittest.main()
