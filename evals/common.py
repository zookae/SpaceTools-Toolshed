# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Common evaluation infrastructure.

This module provides the base evaluator class and common utilities for running
evaluations across different datasets.
"""

import json
import logging
import os
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any, List, Optional
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import BaseModel
from toolshed.formats import convert_messages_to_verl, convert_messages_to_sharegpt, build_training_example

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Base class for dataset evaluators.
    
    Subclasses must implement:
    - parse_example: Convert dataset row to standardized format
    - score_prediction: Score model prediction against ground truth
    - get_result_metadata: Extract dataset-specific result metadata
    
    Optionally override:
    - compute_custom_metrics: Add dataset-specific metric aggregations
    - load_dataset: Use different dataset format (default: parquet)
    """
    
    def __init__(self, model: BaseModel,
                 save_conversations_dir: Optional[str] = None,
                 save_training_dir: Optional[str] = None):
        """
        Initialize the evaluator with a model.
        
        Args:
            model: Model instance that implements the BaseModel interface
            save_conversations_dir: Optional directory to save conversations in VeRL format
            save_training_dir: Optional directory to save training data in ShareGPT format
        """
        self.model = model
        self.save_conversations_dir = save_conversations_dir
        self.save_training_dir = save_training_dir
        
        # Track which examples we've already saved to avoid duplicates on resume
        self._saved_conversation_indices = set()
        self._saved_training_indices = set()
        
        # Enumeration counter for sequential image naming (avoids collisions)
        self._conversation_enum_counter = 0
        self._training_enum_counter = 0
        
        # Thread safety for parallel evaluation
        self._results_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()
    
    @abstractmethod
    def parse_example(self, row: Dict) -> Dict:
        """
        Parse a dataset row into standardized format.
        
        Args:
            row: Raw dataset row (dict from dataframe)
            
        Returns:
            Dictionary with keys:
                - question: str - The question/prompt to send to the model
                - ground_truth: Any - The ground truth answer
                - image: Optional[PIL.Image] - Image if present
                - system_prompt: str - System prompt for the model
                - metadata: Dict[str, Any] - Dataset-specific fields (e.g., index, type)
        """
        pass
    
    @abstractmethod
    def score_prediction(self, prediction: str, parsed_example: Dict, messages: Optional[List[Dict]] = None):
        """
        Score a model prediction against ground truth.
        
        Args:
            prediction: Model's output string
            parsed_example: Output from parse_example()
            messages: Optional conversation history (for parallel execution)
            
        Returns:
            float or Dict: Score between 0.0 and 1.0, or a dict with 'score' key (and optionally 'scoring_error')
        """
        pass
    
    @abstractmethod
    def get_result_metadata(self, parsed_example: Dict, score: float) -> Dict:
        """
        Extract dataset-specific metadata for results.
        
        Args:
            parsed_example: Output from parse_example()
            score: Score from score_prediction()
            
        Returns:
            Dict with dataset-specific fields like question_type, category, etc.
        """
        pass
    
    def compute_custom_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Compute dataset-specific metrics (e.g., per-type breakdown).
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dict with custom metrics to add to aggregated results
        """
        return {}
    
    def should_include_in_training_data(self, result: Dict[str, Any]) -> bool:
        """
        Determine whether a result should be included in training data.
        
        Default implementation includes all non-error results with score > 0.5.
        Override in subclass for dataset-specific filtering criteria.
        
        Args:
            result: Evaluation result dictionary
            
        Returns:
            True if result should be included in training data, False otherwise
        """
        score = result.get('score', 0.0)
        return score > 0.5
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Default implementation loads parquet files. Override for other formats.
        
        Args:
            dataset_path: Path to dataset file
            
        Returns:
            Pandas DataFrame
        """
        return pd.read_parquet(dataset_path)
    
    def evaluate(self, dataset_path: str, limit: Optional[int] = None,
                 verbose: bool = True,
                 checkpoint_data: Optional[Dict[str, Any]] = None,
                 checkpoint_path: Optional[str] = None,
                 checkpoint_interval: int = 10,
                 start_index: Optional[int] = None,
                 end_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset_path: Path to the dataset file
            limit: Optional limit on number of examples to evaluate
            verbose: Whether to show progress bar
            checkpoint_data: Optional checkpoint data to resume from
            checkpoint_path: Optional path to save checkpoints
            checkpoint_interval: Save checkpoint every N examples (default: 10)
            start_index: Optional start index for dataset slice (inclusive)
            end_index: Optional end index for dataset slice (exclusive)
            
        Returns:
            Dictionary containing evaluation results
        """
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        df = self.load_dataset(dataset_path)
        
        # Apply index-based slicing for parallel execution
        if start_index is not None or end_index is not None:
            start = start_index if start_index is not None else 0
            end = end_index if end_index is not None else len(df)
            df = df.iloc[start:end]
            logger.info(f"Sliced dataset to indices [{start}:{end}), {len(df)} examples")
        
        if limit:
            df = df.head(limit)
            logger.info(f"Limiting evaluation to {limit} examples")
        
        # Load existing results from checkpoint
        results = []
        processed_indices = set()
        if checkpoint_data and 'detailed_results' in checkpoint_data:
            results = checkpoint_data['detailed_results']
            # Build set of already-processed indices
            processed_indices = {r['index'] for r in results if 'index' in r}
            
            # Restore tracking state from checkpoint (fast resume)
            if 'tracking_state' in checkpoint_data:
                tracking = checkpoint_data['tracking_state']
                self._saved_conversation_indices = set(tracking.get('saved_conversation_indices', []))
                self._saved_training_indices = set(tracking.get('saved_training_indices', []))
                self._conversation_enum_counter = tracking.get('conversation_enum_counter', 0)
                self._training_enum_counter = tracking.get('training_enum_counter', 0)
                logger.info(f"Restored tracking state: {len(self._saved_conversation_indices)} conversations, "
                           f"{len(self._saved_training_indices)} training examples, "
                           f"counters at {self._conversation_enum_counter}/{self._training_enum_counter}")
            else:
                # Backward compatibility: old checkpoint without tracking state
                # Scan existing files to rebuild tracking (fallback, still much faster than regeneration)
                logger.warning("Checkpoint missing tracking_state (old format). Scanning existing output files...")
                self._rebuild_tracking_from_existing_files(step=0)
            
            # Detect and regenerate missing outputs (e.g., from interrupted regeneration)
            regenerated_any = False
            if self.save_conversations_dir or self.save_training_dir:
                missing_outputs = self._detect_missing_outputs(results)
                if missing_outputs:
                    logger.info(f"Detected {len(missing_outputs)} results with missing outputs. "
                               f"Will regenerate incrementally...")
                    self._regenerate_missing_outputs_incremental(
                        missing_outputs, 
                        dataset_path, 
                        step=0,
                        checkpoint_path=checkpoint_path,
                        checkpoint_interval=checkpoint_interval,
                        all_results=results
                    )
                    regenerated_any = True
                    
                    # Convert training JSONL to JSON format after regeneration
                    if self.save_training_dir:
                        self._convert_training_jsonl_to_json(step=0)
                    
                    # Save final checkpoint after regeneration to preserve updated tracking state
                    if checkpoint_path:
                        self._save_checkpoint(results, checkpoint_path)
                        logger.info(f"Final checkpoint saved after regenerating outputs")
            
            logger.info(f"Loaded {len(results)} existing results, will skip {len(processed_indices)} already-processed examples")
        else:
            # No checkpoint exists - clear any existing output files to start fresh
            if self.save_conversations_dir or self.save_training_dir:
                logger.info("No checkpoint found - clearing any existing output files to start fresh")
                self._clear_output_files(step=0)
                self._reset_tracking_state()
        
        iterator = tqdm(df.iterrows(), total=len(df), desc="Evaluating") if verbose else df.iterrows()
        examples_processed_since_checkpoint = 0
        
        for idx, row in iterator:
            # Get the example index from metadata (dataset-specific)
            example_idx = self._get_example_index(row, idx)
            
            # Skip if already processed
            if example_idx in processed_indices:
                if verbose:
                    iterator.set_postfix({"status": "skipped (cached)"})
                continue
            
            try:
                example_result = self._evaluate_example(row, example_idx)
                results.append(example_result)
                processed_indices.add(example_idx)
                examples_processed_since_checkpoint += 1
                
                # Save conversations and training data incrementally
                self._save_incremental_outputs([example_result])
                
                # Save checkpoint periodically
                if checkpoint_path and examples_processed_since_checkpoint >= checkpoint_interval:
                    self._save_checkpoint(results, checkpoint_path)
                    examples_processed_since_checkpoint = 0
                    if verbose:
                        iterator.set_postfix({"status": "checkpoint saved"})
                        
            except KeyboardInterrupt:
                # Allow clean exit on Ctrl+C
                raise
            except Exception as e:
                # Don't hide errors - let them crash for debugging
                logger.error(f"Error evaluating example {idx}: {e}")
                raise
        
        # Save final checkpoint if there were any new results
        if checkpoint_path and examples_processed_since_checkpoint > 0:
            self._save_checkpoint(results, checkpoint_path)
            logger.info(f"Final checkpoint saved with {len(results)} total results")
        
        # Compute metrics
        aggregated_results = self._aggregate_results(results)
        
        # Convert training JSONL to JSON format if training data was saved
        if self.save_training_dir:
            self._convert_training_jsonl_to_json(step=0)
        
        return aggregated_results
    
    def evaluate_parallel(self, dataset_path: str, num_workers: int = 4,
                          limit: Optional[int] = None,
                          verbose: bool = True,
                          checkpoint_data: Optional[Dict[str, Any]] = None,
                          checkpoint_path: Optional[str] = None,
                          checkpoint_interval: int = 10,
                          start_index: Optional[int] = None,
                          end_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset with parallel workers.
        
        Args:
            dataset_path: Path to the dataset file
            num_workers: Number of parallel conversation threads
            limit: Optional limit on number of examples to evaluate
            verbose: Whether to show progress bar
            checkpoint_data: Optional checkpoint data to resume from
            checkpoint_path: Optional path to save checkpoints
            checkpoint_interval: Save checkpoint every N examples (default: 10)
            start_index: Optional start index for dataset slice (inclusive)
            end_index: Optional end index for dataset slice (exclusive)
            
        Returns:
            Dictionary containing evaluation results
        """
        if num_workers <= 1:
            # Fall back to sequential evaluation
            return self.evaluate(dataset_path, limit, verbose, checkpoint_data,
                               checkpoint_path, checkpoint_interval, start_index, end_index)
        
        logger.info(f"Starting parallel evaluation with {num_workers} workers")
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        df = self.load_dataset(dataset_path)
        
        # Apply index-based slicing for parallel execution
        if start_index is not None or end_index is not None:
            start = start_index if start_index is not None else 0
            end = end_index if end_index is not None else len(df)
            df = df.iloc[start:end]
            logger.info(f"Sliced dataset to indices [{start}:{end}), {len(df)} examples")
        
        if limit:
            df = df.head(limit)
            logger.info(f"Limiting evaluation to {limit} examples")
        
        # Load existing results from checkpoint
        results = []
        processed_indices = set()
        if checkpoint_data and 'detailed_results' in checkpoint_data:
            results = checkpoint_data['detailed_results']
            # Build set of already-processed indices
            processed_indices = {r['index'] for r in results if 'index' in r}
            
            # Restore tracking state from checkpoint
            if 'tracking_state' in checkpoint_data:
                tracking = checkpoint_data['tracking_state']
                self._saved_conversation_indices = set(tracking.get('saved_conversation_indices', []))
                self._saved_training_indices = set(tracking.get('saved_training_indices', []))
                self._conversation_enum_counter = tracking.get('conversation_enum_counter', 0)
                self._training_enum_counter = tracking.get('training_enum_counter', 0)
                logger.info(f"Restored tracking state: {len(self._saved_conversation_indices)} conversations, "
                           f"{len(self._saved_training_indices)} training examples, "
                           f"counters at {self._conversation_enum_counter}/{self._training_enum_counter}")
            else:
                # Backward compatibility: old checkpoint without tracking state
                logger.warning("Checkpoint missing tracking_state (old format). Scanning existing output files...")
                self._rebuild_tracking_from_existing_files(step=0)
            
            # Detect and regenerate missing outputs
            regenerated_any = False
            if self.save_conversations_dir or self.save_training_dir:
                missing_outputs = self._detect_missing_outputs(results)
                if missing_outputs:
                    logger.info(f"Detected {len(missing_outputs)} results with missing outputs. "
                               f"Will regenerate incrementally...")
                    self._regenerate_missing_outputs_incremental(
                        missing_outputs, 
                        dataset_path, 
                        step=0,
                        checkpoint_path=checkpoint_path,
                        checkpoint_interval=checkpoint_interval,
                        all_results=results
                    )
                    regenerated_any = True
                    
                    # Convert training JSONL to JSON format after regeneration
                    if self.save_training_dir:
                        self._convert_training_jsonl_to_json(step=0)
                    
                    # Save final checkpoint after regeneration
                    if checkpoint_path:
                        self._save_checkpoint(results, checkpoint_path)
                        logger.info(f"Final checkpoint saved after regenerating outputs")
            
            logger.info(f"Loaded {len(results)} existing results, will skip {len(processed_indices)} already-processed examples")
        else:
            # No checkpoint exists - clear any existing output files
            if self.save_conversations_dir or self.save_training_dir:
                logger.info("No checkpoint found - clearing any existing output files to start fresh")
                self._clear_output_files(step=0)
                self._reset_tracking_state()
        
        # Build list of examples to process
        examples_to_process = []
        for idx, row in df.iterrows():
            example_idx = self._get_example_index(row, idx)
            if example_idx not in processed_indices:
                examples_to_process.append((idx, row, example_idx))
        
        if not examples_to_process:
            logger.info("All examples already processed, skipping evaluation")
            return self._aggregate_results(results)
        
        logger.info(f"Processing {len(examples_to_process)} examples with {num_workers} workers")
        
        # Shared state for tracking progress
        examples_processed_since_checkpoint = 0
        checkpoint_counter_lock = threading.Lock()
        
        # Worker function to process a single example
        def process_example(example_data):
            nonlocal examples_processed_since_checkpoint
            
            idx, row, example_idx = example_data
            
            # Create worker model instance (shares toolkit but independent session)
            worker_model = self._create_worker_model()
            
            try:
                # Evaluate the example
                example_result = self._evaluate_example_with_model(row, example_idx, worker_model)
                
                # Thread-safe result collection
                with self._results_lock:
                    results.append(example_result)
                    processed_indices.add(example_idx)
                    
                    # Save outputs incrementally
                    self._save_incremental_outputs([example_result])
                
                # Thread-safe checkpoint counter and saving
                with checkpoint_counter_lock:
                    examples_processed_since_checkpoint += 1
                    should_checkpoint = (checkpoint_path and 
                                       examples_processed_since_checkpoint >= checkpoint_interval)
                    
                    if should_checkpoint:
                        # Save checkpoint within the counter lock to ensure exact intervals
                        with self._checkpoint_lock:
                            self._save_checkpoint(results, checkpoint_path)
                        examples_processed_since_checkpoint = 0
                
                return {'success': True, 'index': example_idx}
                
            except KeyboardInterrupt:
                # Allow clean exit on Ctrl+C
                raise
            except Exception as e:
                # Don't hide errors - let them crash for debugging
                logger.error(f"Error evaluating example {idx}: {e}", exc_info=True)
                raise
        
        # Execute in parallel with progress tracking
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_example, ex) for ex in examples_to_process]
            
            # Progress bar tracking
            if verbose:
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Evaluating (parallel)"):
                    pass
            else:
                # Wait for completion without progress bar
                for future in as_completed(futures):
                    try:
                        future.result()  # Raise any exceptions
                    except Exception as e:
                        logger.error(f"Worker raised exception: {e}")
        
        # Save final checkpoint if there were any new results
        if checkpoint_path and examples_processed_since_checkpoint > 0:
            self._save_checkpoint(results, checkpoint_path)
            logger.info(f"Final checkpoint saved with {len(results)} total results")
        
        # Compute metrics
        aggregated_results = self._aggregate_results(results)
        
        # Convert training JSONL to JSON format if training data was saved
        if self.save_training_dir:
            self._convert_training_jsonl_to_json(step=0)
        
        return aggregated_results
    
    def _create_worker_model(self):
        """
        Create a new model instance for a worker thread.
        
        Each worker gets its own model instance to maintain independent sessions,
        but they share the same toolkit (which is thread-safe via Ray).
        
        Returns:
            New model instance with same configuration as self.model
        """
        # Import here to avoid circular dependency
        from .models import UnifiedModel
        
        # Check if the model is a UnifiedModel (most common case)
        if isinstance(self.model, UnifiedModel):
            # Create new instance with same configuration
            return UnifiedModel(
                toolkit=self.model.toolkit,  # Shared toolkit (thread-safe via Ray)
                provider=self.model.provider,
                model=self.model.model_name,
                enable_variables=self.model.integration.enable_variables if hasattr(self.model.integration, 'enable_variables') else True,
                max_iterations=self.model.max_iterations,
                use_web_demo_prompt=self.model.use_web_demo_prompt
            )
        else:
            # For other model types, log warning and return the original model
            # This may not be thread-safe, but we can't clone arbitrary models
            logger.warning(f"Cannot clone model of type {type(self.model).__name__} for parallel execution. "
                         f"Sharing single model instance across workers (may not be thread-safe).")
            return self.model
    
    def _evaluate_example_with_model(self, row: Dict, example_idx: int, model) -> Dict:
        """
        Evaluate a single example using a specific model instance.
        
        This is similar to _evaluate_example but takes a model parameter
        instead of using self.model, enabling parallel execution.
        
        Args:
            row: Dataset row
            example_idx: Example index
            model: Model instance to use for this evaluation
            
        Returns:
            Result dictionary
        """
        # Parse example using dataset-specific logic
        parsed = self.parse_example(row)
        
        # Get model prediction
        images = [parsed['image']] if parsed.get('image') is not None else None
        
        # Prepare prediction kwargs
        predict_kwargs = {
            'text': parsed['question'],
            'images': images,
            'system_prompt': parsed['system_prompt']
        }
        
        model_output = model.predict(**predict_kwargs)
        
        # Get conversation history before scoring (for parallel execution compatibility)
        messages = None
        if hasattr(model, 'get_conversation_history'):
            messages = model.get_conversation_history()
        
        # Score prediction (can return float or dict)
        # Pass messages to support parallel execution where model isn't accessible
        score_result = self.score_prediction(model_output, parsed, messages=messages)
        
        # Handle dict or float return value
        if isinstance(score_result, dict):
            score = score_result.get('score', 0.0)
            score_metadata = {k: v for k, v in score_result.items() if k != 'score'}
        else:
            score = score_result
            score_metadata = {}
        
        # Messages already captured before scoring (reuse the same variable)
        
        # Build result dictionary
        result = {
            'index': example_idx,
            'score': score,
            'model_output': model_output,
            'prompt': parsed['system_prompt']
        }
        
        # Add score metadata (e.g., scoring_error, acc, fmt)
        result.update(score_metadata)
        
        # Add dataset-specific metadata
        result.update(self.get_result_metadata(parsed, score))
        
        # Store image for conversation saving (prefix with _ to indicate internal use)
        if parsed.get('image') is not None:
            result['_image'] = parsed['image']
        elif parsed.get('_training_image') is not None:
            # Handle mock_robot_mode where image is not in prompt but needed for training
            result['_image'] = parsed['_training_image']
        
        # Add messages if available
        if messages:
            result['messages'] = messages
        
        return result
    
    def _get_example_index(self, row: Dict, fallback_idx: int) -> int:
        """
        Extract example index from row. Can be overridden for dataset-specific logic.
        
        Default implementation tries common patterns.
        """
        # Try common patterns
        if 'extra_info' in row:
            extra_info = row['extra_info']
            if isinstance(extra_info, dict) and 'index' in extra_info:
                return extra_info['index']
        if 'index' in row:
            return row['index']
        if 'id' in row:
            return row['id']
        return fallback_idx
    
    def _evaluate_example(self, row: Dict, example_idx: int) -> Dict:
        """Evaluate a single example using the parse/predict/score pipeline."""
        # Parse example using dataset-specific logic
        parsed = self.parse_example(row)
        
        # Get model prediction
        images = [parsed['image']] if parsed.get('image') is not None else None
        
        # Prepare prediction kwargs
        predict_kwargs = {
            'text': parsed['question'],
            'images': images,
            'system_prompt': parsed['system_prompt']
        }
        
        model_output = self.model.predict(**predict_kwargs)
        
        # Get conversation history before scoring
        messages = None
        if hasattr(self.model, 'get_conversation_history'):
            messages = self.model.get_conversation_history()
        
        # Score prediction (can return float or dict)
        score_result = self.score_prediction(model_output, parsed, messages=messages)
        
        # Handle dict or float return value
        if isinstance(score_result, dict):
            score = score_result.get('score', 0.0)
            score_metadata = {k: v for k, v in score_result.items() if k != 'score'}
        else:
            score = score_result
            score_metadata = {}
        
        # Messages already captured before scoring (reuse the same variable)
        
        # Build result dictionary
        result = {
            'index': example_idx,
            'score': score,
            'model_output': model_output,
            'prompt': parsed['system_prompt']
        }
        
        # Add score metadata (e.g., scoring_error, acc, fmt)
        result.update(score_metadata)
        
        # Add dataset-specific metadata
        result.update(self.get_result_metadata(parsed, score))
        
        # Store image for conversation saving (prefix with _ to indicate internal use)
        if parsed.get('image') is not None:
            result['_image'] = parsed['image']
        elif parsed.get('_training_image') is not None:
            # Handle mock_robot_mode where image is not in prompt but needed for training
            result['_image'] = parsed['_training_image']
        
        # Add messages if available
        if messages:
            result['messages'] = messages
        
        return result
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results into final metrics."""
        # Filter out errors - also exclude None scores and scoring_error results
        valid_results = [r for r in results if 'error' not in r and 'scoring_error' not in r and r.get('score') is not None]
        error_results = [r for r in results if 'error' in r or 'scoring_error' in r or r.get('score') is None]
        
        # Count specifically scoring errors
        scoring_error_results = [r for r in results if 'scoring_error' in r]
        
        if not valid_results:
            return {
                'num_examples': len(results),
                'num_errors': len(error_results),
                'num_scoring_errors': len(scoring_error_results),
                'average_score': 0.0,
                'accuracy': 0.0,
                'error_rate': 1.0,
                'detailed_results': results
            }
        
        scores = [r['score'] for r in valid_results]
        
        # Base metrics
        base_metrics = {
            'num_examples': len(results),
            'num_valid': len(valid_results),
            'num_errors': len(error_results),
            'num_scoring_errors': len(scoring_error_results),
            'error_rate': len(error_results) / len(results) if results else None,
            'average_score': sum(scores) / len(scores) if scores else None,
            'num_correct': sum(1 for r in valid_results if r.get('acc', r['score']) > 0.5),
            'accuracy': sum(1 for r in valid_results if r.get('acc', r['score']) > 0.5) / len(valid_results) if valid_results else None,
            'detailed_results': results
        }
        
        # Add custom metrics from subclass
        custom_metrics = self.compute_custom_metrics(valid_results)
        base_metrics.update(custom_metrics)
        
        return base_metrics
    
    def _save_checkpoint(self, results: List[Dict], checkpoint_path: str):
        """
        Save intermediate checkpoint during evaluation.
        
        Args:
            results: Current list of results
            checkpoint_path: Path to save checkpoint
        """
        # Compute partial metrics for the checkpoint
        partial_metrics = self._aggregate_results(results)
        
        # Prepare checkpoint data
        detailed_results = []
        for r in results:
            # Create a copy without internal fields (like _image)
            result_copy = {k: v for k, v in r.items() if not k.startswith('_')}
            detailed_results.append(result_copy)
        
        checkpoint_data = {
            'summary': {k: v for k, v in partial_metrics.items() if k != 'detailed_results'},
            'detailed_results': detailed_results,
            'checkpoint': True,
            'checkpoint_time': datetime.now().isoformat(),
            'tracking_state': {
                'saved_conversation_indices': list(self._saved_conversation_indices),
                'saved_training_indices': list(self._saved_training_indices),
                'conversation_enum_counter': self._conversation_enum_counter,
                'training_enum_counter': self._training_enum_counter
            }
        }
        
        # Save atomically by writing to temp file then renaming
        temp_path = checkpoint_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        os.replace(temp_path, checkpoint_path)
        
        logger.debug(f"Checkpoint saved: {len(results)} results to {checkpoint_path}")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to a JSON file."""
        # Remove detailed results for summary
        summary = {k: v for k, v in results.items() if k != 'detailed_results'}
        
        # Remove internal fields from detailed results
        detailed_results = []
        for r in results['detailed_results']:
            result_copy = {k: v for k, v in r.items() if not k.startswith('_')}
            detailed_results.append(result_copy)
        
        output_data = {
            'summary': summary,
            'detailed_results': detailed_results,
            'tracking_state': {
                'saved_conversation_indices': list(self._saved_conversation_indices),
                'saved_training_indices': list(self._saved_training_indices),
                'conversation_enum_counter': self._conversation_enum_counter,
                'training_enum_counter': self._training_enum_counter
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    # ============================================================================
    # Image Loading Utilities
    # ============================================================================
    
    def load_image_from_data_uri(self, data_uri: str) -> Image.Image:
        """Convert data URI to PIL Image."""
        header, data = data_uri.split(',', 1)
        image_data = base64.b64decode(data)
        return Image.open(io.BytesIO(image_data))
    
    def load_image_from_path(self, path: str) -> Image.Image:
        """Load image from file path."""
        return Image.open(path)
    
    def load_image_from_row(self, row: Dict) -> Optional[Image.Image]:
        """
        Load image from a dataset row.
        
        Handles common formats:
        - Dict: {'image': 'data:image/jpeg;base64,...'}
        - String: 'data:image/...' or '/path/to/image.jpg'
        
        Args:
            row: Dataset row with 'images' field
            
        Returns:
            PIL Image or None if not found
        """
        if 'images' not in row or not row['images'] or len(row['images']) == 0:
            return None
        
        image_field = row['images'][0]
        
        if isinstance(image_field, dict):
            # Dict format: {'image': 'data:image/jpeg;base64,...'}
            image_data = image_field.get('image')
            if image_data:
                return self.load_image_from_data_uri(image_data)
        elif isinstance(image_field, str):
            # String format: '/path/to/image.jpg' or 'data:image/...'
            if image_field.startswith('data:image'):
                return self.load_image_from_data_uri(image_field)
            else:
                # File path - load from disk
                return self.load_image_from_path(image_field)
        else:
            raise ValueError(f"Unknown image field format (type: {type(image_field)}). Expected dict or str, got: {image_field}")
        
        return None
    
    # ============================================================================
    # Incremental Output Saving
    # ============================================================================
    
    def _save_incremental_outputs(self, new_results: List[Dict]):
        """
        Save conversations and training data incrementally for new results.
        
        Args:
            new_results: List of newly processed results (typically just one)
        """
        for result in new_results:
            if self.save_conversations_dir:
                self._save_single_conversation_result(result, step=0)
            if self.save_training_dir:
                self._save_single_training_result(result, step=0)
    
    def _save_single_conversation_result(self, result: Dict[str, Any], step: int = 0):
        """Save a single conversation result incrementally."""
        if not self.save_conversations_dir:
            return
        
        # Skip if already saved or has errors (including scoring errors)
        result_idx = result.get('index', -1)
        if result_idx in self._saved_conversation_indices:
            return
        if 'error' in result or 'scoring_error' in result or result.get('score') is None:
            return
        
        os.makedirs(self.save_conversations_dir, exist_ok=True)
        filename = os.path.join(self.save_conversations_dir, f"{step}.jsonl")
        images_dir = os.path.join(self.save_conversations_dir, f"images_{step}")
        os.makedirs(images_dir, exist_ok=True)
        
        # Assign enumeration index for this sample
        enum_idx = self._conversation_enum_counter
        self._conversation_enum_counter += 1
        
        # Build basic entry with metadata
        entry = {
            "input": result.get('question', ''),
            "output": result.get('model_output', ''),
            "score": result.get('score', 0.0),
            "step": step,
            "index": result_idx
        }
        
        # Add dataset-specific fields
        for key in ['question', 'question_type', 'ground_truth']:
            if key in result:
                entry[key] = result[key]
        
        # Save image if present
        image_filename = None
        if '_image' in result and result['_image'] is not None:
            try:
                img_filename = f"s{enum_idx:03d}_img0000.png"
                img_path = os.path.join(images_dir, img_filename)
                result['_image'].save(img_path)
                image_filename = f"images_{step}/{img_filename}"
                entry['image_path'] = image_filename
            except Exception as e:
                logger.warning(f"Could not save image for sample {result_idx}: {e}")
        
        # Convert conversation messages to VeRL format
        if 'messages' in result and result['messages']:
            verl_messages = convert_messages_to_verl(result['messages'])
            # Update image references
            if image_filename:
                for msg in verl_messages:
                    if 'content' in msg and isinstance(msg['content'], list):
                        for content_item in msg['content']:
                            if content_item.get('type') == 'image':
                                content_item['image_path'] = image_filename
            entry['messages'] = verl_messages
        
        # Append to file
        with open(filename, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        
        self._saved_conversation_indices.add(result_idx)
    
    def _save_single_training_result(self, result: Dict[str, Any], step: int = 0):
        """Save a single training data result incrementally."""
        if not self.save_training_dir:
            return
        
        # Skip if already saved or has errors (including scoring errors)
        result_idx = result.get('index', -1)
        if result_idx in self._saved_training_indices:
            return
        if 'error' in result or 'scoring_error' in result or result.get('score') is None:
            return
        
        # Check dataset-specific inclusion criteria
        if not self.should_include_in_training_data(result):
            return
        
        # Check if we have messages to convert
        if 'messages' not in result or not result['messages']:
            logger.warning(f"No messages found for example {result_idx}, skipping training data")
            return
        
        os.makedirs(self.save_training_dir, exist_ok=True)
        training_file = os.path.join(self.save_training_dir, f"train_{step}.jsonl")
        images_dir = os.path.join(self.save_training_dir, f"images_{step}")
        os.makedirs(images_dir, exist_ok=True)
        
        # Assign enumeration index
        enum_idx = self._training_enum_counter
        self._training_enum_counter += 1
        
        score = result.get('score', 0.0)
        
        # Save image if present and get path
        image_paths = []
        if '_image' in result and result['_image'] is not None:
            try:
                img_filename = f"s{enum_idx:03d}_img0000.png"
                img_full_path = os.path.join(images_dir, img_filename)
                result['_image'].save(img_full_path)
                image_paths.append(f"images_{step}/{img_filename}")
            except Exception as e:
                logger.warning(f"Could not save image for example {result_idx}: {e}")
        
        # Get tool schemas from model if available
        tool_schemas = self._get_tool_schemas_from_model()
        
        # Get original system prompt if available (for training data)
        system_prompt_override = result.get('system_prompt_original')
        
        # Build training example using format-specific builder
        try:
            example = build_training_example(
                messages=result['messages'],
                task_id=f"{self._get_dataset_name()}_{step}_{enum_idx:05d}",
                dataset_index=result_idx,
                score=score,
                image_paths=image_paths,
                tool_schemas=tool_schemas,
                system_prompt_override=system_prompt_override,
                # Pass additional dataset-specific fields
                ground_truth=result.get('ground_truth'),
                question_type=result.get('question_type')
            )
        except Exception as e:
            logger.error(f"Failed to build training example for {result_idx}: {e}")
            return
        
        # Append to JSONL
        with open(training_file, 'a') as f:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        self._saved_training_indices.add(result_idx)
    
    def _get_dataset_name(self) -> str:
        """Get dataset name for task IDs. Override in subclass if needed."""
        return self.__class__.__name__.replace('Evaluator', '').lower()
    
    def _get_tool_schemas_from_model(self) -> Optional[List[Dict]]:
        """
        Get tool schemas from the model if available.
        
        Returns:
            List of tool schema dicts or None if not available
        """
        # Check if model has integration with tool schemas
        if hasattr(self.model, 'integration') and self.model.integration:
            try:
                # Get raw schemas (OpenAI format) from integration
                return self.model.integration.get_raw_schemas()
            except Exception as e:
                logger.debug(f"Could not get tool schemas from model: {e}")
                return None
        return None
    
    def _convert_to_sharegpt_format(self, result: Dict) -> List[Dict]:
        """
        Convert evaluation result to ShareGPT format messages.
        
        Args:
            result: Evaluation result with messages
            
        Returns:
            List of ShareGPT formatted messages
        """
        # Get original messages from conversation history
        if 'messages' not in result or not result['messages']:
            # Fallback: create minimal messages from question/answer
            sharegpt_messages = []
            if 'prompt' in result:
                sharegpt_messages.append({
                    "role": "system",
                    "content": result['prompt']
                })
            if 'question' in result:
                sharegpt_messages.append({
                    "role": "user",
                    "content": result['question']
                })
            if 'model_output' in result:
                sharegpt_messages.append({
                    "role": "assistant",
                    "content": result['model_output']
                })
            return sharegpt_messages
        
        # Convert messages using format converter (old format)
        return convert_messages_to_sharegpt(result['messages'])
    
    # ============================================================================
    # Output File Management
    # ============================================================================
    
    def _clear_output_files(self, step: int = 0):
        """Clear/delete all output files for a given step."""
        import shutil
        
        if self.save_conversations_dir:
            conv_file = os.path.join(self.save_conversations_dir, f"{step}.jsonl")
            images_dir = os.path.join(self.save_conversations_dir, f"images_{step}")
            if os.path.exists(conv_file):
                os.remove(conv_file)
                logger.debug(f"Deleted {conv_file}")
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)
                logger.debug(f"Deleted {images_dir}")
        
        if self.save_training_dir:
            train_file = os.path.join(self.save_training_dir, f"train_{step}.jsonl")
            images_dir = os.path.join(self.save_training_dir, f"images_{step}")
            if os.path.exists(train_file):
                os.remove(train_file)
                logger.debug(f"Deleted {train_file}")
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)
                logger.debug(f"Deleted {images_dir}")
    
    def _reset_tracking_state(self):
        """Reset tracking indices and enumeration counters."""
        self._saved_conversation_indices = set()
        self._saved_training_indices = set()
        self._conversation_enum_counter = 0
        self._training_enum_counter = 0
    
    def _rebuild_tracking_from_existing_files(self, step: int = 0):
        """
        Rebuild tracking state by scanning existing output files (fallback for old checkpoints).
        
        This is much faster than regeneration since it only reads metadata, not images.
        
        Args:
            step: Step number for file naming (default: 0)
        """
        self._reset_tracking_state()
        
        # Scan conversation file
        if self.save_conversations_dir:
            conv_file = os.path.join(self.save_conversations_dir, f"{step}.jsonl")
            if os.path.exists(conv_file):
                with open(conv_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            idx = entry.get('index')
                            if idx is not None:
                                self._saved_conversation_indices.add(idx)
                        except Exception as e:
                            logger.debug(f"Error parsing conversation line: {e}")
                self._conversation_enum_counter = len(self._saved_conversation_indices)
                logger.info(f"Rebuilt conversation tracking: {len(self._saved_conversation_indices)} entries, "
                           f"counter at {self._conversation_enum_counter}")
        
        # Scan training file
        if self.save_training_dir:
            train_file = os.path.join(self.save_training_dir, f"train_{step}.jsonl")
            if os.path.exists(train_file):
                with open(train_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            # Training data stores dataset_index in metadata
                            idx = entry.get('dataset_index')
                            if idx is not None:
                                self._saved_training_indices.add(idx)
                        except Exception as e:
                            logger.debug(f"Error parsing training line: {e}")
                self._training_enum_counter = len(self._saved_training_indices)
                logger.info(f"Rebuilt training tracking: {len(self._saved_training_indices)} entries, "
                           f"counter at {self._training_enum_counter}")
    
    def _detect_missing_outputs(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect results that are missing from output files.
        
        Compares checkpoint results against tracking state to find results that
        should have outputs but don't (e.g., from interrupted regeneration).
        
        Args:
            results: List of result dictionaries from checkpoint
            
        Returns:
            List of results that need outputs regenerated
        """
        missing = []
        
        for result in results:
            # Skip errors and scoring errors
            if 'error' in result or 'scoring_error' in result or result.get('score') is None:
                continue
            
            result_idx = result.get('index', -1)
            needs_regeneration = False
            
            # Check if conversation output is missing
            if self.save_conversations_dir:
                if result_idx not in self._saved_conversation_indices:
                    needs_regeneration = True
            
            # Check if training output is missing (if it should be included)
            if self.save_training_dir:
                if self.should_include_in_training_data(result):
                    if result_idx not in self._saved_training_indices:
                        needs_regeneration = True
            
            if needs_regeneration:
                missing.append(result)
        
        return missing
    
    def _regenerate_missing_outputs_incremental(self, missing_results: List[Dict[str, Any]], 
                                                dataset_path: str, step: int = 0,
                                                checkpoint_path: Optional[str] = None,
                                                checkpoint_interval: int = 100,
                                                all_results: Optional[List[Dict]] = None):
        """
        Regenerate outputs for results missing from output files.
        
        This is incremental and interruptible - it checks tracking state before each save
        so it can be stopped and resumed across multiple runs. Saves checkpoint periodically
        during regeneration.
        
        Args:
            missing_results: List of results needing output regeneration
            dataset_path: Path to original dataset file
            step: Step number for file naming (default: 0)
            checkpoint_path: Optional path to save periodic checkpoints during regeneration
            checkpoint_interval: Save checkpoint every N regenerated results (default: 100)
            all_results: Full list of results for checkpoint saving
        """
        logger.info(f"Starting incremental regeneration for {len(missing_results)} results...")
        
        # Load dataset once for image lookup
        df = self.load_dataset(dataset_path)
        
        # Build index mapping for fast lookup
        dataset_by_index = {}
        for idx, row in df.iterrows():
            example_idx = self._get_example_index(row, idx)
            dataset_by_index[example_idx] = row
        
        # Regenerate each missing result (with progress tracking)
        regenerated_count = 0
        regenerated_since_checkpoint = 0
        
        for result in tqdm(missing_results, desc="Regenerating outputs", leave=False):
            result_idx = result.get('index')
            
            # Skip if we don't have the dataset row
            if result_idx not in dataset_by_index:
                logger.warning(f"Cannot regenerate index {result_idx}: not found in dataset")
                continue
            
            # Re-attach image from dataset
            dataset_row = dataset_by_index[result_idx]
            image = self.load_image_from_row(dataset_row)
            if image:
                result['_image'] = image
            
            # Save using existing single-save functions
            # These check tracking state internally and skip if already saved
            self._save_single_conversation_result(result, step)
            self._save_single_training_result(result, step)
            
            regenerated_count += 1
            regenerated_since_checkpoint += 1
            
            # Save checkpoint periodically during regeneration
            if checkpoint_path and all_results and regenerated_since_checkpoint >= checkpoint_interval:
                self._save_checkpoint(all_results, checkpoint_path)
                logger.info(f"Checkpoint saved during regeneration ({regenerated_count}/{len(missing_results)} done)")
                regenerated_since_checkpoint = 0
        
        logger.info(f"Regenerated outputs: {regenerated_count} results processed, "
                   f"{len(self._saved_conversation_indices)} total conversations, "
                   f"{len(self._saved_training_indices)} total training examples")
    
    def _convert_training_jsonl_to_json(self, step: int = 0):
        """
        Convert training JSONL file to nicely indented JSON format with a list.
        
        Args:
            step: Step number for file naming (default: 0)
        """
        if not self.save_training_dir:
            return
        
        jsonl_file = os.path.join(self.save_training_dir, f"train_{step}.jsonl")
        json_file = os.path.join(self.save_training_dir, f"train_{step}.json")
        
        # Check if JSONL file exists
        if not os.path.exists(jsonl_file):
            logger.debug(f"No training JSONL file found at {jsonl_file}, skipping conversion")
            return
        
        # Read all lines from JSONL
        examples = []
        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        examples.append(json.loads(line))
            
            # Write as nicely indented JSON array
            with open(json_file, 'w') as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Converted training data to JSON format: {json_file} ({len(examples)} examples)")
        except Exception as e:
            logger.warning(f"Failed to convert training JSONL to JSON: {e}")
    
    def regenerate_outputs_from_checkpoint(self, results: List[Dict[str, Any]],
                                          dataset_path: str, step: int = 0):
        """
        Regenerate all output files from checkpoint results.
        
        This provides a clean slate by deleting existing files and rebuilding.
        Images are re-loaded from the original dataset.
        
        Args:
            results: List of result dicts from checkpoint
            dataset_path: Path to original dataset file
            step: Step number for file naming (default: 0)
        """
        logger.info(f"Regenerating outputs from {len(results)} checkpoint results...")
        
        # Load dataset once for image lookup
        df = self.load_dataset(dataset_path)
        
        # Build index mapping for fast lookup
        dataset_by_index = {}
        for idx, row in df.iterrows():
            example_idx = self._get_example_index(row, idx)
            dataset_by_index[example_idx] = row
        
        # Clear existing output files
        self._clear_output_files(step)
        
        # Reset tracking state
        self._reset_tracking_state()
        
        # Regenerate each result
        for result in results:
            # Skip errors (including scoring errors)
            if 'error' in result or 'scoring_error' in result or result.get('score') is None:
                continue
            
            # Re-attach image from dataset
            dataset_idx = result.get('index')
            if dataset_idx in dataset_by_index:
                dataset_row = dataset_by_index[dataset_idx]
                image = self.load_image_from_row(dataset_row)
                if image:
                    result['_image'] = image
            
            # Save using existing single-save functions
            self._save_single_conversation_result(result, step)
            self._save_single_training_result(result, step)
        
        # Convert training JSONL to JSON format
        if self.save_training_dir:
            self._convert_training_jsonl_to_json(step)
        
        logger.info(f"Regenerated outputs: {len(self._saved_conversation_indices)} conversations, "
                   f"{len(self._saved_training_indices)} training examples")

