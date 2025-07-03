# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Command-line interface for running evaluations.

This script provides a simple way to evaluate models on various datasets
with flexible configuration options.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import os
from typing import Dict, Any, Optional

# Add toolshed to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from toolshed import start_toolkit, get_toolkit
from .models import UnifiedModel
from .robospatial import RoboSpatialEvaluator
from .refspatial import RefSpatialEvaluator
from .bop_ask import BopAskEvaluator
from .grasp_prediction import GraspPredictionEvaluator
from .spatialbench import SpatialBenchEvaluator
from .blink import BlinkEvaluator
from .cvbench import CVBenchEvaluator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate models on various datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate RoboSpatial with tools (OpenAI)
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider openai --model gpt-4o --use-tools
  
  # Evaluate RefSpatial with tools (OpenAI)
  python -m evals.run --dataset data/refspatial_multiturn/depth.parquet --dataset-type refspatial --provider openai --model gpt-4o --use-tools
  
  # Evaluate SpatialBench with tools (OpenAI)
  python -m evals.run --dataset data/spatialbench_positional/test.parquet --dataset-type spatialbench --provider openai --model gpt-4o --use-tools --think-output-mode force
  
  # Evaluate BLINK with tools (OpenAI)
  python -m evals.run --dataset data/blink_relative_depth/val.parquet --dataset-type blink --provider openai --model gpt-4o --use-tools --think-output-mode force
  
  # Evaluate CVBench with tools (OpenAI)
  python -m evals.run --dataset data/cvbench_3d_depth/test.parquet --dataset-type cvbench --provider openai --model gpt-4o --use-tools --think-output-mode force
  
  # Evaluate BOP-ASK with tools (OpenAI)
  python -m evals.run --dataset data/bop_ask_bench/pose.parquet --dataset-type bop_ask --provider openai --model gpt-4o --use-tools --think-output-mode force
  
  # Evaluate BOP-ASK with tools (Anthropic)
  python -m evals.run --dataset data/bop_ask_bench/spatial_reasoning.parquet --dataset-type bop_ask --provider anthropic --model claude-opus-4-1-20250805 --use-tools --think-output-mode suppress
  
  # Evaluate BOP-ASK grasp with mock robot
  python -m evals.run --dataset data/bop_ask_bench/grasp.parquet --dataset-type bop_ask --provider openai --model gpt-4o --use-tools --bop-mock-robot
  
  # Evaluate without tools (baseline)
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider openai --model gpt-4o
  
  # Evaluate with AWS Bedrock
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider bedrock --model us.anthropic.claude-sonnet-4-20250514-v1:0 --use-tools
  
  # Evaluate with NVIDIA LLM Gateway (OpenAI models)
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider llm_gateway_openai --model gpt-4o --use-tools
  
  # Evaluate with only specific tools
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider openai --model gpt-4o --use-tools --enable-tools vlm,code_executor
  
  # Save results, conversations, and training data to output directory
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider openai --model gpt-4o --use-tools --output ./results --save-conversations --save-training-data
  
  # Quick test with limit
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider openai --model gpt-4o-mini --limit 10
  
  # Resume evaluation from checkpoint (useful for cluster time limits)
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider openai --model gpt-4o --use-tools --output ./results --resume --checkpoint-interval 10
  
  # Run with 4 parallel workers to speed up evaluation (tool utilization)
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider openai --model gpt-4o --use-tools --num-workers 4
  
  # Run multiple instances in parallel (each with unique namespace and dataset partition)
  # Example: split 1000-example dataset into 4 parallel instances (250 examples each)
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider openai --model gpt-4o --use-tools --namespace toolshed_1 --router-name router_1 --output ./results_1 --start-index 0 --end-index 250 --enum-offset 0
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider openai --model gpt-4o --use-tools --namespace toolshed_2 --router-name router_2 --output ./results_2 --start-index 250 --end-index 500 --enum-offset 250
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider openai --model gpt-4o --use-tools --namespace toolshed_3 --router-name router_3 --output ./results_3 --start-index 500 --end-index 750 --enum-offset 500
  python -m evals.run --dataset data/robospatial_home/test.parquet --dataset-type robospatial --provider openai --model gpt-4o --use-tools --namespace toolshed_4 --router-name router_4 --output ./results_4 --start-index 750 --end-index 1000 --enum-offset 750
        """
    )
    
    parser.add_argument('--dataset', required=True, help='Path to dataset file')
    parser.add_argument('--dataset-type', required=True, choices=['robospatial', 'refspatial', 'bop_ask', 'grasp_prediction', 'spatialbench', 'blink', 'cvbench'],
                       help='Type of dataset to evaluate on')
    parser.add_argument('--provider', default='openai', choices=['openai', 'anthropic', 'bedrock', 'llm_gateway_openai'],
                       help='LLM provider (default: openai)')
    parser.add_argument('--model', default='gpt-4o',
                       help='Model name (default: gpt-4o)')
    parser.add_argument('--use-tools', action='store_true', 
                       help='Use toolshed tools (requires GPU for vision tools)')
    parser.add_argument('--output', help='Output directory for results (contains checkpoint.json, convos/, and training_data/)')
    parser.add_argument('--limit', type=int, help='Limit number of examples to evaluate')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    parser.add_argument('--toolkit-config', type=str, 
                       help='Path to custom toolkit configuration YAML')
    parser.add_argument('--no-variables', action='store_true',
                       help='Disable variable output from tools')
    parser.add_argument('--no-images', action='store_true',
                       help='Disable image output from tools')
    parser.add_argument('--max-iterations', type=int, default=20,
                       help='Maximum number of tool call iterations (default: 20)')
    parser.add_argument('--enable-tools', type=str,
                       help='Comma-separated list of tools to enable (e.g., vlm,code_executor). If not specified, all tools are enabled.')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if output directory exists (requires --output)')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N examples (default: 10)')
    parser.add_argument('--save-conversations', action='store_true',
                       help='Save conversation dumps (stored in convos/ subdirectory of --output)')
    parser.add_argument('--save-training-data', action='store_true',
                       help='Save training data in ShareGPT format (stored in training_data/ subdirectory of --output)')
    
    # Dataset-specific arguments
    parser.add_argument('--point-eval-method', default='convex_hull',
                       help='[RoboSpatial] Point evaluation method for scoring (default: convex_hull)')
    parser.add_argument('--think-output-mode', default=None, choices=['suppress', 'force'],
                       help='[RoboSpatial/RefSpatial/BOP-ASK/SpatialBench/BLINK/CVBench] Mode for thinking output (default: None). '
                       '"suppress" is recommended for Anthropic models as they already provide reasoning; '
                       '"force" is recommended for OpenAI models since they don\'t provide reasoning')
    parser.add_argument('--rs-system-prompt', default="default", choices=['default', 'expanded'],
                       help='[RoboSpatial/RefSpatial] System prompt to use for the model')
    parser.add_argument('--bop-system-prompt', default='default', choices=['default'],
                       help='[BOP-ASK] System prompt mode (default: default)')
    parser.add_argument('--bop-mock-robot', action='store_true',
                       help='[BOP-ASK] Enable mock robot mode for grasp execution (requires depth estimator). Only supports grasp questions.')
    parser.add_argument('--no-system-prompt', action='store_true',
                       help='[All datasets] Disable system prompt for the model. Applies to all datasets and providers.')
    parser.add_argument('--namespace', default='toolshed',
                       help='Ray namespace for toolkit isolation (default: toolshed). Use unique values to run multiple instances in parallel.')
    parser.add_argument('--router-name', default='toolshed_router',
                       help='Router actor name (default: toolshed_router). Use unique values to run multiple instances in parallel.')
    
    # Dataset partitioning for parallel execution
    parser.add_argument('--start-index', type=int, default=None,
                       help='Start index for dataset slice (inclusive). Use with --end-index to partition dataset across multiple instances.')
    parser.add_argument('--end-index', type=int, default=None,
                       help='End index for dataset slice (exclusive). Use with --start-index to partition dataset across multiple instances.')
    parser.add_argument('--enum-offset', type=int, default=0,
                       help='Offset for enumeration counters (for image file naming). Set to avoid collisions when running parallel instances. '
                       'Recommended: use start-index value as enum-offset.')
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Number of parallel conversation threads (default: 1). Higher values increase tool utilization and speed up evaluation. Recommended: match number of tool actors.')
    
    return parser.parse_args()


def get_default_toolkit_config(enable_variables=True, enable_images=True, enable_tools=None, 
                               router_name='toolshed_router', namespace='toolshed'):
    """Get default toolkit configuration for evaluation."""
    tools = {
        'roborefer': {
            'num_actors': 8,
            'resources': {'num_gpus': 0.6},
            "conda_env": "tool_roborefer",
            'timeout': 600,
            "args": {
                "model_path": "/lustre/fsw/portfolios/nvr/users/siyic/projects/RoboRefer/models/RoboRefer-8B-SFT",
                'no_output_image': not enable_images,
                'no_output_vars': not enable_variables,
                'exclude_methods': ['general_query'],
                'exclude_behavior': 'error'
            },
        },
        # 'vlm': {
        #     'num_actors': 1,
        #     'resources': {'num_gpus': 1.0},
        #     'conda_env': 'tool_vlm',
        #     'timeout': 600,
        #     'args': {
        #         'no_output_image': not enable_images,
        #         'no_output_vars': not enable_variables,
        #         'model_name': 'allenai/Molmo-7B-D-0924',
        #         'dtype': 'float16'
        #     }
        # },
        'sam2': {
            'num_actors': 8,
            'resources': {'num_gpus': 0.1},
            'conda_env': 'tool_sam2',
            'timeout': 600,
            'args': {
                'no_output_image': not enable_images,
                'no_output_vars': not enable_variables
            }
        },
        'depth_estimator': {
            'num_actors': 8,
            'resources': {'num_gpus': 0.1},
            'conda_env': 'tool_depth',
            'timeout': 600,
            'args': {
                'checkpoint_path': '/lustre/fsw/portfolios/nvr/users/vblukis/checkpoints/depth_pro.pt',
                'no_output_image': not enable_images,
                'no_output_vars': not enable_variables
            }
        },
        'code_executor': {
            'num_actors': 8,
            'resources': {'num_cpus': 1.0, 'num_gpus': 0},
            'conda_env': 'verlshed',  # Using the main toolshed environment
            'timeout': 600,
            'args': {
                'no_output_image': not enable_images,
                'no_output_vars': not enable_variables,
                'router_name': router_name,
                'namespace': namespace
            }
        },
        "bounding_box": {
            "num_actors": 8,
            "resources": {"num_gpus": 0},
            "conda_env": "tool_bbox",
            "args" : {
                "no_output_image": not enable_images,
                "no_output_vars": not enable_variables,
            }
        },
        "vision_ops": {
            "num_actors": 8,
            "resources": {"num_gpus": 0},
            "conda_env": "tool_depth",
            "args" : {
                "no_output_image": not enable_images,
                "no_output_vars": not enable_variables,
                "exclude_methods": ['mask_crop','point_crop', 'project_3d_points_to_2d', 'project_2d_points_to_3d', 'draw_polygon'],
            }
        },
        "grasp_generator": {
            "num_actors": 8,
            "resources": {"num_gpus": 0.1},
            "conda_env": "tool_graspgen",
            "timeout": 600,
            "args": {
                "gripper_config": "/home/vblukis/vb/code/toolshed/graspgen_franka_panda.yml",
                "no_output_image": not enable_images,
                "no_output_vars": not enable_variables
            }
        },
        "mock_robot": {
            "num_actors": 8,
            "resources": {"num_cpus": 0.0, "num_gpus": 0},
            "conda_env": "verlshed",
            "timeout": 600,
            "args": {
                "no_output_image": not enable_images,
                "no_output_vars": not enable_variables,
            }
        }
    }

    # Filter tools based on enable_tools if the filter is provided
    if enable_tools:
        tools = {tool_name: tools[tool_name] for tool_name in enable_tools if tool_name in tools}

    print(f"\nUsing tools: {[tool_name for tool_name in tools.keys()]}\n")

    return tools


def load_toolkit_config(config_path: str = None, enable_variables=True, enable_images=True, enable_tools=None,
                       router_name='toolshed_router', namespace='toolshed'):
    """Load toolkit configuration from file or use default."""
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Apply enable_variables and enable_images to loaded config
        for tool_name, tool_config in config.items():
            if 'args' in tool_config:
                tool_config['args']['no_output_vars'] = not enable_variables
                tool_config['args']['no_output_image'] = not enable_images
                # Add namespace/router_name to code_executor if present
                if tool_name == 'code_executor':
                    tool_config['args']['router_name'] = router_name
                    tool_config['args']['namespace'] = namespace
        
        # Filter tools based on enable_tools if specified
        if enable_tools:
            config = {tool_name: config[tool_name] for tool_name in enable_tools if tool_name in config}
        
        return config
    else:
        return get_default_toolkit_config(enable_variables, enable_images, enable_tools, router_name, namespace)


def initialize_toolkit(config_path: str = None, enable_variables=True, enable_images=True, enable_tools=None,
                      router_name='toolshed_router', namespace='toolshed'):
    """Initialize the toolkit with tools."""
    # Load configuration
    tool_config = load_toolkit_config(config_path, enable_variables, enable_images, enable_tools,
                                     router_name, namespace)
    
    # Start toolkit
    logger.info(f"Starting Toolshed with router_name='{router_name}', namespace='{namespace}':")
    for tool_name, config in tool_config.items():
        logger.info(f"  {tool_name}: {config.get('num_actors', 1)} actors, "
                   f"{config.get('resources', {})}")
    
    try:
        router = start_toolkit(tool_config, router_name=router_name, namespace=namespace,
                             detached=False, dashboard=False)
        logger.info("Toolshed started successfully")
        return router
    except Exception as e:
        logger.error(f"Failed to start toolkit: {e}")
        raise


def initialize_model(args):
    """
    Initialize model with or without toolkit based on arguments.
    
    Returns:
        tuple: (model, model_type, router)
    """
    if args.use_tools:
        logger.info(f"Initializing {args.provider}/{args.model} with Toolshed tools...")
        enable_variables = not args.no_variables
        enable_images = not args.no_images
        
        # Parse enabled tools if specified
        enable_tools = None
        if args.enable_tools:
            enable_tools = [tool.strip() for tool in args.enable_tools.split(',')]
            logger.info(f"Enabling only these tools: {enable_tools}")
        
        router = initialize_toolkit(args.toolkit_config, enable_variables, enable_images, enable_tools,
                                   args.router_name, args.namespace)
        toolkit = get_toolkit(router_name=args.router_name, namespace=args.namespace)
        model = UnifiedModel(
            toolkit=toolkit,
            provider=args.provider,
            model=args.model,
            enable_variables=enable_variables,
            max_iterations=args.max_iterations
        )
        model_type = "with_tools"
    else:
        logger.info(f"Initializing {args.provider}/{args.model} without tools...")
        model = UnifiedModel(
            toolkit=None,  # No toolkit = no tools
            provider=args.provider,
            model=args.model,
            enable_variables=False,
            max_iterations=1  # No tool calls needed
        )
        model_type = "no_tools"
        router = None
    
    return model, model_type, router

keep_alive = None
def create_evaluator(dataset_type: str, model, args):
    """
    Create evaluator instance based on dataset type.
    
    Args:
        dataset_type: Type of dataset ('robospatial', etc.)
        model: Initialized model instance
        args: Command line arguments
        
    Returns:
        Evaluator instance
    """
    # Setup output directories if requested
    conversations_dir = None
    training_dir = None
    
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.save_conversations:
            conversations_dir = output_dir / "convos"
            conversations_dir.mkdir(parents=True, exist_ok=True)
        
        if args.save_training_data:
            training_dir = output_dir / "training_data"
            training_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine system prompt based on --no-system-prompt flag
    if args.no_system_prompt:
        robospatial_system_prompt = None
        bop_ask_system_prompt = None
        blink_system_prompt = None
        cvbench_system_prompt = None
        grasp_system_prompt_mode = None
    else:
        robospatial_system_prompt = args.rs_system_prompt
        bop_ask_system_prompt = args.bop_system_prompt
        blink_system_prompt = 'default'
        cvbench_system_prompt = 'default'
        grasp_system_prompt_mode = 'default'
    
    # Create dataset-specific evaluator
    if dataset_type == 'robospatial':
        evaluator = RoboSpatialEvaluator(
            model,
            point_evaluation_method=args.point_eval_method,
            save_conversations_dir=str(conversations_dir) if conversations_dir else None,
            save_training_dir=str(training_dir) if training_dir else None,
            think_output_mode=args.think_output_mode,
            system_prompt=robospatial_system_prompt
        )
    elif dataset_type == 'refspatial':
        evaluator = RefSpatialEvaluator(
            model,
            save_conversations_dir=str(conversations_dir) if conversations_dir else None,
            save_training_dir=str(training_dir) if training_dir else None,
            think_output_mode=args.think_output_mode,
            system_prompt=robospatial_system_prompt
        )
    elif dataset_type == 'bop_ask':
        depth_toolkit = None
        
        # Setup depth toolkit if mock_robot mode enabled
        if args.bop_mock_robot:
            from .mock_depth_prep import initialize_depth_toolkit
            logger.info("Starting depth preprocessing toolkit for BOP-ASK mock robot mode...")
            depth_router, depth_toolkit = initialize_depth_toolkit()
            global keep_alive
            keep_alive = depth_router
            logger.info("Depth preprocessing toolkit ready")
        
        evaluator = BopAskEvaluator(
            model,
            save_conversations_dir=str(conversations_dir) if conversations_dir else None,
            save_training_dir=str(training_dir) if training_dir else None,
            system_prompt=bop_ask_system_prompt,
            think_output_mode=args.think_output_mode,
            mock_robot_mode=args.bop_mock_robot,
            depth_toolkit=depth_toolkit
        )
    elif dataset_type == 'grasp_prediction':
        # For grasp prediction, we need a separate depth toolkit for preprocessing
        # Start a separate toolkit with depth_estimator in different namespace
        depth_config = {
            'depth_estimator': {
                'num_actors': 8,
                'resources': {'num_gpus': 0.5},
                'conda_env': 'tool_depth',
                'timeout': 600,
                'args': {
                    'checkpoint_path': '/lustre/fsw/portfolios/nvr/users/vblukis/checkpoints/depth_pro.pt',
                    'no_output_image': True,  # Don't need viz for preprocessing
                    'no_output_vars': True,
                }
            }
        }
        
        logger.info("Starting depth preprocessing toolkit for grasp prediction...")
        depth_router = start_toolkit(
            depth_config, 
            router_name='depth_prep_router', 
            namespace='depth_prep',
            detached=False,
            dashboard=False
        )
        depth_toolkit = get_toolkit(router_name='depth_prep_router', namespace='depth_prep')
        logger.info("Depth preprocessing toolkit ready")
        
        evaluator = GraspPredictionEvaluator(
            model,
            depth_estimator_toolkit=(depth_router, depth_toolkit),
            save_conversations_dir=str(conversations_dir) if conversations_dir else None,
            save_training_dir=str(training_dir) if training_dir else None,
            system_prompt_mode=grasp_system_prompt_mode
        )
    elif dataset_type == 'spatialbench':
        evaluator = SpatialBenchEvaluator(
            model,
            save_conversations_dir=str(conversations_dir) if conversations_dir else None,
            save_training_dir=str(training_dir) if training_dir else None,
            think_output_mode=args.think_output_mode,
            system_prompt=robospatial_system_prompt
        )
    elif dataset_type == 'blink':
        evaluator = BlinkEvaluator(
            model,
            save_conversations_dir=str(conversations_dir) if conversations_dir else None,
            save_training_dir=str(training_dir) if training_dir else None,
            think_output_mode=args.think_output_mode,
            system_prompt=blink_system_prompt
        )
    elif dataset_type == 'cvbench':
        evaluator = CVBenchEvaluator(
            model,
            save_conversations_dir=str(conversations_dir) if conversations_dir else None,
            save_training_dir=str(training_dir) if training_dir else None,
            think_output_mode=args.think_output_mode,
            system_prompt=cvbench_system_prompt
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Set enumeration offset for parallel execution (must be done after instantiation)
    if args.enum_offset > 0:
        evaluator._conversation_enum_counter = args.enum_offset
        evaluator._training_enum_counter = args.enum_offset
        logger.info(f"Set enumeration offset to {args.enum_offset} for parallel execution")
    
    return evaluator


def load_checkpoint(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load existing results from checkpoint file.
    
    Args:
        output_dir: Path to output directory containing checkpoint.json
        
    Returns:
        Dictionary with existing results or None if file doesn't exist
    """
    checkpoint_file = Path(output_dir) / "checkpoint.json"
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        # Validate checkpoint structure
        if 'detailed_results' not in data:
            logger.warning(f"Checkpoint file {checkpoint_file} has invalid format, starting fresh")
            return None
        
        logger.info(f"Loaded checkpoint with {len(data['detailed_results'])} existing results")
        return data
    except Exception as e:
        logger.warning(f"Failed to load checkpoint from {checkpoint_file}: {e}, starting fresh")
        return None


def setup_logging(debug: bool):
    """Configure logging level based on debug flag."""
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('toolshed').setLevel(logging.INFO)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('toolshed').setLevel(logging.DEBUG)
        # Suppress noisy loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    setup_logging(args.debug)
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    # Initialize model
    model, model_type, router = initialize_model(args)
    
    # Create evaluator
    evaluator = create_evaluator(args.dataset_type, model, args)
    
    # Setup checkpoint path
    checkpoint_path = None
    if args.output:
        checkpoint_path = Path(args.output) / "checkpoint.json"
    
    # Handle resume from checkpoint
    checkpoint_data = None
    if args.resume:
        if not args.output:
            logger.error("--resume requires --output to be specified")
            sys.exit(1)
        checkpoint_data = load_checkpoint(args.output)
        if checkpoint_data:
            logger.info(f"Resuming from checkpoint with {len(checkpoint_data['detailed_results'])} completed examples")
    
    # Run evaluation
    logger.info(f"Starting evaluation on {dataset_path}")
    
    # Log dataset partitioning if specified
    if args.start_index is not None or args.end_index is not None:
        start_str = str(args.start_index) if args.start_index is not None else "0"
        end_str = str(args.end_index) if args.end_index is not None else "end"
        logger.info(f"Processing dataset slice: [{start_str}:{end_str})")
    
    # Choose evaluation method based on num_workers
    if args.num_workers > 1:
        logger.info(f"Using parallel evaluation with {args.num_workers} workers")
        results = evaluator.evaluate_parallel(
            str(dataset_path),
            num_workers=args.num_workers,
            limit=args.limit,
            verbose=not args.no_progress,
            checkpoint_data=checkpoint_data,
            checkpoint_path=str(checkpoint_path) if (args.resume and checkpoint_path) else None,
            checkpoint_interval=args.checkpoint_interval,
            start_index=args.start_index,
            end_index=args.end_index
        )
    else:
        results = evaluator.evaluate(
            str(dataset_path), 
            limit=args.limit,
            verbose=not args.no_progress,
            checkpoint_data=checkpoint_data,
            checkpoint_path=str(checkpoint_path) if (args.resume and checkpoint_path) else None,
            checkpoint_interval=args.checkpoint_interval,
            start_index=args.start_index,
            end_index=args.end_index
        )
    
    # Print summary (dataset-specific formatting)
    if hasattr(evaluator, 'print_results_summary'):
        evaluator.print_results_summary(results)
    else:
        # Fallback generic summary
        print(f"\nEvaluation complete: {results['accuracy']:.2%} accuracy, "
              f"{results['average_score']:.3f} avg score")
    
    # Save results if requested
    if args.output:
        # Add metadata
        results['metadata'] = {
            'provider': args.provider,
            'model': args.model,
            'model_type': model_type,
            'use_tools': args.use_tools,
            'enable_variables': not args.no_variables if args.use_tools else None,
            'enable_images': not args.no_images if args.use_tools else None,
            'dataset': str(dataset_path),
            'dataset_type': args.dataset_type,
            'timestamp': datetime.now().isoformat(),
            'limit': args.limit
        }
        
        evaluator.save_results(results, str(checkpoint_path))
        logger.info(f"Results saved to {checkpoint_path}")
        if args.save_conversations:
            logger.info(f"Conversations saved to {Path(args.output) / 'convos'}")
        if args.save_training_data:
            logger.info(f"Training data saved to {Path(args.output) / 'training_data'}")


if __name__ == '__main__':
    main()
