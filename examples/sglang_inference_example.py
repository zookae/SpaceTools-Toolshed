#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Example: Using fine-tuned Qwen model via sglang with Toolshed.

This script demonstrates how to use a fine-tuned VLM model (e.g., Qwen2.5-VL)
hosted on an sglang server with Toolshed's vision tools for multi-turn
reasoning with tool calls.

This is a self-contained example that launches toolshed internally.

Prerequisites:
    1. Start sglang server first:
       python sglang_server_launch.py --model-path /path/to/model
    
    2. Run this example:
       python sglang_inference_example.py --image test.jpg

Example tasks:
    - Depth reasoning: "Which point is closer to the camera?"
    - Object detection: "Detect all instances of 'cup' in the image"
    - Spatial reasoning: "What is the distance between the red object and the blue object?"
"""

import argparse
import sys
from pathlib import Path

import ray
from PIL import Image

from toolshed import start_toolkit, get_toolkit
from toolshed.agent import create_tool_agent


def launch_toolshed():
    """
    Launch toolshed with a minimal set of vision tools.
    
    Returns:
        ToolkitClient instance
    """
    print("\n1. Starting Toolshed with vision tools...")
    print("   (This may take a minute on first run...)")
    
    # Configure tools - adjust paths as needed
    tool_configs = {
        'sam2': {
            'num_actors': 1,
            'resources': {'num_gpus': 0.2},
            'timeout': 600,
            'args': {
                'no_output_image': True,
                'no_output_vars': False,
            }
        },
        'depth_estimator': {
            'num_actors': 1,
            'resources': {'num_gpus': 0.2},
            'timeout': 600,
            'args': {
                # Uncomment and set if you have a custom checkpoint
                # 'checkpoint_path': '/path/to/depth_pro.pt',
                'no_output_image': True,
                'no_output_vars': False,
            }
        },
        'vision_ops': {
            'num_actors': 1,
            'resources': {'num_cpus': 1.0, 'num_gpus': 0},
            'timeout': 600,
            'args': {
                'no_output_image': True,
                'no_output_vars': False,
            }
        },
        # Uncomment if you have roborefer model
        # 'roborefer': {
        #     'num_actors': 1,
        #     'resources': {'num_gpus': 1.0},
        #     'timeout': 600,
        #     'args': {
        #         'model_path': '/path/to/RoboRefer-8B-SFT',
        #         'no_output_image': True,
        #         'no_output_vars': False,
        #     }
        # },
    }
    
    try:
        toolkit = start_toolkit(
            tool_configs,
            router_name='toolshed_router',
            namespace='toolshed',
            detached=False,
            dashboard=False
        )
        print("✓ Toolshed started successfully")
        return toolkit
    except Exception as e:
        print(f"✗ Failed to start Toolshed: {e}")
        raise


def run_inference_example(
    image_path: str,
    query: str,
    sglang_url: str = "http://localhost:30000/v1",
    system_prompt: str = None,
    max_iterations: int = 5,
):
    """
    Run inference with fine-tuned Qwen model via sglang.
    
    Args:
        image_path: Path to input image
        query: Question to ask about the image
        sglang_url: Base URL for sglang server
        system_prompt: Optional system prompt (default uses spatial reasoning prompt)
        max_iterations: Maximum tool call iterations
    """
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    print("=" * 80)
    print("SGLang + Toolshed Inference Example")
    print("=" * 80)
    
    # Try to connect to existing toolshed, or launch new one
    print("\n1. Setting up Toolshed...")
    router_actor = launch_toolshed()
    toolkit = get_toolkit(router_name="toolshed_router", namespace="toolshed")
    
    # Create sglang integration
    print(f"\n2. Creating sglang integration (server: {sglang_url})...")
    try:
        llm = create_tool_agent(
            toolkit,
            provider="sglang",
            model="Qwen2.5-VL-3B-Instruct",
            base_url=sglang_url
        )
        print("✓ Created sglang integration")
    except Exception as e:
        print(f"✗ Failed to create integration: {e}")
        print("\nMake sure sglang server is running:")
        print("  python sglang_server_launch.py --model-path /path/to/model")
        sys.exit(1)
    
    # Generate tool schemas
    print("\n3. Generating tool schemas from Toolshed...")
    llm.generate_schemas()
    available_tools = llm.get_available_tools()
    print(f"✓ Generated {len(available_tools)} tool schemas:")
    for tool in available_tools[:5]:  # Show first 5
        print(f"   - {tool}")
    if len(available_tools) > 5:
        print(f"   ... and {len(available_tools) - 5} more")
    
    # Load image
    print(f"\n4. Loading image: {image_path}")
    try:
        image = Image.open(image_path)
        print(f"✓ Loaded image ({image.size[0]}x{image.size[1]})")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        sys.exit(1)
    
    # Default system prompt for spatial reasoning
    if system_prompt is None:
        system_prompt = """You are an expert in 3D spatial reasoning for robotics. Given an image and a spatial reasoning question, follow this process:

1. First, think about the reasoning process as an internal monologue the first time you receive the question, and every time you receive new information.
Your reasoning process MUST be enclosed within <think> </think> tags.
2. After thinking, if you need additional information to answer the question, such as specific object location in the image or information about depth or geometry, call the appropriate vision tools.
3. When you receive a tool response that contains reasonable information, use that information to continue your analysis on the question.
4. Once no further visual analysis or tool calls are needed, you MUST provide your final answer inside <answer> and </answer> tags without detailed illustrations.

Example answer format: <answer> <Your final answer here> </answer>."""
    
    # Run inference
    print("\n5. Running inference...")
    print(f"Query: {query}")
    print("\n" + "-" * 80)
    
    try:
        # Create session
        session_id = "demo_session"
        session = llm.create_session(session_id, initial_images=[])
        
        # Add system prompt
        session.add_system_message(system_prompt)
        
        # Add user message with image
        session.add_user_message(query, images=[image])
        
        # Get response with steps for better visibility
        import asyncio
        result = asyncio.run(llm.get_response_async(
            session_id,
            max_iterations=max_iterations,
            track_steps=True
        ))
        
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"\nIterations: {result['iterations']}")
        print(f"Tool calls made: {result['tool_calls_made']}")
        
        # Show reasoning steps
        if 'reasoning' in result:
            print("\nReasoning process:")
            for i, r in enumerate(result['reasoning'], 1):
                print(f"\n[Step {i}]")
                print(r['content'][:500])
                if len(r['content']) > 500:
                    print("...")
        
        # Show tool calls
        if 'steps' in result:
            tool_steps = [s for s in result['steps'] if s['type'] == 'tool_result']
            if tool_steps:
                print("\nTool calls:")
                for step in tool_steps:
                    print(f"\n  {step['tool_name']}:")
                    print(f"    {step['result']}")
        
        # Show final answer
        print("\n" + "=" * 80)
        print("FINAL ANSWER")
        print("=" * 80)
        print(f"\n{result['response']}")
        print("\n" + "=" * 80)
        
        # Clean up
        llm.clear_session(session_id)
        
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✓ Inference complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned Qwen via sglang",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default="Which point is closer to the camera?",
        help="Question to ask about the image"
    )
    
    parser.add_argument(
        "--sglang-url",
        type=str,
        default="http://localhost:30000/v1",
        help="Base URL for sglang server (default: http://localhost:30000/v1)"
    )
    
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (default: spatial reasoning prompt)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum tool call iterations (default: 5)"
    )
    
    args = parser.parse_args()
    
    run_inference_example(
        image_path=args.image,
        query=args.query,
        sglang_url=args.sglang_url,
        system_prompt=args.system_prompt,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
