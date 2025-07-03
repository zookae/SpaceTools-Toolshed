#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Compare OpenAI, Anthropic, and fine-tuned Qwen models on the same task.

This script demonstrates side-by-side comparison of different LLM providers
using the same Toolshed tools and the same input. Useful for evaluating
the performance of fine-tuned models against commercial APIs.

Prerequisites:
    1. For Qwen: Start sglang server
    2. Set API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY (optional)

Usage:
    # Compare all three models
    python compare_llm_models.py --image test.jpg

    # Compare specific models only
    python compare_llm_models.py --image test.jpg --models qwen openai

    # Use custom query
    python compare_llm_models.py --image test.jpg \\
        --query "What is the distance between the two objects?"
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import List, Optional

import ray
from PIL import Image

from toolshed import start_toolkit, get_toolkit
from toolshed.agent import create_tool_agent


SYSTEM_PROMPT = """You are an expert in 3D spatial reasoning for robotics. Given an image and a spatial reasoning question, follow this process:

1. First, think about the reasoning process as an internal monologue the first time you receive the question, and every time you receive new information.
Your reasoning process MUST be enclosed within <think> </think> tags.
2. After thinking, if you need additional information to answer the question, such as specific object location in the image or information about depth or geometry, call the appropriate vision tools.
3. When you receive a tool response that contains reasonable information, use that information to continue your analysis on the question.
4. Once no further visual analysis or tool calls are needed, you MUST provide your final answer inside <answer> and </answer> tags without detailed illustrations.

Example answer format: <answer> <Your final answer here> </answer>."""


def launch_toolshed():
    """Launch toolshed with minimal vision tools."""
    print("\nLaunching Toolshed with vision tools...")
    print("(This may take a minute on first run...)")
    
    tool_configs = {
        'sam2': {
            'num_actors': 1,
            'resources': {'num_gpus': 0.2},
            'timeout': 600,
            'args': {'no_output_image': True, 'no_output_vars': False}
        },
        'depth_estimator': {
            'num_actors': 1,
            'resources': {'num_gpus': 0.2},
            'timeout': 600,
            'args': {'no_output_image': True, 'no_output_vars': False}
        },
        'vision_ops': {
            'num_actors': 1,
            'resources': {'num_cpus': 1.0, 'num_gpus': 0},
            'timeout': 600,
            'args': {'no_output_image': True, 'no_output_vars': False}
        },
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


def compare_models(
    image_path: str,
    query: str,
    models: List[str],
    sglang_url: str = "http://localhost:30000/v1",
    max_iterations: int = 5,
):
    """
    Compare multiple LLM models on the same task.
    
    Args:
        image_path: Path to input image
        query: Question to ask
        models: List of model names to compare ('openai', 'anthropic', 'qwen')
        sglang_url: Base URL for sglang server (for qwen)
        max_iterations: Maximum tool call iterations
    """
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    print("=" * 80)
    print("LLM Model Comparison with Toolshed")
    print("=" * 80)
    
    # Connect to Toolshed or launch new one
    print("\n1. Setting up Toolshed...")
    try:
        toolkit = get_toolkit(router_name="toolshed_router", namespace="toolshed")
        print("✓ Connected to existing Toolshed router")
    except Exception as e:
        print("   No existing router found, launching new one...")
        toolkit = launch_toolshed()
    
    # Load image
    print(f"\n2. Loading image: {image_path}")
    try:
        image = Image.open(image_path)
        print(f"✓ Loaded image ({image.size[0]}x{image.size[1]})")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        sys.exit(1)
    
    # Create integrations for requested models
    print("\n3. Creating LLM integrations...")
    integrations = {}
    
    if 'openai' in models:
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠ Warning: OPENAI_API_KEY not set, skipping OpenAI")
        else:
            try:
                integrations['OpenAI (GPT-4o)'] = create_tool_agent(
                    toolkit,
                    provider="openai",
                    model="gpt-4o"
                )
                print("✓ Created OpenAI integration")
            except Exception as e:
                print(f"⚠ Failed to create OpenAI integration: {e}")
    
    if 'anthropic' in models:
        if not os.getenv('ANTHROPIC_API_KEY'):
            print("⚠ Warning: ANTHROPIC_API_KEY not set, skipping Anthropic")
        else:
            try:
                integrations['Anthropic (Claude Opus 4)'] = create_tool_agent(
                    toolkit,
                    provider="anthropic",
                    model="claude-opus-4-1-20250805"
                )
                print("✓ Created Anthropic integration")
            except Exception as e:
                print(f"⚠ Failed to create Anthropic integration: {e}")
    
    if 'qwen' in models or 'sglang' in models:
        try:
            integrations['Qwen2.5-VL-3B (fine-tuned)'] = create_tool_agent(
                toolkit,
                provider="sglang",
                model="Qwen2.5-VL-3B-Instruct",
                base_url=sglang_url
            )
            print("✓ Created SGLang/Qwen integration")
        except Exception as e:
            print(f"⚠ Failed to create SGLang integration: {e}")
            print(f"   Make sure sglang server is running at {sglang_url}")
    
    if not integrations:
        print("\n✗ No models available for comparison")
        sys.exit(1)
    
    # Generate schemas for all integrations
    print("\n4. Generating tool schemas...")
    for name, llm in integrations.items():
        llm.generate_schemas()
    print(f"✓ Generated schemas for {len(integrations)} model(s)")
    
    # Run inference for each model
    print("\n5. Running inference...")
    print(f"Query: {query}")
    print("\n" + "=" * 80)
    
    results = {}
    for name, llm in integrations.items():
        print(f"\n{'=' * 80}")
        print(f"Running: {name}")
        print('=' * 80)
        
        try:
            # Create session
            session_id = f"session_{name.replace(' ', '_').replace('(', '').replace(')', '')}"
            session = llm.create_session(session_id, initial_images=[])
            session.add_system_message(SYSTEM_PROMPT)
            session.add_user_message(query, images=[image])
            
            # Time the inference
            start_time = time.time()
            result = llm.get_response(session_id, max_iterations=max_iterations)
            elapsed_time = time.time() - start_time
            
            # Extract final answer (between <answer> tags if present)
            response_text = result['response']
            import re
            answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL | re.IGNORECASE)
            if answer_match:
                final_answer = answer_match.group(1).strip()
            else:
                # No answer tags, use full response
                final_answer = response_text.strip()
            
            results[name] = {
                'answer': final_answer,
                'full_response': response_text,
                'iterations': result['iterations'],
                'tool_calls': result['tool_calls_made'],
                'time': elapsed_time
            }
            
            print(f"✓ Completed in {elapsed_time:.2f}s")
            print(f"  Iterations: {result['iterations']}, Tool calls: {result['tool_calls_made']}")
            print(f"  Answer: {final_answer[:200]}...")
            
            # Clean up
            llm.clear_session(session_id)
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[name] = {'error': str(e)}
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    for name, result in results.items():
        print(f"\n{name}")
        print("-" * 40)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            continue
        
        print(f"Time: {result['time']:.2f}s")
        print(f"Iterations: {result['iterations']}")
        print(f"Tool calls: {result['tool_calls']}")
        print(f"\nFinal Answer:")
        print(result['answer'])
    
    print("\n" + "=" * 80)
    print("✓ Comparison complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM models on the same task with Toolshed",
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
        "--models",
        nargs='+',
        choices=['openai', 'anthropic', 'qwen', 'sglang', 'all'],
        default=['all'],
        help="Models to compare (default: all)"
    )
    
    parser.add_argument(
        "--sglang-url",
        type=str,
        default="http://localhost:30000/v1",
        help="Base URL for sglang server (default: http://localhost:30000/v1)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum tool call iterations (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Handle 'all' option
    if 'all' in args.models:
        models = ['openai', 'anthropic', 'qwen']
    else:
        models = args.models
    
    compare_models(
        image_path=args.image,
        query=args.query,
        models=models,
        sglang_url=args.sglang_url,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
