#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Example of using the unified LLM integration with step-by-step callbacks.

This script demonstrates how to use the LLM integration with real-time step callbacks
to monitor the model's progress. Supports both text-only and vision (image) queries.

Usage:
    # Text-only example with calculator tools:
    python examples/agentic.py --config configs/minimal.json "Compute 5! and repeat it backwards"

    # Vision example with image input:
    python examples/agentic.py --config configs/vision_full.json --image examples/media/example_image.jpg "What objects are in this image?"
"""

import os
import sys
import argparse
import asyncio
import json
from pathlib import Path

from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from toolshed import start_toolkit, get_toolkit, shutdown_toolkit
from toolshed.agent import create_tool_agent


def print_step(step):
    """Callback function to print step information to console."""
    step_type = step.get("type", "unknown")
    status = step.get("status", "")   # Integration-constructed status message
    message = step.get("message", "") # Actual LLM content (only in reasoning/complete steps)
    
    if step_type == "reasoning":
        # Print the full LLM message
        print(f"\n💭 Agent message:\n{message}")
    elif step_type == "tool_decision":
        print(f"\n{status}")
        tool_calls = step.get("tool_calls", [])
        for tc in tool_calls:
            print(f"  - {tc['name']}")
    elif step_type == "tool_executing":
        print(f"  {status}")
    elif step_type == "tool_result":
        print(f"  {status}")
    elif step_type == "synthesizing":
        print(f"\n{status}")
    elif step_type == "complete":
        # Print final LLM message
        print(f"\n💭 Final agent message:\n{message}")


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM integration example with step-by-step output"
    )
    parser.add_argument(
        "--provider",
        default="bedrock",
        choices=["openai", "anthropic", "bedrock"],
        help="LLM provider to use (default: bedrock)"
    )
    parser.add_argument(
        "--model",
        default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="Model to use (default: us.anthropic.claude-sonnet-4-5-20250929-v1:0)"
    )
    parser.add_argument(
        "--config",
        default="configs/minimal.json",
        help="Path to JSON config file for tool configurations (default: configs/minimal.json)"
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to image file for vision queries (optional)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Compute 5! and repeat it to me backwards",
        help="Query string to send to the model"
    )
    
    args = parser.parse_args()
    
    # Load tool config from file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return 1
    
    print(f"Loading tool config from {config_path}")
    with open(config_path, 'r') as f:
        tool_configs = json.load(f)
    
    # Load image if provided
    image = None
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return 1
        print(f"Loading image from {args.image}")
        image = Image.open(args.image)
        # Resize large images to avoid token limits
        max_size = (1024, 1024)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Initialize toolkit with tools from config
    print(f"Starting toolkit with tools: {list(tool_configs.keys())}")
    try:
        handle = start_toolkit(tool_configs, detached=False, dashboard=False)
        toolkit = get_toolkit()
        
        # Create LLM integration
        print(f"Initializing {args.provider} integration with model {args.model}")
        agent = create_tool_agent(
            toolkit,
            provider=args.provider,
            model=args.model
        )
        
        # Create a session with optional initial image
        session_id = "example_session"
        images = [image] if image else None
        session = agent.create_session(session_id, initial_images=images)
        
        # Add user message with optional image
        print(f"\nQuery: {args.query}")
        if image:
            print(f"Image: {args.image} ({image.width}x{image.height})")
        print("=" * 80)
        session.add_user_message(args.query, images=images)
        
        # Get response with real-time step callbacks
        response = asyncio.run(agent.get_response_async(
            session_id,
            step_callback=print_step,
            max_iterations=10
        ))
        
        # Print final response
        print("\n" + "=" * 80)
        print("\nFINAL RESPONSE:")
        print("-" * 80)
        print(response['response'])
        print("-" * 80)
        print(f"\nTotal iterations: {response['iterations']}")
        print(f"Total tool calls: {response['tool_calls_made']}")
        
        # Clean up
        agent.clear_session(session_id)
        
    finally:
        # Always shutdown toolkit
        print("\nShutting down toolkit...")
        shutdown_toolkit()
    
    return 0


if __name__ == "__main__":
    exit(main())
