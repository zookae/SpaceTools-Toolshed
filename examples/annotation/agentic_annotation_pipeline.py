#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Multi-agent annotation pipeline for processing multiple images in parallel.

This script uses the agentic workflow to annotate multiple images concurrently,
finding 3D volumes of objects and saving them in JSON format.
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from toolshed import start_toolkit, get_toolkit, shutdown_toolkit
from toolshed.agent import create_tool_agent


ANNOTATION_PROMPT = """Use the available computer vision tools (pointing at objects, estimating depth, segmentation, bounding box fitting) to annotate the image and save the annotation using the save_data tool. Specifically, find the 3D volume of each object in the image and save it in a JSON format:
{"objects": [{"name": "<short name of object>", "volume": <volume in cm^3>}, ...]}.

If any object fails to be detected, or the bounding box cannot be fitted (e.g. mask is empty), skip that object. Do not eyeball things yourself, your job is to orchestrate the tools.
"""


# Global progress tracker
agent_progress: dict[int, dict] = {}  # {agent_id: {"status": str, "tools": list}}
progress_lock = asyncio.Lock()


def _print_progress():
    """Print the current progress summary."""
    print("=" * 60)
    print("PROGRESS")
    print("=" * 60)
    for aid in sorted(agent_progress.keys()):
        info = agent_progress[aid]
        status = info["status"]
        tools = info["tools"]
        tools_str = " -> ".join(tools) if tools else ""
        print(f"Agent {aid} - {status} : {tools_str}\n")
    print("=" * 60)


async def _update_progress(agent_id: int, tool_name: str):
    """Update progress for an agent and print summary."""
    async with progress_lock:
        agent_progress[agent_id]["tools"].append(tool_name)
        agent_progress[agent_id]["status"] = "RUNNING!"
        _print_progress()


async def _set_agent_status(agent_id: int, status: str):
    """Set status for an agent and print summary."""
    async with progress_lock:
        if agent_id not in agent_progress:
            agent_progress[agent_id] = {"status": status, "tools": []}
        else:
            agent_progress[agent_id]["status"] = status
        if status != "STARTING":
            _print_progress()


async def annotate_image(agent, image_path: str, image_id: int):
    """Annotate a single image using the agent."""
    session_id = f"annotation_{image_id}"
    unique_id = f"annotation_{image_id}"
    
    # Initialize progress tracking for this agent
    await _set_agent_status(image_id, "STARTING")
    
    try:
        def step_callback(step):
            if step.get("type") == "tool_result":
                tool_name = step.get("tool_name", "unknown")
                asyncio.create_task(_update_progress(image_id, tool_name))
        
        image = Image.open(image_path)
        max_size = (1024, 1024)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        session = agent.create_session(session_id, initial_images=[image])
        
        id_prompt = f"""When saving annotations using the save_data tool, you must use the unique identifier "{unique_id}" as the 'name' parameter."""
        session.add_system_message("You are a helpful assistant that annotates images using computer vision tools.")
        
        session.add_user_message(ANNOTATION_PROMPT + "\n" + id_prompt, images=[image])
        
        response = await agent.get_response_async(
            session_id,
            max_iterations=20,
            step_callback=step_callback
        )
        
        await _set_agent_status(image_id, "FINISHED")
        
        agent.clear_session(session_id)
        
        return {
            "image_id": image_id,
            "image_path": image_path,
            "response": response["response"],
            "iterations": response["iterations"],
            "tool_calls": response["tool_calls_made"]
        }
    except Exception as e:
        await _set_agent_status(image_id, f"FAILED: {e}")
        agent.clear_session(session_id)
        return {
            "image_id": image_id,
            "image_path": image_path,
            "error": str(e),
            "response": None,
            "iterations": 0,
            "tool_calls": 0
        }


async def main():
    parser = argparse.ArgumentParser(description="Multi-agent annotation pipeline")
    parser.add_argument(
        "--config",
        default="examples/annotation/annotation_config.json",
        help="Path to tool config JSON file"
    )
    parser.add_argument(
        "--provider",
        default="bedrock",
        choices=["openai", "anthropic", "bedrock"],
        help="LLM provider"
    )
    parser.add_argument(
        "--model",
        default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="Model name"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=4,
        help="Number of images to process"
    )
    parser.add_argument(
        "--save-path",
        default="outputs/annotation",
        help="Path to save annotations"
    )
    parser.add_argument(
        "--skip-launch",
        action="store_true",
        help="Skip launching toolshed (assumes it's already running)"
    )
    
    args = parser.parse_args()
    
    # Prepare image list
    image1_path = "examples/media/example_image.jpg"
    image_paths = [image1_path] * args.num_images
    
    print(f"Processing {len(image_paths)} images")
    
    try:
        if args.skip_launch:
            print("Connecting to existing toolshed instance...")
            toolkit = get_toolkit()
        else:
            # Load tool config
            config_path = Path(args.config)
            if not config_path.exists():
                print(f"Error: Config file not found at {config_path}")
                return 1
            
            with open(config_path, 'r') as f:
                tool_configs = json.load(f)
            
            # Update save_path in config (use absolute path for Ray workers)
            tool_configs["save_data"]["args"]["save_path"] = str(Path(args.save_path).resolve())
            
            print(f"Starting toolkit with {len(tool_configs)} tools...")
            handle = start_toolkit(tool_configs, detached=False, dashboard=True, dashboard_port=7001)
            toolkit = get_toolkit()
        
        print(f"Initializing {args.provider} agent with model {args.model}")
        agent = create_tool_agent(
            toolkit,
            provider=args.provider,
            model=args.model,
            enable_variables=True,
            inject_variable_instructions=True,
        )
        
        print(f"\nProcessing {len(image_paths)} images in parallel...")
        tasks = [
            annotate_image(agent, img_path, i)
            for i, img_path in enumerate(image_paths)
        ]
        
        results = await asyncio.gather(*tasks)
        
        print("\n" + "=" * 80)
        print("ANNOTATION RESULTS")
        print("=" * 80)
        for result in results:
            print(f"\nImage {result['image_id']}: {result['image_path']}")
            if result.get("error"):
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Iterations: {result['iterations']}")
                print(f"  Tool calls: {result['tool_calls']}")
                print(f"  Response preview: {result['response'][:200]}...")
        
        print(f"\nAll annotations saved to: {args.save_path}")
        
    finally:
        if not args.skip_launch:
            print("\nShutting down toolkit...")
            shutdown_toolkit()
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
