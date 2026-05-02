#!/usr/bin/env python3
"""Run the Isaac ROS manipulation pick-and-place demo through Toolshed tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toolshed import get_toolkit, shutdown_toolkit, start_toolkit
from toolshed.agent import create_tool_agent


DEFAULT_QUERY = """Replicate the Isaac Sim pick-and-place tutorial using the available Isaac ROS tools.

Use tool calls, not assumptions:
1. Prepare or inspect the pick-and-place workflow config.
   - Use use_ground_truth_pose_in_sim=false for this multi-object tutorial; the
     behavior tree needs the /get_objects server.
2. Check the Isaac Sim ROS bridge and ROS graph.
3. Start the pick-and-place workflow if the action server is not available.
   Use isaac_ros_launch.start_recipe with recipe_name=pick_and_place and
   manipulator_workflow_config set to the config path returned by the config tool.
4. Capture scene/perception state with render artifacts when possible.
5. Send the tutorial single-bin goal through the generic ROS action primitive
   as the first manipulation attempt. Use action_name=/multi_object_pick_and_place,
   action_type=isaac_ros_manipulation_interfaces/action/MultiObjectPickAndPlace,
   and this goal:
   {"target_poses":{"header":{"frame_id":"base_link"},"poses":[{"position":{"x":-0.25,"y":-0.35,"z":0.5},"orientation":{"x":-0.677772,"y":0.734752,"z":0.020993,"w":0.017994}}]},"class_ids":[],"mode":0}
6. Inspect the scene and relevant logs after the action returns.

Stop and report concrete missing resources if Isaac Sim, ROS 2, the Isaac ROS
container, camera topics, or the action server are not available."""


SYSTEM_PROMPT = """You are controlling an Isaac ROS manipulation demo through Toolshed tools.

Prefer the fundamental primitives when deciding what to do next:
- inspect graph resources before sending actions;
- check launch process state and logs after starting anything;
- render or snapshot camera topics before manipulation;
- use isaac_ros_launch.start_recipe for the workflow and isaac_ros_action.send_goal
  for /multi_object_pick_and_place unless shortcut tools were explicitly enabled;
- pass the prepared manipulator_workflow_config path when starting
  pick_and_place_workflow;
- treat missing ROS topics/actions as recoverable state and report the exact names;
- do not fabricate successful robot motion without a tool result.

The tutorial action is /multi_object_pick_and_place. This action is NVIDIA's
behavior-tree tutorial API, not a Toolshed script. The current source package
uses isaac_ros_manipulation_interfaces/action/MultiObjectPickAndPlace, while
some older docs show isaac_manipulator_interfaces/action/MultiObjectPickAndPlace.

For this multi-object pick-and-place behavior tree, keep use_ground_truth_pose_in_sim
false. Setting it true prevents the object-info servers from launching, and the
behavior tree will wait forever for /get_objects.

The known-good debug path starts with the single-bin pick-and-place goal on a
fresh scene. Multi-bin goals can mutate object/planning state before fallback,
so avoid multi-bin as the first manipulation attempt unless the user explicitly
asks for it."""


def print_step(step: dict[str, Any]) -> None:
    step_type = step.get("type", "unknown")
    status = step.get("status", "")
    message = step.get("message", "")

    if step_type == "reasoning" and message:
        print(f"\nAgent message:\n{message}", flush=True)
    elif step_type == "tool_decision":
        print(f"\n{status}", flush=True)
        for tool_call in step.get("tool_calls", []):
            arguments = tool_call.get("arguments") or {}
            print(f"  - {tool_call['name']} args={json.dumps(arguments, sort_keys=True)}", flush=True)
    elif step_type == "tool_result":
        print(f"  {status}", flush=True)
        result = step.get("full_result") or step.get("result") or ""
        if result:
            print(f"    result: {result[:1200]}", flush=True)
    elif step_type in {"tool_executing", "synthesizing"}:
        print(f"  {status}", flush=True)
    elif step_type == "complete" and message:
        print(f"\nFinal agent message:\n{message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Isaac ROS pick-and-place demo with an agent composing Toolshed tools."
    )
    parser.add_argument(
        "--config",
        default="configs/isaac_ros_manipulation_pick_place.json",
        help="Toolshed JSON config to load.",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("ISAAC_DEMO_PROVIDER", "nvidia"),
        choices=["openai", "nvidia", "nvidia_openai", "llm_gateway_openai", "anthropic", "bedrock", "sglang"],
        help="LLM provider.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("ISAAC_DEMO_MODEL"),
        help="Model name for the selected provider.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=16,
        help="Maximum agent tool-use iterations.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the Toolshed dashboard.",
    )
    parser.add_argument(
        "--cleanup-recipe",
        action="append",
        default=["pick_and_place_workflow"],
        help="Launch recipe to stop before shutdown. Repeatable. Use an empty value to skip.",
    )
    parser.add_argument(
        "--allow-tutorial-shortcuts",
        action="store_true",
        help="Expose tutorial convenience wrappers such as isaac_pick_place to the agent.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Agent task prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cleanup_recipes = [recipe for recipe in (args.cleanup_recipe or []) if recipe]
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open("r", encoding="utf-8") as fp:
        tool_configs = json.load(fp)

    if not args.allow_tutorial_shortcuts:
        tool_configs.pop("isaac_pick_place", None)

    print(f"Loading tool config from {config_path}")
    print(f"Starting toolkit with {len(tool_configs)} tools")
    if not args.allow_tutorial_shortcuts:
        print("Tutorial shortcut tools disabled; agent will use generic ROS primitives.")

    toolkit = None
    try:
        router = start_toolkit(tool_configs, detached=False, dashboard=args.dashboard)
        toolkit = get_toolkit()
        agent = create_tool_agent(toolkit, provider=args.provider, model=args.model)

        session_id = "isaac_ros_pick_place_demo"
        session = agent.create_session(session_id)
        session.add_system_message(SYSTEM_PROMPT)
        session.add_user_message(args.query)

        print(f"Running agent with provider={args.provider} model={args.model}")
        print("=" * 80)
        response = asyncio.run(
            agent.get_response_async(
                session_id,
                step_callback=print_step,
                max_iterations=args.max_iterations,
            )
        )

        print("\n" + "=" * 80)
        print("FINAL RESPONSE")
        print("-" * 80)
        print(response["response"])
        print("-" * 80)
        print(f"Iterations: {response['iterations']}")
        print(f"Tool calls: {response['tool_calls_made']}")
        agent.clear_session(session_id)
    finally:
        if toolkit is not None:
            for recipe in cleanup_recipes:
                try:
                    result = toolkit.call_tool("isaac_ros_launch", "stop_recipe", recipe)
                    print(f"Cleanup stopped {recipe}: {getattr(result, 'text', result)}", flush=True)
                except Exception as exc:
                    print(f"Cleanup could not stop {recipe}: {exc}", file=sys.stderr, flush=True)
        print("\nShutting down toolkit")
        shutdown_toolkit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
