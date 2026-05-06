#!/usr/bin/env python3
"""Run the Isaac ROS pick-and-place tutorial with a fixed tool sequence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toolshed import get_toolkit, shutdown_toolkit, start_toolkit


SCRIPTED_TOOL_NAMES = {
    "isaac_ros_config",
    "isaac_ros_graph",
    "isaac_ros_launch",
    "isaac_ros_scene",
    "isaac_perception",
    "isaac_pick_place",
}

REQUIRED_TOPICS = [
    "/front_stereo_camera/left/image_raw",
    "/front_stereo_camera/depth/ground_truth",
]
REQUIRED_SERVICES = [
    "/get_objects",
    "/get_object_pose",
]
REQUIRED_ACTIONS = [
    "/multi_object_pick_and_place",
    "/cumotion/motion_plan",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Isaac ROS pick-and-place tutorial without an LLM."
    )
    parser.add_argument(
        "--config",
        default="configs/isaac_ros_manipulation_pick_place.json",
        help="Toolshed JSON config to load.",
    )
    parser.add_argument(
        "--output-summary",
        default="",
        help="Destination JSON summary. Defaults to <configured output_dir>/direct_summary.json.",
    )
    parser.add_argument(
        "--readiness-timeout-s",
        type=float,
        default=180.0,
        help="Timeout for ROS graph resources to appear.",
    )
    parser.add_argument(
        "--action-timeout-s",
        type=float,
        default=900.0,
        help="Timeout for the pick-and-place tutorial action.",
    )
    parser.add_argument(
        "--pre-action-settle-s",
        type=float,
        default=float(os.getenv("ISAAC_SCRIPTED_PRE_ACTION_SETTLE_S", "120")),
        help="Seconds to let ROS simulated time, TF, and joint states settle before the action.",
    )
    parser.add_argument(
        "--log-lines",
        type=int,
        default=500,
        help="Workflow log lines to include in the final summary.",
    )
    parser.add_argument(
        "--no-restart-workflow",
        action="store_true",
        help="Do not stop an already-running workflow before launch.",
    )
    parser.add_argument(
        "--all-config-tools",
        action="store_true",
        help="Start every tool from the config instead of the scripted subset.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the Toolshed dashboard.",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def value_ok(value: Any) -> bool:
    return bool(value.get("ok")) if isinstance(value, dict) else False


def configured_output_dir(tool_configs: dict[str, Any], config_path: Path) -> Path:
    for tool_config in tool_configs.values():
        args = tool_config.get("args", {}) if isinstance(tool_config, dict) else {}
        output_dir = args.get("output_dir")
        if output_dir:
            return Path(output_dir).resolve()
    return config_path.resolve().parent


def filtered_tool_config(tool_configs: dict[str, Any], include_all: bool) -> dict[str, Any]:
    if include_all:
        return tool_configs
    missing = sorted(SCRIPTED_TOOL_NAMES.difference(tool_configs))
    if missing:
        raise RuntimeError(f"Config is missing scripted tutorial tools: {', '.join(missing)}")
    return {name: tool_configs[name] for name in sorted(SCRIPTED_TOOL_NAMES)}


class ScriptedRun:
    def __init__(self, toolkit: Any, summary_path: Path) -> None:
        self.toolkit = toolkit
        self.summary_path = summary_path
        self.summary: dict[str, Any] = {
            "mode": "scripted_tutorial",
            "llm_api_called": False,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "steps": [],
        }

    def write_summary(self) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(jsonable(self.summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def call(self, label: str, tool: str, method: str, *args: Any, **kwargs: Any) -> Any:
        print(f"\n[{label}] {tool}.{method}", flush=True)
        if args:
            print(f"  args={json.dumps(jsonable(args), sort_keys=True)}", flush=True)
        if kwargs:
            print(f"  kwargs={json.dumps(jsonable(kwargs), sort_keys=True)}", flush=True)
        entry: dict[str, Any] = {
            "label": label,
            "tool": tool,
            "method": method,
            "args": jsonable(args),
            "kwargs": jsonable(kwargs),
        }
        self.summary["steps"].append(entry)
        try:
            result = self.toolkit.call_tool(tool, method, *args, **kwargs)
        except Exception as exc:
            entry.update(
                {
                    "ok": False,
                    "exception": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            self.write_summary()
            raise

        value = getattr(result, "value", result)
        text = getattr(result, "text", "")
        is_error = bool(getattr(result, "is_error", False))
        entry.update(
            {
                "ok": bool(value.get("ok", not is_error)) if isinstance(value, dict) else not is_error,
                "is_error": is_error,
                "text": text,
                "value": jsonable(value),
            }
        )
        if isinstance(value, dict):
            print(
                "  "
                + json.dumps(
                    {
                        "ok": value.get("ok"),
                        "state": value.get("state"),
                        "summary": value.get("summary"),
                        "missing": value.get("missing", []),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        elif text:
            print(f"  {text[:1000]}", flush=True)
        self.write_summary()
        return result


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    tool_configs = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = configured_output_dir(tool_configs, config_path)
    summary_path = Path(args.output_summary) if args.output_summary else output_dir / "direct_summary.json"
    run_tool_configs = filtered_tool_config(tool_configs, args.all_config_tools)

    print(f"Loading tool config from {config_path}")
    print(f"Starting scripted toolkit with {len(run_tool_configs)} tools")
    print(f"Writing scripted summary to {summary_path}")

    router = None
    toolkit = None
    runner: ScriptedRun | None = None
    workflow_started = False
    exit_code = 1
    try:
        router = start_toolkit(run_tool_configs, detached=False, dashboard=args.dashboard)
        toolkit = get_toolkit()
        runner = ScriptedRun(toolkit, summary_path)

        config = runner.call("prepare_config", "isaac_ros_config", "prepare_pick_and_place_config")
        config_value = config.value
        config_path_for_launch = config_value["artifacts"]["config_path"]

        runner.call("graph_before_launch", "isaac_ros_graph", "list_graph", include_tf=False)
        launch = runner.call(
            "start_workflow",
            "isaac_pick_place",
            "start_workflow",
            config_path=config_path_for_launch,
            restart=not args.no_restart_workflow,
        )
        workflow_started = bool(launch.value.get("ok"))

        readiness = runner.call(
            "wait_for_tutorial_resources",
            "isaac_ros_graph",
            "wait_for",
            required_topics=json.dumps(REQUIRED_TOPICS),
            required_services=json.dumps(REQUIRED_SERVICES),
            required_actions=json.dumps(REQUIRED_ACTIONS),
            timeout_s=args.readiness_timeout_s,
            poll_s=2.0,
        )
        scene_before = runner.call(
            "scene_before",
            "isaac_ros_scene",
            "check_ready",
            sample_timeout_s=3.0,
        )
        inspect_before = runner.call(
            "inspect_before",
            "isaac_perception",
            "inspect_scene",
        )
        workflow_outputs = runner.call(
            "workflow_outputs_before_action",
            "isaac_ros_launch",
            "inspect_recipe_outputs",
            "pick_and_place_workflow",
            sample_timeout_s=3.0,
            render_images=True,
        )
        action_server = runner.call(
            "watch_action_before_goal",
            "isaac_pick_place",
            "watch_goal",
        )
        if args.pre_action_settle_s > 0:
            print(
                f"\n[pre_action_settle] sleeping {args.pre_action_settle_s:.1f}s before tutorial action",
                flush=True,
            )
            runner.summary["pre_action_settle_s"] = args.pre_action_settle_s
            runner.write_summary()
            time.sleep(args.pre_action_settle_s)
        scene_at_action = runner.call(
            "scene_at_action",
            "isaac_ros_scene",
            "check_ready",
            sample_timeout_s=3.0,
        )
        action = runner.call(
            "send_single_bin_goal",
            "isaac_pick_place",
            "send_single_bin_goal",
            timeout_s=args.action_timeout_s,
        )
        inspect_after = runner.call(
            "inspect_after",
            "isaac_perception",
            "inspect_scene",
        )
        logs = runner.call(
            "workflow_logs",
            "isaac_ros_launch",
            "logs_recipe",
            "pick_and_place_workflow",
            max_lines=args.log_lines,
        )

        runner.summary.update(
            {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "config": jsonable(config.value),
                "readiness": jsonable(readiness.value),
                "scene_before": jsonable(scene_before.value),
                "inspect_before": jsonable(inspect_before.value),
                "workflow_outputs_before_action": jsonable(workflow_outputs.value),
                "action_server": jsonable(action_server.value),
                "scene_at_action": jsonable(scene_at_action.value),
                "action": jsonable(action.value),
                "inspect_after": jsonable(inspect_after.value),
                "logs": jsonable(logs.value),
            }
        )
        success = bool(
            value_ok(config.value)
            and value_ok(readiness.value)
            and value_ok(scene_before.value)
            and value_ok(inspect_before.value)
            and value_ok(workflow_outputs.value)
            and value_ok(action_server.value)
            and value_ok(scene_at_action.value)
            and value_ok(action.value)
            and value_ok(inspect_after.value)
        )
        runner.summary["success"] = success
        exit_code = 0 if success else 2
        return exit_code
    except Exception:
        if runner is not None:
            runner.summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            runner.summary["success"] = False
            runner.summary["traceback"] = traceback.format_exc()
            runner.write_summary()
        traceback.print_exc()
        return 1
    finally:
        if toolkit is not None:
            try:
                result = toolkit.call_tool("isaac_ros_launch", "stop_recipe", "pick_and_place_workflow")
                print(f"\n[cleanup] stop_recipe: {getattr(result, 'text', result)}", flush=True)
                if runner is not None:
                    runner.summary["cleanup"] = {"stop_recipe": jsonable(getattr(result, "value", result))}
            except Exception as exc:
                print(f"\n[cleanup] could not stop workflow: {exc}", file=sys.stderr, flush=True)
                if runner is not None:
                    runner.summary["cleanup"] = {
                        "stop_recipe": {"ok": False, "exception": repr(exc)},
                        "workflow_started": workflow_started,
                    }
            if runner is not None:
                runner.summary.setdefault("finished_at", datetime.now(timezone.utc).isoformat())
                runner.write_summary()
        print("\nShutting down toolkit")
        _ = router
        shutdown_toolkit()


if __name__ == "__main__":
    raise SystemExit(main())
