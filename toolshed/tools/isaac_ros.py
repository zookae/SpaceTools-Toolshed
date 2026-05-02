"""Model-callable Toolshed tools for Isaac ROS manipulation workflows."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from PIL import Image
import yaml

from toolshed.tool_result import ToolResult
from toolshed.tools.base import BaseTool, tool_method
from toolshed.tools.isaac_ros_core import (
    DEFAULT_ACTION_TYPE,
    DEFAULT_OUTPUT_DIR,
    CommandResult,
    LaunchRecipe,
    RECIPES,
    RECIPE_ALIASES,
    RosCommandRunner,
    parse_overrides,
    parse_pose,
    parse_string_list,
    recent_logs_from_results,
    tail_lines,
)


DEFAULT_SINGLE_BIN_DROP_POSE = [-0.25, -0.35, 0.50, -0.677772, 0.734752, 0.020993, 0.017994]
DEFAULT_MULTI_BIN_POSES = [
    [-0.25, -0.35, 0.50, -0.677772, 0.734752, 0.020993, 0.017994],
    [-0.25, 0.40, 0.40, -0.677772, 0.734752, 0.020993, 0.017994],
]
DEFAULT_MULTI_BIN_CLASS_IDS = ["22", "3"]

_DEPTH_RENDER_SCRIPT = r"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


topic = sys.argv[1]
output_path = Path(sys.argv[2])
timeout_s = float(sys.argv[3])
bridge = CvBridge()
captured = {"msg": None}


class CaptureNode(Node):
    def __init__(self):
        super().__init__("toolshed_depth_saver")
        self.create_subscription(Image, topic, self._callback, qos_profile_sensor_data)

    def _callback(self, msg):
        if captured["msg"] is None:
            captured["msg"] = msg


def normalize_depth(array):
    depth = np.asarray(array, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros(depth.shape, dtype=np.uint8)
    valid = depth[finite]
    lo = float(np.percentile(valid, 1))
    hi = float(np.percentile(valid, 99))
    if hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max())
    if hi <= lo:
        return np.zeros(depth.shape, dtype=np.uint8)
    scaled = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    scaled[~finite] = 0.0
    return (scaled * 255.0).astype(np.uint8)


rclpy.init()
node = CaptureNode()
deadline = time.monotonic() + timeout_s
try:
    while captured["msg"] is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if captured["msg"] is None:
        print(f"No image received on {topic} within {timeout_s}s", file=sys.stderr)
        sys.exit(2)
    msg = captured["msg"]
    array = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    rendered = normalize_depth(array)
    rgb = np.repeat(rendered[:, :, None], 3, axis=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(rgb).save(output_path)
    print(json.dumps({"encoding": msg.encoding, "height": msg.height, "width": msg.width, "output": str(output_path)}))
finally:
    node.destroy_node()
    rclpy.shutdown()
"""


def _runner_or_default(runner: Any | None) -> Any:
    return runner if runner is not None else RosCommandRunner()


def _stateful_payload(
    *,
    ok: bool,
    state: str,
    summary: str,
    missing: list[str] | None = None,
    next_suggested_actions: list[str] | None = None,
    artifacts: dict[str, Any] | None = None,
    recent_logs: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "ok": ok,
        "state": state,
        "summary": summary,
        "missing": missing or [],
        "next_suggested_actions": next_suggested_actions or [],
        "artifacts": artifacts or {},
        "recent_logs": recent_logs or [],
    }
    payload.update(extra)
    return payload


def _result_text(payload: dict[str, Any]) -> str:
    missing = payload.get("missing") or []
    suffix = f" Missing: {', '.join(missing)}." if missing else ""
    warnings = payload.get("warnings") or []
    warning_suffix = f" Warnings: {'; '.join(str(warning) for warning in warnings)}." if warnings else ""
    config = payload.get("config") or {}
    effective_keys = ("use_ground_truth_pose_in_sim", "enable_nvblox", "enable_rviz_visualization")
    effective_values = [f"{key}={config[key]}" for key in effective_keys if key in config]
    effective_suffix = f" Effective config: {', '.join(effective_values)}." if effective_values else ""
    text = f"{payload['summary']} State: {payload['state']}.{suffix}{warning_suffix}{effective_suffix}"
    critical = _agent_feedback_lines(payload)
    if critical:
        text += " Critical logs: " + " | ".join(critical)
    return text


def _agent_feedback_lines(payload: dict[str, Any], max_lines: int = 32) -> list[str]:
    lines: list[str] = []
    for key in ("workflow_error_summary", "error_summary", "action_error_summary"):
        for line in payload.get(key) or []:
            if line and line not in lines:
                lines.append(str(line))
    return lines[-max_lines:]


def _action_count(info_text: str, label: str) -> int:
    match = re.search(rf"{re.escape(label)}:\s*([0-9]+)", info_text)
    return int(match.group(1)) if match else 0


def _items_from_output(result: CommandResult) -> list[str]:
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _parse_int_list(value: str | list[int] | tuple[int, ...] | None) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [int(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _critical_log_lines(logs: list[str], max_lines: int = 80) -> list[str]:
    patterns = (
        "[ERROR]",
        "Traceback",
        "IndexError",
        "Exception",
        "CUDA",
        "joint_state was not received",
        "Failed to plan",
        "MoveItErrorCodes",
        "Error raised in execute callback",
        "Error message:",
        "[Plan To Grasp]",
        "[Read Grasp Poses]",
        "Goal finished with status:",
        "workflow_status:",
    )
    critical = []
    traceback_remaining = 0
    for line in logs:
        is_traceback_start = "Traceback (most recent call last):" in line
        if is_traceback_start:
            critical.append(line)
            traceback_remaining = 16
            continue
        if traceback_remaining > 0:
            critical.append(line)
            traceback_remaining -= 1
            if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*(Error|Exception):", line):
                traceback_remaining = 0
            continue
        if any(pattern in line for pattern in patterns):
            critical.append(line)
    return critical[-max_lines:]


def _logs_result_text(payload: dict[str, Any]) -> str:
    return _result_text(payload)


def _resolved_package(runner: Any, package_name: str) -> str:
    if hasattr(runner, "resolve_package"):
        return runner.resolve_package(package_name)
    return package_name


def _resolved_action_type(runner: Any, action_type: str) -> str:
    if hasattr(runner, "resolve_action_type"):
        return runner.resolve_action_type(action_type)
    return action_type


def _resolved_interface_type(runner: Any, interface_type: str) -> str:
    if hasattr(runner, "resolve_interface_type"):
        return runner.resolve_interface_type(interface_type)
    if hasattr(runner, "resolve_action_type"):
        return runner.resolve_action_type(interface_type)
    return interface_type


def _recipe_command(recipe: LaunchRecipe, overrides: Mapping[str, Any], runner: Any) -> list[str]:
    launch_args = dict(recipe.default_args)
    launch_args.update(overrides)
    package = _resolved_package(runner, recipe.package)

    if recipe.name == "pick_and_place_workflow":
        config_key = "manipulator_workflow_config"
        config_value = str(launch_args.get(config_key, ""))
        if config_value and "/" not in config_value and hasattr(runner, "package_share_path"):
            share_path = runner.package_share_path(recipe.package)
            if share_path:
                launch_args[config_key] = str(Path(share_path) / "params" / config_value)

    command = ["ros2", "launch", package, recipe.launch_file]
    command.extend(f"{key}:={value}" for key, value in launch_args.items())
    return command


def _container_default_config_path(runner: Any) -> str | None:
    return "/tmp/toolshed_pick_place_config.yaml" if getattr(runner, "container_name", None) else None


def _path_exists_for_runner(runner: Any, path: str) -> bool:
    if hasattr(runner, "read_file"):
        try:
            return runner.read_file(path, timeout_s=3).ok
        except TypeError:
            return runner.read_file(path).ok
        except Exception:
            return False
    return Path(path).exists()


def _read_default_workflow_config(runner: Any) -> dict[str, Any]:
    share_path = runner.package_share_path("isaac_ros_manipulation_bringup") if hasattr(runner, "package_share_path") else None
    candidate = str(Path(share_path) / "params" / "sim_launch_params.yaml") if share_path else ""
    if candidate and hasattr(runner, "read_file"):
        result = runner.read_file(candidate)
        if result.ok:
            return yaml.safe_load(result.stdout) or {}
    return {}


def _read_yaml_file(runner: Any, path: str | Path) -> dict[str, Any]:
    if hasattr(runner, "read_file"):
        result = runner.read_file(path)
        if result.ok:
            return yaml.safe_load(result.stdout) or {}
        return {}
    source = Path(path)
    if source.exists():
        return yaml.safe_load(source.read_text(errors="replace")) or {}
    return {}


def _artifact_paths(runner: Any, output_dir: Path, filename: str) -> tuple[str, Path]:
    host_path = output_dir / filename
    if getattr(runner, "container_name", None):
        return f"/tmp/{filename}", host_path
    return str(host_path), host_path


def _pick_place_params_dir(runner: Any) -> str:
    resolved = _resolved_package(runner, "isaac_ros_manipulation_pick_and_place")
    candidates = []
    if hasattr(runner, "package_share_path"):
        share_path = runner.package_share_path("isaac_ros_manipulation_pick_and_place")
        if share_path:
            candidates.append(str(Path(share_path) / "params"))
    candidates.append(f"/opt/ros/jazzy/share/{resolved}/params")
    candidates.append(f"/opt/ros/humble/share/{resolved}/params")
    for candidate in candidates:
        behavior_tree = Path(candidate) / "multi_object_pick_and_place_behavior_tree_params.yaml"
        if hasattr(runner, "path_exists"):
            if runner.path_exists(behavior_tree):
                return candidate
        elif behavior_tree.exists():
            return candidate
    return candidates[0]


def _robot_description_config_dir(runner: Any) -> str:
    resolved = _resolved_package(runner, "isaac_manipulator_robot_description")
    candidates = []
    if hasattr(runner, "package_share_path"):
        share_path = runner.package_share_path("isaac_manipulator_robot_description")
        if share_path:
            candidates.append(str(Path(share_path) / "config"))
    candidates.append(f"/opt/ros/jazzy/share/{resolved}/config")
    candidates.append(f"/opt/ros/humble/share/{resolved}/config")
    return candidates[0]


def _pose_array_pose(pose: list[float]) -> dict[str, Any]:
    x, y, z, qx, qy, qz, qw = pose
    return {
        "position": {"x": x, "y": y, "z": z},
        "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
    }


def build_single_bin_goal(
    drop_pose: str | list[float] | tuple[float, ...] | None = None,
    frame_id: str = "base_link",
) -> dict[str, Any]:
    """Build the pick-and-place action goal for single-bin mode."""
    pose = parse_pose(drop_pose or DEFAULT_SINGLE_BIN_DROP_POSE)
    return {
        "target_poses": {
            "header": {"frame_id": frame_id},
            "poses": [_pose_array_pose(pose)],
        },
        "class_ids": [],
        "mode": 0,
    }


def build_multi_bin_goal(
    target_poses: str | list[list[float]] | None = None,
    class_ids: str | list[str] | None = None,
    frame_id: str = "base_link",
) -> dict[str, Any]:
    """Build the pick-and-place action goal for multi-bin mode."""
    if isinstance(target_poses, str):
        target_poses = json.loads(target_poses)
    poses = target_poses or DEFAULT_MULTI_BIN_POSES
    parsed_poses = [parse_pose(pose) for pose in poses]
    parsed_class_ids = parse_string_list(class_ids) or DEFAULT_MULTI_BIN_CLASS_IDS
    if len(parsed_class_ids) != len(parsed_poses):
        raise ValueError("Multi-bin mode requires class_ids and target_poses to have equal length")
    return {
        "target_poses": {
            "header": {"frame_id": frame_id},
            "poses": [_pose_array_pose(pose) for pose in parsed_poses],
        },
        "class_ids": parsed_class_ids,
        "mode": 1,
    }


class _IsaacTool(BaseTool):
    def __init__(
        self,
        *,
        runner: Any | None = None,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        no_output_image: bool = False,
        no_output_vars: bool = False,
        exclude_methods: list[str] | None = None,
        exclude_behavior: str = "warn",
    ) -> None:
        super().__init__(no_output_image, no_output_vars, exclude_methods, exclude_behavior)
        self._runner = _runner_or_default(runner)
        self._output_dir = Path(output_dir)


class IsaacRosGraphTool(_IsaacTool):
    """Inspect the live ROS graph for agent decision making."""

    def get_name(self) -> str:
        return "isaac_ros_graph"

    @tool_method
    def list_graph(self, include_tf: bool = True) -> ToolResult[dict[str, Any]]:
        """
        List the currently visible ROS graph resources.

        [[if:text]]Text output: Summary of nodes, topics, services, actions, missing checks, and suggested next tool calls.[[/if:text]]

        Args:
            include_tf: Whether to run a best-effort TF frame query.

        Returns:
            dict: Observability payload with nodes, topics, services, actions, frames, logs, and next suggested actions.
        """
        commands = {
            "nodes": ["ros2", "node", "list"],
            "topics": ["ros2", "topic", "list"],
            "services": ["ros2", "service", "list"],
            "actions": ["ros2", "action", "list"],
        }
        results = {name: self._runner.run(args, timeout_s=10) for name, args in commands.items()}
        frames: list[str] = []
        if include_tf:
            tf_result = self._runner.run(["ros2", "run", "tf2_tools", "view_frames"], timeout_s=10)
            results["frames"] = tf_result
            frames = _items_from_output(tf_result)

        missing = [name for name, result in results.items() if not result.ok and name != "frames"]
        ok = not missing
        payload = _stateful_payload(
            ok=ok,
            state="ready" if ok else "partial",
            summary="ROS graph queried successfully" if ok else "ROS graph query is incomplete",
            missing=missing,
            next_suggested_actions=[
                "Call isaac_ros_scene.check_ready before starting manipulation.",
                "Call isaac_ros_launch.start_recipe for missing launch components.",
                "Call isaac_perception.inspect_scene before sending a pick-and-place goal.",
            ],
            recent_logs=recent_logs_from_results(list(results.values())),
            nodes=_items_from_output(results["nodes"]),
            topics=_items_from_output(results["topics"]),
            services=_items_from_output(results["services"]),
            actions=_items_from_output(results["actions"]),
            frames=frames,
        )
        return ToolResult(payload, text=_result_text(payload))

    @tool_method
    def wait_for(
        self,
        required_nodes: str = "",
        required_topics: str = "",
        required_services: str = "",
        required_actions: str = "",
        timeout_s: float = 30.0,
        poll_s: float = 2.0,
    ) -> ToolResult[dict[str, Any]]:
        """
        Wait until required ROS graph resources appear.

        [[if:text]]Text output: Which required resources are present or missing and what the agent should try next.[[/if:text]]

        Args:
            required_nodes: JSON array or comma-separated node names.
            required_topics: JSON array or comma-separated topic names.
            required_services: JSON array or comma-separated service names.
            required_actions: JSON array or comma-separated action names.
            timeout_s: Maximum wait time in seconds.
            poll_s: Poll interval in seconds.

        Returns:
            dict: Observability payload with current graph state and missing resources.
        """
        required = {
            "nodes": parse_string_list(required_nodes),
            "topics": parse_string_list(required_topics),
            "services": parse_string_list(required_services),
            "actions": parse_string_list(required_actions),
        }
        deadline = __import__("time").time() + float(timeout_s)
        graph: dict[str, Any] = {}
        missing: list[str] = []
        resource_category_corrections: list[dict[str, str]] = []
        while True:
            graph = self.list_graph(include_tf=False).value
            missing = []
            resource_category_corrections = []
            for kind, items in required.items():
                for item in items:
                    if item in graph.get(kind, []):
                        continue
                    if kind == "services" and item in graph.get("actions", []):
                        resource_category_corrections.append(
                            {"requested": f"services:{item}", "actual": f"actions:{item}"}
                        )
                        continue
                    missing.append(f"{kind}:{item}")
            if not missing or __import__("time").time() >= deadline:
                break
            __import__("time").sleep(float(poll_s))

        ok = not missing
        payload = _stateful_payload(
            ok=ok,
            state="ready" if ok else "waiting_timeout",
            summary="Required ROS resources are available" if ok else "Timed out waiting for ROS resources",
            missing=missing,
            next_suggested_actions=[
                "Inspect isaac_ros_launch.status_recipe for the launch component that should create the missing resource.",
                "Inspect isaac_ros_launch.logs_recipe for errors.",
            ],
            nodes=graph.get("nodes", []),
            topics=graph.get("topics", []),
            services=graph.get("services", []),
            actions=graph.get("actions", []),
            resource_category_corrections=resource_category_corrections,
        )
        return ToolResult(payload, text=_result_text(payload))


class IsaacRosLaunchTool(_IsaacTool):
    """Start, stop, and inspect registered Isaac ROS launch recipes."""

    def get_name(self) -> str:
        return "isaac_ros_launch"

    @tool_method
    def list_recipes(self) -> ToolResult[dict[str, Any]]:
        """
        List registered Isaac ROS launch recipes.

        [[if:text]]Text output: Recipe names and short summaries for the agent.[[/if:text]]

        Returns:
            dict: Registered recipes keyed by recipe name.
        """
        payload = _stateful_payload(
            ok=True,
            state="ready",
            summary=f"Found {len(RECIPES)} Isaac ROS launch recipes",
            next_suggested_actions=["Call describe_recipe for details or start_recipe to launch one."],
            recipes={name: recipe.to_dict() for name, recipe in RECIPES.items()},
        )
        return ToolResult(payload, text=_result_text(payload))

    @tool_method
    def describe_recipe(self, recipe_name: str) -> ToolResult[dict[str, Any]]:
        """
        Describe an Isaac ROS launch recipe.

        [[if:text]]Text output: Launch command, produced ROS resources, and render topics.[[/if:text]]

        Args:
            recipe_name: Name of the recipe to describe.

        Returns:
            dict: Recipe metadata and launch command.
        """
        recipe = _recipe(recipe_name)
        payload = _stateful_payload(
            ok=True,
            state="ready",
            summary=f"Recipe {recipe.name}: {recipe.summary}",
            next_suggested_actions=["Call start_recipe to launch this component."],
            recipe=recipe.to_dict(),
            command=_recipe_command(recipe, {}, self._runner),
        )
        return ToolResult(payload, text=_result_text(payload))

    @tool_method
    def start_recipe(
        self,
        recipe_name: str,
        overrides: str = "",
        restart: bool = False,
    ) -> ToolResult[dict[str, Any]]:
        """
        Start an Isaac ROS launch recipe.

        [[if:text]]Text output: Process state, launch command, produced resources, logs, and suggested next checks.[[/if:text]]

        Args:
            recipe_name: Name of the launch recipe to start.
            overrides: JSON object or comma-separated key=value launch-argument overrides.
            restart: Stop the recipe first if it is already running.

        Returns:
            dict: Observability payload with process state and recipe resources.
        """
        recipe = _recipe(recipe_name)
        if restart:
            self._runner.stop(recipe.name)
        parsed_overrides = parse_overrides(overrides)
        warnings: list[str] = []
        if recipe.name == "pick_and_place_workflow" and "manipulator_workflow_config" not in parsed_overrides:
            default_config_path = _container_default_config_path(self._runner)
            if default_config_path and _path_exists_for_runner(self._runner, default_config_path):
                parsed_overrides["manipulator_workflow_config"] = default_config_path
                warnings.append(
                    f"Using prepared pick-and-place config overlay {default_config_path}."
                )
            elif overrides:
                warnings.append(
                    "Pick-and-place workflow launch arguments are read from "
                    "manipulator_workflow_config; call prepare_pick_and_place_config "
                    "or pass manipulator_workflow_config explicitly."
                )
        command = _recipe_command(recipe, parsed_overrides, self._runner)
        compatibility_results = []
        if hasattr(self._runner, "ensure_moveit_compatibility"):
            compatibility_results.append(self._runner.ensure_moveit_compatibility())
        if recipe.name in {"pick_and_place_workflow", "cumotion"} and hasattr(
            self._runner, "ensure_cumotion_goalset_compatibility"
        ):
            compatibility_results.append(self._runner.ensure_cumotion_goalset_compatibility())
        log_path = self._recipe_log_path(recipe.name)
        try:
            state = self._runner.start(recipe.name, command, log_path)
            ok = state.running
            payload = _stateful_payload(
                ok=ok,
                state="running" if ok else "exited",
                summary=f"Started launch recipe {recipe.name}" if ok else f"Launch recipe {recipe.name} exited",
                next_suggested_actions=[
                    "Call status_recipe to verify readiness.",
                    "Call isaac_ros_graph.wait_for with produced actions/topics before using downstream tools.",
                    "Call isaac_perception.inspect_scene once camera topics are available.",
                ],
                artifacts={"log_path": state.log_path},
                recent_logs=(
                    recent_logs_from_results(compatibility_results) if compatibility_results else []
                ) + self._recipe_logs(recipe.name, max_lines=20),
                recipe=recipe.name,
                command=command,
                process=state.__dict__,
                produced_topics=recipe.produced_topics,
                produced_actions=recipe.produced_actions,
                produced_services=recipe.produced_services,
                render_topics=recipe.render_topics,
                warnings=warnings,
            )
        except Exception as exc:
            payload = _stateful_payload(
                ok=False,
                state="failed_to_start",
                summary=f"Failed to start launch recipe {recipe.name}: {exc}",
                missing=[recipe.name],
                next_suggested_actions=["Fix the launch error and retry with restart=true."],
                recent_logs=[str(exc)],
                recipe=recipe.name,
                command=command,
            )
        return ToolResult(payload, text=_result_text(payload))

    @tool_method
    def stop_recipe(self, recipe_name: str) -> ToolResult[dict[str, Any]]:
        """
        Stop a running Isaac ROS launch recipe.

        [[if:text]]Text output: Final process state and recent logs.[[/if:text]]

        Args:
            recipe_name: Name of the launch recipe to stop.

        Returns:
            dict: Observability payload with final process state.
        """
        recipe = _recipe(recipe_name)
        state = self._runner.stop(recipe.name)
        if recipe.cleanup_patterns and hasattr(self._runner, "stop_remote_patterns"):
            self._runner.stop_remote_patterns(recipe.cleanup_patterns)
            state = self._runner.status(recipe.name)
        log_path = state.log_path or str(self._recipe_log_path(recipe.name))
        payload = _stateful_payload(
            ok=True,
            state="stopped",
            summary=f"Stopped launch recipe {recipe.name}",
            next_suggested_actions=["Call start_recipe to launch it again if needed."],
            artifacts={"log_path": log_path},
            recent_logs=self._recipe_logs(recipe.name, max_lines=20),
            recipe=recipe.name,
            process=state.__dict__,
        )
        return ToolResult(payload, text=_result_text(payload))

    @tool_method
    def status_recipe(self, recipe_name: str) -> ToolResult[dict[str, Any]]:
        """
        Inspect a launch recipe process and expected resources.

        [[if:text]]Text output: Process status, expected topics/actions/services, render topics, and recent logs.[[/if:text]]

        Args:
            recipe_name: Name of the launch recipe.

        Returns:
            dict: Observability payload with process and recipe metadata.
        """
        recipe = _recipe(recipe_name)
        state = self._runner.status(recipe.name)
        log_path = state.log_path or str(self._recipe_log_path(recipe.name))
        payload = _stateful_payload(
            ok=state.running or state.returncode == 0,
            state="running" if state.running else "stopped",
            summary=f"Launch recipe {recipe.name} status inspected",
            next_suggested_actions=[
                "Call isaac_ros_graph.wait_for with this recipe's produced resources.",
                "Call logs_recipe if state is stopped or warnings appear.",
            ],
            artifacts={"log_path": log_path},
            recent_logs=self._recipe_logs(recipe.name, max_lines=20),
            recipe=recipe.to_dict(),
            process=state.__dict__,
        )
        return ToolResult(payload, text=_result_text(payload))

    @tool_method
    def logs_recipe(self, recipe_name: str, max_lines: int = 80) -> ToolResult[dict[str, Any]]:
        """
        Return recent logs for a launch recipe.

        [[if:text]]Text output: Tail of the recipe log, warnings, and errors.[[/if:text]]

        Args:
            recipe_name: Name of the launch recipe.
            max_lines: Maximum number of log lines to return.

        Returns:
            dict: Observability payload with recent logs.
        """
        recipe = _recipe(recipe_name)
        logs = self._recipe_logs(recipe.name, max_lines=max_lines)
        error_summary = _critical_log_lines(logs)
        payload = _stateful_payload(
            ok=True,
            state="logs_ready",
            summary=f"Loaded {len(logs)} log lines for {recipe.name}",
            next_suggested_actions=["Use error lines to decide whether to restart or inspect missing ROS resources."],
            artifacts={"log_path": str(self._recipe_log_path(recipe.name))},
            recent_logs=logs,
            error_summary=error_summary,
            recipe=recipe.name,
        )
        return ToolResult(payload, text=_logs_result_text(payload))

    @tool_method
    def inspect_recipe_outputs(
        self,
        recipe_name: str,
        sample_timeout_s: float = 3.0,
        render_images: bool = False,
    ) -> ToolResult[dict[str, Any]]:
        """
        Inspect the ROS topics, actions, and services produced by a launch recipe.

        [[if:text]]Text output: Produced resource readiness, missing outputs, optional render artifacts, and next checks.[[/if:text]]
        [[if:image]]Image output: Optional renders for image render topics when render_images=true.[[/if:image]]

        Args:
            recipe_name: Name of the launch recipe.
            sample_timeout_s: Seconds to sample each produced topic.
            render_images: Whether to render image-like render topics after inspection.

        Returns:
            dict: Observability payload for this recipe's advertised API surface.
        """
        recipe = _recipe(recipe_name)
        topic_tool = IsaacRosTopicTool(runner=self._runner, output_dir=self._output_dir)
        action_tool = IsaacRosActionTool(runner=self._runner, output_dir=self._output_dir)
        graph_tool = IsaacRosGraphTool(runner=self._runner, output_dir=self._output_dir)

        topics = sorted(set(recipe.produced_topics + recipe.render_topics))
        topic_results: dict[str, Any] = {}
        images: list[Image.Image] = []
        render_artifacts: dict[str, Any] = {}
        for topic in topics:
            topic_result = topic_tool.inspect_topic(topic, sample_timeout_s=sample_timeout_s)
            topic_results[topic] = topic_result.value
            topic_type = str(topic_result.value.get("topic_type", ""))
            if render_images and topic_result.value.get("ok") and topic in recipe.render_topics:
                renderer = topic_tool.render_depth if "depth" in topic.lower() else topic_tool.render_image
                if topic_type == "sensor_msgs/msg/Image" or "depth" in topic.lower():
                    render_result = renderer(topic, timeout_s=max(3.0, sample_timeout_s))
                    render_artifacts[topic] = render_result.value.get("artifacts", {})
                    if render_result.image:
                        images.extend(render_result.image)

        action_results = {
            action: action_tool.watch_action(action, timeout_s=10).value
            for action in recipe.produced_actions
        }
        graph = graph_tool.list_graph(include_tf=False).value
        services = []
        missing_services = []
        graph_services = set(graph.get("services", []))
        for service in recipe.produced_services:
            services.append({"service": service, "available": service in graph_services})
            if service not in graph_services:
                missing_services.append(service)

        missing = []
        for topic, result in topic_results.items():
            if not result.get("ok"):
                missing.extend(result.get("missing") or [topic])
        for action, result in action_results.items():
            if not result.get("ok"):
                missing.extend(result.get("missing") or [action])
        missing.extend(missing_services)
        payload = _stateful_payload(
            ok=not missing,
            state="outputs_ready" if not missing else "missing_outputs",
            summary=(
                f"Recipe {recipe.name} produced resources are ready"
                if not missing
                else f"Recipe {recipe.name} is missing {len(missing)} produced resource(s)"
            ),
            missing=missing,
            next_suggested_actions=[
                "Call logs_recipe for this recipe if expected outputs are missing.",
                "Call start_recipe or the recipe-specific start tool if the recipe is stopped.",
                "Use isaac_ros_topic.snapshot or render_image for deeper topic inspection.",
            ],
            artifacts={"log_path": str(self._recipe_log_path(recipe.name)), "renders": render_artifacts},
            recent_logs=self._recipe_logs(recipe.name, max_lines=30),
            recipe=recipe.to_dict(),
            topics=topic_results,
            actions=action_results,
            services=services,
        )
        return ToolResult(payload, text=_result_text(payload), image=images or None)

    def _recipe_log_path(self, recipe_name: str) -> Path:
        return self._output_dir / "logs" / f"{recipe_name}.log"

    def _recipe_logs(self, recipe_name: str, max_lines: int = 80) -> list[str]:
        logs = self._runner.logs(recipe_name, max_lines=max_lines)
        if logs:
            return logs
        log_path = self._recipe_log_path(recipe_name)
        if not log_path.exists():
            return []
        return tail_lines(log_path.read_text(errors="replace"), max_lines=max_lines)


class IsaacRosTopicTool(_IsaacTool):
    """Inspect and render ROS topics."""

    def get_name(self) -> str:
        return "isaac_ros_topic"

    @tool_method
    def inspect_topic(self, topic: str, sample_timeout_s: float = 5.0) -> ToolResult[dict[str, Any]]:
        """
        Inspect topic metadata and whether messages are actively streaming.

        [[if:text]]Text output: Topic existence, type/publisher metadata, sample-rate status, and suggested next checks.[[/if:text]]

        Args:
            topic: ROS topic name to inspect.
            sample_timeout_s: Maximum time in seconds to wait for rate samples.

        Returns:
            dict: Observability payload with metadata and activity state.
        """
        info = self._runner.run(["ros2", "topic", "info", topic], timeout_s=10)
        topic_type = self._runner.run(["ros2", "topic", "type", topic], timeout_s=10)
        hz = self._runner.run(["ros2", "topic", "hz", topic], timeout_s=sample_timeout_s)
        exists = info.ok
        streaming = hz.ok or "average rate:" in hz.stdout
        missing = [] if exists else [topic]
        if exists and not streaming:
            missing = [f"{topic}:messages"]
        payload = _stateful_payload(
            ok=exists and streaming,
            state="streaming" if streaming else ("not_streaming" if exists else "missing"),
            summary=(
                f"Topic {topic} is publishing messages"
                if streaming
                else f"Topic {topic} exists but no messages were observed"
                if exists
                else f"Topic {topic} is missing"
            ),
            missing=missing,
            next_suggested_actions=[
                "If a camera topic is not streaming, verify Isaac Sim is open and playing.",
                "Inspect the launch logs for the component that should publish this topic.",
            ],
            recent_logs=recent_logs_from_results([info, topic_type, hz]),
            topic=topic,
            info=info.stdout,
            topic_type=topic_type.stdout.strip(),
            hz=hz.stdout,
            command={"info": info.args, "type": topic_type.args, "hz": hz.args},
        )
        return ToolResult(payload, text=_result_text(payload))

    @tool_method
    def snapshot(self, topic: str, timeout_s: float = 5.0) -> ToolResult[dict[str, Any]]:
        """
        Capture one textual message from a ROS topic.

        [[if:text]]Text output: Topic snapshot text and command status.[[/if:text]]

        Args:
            topic: ROS topic name to inspect.
            timeout_s: Maximum time in seconds to wait for one message.

        Returns:
            dict: Observability payload with message text and command logs.
        """
        result = self._runner.run(["ros2", "topic", "echo", "--once", topic], timeout_s=timeout_s)
        payload = _stateful_payload(
            ok=result.ok,
            state="message_received" if result.ok else "no_message",
            summary=f"Snapshot for topic {topic}" if result.ok else f"No snapshot for topic {topic}",
            missing=[] if result.ok else [topic],
            next_suggested_actions=[
                "If this topic is missing, inspect the launch recipe that should publish it.",
                "If this is an image topic, call render_image for visual feedback.",
            ],
            recent_logs=recent_logs_from_results([result]),
            topic=topic,
            message=result.stdout,
            command=result.args,
        )
        return ToolResult(payload, text=_result_text(payload))

    @tool_method(requires_image_output=True)
    def render_image(self, topic: str, timeout_s: float = 5.0) -> ToolResult[dict[str, Any]]:
        """
        Save and return a render from a ROS image topic.

        [[if:text]]Text output: Render artifact path and command status.[[/if:text]]
        [[if:image]]Image output: The saved image if image_view can capture the topic.[[/if:image]]

        Args:
            topic: ROS image topic name.
            timeout_s: Maximum capture time in seconds.

        Returns:
            dict: Observability payload with artifact path and render state.
        """
        return self._render_topic(topic, "image", timeout_s)

    @tool_method(requires_image_output=True)
    def render_depth(self, topic: str, timeout_s: float = 5.0) -> ToolResult[dict[str, Any]]:
        """
        Save and return a render from a ROS depth image topic.

        [[if:text]]Text output: Depth render artifact path and command status.[[/if:text]]
        [[if:image]]Image output: The saved depth image if image_view can capture the topic.[[/if:image]]

        Args:
            topic: ROS depth image topic name.
            timeout_s: Maximum capture time in seconds.

        Returns:
            dict: Observability payload with artifact path and render state.
        """
        return self._render_depth_topic(topic, timeout_s)

    def _render_depth_topic(self, topic: str, timeout_s: float) -> ToolResult[dict[str, Any]]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        safe_topic = topic.strip("/").replace("/", "_") or "root"
        output_path = self._output_dir / f"depth_{safe_topic}.png"
        capture_path: str | Path = output_path
        if getattr(self._runner, "container_name", None):
            capture_path = f"/tmp/{output_path.name}"
        command = ["python3", "-c", _DEPTH_RENDER_SCRIPT, topic, str(capture_path), str(float(timeout_s))]
        result = self._runner.run(command, timeout_s=timeout_s + 10)
        fetch_result = None
        if capture_path != output_path and not output_path.exists() and hasattr(self._runner, "fetch_file"):
            fetch_result = self._runner.fetch_file(capture_path, output_path)
        image = None
        if output_path.exists() and not self.no_output_image:
            image = Image.open(output_path).convert("RGB")
        ok = result.ok and output_path.exists()
        log_results = [result]
        if fetch_result is not None:
            log_results.append(fetch_result)
        payload = _stateful_payload(
            ok=ok,
            state="render_ready" if ok else "render_unavailable",
            summary=f"Rendered depth topic {topic}" if ok else f"Could not render depth topic {topic}",
            missing=[] if ok else [topic],
            next_suggested_actions=[
                "If render is unavailable, call snapshot to confirm topic data.",
                "Inspect launch logs for camera bridge or depth image conversion errors.",
            ],
            artifacts={"image_path": str(output_path), "capture_path": str(capture_path)},
            recent_logs=recent_logs_from_results(log_results),
            topic=topic,
            command=result.args,
        )
        return ToolResult(payload, text=_result_text(payload), image=[image] if image else None)

    def _render_topic(self, topic: str, prefix: str, timeout_s: float) -> ToolResult[dict[str, Any]]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        safe_topic = topic.strip("/").replace("/", "_") or "root"
        output_path = self._output_dir / f"{prefix}_{safe_topic}.png"
        capture_path: str | Path = output_path
        if getattr(self._runner, "container_name", None):
            capture_path = f"/tmp/{output_path.name}"
        command = [
            "timeout",
            str(float(timeout_s)),
            "ros2",
            "run",
            "image_view",
            "image_saver",
            "--ros-args",
            "-r",
            f"image:={topic}",
            "-p",
            f"filename_format:={capture_path}",
        ]
        result = self._runner.run(command, timeout_s=timeout_s + 5)
        fetch_result = None
        if capture_path != output_path and not output_path.exists() and hasattr(self._runner, "fetch_file"):
            fetch_result = self._runner.fetch_file(capture_path, output_path)
        image = None
        if output_path.exists() and not self.no_output_image:
            image = Image.open(output_path).convert("RGB")
        ok = result.ok and output_path.exists()
        log_results = [result]
        if fetch_result is not None:
            log_results.append(fetch_result)
        payload = _stateful_payload(
            ok=ok,
            state="render_ready" if ok else "render_unavailable",
            summary=f"Rendered topic {topic}" if ok else f"Could not render topic {topic}",
            missing=[] if ok else [topic],
            next_suggested_actions=[
                "If render is unavailable, call snapshot to confirm topic data.",
                "Inspect launch logs for camera bridge or image_view errors.",
            ],
            artifacts={"image_path": str(output_path), "capture_path": str(capture_path)},
            recent_logs=recent_logs_from_results(log_results),
            topic=topic,
            command=result.args,
        )
        return ToolResult(payload, text=_result_text(payload), image=[image] if image else None)


class IsaacRosActionTool(_IsaacTool):
    """Send and inspect ROS actions."""

    def get_name(self) -> str:
        return "isaac_ros_action"

    @tool_method
    def send_goal(
        self,
        action_name: str,
        action_type: str,
        goal_yaml: str,
        timeout_s: float = 120.0,
    ) -> ToolResult[dict[str, Any]]:
        """
        Send a ROS action goal and stream feedback until completion or timeout.

        [[if:text]]Text output: Action command state, feedback/result text, and next suggested inspection calls.[[/if:text]]

        Args:
            action_name: ROS action name, for example /multi_object_pick_and_place.
            action_type: ROS action type.
            goal_yaml: Goal payload as YAML or JSON text.
            timeout_s: Maximum time in seconds to wait for the action command.

        Returns:
            dict: Observability payload with action state, logs, and goal text.
        """
        preflight = self._runner.run(["ros2", "action", "info", action_name], timeout_s=min(10.0, timeout_s))
        server_count = _action_count(preflight.stdout, "Action servers")
        client_count = _action_count(preflight.stdout, "Action clients")
        if not preflight.ok or server_count < 1:
            payload = _stateful_payload(
                ok=False,
                state="no_action_server" if preflight.ok else "action_info_failed",
                summary=f"Action {action_name} has no available server",
                missing=[action_name],
                next_suggested_actions=[
                    "Start the launch recipe that provides this action.",
                    "Call isaac_ros_launch.status_recipe and logs_recipe for the recipe.",
                    "Call isaac_ros_graph.wait_for before sending the goal again.",
                ],
                recent_logs=recent_logs_from_results([preflight]),
                action_name=action_name,
                action_type=action_type,
                goal=goal_yaml,
                command=preflight.args,
                stdout=preflight.stdout,
                stderr=preflight.stderr,
                action_servers=server_count,
                action_clients=client_count,
                terminal_status="",
                workflow_status="",
            )
            return ToolResult(payload, text=_result_text(payload))

        command = ["ros2", "action", "send_goal", "--feedback", action_name, action_type, goal_yaml]
        command[5] = _resolved_action_type(self._runner, action_type)
        result = self._runner.run(command, timeout_s=timeout_s)
        accepted = "Goal accepted" in result.stdout
        terminal_match = re.search(r"Goal finished with status:\s*([A-Z_]+)", result.stdout)
        terminal_status = terminal_match.group(1) if terminal_match else ""
        workflow_match = re.search(r"workflow_status:\s*([0-9]+)", result.stdout)
        workflow_status = workflow_match.group(1) if workflow_match else ""
        timed_out = result.returncode == 124 or "Timed out after" in result.stderr
        terminal_failure = terminal_status and terminal_status != "SUCCEEDED"
        command_logs = recent_logs_from_results([result], max_lines=80)
        action_error_summary = _critical_log_lines(command_logs, max_lines=24)
        if (terminal_failure or not result.ok) and not action_error_summary:
            action_error_summary = [line for line in command_logs if line.strip()][-12:]
        if terminal_status == "SUCCEEDED":
            state = "goal_succeeded"
            summary = f"Action goal for {action_name} finished with status SUCCEEDED"
        elif terminal_failure:
            state = f"goal_{terminal_status.lower()}"
            summary = f"Action goal for {action_name} finished with status {terminal_status}"
        elif result.ok:
            state = "goal_sent"
            summary = f"Action goal sent to {action_name}"
        elif timed_out and accepted:
            state = "goal_timeout"
            summary = f"Action goal for {action_name} timed out after being accepted"
        elif timed_out:
            state = "action_timeout"
            summary = f"Timed out waiting on action {action_name}"
        else:
            state = "goal_failed"
            summary = f"Action goal failed for {action_name}"
        payload = _stateful_payload(
            ok=result.ok and not terminal_failure,
            state=state,
            summary=summary,
            missing=[] if result.ok or accepted else [action_name],
            next_suggested_actions=[
                "Call isaac_perception.inspect_scene to verify the scene changed.",
                "Call watch_action to inspect the action server if the goal failed.",
                "Call isaac_ros_launch.logs_recipe for orchestration logs.",
            ],
            recent_logs=command_logs,
            action_error_summary=action_error_summary,
            action_name=action_name,
            action_type=action_type,
            goal=goal_yaml,
            command=result.args,
            stdout=result.stdout,
            stderr=result.stderr,
            terminal_status=terminal_status,
            workflow_status=workflow_status,
            action_servers=server_count,
            action_clients=client_count,
        )
        return ToolResult(payload, text=_result_text(payload))

    @tool_method
    def watch_action(self, action_name: str, timeout_s: float = 10.0) -> ToolResult[dict[str, Any]]:
        """
        Inspect a ROS action server.

        [[if:text]]Text output: Action info output, command status, and suggested next steps.[[/if:text]]

        Args:
            action_name: ROS action name.
            timeout_s: Maximum command time in seconds.

        Returns:
            dict: Observability payload with action server info.
        """
        result = self._runner.run(["ros2", "action", "info", action_name], timeout_s=timeout_s)
        server_count = _action_count(result.stdout, "Action servers")
        client_count = _action_count(result.stdout, "Action clients")
        available = result.ok and server_count > 0
        payload = _stateful_payload(
            ok=available,
            state="available" if available else ("no_action_server" if result.ok else "missing"),
            summary=(
                f"Action {action_name} has {server_count} server(s)"
                if available
                else f"Action {action_name} has no available server"
            ),
            missing=[] if available else [action_name],
            next_suggested_actions=["Start or inspect the launch recipe that provides this action."],
            recent_logs=recent_logs_from_results([result]),
            action_name=action_name,
            info=result.stdout,
            action_servers=server_count,
            action_clients=client_count,
        )
        return ToolResult(payload, text=_result_text(payload))


class IsaacRosServiceTool(_IsaacTool):
    """Inspect and call ROS services."""

    def get_name(self) -> str:
        return "isaac_ros_service"

    @tool_method
    def inspect_service(self, service_name: str, timeout_s: float = 10.0) -> ToolResult[dict[str, Any]]:
        """
        Inspect a ROS service.

        [[if:text]]Text output: Service availability, type, command logs, and suggested next steps.[[/if:text]]

        Args:
            service_name: ROS service name.
            timeout_s: Maximum command time in seconds.

        Returns:
            dict: Observability payload with service type and availability.
        """
        service_type = self._runner.run(["ros2", "service", "type", service_name], timeout_s=timeout_s)
        service_list = self._runner.run(["ros2", "service", "list"], timeout_s=timeout_s)
        services = _items_from_output(service_list)
        available = service_type.ok and service_name in services
        payload = _stateful_payload(
            ok=available,
            state="available" if available else "missing",
            summary=(
                f"Service {service_name} is available"
                if available
                else f"Service {service_name} is not available"
            ),
            missing=[] if available else [service_name],
            next_suggested_actions=[
                "Start or inspect the launch recipe that provides this service.",
                "Call isaac_ros_graph.list_graph to inspect available services.",
            ],
            recent_logs=recent_logs_from_results([service_type, service_list]),
            service_name=service_name,
            service_type=service_type.stdout.strip(),
            services=services,
            command={"type": service_type.args, "list": service_list.args},
        )
        return ToolResult(payload, text=_result_text(payload))

    @tool_method
    def call_service(
        self,
        service_name: str,
        service_type: str,
        request_yaml: str = "{}",
        timeout_s: float = 30.0,
    ) -> ToolResult[dict[str, Any]]:
        """
        Call a ROS service.

        [[if:text]]Text output: Service call result, stdout/stderr, command, and next suggested checks.[[/if:text]]

        Args:
            service_name: ROS service name.
            service_type: ROS service type.
            request_yaml: Request payload as YAML or JSON text.
            timeout_s: Maximum command time in seconds.

        Returns:
            dict: Observability payload with service call output.
        """
        resolved_type = _resolved_interface_type(self._runner, service_type)
        preflight = self.inspect_service(service_name, timeout_s=min(10.0, timeout_s))
        if not preflight.value.get("ok", False):
            payload = dict(preflight.value)
            payload.update(
                {
                    "service_type": resolved_type,
                    "request": request_yaml,
                    "summary": f"Service {service_name} is unavailable; not calling it",
                }
            )
            return ToolResult(payload, text=_result_text(payload))

        command = ["ros2", "service", "call", service_name, resolved_type, request_yaml]
        result = self._runner.run(command, timeout_s=timeout_s)
        payload = _stateful_payload(
            ok=result.ok,
            state="service_called" if result.ok else "service_call_failed",
            summary=(
                f"Service {service_name} call completed"
                if result.ok
                else f"Service {service_name} call failed"
            ),
            missing=[] if result.ok else [service_name],
            next_suggested_actions=[
                "Inspect dependent topics/actions to verify the service changed state.",
                "Call isaac_ros_launch.logs_recipe for the recipe that owns this service if it failed.",
            ],
            recent_logs=recent_logs_from_results([result]),
            service_name=service_name,
            service_type=resolved_type,
            request=request_yaml,
            command=result.args,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        return ToolResult(payload, text=_result_text(payload))


class IsaacRosConfigTool(_IsaacTool):
    """Prepare validated Isaac ROS manipulation config overlays."""

    def get_name(self) -> str:
        return "isaac_ros_config"

    @tool_method
    def prepare_pick_and_place_config(
        self,
        output_path: str = "",
        use_ground_truth_pose_in_sim: bool = False,
        enable_nvblox: bool = False,
        enable_rviz_visualization: bool = False,
    ) -> ToolResult[dict[str, Any]]:
        """
        Write a simulation pick-and-place workflow config overlay.

        [[if:text]]Text output: Config path, core values, and required restart note.[[/if:text]]

        Args:
            output_path: Destination YAML path. Defaults to outputs/isaac_ros/pick_place_config.yaml.
            use_ground_truth_pose_in_sim: Keep false for the multi-object pick-and-place
                tutorial. True disables object detection/object-info servers, so the behavior
                tree cannot satisfy its /get_objects dependency.
            enable_nvblox: Whether to enable nvblox ESDF reconstruction.
            enable_rviz_visualization: Whether to start RViz from the workflow launch.

        Returns:
            dict: Observability payload with config artifact path and selected values.
        """
        default_remote_path = _container_default_config_path(self._runner)
        path = output_path or default_remote_path or str(self._output_dir / "pick_place_config.yaml")
        host_artifact_path = self._output_dir / "pick_place_config.yaml"
        config = _read_default_workflow_config(self._runner)
        params_dir = _pick_place_params_dir(self._runner)
        bt_path, host_bt_path = _artifact_paths(
            self._runner,
            self._output_dir,
            "multi_object_pick_and_place_behavior_tree_params.yaml",
        )
        blackboard_path, host_blackboard_path = _artifact_paths(
            self._runner,
            self._output_dir,
            "multi_object_pick_and_place_blackboard_params.yaml",
        )
        ros2_control_path, host_ros2_control_path = _artifact_paths(
            self._runner,
            self._output_dir,
            "ros2_control_controllers_sim_toolshed.yaml",
        )
        behavior_tree_config = _read_yaml_file(
            self._runner,
            Path(params_dir) / "multi_object_pick_and_place_behavior_tree_params.yaml",
        )
        blackboard_config = _read_yaml_file(
            self._runner,
            Path(params_dir) / "multi_object_pick_and_place_blackboard_params.yaml",
        )
        robot_config_dir = _robot_description_config_dir(self._runner)
        ros2_control_config = _read_yaml_file(
            self._runner,
            Path(robot_config_dir) / "ros2_control_controllers_sim.yaml",
        )
        controller_params = (
            ros2_control_config
            .setdefault("controller_manager", {})
            .setdefault("ros__parameters", {})
        )
        controller_params.pop("joint_state_broadcaster", None)
        ros2_control_config.pop("joint_state_broadcaster", None)
        pick_place_bt = (
            behavior_tree_config
            .setdefault("behavior_tree_params", {})
            .setdefault("multi_object_pick_and_place", {})
        )
        pick_place_bt.setdefault("pose_estimation", {})["camera_frame_id"] = "front_stereo_camera_left"
        switch_controllers = pick_place_bt.setdefault("switch_controllers", {})
        for controller_group in ("arm", "tool"):
            controller_config = switch_controllers.setdefault(controller_group, {})
            controller_config["controllers_to_activate"] = []
            controller_config["controllers_to_deactivate"] = []
        blackboard_params = blackboard_config.setdefault("blackboard_params", {})
        supported_objects = blackboard_params.setdefault("supported_objects", {})
        supported_objects["mac_and_cheese"] = {
            "class_id": "22",
            "grasp_file_path": (
                "$(ros2 pkg prefix --share isaac_manipulator_robot_description)"
                "/config/robotiq_2f_140_grasps_mac_and_cheese.yaml"
            ),
            "mesh_file_path": (
                "${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_foundationpose/"
                "Mac_and_cheese_0_1/Mac_and_cheese_0_1.obj"
            ),
        }
        supported_objects["soup_can"] = {
            "class_id": "3",
            "grasp_file_path": (
                "$(ros2 pkg prefix --share isaac_manipulator_robot_description)"
                "/config/robotiq_2f_140_grasps_soup_can.yaml"
            ),
            "mesh_file_path": "${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_foundationpose/soup_can/soup_can.obj",
        }
        bt_text = yaml.safe_dump(behavior_tree_config, sort_keys=False)
        if hasattr(self._runner, "write_file"):
            write_bt_result = self._runner.write_file(bt_path, bt_text)
            if not write_bt_result.ok:
                raise RuntimeError(write_bt_result.stderr or f"Failed to write behavior tree config to {bt_path}")
            if str(bt_path) != str(host_bt_path):
                host_bt_path.parent.mkdir(parents=True, exist_ok=True)
                host_bt_path.write_text(bt_text)
        else:
            host_bt_path.parent.mkdir(parents=True, exist_ok=True)
            host_bt_path.write_text(bt_text)
        blackboard_text = yaml.safe_dump(blackboard_config, sort_keys=False)
        if hasattr(self._runner, "write_file"):
            write_blackboard_result = self._runner.write_file(blackboard_path, blackboard_text)
            if not write_blackboard_result.ok:
                raise RuntimeError(
                    write_blackboard_result.stderr
                    or f"Failed to write blackboard config to {blackboard_path}"
                )
            if str(blackboard_path) != str(host_blackboard_path):
                host_blackboard_path.parent.mkdir(parents=True, exist_ok=True)
                host_blackboard_path.write_text(blackboard_text)
        else:
            host_blackboard_path.parent.mkdir(parents=True, exist_ok=True)
            host_blackboard_path.write_text(blackboard_text)
        ros2_control_text = yaml.safe_dump(ros2_control_config, sort_keys=False)
        if hasattr(self._runner, "write_file"):
            write_ros2_control_result = self._runner.write_file(ros2_control_path, ros2_control_text)
            if not write_ros2_control_result.ok:
                raise RuntimeError(
                    write_ros2_control_result.stderr
                    or f"Failed to write ros2_control config to {ros2_control_path}"
                )
            if str(ros2_control_path) != str(host_ros2_control_path):
                host_ros2_control_path.parent.mkdir(parents=True, exist_ok=True)
                host_ros2_control_path.write_text(ros2_control_text)
        else:
            host_ros2_control_path.parent.mkdir(parents=True, exist_ok=True)
            host_ros2_control_path.write_text(ros2_control_text)
        requested_ground_truth_pose = bool(use_ground_truth_pose_in_sim)
        requested_rviz_visualization = bool(enable_rviz_visualization)
        requested_nvblox = bool(enable_nvblox)
        warnings = []
        if requested_ground_truth_pose:
            warnings.append(
                "use_ground_truth_pose_in_sim=true is incompatible with the multi-object "
                "pick-and-place behavior tree because it prevents /get_objects from launching; "
                "writing false instead."
            )
        if requested_rviz_visualization:
            warnings.append(
                "enable_rviz_visualization=true is incompatible with the headless Isaac Sim "
                "runner because RViz requires an interactive display; writing false instead."
            )
        if requested_nvblox:
            warnings.append(
                "enable_nvblox=true is not stable in this headless tutorial run because the "
                "cuMotion planner can abort while waiting for ESDF data; writing false instead."
            )
        use_ground_truth_pose_in_sim = False
        enable_rviz_visualization = False
        enable_nvblox = False
        config.update(
            {
                "camera_type": "ISAAC_SIM",
                "num_cameras": 1,
                "workflow_type": "PICK_AND_PLACE",
                "ur_type": "ur10e",
                "gripper_type": "robotiq_2f_140",
                "use_sim_time": "true",
                "setup": "sim_test_bench",
                "object_detection_type": "RTDETR",
                "pose_estimation_type": "FOUNDATION_POSE",
                "segmentation_type": "NONE",
                "use_ground_truth_pose_in_sim": str(use_ground_truth_pose_in_sim).lower(),
                "enable_nvblox": str(enable_nvblox).lower(),
                "enable_rviz_visualization": str(enable_rviz_visualization).lower(),
                "controller_spawner_timeout": 60,
                "ros2_controllers_file_path": str(ros2_control_path),
                "sim_gt_asset_frame_id": "soup_can",
                "object_class_id": "3",
                "rt_detr_confidence_threshold": "0.5",
                "behavior_tree_config_file": str(bt_path),
                "blackboard_config_file": str(blackboard_path),
            }
        )
        config_text = yaml.safe_dump(config, sort_keys=False)
        if hasattr(self._runner, "write_file"):
            write_result = self._runner.write_file(path, config_text)
            if not write_result.ok:
                raise RuntimeError(write_result.stderr or f"Failed to write config to {path}")
            if str(path) != str(host_artifact_path):
                host_artifact_path.parent.mkdir(parents=True, exist_ok=True)
                host_artifact_path.write_text(config_text)
        else:
            local_path = Path(path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(config_text)
        payload = _stateful_payload(
            ok=True,
            state="config_ready",
            summary=f"Wrote pick-and-place config overlay to {path}",
            next_suggested_actions=[
                f"Call isaac_ros_launch.start_recipe for pick_and_place with manipulator_workflow_config={path}.",
                "Restart the pick-and-place workflow after changing this file.",
            ],
            artifacts={
                "config_path": str(path),
                "host_config_path": str(host_artifact_path),
                "behavior_tree_config_path": str(bt_path),
                "host_behavior_tree_config_path": str(host_bt_path),
                "blackboard_config_path": str(blackboard_path),
                "host_blackboard_config_path": str(host_blackboard_path),
                "ros2_control_config_path": str(ros2_control_path),
                "host_ros2_control_config_path": str(host_ros2_control_path),
            },
            config=config,
            behavior_tree_config=behavior_tree_config,
            blackboard_config=blackboard_config,
            ros2_control_config=ros2_control_config,
            warnings=warnings,
            requested_use_ground_truth_pose_in_sim=requested_ground_truth_pose,
            requested_enable_rviz_visualization=requested_rviz_visualization,
            requested_enable_nvblox=requested_nvblox,
        )
        return ToolResult(payload, text=_result_text(payload))


class IsaacRosSceneTool(_IsaacTool):
    """Inspect Isaac Sim bridge readiness and scene observations."""

    def get_name(self) -> str:
        return "isaac_ros_scene"

    @tool_method
    def check_ready(self, sample_timeout_s: float = 3.0) -> ToolResult[dict[str, Any]]:
        """
        Check whether Isaac Sim ROS bridge topics needed by manipulation are visible.

        [[if:text]]Text output: Missing scene resources, topic activity, and suggested next calls.[[/if:text]]

        Args:
            sample_timeout_s: Maximum time in seconds to wait for message-rate samples per required topic.

        Returns:
            dict: Observability payload with graph readiness for Isaac Sim camera and joint topics.
        """
        graph = IsaacRosGraphTool(runner=self._runner, output_dir=self._output_dir).list_graph(include_tf=False).value
        required_topics = ["/front_stereo_camera/left/image_raw", "/front_stereo_camera/depth/ground_truth"]
        missing = [topic for topic in required_topics if topic not in graph.get("topics", [])]
        topic_tool = IsaacRosTopicTool(runner=self._runner, output_dir=self._output_dir)
        topic_activity = {
            topic: topic_tool.inspect_topic(topic, sample_timeout_s=sample_timeout_s).value
            for topic in required_topics
            if topic not in missing
        }
        inactive = [
            topic
            for topic, state in topic_activity.items()
            if state.get("state") != "streaming"
        ]
        all_missing = missing + [f"{topic}:messages" for topic in inactive]
        ok = not all_missing
        payload = _stateful_payload(
            ok=ok,
            state=(
                "ready"
                if ok
                else "missing_scene_topics"
                if missing
                else "scene_topics_not_streaming"
            ),
            summary=(
                "Isaac Sim ROS bridge is ready"
                if ok
                else "Isaac Sim ROS bridge has topics but no camera/depth messages"
                if inactive and not missing
                else "Isaac Sim ROS bridge is missing topics"
            ),
            missing=all_missing,
            next_suggested_actions=[
                "Start the Isaac Sim demo wrapper if scene topics are missing.",
                "If topics exist but do not stream, open Isaac Sim, load the tutorial USD, and press Play.",
                "Call isaac_ros_topic.render_image on /front_stereo_camera/left/image_raw.",
            ],
            topics=graph.get("topics", []),
            actions=graph.get("actions", []),
            topic_activity=topic_activity,
        )
        return ToolResult(payload, text=_result_text(payload))


class IsaacPerceptionTool(_IsaacTool):
    """Collect scene perception feedback for agent decisions."""

    def get_name(self) -> str:
        return "isaac_perception"

    @tool_method
    def inspect_scene(
        self,
        rgb_topic: str = "/front_stereo_camera/left/image_raw",
        depth_topic: str = "/front_stereo_camera/depth/ground_truth",
    ) -> ToolResult[dict[str, Any]]:
        """
        Inspect the manipulation scene using graph state and render topics.

        [[if:text]]Text output: Scene resource state, render artifacts, and suggested next calls.[[/if:text]]
        [[if:image]]Image output: RGB/depth render images when image capture succeeds.[[/if:image]]

        Args:
            rgb_topic: RGB camera topic to render.
            depth_topic: Depth topic to render.

        Returns:
            dict: Observability payload with graph, render artifacts, detections topics, actions, and logs.
        """
        graph = IsaacRosGraphTool(runner=self._runner, output_dir=self._output_dir).list_graph(include_tf=False).value
        topic_tool = IsaacRosTopicTool(runner=self._runner, output_dir=self._output_dir)
        rgb_state = topic_tool.inspect_topic(rgb_topic, sample_timeout_s=3)
        depth_state = topic_tool.inspect_topic(depth_topic, sample_timeout_s=3)
        rgb = topic_tool.render_image(rgb_topic)
        depth = topic_tool.render_depth(depth_topic)
        images = []
        images.extend(rgb.image)
        images.extend(depth.image)
        payload = _stateful_payload(
            ok=graph.get("ok", False),
            state="scene_inspected",
            summary="Scene inspection complete",
            missing=graph.get("missing", []),
            next_suggested_actions=[
                "If detections and pose topics are present, send a pick-and-place goal.",
                "If camera renders are missing, inspect Isaac Sim bridge readiness.",
            ],
            artifacts={
                "rgb": rgb.value.get("artifacts", {}),
                "depth": depth.value.get("artifacts", {}),
            },
            recent_logs=(
                rgb_state.value.get("recent_logs", [])
                + depth_state.value.get("recent_logs", [])
                + rgb.value.get("recent_logs", [])
                + depth.value.get("recent_logs", [])
            )[-20:],
            topics=graph.get("topics", []),
            actions=graph.get("actions", []),
            topic_activity={rgb_topic: rgb_state.value, depth_topic: depth_state.value},
            detection_topics=[topic for topic in graph.get("topics", []) if "detection" in topic],
            pose_topics=[topic for topic in graph.get("topics", []) if "pose" in topic],
        )
        return ToolResult(payload, text=_result_text(payload), image=images or None)


class IsaacPickPlaceTool(_IsaacTool):
    """High-level pick-and-place helpers built from ROS action and launch primitives."""

    def get_name(self) -> str:
        return "isaac_pick_place"

    @tool_method
    def start_workflow(
        self,
        config_path: str = "",
        restart: bool = False,
    ) -> ToolResult[dict[str, Any]]:
        """
        Start the top-level Isaac ROS pick-and-place workflow.

        [[if:text]]Text output: Launch state, expected action server, render topics, logs, and suggested readiness checks.[[/if:text]]

        Args:
            config_path: Workflow YAML path or file name.
            restart: Stop the workflow first if already running.

        Returns:
            dict: Observability payload with launch process and produced resources.
        """
        if not config_path:
            config_result = IsaacRosConfigTool(
                runner=self._runner, output_dir=self._output_dir
            ).prepare_pick_and_place_config()
            config_path = config_result.value["artifacts"]["config_path"]
        overrides = json.dumps({"manipulator_workflow_config": config_path})
        return IsaacRosLaunchTool(runner=self._runner, output_dir=self._output_dir).start_recipe(
            "pick_and_place_workflow", overrides=overrides, restart=restart
        )

    @tool_method
    def send_single_bin_goal(
        self,
        drop_pose: str = "",
        frame_id: str = "base_link",
        timeout_s: float = 180.0,
    ) -> ToolResult[dict[str, Any]]:
        """
        Send the tutorial single-bin pick-and-place action goal.

        [[if:text]]Text output: Action feedback/result text, goal payload, and suggested scene-inspection call.[[/if:text]]

        Args:
            drop_pose: Optional JSON seven-value pose [x, y, z, qx, qy, qz, qw].
            frame_id: Frame for the target pose.
            timeout_s: Maximum action wait time in seconds.

        Returns:
            dict: Observability payload with action result and goal.
        """
        goal = build_single_bin_goal(drop_pose or None, frame_id=frame_id)
        return self._send_pick_place_goal(goal, timeout_s)

    @tool_method
    def send_multi_bin_goal(
        self,
        target_poses: str = "",
        class_ids: str = "",
        frame_id: str = "base_link",
        timeout_s: float = 180.0,
    ) -> ToolResult[dict[str, Any]]:
        """
        Send the tutorial multi-bin pick-and-place action goal.

        [[if:text]]Text output: Action feedback/result text, goal payload, and suggested scene-inspection call.[[/if:text]]

        Args:
            target_poses: Optional JSON array of seven-value target poses.
            class_ids: Optional JSON array or comma-separated class IDs.
            frame_id: Frame for target poses.
            timeout_s: Maximum action wait time in seconds.

        Returns:
            dict: Observability payload with action result and goal.
        """
        try:
            goal = build_multi_bin_goal(target_poses or None, class_ids or None, frame_id=frame_id)
        except ValueError as exc:
            payload = _stateful_payload(
                ok=False,
                state="invalid_goal",
                summary=f"Invalid multi-bin pick-and-place goal: {exc}",
                missing=[],
                next_suggested_actions=[
                    "Call send_multi_bin_goal with equal-length target_poses and class_ids.",
                    "Omit both target_poses and class_ids to use the tutorial defaults.",
                    "Call send_single_bin_goal if one drop pose is sufficient.",
                ],
                recent_logs=[str(exc)],
                target_poses=target_poses,
                class_ids=class_ids,
                frame_id=frame_id,
            )
            return ToolResult(payload, text=_result_text(payload))
        return self._send_pick_place_goal(goal, timeout_s)

    @tool_method
    def watch_goal(self) -> ToolResult[dict[str, Any]]:
        """
        Inspect the pick-and-place action server.

        [[if:text]]Text output: Action server status and suggested next calls.[[/if:text]]

        Returns:
            dict: Observability payload for /multi_object_pick_and_place.
        """
        return IsaacRosActionTool(runner=self._runner, output_dir=self._output_dir).watch_action(
            "/multi_object_pick_and_place"
        )

    def _send_pick_place_goal(self, goal: dict[str, Any], timeout_s: float) -> ToolResult[dict[str, Any]]:
        result = IsaacRosActionTool(runner=self._runner, output_dir=self._output_dir).send_goal(
            "/multi_object_pick_and_place",
            _resolved_action_type(self._runner, DEFAULT_ACTION_TYPE),
            json.dumps(goal),
            timeout_s=timeout_s,
        )
        result.value["goal"] = goal
        result.value["next_suggested_actions"] = [
            "Call isaac_perception.inspect_scene to verify object movement.",
            "Call isaac_ros_launch.logs_recipe on pick_and_place_workflow if execution failed.",
        ]
        if not result.value.get("ok", False):
            result.value["next_suggested_actions"] = [
                "Call isaac_ros_launch.logs_recipe on pick_and_place_workflow and inspect the concrete error lines.",
                "Call isaac_pick_place.watch_goal to confirm the action server is still available.",
                "If a multi-bin goal failed or timed out, consider send_single_bin_goal with the default tutorial drop pose.",
            ]
            log_result = IsaacRosLaunchTool(
                runner=self._runner, output_dir=self._output_dir
            ).logs_recipe("pick_and_place_workflow", max_lines=400)
            result.value["workflow_error_summary"] = log_result.value.get("error_summary", [])
            result.value["recent_logs"] = (
                result.value.get("recent_logs", []) + result.value["workflow_error_summary"]
            )[-40:]
            result.value.setdefault("artifacts", {})["workflow_log_path"] = log_result.value.get(
                "artifacts", {}
            ).get("log_path", str(self._output_dir / "logs" / "pick_and_place_workflow.log"))
        result.text = _result_text(result.value)
        return result


class IsaacObjectInfoTool(_IsaacTool):
    """Call object detection, object-info, and object-cache APIs used by the workflow."""

    def get_name(self) -> str:
        return "isaac_object_info"

    @tool_method
    def detect_objects(self, timeout_s: float = 60.0) -> ToolResult[dict[str, Any]]:
        """
        Trigger the object detection action.

        [[if:text]]Text output: Action result, detected-object payload text, and next suggested checks.[[/if:text]]

        Args:
            timeout_s: Maximum action wait time in seconds.

        Returns:
            dict: Observability payload with detection action output.
        """
        return IsaacRosActionTool(runner=self._runner, output_dir=self._output_dir).send_goal(
            "/detect_objects",
            _resolved_action_type(
                self._runner,
                "isaac_ros_manipulation_interfaces/action/DetectObjects",
            ),
            "{}",
            timeout_s=timeout_s,
        )

    @tool_method
    def get_objects(
        self,
        use_initial_hint: bool = False,
        initial_hint: str = "",
        timeout_s: float = 60.0,
    ) -> ToolResult[dict[str, Any]]:
        """
        Query the workflow object-info action.

        [[if:text]]Text output: Object-info action result, object IDs/classes/poses in stdout, and next suggested checks.[[/if:text]]

        Args:
            use_initial_hint: Whether to provide an initial 3D hint to the object selector.
            initial_hint: Optional JSON point {"x": ..., "y": ..., "z": ...}.
            timeout_s: Maximum action wait time in seconds.

        Returns:
            dict: Observability payload with /get_objects action output.
        """
        hint = {"x": 0.0, "y": 0.0, "z": 0.0}
        if initial_hint:
            parsed_hint = json.loads(initial_hint)
            hint = {axis: float(parsed_hint.get(axis, 0.0)) for axis in ("x", "y", "z")}
        goal = {"use_initial_hint": bool(use_initial_hint), "initial_hint": hint}
        return IsaacRosActionTool(runner=self._runner, output_dir=self._output_dir).send_goal(
            "/get_objects",
            _resolved_action_type(
                self._runner,
                "isaac_ros_manipulation_interfaces/action/GetObjects",
            ),
            json.dumps(goal),
            timeout_s=timeout_s,
        )

    @tool_method
    def get_object_pose(
        self,
        object_id: int,
        class_id: str = "",
        timeout_s: float = 30.0,
    ) -> ToolResult[dict[str, Any]]:
        """
        Query the pose for one object ID returned by get_objects.

        [[if:text]]Text output: Object pose action result and stdout/stderr.[[/if:text]]

        Args:
            object_id: Object ID from /get_objects.
            class_id: Optional object class ID.
            timeout_s: Maximum action wait time in seconds.

        Returns:
            dict: Observability payload with /get_object_pose action output.
        """
        goal = {"object_id": int(object_id), "class_id": str(class_id)}
        return IsaacRosActionTool(runner=self._runner, output_dir=self._output_dir).send_goal(
            "/get_object_pose",
            _resolved_action_type(
                self._runner,
                "isaac_ros_manipulation_interfaces/action/GetObjectPose",
            ),
            json.dumps(goal),
            timeout_s=timeout_s,
        )

    @tool_method
    def add_segmentation_mask(self, object_id: int, timeout_s: float = 30.0) -> ToolResult[dict[str, Any]]:
        """
        Add or refresh a segmentation mask for one known object.

        [[if:text]]Text output: Segmentation-mask action result and stdout/stderr.[[/if:text]]

        Args:
            object_id: Object ID from /get_objects.
            timeout_s: Maximum action wait time in seconds.

        Returns:
            dict: Observability payload with /add_segmentation_mask action output.
        """
        goal = {"object_id": int(object_id)}
        return IsaacRosActionTool(runner=self._runner, output_dir=self._output_dir).send_goal(
            "/add_segmentation_mask",
            _resolved_action_type(
                self._runner,
                "isaac_ros_manipulation_interfaces/action/AddSegmentationMask",
            ),
            json.dumps(goal),
            timeout_s=timeout_s,
        )

    @tool_method
    def clear_objects(self, object_ids: str = "", timeout_s: float = 30.0) -> ToolResult[dict[str, Any]]:
        """
        Clear cached object records from the object-info server.

        [[if:text]]Text output: ClearObjects service result and stdout/stderr.[[/if:text]]

        Args:
            object_ids: Optional JSON array or comma-separated object IDs. Empty clears all cached objects.
            timeout_s: Maximum service-call time in seconds.

        Returns:
            dict: Observability payload with /clear_objects service output.
        """
        request = {"object_ids": _parse_int_list(object_ids)}
        return IsaacRosServiceTool(runner=self._runner, output_dir=self._output_dir).call_service(
            "/clear_objects",
            _resolved_interface_type(
                self._runner,
                "isaac_ros_manipulation_interfaces/srv/ClearObjects",
            ),
            json.dumps(request),
            timeout_s=timeout_s,
        )


class IsaacSegmentationTool(_IsaacTool):
    """Call SAM-compatible segmentation actions exposed by Isaac ROS manipulation."""

    def get_name(self) -> str:
        return "isaac_segmentation"

    @tool_method
    def segment_anything_point(
        self,
        x: float,
        y: float,
        timeout_s: float = 60.0,
    ) -> ToolResult[dict[str, Any]]:
        """
        Run the Segment Anything action with a 2D point hint.

        [[if:text]]Text output: Segmentation action result, detection/mask stdout, and next checks.[[/if:text]]

        Args:
            x: Image-space x coordinate for the point hint.
            y: Image-space y coordinate for the point hint.
            timeout_s: Maximum action wait time in seconds.

        Returns:
            dict: Observability payload with /segment_anything action output.
        """
        goal = {
            "use_point_hint": True,
            "initial_hint_point": {"x": float(x), "y": float(y)},
            "initial_hint_bbox": {
                "center": {"position": {"x": 0.0, "y": 0.0}, "theta": 0.0},
                "size_x": 0.0,
                "size_y": 0.0,
            },
        }
        return self._send_segment_anything_goal(goal, timeout_s)

    @tool_method
    def segment_anything_box(
        self,
        center_x: float,
        center_y: float,
        size_x: float,
        size_y: float,
        theta: float = 0.0,
        timeout_s: float = 60.0,
    ) -> ToolResult[dict[str, Any]]:
        """
        Run the Segment Anything action with a 2D bounding-box hint.

        [[if:text]]Text output: Segmentation action result, detection/mask stdout, and next checks.[[/if:text]]

        Args:
            center_x: Image-space box center x.
            center_y: Image-space box center y.
            size_x: Box width.
            size_y: Box height.
            theta: Box rotation angle.
            timeout_s: Maximum action wait time in seconds.

        Returns:
            dict: Observability payload with /segment_anything action output.
        """
        goal = {
            "use_point_hint": False,
            "initial_hint_point": {"x": 0.0, "y": 0.0},
            "initial_hint_bbox": {
                "center": {
                    "position": {"x": float(center_x), "y": float(center_y)},
                    "theta": float(theta),
                },
                "size_x": float(size_x),
                "size_y": float(size_y),
            },
        }
        return self._send_segment_anything_goal(goal, timeout_s)

    def _send_segment_anything_goal(self, goal: dict[str, Any], timeout_s: float) -> ToolResult[dict[str, Any]]:
        return IsaacRosActionTool(runner=self._runner, output_dir=self._output_dir).send_goal(
            "/segment_anything",
            _resolved_action_type(
                self._runner,
                "isaac_ros_manipulation_interfaces/action/SegmentAnything",
            ),
            json.dumps(goal),
            timeout_s=timeout_s,
        )


class IsaacGripperTool(_IsaacTool):
    """Open and close the simulated Robotiq gripper through ROS actions."""

    def get_name(self) -> str:
        return "isaac_gripper"

    @tool_method
    def open(self, position: float = 0.0, max_effort: float = 10.0) -> ToolResult[dict[str, Any]]:
        """
        Open the Robotiq gripper.

        [[if:text]]Text output: Gripper command action state and feedback.[[/if:text]]

        Args:
            position: Target gripper position.
            max_effort: Maximum gripper effort.

        Returns:
            dict: Observability payload with action result.
        """
        return self._send_gripper_goal(position, max_effort)

    @tool_method
    def close(self, position: float = 0.55, max_effort: float = 10.0) -> ToolResult[dict[str, Any]]:
        """
        Close the Robotiq gripper.

        [[if:text]]Text output: Gripper command action state and feedback.[[/if:text]]

        Args:
            position: Target gripper position.
            max_effort: Maximum gripper effort.

        Returns:
            dict: Observability payload with action result.
        """
        return self._send_gripper_goal(position, max_effort)

    def _send_gripper_goal(self, position: float, max_effort: float) -> ToolResult[dict[str, Any]]:
        goal = {"command": {"position": float(position), "max_effort": float(max_effort)}}
        return IsaacRosActionTool(runner=self._runner, output_dir=self._output_dir).send_goal(
            "/robotiq_gripper_controller/gripper_cmd",
            "control_msgs/action/GripperCommand",
            json.dumps(goal),
            timeout_s=30,
        )


class IsaacLaunchRecipeTool(IsaacRosLaunchTool):
    """Launch-tool facade for one registered recipe."""

    recipe_name = ""
    schema_name = ""

    def __init__(self, recipe_name: str | None = None, schema_name: str | None = None, **kwargs: Any) -> None:
        generic_methods = [
            "list_recipes",
            "describe_recipe",
            "start_recipe",
            "stop_recipe",
            "status_recipe",
            "logs_recipe",
            "inspect_recipe_outputs",
        ]
        exclude_methods = set(kwargs.pop("exclude_methods", []) or [])
        kwargs["exclude_methods"] = sorted(exclude_methods.union(generic_methods))
        super().__init__(**kwargs)
        self._recipe_name = recipe_name or self.recipe_name
        self._schema_name = schema_name or self.schema_name or f"isaac_{self._recipe_name}"

    def get_name(self) -> str:
        return self._schema_name

    @tool_method
    def start(self, overrides: str = "", restart: bool = False) -> ToolResult[dict[str, Any]]:
        """
        Start this Isaac ROS launch recipe.

        [[if:text]]Text output: Process state, resources produced by this recipe, render topics, logs, and next checks.[[/if:text]]

        Args:
            overrides: JSON object or comma-separated key=value launch-argument overrides.
            restart: Stop the recipe first if it is already running.

        Returns:
            dict: Observability payload with launch process and expected resources.
        """
        return self.start_recipe(self._recipe_name, overrides=overrides, restart=restart)

    @tool_method
    def stop(self) -> ToolResult[dict[str, Any]]:
        """
        Stop this Isaac ROS launch recipe.

        [[if:text]]Text output: Final process state and recent logs.[[/if:text]]

        Returns:
            dict: Observability payload with process state.
        """
        return self.stop_recipe(self._recipe_name)

    @tool_method
    def status(self) -> ToolResult[dict[str, Any]]:
        """
        Inspect this Isaac ROS launch recipe.

        [[if:text]]Text output: Process state, expected resources, render topics, and recent logs.[[/if:text]]

        Returns:
            dict: Observability payload with recipe status.
        """
        return self.status_recipe(self._recipe_name)

    @tool_method
    def logs(self, max_lines: int = 80) -> ToolResult[dict[str, Any]]:
        """
        Return recent logs for this Isaac ROS launch recipe.

        [[if:text]]Text output: Tail of launch logs for this recipe.[[/if:text]]

        Args:
            max_lines: Maximum log lines to return.

        Returns:
            dict: Observability payload with recent logs.
        """
        return self.logs_recipe(self._recipe_name, max_lines=max_lines)

    @tool_method
    def inspect_outputs(
        self,
        sample_timeout_s: float = 3.0,
        render_images: bool = False,
    ) -> ToolResult[dict[str, Any]]:
        """
        Inspect this recipe's produced ROS topics, actions, and services.

        [[if:text]]Text output: Produced resource readiness, missing outputs, optional render artifacts, and next checks.[[/if:text]]
        [[if:image]]Image output: Optional renders for image-like render topics when render_images=true.[[/if:image]]

        Args:
            sample_timeout_s: Seconds to sample each produced topic.
            render_images: Whether to render image-like render topics after inspection.

        Returns:
            dict: Observability payload for this recipe's advertised API surface.
        """
        return self.inspect_recipe_outputs(
            self._recipe_name,
            sample_timeout_s=sample_timeout_s,
            render_images=render_images,
        )

    @tool_method
    def describe(self) -> ToolResult[dict[str, Any]]:
        """
        Describe this Isaac ROS launch recipe.

        [[if:text]]Text output: Recipe command, dependencies, produced resources, and render topics.[[/if:text]]

        Returns:
            dict: Recipe metadata and launch command.
        """
        return self.describe_recipe(self._recipe_name)


def _recipe(recipe_name: str) -> LaunchRecipe:
    recipe_name = RECIPE_ALIASES.get(recipe_name, recipe_name)
    try:
        return RECIPES[recipe_name]
    except KeyError as exc:
        raise ValueError(f"Unknown Isaac ROS recipe '{recipe_name}'. Available: {sorted(RECIPES)}") from exc


def _make_recipe_tool_class(recipe_name: str):
    schema_name = f"isaac_{recipe_name}"
    class_name = "".join(part.title() for part in schema_name.split("_")) + "Tool"
    return type(
        class_name,
        (IsaacLaunchRecipeTool,),
        {
            "recipe_name": recipe_name,
            "schema_name": schema_name,
            "__module__": __name__,
        },
    )


for _recipe_name in [
    "cumotion",
    "dope",
    "ess",
    "foundationpose",
    "foundationstereo",
    "grounding_dino",
    "nvblox",
    "object_following",
    "pose_to_pose",
    "realsense",
    "rtdetr",
    "segment_anything",
    "segment_anything2",
    "static_transforms",
]:
    globals()["".join(part.title() for part in f"isaac_{_recipe_name}".split("_")) + "Tool"] = (
        _make_recipe_tool_class(_recipe_name)
    )
