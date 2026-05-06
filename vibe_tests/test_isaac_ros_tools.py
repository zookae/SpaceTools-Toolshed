"""Tests for Isaac ROS Toolshed primitives.

These tests use fake command/process runners so they do not require ROS 2,
Docker, or Isaac Sim. They lock the agent-facing contract: commands are
constructed correctly and tool results include enough state for decision making.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import time
from pathlib import Path

import pytest
import yaml

from toolshed.tool_manifest import TOOL_MANIFEST
from toolshed.tool_result import ToolResult
from toolshed.integration.providers.bedrock_provider import _redact_secrets
from toolshed.integration.providers import nvidia_openai_provider
from toolshed.integration.providers.nvidia_openai_provider import NvidiaOpenAIProvider
from toolshed.tools.isaac_ros import (
    IsaacNvbloxTool,
    IsaacObjectInfoTool,
    IsaacPickPlaceTool,
    IsaacRtdetrTool,
    IsaacRosActionTool,
    IsaacRosConfigTool,
    IsaacRosGraphTool,
    IsaacRosLaunchTool,
    IsaacRosParamTool,
    IsaacRosServiceTool,
    IsaacRosSceneTool,
    IsaacRosTfTool,
    IsaacRosTopicTool,
    IsaacSegmentationTool,
    build_multi_bin_goal,
    build_single_bin_goal,
)
from toolshed.tools.isaac_ros_core import (
    CommandResult,
    DEFAULT_ACTION_TYPE,
    ProcessState,
    RECIPES,
    RosCommandRunner,
)


class FakeRunner:
    def __init__(self, responses: dict[tuple[str, ...], CommandResult] | None = None):
        self.responses = responses or {}
        self.calls: list[list[str]] = []

    def run(self, args, timeout_s=30):
        args = [str(arg) for arg in args]
        self.calls.append(args)
        return self.responses.get(
            tuple(args),
            CommandResult(args=args, returncode=0, stdout="", stderr=""),
        )

    def start(self, name, args, log_path, env=None):
        self.calls.append(["START", name, *[str(arg) for arg in args]])
        return ProcessState(
            name=name,
            command=[str(arg) for arg in args],
            running=True,
            returncode=None,
            log_path=str(log_path),
        )

    def stop(self, name):
        self.calls.append(["STOP", name])
        return ProcessState(name=name, command=[], running=False, returncode=0, log_path=None)

    def status(self, name):
        return ProcessState(
            name=name,
            command=["ros2", "launch"],
            running=True,
            returncode=None,
            log_path="/tmp/fake.log",
        )

    def logs(self, name, max_lines=80):
        return ["[INFO] fake launch ready", "[WARN] fake warning"]


def test_ros_command_runner_builds_docker_exec_with_ros_environment():
    runner = RosCommandRunner(container_name="isaac", ros_domain_id="42")

    command = runner.wrap_ros_command(["ros2", "node", "list"])

    assert command[:3] == ["docker", "exec", "-i"]
    assert "isaac" in command
    shell_command = command[-1]
    assert "export ROS_DOMAIN_ID=42" in shell_command
    assert "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" in shell_command
    assert "/workspaces/isaac_ros-dev/install/setup.bash" in shell_command
    assert "ros2 node list" in shell_command


def test_cumotion_overlay_defers_result_publication_until_callback_returns():
    source = inspect.getsource(RosCommandRunner.ensure_cumotion_goalset_compatibility)

    assert "result_timeout=2147483647" in source
    assert "rclpy_implementation as _rclpy" in source
    assert "goal_handle._update_state(_rclpy.GoalEvent.SUCCEED)" in source
    assert 'patched_lines.append(f"{indent}self._toolshed_mark_goal_succeeded(goal_handle)\\n")' in source
    assert 'patched_lines.append(f"{indent}goal_handle.succeed()\\n")' not in source


def test_isaac_sim_camera_resolution_overlay_is_env_driven():
    source = inspect.getsource(RosCommandRunner.ensure_isaac_sim_camera_resolution_compatibility)

    assert "ISAAC_IMAGE_PUBLISHER_WIDTH" in source
    assert "ISAAC_IMAGE_PUBLISHER_HEIGHT" in source
    assert "HAWK_IMAGE_WIDTH" in source
    assert "HAWK_IMAGE_HEIGHT" in source
    assert "__init__.py" in source
    assert "init_text" in source
    assert "toolshed_isaac_ros_manipulator_overlay" in source


def test_recipe_registry_contains_launch_include_recipes_with_observability():
    expected = {
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
    }

    assert expected.issubset(RECIPES)
    rtdetr = RECIPES["rtdetr"]
    assert rtdetr.package == "isaac_ros_manipulation_bringup"
    assert rtdetr.launch_file == "launch/include/rtdetr.launch.py"
    assert "/detections" in rtdetr.produced_topics
    assert rtdetr.render_topics
    assert "/get_objects" in RECIPES["pick_and_place_workflow"].produced_actions
    assert "/isaac_joint_states" in RECIPES["pick_and_place_workflow"].required_topics
    assert RECIPES["pick_and_place_workflow"].cleanup_patterns


def test_graph_list_returns_stateful_payload_with_next_actions():
    runner = FakeRunner(
        {
            ("ros2", "node", "list"): CommandResult(
                ["ros2", "node", "list"], 0, "/foo\n/bar\n", ""
            ),
            ("ros2", "topic", "list"): CommandResult(
                ["ros2", "topic", "list"], 0, "/image\n/depth\n", ""
            ),
            ("ros2", "service", "list"): CommandResult(
                ["ros2", "service", "list"], 0, "/reset\n", ""
            ),
            ("ros2", "action", "list"): CommandResult(
                ["ros2", "action", "list"], 0, "/multi_object_pick_and_place\n", ""
            ),
            ("ros2", "run", "tf2_tools", "view_frames"): CommandResult(
                ["ros2", "run", "tf2_tools", "view_frames"], 1, "", "tf unavailable"
            ),
        }
    )
    tool = IsaacRosGraphTool(runner=runner)

    result = tool.list_graph()

    assert isinstance(result, ToolResult)
    assert result.value["ok"] is True
    assert result.value["state"] == "ready"
    assert "/foo" in result.value["nodes"]
    assert "/multi_object_pick_and_place" in result.value["actions"]
    assert result.value["next_suggested_actions"]
    assert "recent_logs" in result.value


def test_graph_wait_for_corrects_action_requested_as_service():
    runner = FakeRunner(
        {
            ("ros2", "node", "list"): CommandResult(["ros2", "node", "list"], 0, "", ""),
            ("ros2", "topic", "list"): CommandResult(["ros2", "topic", "list"], 0, "", ""),
            ("ros2", "service", "list"): CommandResult(["ros2", "service", "list"], 0, "", ""),
            ("ros2", "action", "list"): CommandResult(
                ["ros2", "action", "list"], 0, "/get_objects\n", ""
            ),
        }
    )
    tool = IsaacRosGraphTool(runner=runner)

    result = tool.wait_for(required_services="/get_objects", timeout_s=0, poll_s=0)

    assert result.value["ok"] is True
    assert result.value["missing"] == []
    assert result.value["resource_category_corrections"] == [
        {"requested": "services:/get_objects", "actual": "actions:/get_objects"}
    ]


def test_graph_wait_for_stable_state_samples_clock_and_joint_state(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda duration: sleeps.append(duration))
    runner = FakeRunner(
        {
            ("ros2", "topic", "echo", "--once", "/clock"): CommandResult(
                ["ros2", "topic", "echo", "--once", "/clock"], 0, "clock: ok", ""
            ),
            ("ros2", "topic", "echo", "--once", "/isaac_joint_states"): CommandResult(
                ["ros2", "topic", "echo", "--once", "/isaac_joint_states"], 0, "joint: ok", ""
            ),
        }
    )

    result = IsaacRosGraphTool(runner=runner).wait_for_stable_state(duration_s=2.5)

    assert result.value["ok"] is True
    assert result.value["state"] == "stable_wait_completed"
    assert result.value["samples"] == {
        "before_clock": True,
        "before_joint": True,
        "after_clock": True,
        "after_joint": True,
    }
    assert sleeps == [2.5]
    assert runner.calls == [
        ["ros2", "topic", "echo", "--once", "/clock"],
        ["ros2", "topic", "echo", "--once", "/isaac_joint_states"],
        ["ros2", "topic", "echo", "--once", "/clock"],
        ["ros2", "topic", "echo", "--once", "/isaac_joint_states"],
    ]


def test_graph_inspect_node_returns_raw_node_interfaces():
    runner = FakeRunner(
        {
            ("ros2", "node", "info", "/planner"): CommandResult(
                ["ros2", "node", "info", "/planner"],
                0,
                "Subscribers:\n  /joint_states: sensor_msgs/msg/JointState\nAction Servers:\n  /motion_plan\n",
                "",
            )
        }
    )

    result = IsaacRosGraphTool(runner=runner).inspect_node("/planner")

    assert result.value["ok"] is True
    assert result.value["state"] == "node_ready"
    assert "/motion_plan" in result.value["info"]


def test_launch_recipe_start_expands_to_ros2_launch_and_returns_status():
    runner = FakeRunner()
    tool = IsaacRosLaunchTool(runner=runner, output_dir=Path("/tmp/isaac-test"))

    result = tool.start_recipe("rtdetr", overrides='{"object_class_id": "3"}')

    assert result.value["ok"] is True
    assert result.value["recipe"] == "rtdetr"
    start_call = runner.calls[0]
    assert start_call[:3] == ["START", "rtdetr", "ros2"]
    assert "isaac_ros_manipulation_bringup" in start_call
    assert "launch/include/rtdetr.launch.py" in start_call
    assert "object_class_id:=3" in start_call
    assert result.value["produced_topics"]
    assert result.value["render_topics"]


def test_launch_recipe_aliases_resolve_common_pick_place_names():
    runner = FakeRunner()
    tool = IsaacRosLaunchTool(runner=runner, output_dir=Path("/tmp/isaac-test"))

    result = tool.describe_recipe("pick_place")

    assert result.value["ok"] is True
    assert result.value["recipe"]["name"] == "pick_and_place_workflow"


def test_unknown_launch_recipe_returns_structured_feedback():
    result = IsaacRosLaunchTool(runner=FakeRunner()).describe_recipe("pick_and_place_tutorial")

    assert result.value["ok"] is False
    assert result.value["state"] == "unknown_recipe"
    assert "pick_and_place_workflow" in result.value["available_recipes"]
    assert "Call list_recipes" in result.value["next_suggested_actions"][0]


def test_stop_recipe_applies_registered_remote_cleanup_patterns():
    class CleanupRunner(FakeRunner):
        def stop_remote_patterns(self, patterns):
            self.calls.append(["REMOTE_CLEANUP", *patterns])

        def status(self, name):
            return ProcessState(
                name=name,
                command=[],
                running=False,
                returncode=0,
                log_path="/tmp/fake.log",
            )

    runner = CleanupRunner()
    tool = IsaacRosLaunchTool(runner=runner, output_dir=Path("/tmp/isaac-test"))

    result = tool.stop_recipe("pick_and_place_workflow")

    assert result.value["ok"] is True
    assert ["STOP", "pick_and_place_workflow"] in runner.calls
    cleanup_call = next(call for call in runner.calls if call[0] == "REMOTE_CLEANUP")
    assert "workflows.launch.py manipulator_workflow_config:=" in cleanup_call


def test_pick_place_goal_builders_match_tutorial_payload_shape():
    single = build_single_bin_goal(
        drop_pose=[-0.25, -0.35, 0.50, -0.677772, 0.734752, 0.020993, 0.017994]
    )
    multi = build_multi_bin_goal(
        target_poses=[
            [-0.25, -0.35, 0.50, -0.677772, 0.734752, 0.020993, 0.017994],
            [-0.25, 0.40, 0.40, -0.677772, 0.734752, 0.020993, 0.017994],
        ],
        class_ids=["22", "3"],
    )

    assert single["mode"] == 0
    assert single["class_ids"] == []
    assert single["target_poses"]["header"]["frame_id"] == "base_link"
    assert single["target_poses"]["poses"][0]["orientation"]["w"] == 0.017994
    assert multi["mode"] == 1
    assert multi["class_ids"] == ["22", "3"]


def test_topic_inspection_and_snapshot_return_parsed_schema_and_message():
    topic = "/joint_states"
    responses = {
        ("ros2", "topic", "info", topic): CommandResult(["ros2", "topic", "info", topic], 0, "Publisher count: 1\n", ""),
        ("ros2", "topic", "type", topic): CommandResult(["ros2", "topic", "type", topic], 0, "sensor_msgs/msg/JointState\n", ""),
        ("ros2", "topic", "hz", topic): CommandResult(["ros2", "topic", "hz", topic], 0, "average rate: 10.0\n", ""),
        ("ros2", "interface", "show", "sensor_msgs/msg/JointState"): CommandResult(
            ["ros2", "interface", "show", "sensor_msgs/msg/JointState"],
            0,
            "std_msgs/Header header\nstring[] name\nfloat64[] position\n",
            "",
        ),
        ("ros2", "topic", "echo", "--once", topic): CommandResult(
            ["ros2", "topic", "echo", "--once", topic],
            0,
            "name: ['shoulder_pan_joint']\nposition: [1.2]\n",
            "",
        ),
    }
    tool = IsaacRosTopicTool(runner=FakeRunner(responses))

    inspected = tool.inspect_topic(topic)
    snapshot = tool.snapshot(topic)

    assert inspected.value["schema"]["fields"]["message"][0]["name"] == "header"
    assert snapshot.value["parsed_message"]["name"] == ["shoulder_pan_joint"]
    assert snapshot.value["parsed_message"]["position"] == [1.2]


def test_action_and_pick_place_tools_return_action_state_and_observability():
    info_result = CommandResult(
        ["ros2", "action", "info", "/multi_object_pick_and_place"],
        0,
        "Action clients: 0\nAction servers: 1\n    /multi_object_pick_and_place\n",
        "",
    )
    runner = FakeRunner({("ros2", "action", "info", "/multi_object_pick_and_place"): info_result})
    action_tool = IsaacRosActionTool(runner=runner)
    pick_tool = IsaacPickPlaceTool(runner=runner)

    action_result = action_tool.send_goal(
        "/multi_object_pick_and_place",
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
        '{"mode": 0}',
        timeout_s=10,
    )
    pick_result = pick_tool.send_single_bin_goal(timeout_s=10)

    assert action_result.value["ok"] is False
    assert action_result.value["state"] == "invalid_action_timeout"
    assert action_result.value["action_name"] == "/multi_object_pick_and_place"
    assert action_result.value["recommended_timeout_s"] == 900.0
    assert pick_result.value["goal"]["mode"] == 0
    assert pick_result.value["state"] == "invalid_action_timeout"
    assert pick_result.value["next_suggested_actions"]
    assert not any(call[:4] == ["ros2", "action", "send_goal", "--feedback"] for call in runner.calls)


def test_action_info_requires_a_server_not_just_a_client():
    runner = FakeRunner(
        {
            ("ros2", "action", "info", "/multi_object_pick_and_place"): CommandResult(
                ["ros2", "action", "info", "/multi_object_pick_and_place"],
                0,
                "Action clients: 1\n    /client\nAction servers: 0\n",
                "",
            )
        }
    )
    action_tool = IsaacRosActionTool(runner=runner)

    watch_result = action_tool.watch_action("/multi_object_pick_and_place")
    send_result = action_tool.send_goal(
        "/multi_object_pick_and_place",
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
        '{"mode": 0}',
        timeout_s=10,
    )

    assert watch_result.value["ok"] is False
    assert watch_result.value["state"] == "no_action_server"
    assert watch_result.value["action_clients"] == 1
    assert watch_result.value["action_servers"] == 0
    assert send_result.value["state"] == "no_action_server"
    assert not any(call[:4] == ["ros2", "action", "send_goal", "--feedback"] for call in runner.calls)


def test_pick_place_action_rejects_timeout_that_would_kill_client_too_early():
    info_command = ("ros2", "action", "info", "/multi_object_pick_and_place")
    runner = FakeRunner(
        {
            info_command: CommandResult(
                list(info_command),
                0,
                "Action clients: 0\nAction servers: 1\n",
                "",
            ),
        }
    )
    action_tool = IsaacRosActionTool(runner=runner)

    result = action_tool.send_goal(
        "/multi_object_pick_and_place",
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
        '{"mode": 0}',
        timeout_s=45,
    )

    assert result.value["ok"] is False
    assert result.value["state"] == "invalid_action_timeout"
    assert result.value["missing"] == []
    assert result.value["recommended_timeout_s"] == 900.0
    assert "Recommended timeout: 900.0s." in result.text
    assert not any(call[:4] == ["ros2", "action", "send_goal", "--feedback"] for call in runner.calls)


def test_action_timeout_after_acceptance_is_not_reported_as_missing_server():
    info_command = ("ros2", "action", "info", "/slow_action")
    goal_command = (
        "ros2",
        "action",
        "send_goal",
        "--feedback",
        "/slow_action",
        "example_interfaces/action/Fibonacci",
        '{"order": 10}',
    )
    runner = FakeRunner(
        {
            info_command: CommandResult(
                list(info_command),
                0,
                "Action clients: 0\nAction servers: 1\n",
                "",
            ),
            goal_command: CommandResult(
                list(goal_command),
                124,
                "Goal accepted with ID: abc123\n",
                "Timed out after 45s",
            )
        }
    )
    action_tool = IsaacRosActionTool(runner=runner)

    result = action_tool.send_goal(
        "/slow_action",
        "example_interfaces/action/Fibonacci",
        '{"order": 10}',
        timeout_s=45,
    )

    assert result.value["ok"] is False
    assert result.value["state"] == "goal_timeout"
    assert result.value["missing"] == []
    assert any("Timed out after 45s" in line for line in result.value["recent_logs"])


def test_action_aborted_terminal_status_is_not_reported_as_success():
    info_command = ("ros2", "action", "info", "/multi_object_pick_and_place")
    goal_text = json.dumps(build_single_bin_goal())
    goal_command = (
        "ros2",
        "action",
        "send_goal",
        "--feedback",
        "/multi_object_pick_and_place",
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
        goal_text,
    )
    runner = FakeRunner(
        {
            info_command: CommandResult(
                list(info_command),
                0,
                "Action clients: 0\nAction servers: 1\n",
                "",
            ),
            ("ros2", "interface", "show", "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace"): CommandResult(
                ["ros2", "interface", "show", "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace"],
                0,
                "geometry_msgs/PoseArray target_poses\nstring[] class_ids\nint32 mode\n---\nint32 workflow_status\n---\nstring current_state\n",
                "",
            ),
            goal_command: CommandResult(
                list(goal_command),
                0,
                "Goal accepted with ID: abc123\nResult:\n    workflow_status: 3\nGoal finished with status: ABORTED\n",
                "",
            )
        }
    )
    action_tool = IsaacRosActionTool(runner=runner)

    result = action_tool.send_goal(
        "/multi_object_pick_and_place",
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
        goal_text,
        timeout_s=900,
    )

    assert result.value["ok"] is False
    assert result.value["state"] == "goal_aborted"
    assert result.value["terminal_status"] == "ABORTED"
    assert result.value["workflow_status"] == "3"
    assert result.value["missing"] == []
    assert "Goal finished with status: ABORTED" in result.text


def test_action_succeeded_terminal_status_is_reported_as_success():
    info_command = ("ros2", "action", "info", "/multi_object_pick_and_place")
    goal_text = json.dumps(build_single_bin_goal())
    goal_command = (
        "ros2",
        "action",
        "send_goal",
        "--feedback",
        "/multi_object_pick_and_place",
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
        goal_text,
    )
    runner = FakeRunner(
        {
            info_command: CommandResult(
                list(info_command),
                0,
                "Action clients: 0\nAction servers: 1\n",
                "",
            ),
            ("ros2", "interface", "show", "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace"): CommandResult(
                ["ros2", "interface", "show", "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace"],
                0,
                "geometry_msgs/PoseArray target_poses\nstring[] class_ids\nint32 mode\n---\nint32 workflow_status\n---\nstring current_state\n",
                "",
            ),
            goal_command: CommandResult(
                list(goal_command),
                0,
                "Goal accepted with ID: abc123\nResult:\n    workflow_status: 2\nGoal finished with status: SUCCEEDED\n",
                "",
            )
        }
    )
    action_tool = IsaacRosActionTool(runner=runner)

    result = action_tool.send_goal(
        "/multi_object_pick_and_place",
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
        goal_text,
        timeout_s=900,
    )

    assert result.value["ok"] is True
    assert result.value["state"] == "goal_succeeded"
    assert result.value["terminal_status"] == "SUCCEEDED"
    assert result.value["workflow_status"] == "2"
    assert result.value["parsed_result"]["workflow_status"] == 2
    assert result.value["schema"]["fields"]["goal"][0]["name"] == "target_poses"
    assert "finished with status SUCCEEDED" in result.text


def test_action_aborted_goal_surfaces_execute_trajectory_failure():
    info_command = ("ros2", "action", "info", "/multi_object_pick_and_place")
    goal_text = json.dumps(build_single_bin_goal())
    goal_command = (
        "ros2",
        "action",
        "send_goal",
        "--feedback",
        "/multi_object_pick_and_place",
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
        goal_text,
    )

    class LogRunner(FakeRunner):
        def logs(self, name, max_lines=80):
            assert name == "pick_and_place_workflow"
            return [
                "[INFO] [tree]: [Execute Lift] Starting trajectory execution for object_id=0, trajectory_index=1",
                "[ERROR] [move_group.moveit.moveit.ros.trajectory_execution_manager]: Invalid Trajectory: start point deviates from current robot state more than 0.1 at joint 'shoulder_pan_joint'.",
                "[INFO] [move_group.moveit.moveit.ros.move_group.clear_octomap_service]: Execution completed: ABORTED",
                "[ERROR] [tree]: [Execute Lift] execute_trajectory action server aborted",
                "[ERROR] [tree]: [Execute Lift] Action execute_trajectory failed",
            ]

    runner = LogRunner(
        {
            info_command: CommandResult(list(info_command), 0, "Action servers: 1\n", ""),
            (
                "ros2",
                "interface",
                "show",
                "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
            ): CommandResult(
                [
                    "ros2",
                    "interface",
                    "show",
                    "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
                ],
                0,
                "geometry_msgs/PoseArray target_poses\nstring[] class_ids\nint32 mode\n---\nint32 workflow_status\n---\nstring current_state\n",
                "",
            ),
            goal_command: CommandResult(
                list(goal_command),
                0,
                "Goal accepted with ID: abc123\nResult:\n    workflow_status: 0\nGoal finished with status: ABORTED\n",
                "",
            ),
        }
    )

    result = IsaacRosActionTool(runner=runner).send_goal(
        "/multi_object_pick_and_place",
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
        goal_text,
        timeout_s=900,
    )

    assert result.value["state"] == "goal_aborted"
    assert any("Invalid Trajectory" in line for line in result.value["action_error_summary"])
    assert any("execute_trajectory action server aborted" in line for line in result.value["action_error_summary"])
    assert "shoulder_pan_joint" in result.text


def test_action_accepted_without_terminal_status_is_not_success():
    info_command = ("ros2", "action", "info", "/multi_object_pick_and_place")
    goal_text = json.dumps(build_single_bin_goal())
    goal_command = (
        "ros2",
        "action",
        "send_goal",
        "--feedback",
        "/multi_object_pick_and_place",
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
        goal_text,
    )

    class LogRunner(FakeRunner):
        def logs(self, name, max_lines=80):
            assert name == "pick_and_place_workflow"
            return [
                "[ERROR] [cumotion_planner]: Toolshed compatibility: plan_grasp failed; status=No grasp in goal set was reachable.",
                "Traceback (most recent call last):",
                "  File \"/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/action/server.py\", line 377, in _execute_goal",
                "KeyError: b'goal'",
            ]

    runner = LogRunner(
        {
            info_command: CommandResult(
                list(info_command),
                0,
                "Action clients: 0\nAction servers: 1\n",
                "",
            ),
            ("ros2", "interface", "show", "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace"): CommandResult(
                ["ros2", "interface", "show", "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace"],
                0,
                "geometry_msgs/PoseArray target_poses\nstring[] class_ids\nint32 mode\n---\nint32 workflow_status\n---\nstring current_state\n",
                "",
            ),
            goal_command: CommandResult(
                list(goal_command),
                0,
                "Goal accepted with ID: abc123\nFeedback:\n    current_state: planning\n",
                "",
            )
        }
    )

    result = IsaacRosActionTool(runner=runner).send_goal(
        "/multi_object_pick_and_place",
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
        goal_text,
        timeout_s=900,
    )

    assert result.value["ok"] is False
    assert result.value["state"] == "goal_status_unknown"
    assert result.value["terminal_status"] == ""
    assert result.value["missing"] == []
    assert any("plan_grasp failed" in line for line in result.value["action_error_summary"])
    assert "no terminal status" in result.text
    assert "KeyError" in result.text


def test_action_rejects_goal_missing_schema_fields_before_send():
    info_command = ("ros2", "action", "info", "/multi_object_pick_and_place")
    runner = FakeRunner(
        {
            info_command: CommandResult(list(info_command), 0, "Action servers: 1\n", ""),
            ("ros2", "interface", "show", "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace"): CommandResult(
                ["ros2", "interface", "show", "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace"],
                0,
                "geometry_msgs/PoseArray target_poses\nstring[] class_ids\nint32 mode\n---\nint32 workflow_status\n---\nstring current_state\n",
                "",
            ),
        }
    )

    result = IsaacRosActionTool(runner=runner).send_goal(
        "/multi_object_pick_and_place",
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace",
        "{}",
        timeout_s=45,
    )

    assert result.value["ok"] is False
    assert result.value["state"] == "invalid_goal_schema"
    assert result.value["missing"] == ["target_poses", "class_ids", "mode"]
    assert result.value["goal_template"]["target_poses"]["poses"]
    assert "Goal template:" in result.text
    assert not any(call[:4] == ["ros2", "action", "send_goal", "--feedback"] for call in runner.calls)


def test_action_schema_template_uses_only_top_level_goal_fields():
    action = "/multi_object_pick_and_place"
    action_type = "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace"
    responses = {
        ("ros2", "action", "info", action): CommandResult(
            ["ros2", "action", "info", action], 0, "Action servers: 1\n", ""
        ),
        ("ros2", "action", "list", "-t"): CommandResult(
            ["ros2", "action", "list", "-t"], 0, f"{action} [{action_type}]\n", ""
        ),
        ("ros2", "interface", "show", action_type): CommandResult(
            ["ros2", "interface", "show", action_type],
            0,
            (
                "uint8 mode\n"
                "geometry_msgs/PoseArray target_poses\n"
                "  std_msgs/Header header\n"
                "    builtin_interfaces/Time stamp\n"
                "      int32 sec\n"
                "      uint32 nanosec\n"
                "    string frame_id\n"
                "  Pose[] poses\n"
                "    Point position\n"
                "      float64 x\n"
                "      float64 y\n"
                "      float64 z\n"
                "    Quaternion orientation\n"
                "      float64 x\n"
                "      float64 y\n"
                "      float64 z\n"
                "      float64 w\n"
                "string[] class_ids\n"
                "---\n"
                "int32 workflow_status\n"
                "---\n"
                "string current_state\n"
            ),
            "",
        ),
    }

    result = IsaacRosActionTool(runner=FakeRunner(responses)).watch_action(action)

    assert result.value["schema_summary"] == "mode:uint8, target_poses:geometry_msgs/PoseArray, class_ids:string[]"
    assert set(result.value["goal_template"]) == {"mode", "target_poses", "class_ids"}
    assert "header" not in result.value["goal_template"]


def test_action_rejects_empty_pose_array_from_goal_template():
    action = "/multi_object_pick_and_place"
    action_type = "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace"
    responses = {
        ("ros2", "action", "info", action): CommandResult(
            ["ros2", "action", "info", action], 0, "Action servers: 1\n", ""
        ),
        ("ros2", "interface", "show", action_type): CommandResult(
            ["ros2", "interface", "show", action_type],
            0,
            "uint8 mode\ngeometry_msgs/PoseArray target_poses\nstring[] class_ids\n---\nint32 workflow_status\n---\n",
            "",
        ),
    }
    goal = {
        "mode": 0,
        "target_poses": {"header": {"frame_id": "base_link"}, "poses": []},
        "class_ids": [],
    }

    result = IsaacRosActionTool(runner=FakeRunner(responses)).send_goal(
        action,
        action_type,
        json.dumps(goal),
        timeout_s=45,
    )

    assert result.value["ok"] is False
    assert result.value["state"] == "invalid_goal_schema"
    assert result.value["missing"] == ["target_poses.poses[0]"]


def test_service_call_returns_schema_and_parsed_response():
    service = "/controller_manager/list_controllers"
    service_type = "controller_manager_msgs/srv/ListControllers"
    responses = {
        ("ros2", "service", "type", service): CommandResult(["ros2", "service", "type", service], 0, f"{service_type}\n", ""),
        ("ros2", "service", "list"): CommandResult(["ros2", "service", "list"], 0, f"{service}\n", ""),
        ("ros2", "interface", "show", service_type): CommandResult(
            ["ros2", "interface", "show", service_type],
            0,
            "---\nstring name\nstring state\n",
            "",
        ),
        ("ros2", "service", "call", service, service_type, "{}"): CommandResult(
            ["ros2", "service", "call", service, service_type, "{}"],
            0,
            "response:\n  controller:\n  - name: joint_state_broadcaster\n    state: active\n",
            "",
        ),
    }
    tool = IsaacRosServiceTool(runner=FakeRunner(responses))

    inspected = tool.inspect_service(service)
    called = tool.call_service(service, service_type, "{}")

    assert inspected.value["schema"]["fields"]["response"][0]["name"] == "name"
    assert called.value["parsed_request"] == {}
    assert called.value["parsed_response"]["controller"][0]["state"] == "active"


def test_service_inspection_points_to_action_when_resource_is_action():
    resource = "/cumotion/motion_plan"
    responses = {
        ("ros2", "service", "type", resource): CommandResult(["ros2", "service", "type", resource], 1, "", "Unknown service"),
        ("ros2", "service", "list"): CommandResult(["ros2", "service", "list"], 0, "", ""),
        ("ros2", "action", "list"): CommandResult(["ros2", "action", "list"], 0, f"{resource}\n", ""),
    }

    result = IsaacRosServiceTool(runner=FakeRunner(responses)).inspect_service(resource)

    assert result.value["ok"] is False
    assert result.value["state"] == "resource_is_action"
    assert result.value["missing"] == []
    assert result.value["resource_category_corrections"] == [
        {"requested": f"services:{resource}", "actual": f"actions:{resource}"}
    ]
    assert "watch_action" in result.value["next_suggested_actions"][0]


def test_param_and_tf_tools_expose_low_level_ros_state():
    runner = FakeRunner(
        {
            ("ros2", "param", "list", "/node"): CommandResult(
                ["ros2", "param", "list", "/node"], 0, "  use_sim_time\n  threshold\n", ""
            ),
            ("ros2", "param", "get", "/node", "use_sim_time"): CommandResult(
                ["ros2", "param", "get", "/node", "use_sim_time"], 0, "Boolean value is: True\n", ""
            ),
            ("ros2", "param", "set", "/node", "threshold", "0.5"): CommandResult(
                ["ros2", "param", "set", "/node", "threshold", "0.5"], 0, "Set parameter successful\n", ""
            ),
            ("ros2", "param", "dump", "/node"): CommandResult(
                ["ros2", "param", "dump", "/node"], 0, "/node:\n  ros__parameters:\n    threshold: 0.5\n", ""
            ),
            ("ros2", "run", "tf2_tools", "view_frames"): CommandResult(
                ["ros2", "run", "tf2_tools", "view_frames"], 0, "base_link\nfront_stereo_camera_left\n", ""
            ),
            ("ros2", "run", "tf2_ros", "tf2_echo", "base_link", "front_stereo_camera_left", "--once"): CommandResult(
                ["ros2", "run", "tf2_ros", "tf2_echo", "base_link", "front_stereo_camera_left", "--once"],
                0,
                "At time 0.0\n- Translation: [0.1, 0.2, 0.3]\n",
                "",
            ),
        }
    )
    param_tool = IsaacRosParamTool(runner=runner)
    tf_tool = IsaacRosTfTool(runner=runner)

    assert param_tool.list_params("/node").value["params"] == ["use_sim_time", "threshold"]
    assert param_tool.get_param("/node", "use_sim_time").value["ok"] is True
    assert param_tool.set_param("/node", "threshold", "0.5").value["parsed_value"] == 0.5
    assert param_tool.dump_params("/node").value["params"]["/node"]["ros__parameters"]["threshold"] == 0.5
    assert "base_link" in tf_tool.list_frames().value["frames"]
    assert tf_tool.lookup_transform("base_link", "front_stereo_camera_left").value["ok"] is True


def test_launch_logs_surface_critical_error_lines():
    class LogRunner(FakeRunner):
        def logs(self, name, max_lines=80):
            return [
                "[INFO] ready",
                "[ERROR] [tree]: [Plan To Grasp] Failed to plan grasp for object_id=1",
                "Traceback (most recent call last):",
                "  File \"/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/action/server.py\", line 357, in _execute_goal",
                "    execute_result = await await_or_execute(execute_callback, goal_handle)",
                "IndexError: index 99 is out of bounds for dimension 1 with size 1",
            ]

    result = IsaacRosLaunchTool(
        runner=LogRunner(), output_dir=Path("/tmp/isaac-test")
    ).logs_recipe("pick_and_place_workflow")

    assert result.value["error_summary"] == [
        "[ERROR] [tree]: [Plan To Grasp] Failed to plan grasp for object_id=1",
        "Traceback (most recent call last):",
        "  File \"/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/action/server.py\", line 357, in _execute_goal",
        "    execute_result = await await_or_execute(execute_callback, goal_handle)",
        "IndexError: index 99 is out of bounds for dimension 1 with size 1",
    ]
    assert "Critical logs:" in result.text
    assert "_execute_goal" in result.text


def test_pick_place_launch_uses_prepared_config_overlay_when_available():
    class ContainerRunner(FakeRunner):
        container_name = "isaac_ros_dev_container"

        def read_file(self, path, timeout_s=10):
            return CommandResult(["read", str(path)], 0, "workflow_type: PICK_AND_PLACE\n", "")

    runner = ContainerRunner()

    result = IsaacRosLaunchTool(
        runner=runner, output_dir=Path("/tmp/isaac-test")
    ).start_recipe("pick_and_place", overrides="enable_nvblox=false")

    assert result.value["ok"] is True
    assert "manipulator_workflow_config:=/tmp/toolshed_pick_place_config.yaml" in result.value["command"]
    assert any("prepared pick-and-place config" in warning for warning in result.value["warnings"])
    assert "Warnings:" in result.text


def test_pick_place_launch_installs_isaac_sim_camera_resolution_overlay():
    class CompatRunner(FakeRunner):
        container_name = "isaac_ros_dev_container"

        def read_file(self, path, timeout_s=10):
            return CommandResult(["read", str(path)], 0, "workflow_type: PICK_AND_PLACE\n", "")

        def ensure_isaac_sim_camera_resolution_compatibility(self):
            self.calls.append(["ENSURE_ISAAC_SIM_CAMERA_RESOLUTION"])
            return CommandResult(["ensure_isaac_sim_camera_resolution_compatibility"], 0, "ok", "")

    runner = CompatRunner()

    result = IsaacRosLaunchTool(
        runner=runner, output_dir=Path("/tmp/isaac-test")
    ).start_recipe("pick_and_place")

    assert result.value["ok"] is True
    assert ["ENSURE_ISAAC_SIM_CAMERA_RESOLUTION"] in runner.calls
    assert "ok" in result.value["recent_logs"]


def test_pick_place_recipe_treats_perception_topics_as_event_triggered():
    responses = {
        ("ros2", "node", "list"): CommandResult(["ros2", "node", "list"], 0, "/tree\n", ""),
        ("ros2", "topic", "list"): CommandResult(["ros2", "topic", "list"], 0, "", ""),
        ("ros2", "service", "list"): CommandResult(["ros2", "service", "list"], 0, "", ""),
        ("ros2", "action", "list"): CommandResult(["ros2", "action", "list"], 0, "", ""),
    }
    for topic, topic_type, streaming in [
        ("/detections", "vision_msgs/msg/Detection3DArray", False),
        ("/pose_estimation/output", "geometry_msgs/msg/PoseArray", False),
        ("/front_stereo_camera/left/image_raw", "sensor_msgs/msg/Image", True),
        ("/front_stereo_camera/depth/ground_truth", "sensor_msgs/msg/Image", True),
    ]:
        responses[("ros2", "topic", "info", topic)] = CommandResult(
            ["ros2", "topic", "info", topic], 0, f"Type: {topic_type}\nPublisher count: 1\n", ""
        )
        responses[("ros2", "topic", "type", topic)] = CommandResult(
            ["ros2", "topic", "type", topic], 0, f"{topic_type}\n", ""
        )
        responses[("ros2", "topic", "hz", topic)] = CommandResult(
            ["ros2", "topic", "hz", topic],
            0 if streaming else 124,
            "average rate: 30.0\n" if streaming else "",
            "" if streaming else "Timed out",
        )
    for action in [
        "/multi_object_pick_and_place",
        "/get_objects",
        "/get_object_pose",
        "/cumotion/motion_plan",
    ]:
        responses[("ros2", "action", "info", action)] = CommandResult(
            ["ros2", "action", "info", action], 0, "Action clients: 0\nAction servers: 1\n", ""
        )

    result = IsaacRosLaunchTool(
        runner=FakeRunner(responses), output_dir=Path("/tmp/isaac-test")
    ).inspect_recipe_outputs("pick_and_place_workflow")

    assert result.value["ok"] is True
    assert result.value["state"] == "ready_waiting_for_triggered_outputs"
    assert result.value["missing"] == []
    assert result.value["deferred_outputs"] == ["/detections", "/pose_estimation/output"]
    assert "Deferred/event-triggered outputs" in result.text


def test_pick_place_failed_goal_includes_workflow_error_summary():
    goal = build_single_bin_goal()
    info_command = ("ros2", "action", "info", "/multi_object_pick_and_place")
    goal_command = (
        "ros2",
        "action",
        "send_goal",
        "--feedback",
        "/multi_object_pick_and_place",
        DEFAULT_ACTION_TYPE,
        json.dumps(goal),
    )

    class LogRunner(FakeRunner):
        def logs(self, name, max_lines=80):
            return [
                "[INFO] [tree]: [Read Grasp Poses] Successfully loaded 1 grasp poses for object_id=1, class_id=3",
                "[INFO] [tree]: [Plan To Grasp] Object info for active_obj_id=1:",
                "[INFO] [tree]: [Plan To Grasp]   status: IN_MOTION",
                "[ERROR] [cumotion_planner]: joint_state was not received from /isaac_joint_states",
                "Traceback (most recent call last):",
                "  File \"/opt/ros/jazzy/lib/python3.12/site-packages/isaac_ros_cumotion/cumotion_goal_set_planner.py\", line 202, in motion_plan_execute_callback",
                "IndexError: index 99 is out of bounds for dimension 1 with size 5",
            ]

    runner = LogRunner(
        {
            info_command: CommandResult(list(info_command), 0, "Action servers: 1\n", ""),
            goal_command: CommandResult(
                list(goal_command),
                0,
                "Goal accepted with ID: abc123\nResult:\n    workflow_status: 3\nGoal finished with status: ABORTED\n",
                "",
            ),
        }
    )

    result = IsaacPickPlaceTool(runner=runner, output_dir=Path("/tmp/isaac-test")).send_single_bin_goal()

    assert result.value["state"] == "goal_aborted"
    assert "workflow_error_summary" in result.value
    assert any("joint_state was not received" in line for line in result.value["workflow_error_summary"])
    assert any("index 99" in line for line in result.value["recent_logs"])
    assert "Critical logs:" in result.text
    assert "joint_state was not received" in result.text
    assert "motion_plan_execute_callback" in result.text


def test_pick_place_config_defaults_are_headless_safe(tmp_path):
    runner = FakeRunner()
    output_path = tmp_path / "pick_place_config.yaml"
    tool = IsaacRosConfigTool(runner=runner, output_dir=tmp_path)

    result = tool.prepare_pick_and_place_config(output_path=str(output_path))

    assert result.value["ok"] is True
    assert "isaac_ros_launch.start_recipe" in result.value["next_suggested_actions"][0]
    assert result.value["config"]["enable_rviz_visualization"] == "false"
    assert result.value["config"]["controller_spawner_timeout"] == 60
    assert result.value["config"]["behavior_tree_config_file"].endswith(
        "multi_object_pick_and_place_behavior_tree_params.yaml"
    )
    assert result.value["config"]["blackboard_config_file"].endswith(
        "multi_object_pick_and_place_blackboard_params.yaml"
    )
    assert (
        result.value["behavior_tree_config"]["behavior_tree_params"]["multi_object_pick_and_place"]
        ["pose_estimation"]["camera_frame_id"]
        == "front_stereo_camera_left"
    )
    supported_objects = result.value["blackboard_config"]["blackboard_params"]["supported_objects"]
    assert supported_objects["mac_and_cheese"]["grasp_file_path"].endswith(
        "robotiq_2f_140_grasps_mac_and_cheese.yaml"
    )
    assert supported_objects["soup_can"]["grasp_file_path"].endswith(
        "robotiq_2f_140_grasps_soup_can.yaml"
    )
    switch_controllers = (
        result.value["behavior_tree_config"]["behavior_tree_params"]["multi_object_pick_and_place"]
        ["switch_controllers"]
    )
    assert switch_controllers["arm"]["controllers_to_activate"] == []
    assert switch_controllers["arm"]["controllers_to_deactivate"] == []
    assert switch_controllers["tool"]["controllers_to_activate"] == []
    assert switch_controllers["tool"]["controllers_to_deactivate"] == []
    assert output_path.exists()
    assert Path(result.value["artifacts"]["host_behavior_tree_config_path"]).exists()
    assert Path(result.value["artifacts"]["host_blackboard_config_path"]).exists()
    assert Path(result.value["artifacts"]["host_moveit_controllers_config_path"]).exists()
    assert result.value["config"]["moveit_controllers_file_path"].endswith(
        "moveit_sim_controllers_toolshed.yaml"
    )
    assert (
        result.value["moveit_controllers_config"]["trajectory_execution"]["allowed_start_tolerance"]
        == 1.0
    )


def test_pick_place_config_surfaces_forced_headless_values_in_text(tmp_path):
    runner = FakeRunner()
    tool = IsaacRosConfigTool(runner=runner, output_dir=tmp_path)

    result = tool.prepare_pick_and_place_config(enable_nvblox=True)

    assert result.value["requested_enable_nvblox"] is True
    assert result.value["config"]["enable_nvblox"] == "false"
    assert result.value["warnings"]
    assert "enable_nvblox=true is not stable" in result.text
    assert "enable_nvblox=false" in result.text


def test_pick_place_config_removes_crashing_joint_state_broadcaster_from_sim_overlay(tmp_path):
    class FileRunner(FakeRunner):
        container_name = "isaac"

        def __init__(self, files):
            super().__init__()
            self.files = files

        def package_share_path(self, package_name):
            return f"/share/{package_name}"

        def path_exists(self, path):
            return str(path) in self.files

        def read_file(self, path, timeout_s=10):
            path = str(path)
            if path not in self.files:
                return CommandResult(["cat", path], 1, "", "missing")
            return CommandResult(["cat", path], 0, self.files[path], "")

        def write_file(self, path, text, timeout_s=10):
            self.files[str(path)] = text
            return CommandResult(["write", str(path)], 0, "", "")

    files = {
        "/share/isaac_manipulator_pick_and_place/params/multi_object_pick_and_place_behavior_tree_params.yaml": yaml.safe_dump(
            {
                "behavior_tree_params": {
                    "multi_object_pick_and_place": {
                        "pose_estimation": {"camera_frame_id": "camera_1_color_optical_frame"},
                        "switch_controllers": {
                            "arm": {"controllers_to_activate": ["scaled_joint_trajectory_controller"]},
                            "tool": {"controllers_to_activate": ["robotiq_gripper_controller"]},
                        },
                    }
                }
            }
        ),
        "/share/isaac_manipulator_robot_description/config/ros2_control_controllers_sim.yaml": yaml.safe_dump(
            {
                "controller_manager": {
                    "ros__parameters": {
                        "joint_state_broadcaster": {
                            "type": "joint_state_broadcaster/JointStateBroadcaster"
                        },
                        "scaled_joint_trajectory_controller": {
                            "type": "joint_trajectory_controller/JointTrajectoryController"
                        },
                    }
                },
                "joint_state_broadcaster": {"ros__parameters": {}},
            }
        ),
        "/share/isaac_manipulator_robot_description/config/moveit_sim_controllers.yaml": yaml.safe_dump(
            {
                "trajectory_execution": {"allowed_start_tolerance": 0.1},
                "moveit_simple_controller_manager": {},
            }
        ),
    }
    runner = FileRunner(files)
    tool = IsaacRosConfigTool(runner=runner, output_dir=tmp_path)

    result = tool.prepare_pick_and_place_config()

    ros2_control_path = result.value["artifacts"]["ros2_control_config_path"]
    ros2_control_config = yaml.safe_load(runner.files[ros2_control_path])
    controller_params = ros2_control_config["controller_manager"]["ros__parameters"]
    assert "joint_state_broadcaster" not in controller_params
    assert "joint_state_broadcaster" not in ros2_control_config
    assert result.value["config"]["ros2_controllers_file_path"] == ros2_control_path
    moveit_controllers_path = result.value["artifacts"]["moveit_controllers_config_path"]
    moveit_controllers_config = yaml.safe_load(runner.files[moveit_controllers_path])
    assert moveit_controllers_config["trajectory_execution"]["allowed_start_tolerance"] == 1.0
    assert result.value["config"]["moveit_controllers_file_path"] == moveit_controllers_path


def test_pick_place_config_rejects_ground_truth_pose_for_multi_object_bt(tmp_path):
    runner = FakeRunner()
    output_path = tmp_path / "pick_place_config.yaml"
    tool = IsaacRosConfigTool(runner=runner, output_dir=tmp_path)

    result = tool.prepare_pick_and_place_config(
        output_path=str(output_path),
        use_ground_truth_pose_in_sim=True,
    )

    assert result.value["ok"] is True
    assert result.value["requested_use_ground_truth_pose_in_sim"] is True
    assert result.value["config"]["use_ground_truth_pose_in_sim"] == "false"
    assert "/get_objects" in result.value["warnings"][0]


def test_pick_place_config_rejects_rviz_for_headless_runner(tmp_path):
    runner = FakeRunner()
    output_path = tmp_path / "pick_place_config.yaml"
    tool = IsaacRosConfigTool(runner=runner, output_dir=tmp_path)

    result = tool.prepare_pick_and_place_config(
        output_path=str(output_path),
        enable_rviz_visualization=True,
    )

    assert result.value["ok"] is True
    assert result.value["requested_enable_rviz_visualization"] is True
    assert result.value["config"]["enable_rviz_visualization"] == "false"
    assert "RViz requires an interactive display" in result.value["warnings"][0]


def test_pick_place_config_rejects_nvblox_for_headless_tutorial(tmp_path):
    runner = FakeRunner()
    output_path = tmp_path / "pick_place_config.yaml"
    tool = IsaacRosConfigTool(runner=runner, output_dir=tmp_path)

    result = tool.prepare_pick_and_place_config(
        output_path=str(output_path),
        enable_nvblox=True,
    )

    assert result.value["ok"] is True
    assert result.value["requested_enable_nvblox"] is True
    assert result.value["config"]["enable_nvblox"] == "false"
    assert "ESDF data" in result.value["warnings"][0]


def test_bedrock_provider_redacts_tokens_from_error_text():
    text = "Unauthorized with token: sk-testsecret and Authorization: Bearer nvapi-secret"

    redacted = _redact_secrets(text)

    assert "sk-testsecret" not in redacted
    assert "nvapi-secret" not in redacted
    assert "[redacted]" in redacted


def test_nvidia_provider_uses_nvidia_api_key_and_inference_api(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["api_kwargs"] = kwargs
            return object()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(nvidia_openai_provider, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.delenv("LLM_GATEWAY_TOKEN", raising=False)
    provider = NvidiaOpenAIProvider()

    provider.create_client()
    asyncio.run(provider.call_api_async([{"role": "user", "content": "hi"}], [], max_completion_tokens=123))

    assert captured["client_kwargs"]["api_key"] == "nvapi-test"
    assert captured["client_kwargs"]["base_url"] == "https://inference-api.nvidia.com/v1"
    assert captured["api_kwargs"]["model"] == "openai/openai/gpt-5.5"
    assert captured["api_kwargs"]["max_tokens"] == 123
    assert "temperature" not in captured["api_kwargs"]
    assert "tools" not in captured["api_kwargs"]


def test_scene_readiness_reports_topics_that_exist_but_do_not_stream():
    rgb = "/front_stereo_camera/left/image_raw"
    depth = "/front_stereo_camera/depth/ground_truth"
    responses = {
        ("ros2", "node", "list"): CommandResult(["ros2", "node", "list"], 0, "", ""),
        ("ros2", "topic", "list"): CommandResult(
            ["ros2", "topic", "list"], 0, f"{rgb}\n{depth}\n", ""
        ),
        ("ros2", "service", "list"): CommandResult(["ros2", "service", "list"], 0, "", ""),
        ("ros2", "action", "list"): CommandResult(["ros2", "action", "list"], 0, "", ""),
        ("ros2", "topic", "info", rgb): CommandResult(["ros2", "topic", "info", rgb], 0, "Publisher count: 1\n", ""),
        ("ros2", "topic", "type", rgb): CommandResult(["ros2", "topic", "type", rgb], 0, "sensor_msgs/msg/Image\n", ""),
        ("ros2", "topic", "hz", rgb): CommandResult(["ros2", "topic", "hz", rgb], 124, "", "Timed out after 1s"),
        ("ros2", "topic", "info", depth): CommandResult(["ros2", "topic", "info", depth], 0, "Publisher count: 1\n", ""),
        ("ros2", "topic", "type", depth): CommandResult(["ros2", "topic", "type", depth], 0, "sensor_msgs/msg/Image\n", ""),
        ("ros2", "topic", "hz", depth): CommandResult(["ros2", "topic", "hz", depth], 124, "", "Timed out after 1s"),
    }
    runner = FakeRunner(responses)

    result = IsaacRosSceneTool(runner=runner).check_ready(sample_timeout_s=1)

    assert result.value["ok"] is False
    assert result.value["state"] == "scene_topics_not_streaming"
    assert f"{rgb}:messages" in result.value["missing"]
    assert result.value["topic_activity"][rgb]["state"] == "not_streaming"


def test_tool_schemas_are_model_callable():
    schemas = IsaacRosLaunchTool(runner=FakeRunner()).get_openai_schemas()
    names = {schema["function"]["name"] for schema in schemas}

    assert "isaac_ros_launch.start_recipe" in names
    assert "isaac_ros_launch.status_recipe" in names
    start_schema = next(s for s in schemas if s["function"]["name"] == "isaac_ros_launch.start_recipe")
    assert "recipe_name" in start_schema["function"]["parameters"]["required"]

    rtdetr_schema_names = {
        schema["function"]["name"]
        for schema in IsaacRtdetrTool(runner=FakeRunner()).get_openai_schemas()
    }
    assert "isaac_rtdetr.start" in rtdetr_schema_names
    assert "isaac_rtdetr.start_recipe" not in rtdetr_schema_names

    nvblox_schema_names = {
        schema["function"]["name"]
        for schema in IsaacNvbloxTool(runner=FakeRunner()).get_openai_schemas()
    }
    assert "isaac_nvblox.inspect_outputs" in nvblox_schema_names

    service_schema_names = {
        schema["function"]["name"]
        for schema in IsaacRosServiceTool(runner=FakeRunner()).get_openai_schemas()
    }
    assert "isaac_ros_service.inspect_service" in service_schema_names
    assert "isaac_ros_service.call_service" in service_schema_names

    param_schema_names = {
        schema["function"]["name"]
        for schema in IsaacRosParamTool(runner=FakeRunner()).get_openai_schemas()
    }
    assert "isaac_ros_param.list_params" in param_schema_names
    assert "isaac_ros_param.get_param" in param_schema_names
    assert "isaac_ros_param.set_param" in param_schema_names

    tf_schema_names = {
        schema["function"]["name"]
        for schema in IsaacRosTfTool(runner=FakeRunner()).get_openai_schemas()
    }
    assert "isaac_ros_tf.list_frames" in tf_schema_names
    assert "isaac_ros_tf.lookup_transform" in tf_schema_names

    object_info_schema_names = {
        schema["function"]["name"]
        for schema in IsaacObjectInfoTool(runner=FakeRunner()).get_openai_schemas()
    }
    assert "isaac_object_info.get_objects" in object_info_schema_names
    assert "isaac_object_info.detect_objects" in object_info_schema_names
    assert "isaac_object_info.clear_objects" in object_info_schema_names

    segmentation_schema_names = {
        schema["function"]["name"]
        for schema in IsaacSegmentationTool(runner=FakeRunner()).get_openai_schemas()
    }
    assert "isaac_segmentation.segment_anything_point" in segmentation_schema_names
    assert "isaac_segmentation.segment_anything_box" in segmentation_schema_names


def test_isaac_tools_are_registered_in_manifest_and_demo_config():
    expected_tools = {
        "isaac_ros_graph",
        "isaac_ros_launch",
        "isaac_ros_topic",
        "isaac_ros_action",
        "isaac_ros_config",
        "isaac_ros_scene",
        "isaac_perception",
        "isaac_pick_place",
        "isaac_gripper",
        "isaac_ros_service",
        "isaac_ros_param",
        "isaac_ros_tf",
        "isaac_object_info",
        "isaac_segmentation",
        "isaac_cumotion",
        "isaac_dope",
        "isaac_ess",
        "isaac_foundationpose",
        "isaac_foundationstereo",
        "isaac_grounding_dino",
        "isaac_nvblox",
        "isaac_object_following",
        "isaac_pose_to_pose",
        "isaac_realsense",
        "isaac_rtdetr",
        "isaac_segment_anything",
        "isaac_segment_anything2",
        "isaac_static_transforms",
    }

    assert expected_tools.issubset(TOOL_MANIFEST)
    for tool_name in expected_tools:
        module_name, class_name = TOOL_MANIFEST[tool_name].split(":", 1)
        tool_class = getattr(importlib.import_module(module_name), class_name)
        assert tool_class().get_name() == tool_name

    config_path = Path(__file__).resolve().parents[1] / "configs" / "isaac_ros_manipulation_pick_place.json"
    demo_config = json.loads(config_path.read_text())
    assert expected_tools.issubset(demo_config)
    assert demo_config["isaac_ros_topic"]["args"]["no_output_image"] is False

    strict_config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "isaac_ros_manipulation_pick_place_ros_strict.json"
    )
    strict_config = json.loads(strict_config_path.read_text())
    assert set(strict_config) == {
        "isaac_ros_graph",
        "isaac_ros_launch",
        "isaac_ros_topic",
        "isaac_ros_action",
        "isaac_ros_service",
        "isaac_ros_param",
        "isaac_ros_tf",
    }
    assert "isaac_pick_place" not in strict_config
    assert "isaac_ros_config" not in strict_config

    visual_config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "isaac_ros_manipulation_pick_place_ros_visual.json"
    )
    visual_config = json.loads(visual_config_path.read_text())
    assert set(strict_config).issubset(visual_config)
    assert {"visual_io", "vision_ops"}.issubset(visual_config)
