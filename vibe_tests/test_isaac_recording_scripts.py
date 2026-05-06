"""Regression coverage for Isaac pick-place recording scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_script_module(relative_path: str):
    path = ROOT_DIR / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_headless_recorder_rejects_sparse_videos_like_the_one_second_capture():
    module = load_script_module("scripts/isaac_ros/headless_isaac_sim.py")

    assert module.video_duration_s(6, 6.0) == 1.0
    error = module.recording_quality_error(elapsed_s=516.0, frames_written=6, fps=6.0)

    assert error is not None
    assert "Recording is too sparse" in error
    assert module.recording_quality_error(elapsed_s=10.0, frames_written=60, fps=6.0) is None


def test_headless_video_camera_vector_validation_is_strict():
    module = load_script_module("scripts/isaac_ros/headless_isaac_sim.py")

    assert module.parse_vec3("1.0, -2, 3.5", "--video-camera-eye") == (1.0, -2.0, 3.5)
    with pytest.raises(ValueError):
        module.parse_vec3("1,2", "--video-camera-eye")


def test_headless_status_serializes_usd_vector_like_values():
    module = load_script_module("scripts/isaac_ros/headless_isaac_sim.py")

    class Vec2Like:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return (640, 480)[index]

    assert module.stage_value(Vec2Like()) == [640, 480]


def test_direct_recorder_defaults_to_wide_overview_camera(monkeypatch):
    module = load_script_module("scripts/isaac_ros/headless_isaac_sim.py")
    monkeypatch.setattr(module.sys, "argv", ["headless_isaac_sim.py"])

    args = module.parse_args()

    assert args.video_camera_eye == "-1.2,-3.4,3.0"
    assert args.video_camera_target == "0.0,-0.15,0.35"
    assert args.physics_steps_per_second == 60
    assert args.disable_gpu_dynamics is True
    assert args.image_publisher_width == 640
    assert args.image_publisher_height == 480


def test_headless_stage_optimization_sets_physics_and_image_resolution():
    module = load_script_module("scripts/isaac_ros/headless_isaac_sim.py")

    class MissingAttr:
        def __bool__(self):
            return False

        def IsValid(self):
            return False

    class Attr:
        def __init__(self, value):
            self.value = value

        def __bool__(self):
            return True

        def IsValid(self):
            return True

        def Get(self):
            return self.value

        def Set(self, value):
            self.value = value

    class Prim:
        def __init__(self, path, type_name, attrs):
            self.path = path
            self.type_name = type_name
            self.attrs = attrs

        def __bool__(self):
            return True

        def IsValid(self):
            return True

        def GetPath(self):
            return self.path

        def GetTypeName(self):
            return self.type_name

        def GetAttribute(self, name):
            return self.attrs.get(name, MissingAttr())

    class Stage:
        def __init__(self, prims):
            self.prims = prims

        def GetPrimAtPath(self, path):
            return self.prims[path]

        def Traverse(self):
            return list(self.prims.values())

    physics = Prim(
        "/World/PhysicsScene",
        "PhysicsScene",
        {
            "physxScene:timeStepsPerSecond": Attr(120),
            "physxScene:enableGPUDynamics": Attr(True),
        },
    )
    render_product = Prim("/Render/Product", "RenderProduct", {"resolution": Attr((1900, 1200))})
    graph_node = Prim(
        "/ActionGraph/CreateRenderProduct",
        "OmniGraphNode",
        {
            "node:type": Attr("isaacsim.core.nodes.IsaacCreateRenderProduct"),
            "inputs:width": Attr(1900),
            "inputs:height": Attr(1200),
        },
    )
    stage = Stage({prim.path: prim for prim in [physics, render_product, graph_node]})

    result = module.configure_stage_for_pick_place(
        stage,
        physics_scene_path="/World/PhysicsScene",
        physics_steps_per_second=60,
        disable_gpu_dynamics=True,
        image_publisher_width=640,
        image_publisher_height=480,
    )

    assert physics.attrs["physxScene:timeStepsPerSecond"].Get() == 60
    assert physics.attrs["physxScene:enableGPUDynamics"].Get() is False
    assert render_product.attrs["resolution"].Get() == (640, 480)
    assert graph_node.attrs["inputs:width"].Get() == 640
    assert graph_node.attrs["inputs:height"].Get() == 480
    assert result["physics"]["updates"]
    assert result["image_publishers"]["updates"]


def test_start_script_routes_recording_to_isaac_sim_before_ros_topic_fallback():
    source = (ROOT_DIR / "scripts/isaac_ros/start_pick_place_demo.sh").read_text(encoding="utf-8")

    assert "ISAAC_SIM_VIDEO_ACTIVE" in source
    assert '"--video-output"' in source
    assert '"--video-status-json"' in source
    assert "Using direct Isaac Sim video recorder" in source
    assert 'if [[ "${ISAAC_SIM_VIDEO_ACTIVE}" == "1" ]]' in source
    assert 'ISAAC_VIDEO_CAMERA_EYE="${ISAAC_VIDEO_CAMERA_EYE:--1.2,-3.4,3.0}"' in source
    assert 'ISAAC_VIDEO_CAMERA_TARGET="${ISAAC_VIDEO_CAMERA_TARGET:-0.0,-0.15,0.35}"' in source
    assert '"--video-camera-eye=${ISAAC_VIDEO_CAMERA_EYE}"' in source
    assert 'ISAAC_PHYSICS_STEPS_PER_SECOND="${ISAAC_PHYSICS_STEPS_PER_SECOND:-60}"' in source
    assert 'ISAAC_IMAGE_PUBLISHER_WIDTH="${ISAAC_IMAGE_PUBLISHER_WIDTH:-640}"' in source
    assert 'ISAAC_IMAGE_PUBLISHER_HEIGHT="${ISAAC_IMAGE_PUBLISHER_HEIGHT:-480}"' in source
    assert "export ISAAC_IMAGE_PUBLISHER_WIDTH" in source
    assert "export ISAAC_IMAGE_PUBLISHER_HEIGHT" in source
    assert '"--physics-steps-per-second"' in source
    assert '"--image-publisher-width"' in source
    assert '"--disable-gpu-dynamics"' in source
    assert 'ISAAC_VIDEO_OVERLAY="${ISAAC_VIDEO_OVERLAY:-auto}"' in source
    assert "overlay_demo_trace_video.py" in source
    assert "overlay_video_trace_if_available" in source
    assert "docker cp" in source  # ROS topic recorder remains available as fallback.


def test_start_script_can_route_to_scripted_tutorial_without_agent():
    source = (ROOT_DIR / "scripts/isaac_ros/start_pick_place_demo.sh").read_text(encoding="utf-8")

    assert 'ISAAC_DEMO_RUN_MODE="${ISAAC_DEMO_RUN_MODE:-agentic}"' in source
    assert "examples/isaac_ros_pick_place_agentic_demo.py" in source
    assert "examples/isaac_ros_pick_place_scripted_demo.py" in source
    assert "SCRIPTED_STDOUT_LOG" in source


def test_scripted_tutorial_runner_matches_current_tool_contracts():
    source = (
        ROOT_DIR / "SpaceTools-Toolshed/examples/isaac_ros_pick_place_scripted_demo.py"
    ).read_text(encoding="utf-8")

    assert "router = start_toolkit" in source
    assert "--pre-action-settle-s" in source
    assert '"inspect_scene"' in source
    assert "render_rgb" not in source
    assert "render_depth" not in source


def test_ros_topic_recorder_marks_sparse_fallback_recordings_failed():
    source = (ROOT_DIR / "scripts/isaac_ros/record_ros_image_video.py").read_text(encoding="utf-8")

    assert "def recording_quality_error" in source
    assert "Recording is too sparse" in source
    assert "video_duration_s" in source
    assert "rc = 6" in source


def test_trace_overlay_script_builds_observable_tool_call_events(tmp_path):
    module = load_script_module("scripts/isaac_ros/overlay_demo_trace_video.py")
    summary_path = tmp_path / "direct_summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    summary = {
        "llm_api_called": False,
        "mode": "scripted_tutorial",
        "steps": [
            {
                "label": "prepare_config",
                "tool": "isaac_ros_config",
                "method": "prepare_pick_and_place_config",
                "text": "Wrote pick-and-place config overlay",
            },
            {
                "label": "graph_before_launch",
                "tool": "isaac_ros_graph",
                "method": "list_graph",
                "text": "ROS graph queried successfully",
            },
        ],
        "action": {
            "terminal_status": "SUCCEEDED",
            "parsed_result": {"workflow_status": 2},
            "parsed_feedback": [{"message": "Object 1 (class: 3) completed successfully"}],
        },
    }

    info = module.VideoInfo(width=960, height=600, duration_s=120.0)
    events = module.build_overlay_events(summary, summary_path, info)
    ass_path = tmp_path / "overlay.ass"
    module.write_ass(ass_path, info, events)
    ass_text = ass_path.read_text(encoding="utf-8")

    assert any("isaac_ros_config.prepare_pick_and_place_config" in event.text for event in events)
    assert any("isaac_pick_place.send_single_bin_goal" in event.text for event in events)
    assert "no LLM API" in ass_text
    assert "workflow_status=2" in ass_text
    assert "/NObject" not in ass_text


def test_trace_overlay_script_builds_agentic_tool_call_events(tmp_path):
    module = load_script_module("scripts/isaac_ros/overlay_demo_trace_video.py")
    summary_path = tmp_path / "direct_summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    summary = {
        "mode": "agentic_low_level",
        "llm_api_called": True,
        "response": "The ROS action completed successfully.",
        "steps": [
            {
                "type": "tool_decision",
                "tool_calls": [
                    {
                        "name": "isaac_ros_graph__wait_for",
                        "arguments": {"required_actions": "[\"/multi_object_pick_and_place\"]"},
                    }
                ],
            },
            {
                "type": "tool_result",
                "tool_name": "isaac_ros_action__send_goal",
                "result": "Action goal finished with status SUCCEEDED",
            },
        ],
    }

    info = module.VideoInfo(width=960, height=600, duration_s=120.0)
    events = module.build_overlay_events(summary, summary_path, info)
    ass_path = tmp_path / "overlay.ass"
    module.write_ass(ass_path, info, events)
    ass_text = ass_path.read_text(encoding="utf-8")

    assert "AGENTIC ISAAC ROS PICK/PLACE" in ass_text
    assert "isaac_ros_graph.wait_for" in ass_text
    assert "isaac_ros_action.send_goal" in ass_text
    assert "no LLM API" not in ass_text


def test_trace_overlay_ffmpeg_call_uses_absolute_video_paths(monkeypatch, tmp_path):
    module = load_script_module("scripts/isaac_ros/overlay_demo_trace_video.py")
    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, cwd, check):
        calls.append({"command": command, "cwd": cwd, "check": check})

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module.burn_overlay(Path("input.mp4"), Path("trace/overlay.ass"), Path("out/overlay.mp4"))

    command = calls[0]["command"]
    assert command[command.index("-i") + 1] == str((tmp_path / "input.mp4").resolve())
    assert command[-1] == str((tmp_path / "out/overlay.mp4").resolve())
    assert calls[0]["cwd"] == str((tmp_path / "trace").resolve())
