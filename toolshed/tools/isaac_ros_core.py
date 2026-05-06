"""Shared helpers for Isaac ROS manipulation Toolshed tools."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shlex
import signal
import shutil
import subprocess
import time
from typing import Any, Mapping


DEFAULT_RMW_IMPLEMENTATION = "rmw_cyclonedds_cpp"
DEFAULT_ACTION_TYPE = "isaac_ros_manipulation_interfaces/action/MultiObjectPickAndPlace"
DEFAULT_OUTPUT_DIR = Path("outputs/isaac_ros")

PACKAGE_ALIASES = {
    "isaac_ros_manipulation_bringup": "isaac_manipulator_bringup",
    "isaac_ros_manipulation_pick_and_place": "isaac_manipulator_pick_and_place",
}
ACTION_TYPE_ALIASES = {
    "isaac_ros_manipulation_interfaces/action/MultiObjectPickAndPlace": (
        "isaac_manipulator_interfaces/action/MultiObjectPickAndPlace"
    ),
}
for _interface_kind, _names in {
    "action": [
        "AddSegmentationMask",
        "DetectObjects",
        "EstimatePoseDope",
        "EstimatePoseFoundationPose",
        "GetObjectPose",
        "GetObjects",
        "GetSelectedObject",
        "MultiObjectPickAndPlace",
        "PickAndPlace",
        "SegmentAnything",
    ],
    "srv": [
        "AddMeshToObject",
        "AssignNameToObject",
        "ClearObjects",
    ],
}.items():
    for _name in _names:
        ACTION_TYPE_ALIASES[
            f"isaac_ros_manipulation_interfaces/{_interface_kind}/{_name}"
        ] = f"isaac_manipulator_interfaces/{_interface_kind}/{_name}"


@dataclass(frozen=True)
class CommandResult:
    """Result of a shell command executed by a runner."""

    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class ProcessState:
    """Serializable state for a long-running process."""

    name: str
    command: list[str]
    running: bool
    returncode: int | None
    log_path: str | None


@dataclass(frozen=True)
class LaunchRecipe:
    """A named Isaac ROS launch recipe plus observability metadata."""

    name: str
    package: str
    launch_file: str
    summary: str
    default_args: dict[str, Any] = field(default_factory=dict)
    required_topics: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)
    required_services: list[str] = field(default_factory=list)
    produced_topics: list[str] = field(default_factory=list)
    produced_actions: list[str] = field(default_factory=list)
    produced_services: list[str] = field(default_factory=list)
    event_driven_topics: list[str] = field(default_factory=list)
    frames: list[str] = field(default_factory=list)
    render_topics: list[str] = field(default_factory=list)
    cleanup_patterns: list[str] = field(default_factory=list)

    def command(self, overrides: Mapping[str, Any] | None = None) -> list[str]:
        launch_args = dict(self.default_args)
        launch_args.update(overrides or {})
        command = ["ros2", "launch", self.package, self.launch_file]
        command.extend(f"{key}:={value}" for key, value in launch_args.items())
        return command

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "package": self.package,
            "launch_file": self.launch_file,
            "summary": self.summary,
            "default_args": self.default_args,
            "required_topics": self.required_topics,
            "required_actions": self.required_actions,
            "required_services": self.required_services,
            "produced_topics": self.produced_topics,
            "produced_actions": self.produced_actions,
            "produced_services": self.produced_services,
            "event_driven_topics": self.event_driven_topics,
            "frames": self.frames,
            "render_topics": self.render_topics,
            "cleanup_patterns": self.cleanup_patterns,
        }


def _include_recipe(
    name: str,
    summary: str,
    *,
    produced_topics: list[str] | None = None,
    produced_actions: list[str] | None = None,
    produced_services: list[str] | None = None,
    required_topics: list[str] | None = None,
    required_actions: list[str] | None = None,
    required_services: list[str] | None = None,
    frames: list[str] | None = None,
    render_topics: list[str] | None = None,
    cleanup_patterns: list[str] | None = None,
    default_args: dict[str, Any] | None = None,
) -> LaunchRecipe:
    return LaunchRecipe(
        name=name,
        package="isaac_ros_manipulation_bringup",
        launch_file=f"launch/include/{name}.launch.py",
        summary=summary,
        default_args=default_args or {},
        required_topics=required_topics or [],
        required_actions=required_actions or [],
        required_services=required_services or [],
        produced_topics=produced_topics or [],
        produced_actions=produced_actions or [],
        produced_services=produced_services or [],
        frames=frames or [],
        render_topics=render_topics or [],
        cleanup_patterns=cleanup_patterns or [],
    )


RECIPES: dict[str, LaunchRecipe] = {
    "cumotion": _include_recipe(
        "cumotion",
        "cuMotion planner, robot segmenter, and optional object attachment.",
        required_topics=["/isaac_joint_states"],
        produced_actions=["/cumotion/motion_plan", "/attach_object"],
        produced_topics=["/cumotion/camera_1/robot_mask", "/cumotion/camera_1/world_depth"],
        render_topics=["/cumotion/camera_1/world_depth"],
    ),
    "dope": _include_recipe(
        "dope",
        "DOPE TensorRT pose-estimation graph.",
        required_topics=["/camera_1/color/image_raw", "/camera_1/color/camera_info"],
        produced_topics=["/pose_estimation/output"],
        render_topics=["/camera_1/color/image_raw"],
    ),
    "ess": _include_recipe(
        "ess",
        "ESS stereo depth estimation graph.",
        required_topics=["/left/image_rect", "/right/image_rect"],
        produced_topics=["/depth_image"],
        render_topics=["/depth_image"],
    ),
    "foundationpose": _include_recipe(
        "foundationpose",
        "FoundationPose pose-estimation graph fed by RGB, depth, camera info, and detections.",
        required_topics=["/foundation_pose_server/image", "/foundation_pose_server/depth"],
        produced_topics=["/pose_estimation/output", "/foundation_pose_server/pose_estimation/output"],
        render_topics=["/foundation_pose_server/image", "/foundation_pose_server/depth"],
    ),
    "foundationstereo": _include_recipe(
        "foundationstereo",
        "FoundationStereo depth-estimation graph.",
        required_topics=["/left/image_rect", "/right/image_rect"],
        produced_topics=["/depth_image"],
        render_topics=["/depth_image"],
    ),
    "grounding_dino": _include_recipe(
        "grounding_dino",
        "GroundingDINO detection and mask-generation graph.",
        required_topics=["/object_detection_server/image_rect"],
        produced_topics=["/detections", "/detections_output", "/segmentation"],
        render_topics=["/object_detection_server/image_rect", "/segmentation"],
    ),
    "nvblox": _include_recipe(
        "nvblox",
        "nvblox ESDF reconstruction graph for collision-aware planning.",
        required_topics=["/isaac_joint_states"],
        produced_topics=["/nvblox_node/esdf", "/nvblox_node/mesh"],
        render_topics=["/front_stereo_camera/depth/ground_truth"],
    ),
    "object_following": _include_recipe(
        "object_following",
        "Object-following orchestration node.",
        required_actions=["/cumotion/motion_plan"],
        produced_topics=["/goal_frame"],
    ),
    "pose_to_pose": _include_recipe(
        "pose_to_pose",
        "Pose-to-pose orchestration node.",
        required_actions=["/cumotion/motion_plan"],
        produced_topics=["/goal_frame"],
    ),
    "realsense": _include_recipe(
        "realsense",
        "RealSense camera driver graph.",
        produced_topics=[
            "/camera_1/color/image_raw",
            "/camera_1/color/camera_info",
            "/camera_1/aligned_depth_to_color/image_raw",
        ],
        render_topics=["/camera_1/color/image_raw", "/camera_1/aligned_depth_to_color/image_raw"],
    ),
    "rtdetr": _include_recipe(
        "rtdetr",
        "RT-DETR object detection and mask-generation graph.",
        required_topics=["/object_detection_server/image_rect"],
        produced_topics=["/detections", "/detections_output", "/segmentation"],
        render_topics=["/object_detection_server/image_rect", "/segmentation"],
    ),
    "segment_anything": _include_recipe(
        "segment_anything",
        "SAM point/detection-triggered segmentation graph.",
        required_topics=["/segment_anything_server/image"],
        produced_topics=["/segment_anything/binary_segmentation_mask"],
        render_topics=["/segment_anything/binary_segmentation_mask"],
    ),
    "segment_anything2": _include_recipe(
        "segment_anything2",
        "SAM2 point-triggered segmentation graph.",
        required_topics=["/segment_anything_server/image"],
        produced_topics=["/segment_anything2/binary_segmentation_mask"],
        render_topics=["/segment_anything2/binary_segmentation_mask"],
    ),
    "static_transforms": _include_recipe(
        "static_transforms",
        "Static TF publishers for cameras, targets, object grasp frames, and setup calibration.",
        frames=["world", "base_link", "goal_frame", "target1_frame", "target2_frame"],
    ),
    "pick_and_place_workflow": LaunchRecipe(
        name="pick_and_place_workflow",
        package="isaac_ros_manipulation_bringup",
        launch_file="workflows.launch.py",
        summary="Top-level Isaac ROS manipulation pick-and-place workflow.",
        default_args={"manipulator_workflow_config": "sim_launch_params.yaml"},
        required_topics=["/front_stereo_camera/left/image_raw", "/isaac_joint_states"],
        produced_actions=[
            "/multi_object_pick_and_place",
            "/get_objects",
            "/get_object_pose",
            "/cumotion/motion_plan",
        ],
        produced_topics=["/detections", "/pose_estimation/output"],
        event_driven_topics=["/detections", "/pose_estimation/output"],
        render_topics=["/front_stereo_camera/left/image_raw", "/front_stereo_camera/depth/ground_truth"],
        cleanup_patterns=[
            "workflows.launch.py manipulator_workflow_config:=",
            "isaac_manipulator_pick_and_place/multi_object_pick_and_place",
            "ros2 run isaac_manipulator_pick_and_place multi_object_pick_and_place",
            "cumotion_goal_set_planner_node",
            "isaac_sim_gripper_driver.py",
        ],
    ),
}

RECIPE_ALIASES: dict[str, str] = {
    "pick_place": "pick_and_place_workflow",
    "pick_and_place": "pick_and_place_workflow",
    "pick_place_workflow": "pick_and_place_workflow",
}


def parse_overrides(overrides: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Parse JSON or comma-separated launch-argument overrides."""
    if overrides is None or overrides == "":
        return {}
    if isinstance(overrides, Mapping):
        return dict(overrides)

    text = str(overrides).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        return dict(parsed)

    result: dict[str, Any] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(
                "Overrides must be JSON object text or comma-separated key=value pairs"
            )
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def parse_string_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Parse a JSON array or comma-separated string into a list of strings."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_pose(value: str | list[float] | tuple[float, ...]) -> list[float]:
    """Parse a seven-value pose [x, y, z, qx, qy, qz, qw]."""
    if isinstance(value, str):
        value = json.loads(value)
    pose = [float(item) for item in value]
    if len(pose) != 7:
        raise ValueError("Pose must contain exactly 7 values: [x, y, z, qx, qy, qz, qw]")
    return pose


def tail_lines(text: str, max_lines: int = 20) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return [""]
    return lines[-max_lines:]


def recent_logs_from_results(results: list[CommandResult], max_lines: int = 20) -> list[str]:
    text = "\n".join(part for result in results for part in (result.stderr, result.stdout) if part)
    return tail_lines(text, max_lines=max_lines)


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


class RosCommandRunner:
    """Runs ROS 2 commands locally or inside an Isaac ROS Docker container."""

    def __init__(
        self,
        *,
        container_name: str | None = None,
        ros_domain_id: str | int | None = None,
        rmw_implementation: str = DEFAULT_RMW_IMPLEMENTATION,
        isaac_ros_ws: str | None = None,
        workdir: str | None = None,
    ) -> None:
        self.container_name = container_name or os.getenv("ISAAC_ROS_CONTAINER")
        self.ros_domain_id = str(ros_domain_id if ros_domain_id is not None else os.getenv("ROS_DOMAIN_ID", "0"))
        self.rmw_implementation = rmw_implementation or os.getenv(
            "RMW_IMPLEMENTATION", DEFAULT_RMW_IMPLEMENTATION
        )
        self.isaac_ros_ws = isaac_ros_ws or os.getenv("ISAAC_ROS_WS")
        self.isaac_ros_setup = os.getenv("ISAAC_ROS_SETUP")
        self.workdir = workdir or self.isaac_ros_ws
        self._processes: dict[str, subprocess.Popen] = {}
        self._log_paths: dict[str, Path] = {}
        self._remote_pid_files: dict[str, str] = {}
        self._package_cache: dict[str, str] = {}
        self._interface_cache: dict[str, str] = {}
        self._pythonpath_prepend: list[str] = []

    @staticmethod
    def remote_pid_file_for(name: str) -> str:
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
        return f"/tmp/toolshed_{safe_name}.pid"

    def wrap_ros_command(self, args: list[str]) -> list[str]:
        if not self.container_name:
            return [str(arg) for arg in args]
        return self._container_command(args)

    def _container_command(self, args: list[str], pid_file: str | None = None) -> list[str]:
        exports = [
            f"export ROS_DOMAIN_ID={shlex.quote(str(self.ros_domain_id))}",
            f"export RMW_IMPLEMENTATION={shlex.quote(self.rmw_implementation)}",
        ]
        if self.isaac_ros_ws:
            exports.append(f"export ISAAC_ROS_WS={shlex.quote(self.isaac_ros_ws)}")
        exports.append(self._setup_command())
        if self._pythonpath_prepend:
            pythonpath = ":".join(self._pythonpath_prepend)
            exports.append(f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}")
        if self.workdir:
            exports.append(f"cd {shlex.quote(self.workdir)}")
        command_text = shlex.join([str(arg) for arg in args])
        if pid_file:
            command_text = f"echo $$ > {shlex.quote(pid_file)}; exec {command_text}"
        shell_command = "; ".join([*exports, command_text])
        return ["docker", "exec", "-i", self.container_name, "bash", "-lc", shell_command]

    def _setup_command(self) -> str:
        if self.isaac_ros_setup:
            return f"source {shlex.quote(self.isaac_ros_setup)}"
        if self.isaac_ros_ws:
            setup_path = shlex.quote(str(Path(self.isaac_ros_ws) / "install" / "setup.bash"))
            return (
                f"if [ -f {setup_path} ]; then source {setup_path}; "
                "elif [ -f /opt/ros/jazzy/setup.bash ]; then source /opt/ros/jazzy/setup.bash; "
                "elif [ -f /opt/ros/humble/setup.bash ]; then source /opt/ros/humble/setup.bash; fi"
            )
        return (
            "if [ -f /workspaces/isaac_ros-dev/install/setup.bash ]; then "
            "source /workspaces/isaac_ros-dev/install/setup.bash; "
            "elif [ -f /opt/ros/jazzy/setup.bash ]; then source /opt/ros/jazzy/setup.bash; "
            "elif [ -f /opt/ros/humble/setup.bash ]; then source /opt/ros/humble/setup.bash; fi"
        )

    def run(self, args: list[str], timeout_s: float = 30.0) -> CommandResult:
        raw_args = [str(arg) for arg in args]
        wrapped_with_timeout = False
        if self.container_name and (not raw_args or raw_args[0] != "timeout"):
            command = self._container_command(["timeout", str(float(timeout_s)), *raw_args])
            subprocess_timeout = timeout_s + 10
            wrapped_with_timeout = True
        else:
            command = self.wrap_ros_command(raw_args)
            subprocess_timeout = timeout_s
        try:
            proc = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=subprocess_timeout,
            )
            stderr = proc.stderr
            if wrapped_with_timeout and proc.returncode == 124 and "Timed out after" not in stderr:
                stderr = (stderr + f"\nTimed out after {timeout_s}s").lstrip()
            return CommandResult(command, proc.returncode, proc.stdout, stderr)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command,
                124,
                _timeout_text(exc.stdout),
                _timeout_text(exc.stderr) + f"\nTimed out after {timeout_s}s",
            )

    def resolve_package(self, package_name: str) -> str:
        """Resolve current Isaac ROS package names to installed release aliases."""
        if package_name in self._package_cache:
            return self._package_cache[package_name]

        if self.run(["ros2", "pkg", "prefix", package_name], timeout_s=10).ok:
            self._package_cache[package_name] = package_name
            return package_name

        alias = PACKAGE_ALIASES.get(package_name)
        if alias and self.run(["ros2", "pkg", "prefix", alias], timeout_s=10).ok:
            self._package_cache[package_name] = alias
            return alias

        self._package_cache[package_name] = package_name
        return package_name

    def package_share_path(self, package_name: str) -> str | None:
        resolved_package = self.resolve_package(package_name)
        result = self.run(["ros2", "pkg", "prefix", "--share", resolved_package], timeout_s=10)
        if not result.ok:
            return None
        return result.stdout.strip() or None

    def path_exists(self, path: str | Path) -> bool:
        if self.container_name:
            return self.run(["test", "-e", str(path)], timeout_s=5).ok
        return Path(path).exists()

    def read_file(self, path: str | Path, timeout_s: float = 10.0) -> CommandResult:
        path = str(path)
        if self.container_name:
            return self.run(["cat", path], timeout_s=timeout_s)
        source = Path(path)
        if not source.exists():
            return CommandResult(["cat", path], 1, "", f"Source file does not exist: {source}")
        return CommandResult(["cat", path], 0, source.read_text(errors="replace"), "")

    def write_file(self, path: str | Path, text: str, timeout_s: float = 10.0) -> CommandResult:
        path = str(path)
        if self.container_name:
            command = [
                "docker",
                "exec",
                "-i",
                self.container_name,
                "bash",
                "-lc",
                f"mkdir -p {shlex.quote(str(Path(path).parent))}; cat > {shlex.quote(path)}",
            ]
            try:
                proc = subprocess.run(
                    command,
                    input=text,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                return CommandResult(command, proc.returncode, proc.stdout, proc.stderr)
            except subprocess.TimeoutExpired as exc:
                return CommandResult(
                    command,
                    124,
                    _timeout_text(exc.stdout),
                    _timeout_text(exc.stderr) + f"\nTimed out after {timeout_s}s",
                )

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        return CommandResult(["write", path], 0, "", "")

    def resolve_interface_type(self, interface_type: str) -> str:
        """Resolve current Isaac ROS interface names to installed release aliases."""
        if interface_type in self._interface_cache:
            return self._interface_cache[interface_type]

        if self.run(["ros2", "interface", "show", interface_type], timeout_s=10).ok:
            self._interface_cache[interface_type] = interface_type
            return interface_type

        alias = ACTION_TYPE_ALIASES.get(interface_type)
        if alias and self.run(["ros2", "interface", "show", alias], timeout_s=10).ok:
            self._interface_cache[interface_type] = alias
            return alias

        self._interface_cache[interface_type] = interface_type
        return interface_type

    def resolve_action_type(self, action_type: str) -> str:
        """Resolve current Isaac ROS action names to installed release aliases."""
        return self.resolve_interface_type(action_type)

    def ensure_moveit_compatibility(self) -> CommandResult:
        """Create patch-version symlinks needed by some Isaac ROS binary releases."""
        if not self.container_name:
            return CommandResult(["ensure_moveit_compatibility"], 0, "", "Not running in a container")

        script = r"""
set -e
source_file=""
if [ -f /workspaces/isaac_ros-dev/install/setup.bash ]; then
  . /workspaces/isaac_ros-dev/install/setup.bash
elif [ -f /opt/ros/jazzy/setup.bash ]; then
  . /opt/ros/jazzy/setup.bash
elif [ -f /opt/ros/humble/setup.bash ]; then
  . /opt/ros/humble/setup.bash
fi
ros_lib="/opt/ros/${ROS_DISTRO:-jazzy}/lib"
if [ ! -d "$ros_lib" ]; then
  exit 0
fi
missing="$(LD_LIBRARY_PATH="$ros_lib:${LD_LIBRARY_PATH:-}" ldd "$ros_lib/libisaac_ros_cumotion_moveit.so" 2>/dev/null | awk '/libmoveit_.*\.so\.2\.12\.3.*not found/ {print $1}' || true)"
for lib in $missing; do
  compat="$ros_lib/$lib"
  target="${compat%.2.12.3}.2.12.4"
  if [ -e "$target" ] && [ ! -e "$compat" ]; then
    ln -s "$(basename "$target")" "$compat"
    echo "linked $compat -> $(basename "$target")"
  fi
done
LD_LIBRARY_PATH="$ros_lib:${LD_LIBRARY_PATH:-}" ldd "$ros_lib/libisaac_ros_cumotion_moveit.so" 2>/dev/null | grep 'not found' || true
"""
        command = ["docker", "exec", "-i", self.container_name, "bash", "-lc", script]
        try:
            proc = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
            return CommandResult(command, proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command,
                124,
                _timeout_text(exc.stdout),
                _timeout_text(exc.stderr) + "\nTimed out after 30s",
            )

    def ensure_isaac_sim_camera_resolution_compatibility(self) -> CommandResult:
        """Install an import overlay so Isaac Sim launch constants match headless camera size."""
        width = int(os.getenv("ISAAC_IMAGE_PUBLISHER_WIDTH", "1920"))
        height = int(os.getenv("ISAAC_IMAGE_PUBLISHER_HEIGHT", "1200"))
        overlay_root = "/tmp/toolshed_isaac_ros_manipulator_overlay"
        if overlay_root not in self._pythonpath_prepend:
            self._pythonpath_prepend.insert(0, overlay_root)

        if not self.container_name:
            return CommandResult(
                ["ensure_isaac_sim_camera_resolution_compatibility"],
                0,
                f"HAWK_IMAGE_WIDTH={width}\nHAWK_IMAGE_HEIGHT={height}\noverlay={overlay_root}",
                "Not running in a container",
            )

        script = f"""
set -e
python3 - <<'PY'
from pathlib import Path
import re

width = {width}
height = {height}
overlay_root = Path("/tmp/toolshed_isaac_ros_manipulator_overlay")
package_dir = overlay_root / "isaac_manipulator_ros_python_utils"
package_dir.mkdir(parents=True, exist_ok=True)

candidates = sorted(
    list(Path("/workspaces/isaac_ros-dev").glob(
        "*/isaac_manipulator_ros_python_utils/isaac_manipulator_ros_python_utils/constants.py"
    ))
    + list(Path("/opt/ros").glob(
        "*/lib/python*/site-packages/isaac_manipulator_ros_python_utils/constants.py"
    ))
)
if not candidates:
    raise SystemExit("could not find installed isaac_manipulator_ros_python_utils constants")
source = candidates[-1]
init_source = source.parent / "__init__.py"
if not init_source.exists():
    raise SystemExit("could not find installed isaac_manipulator_ros_python_utils __init__.py")
init_text = init_source.read_text()
(package_dir / "__init__.py").write_text(
    "from pkgutil import extend_path\\n"
    "__path__ = extend_path(__path__, __name__)\\n"
    + init_text
)
text = source.read_text()
text, width_count = re.subn(r"^HAWK_IMAGE_WIDTH\\s*=\\s*\\d+\\s*$", f"HAWK_IMAGE_WIDTH = {{width}}", text, count=1, flags=re.MULTILINE)
text, height_count = re.subn(r"^HAWK_IMAGE_HEIGHT\\s*=\\s*\\d+\\s*$", f"HAWK_IMAGE_HEIGHT = {{height}}", text, count=1, flags=re.MULTILINE)
if width_count != 1 or height_count != 1:
    raise SystemExit("HAWK image constant patch anchor not found")
target = package_dir / "constants.py"
target.write_text(text)
print(f"overlay={{overlay_root}}")
print(f"source={{source}}")
print(f"target={{target}}")
print(f"HAWK_IMAGE_WIDTH={{width}}")
print(f"HAWK_IMAGE_HEIGHT={{height}}")
PY
python3 -m py_compile /tmp/toolshed_isaac_ros_manipulator_overlay/isaac_manipulator_ros_python_utils/constants.py
"""
        command = ["docker", "exec", "-i", self.container_name, "bash", "-lc", script]
        try:
            proc = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
            return CommandResult(command, proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command,
                124,
                _timeout_text(exc.stdout),
                _timeout_text(exc.stderr) + "\nTimed out after 30s",
            )

    def ensure_cumotion_goalset_compatibility(self) -> CommandResult:
        """Install an import overlay for cuMotion goal-set planner compatibility fixes."""
        overlay_root = "/tmp/toolshed_isaac_ros_cumotion_overlay"
        if overlay_root not in self._pythonpath_prepend:
            self._pythonpath_prepend.insert(0, overlay_root)

        if not self.container_name:
            return CommandResult(
                ["ensure_cumotion_goalset_compatibility"],
                0,
                overlay_root,
                "Not running in a container",
            )

        script = r"""
set -e
python3 - <<'PY'
from pathlib import Path

overlay_root = Path("/tmp/toolshed_isaac_ros_cumotion_overlay")
package_dir = overlay_root / "isaac_ros_cumotion"
package_dir.mkdir(parents=True, exist_ok=True)
(package_dir / "__init__.py").write_text(
    "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
)

candidates = sorted(
    Path("/opt/ros").glob("*/lib/python*/site-packages/isaac_ros_cumotion/cumotion_goal_set_planner.py")
)
if not candidates:
    raise SystemExit("could not find installed isaac_ros_cumotion goal-set planner")
source = candidates[-1]
text = source.read_text()

old_threading_import = "from typing import List\n"
new_threading_import = "from typing import List\nimport threading\n"
if old_threading_import not in text:
    raise SystemExit("threading import patch anchor not found")
text = text.replace(old_threading_import, new_threading_import, 1)

old_rclpy_import = "import rclpy\nfrom rclpy.action import ActionServer\n"
new_rclpy_import = (
    "import rclpy\n"
    "from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy\n"
    "from rclpy.action import ActionServer\n"
)
if old_rclpy_import not in text:
    raise SystemExit("rclpy goal-event import patch anchor not found")
text = text.replace(old_rclpy_import, new_rclpy_import, 1)

old_lock_init = (
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self._goal_set_planner_server = ActionServer(\n"
)
new_lock_init = (
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self._toolshed_motion_plan_lock = threading.Lock()\n"
    "        self._goal_set_planner_server = ActionServer(\n"
)
if old_lock_init not in text:
    raise SystemExit("planner lock init patch anchor not found")
text = text.replace(old_lock_init, new_lock_init, 1)

old_action_server_timeout = (
    "        self._goal_set_planner_server = ActionServer(\n"
    "            self, MotionPlan, 'cumotion/motion_plan', self.motion_plan_execute_callback\n"
    "        )\n"
)
new_action_server_timeout = (
    "        self._goal_set_planner_server = ActionServer(\n"
    "            self,\n"
    "            MotionPlan,\n"
    "            'cumotion/motion_plan',\n"
    "            self.motion_plan_execute_callback,\n"
    "            result_timeout=2147483647,\n"
    "        )\n"
)
if old_action_server_timeout not in text:
    raise SystemExit("action result timeout patch anchor not found")
text = text.replace(old_action_server_timeout, new_action_server_timeout, 1)

old_callback_def = "    def motion_plan_execute_callback(self, goal_handle):\n"
new_callback_def = (
    "    def motion_plan_execute_callback(self, goal_handle):\n"
    "        with self._toolshed_motion_plan_lock:\n"
    "            return self._toolshed_motion_plan_execute_callback(goal_handle)\n"
    "\n"
    "    def _toolshed_mark_goal_succeeded(self, goal_handle):\n"
    "        try:\n"
    "            if getattr(goal_handle, 'is_active', False):\n"
    "                goal_handle._update_state(_rclpy.GoalEvent.SUCCEED)\n"
    "        except Exception as exc:\n"
    "            self.get_logger().warning(\n"
    "                f'Toolshed compatibility: could not mark motion plan goal succeeded: {exc}'\n"
    "            )\n"
    "\n"
    "    def _toolshed_motion_plan_execute_callback(self, goal_handle):\n"
)
if old_callback_def not in text:
    raise SystemExit("planner lock callback patch anchor not found")
text = text.replace(old_callback_def, new_callback_def, 1)

old_early_succeed = (
    "        goal_handle.succeed()\n"
    "        self.motion_gen.reset(reset_seed=False)\n"
)
new_early_succeed = (
    "        # Toolshed compatibility: report action completion only after the\n"
    "        # MotionPlan.Result has been populated. Completing the goal here lets\n"
    "        # clients observe the default false result before planning runs.\n"
    "        self.motion_gen.reset(reset_seed=False)\n"
)
if old_early_succeed not in text:
    raise SystemExit("early action succeed patch anchor not found")
text = text.replace(old_early_succeed, new_early_succeed, 1)

old_clear = "            self._CumotionActionServer__js_buffer = None\n"
new_clear = (
    "            # Toolshed compatibility: keep the latest joint-state sample available for\n"
    "            # rapid behavior-tree retries instead of failing between Isaac joint ticks.\n"
)
if old_clear not in text:
    raise SystemExit("joint-state buffer patch anchor not found")
text = text.replace(old_clear, new_clear, 1)

old_graph_attempt = (
    "                MotionGenPlanConfig(\n"
    "                    max_attempts=self._CumotionActionServer__max_attempts,\n"
    "                    enable_graph_attempt=1,\n"
    "                    time_dilation_factor=time_dilation_factor,\n"
    "                ),\n"
    "                grasp_approach_offset="
)
new_graph_attempt = (
    "                MotionGenPlanConfig(\n"
    "                    max_attempts=self._CumotionActionServer__max_attempts,\n"
    "                    enable_graph_attempt=0,\n"
    "                    time_dilation_factor=time_dilation_factor,\n"
    "                ),\n"
    "                grasp_approach_offset="
)
if old_graph_attempt not in text:
    raise SystemExit("plan-grasp graph-attempt patch anchor not found")
text = text.replace(old_graph_attempt, new_graph_attempt, 1)

old_goalset_read = (
    "            success, error_code, poses = self.get_goal_poses(plan_req)\n"
    "            self.get_logger().info(f'Success, Error Code): {success}, {error_code}!')\n"
)
new_goalset_read = (
    "            success, error_code, poses = self.get_goal_poses(plan_req)\n"
    "            original_goalset_count = int(getattr(poses, 'n_goalset', 1) or 1)\n"
    "            if success and 0 < original_goalset_count < 100:\n"
    "                repeat_count = (100 + original_goalset_count - 1) // original_goalset_count\n"
    "                poses = Pose(\n"
    "                    position=poses.position.repeat(1, repeat_count, 1)[:, :100, :].contiguous(),\n"
    "                    quaternion=poses.quaternion.repeat(1, repeat_count, 1)[:, :100, :].contiguous(),\n"
    "                    normalize_rotation=False,\n"
    "                )\n"
    "                self.get_logger().info(\n"
    "                    f'Toolshed compatibility: padded grasp goalset from '\n"
    "                    f'{original_goalset_count} to {poses.n_goalset}'\n"
    "                )\n"
    "            self.get_logger().info(f'Success, Error Code): {success}, {error_code}!')\n"
)
if old_goalset_read not in text:
    raise SystemExit("goalset padding patch anchor not found")
text = text.replace(old_goalset_read, new_goalset_read, 1)

old_goal_index = "                result.goal_index = grasp_plan_result.goalset_index.item()\n"
new_goal_index = (
    "                result.goal_index = grasp_plan_result.goalset_index.item()\n"
    "                if original_goalset_count > 0:\n"
    "                    result.goal_index = result.goal_index % original_goalset_count\n"
)
if old_goal_index not in text:
    raise SystemExit("goal-index remap patch anchor not found")
text = text.replace(old_goal_index, new_goal_index, 1)

old_grasp_failure = (
    "            else:\n"
    "                result.success = False\n"
    "                result.message = grasp_plan_result.status\n"
)
new_grasp_failure = (
    "            else:\n"
    "                result.success = False\n"
    "                result.message = grasp_plan_result.status\n"
    "                goalset_result = getattr(grasp_plan_result, 'goalset_result', None)\n"
    "                approach_result = getattr(grasp_plan_result, 'approach_result', None)\n"
    "                retract_result = getattr(grasp_plan_result, 'retract_result', None)\n"
    "                self.get_logger().error(\n"
    "                    'Toolshed compatibility: plan_grasp failed; '\n"
    "                    f'status={grasp_plan_result.status}; '\n"
    "                    f'goalset_success={getattr(goalset_result, \"success\", None)}; '\n"
    "                    f'goalset_status={getattr(goalset_result, \"status\", None)}; '\n"
    "                    f'goalset_index={getattr(goalset_result, \"goalset_index\", None)}; '\n"
    "                    f'approach_success={getattr(approach_result, \"success\", None)}; '\n"
    "                    f'approach_status={getattr(approach_result, \"status\", None)}; '\n"
    "                    f'retract_success={getattr(retract_result, \"success\", None)}; '\n"
    "                    f'retract_status={getattr(retract_result, \"status\", None)}'\n"
    "                )\n"
)
if old_grasp_failure not in text:
    raise SystemExit("plan-grasp failure logging patch anchor not found")
text = text.replace(old_grasp_failure, new_grasp_failure, 1)

return_count = text.count("return result\n")
if return_count < 5:
    raise SystemExit("unexpected number of action result returns")
patched_lines = []
for line in text.splitlines(keepends=True):
    if line.lstrip() == "return result\n":
        indent = line[: len(line) - len(line.lstrip())]
        patched_lines.append(f"{indent}self._toolshed_mark_goal_succeeded(goal_handle)\n")
    patched_lines.append(line)
text = "".join(patched_lines)

target = package_dir / "cumotion_goal_set_planner.py"
target.write_text(text)
print(f"overlay={overlay_root}")
print(f"source={source}")
print(f"target={target}")
PY
python3 -m py_compile /tmp/toolshed_isaac_ros_cumotion_overlay/isaac_ros_cumotion/cumotion_goal_set_planner.py
"""
        command = ["docker", "exec", "-i", self.container_name, "bash", "-lc", script]
        try:
            proc = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
            return CommandResult(command, proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command,
                124,
                _timeout_text(exc.stdout),
                _timeout_text(exc.stderr) + "\nTimed out after 30s",
            )

    def fetch_file(self, source_path: str | Path, dest_path: str | Path, timeout_s: float = 10.0) -> CommandResult:
        source_path = str(source_path)
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if self.container_name:
            command = ["docker", "cp", f"{self.container_name}:{source_path}", str(dest_path)]
            try:
                proc = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_s)
                return CommandResult(command, proc.returncode, proc.stdout, proc.stderr)
            except subprocess.TimeoutExpired as exc:
                return CommandResult(
                    command,
                    124,
                    _timeout_text(exc.stdout),
                    _timeout_text(exc.stderr) + f"\nTimed out after {timeout_s}s",
                )

        command = ["cp", source_path, str(dest_path)]
        source = Path(source_path)
        if source.resolve() == dest_path.resolve():
            return CommandResult(command, 0 if dest_path.exists() else 1, "", "")
        if not source.exists():
            return CommandResult(command, 1, "", f"Source file does not exist: {source}")
        shutil.copyfile(source, dest_path)
        return CommandResult(command, 0, "", "")

    def start(
        self,
        name: str,
        args: list[str],
        log_path: str | Path,
        env: Mapping[str, str] | None = None,
    ) -> ProcessState:
        self._ensure_stopped(name)
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        remote_pid_file = None
        if self.container_name:
            remote_pid_file = self.remote_pid_file_for(name)
            self._remote_pid_files[name] = remote_pid_file
            command = self._container_command([str(arg) for arg in args], pid_file=remote_pid_file)
        else:
            command = self.wrap_ros_command([str(arg) for arg in args])
        log_file = log_path.open("ab")
        proc_env = os.environ.copy()
        proc_env.update(env or {})
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=proc_env,
            start_new_session=True,
        )
        self._processes[name] = proc
        self._log_paths[name] = log_path
        return self.status(name)

    def stop(self, name: str) -> ProcessState:
        proc = self._processes.get(name)
        if self.container_name:
            self._stop_remote_process_tree(name)
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        return self.status(name)

    def _stop_remote_process_tree(self, name: str) -> None:
        if not self.container_name:
            return
        pid_file = self._remote_pid_files.get(name) or self.remote_pid_file_for(name)
        script = (
            f"if [ -f {shlex.quote(pid_file)} ]; then "
            f"pid=$(cat {shlex.quote(pid_file)} 2>/dev/null || true); "
            "kill_descendants() { "
            "local parent=\"$1\" sig=\"$2\" child; "
            "for child in $(pgrep -P \"$parent\" 2>/dev/null || true); do "
            "kill_descendants \"$child\" \"$sig\"; "
            "kill -\"$sig\" \"$child\" 2>/dev/null || true; "
            "done; "
            "}; "
            "if [ -n \"$pid\" ]; then "
            "kill -TERM -- \"-$pid\" 2>/dev/null || true; "
            "kill_descendants \"$pid\" TERM; "
            "kill -TERM \"$pid\" 2>/dev/null || true; "
            "sleep 3; "
            "kill -KILL -- \"-$pid\" 2>/dev/null || true; "
            "kill_descendants \"$pid\" KILL; "
            "kill -KILL \"$pid\" 2>/dev/null || true; "
            "fi; "
            f"rm -f {shlex.quote(pid_file)}; "
            "fi"
        )
        subprocess.run(
            ["docker", "exec", "-i", self.container_name, "bash", "-lc", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def stop_remote_patterns(self, patterns: list[str]) -> None:
        if not self.container_name or not patterns:
            return
        quoted_patterns = " ".join(shlex.quote(pattern) for pattern in patterns)
        script = (
            f"for pat in {quoted_patterns}; do "
            "for pid in $(pgrep -f \"$pat\" 2>/dev/null || true); do "
            "[ \"$$\" = \"$pid\" ] && continue; "
            "kill -TERM \"$pid\" 2>/dev/null || true; "
            "done; "
            "done; "
            "sleep 2; "
            f"for pat in {quoted_patterns}; do "
            "for pid in $(pgrep -f \"$pat\" 2>/dev/null || true); do "
            "[ \"$$\" = \"$pid\" ] && continue; "
            "kill -KILL \"$pid\" 2>/dev/null || true; "
            "done; "
            "done"
        )
        subprocess.run(
            ["docker", "exec", "-i", self.container_name, "bash", "-lc", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def status(self, name: str) -> ProcessState:
        proc = self._processes.get(name)
        log_path = self._log_paths.get(name)
        if proc is None:
            remote_state = self._remote_status(name)
            if remote_state is not None:
                return ProcessState(
                    name=name,
                    command=remote_state["command"],
                    running=remote_state["running"],
                    returncode=None if remote_state["running"] else remote_state["returncode"],
                    log_path=str(log_path) if log_path else None,
                )
            return ProcessState(name=name, command=[], running=False, returncode=None, log_path=None)
        return ProcessState(
            name=name,
            command=[str(arg) for arg in proc.args],
            running=proc.poll() is None,
            returncode=proc.poll(),
            log_path=str(log_path) if log_path else None,
        )

    def _remote_status(self, name: str) -> dict[str, Any] | None:
        if not self.container_name:
            return None
        pid_file = self._remote_pid_files.get(name) or self.remote_pid_file_for(name)
        script = (
            f"pid_file={shlex.quote(pid_file)}; "
            "if [ ! -f \"$pid_file\" ]; then exit 4; fi; "
            "pid=$(cat \"$pid_file\" 2>/dev/null || true); "
            "if [ -z \"$pid\" ]; then exit 4; fi; "
            "if kill -0 \"$pid\" 2>/dev/null; then "
            "printf 'running\\n'; ps -p \"$pid\" -o args= 2>/dev/null || true; "
            "else printf 'stopped\\n'; exit 3; fi"
        )
        try:
            proc = subprocess.run(
                ["docker", "exec", "-i", self.container_name, "bash", "-lc", script],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            return None
        lines = proc.stdout.splitlines()
        if lines and lines[0] == "running":
            return {
                "running": True,
                "returncode": None,
                "command": [lines[1]] if len(lines) > 1 and lines[1] else [],
            }
        if lines and lines[0] == "stopped":
            return {"running": False, "returncode": proc.returncode, "command": []}
        return None

    def logs(self, name: str, max_lines: int = 80) -> list[str]:
        log_path = self._log_paths.get(name)
        if log_path is None or not log_path.exists():
            return []
        return tail_lines(log_path.read_text(errors="replace"), max_lines=max_lines)

    def _ensure_stopped(self, name: str) -> None:
        proc = self._processes.get(name)
        if proc is not None and proc.poll() is None:
            raise RuntimeError(f"Process '{name}' is already running")
