# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Mock robot tool for simulation and data collection.

This tool simulates robot operations without requiring actual hardware.
It uses mock data (depth, pointcloud, images) that is injected per-session
to provide realistic responses to the LLM while enabling conversation data collection.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import ray

from toolshed.tools.base import BaseTool, tool_method
from toolshed.tool_result import ToolResult

logger = logging.getLogger(__name__)


class MockRobotTool(BaseTool):
    """Mock robot tool that simulates successful operations using injected mock data.
    
    This tool does NOT communicate with actual robot hardware. Instead, it:
    - Accepts mock data injected per-session (invisible to LLM)
    - Simulates successful robot operations
    - Returns mock sensor data (images, depth maps, point clouds)
    
    Mock data is prepared by the evaluator using a depth estimator, then
    injected automatically by the integration layer on each tool call.
    """
    
    def __init__(
        self,
        no_output_image: bool = False,
        no_output_vars: bool = False,
        exclude_methods: list[str] | None = None,
        exclude_behavior: str = "warn",
    ) -> None:
        super().__init__(
            no_output_image=no_output_image,
            no_output_vars=no_output_vars,
            exclude_methods=exclude_methods,
            exclude_behavior=exclude_behavior,
        )

    def get_name(self) -> str:
        return "robot"

    @tool_method
    def execute_grasp(
        self,
        grasp_pose,
        _mock_data: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Execute a grasp by moving the robot to the specified pose via a pre-grasp point, and closing the gripper.

        [[if:text]]Text output: Status of the grasp execution.[[/if:text]]
        [[if:image]]Image output: View from robot camera after the grasp is executed.[[/if:image]]
        [[if:vars]]Stored variables: $captured_image (PIL Image) after the grasp is executed.[[/if:vars]]

        Args:
            grasp_pose: 4×4 transformation matrix (list or numpy array) representing the grasp pose
                       in the robot's camera frame (OpenCV convention).

        Returns:
            ToolResult: value dict with ``success`` (bool) and ``execution_time_s`` (float).
        """
        # Simulate successful grasp
        value = {
            "success": True,
            "execution_time_s": 2.5,  # Simulated execution time
        }

        # THIS STRING IS IMPORTANT DON'T CHANGE IT WITHOUT CHANGING bop_ask.py
        text = "Grasp execution succeeded."

        return ToolResult(value, text=text)

    @tool_method
    def place_object_at_2d_location(
        self,
        placement_point_2d,
        _mock_data: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Simulate placing object at 2D location (always succeeds).

        [[if:text]]Text output: Confirmation that placement was successful.[[/if:text]]

        Args:
            placement_point_2d: 2D normalized image coordinate [x, y] in the range [0, 1]
                             where the object should be placed.

        Returns:
            ToolResult: value dict with ``success`` (bool) and ``execution_time_s`` (float).
        """
        value = {
            "success": True,
            "execution_time_s": 1.8,
        }

        text = "Placement succeeded."

        return ToolResult(value, text=text)

    #@tool_method
    def place_object_at_3d_location(
        self,
        placement_point_3d,
        _mock_data: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Simulate placing object at 3D location (always succeeds).

        [[if:text]]Text output: Confirmation that placement was successful.[[/if:text]]

        Args:
            placement_point_3d: 3D point [x, y, z] in the robot's camera frame (list or numpy array)
                             where the object should be placed.

        Returns:
            ToolResult: value dict with ``success`` (bool) and ``execution_time_s`` (float).
        """
        value = {
            "success": True,
            "execution_time_s": 1.8,
        }

        text = "Placement succeeded."

        return ToolResult(value, text=text)

    @tool_method
    def capture_image(self, _mock_data: Optional[Dict[str, Any]] = None) -> ToolResult:
        """
        Capture an RGB image from the robot's camera showing the current scene.

        [[if:text]]Text output: Image dimensions and capture status.[[/if:text]]
        Image output: RGB image from robot camera.
        [[if:vars]]Stored variables: $captured_image (PIL Image).[[/if:vars]]

        Returns:
            ToolResult: value dict with ``image_shape`` (tuple); image output as noted above.
        """
        # Extract mock image - FAIL EARLY if missing
        if not _mock_data or "mock_image" not in _mock_data:
            raise RuntimeError(
                "MockRobotTool.capture_image() requires mock_data with 'mock_image'. "
                "This should be injected by the integration layer. "
                "If you're seeing this error, mock_data was not properly prepared or injected."
            )

        # Get image from mock data (might be Ray ObjectRef), then coerce to PIL Image
        image = _mock_data["mock_image"]
        if isinstance(image, ray.ObjectRef):
            image = ray.get(image)

        # After resolution, image must be a PIL Image per prepare_mock_data_from_image()
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"MockRobotTool.capture_image expected PIL.Image.Image, got {type(image)}"
            )

        width, height = image.size

        value = {
            "image_shape": (width, height),
            "image": image,
        }

        text = f"Captured image: {width}×{height} pixels."

        #images = None if self.no_output_image else [image]
        images = [image]
        variables = {} if self.no_output_vars else {"captured_image": image}

        return ToolResult(value, text=text, image=images, variables=variables)

    @tool_method
    def get_depth(self, _mock_data: Optional[Dict[str, Any]] = None) -> ToolResult:
        """
        Retrieve depth map from the robot's depth sensor.

        [[if:text]]Text output: Summary of depth data including image dimensions, focal length, and depth statistics.[[/if:text]]
        [[if:image]]Image output: A colorized depth map visualization where closer objects appear cooler (blue/purple) and distant objects appear warmer (red/yellow).[[/if:image]]
        [[if:vars]]Stored variables: $depth_map (2D numpy array of depth values in meters), $focal_length_px (float, estimated focal length in pixels).[[/if:vars]]

        Returns:
            ToolResult: value dict with ``depth_map`` (numpy array), ``focal_length_px`` (float),
                       ``width`` (int), and ``height`` (int).
        """
        # Extract from mock data - FAIL EARLY if missing
        if not _mock_data:
            raise RuntimeError(
                "MockRobotTool.get_depth() requires mock_data. "
                "This should be injected by the integration layer."
            )

        required_keys = ["mock_depth_map", "mock_focal_length_px", "image_width", "image_height"]
        missing_keys = [key for key in required_keys if key not in _mock_data]
        if missing_keys:
            raise RuntimeError(
                f"MockRobotTool.get_depth() requires mock_data with keys: {missing_keys}. "
                "Mock data should be prepared by the evaluator with depth estimator output."
            )

        # Extract data (handle Ray ObjectRefs)
        depth_map = self._resolve_ray_ref(_mock_data["mock_depth_map"])
        if not isinstance(depth_map, np.ndarray):
            raise TypeError(
                f"MockRobotTool.get_depth expected depth_map as numpy.ndarray, got {type(depth_map)}"
            )
        focal_length_px = _mock_data["mock_focal_length_px"]
        width = _mock_data["image_width"]
        height = _mock_data["image_height"]
        rgb_image = self._resolve_ray_ref(_mock_data.get("mock_image"))

        # Create visualization only if not suppressed
        depth_viz = None
        if not self.no_output_image and rgb_image is not None:
            # Import visualization from real robot or depth estimator tool
            from toolshed.tools.robot import RobotTool
            depth_viz = [RobotTool._create_depth_visualization(rgb_image, depth_map)]

        # Calculate depth statistics
        valid_depths = depth_map[~np.isnan(depth_map) & ~np.isinf(depth_map)]
        if len(valid_depths) > 0:
            min_depth = float(np.min(valid_depths))
            max_depth = float(np.max(valid_depths))
            mean_depth = float(np.mean(valid_depths))
            depth_stats_text = f" Depth range: {min_depth:.2f}m to {max_depth:.2f}m (mean: {mean_depth:.2f}m)."
        else:
            depth_stats_text = ""

        value = {
            "depth_map": depth_map,
            "focal_length_px": focal_length_px,
            "width": width,
            "height": height,
        }

        variables = {} if self.no_output_vars else {
            "depth_map": depth_map,
            "focal_length_px": focal_length_px
        }

        text = (
            f"Retrieved depth map for {width}x{height} image. "
            f"Focal length: {focal_length_px:.1f}px.{depth_stats_text}"
        )
        if not self.no_output_vars:
            text += (
                f" Use $depth_map (numpy array, {height}x{width}) to reference the depth data "
                f"and $focal_length_px (float) to reference the focal length."
            )

        return ToolResult(value, text=text, image=depth_viz, variables=variables)

    @tool_method
    def get_depth_with_pointcloud(
        self, _mock_data: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Retrieve depth map from robot's depth sensor and generate 3D point cloud.

        [[if:text]]Text output: Summary of depth data and point cloud generation including dimensions, focal length, depth statistics, and point cloud size.[[/if:text]]
        [[if:image]]Image output: A colorized depth map visualization where closer objects appear warmer (red/yellow) and distant objects appear cooler (blue/purple).[[/if:image]]
        [[if:vars]]Stored variables: $depth_map (2D array of depth values in meters), $point_cloud (Nx3 array of 3D points), $focal_length_px (float, estimated focal length in pixels).[[/if:vars]]

        Returns:
            ToolResult: value dict with ``depth_map`` (numpy array), ``focal_length_px`` (float),
                       ``point_cloud`` (numpy array), ``width`` (int), and ``height`` (int).
        """
        # First get the depth map
        depth_tool_result = self.get_depth(_mock_data=_mock_data)
        depth_result = depth_tool_result.value

        # Extract point cloud from mock data - FAIL EARLY if missing
        if not _mock_data or "mock_point_cloud" not in _mock_data:
            raise RuntimeError(
                "MockRobotTool.get_depth_with_pointcloud() requires mock_data with 'mock_point_cloud'. "
                "Mock data should be prepared by the evaluator with depth estimator output."
            )

        point_cloud = self._resolve_ray_ref(_mock_data["mock_point_cloud"])

        value = {
            **depth_result,
            "point_cloud": point_cloud,
        }

        width = depth_result["width"]
        height = depth_result["height"]

        # Calculate point cloud statistics
        if len(point_cloud) > 0:
            z_values = point_cloud[:, 2]
            z_min = float(np.min(z_values))
            z_max = float(np.max(z_values))
            z_mean = float(np.mean(z_values))
            pc_stats_text = f" Depth range in point cloud: {z_min:.2f}m to {z_max:.2f}m (mean: {z_mean:.2f}m)."
        else:
            pc_stats_text = ""

        variables = {} if self.no_output_vars else {
            "depth_map": depth_result["depth_map"],
            "point_cloud": point_cloud,
            "focal_length_px": depth_result["focal_length_px"],
        }

        text = f"Retrieved depth map and generated {len(point_cloud):,} 3D points.{pc_stats_text}"
        if not self.no_output_vars:
            text += (
                f" Use $depth_map (numpy array, {height}x{width}), "
                f"$point_cloud (numpy array, {len(point_cloud)}x3) and "
                f"$focal_length_px (float) to reference the results."
            )

        return ToolResult(value, text=text, image=depth_tool_result.image, variables=variables)

    @staticmethod
    def _resolve_ray_ref(obj):
        """Helper to resolve Ray ObjectRefs."""
        if obj is None:
            return None
        if isinstance(obj, ray.ObjectRef):
            return ray.get(obj)
        return obj
