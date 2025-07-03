# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import logging
from typing import Optional, List
from io import BytesIO
import numpy as np

import requests
from PIL import Image, ImageDraw, ImageFont

from toolshed.tools.base import BaseTool, tool_method
from toolshed.tool_result import ToolResult

logger = logging.getLogger(__name__)


class RobotTool(BaseTool):
    """Tool for controlling a robot system via HTTP API."""

    def __init__(
        self,
        robot_api_url: str,
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
        self._robot_api_url = robot_api_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "robot"

    # ------------------------------------------------------------------
    # Tool methods
    # ------------------------------------------------------------------

    @tool_method
    def execute_grasp(
        self,
        grasp_pose,
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
        # Convert to list (handles numpy array, string, or list input)
        grasp_pose = self.retrieve_matrix(grasp_pose, expected_shape=(4, 4))

        # Call robot API endpoint
        response = requests.post(
            f"{self._robot_api_url}/execute_grasp",
            json={"grasp_pose": grasp_pose},
            timeout=60.0,
        )
        response.raise_for_status()

        # Parse npz response
        buffer = BytesIO(response.content)
        data = np.load(buffer, allow_pickle=True)

        success = bool(data["success"])
        execution_time = float(data["execution_time_s"])
        image = Image.fromarray(data["image"])

        value = {
            "success": success,
            "execution_time_s": execution_time,
            "image": image,
        }

        text = (
            f"Grasp execution {'succeeded' if success else 'failed'}."
        )

        variables = {} if self.no_output_vars else {"captured_image": image}

        return ToolResult(value, image=[image], text=text, variables=variables)

    @tool_method
    def place_object_at_2d_location(
        self,
        placement_point_2d,
    ) -> ToolResult:
        """
        Move the robot to a place it's currently held object based on a 2D normalized image coordinate.

        The tool will convert to a 3D placement location automatically by shooting a ray.

        [[if:text]]Text output: Status of the release operation.[[/if:text]]
        [[if:image]]Image output: View from robot camera after the placement is executed.[[/if:image]]
        [[if:vars]]Stored variables: $captured_image (PIL Image) after the placement is executed.[[/if:vars]]

        Args:
            placement_point_2d: 2D normalized image coordinate [x, y] in the range [0, 1]
                             where the object should be placed.

        Returns:
            ToolResult: value dict with ``success`` (bool) and ``execution_time_s`` (float).
        """
        # Debug what we're actually receiving
        print(f"DEBUG place_object_at_2d_location - type: {type(placement_point_2d)}, value: {repr(placement_point_2d)}")

        # Convert to list of floats
        placement_point_2d = self.retrieve_list_of_floats(placement_point_2d)

        print("Calling place_object_at_2d_location with placement_point_2d:", placement_point_2d)

        # Call robot API endpoint
        response = requests.post(
            f"{self._robot_api_url}/place_object_at_2d_location",
            json={"placement_point_2d": placement_point_2d},
            timeout=60.0,
        )
        response.raise_for_status()

        # Parse npz response
        buffer = BytesIO(response.content)
        data = np.load(buffer, allow_pickle=True)

        success = bool(data["success"])
        execution_time = float(data["execution_time_s"])
        image = Image.fromarray(data["image"])

        value = {
            "success": success,
            "execution_time_s": execution_time,
            "image": image,
        }

        text = (
            f"Release operation {'succeeded' if success else 'failed'}."
        )

        variables = {} if self.no_output_vars else {"captured_image": image}

        return ToolResult(value, image=[image], text=text, variables=variables)

    #@tool_method
    def place_object_at_3d_location(
        self,
        placement_point_3d,
    ) -> ToolResult:
        """
        Move the robot to a 3D placement point and open the gripper to place the object.

        [[if:text]]Text output: Status of the placement operation.[[/if:text]]
        [[if:image]]Image output: View from robot camera after the placement is executed.[[/if:image]]
        [[if:vars]]Stored variables: $captured_image (PIL Image) after the placement is executed.[[/if:vars]]

        Args:
            placement_point_3d: 3D point [x, y, z] in the robot's camera frame (list or numpy array)
                             where the object should be placed.

        Returns:
            ToolResult: value dict with ``success`` (bool) and ``execution_time_s`` (float).
        """
        # Convert to list of floats
        placement_point_3d = self.retrieve_list_of_floats(placement_point_3d)

        # Call robot API endpoint
        response = requests.post(
            f"{self._robot_api_url}/place_object_at_3d_location",
            json={"placement_point_3d": placement_point_3d},
            timeout=60.0,
        )
        response.raise_for_status()

        # Parse npz response
        buffer = BytesIO(response.content)
        data = np.load(buffer, allow_pickle=True)

        success = bool(data["success"])
        execution_time = float(data["execution_time_s"])
        image = Image.fromarray(data["image"])

        value = {
            "success": success,
            "execution_time_s": execution_time,
            "image": image,
        }

        text = (
            f"Release operation {'succeeded' if success else 'failed'}."
        )

        variables = {} if self.no_output_vars else {"captured_image": image}

        return ToolResult(value, image=[image], text=text, variables=variables)

    @tool_method
    def capture_image(self) -> ToolResult:
        """
        Capture an RGB image from the robot's camera showing the current scene.

        [[if:text]]Text output: Image dimensions and capture status.[[/if:text]]
        Image output: RGB image from robot camera.
        [[if:vars]]Stored variables: $captured_image (PIL Image).[[/if:vars]]

        Returns:
            ToolResult: value dict with ``image_shape`` (tuple); image output as noted above.
        """
        # Call robot API endpoint
        response = requests.get(
            f"{self._robot_api_url}/capture_image",
            timeout=10.0,
        )
        response.raise_for_status()

        # API now returns PNG image bytes
        image = Image.open(BytesIO(response.content)).convert("RGB")

        value = {
            "image_shape": image.size,  # (width, height)
            "image": image,
        }

        text = f"Captured image: {image.size[0]}×{image.size[1]} pixels."

        images: Optional[List[Image.Image]] = None if self.no_output_image else [image]

        variables = {} if self.no_output_vars else {"captured_image": image}

        return ToolResult(value, text=text, image=images, variables=variables)

    @tool_method
    def get_depth(self) -> ToolResult:
        """
        Retrieve depth map from the robot's depth sensor.

        [[if:text]]Text output: Summary of depth data including image dimensions, focal length, and depth statistics.[[/if:text]]
        [[if:image]]Image output: A colorized depth map visualization where closer objects appear cooler (blue/purple) and distant objects appear warmer (red/yellow).[[/if:image]]
        [[if:vars]]Stored variables: $depth_map (2D numpy array of depth values in meters), $focal_length_px (float, estimated focal length in pixels).[[/if:vars]]

        Returns:
            ToolResult: value dict with ``depth_map`` (numpy array), ``focal_length_px`` (float),
                       ``width`` (int), and ``height`` (int).
        """
        # Call robot API endpoint to get depth data
        response = requests.get(
            f"{self._robot_api_url}/get_depth",
            timeout=10.0,
        )
        response.raise_for_status()

        # Parse npz response
        buffer = BytesIO(response.content)
        data = np.load(buffer)

        # Extract depth data from API response
        depth_map = data["depth_map"]
        focal_length_px = float(data["focal_length_px"])
        width = int(data["width"])
        height = int(data["height"])

        # Create visualization only if not suppressed
        #depth_viz = None if self.no_output_image else [self._create_depth_visualization(rgb_image, depth_map)]
        depth_viz = None if self.no_output_image else [self._create_lean_depth_visualization(depth_map)]

        # Calculate depth statistics for the text description
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
            "depth_map_visualization": depth_viz,
            "focal_length_px": focal_length_px,
            "width": width,
            "height": height
        }

        # Prepare variables only if not suppressed
        variables = {} if self.no_output_vars else {"depth_map": depth_map, "focal_length_px": focal_length_px}

        # Adjust text based on whether variables are suppressed
        text = f"Retrieved depth map for {width}x{height} image from robot. Focal length: {focal_length_px:.1f}px.{depth_stats_text}"
        if not self.no_output_vars:
            text += f" Use $depth_map (numpy array, {height}x{width}) to reference the depth data and $focal_length_px (float) to reference the focal length."

        return ToolResult(value, text=text, image=depth_viz, variables=variables)

    @tool_method
    def get_depth_with_pointcloud(self) -> ToolResult:
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
        depth_tool_result = self.get_depth()
        depth_result = depth_tool_result.value  # Extract the actual dict

        # Generate point cloud
        point_cloud = self._compute_point_cloud(
            depth_result["depth_map"],
            depth_result["focal_length_px"],
            depth_result["width"],
            depth_result["height"]
        )

        # Re-create visualization only if not suppressed
        #depth_viz = None if self.no_output_image else [self._create_depth_visualization(rgb_image, depth_result["depth_map"])]
        #depth_viz = None if self.no_output_image else [self._create_lean_depth_visualization(depth_result["depth_map"])]
        depth_viz = depth_result["depth_map_visualization"]

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

        # Prepare variables only if not suppressed
        variables = {} if self.no_output_vars else {
            "depth_map": depth_result["depth_map"],
            "point_cloud": point_cloud,
            "focal_length_px": float(depth_result["focal_length_px"])
        }

        # Adjust text based on whether variables are suppressed
        text = f"Retrieved depth map and generated {len(point_cloud):,} 3D points from robot data.{pc_stats_text}"
        if not self.no_output_vars:
            text += f" Use $depth_map (numpy array, {height}x{width}), $point_cloud (numpy array, {len(point_cloud)}x3) and $focal_length_px (float) to reference the results."

        return ToolResult(value, text=text, image=depth_viz, variables=variables)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def retrieve_list_of_floats(value):
        """
        Convert various input formats (numpy array, string, list) to a list of floats.

        Args:
            value: Input value which can be a numpy array, string (comma-separated or JSON format), or list.

        Returns:
            list: A list of floats.

        Raises:
            ValueError: If the input cannot be converted to a list of floats.
        """
        import json
        import ray

        # Store original value for error messages
        original_value = value
        original_type = type(value).__name__

        # Convert to list if numpy array
        if isinstance(value, np.ndarray):
            return value.tolist()

        # If it's a string, try to parse it
        if isinstance(value, str):
            # First try JSON parsing (handles "[0.714, 0.713]" format)
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [float(x) for x in parsed]
                else:
                    raise ValueError(
                        f"After parsing JSON string '{value}', got {type(parsed).__name__} instead of list: {parsed}. "
                        f"Original input type: {original_type}, value: {repr(original_value)}"
                    )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Cannot parse string as JSON list: '{value}'. JSON Error: {e}. "
                    f"Original input type: {original_type}, value: {repr(original_value)}"
                )

        # If it's already a list, validate and return
        if isinstance(value, list):
            try:
                return [float(x) for x in value]
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert list elements to floats: {value}. Error: {e}. "
                    f"Original input type: {original_type}, value: {repr(original_value)}"
                )

        # If we reach here, the input type is unexpected
        raise ValueError(
            f"Expected list, numpy array, or JSON string, but got {type(value).__name__}: {repr(value)}. "
            f"Original input type: {original_type}, value: {repr(original_value)}"
        )

    @staticmethod
    def retrieve_matrix(value, expected_shape=(4, 4)):
        """
        Convert various input formats (numpy array, string, list) to a nested list matrix.

        Args:
            value: Input value which can be a numpy array, string (JSON format or comma-separated),
                   nested list, or flat list.
            expected_shape: Expected shape of the matrix as a tuple (rows, cols). Default is (4, 4).

        Returns:
            list: A nested list representing the matrix (e.g., [[...], [...], ...]).

        Raises:
            ValueError: If the input cannot be converted to a matrix of the expected shape.
        """
        import json
        rows, cols = expected_shape
        expected_size = rows * cols

        # Convert numpy array to list
        if isinstance(value, np.ndarray):
            # If it's already the right shape, convert to nested list
            if value.shape == expected_shape:
                return value.tolist()
            # If it's flat, reshape and convert
            elif value.size == expected_size:
                return value.reshape(expected_shape).tolist()
            else:
                raise ValueError(f"Numpy array has shape {value.shape}, expected {expected_shape} or flat array of size {expected_size}")

        # Parse string input
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                # Try comma-separated format
                try:
                    parsed = [float(x.strip()) for x in value.split(",")]
                except ValueError:
                    raise ValueError(f"Cannot parse string as matrix: {value}")

            # Now handle the parsed value (could be nested or flat list)
            value = parsed

        # At this point, value should be a list
        if not isinstance(value, list):
            raise ValueError(f"Expected list, numpy array, or string, got {type(value)}")

        # Check if it's already a nested list with correct shape
        if len(value) == rows and all(isinstance(row, list) and len(row) == cols for row in value):
            # Convert all elements to float to ensure consistency
            return [[float(x) for x in row] for row in value]

        # Check if it's a flat list that we can reshape
        if all(not isinstance(x, list) for x in value):
            if len(value) == expected_size:
                # Convert flat list to nested list
                flat_list = [float(x) for x in value]
                return [flat_list[i * cols:(i + 1) * cols] for i in range(rows)]
            else:
                raise ValueError(f"Flat list has {len(value)} elements, expected {expected_size} for {expected_shape} matrix")

        # If we get here, the structure doesn't match
        raise ValueError(f"Cannot convert value to {expected_shape} matrix. Got: {value}")

    @staticmethod
    def _create_depth_visualization(
        original_image: Image.Image,
        depth_map: np.ndarray,
        colormap: str = 'turbo',
        show_scale: bool = True,
        scale_position: str = 'right',
        scale_width_pct: float = 0.03,
        scale_margin_pct: float = 0.02,
        font_size_pct: float = 0.02
    ) -> Image.Image:
        """Create a visualization of the depth map with a color scale.

        Args:
            original_image: Original PIL image (for getting dimensions)
            depth_map: 2D numpy array of depth values
            colormap: Name of matplotlib colormap to use ('turbo', 'viridis', 'plasma', 'jet', etc.)
            show_scale: Whether to show the depth scale bar
            scale_position: Position of scale bar ('right' or 'left')
            scale_width_pct: Width of scale bar as percentage of image width
            scale_margin_pct: Margin around scale bar as percentage of image width
            font_size_pct: Font size as percentage of image height

        Returns:
            PIL Image with colorized depth map and optional scale bar
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        # Get dimensions
        width, height = original_image.size

        # Normalize depth values to 0-1 range
        valid_depths = depth_map[~np.isnan(depth_map) & ~np.isinf(depth_map)]
        if len(valid_depths) == 0:
            # Handle edge case of all invalid depths
            vmin, vmax = 0, 1
        else:
            # Use percentiles for robust min/max to handle outliers
            vmin = np.percentile(valid_depths, 1)
            vmax = np.percentile(valid_depths, 99)

        # Apply colormap
        cmap = cm.get_cmap(colormap)
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Convert depth map to RGB using colormap
        depth_normalized = norm(depth_map)
        depth_colored = cmap(depth_normalized)

        # Convert to PIL Image (remove alpha channel)
        depth_rgb = (depth_colored[:, :, :3] * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_rgb)

        if not show_scale:
            return depth_image

        # Calculate scale bar dimensions
        scale_width = max(20, int(width * scale_width_pct))
        scale_margin = max(10, int(width * scale_margin_pct))
        total_width = width + scale_width + 3 * scale_margin

        # Create a new image with space for the scale bar
        combined = Image.new('RGB', (total_width, height), color='white')

        # Place the depth image
        if scale_position == 'right':
            combined.paste(depth_image, (0, 0))
            scale_x = width + scale_margin
        else:
            combined.paste(depth_image, (scale_width + 3 * scale_margin, 0))
            scale_x = scale_margin

        # Create the scale bar
        scale_height = height - 2 * scale_margin
        scale_y = scale_margin

        # Generate gradient for scale bar
        gradient = np.linspace(1, 0, scale_height).reshape(-1, 1)
        gradient = np.repeat(gradient, scale_width, axis=1)

        # Apply the same colormap to the gradient
        gradient_colored = cmap(gradient)
        gradient_rgb = (gradient_colored[:, :, :3] * 255).astype(np.uint8)
        gradient_image = Image.fromarray(gradient_rgb)

        # Paste the gradient scale bar
        combined.paste(gradient_image, (scale_x, scale_y))

        # Draw scale labels
        draw = ImageDraw.Draw(combined)

        # Try to get a reasonable font size
        font_size = max(12, int(height * font_size_pct))
        try:
            # Try to use a better font if available
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            # Fall back to default font
            font = ImageFont.load_default()

        # Add tick marks and labels
        num_ticks = 5
        for i in range(num_ticks):
            # Calculate position and value
            y_pos = scale_y + int(i * scale_height / (num_ticks - 1))
            depth_value = vmax - (vmax - vmin) * i / (num_ticks - 1)

            # Draw tick mark
            tick_x = scale_x + scale_width
            draw.line([(tick_x, y_pos), (tick_x + 5, y_pos)], fill='black', width=1)

            # Draw label
            label = f"{depth_value:.1f}m"
            # Get text bounding box for alignment
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            label_x = tick_x + 8
            label_y = y_pos - text_height // 2
            draw.text((label_x, label_y), label, fill='black', font=font)

        # Add title for the scale
        title = "Depth (m)"
        bbox = draw.textbbox((0, 0), title, font=font)
        title_width = bbox[2] - bbox[0]
        title_x = scale_x + (scale_width - title_width) // 2
        title_y = scale_y - font_size - 5
        draw.text((title_x, title_y), title, fill='black', font=font)

        return combined

    @staticmethod
    def _create_lean_depth_visualization(
        depth_map: np.ndarray,
    ) -> Image.Image:
        """Create a depth visualization of the depth map using OpenCV colormap.

        Args:
            depth_map: 2D numpy array of depth values

        Returns:
            PIL Image with colorized depth map (no legend)
        """
        import cv2

        # Get valid depths for normalization
        valid_depths = depth_map[~np.isnan(depth_map) & ~np.isinf(depth_map)]
        if len(valid_depths) == 0:
            # Handle edge case of all invalid depths
            vmin, vmax = 0, 1
        else:
            # Use percentiles for robust min/max to handle outliers
            vmin = np.percentile(valid_depths, 1)
            vmax = np.percentile(valid_depths, 99)

        # Normalize to 0-255 range
        depth_normalized = np.clip((depth_map - vmin) / (vmax - vmin), 0, 1)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # Apply OpenCV colormap (TURBO is similar to matplotlib's turbo)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

        # Convert from BGR to RGB (OpenCV uses BGR)
        depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        return Image.fromarray(depth_rgb)

    @staticmethod
    def _compute_point_cloud(
        depth_map: np.ndarray,
        focal_length_px: float,
        width: int,
        height: int
    ) -> np.ndarray:
        """Convert a depth map to a 3D point cloud.

        Args:
            depth_map: 2D numpy array containing depth values in meters
            focal_length_px: Focal length of the camera in pixels
            width: Width of the image in pixels
            height: Height of the image in pixels

        Returns:
            Numpy array of shape (N, 3) containing x, y, z coordinates
        """
        # Create pixel coordinate grid
        v, u = np.mgrid[0:height, 0:width]

        # Compute principal point (assume center of image)
        cx = width / 2
        cy = height / 2

        # Create normalized image coordinates
        x = (u - cx) / focal_length_px
        y = (v - cy) / focal_length_px

        # Scale by depth to get 3D coordinates
        x = x * depth_map
        y = y * depth_map
        z = depth_map

        # Stack coordinates and reshape to (N, 3)
        points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)

        # Filter out invalid points (NaN, Inf, or very far)
        valid_mask = (~np.isnan(points).any(axis=1) &
                     ~np.isinf(points).any(axis=1) &
                     (points[:, 2] > 0) &  # Positive depth only
                     (points[:, 2] < 100))  # Reasonable max depth (100m)
        points = points[valid_mask]

        return points

