# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Depth Estimation tool using Apple's ml-depth-pro model.

This tool provides monocular depth estimation capabilities:
1. estimate_depth(image): Estimate depth map from a single image
2. estimate_depth_with_pointcloud(image): Estimate depth and generate 3D point cloud

The image argument can be either a PIL.Image.Image object or a Ray
ObjectRef pointing to one (to avoid large argument copies between Ray
workers).

NOTE: This implementation expects the depth_pro package to be installed.
The actor must request at least one GPU when configured via toolshed's
`tool_configs`.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from toolshed.tools.base import BaseTool, tool_method
from toolshed.tool_result import ToolResult

# Heavy dependencies - imported directly since tools are only instantiated
# inside Ray actors running in tool-specific conda environments where
# dependencies are guaranteed to be available.
import depth_pro  # type: ignore
from depth_pro.depth_pro import (
    DepthProConfig,
    DEFAULT_MONODEPTH_CONFIG_DICT,
)
import torch  # type: ignore

logger = logging.getLogger(__name__)


def _find_toolshed_root() -> str:
    """Find toolshed root from package location."""
    import toolshed
    return os.path.dirname(os.path.dirname(os.path.abspath(toolshed.__file__)))


class DepthEstimatorTool(BaseTool):
    """Depth estimation tool using ml-depth-pro."""

    def __init__(self, checkpoint_path: Optional[str] = None, 
                 no_output_image: bool = False, no_output_vars: bool = False,
                 exclude_methods: list[str] = None, exclude_behavior: str = "warn") -> None:
        """Initialize the depth estimation tool.
        
        Args:
            checkpoint_path: Optional path to the depth_pro checkpoint. If not provided,
                           will look for it in standard locations.
            no_output_image: If True, suppress image outputs in ToolResults.
            no_output_vars: If True, suppress variable outputs in ToolResults.
            exclude_methods: List of method names to exclude from schema and optionally block at runtime.
            exclude_behavior: How to handle calls to excluded methods ("warn", "error", or "silent").
        """
        super().__init__(no_output_image=no_output_image, no_output_vars=no_output_vars,
                         exclude_methods=exclude_methods, exclude_behavior=exclude_behavior)

        # Resolve checkpoint path
        if checkpoint_path is None:
            # Default: use checkpoints/depth_pro.pt relative to toolshed root
            toolshed_root = _find_toolshed_root()
            checkpoint_path = os.path.join(toolshed_root, "checkpoints", "depth_pro.pt")
        else:
            # Expand user home directory and resolve relative paths
            checkpoint_path = os.path.expanduser(checkpoint_path)
            if not os.path.isabs(checkpoint_path):
                toolshed_root = _find_toolshed_root()
                checkpoint_path = os.path.join(toolshed_root, checkpoint_path)
        
        # Validate checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Depth checkpoint not found: {checkpoint_path}\n"
                "Download from: https://github.com/apple/ml-depth-pro"
            )

        # Load the model
        logger.info("Loading depth estimation model …")

        config = DepthProConfig(
            patch_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.patch_encoder_preset,
            image_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.image_encoder_preset,
            decoder_features=DEFAULT_MONODEPTH_CONFIG_DICT.decoder_features,
            use_fov_head=DEFAULT_MONODEPTH_CONFIG_DICT.use_fov_head,
            fov_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset,
            checkpoint_uri=checkpoint_path,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model, self._transform = depth_pro.create_model_and_transforms(
            config=config,
            device=device,
            precision=torch.float32,
        )
        self._model.eval()
        logger.info("Depth model loaded on %s", device)

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------
    def get_name(self) -> str:
        return "depth_estimator"

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------
    @tool_method
    def estimate_depth(self, image) -> ToolResult[Dict[str, Any]]:
        """Estimate depth map from a single image.

        [[if:text]]Text output: Summary of depth estimation including image dimensions, focal length, and depth statistics.[[/if:text]]
        [[if:image]]Visual output: A colorized depth map visualization where closer objects appear cooler (blue/purple) and distant objects appear warmer (red/yellow).[[/if:image]]
        [[if:vars]]Stored variables: $depth_map: 2D array of depth values in meters, $focal_length_px: Estimated focal length in pixels[[/if:vars]]

        Args:
            image (Image): Image to analyze for depth estimation.

        Returns:
            dict: Dictionary containing:
                - depth_map: 2D numpy array of depth values in meters
                - focal_length_px: Estimated focal length in pixels
                - width: Image width in pixels
                - height: Image height in pixels
        """
        img = self._resolve_image(image)
        
        # Convert PIL image to numpy array (RGB)
        img_array = np.array(img.convert('RGB'))
        height, width = img_array.shape[:2]
        
        # Transform the image for the model
        img_tensor = self._transform(img_array)
        
        # Add batch dimension if needed
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            # The model is already on the correct device
            prediction = self._model.infer(img_tensor, f_px=None)
            
        # Extract results
        depth_map = prediction["depth"].cpu().numpy()
        focal_length_px = prediction["focallength_px"]
        
        # Convert focal length to float if it's a tensor
        if isinstance(focal_length_px, torch.Tensor):
            focal_length_px = focal_length_px.item()
        
        # Create visualization only if not suppressed
        depth_viz = None if self.no_output_image else self._create_depth_visualization(img, depth_map)
        
        # Calculate depth statistics for the text description
        valid_depths = depth_map[~np.isnan(depth_map) & ~np.isinf(depth_map)]
        if len(valid_depths) > 0:
            min_depth = float(np.min(valid_depths))
            max_depth = float(np.max(valid_depths))
            mean_depth = float(np.mean(valid_depths))
            depth_stats_text = f" Depth range: {min_depth:.2f}m to {max_depth:.2f}m (mean: {mean_depth:.2f}m)."
        else:
            depth_stats_text = ""
        
        result_dict = {
            "depth_map": depth_map,
            "focal_length_px": float(focal_length_px),
            "width": width,
            "height": height
        }
        
        # Prepare variables only if not suppressed
        variables = {} if self.no_output_vars else {"depth_map": depth_map, "focal_length_px": float(focal_length_px)}
        
        # Adjust text based on whether variables are suppressed
        text = f"Estimated depth map for {width}x{height} image. Focal length: {focal_length_px:.1f}px.{depth_stats_text}"
        if not self.no_output_vars:
            text += f" Use $depth_map (numpy array, {height}x{width}) to reference the depth data and $focal_length_px (float) to reference the focal length."
        
        return ToolResult(
            result_dict,  # The actual value for code execution
            text=text,
            image=depth_viz,  # Will be None if no_output_image is True
            variables=variables
        )

    @tool_method
    def estimate_depth_with_pointcloud(self, image) -> ToolResult[Dict[str, Any]]:
        """Estimate depth and generate 3D point cloud from a single image.

        [[if:text]]Text output: Summary of depth estimation and point cloud generation including dimensions, focal length, depth statistics, and point cloud size.[[/if:text]]
        [[if:image]]Visual output: A colorized depth map visualization where closer objects appear warmer (red/yellow) and distant objects appear cooler (blue/purple).[[/if:image]]
        [[if:vars]]Stored variables: 
            - $depth_map: 2D array of depth values in meters
            - $point_cloud: Nx3 array of 3D points (x, y, z) in camera coordinates
            - $focal_length_px: Estimated focal length in pixels[[/if:vars]]

        Args:
            image (Image): Image to analyze for depth and 3D reconstruction.

        Returns:
            dict: Dictionary containing:
                - depth_map: 2D numpy array of depth values in meters
                - focal_length_px: Estimated focal length in pixels
                - point_cloud: Nx3 numpy array of 3D points (x, y, z) in camera coordinates
                - width: Image width in pixels
                - height: Image height in pixels
        """
        # First get the depth map (now returns ToolResult)
        depth_tool_result = self.estimate_depth(image)
        depth_result = depth_tool_result.value  # Extract the actual dict
        
        # Generate point cloud
        point_cloud = self._compute_point_cloud(
            depth_result["depth_map"],
            depth_result["focal_length_px"],
            depth_result["width"],
            depth_result["height"]
        )
        
        # Re-create visualization only if not suppressed
        img = self._resolve_image(image)
        depth_viz = None if self.no_output_image else self._create_depth_visualization(img, depth_result["depth_map"])
        
        result_dict = {
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
        text = f"Generated {width}x{height} depth map and {len(point_cloud):,} 3D points.{pc_stats_text}"
        if not self.no_output_vars:
            text += f" Use $depth_map (numpy array, {height}x{width}), $point_cloud (numpy array, {len(point_cloud)}x3) and $focal_length_px (float) to reference the results."
        
        return ToolResult(
            result_dict,  # The actual value for code execution
            text=text,
            image=depth_viz,  # Will be None if no_output_image is True
            variables=variables
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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
        import io
        
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
    def _resolve_image(image):
        """Turn Ray ObjectRef → PIL.Image if necessary."""

        if not isinstance(image, Image.Image):
            # Best-effort conversion for numpy arrays
            try:
                image = Image.fromarray(image)  # type: ignore[arg-type]
            except Exception:
                raise TypeError(
                    "Unsupported image type for DepthEstimatorTool; expected PIL.Image "
                    "or numpy array.")
        return image

    @staticmethod
    def _compute_point_cloud(depth_map: np.ndarray, 
                           focal_length_px: float,
                           width: int, 
                           height: int) -> np.ndarray:
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