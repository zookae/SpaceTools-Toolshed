# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Image operations tool
=====================

This tool provides basic image manipulation operations:

1. point_crop(image, points): Crop image to minimally encompass given normalized points
2. mask_crop(image, mask): Crop image to masked region with white background for non-masked areas

The tool handles Ray ObjectRefs for efficient distributed processing.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple, Union

import numpy as np
from PIL import Image
from PIL import ImageDraw
from toolshed.tools.base import BaseTool, tool_method
from toolshed.tool_result import ToolResult
from collections import defaultdict

logger = logging.getLogger(__name__)


class VisionOpsTool(BaseTool):
    """Computer vision operations tool for basic image and 3D vision manipulation."""

    def __init__(self, no_output_image: bool = False, no_output_vars: bool = False,
                 exclude_methods: list[str] = None, exclude_behavior: str = "warn"):
        """Initialize the VisionOpsTool.
        
        Args:
            no_output_image: If True, suppress image outputs in ToolResults.
            no_output_vars: If True, suppress variable outputs in ToolResults.
            exclude_methods: List of method names to exclude from schema and optionally block at runtime.
            exclude_behavior: How to handle calls to excluded methods ("warn", "error", or "silent").
        """
        super().__init__(no_output_image=no_output_image, no_output_vars=no_output_vars, 
                         exclude_methods=exclude_methods, exclude_behavior=exclude_behavior)

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------
    def get_name(self) -> str:
        return "vision_ops"

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------
    
    @tool_method
    def project_3d_points_to_2d(self, points: np.ndarray, focal_length_px: float, width: int, height: int) -> ToolResult[np.ndarray]:
        """Project 3D points to 2D image coordinates.
        
        [[if:text]]Text output: String representation of projected 2D points.[[/if:text]]
        [[if:vars]]Stored variables: The projected 2D points in $projected_2d_points variable (numpy array of shape (N, 2)).[[/if:vars]]
        
        Args:
            points: Numpy array or nested list of shape (N, 3) containing x, y, z coordinates
            focal_length_px: Focal length of the camera in pixels
            width: Width of the image in pixels
            height: Height of the image in pixels

        Returns:
            Numpy array of shape (N, 2) containing x, y coordinates
        """
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be a 2D array of shape (N, 3)")
        intrinsic_matrix = np.array([[focal_length_px, 0, width / 2], [0, focal_length_px, height / 2], [0, 0, 1]])
        projected_points = intrinsic_matrix @ points.T
        projected_points = projected_points[:2, :] / projected_points[2, :]

        return ToolResult(
            projected_points, 
            text=f"Projected {len(points)} 3D points to 2D image coordinates.",
            variables={"projected_2d_points": projected_points}
        )
    
    @tool_method
    def project_2d_points_to_3d(self, points: np.ndarray, point_depth: np.ndarray, focal_length_px: float, width: int, height: int) -> ToolResult[np.ndarray]:
        """Project 2D points to 3D coordinates.
        
        [[if:text]]Text output: String representation of projected 3D points.[[/if:text]]
        [[if:vars]]Stored variables: The projected 3D points in $projected_3d_points variable (numpy array of shape (N, 3)).[[/if:vars]]
        
        Args:
            points: Numpy array or nested list of shape (N, 2) containing x, y coordinates
            point_depth: Numpy array or list of shape (N, ) containing depth values for each point
            focal_length_px: Focal length of the camera in pixels
            width: Width of the image in pixels
            height: Height of the image in pixels

        Returns:
            Numpy array of shape (N, 3) containing x, y, z coordinates
        """
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must be a 2D array of shape (N, 2)")
        point_depth = np.asarray(point_depth)
        if point_depth.ndim != 1 or point_depth.shape[0] != len(points):
            raise ValueError("Point depth must be a 1D array of shape (N,)")
        if not np.all(point_depth > 0):
            raise ValueError("Point depth must be positive")
        intrinsic_matrix = np.array([
            [focal_length_px, 0, width / 2], 
            [0, focal_length_px, height / 2], 
            [0, 0, 1]])
        homogeneous_points = np.concatenate((points, np.ones((len(points), 1))), axis=1)
        homogeneous_points = homogeneous_points.T
        # Unproject to get normalized rays first
        normalized_rays = np.linalg.inv(intrinsic_matrix) @ homogeneous_points
        # Then scale by depth to get 3D points
        points_3d = normalized_rays * point_depth.reshape(1, -1)
        points_3d = points_3d.T

        return ToolResult(
            points_3d,
            text=f"Projected {len(points)} 2D points to 3D coordinates.",
            variables={"projected_3d_points": points_3d}
        )

    @tool_method
    def index_at(self, data: Any, u: float, v: float) -> ToolResult[Union[Any, str]]:
        """Get the pixel value in the numpy ndarray `data` at the given normalized coordinates (u,v).
        Note the input arguments are called `data`, `u`, `v`.

        [[if:text]]Text output: Information about the pixel value at the given coordinates.[[/if:text]]
        
        Args:
            data: Numpy ndarray of shape (H, W) or (H, W, C), or PIL Image
            u: Normalized x-coordinate in [0, 1]
            v: Normalized y-coordinate in [0, 1]
        """
        # [[if:vars]]Stored variables: The pixel value at the given coordinates in $index_at_pixel_value variable.[[/if:vars]]

        array = self._resolve_2d_array(data)
        height, width = array.shape[:2]
        pixel_x = int(u * width)
        pixel_y = int(v * height)
        pixel_value = array[pixel_y, pixel_x]
        
        # Convert numpy types to native Python types for better serialization
        if isinstance(pixel_value, np.ndarray):
            pixel_value = pixel_value.tolist()
        elif isinstance(pixel_value, (np.integer, np.floating)):
            pixel_value = pixel_value.item()
        
        return ToolResult(
            pixel_value,
            text=f"Pixel value at ({u:.3f}, {v:.3f}) is {pixel_value}"
            #variables={"index_at_pixel_value": pixel_value}
        )

    @tool_method
    def point_crop(self, image: Any, points: List[Tuple[float, float]]) -> ToolResult[Union[Any, str]]:
        """Crop image to minimally encompass all given points.

        [[if:text]]Text output: Information about the crop operation including pixel coordinates and resulting dimensions.[[/if:text]]
        [[if:image]]Visual output: The cropped region of the image containing all specified points.[[/if:image]]
        [[if:vars]]Stored variables: The cropped image in $cropped_image variable.[[/if:vars]]
        
        Args:
            image (Image): Image to crop.
            points (List[Tuple[float, float]]): List of (x, y) normalized coordinates in [0, 1] range.
                                                Example: [(0.2, 0.3), (0.8, 0.7)] for two points.

        Returns:
            ToolResult: Cropped image on success, or ToolResult with is_error=True on failure.
        """
        try:
            # Resolve image
            img = self._resolve_image(image)
            width, height = img.size
            
            # Validate points
            if not points:
                error_msg = "No points provided for cropping"
                return ToolResult(
                    None,
                    text=f"Error: {error_msg}",
                    is_error=True
                )
            
            # Validate point ranges
            for i, (x, y) in enumerate(points):
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    error_msg = f"Point {i} ({x}, {y}) is outside normalized range [0, 1]"
                    return ToolResult(
                        None,
                        text=f"Error: {error_msg}",
                        is_error=True
                    )
            
            # Convert normalized points to pixel coordinates
            pixel_points = [(int(x * width), int(y * height)) for x, y in points]
            
            # Find bounding box
            x_coords = [p[0] for p in pixel_points]
            y_coords = [p[1] for p in pixel_points]
            
            x_min = max(0, min(x_coords))
            x_max = min(width, max(x_coords))
            y_min = max(0, min(y_coords))
            y_max = min(height, max(y_coords))
            
            # Check for zero-dimension crop
            if x_min >= x_max:
                error_msg = (f"Crop has zero width: all points have the same x-coordinate "
                           f"(x={x_min} in pixel space)")
                return ToolResult(
                    None,
                    text=f"Error: {error_msg}",
                    is_error=True
                )
            
            if y_min >= y_max:
                error_msg = (f"Crop has zero height: all points have the same y-coordinate "
                           f"(y={y_min} in pixel space)")
                return ToolResult(
                    None,
                    text=f"Error: {error_msg}",
                    is_error=True
                )
            
            # Perform the crop
            cropped = img.crop((x_min, y_min, x_max, y_max))
            
            # Update statistics
            
            # Prepare result
            crop_info = (f"Cropped image from ({x_min}, {y_min}) to ({x_max}, {y_max}) "
                        f"[size: {x_max - x_min}x{y_max - y_min}] to encompass {len(points)} points")
            
            # Create variables if not suppressed
            variables = {} if self.no_output_vars else {"cropped_image": cropped}
            
            # Prepare image output if not suppressed
            output_image = None if self.no_output_image else cropped
            
            return ToolResult(
                cropped,  # The actual value for code execution
                text=crop_info + (f" Use $cropped_image (PIL Image, {cropped.width}x{cropped.height}) to reference it." if not self.no_output_vars else ""),
                image=output_image,
                variables=variables
            )
            
        except Exception as exc:
            logger.error(f"point_crop failed: {exc}")
            return ToolResult(
                None,
                text=f"Error: {exc}",
                is_error=True
            )

    @tool_method
    def mask_crop(self, image: Any, mask: Any) -> ToolResult[Union[Any, str]]:
        """Crop image to the masked region with white background for non-masked areas.

        [[if:text]]Text output: Information about the mask crop operation including bounding box coordinates and dimensions.[[/if:text]]
        [[if:image]]Visual output: The cropped image containing only the masked region, with white background for non-masked areas.[[/if:image]]
        [[if:vars]]Stored variables: The masked and cropped image $masked_crop.[[/if:vars]]
        
        Args:
            image (Image): Image to crop.
            mask (Array): Boolean segmentation mask where True indicates pixels to keep.
                         Shape should match image dimensions.

        Returns:
            ToolResult: Masked and cropped image on success, or ToolResult with is_error=True on failure.
        """
        try:
            # Resolve image and mask
            img = self._resolve_image(image)
            mask_array = self._resolve_mask(mask)
            
            # Validate mask dimensions
            if mask_array.shape[:2] != (img.height, img.width):
                error_msg = (f"Mask shape {mask_array.shape[:2]} doesn't match "
                           f"image dimensions ({img.height}, {img.width})")
                return ToolResult(
                    None,
                    text=f"Error: {error_msg}",
                    is_error=True
                )
            
            # Find bounding box of the mask
            rows = np.any(mask_array, axis=1)
            cols = np.any(mask_array, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                error_msg = "Mask is empty - no pixels are masked"
                return ToolResult(
                    None,
                    text=f"Error: {error_msg}",
                    is_error=True
                )
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Add 1 to max values since crop is exclusive on the right/bottom
            x_max += 1
            y_max += 1
            
            # Crop the image and mask
            cropped_img = img.crop((x_min, y_min, x_max, y_max))
            cropped_mask = mask_array[y_min:y_max, x_min:x_max]
            
            # Create white background
            result = Image.new("RGB", cropped_img.size, (255, 255, 255))
            
            # Convert cropped image to RGB if necessary
            if cropped_img.mode != "RGB":
                cropped_img = cropped_img.convert("RGB")
            
            # Apply mask: copy masked pixels from original, leave rest white
            img_array = np.array(cropped_img)
            result_array = np.array(result)
            
            # Apply mask to each channel
            for c in range(3):
                result_array[:, :, c] = np.where(
                    cropped_mask,
                    img_array[:, :, c],
                    255  # White background
                )
            
            # Convert back to PIL Image
            result_img = Image.fromarray(result_array)
            
            # Update statistics
            
            # Calculate mask coverage
            mask_pixels = np.sum(cropped_mask)
            total_pixels = cropped_mask.size
            coverage_pct = (mask_pixels / total_pixels) * 100
            
            # Prepare result
            crop_info = (f"Cropped masked region from ({x_min}, {y_min}) to ({x_max}, {y_max}) "
                        f"[size: {x_max - x_min}x{y_max - y_min}], "
                        f"mask coverage: {coverage_pct:.1f}%")
            
            # Create variables if not suppressed
            variables = {} if self.no_output_vars else {"masked_crop": result_img}
            
            # Prepare image output if not suppressed
            output_image = None if self.no_output_image else result_img
            
            return ToolResult(
                result_img,  # The actual value for code execution
                text=crop_info + (f" Use $masked_crop (PIL Image, {result_img.width}x{result_img.height}) to reference it." if not self.no_output_vars else ""),
                image=output_image,
                variables=variables
            )
            
        except Exception as exc:
            logger.error(f"mask_crop failed: {exc}")
            return ToolResult(
                None,
                text=f"Error: {exc}",
                is_error=True
            )

    @tool_method(requires_image_output=True)
    def draw_polygon(self, image: Any, points: List[Tuple[float, float]]) -> ToolResult[Union[Any, str]]:
        """Draw a polygon on the image.
        
        [[if:text]]Text output: Information about the polygon drawing operation including bounding box coordinates and dimensions.[[/if:text]]
        [[if:image]]Visual output: The image with the polygon drawn on it.[[/if:image]]
        [[if:vars]]Stored variables: The image with the polygon drawn on it in $drawn_polygon variable.[[/if:vars]]
        
        Args:
            image (Image): Image to draw the polygon on.
            points (List[Tuple[float, float]]): List of (x, y) normalized coordinates in [0, 1] range.

        Returns:
            ToolResult: Image with the polygon drawn on it on success, or ToolResult with is_error=True on failure.
        """
        try:
            # Resolve image
            img = self._resolve_image(image)
            width, height = img.size

            # Validate points
            if not points:
                error_msg = "No points provided for drawing polygon"
                return ToolResult(
                    None,
                    text=f"Error: {error_msg}",
                    is_error=True
                )
            
            # Validate point ranges
            for i, (x, y) in enumerate(points):
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    error_msg = f"Point {i} ({x}, {y}) is outside normalized range [0, 1]"
                    return ToolResult(
                        None,
                        text=f"Error: {error_msg}",
                        is_error=True
                    )
            
            # Convert normalized points to pixel coordinates
            pixel_points = [(int(x * width), int(y * height)) for x, y in points]
            
            # Draw polygon
            draw = ImageDraw.Draw(img)
            draw.polygon(pixel_points, fill=None, outline=(255, 0, 0), width=2)
            
            # Update statistics
            
            # Prepare result
            polygon_info = (f"Drawn polygon on image from {len(points)} points")
            
            # Create variables if not suppressed
            variables = {} if self.no_output_vars else {"drawn_polygon": img}
            
            # Prepare image output if not suppressed
            output_image = None if self.no_output_image else img
            
            return ToolResult(
                img,  # The actual value for code execution
                text=polygon_info + (f" Use $drawn_polygon (PIL Image, {img.width}x{img.height}) to reference it." if not self.no_output_vars else ""),
                image=output_image,
                variables=variables
            )
            
        except Exception as exc:
            logger.error(f"draw_polygon failed: {exc}")
            return ToolResult(
                None,
                text=f"Error: {exc}",
                is_error=True
            )
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_2d_array(data):
        """Convert PIL Images or arrays to numpy array.
        
        Accepts PIL Images or numpy arrays with shape (H, W) or (H, W, C).
        Returns a numpy array with validated shape.
        """
        # Convert PIL Image to numpy array
        if isinstance(data, Image.Image):
            data = np.array(data)
        # Validate numpy array
        elif isinstance(data, np.ndarray):
            if data.ndim not in [2, 3]:
                raise ValueError(
                    f"Array must be 2D (H, W) or 3D (H, W, C), got shape {data.shape}"
                )
        else:
            raise TypeError(
                f"Expected PIL Image or numpy array, got {type(data)}"
            )
        
        return data
    
    @staticmethod
    def _resolve_image(image):
        """Convert to PIL.Image if necessary."""
        if not isinstance(image, Image.Image):
            try:
                # Try to convert numpy array or other format to PIL Image
                image = Image.fromarray(image)  # type: ignore[arg-type]
            except Exception:
                raise TypeError(
                    "Unsupported image type for ImageOpsTool; expected PIL.Image "
                    "or numpy array."
                )
        return image
    
    @staticmethod
    def _resolve_mask(mask):
        """Convert to numpy array if necessary."""
        if not isinstance(mask, np.ndarray):
            try:
                mask = np.array(mask)
            except Exception:
                raise TypeError(
                    "Unsupported mask type for ImageOpsTool; expected numpy array "
                    "or PIL Image."
                )
        
        # Ensure boolean mask
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        return mask














