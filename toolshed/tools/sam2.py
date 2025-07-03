# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
SAM2 segmentation tool.

This tool provides interactive instance segmentation powered by
Meta's **SAM-2** model.

Available methods:
1. segment_from_point(image, x, y): Segment the object at a single pixel
   coordinate.
2. segment_from_points(image, points): Segment using multiple foreground
   points.

NOTE: Requires the ``sam2`` package and its dependencies. Install via
``pip install -e .[sam2]``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import torch  # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

from toolshed.tools.base import BaseTool, tool_method
from toolshed.tool_result import ToolResult

logger = logging.getLogger(__name__)


class Sam2SegmentationTool(BaseTool):
    """Interactive instance segmentation using Facebook SAM-2."""

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small", device: Optional[str] = None, 
                 no_output_image: bool = False, no_output_vars: bool = False,
                 exclude_methods: list[str] = None, exclude_behavior: str = "warn"):
        """Create a ``Sam2SegmentationTool`` instance.

        Args:
            model_id: Hugging Face model ID to load. Defaults to
                ``"facebook/sam2.1-hiera-small"`` which offers a good
                speed/quality trade-off (~300 MB checkpoint).
            device: Torch device for inference (``"cuda"`` or ``"cpu"``). If
                *None*, CUDA is used when available.
            no_output_image: If True, suppress image outputs in ToolResults.
            no_output_vars: If True, suppress variable outputs in ToolResults.
            exclude_methods: List of method names to exclude from schema and optionally block at runtime.
            exclude_behavior: How to handle calls to excluded methods ("warn", "error", or "silent").
        """
        super().__init__(no_output_image=no_output_image, no_output_vars=no_output_vars,
                         exclude_methods=exclude_methods, exclude_behavior=exclude_behavior)

        # Choose device automatically if not provided.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        logger.info("Loading SAM2 model '%s' on %s …", model_id, self._device)
        self._predictor = SAM2ImagePredictor.from_pretrained(
            model_id, device=self._device, mask_threshold=0.15
        )
        logger.info("SAM2 model loaded successfully")

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------
    def get_name(self) -> str:  # noqa: D401 – simple return
        return "sam2"

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------
    @tool_method
    def segment_from_point(self, image, x: float, y: float) -> ToolResult[Dict[str, Any]]:
        """Segment the object at normalized coordinates.

        [[if:text]]Text output: Summary of segmentation including mask dimensions and IoU (Intersection over Union) confidence score.[[/if:text]]
        [[if:image]]Visual output: An overlay visualization showing the segmented object highlighted on the original image.[[/if:image]]
        [[if:vars]]Stored variables: $segmentation_mask: Boolean array of shape (H, W) where True indicates segmented pixels[[/if:vars]]

        Args:
            image (Image): Image to segment.
            x (float): X-coordinate in normalized range [0, 1].
            y (float): Y-coordinate in normalized range [0, 1].

        Returns:
            dict: Dictionary containing:
                - mask: Boolean numpy array of shape (H, W) where True indicates segmented pixels
                - iou_score: Predicted IoU score indicating segmentation quality (0-1)
        """
        try:
            img = self._resolve_image(image)
            width, height = img.size
            
            # Validate normalized coordinates
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f"Coordinates ({x}, {y}) must be in range [0, 1]")
            
            # Convert to pixel coordinates
            pixel_x = int(x * width)
            pixel_y = int(y * height)
            
            mask, ious = self._predict_from_points(img, [(pixel_x, pixel_y)])

            best_iou: float = float(np.max(ious))
            
            # Create overlay visualization only if not suppressed
            overlay_image = None if self.no_output_image else self._create_overlay_image(img, mask, points=[(pixel_x, pixel_y)])
            
            result_dict = {"mask": mask, "iou_score": best_iou}
            
            # Prepare variables only if not suppressed
            variables = {} if self.no_output_vars else {"segmentation_mask": mask}
            
            # Adjust text based on whether variables are suppressed
            text = f"Segmented object at normalized coords ({x:.3f}, {y:.3f}) with IoU score: {best_iou:.3f}."
            if not self.no_output_vars:
                text += f" Use $segmentation_mask (boolean array, {mask.shape[0]}x{mask.shape[1]}) to reference it."
            
            return ToolResult(
                result_dict,  # The actual value for code execution
                text=text,
                image=overlay_image,  # Will be None if no_output_image is True
                variables=variables
            )
        except Exception as exc:  # pragma: no cover
            logger.error(f"segment_from_point failed: {exc}")
            return ToolResult(
                None,
                text=f"Error: {exc}",
                is_error=True
            )

    @tool_method
    def segment_from_points(self, image, points: Sequence[Tuple[float, float]]) -> ToolResult[Dict[str, Any]]:
        """Segment an object using multiple normalized coordinates.

        [[if:text]]Text output: Summary of segmentation including mask dimensions and IoU scores for candidate masks.[[/if:text]]
        [[if:image]]Visual output: An overlay visualization showing the segmented object highlighted on the original image.[[/if:image]]
        [[if:vars]]Stored variables: $segmentation_mask: Boolean array of shape (H, W) where True indicates segmented pixels[[/if:vars]]

        Args:
            image (Image): Image to segment.
            points (Sequence[Tuple[float, float]]): Sequence of (x, y) normalized coordinates in [0, 1] range.

        Returns:
            dict: Dictionary containing:
                - mask: Boolean numpy array of shape (H, W) where True indicates segmented pixels
                - iou_scores: 1-D numpy array with IoU scores for candidate masks (best mask is selected)
        """
        if len(points) == 0:
            raise ValueError("At least one point must be provided")

        try:
            img = self._resolve_image(image)
            width, height = img.size
            
            # Validate normalized coordinates
            for i, (x, y) in enumerate(points):
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError(f"Point {i} ({x}, {y}) must be in range [0, 1]")
            
            # Convert to pixel coordinates
            pixel_points = [(int(x * width), int(y * height)) for x, y in points]
            
            mask, ious = self._predict_from_points(img, pixel_points)
            
            # Create overlay visualization only if not suppressed
            overlay_image = None if self.no_output_image else self._create_overlay_image(img, mask, points=pixel_points)
            
            result_dict = {"mask": mask, "iou_scores": ious}
            best_iou = float(np.max(ious))
            
            # Prepare variables only if not suppressed
            variables = {} if self.no_output_vars else {"segmentation_mask": mask}
            
            # Adjust text based on whether variables are suppressed
            text = f"Segmented object using {len(points)} normalized points with best IoU score: {best_iou:.3f}."
            if not self.no_output_vars:
                text += f" Use $segmentation_mask (boolean array, {mask.shape[0]}x{mask.shape[1]}) to reference it."
            
            return ToolResult(
                result_dict,  # The actual value for code execution
                text=text,
                image=overlay_image,  # Will be None if no_output_image is True
                variables=variables
            )
        except Exception as exc:  # pragma: no cover
            logger.error(f"segment_from_points failed: {exc}")
            return ToolResult(
                None,
                text=f"Error: {exc}",
                is_error=True
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_image(image):
        """Convert numpy arrays → PIL.Image."""
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover
                raise TypeError(
                    "Unsupported image type for Sam2SegmentationTool; expected PIL.Image "
                    "or numpy array."
                ) from exc
        return image

    @staticmethod
    def _create_overlay_image(
        image: Image.Image, 
        mask: np.ndarray, 
        points: Optional[Sequence[Tuple[int, int]]] = None,
        mask_color: Tuple[int, int, int] = (0, 255, 0),
        mask_alpha: float = 0.5,
        point_color: Tuple[int, int, int] = (255, 0, 0),
        outline_width_pct: float = 0.005,  # 0.5% of image width
        point_radius_pct: float = 0.005   # 0.5% of image width
    ) -> Image.Image:
        """Create an overlay image with the segmentation mask and optional points.
        
        All annotation sizes are normalized relative to image dimensions for consistency.
        
        Args:
            image: Original PIL image
            mask: Boolean mask array
            points: Optional list of (x, y) coordinates to mark
            mask_color: RGB color for the mask overlay
            mask_alpha: Transparency of the mask overlay (0-1)
            point_color: RGB color for the point markers
            outline_width_pct: Outline width as percentage of image width (default: 1%)
            point_radius_pct: Point marker radius as percentage of image width (default: 0.5%)
            
        Returns:
            PIL Image with mask overlay and point markers
        """
        # Calculate actual pixel sizes from percentages
        # Use image width as reference for both to maintain aspect ratio
        outline_width = max(2, int(image.width * outline_width_pct))
        point_radius = max(3, int(image.width * point_radius_pct))
        
        # Convert to RGBA for transparency support
        overlay = image.convert("RGBA")
        
        # Create a transparent layer for the mask
        mask_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        
        # Convert boolean mask to RGBA with the specified color and alpha
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[mask] = (*mask_color, int(255 * mask_alpha))
        
        # Create PIL Image from the mask
        mask_img = Image.fromarray(mask_rgba, mode="RGBA")
        
        # Composite the mask onto the transparent layer
        mask_layer.paste(mask_img, (0, 0))
        
        # Composite the mask layer onto the original image
        result = Image.alpha_composite(overlay, mask_layer)
        
        # Add white outline to the mask
        
        # Use scipy for morphological operations with circular kernel
        from scipy import ndimage
        
        # Create a circular structuring element for even dilation
        radius = outline_width
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        struct = x*x + y*y <= radius*radius
        
        # Dilate the mask to get outer boundary
        dilated = ndimage.binary_dilation(mask, structure=struct)
        
        # The outline is the dilated region minus the original mask
        outline = dilated & ~mask
        
        # Create RGBA image for the outline
        outline_rgba_array = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        outline_rgba_array[outline] = (255, 255, 255, 255)  # White, fully opaque
        outline_rgba = Image.fromarray(outline_rgba_array, mode="RGBA")
        
        # Composite the outline onto the result
        result = Image.alpha_composite(result, outline_rgba)
        
        # Draw point markers if provided
        if points:
            draw = ImageDraw.Draw(result)
            # Point outline width scales with point size (about 20% of radius)
            point_outline_width = max(1, int(point_radius * 0.2))
            
            for x, y in points:
                # Draw a circle at each point
                draw.ellipse(
                    [x - point_radius, y - point_radius, 
                     x + point_radius, y + point_radius],
                    fill=point_color,
                    outline=(255, 255, 255),  # White outline for visibility
                    width=point_outline_width
                )
        
        # Convert back to RGB (removing alpha channel)
        return result.convert("RGB")

    def _predict_from_points(
        self, image: Image.Image, points_xy: Sequence[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run SAM2 to obtain a segmentation mask from *points_xy*."""
        # Convert image to numpy array (RGB) – predictor expects HWC uint8.
        img_array = np.array(image.convert("RGB"))

        # Compute (if not already cached) image embeddings
        # Note: the predictor caches the last image – but to keep the logic
        # simple and thread-safe we always set the image.
        self._predictor.set_image(img_array)

        # Prepare point tensors (N×2) and labels (foreground=1)
        point_coords = np.array(points_xy, dtype=np.float32)  # [[x, y], ...]
        point_labels = np.ones(len(points_xy), dtype=np.int32)

        # Model inference – disable multi-mask to return a single candidate.
        masks, ious, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,  # let model generate multiple; we'll pick best
            normalize_coords=True,  # convert absolute pixel coords → normalized before inference
            return_logits=False,
        )

        # Select the mask with highest predicted IoU.
        best_idx = int(np.argmax(ious))
        best_mask = masks[best_idx]
        best_iou = float(ious[best_idx])

        # Convert to boolean for cleaner downstream usage
        best_mask_bool = best_mask.astype(bool)

        # Erode the mask by 5px to avoid edge effects
        best_mask_bool = cv2.erode(best_mask_bool.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool)

        return best_mask_bool, ious 