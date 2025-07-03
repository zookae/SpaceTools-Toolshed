#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Example usage of the SAM-2 segmentation tool in *toolshed*.

This script shows how to:
1. Launch the toolkit with the SAM-2 tool enabled
2. Perform a segmentation given a click on the image centre
3. Save the overlay image directly from the tool's output
4. Access the raw mask data from ToolResult variables

NOTE: The first run will download the SAM-2 checkpoint (~300 MB) from the
HuggingFace Hub – make sure you have a stable internet connection.
"""

import os
import sys
from typing import Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ensure the repo root is on the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toolshed import start_toolkit, get_toolkit, shutdown_toolkit  # noqa: E402
from toolshed.tool_result import ToolResult  # noqa: E402


def overlay_mask(img: Image.Image, mask: np.ndarray, colour: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5) -> Image.Image:
    """Return *img* with *mask* blended on top (RGB).
    
    NOTE: This function is kept for demonstration purposes, but the SAM2 tool
    now provides its own overlay visualization in the ToolResult.
    """
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    overlay_np = np.array(overlay)
    overlay_np[mask] = [*colour, int(alpha * 255)]
    overlay = Image.fromarray(overlay_np, mode="RGBA")

    blended = Image.alpha_composite(rgba, overlay)
    return blended.convert("RGB")


def main():
    """Run the example."""
    # Configure the SAM-2 tool – request a GPU if available for speed.
    tool_configs = {
        "sam2": {
            "num_actors": 1,
            "resources": {"num_gpus": 1},
            "conda_env": "tool_sam2",
            "timeout": 120,  # allow time for weight download
        }
    }

    print("Starting toolshed (this may take a minute on first run)…")
    handle = start_toolkit(tool_configs, detached=False)
    toolkit = get_toolkit()

    try:
        # Load an example image (a 512×512 kitchen photo shipped with the repo)
        img_path = os.path.join(os.path.dirname(__file__), "media", "kitchen.png")
        image = Image.open(img_path)
        w, h = image.size

        # Click in the peach
        click_x, click_y = 384, 845
        print(f"Segmenting pixel at ({click_x}, {click_y})…")

        # The tool returns a ToolResult object when called through the toolkit
        result = toolkit.sam2.segment_from_point(image, x=click_x, y=click_y)
        
        # Check for errors
        if result.is_error:
            raise RuntimeError(f"SAM2 tool call failed: {result.text}")
        
        # Access the mask and IoU score from the value
        mask = result.value["mask"]
        iou_score = result.value["iou_score"]
        
        # Save the overlay image if available
        if result.image:
            overlay_image = result.image[0]
            overlay_path = "outputs/sam2_overlay_from_toolkit.png"
            os.makedirs("outputs", exist_ok=True)
            overlay_image.save(overlay_path)
            print(f"Saved tool-generated overlay to '{overlay_path}'")
            
        print(f"Predicted mask with IoU ≈ {iou_score:.3f}")
        
        # Demonstrate accessing ToolResult properties
        if isinstance(result, ToolResult):
            print("\nToolResult properties:")
            print(f"  Text message: {result.text}")
            print(f"  Has overlay image: {len(result.image) > 0}")
            
            # The mask is also available in variables
            if "segmentation_mask" in result.variables:
                mask_from_vars = result.variables["segmentation_mask"]
                print(f"  Mask available in variables: shape {mask_from_vars.shape}")

        # Also create a custom overlay using the original helper function for comparison
        custom_vis = overlay_mask(image, mask, colour=(255, 0, 0), alpha=0.3)  # Red with different alpha
        custom_vis_path = "outputs/sam2_custom_overlay.png"
        os.makedirs("outputs", exist_ok=True)
        custom_vis.save(custom_vis_path)
        print(f"\nSaved custom visualization to '{custom_vis_path}'")
        
        # Test multi-point segmentation
        print("\nTesting multi-point segmentation:")
        points = [(384, 845), (400, 850), (370, 840)]  # Multiple points on the peach
        result_multi = toolkit.sam2.segment_from_points(image, points=points)
        
        # Check for errors
        if result_multi.is_error:
            raise RuntimeError(f"SAM2 multi-point failed: {result_multi.text}")
        
        mask_multi = result_multi.value["mask"]
        iou_scores = result_multi.value["iou_scores"]
        print(f"  Multi-point segmentation IoU scores: {iou_scores}")
        
        # Save the overlay from ToolResult if available
        if result_multi.image:
            overlay_multi = result_multi.image[0]
            overlay_multi_path = "outputs/sam2_multipoint_overlay.png"
            overlay_multi.save(overlay_multi_path)
            print(f"  Saved multi-point overlay to '{overlay_multi_path}'")

        # Show stats
        print("\nTool statistics:")
        print(toolkit.sam2.get_stats())

    finally:
        print("\nShutting down toolkit…")
        shutdown_toolkit()
        print("Done.")


if __name__ == "__main__":
    main() 
