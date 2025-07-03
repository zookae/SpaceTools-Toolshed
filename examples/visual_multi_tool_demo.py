#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""visual_multi_tool_demo.py

End-to-end example showing how multiple *toolshed* vision tools can be
combined to answer the question "**How far is the lemon?**".

Pipeline
--------
1. Use the Vision-Language Model (``vlm``) tool to find pixel
   coordinates corresponding to *lemon*.
2. Feed that coordinate into the ``sam2`` segmentation tool to obtain a
   precise object mask.
3. Run the ``depth_estimator`` to get a per-pixel depth map.
4. Average depth values within the lemon mask and print the estimated
   distance.

Requirements
~~~~~~~~~~~~
• Working GPU(s) and the extras for each tool installed: ``vlm``,
  ``sam2``, and ``depth``.
• The example image should contain a lemon; you can pass an explicit path
  via ``--image``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add repo root to path so the script works when executed directly
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from toolshed import start_toolkit, get_toolkit, shutdown_toolkit  # noqa: E402

DEFAULT_IMAGE = THIS_DIR / "media" / "kitchen.png"


# -----------------------------------------------------------------------------
# Auxiliary visual helpers
# -----------------------------------------------------------------------------

def overlay_mask(img: Image.Image, mask: np.ndarray, colour: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.4) -> Image.Image:
    """Blend *mask* onto *img* and return an RGB image."""
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    # Apply colour to masked pixels
    # Convert to a writable NumPy array – np.asarray may return a read-only view.
    arr = np.asarray(overlay).copy()
    arr[mask] = [*colour, int(alpha * 255)]
    overlay = Image.fromarray(arr, "RGBA")

    return Image.alpha_composite(rgba, overlay).convert("RGB")


# -----------------------------------------------------------------------------
# Main demo logic
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: D401 – script entry
    parser = argparse.ArgumentParser(description="Visual multi-tool demo – How far is the lemon?")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE), help="Path to the input image.")
    parser.add_argument("--object", type=str, default="leftmost peach", help="Object to detect.")
    parser.add_argument("--config", type=str, default="configs/vision_minimal.json", help="Path to the tool configuration file.")
    parser.add_argument("--show", action="store_true", help="Display the visualisation in a GUI window (in addition to saving it).")
    args = parser.parse_args(argv)

    img_path = Path(args.image)
    if not img_path.exists():
        parser.error(f"Image file '{img_path}' not found.")

    # ------------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------------
    output_dir = Path("outputs") / "visual_multi_tool_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(img_path).convert("RGB")

    # ------------------------------------------------------------------
    # 1. Launch toolkit with required tools (1 GPU each)
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file '{config_path}' not found.")
    
    with open(config_path, "r") as f:
        tool_configs = json.load(f)

    print("Starting toolshed cluster … (this may take a while on first run)")
    # Keep the handle alive. Once it gets garbage collected, the toolkit will automatically shut down.
    handle = start_toolkit(tool_configs, detached=False)
    toolkit = get_toolkit()

    try:
        # ------------------------------------------------------------------
        # 2. Use VLM to point at lemons
        # ------------------------------------------------------------------
        print("\nDetecting object with VLM …")
        points = toolkit.vlm.detect_all(image, args.object).value
        if not points:
            raise RuntimeError(f"VLM did not return any points for '{args.object}'.")

        # Use the first point returned (taking integer pixel coords)
        click_x, click_y = map(int, points[0])
        print(f"VLM pointed at object pixel: ({click_x}, {click_y})")

        # ------------------------------------------------------------------
        # 3. Segment using SAM2
        # ------------------------------------------------------------------
        print("Segmenting object with SAM2 …")
        seg_result = toolkit.sam2.segment_from_point(image, x=click_x, y=click_y)
        if seg_result.is_error:
            raise RuntimeError(f"SAM2 error: {seg_result.text}")
        lemon_mask: np.ndarray = seg_result.value["mask"]  # boolean H×W
        print("SAM2 segmentation done – mask obtained.")

        # Save segmentation mask (binary PNG and NumPy array)
        mask_img = Image.fromarray((lemon_mask.astype(np.uint8) * 255))
        mask_img_path = output_dir / "object_mask.png"
        mask_img.save(mask_img_path)
        np.save(output_dir / "object_mask.npy", lemon_mask)
        print(f"Saved object mask to {mask_img_path}")

        # ------------------------------------------------------------------
        # 4. Estimate depth map
        # ------------------------------------------------------------------
        print("Estimating depth for entire image …")
        depth_result = toolkit.depth_estimator.estimate_depth(image).value
        depth_map: np.ndarray = depth_result["depth_map"]  # 2-D float array (meters)
        print("Depth estimation complete.")

        # Save depth map (raw NumPy array and colour-mapped PNG)
        np.save(output_dir / "depth_map.npy", depth_map)
        # Normalise depth for visualisation
        norm_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        plt.imsave(output_dir / "depth_map.png", norm_depth, cmap="plasma")
        print(f"Saved depth map to {output_dir / 'depth_map.npy'} and visualisation to {output_dir / 'depth_map.png'}")

        # ------------------------------------------------------------------
        # 5. Compute average depth for lemon
        # ------------------------------------------------------------------
        lemon_depth_values = depth_map[lemon_mask]
        if lemon_depth_values.size == 0:
            raise RuntimeError("Mask has zero area – segmentation failed.")
        avg_depth_m = float(np.mean(lemon_depth_values))
        print(f"\nEstimated distance to {args.object}: {avg_depth_m:.2f} m")

        # ------------------------------------------------------------------
        # 6. Visualise
        # ------------------------------------------------------------------
        vis_img = overlay_mask(image, lemon_mask)
        # Always save overlay visualisation
        overlay_path = output_dir / "overlay.png"
        vis_img.save(overlay_path)
        print(f"Saved overlay visualisation to {overlay_path}")

        print(f"\Final answer: distance to {args.object} is {avg_depth_m:.2f} m")

        # Optionally show interactive figure when --show flag is provided
        if args.show:
            plt.figure(figsize=(6, 6))
            plt.imshow(vis_img)
            plt.title(f"{args.object} ≈ {avg_depth_m:.2f} m away")
            plt.axis("off")
            plt.show()

    finally:
        print("\nShutting down toolkit …")
        shutdown_toolkit()
        print("Done.")


if __name__ == "__main__":
    main() 
