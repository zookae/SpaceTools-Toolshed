#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""visual_reasoning_scaling_demo.py

Demonstration of multi-GPU, multi-tool visual reasoning in *toolshed*.

The script automatically leverages eight GPUs (3× VLM, 3× depth estimation, 2×
SAM2 segmentation) to analyse an input image.  It proceeds in three stages:

1. Query a Vision-Language Model (VLM) to list the most prominent objects in
   the scene.
2. For each object, run instance segmentation (SAM2) and monocular depth
   estimation in *parallel* across the dedicated actor pools.  The depth of
   each object is computed as the mean depth within its mask.
3. Produce a single annotated image showing all objects with coloured masks,
   bounding boxes, labels, and their estimated distances.

The workload is pipelined via Ray so that all actors remain busy – VLM actors
find objects, SAM2 actors segment them, and depth actors evaluate depth maps
concurrently.

Run with:  python visual_reasoning_scaling_demo.py --image path/to/img.png
"""

from __future__ import annotations

import argparse
import ast
import random
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ray
from PIL import Image, ImageDraw


# -----------------------------------------------------------------------------
# Toolbox import (relative – works when executed directly)
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(THIS_DIR.parent))
from toolshed import start_toolkit, get_toolkit, shutdown_toolkit  # noqa: E402

DEFAULT_IMAGE = THIS_DIR / "media" / "kitchen.png"

# -----------------------------------------------------------------------------
# Visual helpers
# -----------------------------------------------------------------------------

def overlay_mask(img: Image.Image, mask: np.ndarray, colour: Tuple[int, int, int], alpha: float = 0.4) -> Image.Image:  # noqa: D401 – helper
    """Return *img* blended with *mask* in *colour* (RGB tuple)."""
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    arr = np.asarray(overlay).copy()
    arr[mask] = [*colour, int(alpha * 255)]
    overlay = Image.fromarray(arr, "RGBA")

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def random_colour() -> Tuple[int, int, int]:
    """Generate a visually distinct RGB colour."""
    return tuple(int(c * 255) for c in plt.cm.get_cmap("tab10")(random.randint(0, 9))[:3])

# -----------------------------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:  # noqa: D401 – script entry
    parser = argparse.ArgumentParser(description="Visual reasoning scaling demo – multi-GPU pipeline")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE), help="Path to input image")
    parser.add_argument("--max_objects", type=int, default=10, help="Maximum number of objects to process")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file for tool configurations. If provided, overrides the inline tool configs (including conda_env names).")
    parser.add_argument("--show", action="store_true", help="Display the annotated image interactively")
    args = parser.parse_args(argv)

    img_path = Path(args.image)
    if not img_path.exists():
        parser.error(f"Image file '{img_path}' not found.")

    image = Image.open(img_path).convert("RGB")

    # ------------------------------------------------------------------
    # Configure and launch toolkit
    # ------------------------------------------------------------------
    tool_configs = {
        "vlm": {
            "num_actors": 3,
            "resources": {"num_gpus": 1},
            "conda_env": "tool_vlm",
            "timeout": 600,
        },
        "sam2": {
            "num_actors": 2,
            "resources": {"num_gpus": 1},
            "conda_env": "tool_sam2",
            "timeout": 300,
        },
        "depth_estimator": {
            "num_actors": 3,
            "resources": {"num_gpus": 1},
            "conda_env": "tool_depth",
            "timeout": 120,
            "args": {
                "checkpoint_path": "/lustre/fsw/portfolios/nvr/users/vblukis/checkpoints/depth_pro.pt",
            },
        },
    }

    # Override with JSON config if provided
    if args.config is not None:
        import json
        config_path = Path(args.config)
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            tool_configs = json.load(f)

    print("Starting toolshed cluster … (initial model downloads may take a while)")
    handle = start_toolkit(tool_configs, detached=False, dashboard=True, dashboard_port=7001)
    toolkit = get_toolkit()

    # Handy reference for manual, non-blocking calls
    router = handle  # same object; clearer name for below

    try:
        # ------------------------------------------------------------------
        # 1. Ask VLM for prominent objects
        # ------------------------------------------------------------------
        prompt = (
            "Please first describe the image. "
            "Then write a python list of strings containing the names of the prominent objects. "
            "Put your list inside a ```python``` block that should evaluate to a list of strings."
        )
        response = toolkit.vlm.general_query(image, prompt)
        print("VLM object list response:\n", response)

        # Extract the actual string content from ToolResult
        response_text = response.value if hasattr(response, 'value') else str(response)

        # ------------------------------------------------------------------
        # Parse the python list inside a ```python ...``` code block returned
        # by the VLM.  The block may contain an assignment, so we extract the
        # first list literal we encounter and literal_eval it.
        # ------------------------------------------------------------------
        code_block_match = re.search(r"```python(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
        if not code_block_match:
            code_block_match = re.search(r"```(.*?)```", response_text, re.DOTALL)

        if not code_block_match:
            raise RuntimeError("Failed to find a python code block in VLM response")

        code_content = code_block_match.group(1)

        # Attempt to locate a list literal within the code block
        list_match = re.search(r"\[[\s\S]*?\]", code_content)
        if not list_match:
            raise RuntimeError("Failed to extract a list literal from VLM response code block")

        list_str = list_match.group(0)

        try:
            object_names = ast.literal_eval(list_str)
            if not isinstance(object_names, list):
                raise ValueError  # handled by except
            object_names = [str(o) for o in object_names][: args.max_objects]
        except Exception:
            raise RuntimeError("Failed to parse object list from VLM response: " + list_str)

        if len(object_names) == 0:
            raise RuntimeError("VLM did not return any objects to process.")

        print("Objects to process:", object_names)

        # Place image once into the Ray object store to avoid duplicate transfers
        image_ref = ray.put(image)

        # ------------------------------------------------------------------
        # 2. Detect points for each object (run on 3 VLM actors in parallel)
        # ------------------------------------------------------------------
        detect_refs = {
            obj: router.call_tool.remote("vlm", "detect_all", image_ref, obj)
            for obj in object_names
        }
        detect_results = {obj: ray.get(ref).value for obj, ref in detect_refs.items()}

        # Prepare segmentation and depth tasks
        seg_refs = {}
        depth_refs = {}
        for obj, points in detect_results.items():
            if not points:
                print(f"[WARN] No points returned for '{obj}' – skipping")
                continue
            x, y = map(int, points[0])  # use first point
            seg_refs[obj] = router.call_tool.remote("sam2", "segment_from_point", image_ref, x=x, y=y)
            depth_refs[obj] = router.call_tool.remote("depth_estimator", "estimate_depth", image_ref)

        # ------------------------------------------------------------------
        # 3. Gather all results (segmentation + depth)
        # ------------------------------------------------------------------
        object_info = []  # list of dicts for each successful object
        for obj in seg_refs.keys():
            seg_result, depth_result = ray.get([seg_refs[obj], depth_refs[obj]])

            if seg_result.is_error:
                print(f"[ERR] SAM2 error for '{obj}': {seg_result.text}")
                continue
            if depth_result.is_error:
                print(f"[ERR] Depth estimation error for '{obj}': {depth_result.text}")
                continue

            mask: np.ndarray = seg_result.value["mask"]
            depth_map: np.ndarray = depth_result.value["depth_map"]

            if mask.sum() == 0:
                print(f"[WARN] Empty mask for '{obj}' – skipping")
                continue

            # Compute mean depth (metres) within the mask
            mean_depth = float(np.mean(depth_map[mask]))

            # Bounding box
            ys, xs = np.where(mask)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())

            object_info.append(
                {
                    "name": obj,
                    "mask": mask,
                    "bbox": (x_min, y_min, x_max, y_max),
                    "distance": mean_depth,
                    "colour": random_colour(),
                }
            )

        if not object_info:
            raise RuntimeError("No objects were successfully processed.")

        # ------------------------------------------------------------------
        # 4. Visualise combined annotations
        # ------------------------------------------------------------------
        vis_img = image.copy()
        draw = ImageDraw.Draw(vis_img)

        # Draw green points for all detected VLM clicks (before any filtering)
        point_draw = ImageDraw.Draw(vis_img)
        for obj, pts in detect_results.items():
            for x_pt, y_pt in pts:
                r = 4
                point_draw.ellipse([(x_pt - r, y_pt - r), (x_pt + r, y_pt + r)], fill=(0, 255, 0))

        # Overlay per-object annotations (mask, bbox, label)
        # 1) Overlay masks (each call returns a *new* Image instance)
        for info in object_info:
            vis_img = overlay_mask(vis_img, info["mask"], info["colour"], alpha=0.4)

        # 2) Draw bounding boxes + labels on the *final* composited image
        draw = ImageDraw.Draw(vis_img)  # re-instantiate for the updated image
        for info in object_info:
            x_min, y_min, x_max, y_max = info["bbox"]
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=info["colour"], width=2)

            # Two-line label: object name and distance in metres
            label = f"{info['name']}\n{info['distance']:.2f} m"
            draw.multiline_text((x_min, y_min), label, fill=info["colour"])

        # ------------------------------------------------------------------
        # 5. Save / show results
        # ------------------------------------------------------------------
        out_dir = Path("outputs") / "visual_reasoning_scaling_demo"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "annotated.png"
        vis_img.save(out_path)
        print(f"Saved annotated image to {out_path}")

        if args.show:
            plt.figure(figsize=(8, 8))
            plt.imshow(vis_img)
            plt.axis("off")
            plt.title("Visual Reasoning Scaling Demo")
            plt.show()

    finally:
        print("\nShutting down toolkit …")
        shutdown_toolkit()
        print("Done.")


if __name__ == "__main__":
    main() 
