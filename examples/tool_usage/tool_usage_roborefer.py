# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""tool_usage_roborefer.py
Minimal example demonstrating how to use RoboreferTool for point detection.

Steps:
1. Launch toolshed toolkit with one GPU-backed RoboRefer actor.
2. Load a demo image (examples/media/kitchen.png).
3. Ask RoboRefer to detect all instances of a given object.
4. Print returned points and save overlay.

Run inside the *roborefer* conda env with a working GPU:
    python examples/tool_usage_roborefer.py --image /path/to/img.jpg --object "cat"
"""

from __future__ import annotations

from pathlib import Path
import argparse
import logging

import ray
from PIL import Image

from toolshed import start_toolkit, get_toolkit, shutdown_toolkit
from toolshed.tool_result import ToolResult

logging.basicConfig(level=logging.DEBUG)

def parse_args():
    p = argparse.ArgumentParser(description="Roborefer usage example")
    p.add_argument("--image", type=Path, help="Path to input image", required=False)
    p.add_argument("--object", type=str, default="peach", help="Object name to detect")
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("/lustre/fsw/portfolios/nvr/users/siyic/projects/RoboRefer/models/RoboRefer-8B-SFT"),
        help="Path to RoboRefer checkpoint directory",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Configure toolkit with one roborefer actor on a GPU
    tool_configs = {
        "roborefer": {
            "num_actors": 1,
            "resources": {"num_gpus": 1},
            "conda_env": "tool_roborefer",
            "args": {
                "model_path": str(args.model_path),
                'no_output_image': False,
                'no_output_vars': True,
                'exclude_methods': ['general_query'],
                'exclude_behavior': 'error'
            },
        }
    }

    print("Starting toolkit with RoboRefer …")
    handle = start_toolkit(tool_configs)

    try:
        toolkit = get_toolkit()

        # Use provided image or default sample
        if args.image is None:
            img_path = Path(__file__).parent / "media" / "kitchen.png"
        else:
            img_path = args.image
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path)
        img_ref = ray.put(image)

        print(f"Requesting RoboRefer to detect all {args.object}s …")
        result = toolkit.roborefer.detect_all(img_ref, args.object)

        if isinstance(result, ToolResult):
            points = result.value
            print("Detected points:", points)
            print("Result text:", result.text)

            if result.image:
                overlay = (
                    result.image[0] if isinstance(result.image, list) else result.image
                )
                out_dir = Path(__file__).parent.parent / "outputs"
                out_dir.mkdir(exist_ok=True)
                out_path = out_dir / "roborefer_detection_overlay.png"
                overlay.save(out_path)
                print(f"Saved visualization to: {out_path}")
            else:
                print("No visualization available (no detections found)")

            if result.variables:
                for name, val in result.variables.items():
                    print(f"Variable '{name}': {val}")
        else:
            print("Detected points:", result)

    finally:
        shutdown_toolkit()


if __name__ == "__main__":
    main()
