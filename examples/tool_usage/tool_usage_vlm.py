# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""vision_tools.py
A minimal example demonstrating how to use the VLM tool for object
      detection within the toolshed toolkit.

This script:
1. Starts the toolkit with a single VLM actor (GPU).
2. Loads a demo image (examples/media/kitchen.png).
3. Asks the VLM to detect all *peaches* in the image.
4. Prints the returned bounding boxes.
5. Saves the visualization with detected points overlaid.
"""

from pathlib import Path
import os

import ray
from PIL import Image

from toolshed import start_toolkit, get_toolkit, shutdown_toolkit
from toolshed.tool_result import ToolResult

import logging
logging.basicConfig(level=logging.DEBUG)

def main() -> None:
    # Configure the toolkit – one GPU-backed VLM actor is enough for a demo.
    tool_configs = {
        "vlm": {
            "num_actors": 1,
            "resources": {"num_gpus": 1},
            "conda_env": "tool_vlm",
        }
    }

    print("Starting toolkit with VLM …")
    handle = start_toolkit(tool_configs)

    try:
        toolkit = get_toolkit()

        # Load the sample image bundled with the repository.
        img_path = Path(__file__).parent / "media" / "kitchen.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Example image not found: {img_path}")

        image = Image.open(img_path)
        # Put the image in the Ray object store to avoid large copies.
        img_ref = ray.put(image)

        print("Requesting VLM to detect all peaches …")
        result = toolkit.vlm.detect_all(img_ref, "peach")
        
        # Handle the ToolResult properly
        if isinstance(result, ToolResult):
            points = result.value
            print("Detected points:", points)
            print("Result text:", result.text)
            
            # Save the visualization if available
            if result.image:
                overlay_image = result.image[0] if isinstance(result.image, list) else result.image
                output_dir = Path(__file__).parent.parent / "outputs"
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / "vlm_detection_overlay.png"
                overlay_image.save(output_path)
                print(f"Saved visualization to: {output_path}")
            else:
                print("No visualization available (no detections found)")
                
            # Access the points from variables if available
            if result.variables:
                for var_name, var_value in result.variables.items():
                    print(f"Variable '{var_name}': {var_value}")
        else:
            # Fallback for raw response
            print("Detected points:", result)

    finally:
        shutdown_toolkit()


if __name__ == "__main__":
    main() 