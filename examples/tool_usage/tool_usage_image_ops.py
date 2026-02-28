#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Example usage of the ImageOpsTool for point-based and mask-based cropping.

This example demonstrates:
1. Point-based cropping to extract regions of interest from a kitchen scene
2. Mask-based cropping with white background fill
3. Integration with other vision tools (e.g., could be combined with VLM/SAM2)
4. Error handling for invalid inputs

This script uses the sample kitchen image bundled with the repository.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import ray

from toolshed.tools.image_ops import ImageOpsTool
from toolshed.tool_result import ToolResult


def load_kitchen_image():
    """Load the sample kitchen image bundled with the repository."""
    img_path = Path(__file__).parent / ".." / "media" / "kitchen.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Example image not found: {img_path}")
    
    image = Image.open(img_path)
    print(f"Loaded kitchen image: {image.size[0]}x{image.size[1]} pixels")
    return image


def example_point_crop():
    """Demonstrate point-based cropping on the kitchen image."""
    print("\n" + "="*60)
    print("POINT-BASED CROPPING EXAMPLES")
    print("="*60)
    
    tool = ImageOpsTool()
    img = load_kitchen_image()
    
    # Put the image in Ray object store for efficient processing
    img_ref = ray.put(img)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Example 1: Crop to the counter area (left side of image)
    print("\n1. Cropping to counter area (left side):")
    points = [
        (0.0, 0.3),   # Top-left of counter area
        (0.5, 0.9),   # Bottom-right of counter area
    ]
    result = tool.point_crop(img_ref, points)
    print(f"   Result: {result.text}")
    if isinstance(result.value, Image.Image):
        output_path = output_dir / "crop_counter_area.png"
        result.value.save(output_path)
        print(f"   Saved to: {output_path}")
    
    # Example 2: Crop to the sink and window area (center-right)
    print("\n2. Cropping to sink and window area:")
    points = [
        (0.4, 0.1),   # Top-center (window)
        (0.9, 0.7),   # Bottom-right (sink area)
    ]
    result = tool.point_crop(img_ref, points)
    print(f"   Result: {result.text}")
    if isinstance(result.value, Image.Image):
        output_path = output_dir / "crop_sink_window.png"
        result.value.save(output_path)
        print(f"   Saved to: {output_path}")
    
    # Example 3: Crop to a specific object area (simulating VLM detection points)
    print("\n3. Simulating crop from object detection points:")
    # These points could come from VLM.detect_all() for a specific object
    points = [
        (0.3, 0.4),   # Simulated detection point 1
        (0.35, 0.45), # Simulated detection point 2
        (0.32, 0.42), # Simulated detection point 3
    ]
    result = tool.point_crop(img_ref, points)
    print(f"   Result: {result.text}")
    if isinstance(result.value, Image.Image):
        output_path = output_dir / "crop_object_region.png"
        result.value.save(output_path)
        print(f"   Saved to: {output_path}")
    
    # Example 4: Error case - single point
    print("\n4. Error case - single point (zero dimension):")
    points = [(0.5, 0.5)]
    result = tool.point_crop(img_ref, points)
    print(f"   Result: {result.text}")
    
    # Example 5: Error case - points outside range
    print("\n5. Error case - points outside normalized range:")
    points = [(1.5, 0.5), (-0.1, 0.5)]
    result = tool.point_crop(img_ref, points)
    print(f"   Result: {result.text}")


def example_mask_crop():
    """Demonstrate mask-based cropping on the kitchen image."""
    print("\n" + "="*60)
    print("MASK-BASED CROPPING EXAMPLES")
    print("="*60)
    
    tool = ImageOpsTool()
    img = load_kitchen_image()
    width, height = img.size  # 1433 x 949
    
    # Put the image in Ray object store for efficient processing
    img_ref = ray.put(img)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Example 1: Rectangular mask for a specific region (e.g., window area)
    print("\n1. Masking the window area:")
    mask = np.zeros((height, width), dtype=bool)
    # Window is approximately in the upper-middle portion
    mask[50:400, 500:1000] = True
    result = tool.mask_crop(img_ref, mask)
    print(f"   Result: {result.text}")
    if isinstance(result.value, Image.Image):
        output_path = output_dir / "mask_window_area.png"
        result.value.save(output_path)
        print(f"   Saved to: {output_path}")
    
    # Example 2: Circular mask (simulating a region of interest)
    print("\n2. Circular mask for region of interest:")
    mask = np.zeros((height, width), dtype=bool)
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width // 2, height // 2
    radius = 200
    circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    mask[circle_mask] = True
    result = tool.mask_crop(img_ref, mask)
    print(f"   Result: {result.text}")
    if isinstance(result.value, Image.Image):
        output_path = output_dir / "mask_circular_roi.png"
        result.value.save(output_path)
        print(f"   Saved to: {output_path}")
    
    # Example 3: Simulating a segmentation mask (e.g., from SAM2)
    print("\n3. Simulating segmentation mask (could be from SAM2):")
    # This simulates what you might get from SAM2 segmentation
    mask = np.zeros((height, width), dtype=bool)
    # Create multiple regions that might represent segmented objects
    mask[300:500, 100:400] = True    # Object 1
    mask[400:600, 600:800] = True    # Object 2
    mask[200:350, 900:1100] = True   # Object 3
    result = tool.mask_crop(img_ref, mask)
    print(f"   Result: {result.text}")
    if isinstance(result.value, Image.Image):
        output_path = output_dir / "mask_segmentation_sim.png"
        result.value.save(output_path)
        print(f"   Saved to: {output_path}")
    
    # Example 4: Gradient mask (demonstrating partial masking)
    print("\n4. Gradient mask for smooth transition:")
    mask = np.zeros((height, width), dtype=bool)
    # Create a diagonal gradient mask
    for i in range(height):
        for j in range(width):
            if i + j < (height + width) // 2:
                mask[i, j] = True
    result = tool.mask_crop(img_ref, mask)
    print(f"   Result: {result.text}")
    if isinstance(result.value, Image.Image):
        output_path = output_dir / "mask_gradient.png"
        result.value.save(output_path)
        print(f"   Saved to: {output_path}")
    
    # Example 5: Error case - empty mask
    print("\n5. Error case - empty mask:")
    mask = np.zeros((height, width), dtype=bool)
    result = tool.mask_crop(img_ref, mask)
    print(f"   Result: {result.text}")
    
    # Example 6: Error case - wrong dimensions
    print("\n6. Error case - mask with wrong dimensions:")
    mask = np.ones((500, 700), dtype=bool)
    result = tool.mask_crop(img_ref, mask)
    print(f"   Result: {result.text}")


def example_with_suppressed_outputs():
    """Demonstrate usage with suppressed outputs."""
    print("\n" + "="*60)
    print("SUPPRESSED OUTPUT MODE")
    print("="*60)
    
    # Create tool with suppressed outputs
    tool = ImageOpsTool(no_output_image=True, no_output_vars=True)
    img = load_kitchen_image()
    width, height = img.size
    
    # Put the image in Ray object store
    img_ref = ray.put(img)
    
    print("\nWhen outputs are suppressed:")
    print("- no_output_image=True: No image returned in ToolResult")
    print("- no_output_vars=True: No variables stored in context")
    
    # Test point crop
    points = [(0.2, 0.2), (0.6, 0.6)]
    result = tool.point_crop(img_ref, points)
    print(f"\nPoint crop result:")
    print(f"  Text: {result.text}")
    print(f"  Has image: {bool(result.image)}")
    print(f"  Has variables: {bool(result.variables)}")
    print(f"  Raw value type: {type(result.value)}")
    
    # Test mask crop
    mask = np.zeros((height, width), dtype=bool)
    mask[100:300, 100:300] = True
    result = tool.mask_crop(img_ref, mask)
    print(f"\nMask crop result:")
    print(f"  Text: {result.text}")
    print(f"  Has image: {bool(result.image)}")
    print(f"  Has variables: {bool(result.variables)}")
    print(f"  Raw value type: {type(result.value)}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" ImageOpsTool Examples with Kitchen Scene")
    print("="*70)
    print("\nThis example demonstrates the ImageOpsTool capabilities:")
    print("- Point-based cropping to extract regions of interest from a kitchen scene")
    print("- Mask-based cropping with white background fill")
    print("- Integration with Ray for efficient image processing")
    print("- Potential integration with other vision tools (VLM, SAM2)")
    print("- Error handling for invalid inputs")
    print("- Output suppression modes")
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    try:
        example_point_crop()
        example_mask_crop()
        example_with_suppressed_outputs()
        
        print("\n" + "="*70)
        print(" Examples completed!")
        print("="*70)
        print("\nGenerated files in outputs/:")
        print("  Point-based crops:")
        print("    - crop_counter_area.png: Left counter area")
        print("    - crop_sink_window.png: Sink and window region")
        print("    - crop_object_region.png: Simulated object detection crop")
        print("  Mask-based crops:")
        print("    - mask_window_area.png: Window region with white background")
        print("    - mask_circular_roi.png: Circular region of interest")
        print("    - mask_segmentation_sim.png: Simulated segmentation result")
        print("    - mask_gradient.png: Gradient mask demonstration")
    finally:
        # Shutdown Ray if we initialized it
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
