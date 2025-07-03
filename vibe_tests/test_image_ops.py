# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Script-style tests for the ImageOpsTool."""

import numpy as np
from PIL import Image

from toolshed.registry import get_tool_class, list_available_tools
from toolshed.tools.image_ops import ImageOpsTool
from toolshed.tool_result import ToolResult


def test_image_ops_registered():
    """Test that the image_ops tool is registered."""
    assert "image_ops" in list_available_tools()
    assert get_tool_class("image_ops", allow_refresh=True).__name__ == "ImageOpsTool"


def test_point_crop_basic():
    """Test basic point crop functionality."""
    tool = ImageOpsTool()

    # Create a test image
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))

    # Crop to a region
    points = [(0.2, 0.2), (0.8, 0.8)]
    result = tool.point_crop(img, points)

    assert isinstance(result, ToolResult)
    assert isinstance(result.value, Image.Image)
    assert result.value.size == (60, 60)  # 80-20=60 pixels in each dimension
    assert "Cropped image" in result.text


def test_point_crop_single_point_error():
    """Test that single point causes an error (zero dimension)."""
    tool = ImageOpsTool()
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))

    points = [(0.5, 0.5)]
    result = tool.point_crop(img, points)

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert result.value is None
    assert "zero" in result.text.lower()


def test_point_crop_out_of_range():
    """Test that out-of-range points cause an error."""
    tool = ImageOpsTool()
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))

    points = [(1.5, 0.5), (0.5, -0.1)]
    result = tool.point_crop(img, points)

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert result.value is None
    assert "outside normalized range" in result.text


def test_point_crop_same_x_coordinate():
    """Test that points with same x-coordinate cause an error."""
    tool = ImageOpsTool()
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))

    points = [(0.5, 0.2), (0.5, 0.8)]
    result = tool.point_crop(img, points)

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert result.value is None
    assert "zero width" in result.text.lower()


def test_mask_crop_basic():
    """Test basic mask crop functionality."""
    tool = ImageOpsTool()

    # Create a test image
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))

    # Create a mask
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 30:70] = True  # Rectangle in the middle

    result = tool.mask_crop(img, mask)

    assert isinstance(result, ToolResult)
    assert isinstance(result.value, Image.Image)
    assert result.value.size == (40, 60)  # 70-30=40 width, 80-20=60 height
    assert "Cropped masked region" in result.text

    # Check that non-masked areas are white
    result_array = np.array(result.value)
    cropped_mask = mask[20:80, 30:70]
    white_pixels = result_array[~cropped_mask]
    assert np.all(white_pixels == 255)


def test_mask_crop_empty_mask():
    """Test that empty mask causes an error."""
    tool = ImageOpsTool()
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))

    mask = np.zeros((100, 100), dtype=bool)
    result = tool.mask_crop(img, mask)

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert result.value is None
    assert "empty" in result.text.lower()


def test_mask_crop_wrong_dimensions():
    """Test that mask with wrong dimensions causes an error."""
    tool = ImageOpsTool()
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))

    mask = np.ones((50, 50), dtype=bool)
    result = tool.mask_crop(img, mask)

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert result.value is None
    assert "doesn't match" in result.text


def test_suppressed_outputs():
    """Test that output suppression works correctly."""
    tool = ImageOpsTool(no_output_image=True, no_output_vars=True)
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))

    # Test point crop
    points = [(0.2, 0.2), (0.8, 0.8)]
    result = tool.point_crop(img, points)

    assert isinstance(result, ToolResult)
    assert isinstance(result.value, Image.Image)
    assert len(result.image) == 0  # No image in output
    assert len(result.variables) == 0  # No variables stored
    assert "$" not in result.text  # No variable reference in text

    # Test mask crop
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 30:70] = True
    result = tool.mask_crop(img, mask)

    assert isinstance(result, ToolResult)
    assert isinstance(result.value, Image.Image)
    assert len(result.image) == 0  # No image in output
    assert len(result.variables) == 0  # No variables stored
    assert "$" not in result.text  # No variable reference in text


def test_tool_stats():
    """Test that tool statistics are tracked correctly."""
    tool = ImageOpsTool()
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))

    # Initial stats
    stats = tool.get_stats()
    assert stats["total_calls"] == 0
    assert stats["point_crop_calls"] == 0
    assert stats["mask_crop_calls"] == 0

    # Make some calls
    tool.point_crop(img, [(0.2, 0.2), (0.8, 0.8)])
    tool.point_crop(img, [(0.3, 0.3), (0.7, 0.7)])

    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 30:70] = True
    tool.mask_crop(img, mask)

    # Check updated stats
    stats = tool.get_stats()
    assert stats["total_calls"] == 3
    assert stats["point_crop_calls"] == 2
    assert stats["mask_crop_calls"] == 1


def main():
    tests = [
        test_image_ops_registered,
        test_point_crop_basic,
        test_point_crop_single_point_error,
        test_point_crop_out_of_range,
        test_point_crop_same_x_coordinate,
        test_mask_crop_basic,
        test_mask_crop_empty_mask,
        test_mask_crop_wrong_dimensions,
        test_suppressed_outputs,
        test_tool_stats,
    ]

    print("=== Testing ImageOpsTool ===")
    for t in tests:
        t()
        print(f"✓ {t.__name__}")

    print("\n✅ All ImageOpsTool tests passed!")


if __name__ == "__main__":
    main()



