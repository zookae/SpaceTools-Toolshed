# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Test the depth estimator tool.

NOTE: Tools now return ToolResult objects. When accessing tools via get_toolkit(),
the ToolResult is returned directly. Use result.value to get the underlying value.
"""

import numpy as np
from PIL import Image

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toolshed import start_toolkit, get_toolkit, shutdown_toolkit
from toolshed.registry import list_available_tools, get_tool_class
from toolshed.tool_result import ToolResult


def test_depth_estimator_registration():
    """Test that the depth estimator tool is registered."""
    assert "depth_estimator" in list_available_tools()
    assert get_tool_class("depth_estimator", allow_refresh=True).__name__ == "DepthEstimatorTool"


def test_depth_estimator_basic():
    """Test basic depth estimation functionality."""
    # Skip if no GPU available
    try:
        import torch
    except ImportError:
        print("⚠ torch not installed, skipping depth estimation tests")
        return

    if not torch.cuda.is_available():
        print("⚠ GPU not available, skipping depth estimation tests")
        return

    # Configure tool
    tool_configs = {
        "depth_estimator": {
            "num_actors": 1,
            "conda_env": "tool2_depth",
            "resources": {"num_gpus": 1},
            "timeout": 100,
            "args": {
                "checkpoint_path": "checkpoints/depth_pro.pt"
            }
        }
    }

    # Start toolkit
    router_actor = start_toolkit(tool_configs)
    try:
        toolkit = get_toolkit()

        # Create a simple test image
        test_image = Image.new("RGB", (640, 480), color="white")

        # Test depth estimation - now returns ToolResult
        tool_result = toolkit.depth_estimator.estimate_depth(test_image)
        
        # Verify we got a ToolResult
        assert isinstance(tool_result, ToolResult), f"Expected ToolResult, got {type(tool_result)}"
        
        # Extract the actual result dict
        result = tool_result.value

        # Verify results
        assert "depth_map" in result
        assert "focal_length_px" in result
        assert "width" in result
        assert "height" in result

        assert isinstance(result["depth_map"], np.ndarray)
        assert result["depth_map"].shape == (480, 640)
        assert result["width"] == 640
        assert result["height"] == 480
        assert result["focal_length_px"] > 0

        # Test statistics - get_stats returns dict directly
        stats = toolkit.depth_estimator.get_stats()
        assert stats["total_calls"] == 1
        assert stats["depth_calls"] == 1

    finally:
        _ = router_actor
        shutdown_toolkit()


def test_depth_estimator_with_pointcloud():
    """Test depth estimation with point cloud generation."""
    # Skip if no GPU available
    try:
        import torch
    except ImportError:
        print("⚠ torch not installed, skipping pointcloud tests")
        return

    if not torch.cuda.is_available():
        print("⚠ GPU not available, skipping pointcloud tests")
        return

    # Configure tool
    tool_configs = {
        "depth_estimator": {
            "num_actors": 1,
            "resources": {"num_gpus": 1},
            "timeout": 100,
        }
    }

    router_actor = start_toolkit(tool_configs)
    try:
        toolkit = get_toolkit()

        # Create a test image with some structure
        test_image = Image.new("RGB", (320, 240), color="white")
        from PIL import ImageDraw

        draw = ImageDraw.Draw(test_image)
        draw.rectangle([50, 50, 150, 150], fill="red")
        draw.ellipse([170, 90, 270, 190], fill="blue")

        # Test depth with point cloud - now returns ToolResult
        tool_result = toolkit.depth_estimator.estimate_depth_with_pointcloud(test_image)
        
        # Verify we got a ToolResult
        assert isinstance(tool_result, ToolResult), f"Expected ToolResult, got {type(tool_result)}"
        
        # Extract the actual result dict
        result = tool_result.value

        # Verify results
        assert "depth_map" in result
        assert "focal_length_px" in result
        assert "point_cloud" in result
        assert "width" in result
        assert "height" in result

        assert isinstance(result["point_cloud"], np.ndarray)
        assert result["point_cloud"].shape[1] == 3  # X, Y, Z coordinates
        assert len(result["point_cloud"]) > 0

        # Test statistics - get_stats returns dict directly
        stats = toolkit.depth_estimator.get_stats()
        assert stats["total_calls"] == 2  # One for depth, one for pointcloud
        assert stats["depth_calls"] == 1
        assert stats["pointcloud_calls"] == 1

    finally:
        _ = router_actor
        shutdown_toolkit()


if __name__ == "__main__":
    # Run basic registration test
    test_depth_estimator_registration()
    print("✓ Depth estimator tool is registered")

    # Run GPU tests if available
    test_depth_estimator_basic()
    print("✓ Basic depth estimation test passed (or skipped)")

    test_depth_estimator_with_pointcloud()
    print("✓ Point cloud generation test passed (or skipped)")
