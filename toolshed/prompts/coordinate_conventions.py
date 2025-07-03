# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Shared coordinate convention definitions for computer vision tools.

This module defines standardized coordinate systems used across toolshed's
computer vision tools to ensure consistency and enable seamless workflows.
"""

COORDINATE_CONVENTIONS_PROMPT = """
## Coordinate Conventions

**2D Image Coordinates:**
- Format: Normalized coordinates (x, y) as floats in range [0, 1]
- Origin: Top-left corner where (0, 0) = top-left, (1, 1) = bottom-right
- Example: (0.5, 0.5) represents the center of the image
- Rationale: Resolution-independent, enables seamless tool chaining

**3D Camera Coordinates:**
- Format: (x, y, z) in meters using right-handed coordinate system
- X-axis: Points right from camera center (positive = right)
- Y-axis: Points down from camera center (positive = down)  
- Z-axis: Points forward from camera (positive = forward/depth)
- Origin: Camera optical center
- Principal point: Assumed at image center (width/2, height/2)
"""


# Optional helper functions that tools can import and use
def validate_normalized_coords(coords):
    """Validate that coordinates are in [0, 1] range.
    
    Args:
        coords: List of (x, y) coordinate tuples
        
    Returns:
        bool: True if all coordinates are in valid [0, 1] range
    """
    if not coords:
        return True
    return all(0 <= x <= 1 and 0 <= y <= 1 for x, y in coords)


def normalize_pixel_coords(pixel_coords, width, height):
    """Convert pixel coordinates to normalized [0, 1] coordinates.
    
    Args:
        pixel_coords: List of (x, y) pixel coordinate tuples
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        List of (x, y) normalized coordinate tuples in [0, 1] range
    """
    return [(x / width, y / height) for x, y in pixel_coords]


def denormalize_coords(norm_coords, width, height):
    """Convert normalized coordinates to pixel coordinates.
    
    Args:
        norm_coords: List of (x, y) normalized coordinate tuples in [0, 1] range
        width: Image width in pixels  
        height: Image height in pixels
        
    Returns:
        List of (x, y) pixel coordinate tuples as integers
    """
    return [(int(x * width), int(y * height)) for x, y in norm_coords]


def coords_to_camera_3d(norm_coords, depth_map, focal_length_px, width, height):
    """Convert 2D normalized coordinates to 3D camera coordinates.
    
    Args:
        norm_coords: List of (x, y) normalized coordinates in [0, 1] range
        depth_map: 2D numpy array of depth values in meters
        focal_length_px: Focal length in pixels
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        List of (x, y, z) 3D points in camera coordinate system (meters)
    """
    import numpy as np
    
    # Convert normalized to pixel coordinates
    pixel_coords = denormalize_coords(norm_coords, width, height)
    
    # Camera principal point (center of image)
    cx = width / 2
    cy = height / 2
    
    points_3d = []
    for px, py in pixel_coords:
        # Clamp to image bounds
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))
        
        # Get depth at this pixel
        depth = depth_map[py, px]  # Note: numpy uses [row, col] = [y, x]
        
        if np.isnan(depth) or np.isinf(depth) or depth <= 0:
            continue  # Skip invalid depth values
            
        # Convert to 3D camera coordinates
        x = (px - cx) * depth / focal_length_px
        y = (py - cy) * depth / focal_length_px
        z = depth
        
        points_3d.append((x, y, z))
    
    return points_3d
