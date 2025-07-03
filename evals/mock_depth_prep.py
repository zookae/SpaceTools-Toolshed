# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Helper utilities for mock robot mode depth preprocessing.

This module provides depth preprocessing capabilities for mock robot evaluations
by running depth estimation on input images to generate mock sensor data
(depth maps, point clouds, focal lengths).
"""

import logging
from typing import Dict, Any, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


def initialize_depth_toolkit(
    namespace: str = 'mock_depth_prep',
    router_name: str = 'mock_depth_router',
    checkpoint_path: str = '/lustre/fsw/portfolios/nvr/users/vblukis/checkpoints/depth_pro.pt'
) -> Tuple[Any, Any]:
    """Initialize separate depth toolkit for preprocessing mock robot data.
    
    This creates an isolated toolkit instance with only depth_estimator
    in a separate namespace to avoid conflicts with the main toolkit.
    
    Args:
        namespace: Ray namespace for isolation (default: 'mock_depth_prep')
        router_name: Router actor name (default: 'mock_depth_router')
        checkpoint_path: Path to depth model checkpoint
        
    Returns:
        tuple: (depth_router, depth_toolkit) - Router and toolkit instances
    """
    from toolshed import start_toolkit, get_toolkit
    
    logger.info(f"Initializing depth preprocessing toolkit in namespace '{namespace}'...")
    
    depth_config = {
        'depth_estimator': {
            'num_actors': 8,
            'resources': {'num_gpus': 0.1},
            'conda_env': 'tool_depth',
            'timeout': 600,
            'args': {
                'checkpoint_path': checkpoint_path,
                'no_output_image': True,  # Don't need viz for preprocessing
                'no_output_vars': True,
            }
        }
    }
    
    depth_router = start_toolkit(
        depth_config,
        router_name=router_name,
        namespace=namespace,
        detached=False,
        dashboard=False
    )
    depth_toolkit = get_toolkit(router_name=router_name, namespace=namespace)
    logger.info("Depth preprocessing toolkit initialized successfully")
    return depth_router, depth_toolkit


def prepare_mock_data_from_image(image: Image.Image, depth_toolkit) -> Dict[str, Any]:
    """Prepare mock robot data from image using depth estimator.
    
    This runs the depth estimator on the input image to extract:
    - depth_map: 2D array of depth values
    - point_cloud: Nx3 array of 3D points  
    - focal_length_px: Estimated camera focal length
    
    The mock data is prepared once per image and injected into all subsequent
    robot tool calls for that conversation session.
    
    Args:
        image: PIL Image to process
        depth_toolkit: Toolkit instance with depth_estimator tool
        
    Returns:
        Dict with mock data fields:
            - mock_image: Original PIL Image (or Ray ObjectRef)
            - mock_depth_map: Depth map array (or Ray ObjectRef)
            - mock_point_cloud: Point cloud array (or Ray ObjectRef)
            - mock_focal_length_px: Focal length in pixels (float)
            - image_width: Image width in pixels (int)
            - image_height: Image height in pixels (int)
    """
    if depth_toolkit is None:
        logger.warning("No depth toolkit available, returning minimal mock data")
        return {
            "mock_image": image,
            "image_width": image.size[0],
            "image_height": image.size[1],
        }
    
    try:
        # Use depth estimator to get depth and pointcloud
        result = depth_toolkit.depth_estimator.estimate_depth_with_pointcloud(image)
        
        # Use Ray object references for large data to avoid serialization overhead
        import ray
        if ray.is_initialized():
            mock_data = {
                "mock_image": ray.put(image),
                "mock_depth_map": ray.put(result.value["depth_map"]),
                "mock_point_cloud": ray.put(result.value["point_cloud"]),
                "mock_focal_length_px": result.value["focal_length_px"],
                "image_width": result.value["width"],
                "image_height": result.value["height"],
            }
        else:
            # Fallback if Ray is not initialized
            mock_data = {
                "mock_image": image,
                "mock_depth_map": result.value["depth_map"],
                "mock_point_cloud": result.value["point_cloud"],
                "mock_focal_length_px": result.value["focal_length_px"],
                "image_width": result.value["width"],
                "image_height": result.value["height"],
            }
        
        logger.debug(
            f"Prepared mock data: depth_map shape={result.value['depth_map'].shape}, "
            f"point_cloud shape={result.value['point_cloud'].shape}, "
            f"focal_length={result.value['focal_length_px']:.1f}px"
        )
        
        return mock_data
        
    except Exception as e:
        logger.error(f"Failed to prepare mock data: {e}")
        # Return minimal mock data as fallback
        return {
            "mock_image": image,
            "image_width": image.size[0],
            "image_height": image.size[1],
        }

