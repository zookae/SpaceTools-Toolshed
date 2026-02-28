#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Example usage of the depth estimation tool in toolshed.

This example demonstrates how to:
1. Start the toolkit with the depth estimator tool
2. Estimate depth from an image
3. Generate a 3D point cloud from the depth map
4. Visualize the results
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from toolshed import start_toolkit, get_toolkit


def visualize_point_cloud(point_cloud, sample_rate=10):
    """Visualize 3D point cloud."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points for visualization (too many points slow down rendering)
    indices = np.random.choice(len(point_cloud), 
                              size=min(len(point_cloud) // sample_rate, 10000), 
                              replace=False)
    sampled_points = point_cloud[indices]
    
    # Plot the points
    ax.scatter(sampled_points[:, 0], 
               sampled_points[:, 1], 
               sampled_points[:, 2],
               c=sampled_points[:, 2],  # Color by depth
               cmap='viridis',
               s=1,
               alpha=0.5)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'3D Point Cloud ({len(sampled_points)} points shown)')
    
    # Set reasonable axis limits
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([0, 10])
    
    return fig


def main():
    """Main example function."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Depth estimation tool usage example")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file for tool configurations. "
                             "If provided, overrides the inline tool configs (including conda_env names).")
    args = parser.parse_args()

    # Configure the depth estimator tool
    # You may need to adjust the checkpoint path based on your setup
    tool_configs = {
        "depth_estimator": {
            "num_actors": 1,
            "conda_env": "tool_depth",
            "resources": {"num_gpus": 1},  # Request GPU for the tool
            "timeout": 100,
            "args": {
                "checkpoint_path": "checkpoints/depth_pro.pt"
            }
        }
    }

    # Override with JSON config if provided
    if args.config is not None:
        config_path = args.config
        if not os.path.exists(config_path):
            print(f"Error: Config file not found at {config_path}")
            sys.exit(1)
        with open(config_path, "r") as f:
            tool_configs = json.load(f)
    
    print("Starting toolshed with depth estimator...")
    try:
        # Start the toolkit
        print("Starting toolshed with depth estimator...")
        handle = start_toolkit(tool_configs, detached=False)
        
        # Get a client to access tools
        toolkit = get_toolkit()
        
        # Load an example image
        # You can replace this with your own image path
        image_path = os.path.join(os.path.dirname(__file__), "..", "media", "kitchen.png")
        
        image = Image.open(image_path)
        print(f"Loaded image: {image.size}")
        
        # Estimate depth
        print("\nEstimating depth...")
        depth_tool_result = toolkit.depth_estimator.estimate_depth(image)
        
        print(f"Depth estimation complete!")
        
        depth_result = depth_tool_result.value  # Get the actual dictionary
        print(f"\nTool output message: {depth_tool_result.text}")
        
        # The visualization is available in depth_tool_result.image
        toolkit_viz = depth_tool_result.image[0] if depth_tool_result.image else None
            
        print(f"  - Depth map shape: {depth_result['depth_map'].shape}")
        print(f"  - Focal length: {depth_result['focal_length_px']:.2f} pixels")
        print(f"  - Min depth: {depth_result['depth_map'].min():.2f} meters")
        print(f"  - Max depth: {depth_result['depth_map'].max():.2f} meters")
        
        # Generate point cloud
        print("\nGenerating depth with a 3D point cloud...")
        pointcloud_tool_result = toolkit.depth_estimator.estimate_depth_with_pointcloud(image)
        
        pointcloud_result = pointcloud_tool_result.value
        print(f"\nTool output message: {pointcloud_tool_result.text}")
        # The tool also returns a visualization of the depth map
        toolkit_viz_pc = pointcloud_tool_result.image[0] if pointcloud_tool_result.image else None
        
        point_cloud = pointcloud_result['point_cloud']
        print(f"Generated {len(point_cloud):,} 3D points")
        
        # Visualize results
        print("\nVisualizing results...")
        
        # Save the toolkit's visualization if available
        if toolkit_viz is not None:
            # Save the toolkit visualization directly
            output_dir = 'outputs'
            os.makedirs(output_dir, exist_ok=True)
            
            toolkit_viz_path = os.path.join(output_dir, 'depth_toolkit_visualization.png')
            toolkit_viz.save(toolkit_viz_path)
            print(f"Saved toolkit visualization to '{toolkit_viz_path}'")
        
        # Visualize point cloud
        if len(point_cloud) > 0:
            pc_fig = visualize_point_cloud(point_cloud)
            pc_path = os.path.join(output_dir if 'output_dir' in locals() else '.', 'pointcloud_visualization.png')
            pc_fig.savefig(pc_path, dpi=150, bbox_inches='tight')
            print(f"Saved point cloud visualization to '{pc_path}'")
            
        # Get tool statistics
        stats = toolkit.depth_estimator.get_stats()
        print(f"\nTool statistics: {stats}")
        
        # Show that we can access the stored variables
        if depth_tool_result.variables:
            print(f"\nStored variables: {list(depth_tool_result.variables.keys())}")
            # The depth_map is available as $depth_map in the context
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
