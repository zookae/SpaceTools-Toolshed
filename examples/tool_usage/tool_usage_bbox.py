#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Example: Complete bounding box pipeline using VLM, SAM2, Depth Estimator, and BoundingBox tools.

This script demonstrates:
1. Loading an image
2. Using VLM to detect an object (butter)
3. Using SAM2 to segment the object and get a mask
4. Using depth estimator to get depth map and point cloud
5. Computing oriented bounding box with the BoundingBoxTool (both default and PCA methods)
6. Visualizing all outputs including 3D view and 2D overlay

Usage:
$ python examples/tool_usage_bbox.py [--image PATH] [--out FOLDER_NAME] [--query OBJECT]

Requirements:
* Toolshed toolkit dependencies satisfied (see project README)
"""

import os
import sys
import numpy as np
from PIL import Image
import ray
import cv2
import argparse

# Use non-GUI backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to PYTHONPATH so we can import toolshed
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT_DIR)

from toolshed import start_toolkit, get_toolkit, shutdown_toolkit  # noqa: E402
from toolshed.tools.bounding_box import BoundingBoxTool  # noqa: E402
from toolshed.tool_result import ToolResult  # noqa: E402

import open3d as o3d

def _opencv_to_opengl(xyz: np.ndarray) -> np.ndarray:
    """Convert points from OpenCV camera frame to OpenGL camera frame."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be Nx3 array")
    out = xyz.copy()
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out


def _opengl_to_opencv(xyz: np.ndarray) -> np.ndarray:
    """Inverse of _opencv_to_opengl."""
    return _opencv_to_opengl(xyz)


def _project_points(xyz, image_size, fx, fy, cx, cy):
    """Project 3D points to pixel coords."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return np.stack([u, v], axis=1)


def create_bbox_overlay(image, corners_px, edges):
    """Create overlay with bounding box drawn on image using OpenCV."""
    # Convert PIL image to OpenCV format (numpy array, BGR)
    img_array = np.array(image)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Make a copy to draw on
    overlay = img_array.copy()
    
    # Draw the bounding box edges
    # Color is lime green in RGB: (0, 255, 0)
    color = (0, 255, 0)
    thickness = 2
    
    for i, j in edges:
        pt1 = (int(corners_px[i][0]), int(corners_px[i][1]))
        pt2 = (int(corners_px[j][0]), int(corners_px[j][1]))
        cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    # Convert back to PIL Image
    return Image.fromarray(overlay)


def create_points_overlay(image, points_px, color="cyan", size=1.5):
    """Create overlay with points drawn on image."""
    # Subsample if too many points
    max_points = 100_000
    pts = points_px
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    fig, ax = plt.subplots(figsize=(6, 6 * image.height / image.width))
    ax.imshow(image)
    ax.scatter(pts[:, 0], pts[:, 1], s=size, c=color, alpha=0.9, edgecolors='none')
    ax.set_axis_off()
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(height, width, 4)[..., :3]
    overlay = Image.fromarray(buf.copy())
    plt.close(fig)
    return overlay


def visualize_3d_bbox(point_cloud, obb_corners, output_path):
    """Create 3D visualization of point cloud with bounding box."""
    
    # Convert to OpenGL frame for visualization
    pts_gl = _opencv_to_opengl(point_cloud)
    corners_gl = np.array(obb_corners)
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_gl))
    
    # Create oriented bounding box from corners
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(corners_gl)
    )
    
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    
    # Render
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd.paint_uniform_color([0.1, 0.7, 0.9]))
    vis.add_geometry(line_set.paint_uniform_color([1, 0, 0]))
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(True)
    vis.destroy_window()
    
    img_np = (255 * np.asarray(img)).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.save(output_path)
    print(f"Saved 3D visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Complete bounding box pipeline using VLM, SAM2, Depth Estimator, and BoundingBox tools.")
    parser.add_argument("--image", type=str, default=None, help="Path to input image (default: examples/media/kitchen.png)")
    parser.add_argument("--out", type=str, default="bbox", help="Output folder name or absolute path (default: bbox, which becomes outputs/bbox)")
    parser.add_argument("--query", type=str, default="butter", help="Object to detect with VLM (default: butter)")
    args = parser.parse_args()
    
    # Determine output directory
    if os.path.isabs(args.out):
        output_dir = args.out
    else:
        output_dir = os.path.join("outputs", args.out)
    os.makedirs(output_dir, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Initialize all tools
    # ------------------------------------------------------------------
    tool_configs = {
        "vlm": {
            "num_actors": 1,
            "resources": {"num_gpus": 1},
            "conda_env": "tool_vlm",
        },
        "sam2": {
            "num_actors": 1,
            "resources": {"num_gpus": 1},
            "conda_env": "tool_sam2",
        },
        "depth_estimator": {
            "num_actors": 1,
            "resources": {"num_gpus": 1},
            "conda_env": "tool_depth",
            "args": {
                "checkpoint_path": "/lustre/fsw/portfolios/nvr/users/vblukis/checkpoints/depth_pro.pt"
            }
        },
        "bounding_box": {
            "num_actors": 1,
            "resources": {"num_gpus": 0},
            "conda_env": "tool_bbox",
        }
    }
    
    print("Starting toolshed with all tools...")
    handle = start_toolkit(tool_configs, detached=False)
    toolkit = get_toolkit()

    try:
        # Load the image
        if args.image:
            img_path = args.image
        else:
            img_path = os.path.join(ROOT_DIR, "examples", "media", "kitchen.png")
        image = Image.open(img_path)
        print(f"Loaded image: {image.size}")
        
        # Put in Ray object store
        img_ref = ray.put(image)
        
        # ------------------------------------------------------------------
        # Step 1: Use VLM to detect object
        # ------------------------------------------------------------------
        print(f"\n[1/5] Detecting object with VLM (query: '{args.query}')...")
        vlm_result = toolkit.vlm.detect_one(img_ref, args.query)
        
        if isinstance(vlm_result, ToolResult):
            points = vlm_result.value
            print(f"  Found {len(points)} point(s): {points}")
            
            if vlm_result.image:
                vlm_viz_path = os.path.join(output_dir, "01_vlm_detection.png")
                vlm_result.image[0].save(vlm_viz_path)
                print(f"  Saved VLM visualization to {vlm_viz_path}")
        else:
            points = vlm_result
            print(f"  Found {len(points)} point(s): {points}")
        
        if not points:
            print("No glass bowl detected. Exiting.")
            return
        
        # Use the first detected point
        query_point = points[0]
        print(f"  Using query point: {query_point}")
        
        # ------------------------------------------------------------------
        # Step 2: Use SAM2 to segment the object
        # ------------------------------------------------------------------
        print("\n[2/5] Segmenting object with SAM2...")
        sam2_result = toolkit.sam2.segment_from_point(
            img_ref,
            x=query_point[0],
            y=query_point[1]
        )
        
        if isinstance(sam2_result, ToolResult):
            if sam2_result.is_error:
                raise RuntimeError(f"SAM2 tool call failed: {sam2_result.text}")
            result_value = sam2_result.value
            
            mask = result_value['mask']  # singular "mask"
            print(f"  Generated mask shape: {mask.shape}")
            
            if sam2_result.image:
                sam2_viz_path = os.path.join(output_dir, "02_sam2_segmentation.png")
                sam2_result.image[0].save(sam2_viz_path)
                print(f"  Saved SAM2 visualization to {sam2_viz_path}")
            
            # Save mask separately
            mask_path = os.path.join(output_dir, "02_mask.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            print(f"  Saved mask to {mask_path}")
        else:
            mask = sam2_result['mask']  # singular "mask"
            print(f"  Generated mask shape: {mask.shape}")
        
        # ------------------------------------------------------------------
        # Step 3: Estimate depth and generate point cloud
        # ------------------------------------------------------------------
        print("\n[3/5] Estimating depth with depth estimator...")
        depth_result = toolkit.depth_estimator.estimate_depth_with_pointcloud(img_ref)
        
        if isinstance(depth_result, ToolResult):
            point_cloud = depth_result.value['point_cloud']
            focal_length_px = depth_result.value['focal_length_px']
            depth_map = depth_result.value['depth_map']
            print(f"  Generated point cloud: {point_cloud.shape}")
            print(f"  Focal length: {focal_length_px:.2f} px")
            
            if depth_result.image:
                depth_viz_path = os.path.join(output_dir, "03_depth_estimation.png")
                depth_result.image[0].save(depth_viz_path)
                print(f"  Saved depth visualization to {depth_viz_path}")
            
            # Save depth map
            depth_map_path = os.path.join(output_dir, "03_depth_map.npy")
            np.save(depth_map_path, depth_map)
            print(f"  Saved depth map to {depth_map_path}")
        else:
            point_cloud = depth_result['point_cloud']
            focal_length_px = depth_result['focal_length_px']
            depth_map = depth_result['depth_map']
            print(f"  Generated point cloud: {point_cloud.shape}")
        
        # ------------------------------------------------------------------
        # Step 4: Compute bounding box (default method)
        # ------------------------------------------------------------------
        print("\n[4/6] Computing oriented bounding box (default method)...")
        bbox_result = toolkit.bounding_box.compute_bbox(
            point_cloud=point_cloud,
            mask=mask,
            focal_length_px=focal_length_px
        )
        
        if isinstance(bbox_result, ToolResult):
            obb_corners = bbox_result.value['obb_corners_3d']
            corners_px = bbox_result.value['obb_corners_2d']
            extent = bbox_result.value['extent']
            edges = bbox_result.value['edges']
            print(f"  {bbox_result.text}")
            print(f"  Extent (w, h, d): {extent}")
        else:
            obb_corners = bbox_result['obb_corners_3d']
            corners_px = bbox_result['obb_corners_2d']
            extent = bbox_result['extent']
            edges = bbox_result['edges']
        
        # ------------------------------------------------------------------
        # Step 4b: Compute bounding box (PCA method)
        # ------------------------------------------------------------------
        print("\n[5/6] Computing oriented bounding box (PCA method)...")
        bbox_pca_result = toolkit.bounding_box.compute_bbox_pca(
            point_cloud=point_cloud,
            mask=mask,
            focal_length_px=focal_length_px
        )
        
        if isinstance(bbox_pca_result, ToolResult):
            obb_corners_pca = bbox_pca_result.value['obb_corners_3d']
            corners_px_pca = bbox_pca_result.value['obb_corners_2d']
            extent_pca = bbox_pca_result.value['extent']
            edges_pca = bbox_pca_result.value['edges']
            print(f"  {bbox_pca_result.text}")
            print(f"  Extent (w, h, d): {extent_pca}")
        else:
            obb_corners_pca = bbox_pca_result['obb_corners_3d']
            corners_px_pca = bbox_pca_result['obb_corners_2d']
            extent_pca = bbox_pca_result['extent']
            edges_pca = bbox_pca_result['edges']
        
        # ------------------------------------------------------------------
        # Step 6: Create visualizations and save debug artifacts
        # ------------------------------------------------------------------
        print("\n[6/6] Creating visualizations...")
        
        # Save point cloud
        pc_path = os.path.join(output_dir, "04_point_cloud.npy")
        np.save(pc_path, point_cloud)
        print(f"  Saved point cloud to {pc_path}")
        
        # Save masked point cloud
        h, w = mask.shape
        fx = fy = focal_length_px
        cx, cy = w / 2.0, h / 2.0
        pixels = _project_points(point_cloud, (h, w), fx, fy, cx, cy)
        u = pixels[:, 0].astype(int)
        v = pixels[:, 1].astype(int)
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        keep = np.zeros(len(point_cloud), dtype=bool)
        keep[in_bounds] = mask[v[in_bounds], u[in_bounds]]
        masked_pc = point_cloud[keep]
        
        masked_pc_path = os.path.join(output_dir, "04_masked_point_cloud.npy")
        np.save(masked_pc_path, masked_pc)
        print(f"  Saved masked point cloud to {masked_pc_path}")
        
        # Save OBB data
        obb_data = {
            "corners_world": obb_corners,
            "corners_pixels": corners_px,
            "extent": extent,
        }
        obb_path = os.path.join(output_dir, "05_obb_data.npy")
        np.save(obb_path, obb_data, allow_pickle=True)
        print(f"  Saved OBB data to {obb_path}")
        
        # Save camera intrinsics
        intrinsics_path = os.path.join(output_dir, "camera_intrinsics.txt")
        with open(intrinsics_path, "w") as f:
            f.write(f"fx: {fx}\n")
            f.write(f"fy: {fy}\n")
            f.write(f"cx: {cx}\n")
            f.write(f"cy: {cy}\n")
            f.write(f"width: {w}\n")
            f.write(f"height: {h}\n")
        print(f"  Saved camera intrinsics to {intrinsics_path}")
        
        # Create 3D visualization
        #viz_3d_path = os.path.join(output_dir, "06_bbox_3d.png")
        #visualize_3d_bbox(masked_pc, obb_corners, viz_3d_path)
        
        # Create masked points overlay
        masked_pixels = _project_points(masked_pc, (h, w), fx, fy, cx, cy)
        points_overlay = create_points_overlay(image, masked_pixels, color="cyan", size=0.5)
        points_overlay_path = os.path.join(output_dir, "07_masked_points_overlay.png")
        points_overlay.save(points_overlay_path)
        print(f"  Saved masked points overlay to {points_overlay_path}")
        
        # Create bbox overlay
        corners_px_array = np.array(corners_px)
        
        # Use edges from bbox tool output
        bbox_overlay = create_bbox_overlay(image, corners_px_array, edges)
        bbox_overlay_path = os.path.join(output_dir, "08_bbox_overlay.png")
        bbox_overlay.save(bbox_overlay_path)
        print(f"  Saved bbox overlay to {bbox_overlay_path}")
        
        # Save PCA OBB data
        obb_pca_data = {
            "corners_world": obb_corners_pca,
            "corners_pixels": corners_px_pca,
            "extent": extent_pca,
        }
        obb_pca_path = os.path.join(output_dir, "09_pca_obb_data.npy")
        np.save(obb_pca_path, obb_pca_data, allow_pickle=True)
        print(f"  Saved PCA OBB data to {obb_pca_path}")
        
        # Create PCA bbox overlay
        corners_px_pca_array = np.array(corners_px_pca)
        
        # Use edges from bbox tool output
        bbox_pca_overlay = create_bbox_overlay(image, corners_px_pca_array, edges_pca)
        bbox_pca_overlay_path = os.path.join(output_dir, "10_pca_bbox_overlay.png")
        bbox_pca_overlay.save(bbox_pca_overlay_path)
        print(f"  Saved PCA bbox overlay to {bbox_pca_overlay_path}")
        
        # Save original image for reference
        image.save(os.path.join(output_dir, "00_original_image.png"))
        
        print(f"\n✓ All outputs saved to {output_dir}/")
        print(f"\nBounding Box Summary (Default):")
        print(f"  - Extent (w×h×d): {extent[0]:.3f} × {extent[1]:.3f} × {extent[2]:.3f} meters")
        print(f"  - Number of points in bbox: {len(masked_pc)}")
        print(f"  - Center (approx): {np.mean(obb_corners, axis=0).tolist()}")
        print(f"\nBounding Box Summary (PCA):")
        print(f"  - Extent (w×h×d): {extent_pca[0]:.3f} × {extent_pca[1]:.3f} × {extent_pca[2]:.3f} meters")
        print(f"  - Center (approx): {np.mean(obb_corners_pca, axis=0).tolist()}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutdown_toolkit()
        print("\nDone.")


if __name__ == "__main__":
    main()
