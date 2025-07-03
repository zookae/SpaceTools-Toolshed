# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
try:
    import open3d as o3d
    import cv2
except ImportError:
    o3d = None
    cv2 = None

from toolshed.tools.base import BaseTool, tool_method
from toolshed.tool_result import ToolResult

import ray  # for ObjectRef resolution

logger = logging.getLogger(__name__)

def get_edges():
    return [
        # Bottom face (z-)
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face (z+)
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

def _project_points(
    xyz: np.ndarray, image_size: Tuple[int, int], fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Project 3-D points to pixel coords."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return np.stack([u, v], axis=1)


def _opencv_to_opengl(xyz: np.ndarray) -> np.ndarray:
    """Convert points from OpenCV camera frame to OpenGL camera frame.

    Convention:
    - OpenCV camera: +x right, +y down, +z forward
    - OpenGL camera: +x right, +y up,   +z backward (camera looks along -z)

    Mapping: (x, y, z)_cv -> (x, -y, -z)_gl
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be Nx3 array")
    out = xyz.copy()
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out


def _opengl_to_opencv(xyz: np.ndarray) -> np.ndarray:
    """Inverse of _opencv_to_opengl: (x, y, z)_gl -> (x, -y, -z)_cv."""
    return _opencv_to_opengl(xyz)


class BoundingBoxTool(BaseTool):
    """Compute oriented bounding box for a point cloud with optional mask."""

    def __init__(self, no_output_image: bool = False, no_output_vars: bool = False,
                 exclude_methods: list[str] = None, exclude_behavior: str = "warn") -> None:
        super().__init__(no_output_image=no_output_image, no_output_vars=no_output_vars,
                         exclude_methods=exclude_methods, exclude_behavior=exclude_behavior)

    def get_name(self) -> str:
        return "bounding_box"

    # ------------------------------------------------------------------
    # Public tool method
    # ------------------------------------------------------------------

    # #@tool_method
    # def compute_bbox_3d_only(
    #     self,
    #     point_cloud,
    #     mask,
    #     focal_length_px: float,
    # ) -> ToolResult:
    #     """
    #     Compute an oriented bounding box for a masked subset of a point cloud.

    #     [[if:text]]Text output: Summary containing number of input points, the point coordinates in 3d and 2d,
    #     mask shape, box extents, and edges.[[/if:text]]
    #     [[if:vars]]Stored variables: 
    #         - $obb_corners_3d (8×3 list of lists, meters in opencv camera frame)
    #         - $extent (3-element ndarray, extent of the bounding box in meters)
    #         - $edges (list of pairs of integers, edges of the bounding box defined by the indices of the corners)[[/if:vars]]

    #     Args:
    #         point_cloud: scene point cloud, np.ndarray of shape (N, 3) with float values.
    #         mask: np.ndarray of shape (H, W) with boolean values segmenting the target object. Dimensions are used for camera projection.
    #         focal_length_px: Camera focal length in pixels (square pixels assumed).

    #     Returns:
    #         Dict containing:
    #         - obb_corners_3d (8×3 list of lists, meters in opencv camera frame)
    #         - extent (3-element list of floats, extent of the bounding box in meters)
    #         - edges (list of tuples of integers, edges of the bounding box defined by the indices of the corners)
    #     """

    #     pts = self._resolve_pointcloud(point_cloud)
    #     if pts.ndim != 2 or pts.shape[1] != 3:
    #         raise ValueError("Point cloud must be Nx3 array")

    #     mask_array = self._resolve_mask(mask)
    #     h, w = mask_array.shape

    #     # Erode the mask by 5px to avoid edge effects
    #     mask_array = cv2.erode(mask_array.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool)

    #     fx = fy = focal_length_px
    #     cx, cy = w / 2.0, h / 2.0

    #     logger.debug(f"Point cloud shape={pts.shape}, dtype={pts.dtype}")
    #     logger.debug(f"Mask size (w,h)=({w},{h})  focal_length_px={focal_length_px}")
    #     logger.debug(f"Mask shape={mask_array.shape}, dtype={mask_array.dtype}")

    #     # Use OpenCV-frame points to project and apply the mask
    #     pixels = _project_points(pts, (h, w), fx, fy, cx, cy)
    #     u = pixels[:, 0].astype(int)
    #     v = pixels[:, 1].astype(int)
    #     in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    #     keep = np.zeros(len(pts), dtype=bool)
    #     keep[in_bounds] = mask_array[v[in_bounds], u[in_bounds]]
    #     asset_pts = pts[keep]
    #     if len(asset_pts) == 0:
    #         raise RuntimeError(
    #             f"Mask removed all points (total {len(pts)}, kept 0). Check intrinsics and mask alignment."
    #         )

    #     # Convert selected points to OpenGL camera frame for geometry/OBB
    #     asset_pts_gl = _opencv_to_opengl(asset_pts)

    #     # Compute OBB in OpenGL camera frame
    #     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(asset_pts_gl))
    #     obb = pcd.get_minimal_oriented_bounding_box()

    #     # Project OBB corners to pixels for downstream use
    #     corners_gl = np.asarray(obb.get_box_points())
    #     corners_cv = _opengl_to_opencv(corners_gl)
    #     corners_px = _project_points(corners_cv, (h, w), fx, fy, cx, cy)

    #     edges = get_edges()

    #     value = {
    #         # Corners are returned in OpenGL camera coordinates
    #         "obb_corners_3d": corners_cv.tolist(),
    #         "extent": obb.extent.tolist(),
    #         "edges": edges
    #     }
    #     text = (
    #         f"Oriented bounding box computed from {len(asset_pts)}/{len(pts)} points."
    #         f"Corners in 3D: {np.around(corners_cv, 3).tolist()}\n"
    #         f"Edges: {edges}\n"
    #         f"Extent: {np.around(obb.extent, 3).tolist()}\n"
    #         f"Volume: {obb.volume():.3f}\n"
    #     )
    #     self._stats["total_calls"] += 1
    #     self._stats["bbox_calls"] += 1

    #     # Prepare variables if not suppressed
    #     variables = {} if self.no_output_vars else {
    #         "obb_corners_3d": value["obb_corners_3d"],
    #         "edges": value["edges"],
    #     }

    #     return ToolResult(value, text=text, variables=variables)

    @tool_method
    def compute_bbox(
        self,
        point_cloud,
        mask,
        focal_length_px: float,
    ) -> ToolResult:
        """
        Compute an oriented bounding box for a masked subset of a point cloud.

        [[if:text]]Text output: Summary containing number of input points, the point coordinates in 3d and 2d,
        mask shape, box extents, and edges.[[/if:text]]
        [[if:vars]]Stored variables: 
            - $obb_corners_3d (8×3 list of lists, meters in opencv camera frame)
            - $obb_corners_2d (8×2 list of lists, normalized image coordinates)
            - $extent (3-element ndarray, extent of the bounding box in meters)
            - $edges (list of pairs of integers, edges of the bounding box defined by the indices of the corners)[[/if:vars]]

        Args:
            point_cloud: scene point cloud, np.ndarray of shape (N, 3) with float values.
            mask: np.ndarray of shape (H, W) with boolean values segmenting the target object. Dimensions are used for camera projection.
            focal_length_px: Camera focal length in pixels (square pixels assumed).

        Returns:
            Dict containing:
            - obb_corners_3d (8×3 list of lists, meters in opencv camera frame)
            - obb_corners_2d (8×2 list of lists, normalized image coordinates)
            - extent (3-element list of floats, extent of the bounding box in meters)
            - edges (list of tuples of integers, edges of the bounding box defined by the indices of the corners)
        """
        import open3d as o3d
        import cv2

        pts = self._resolve_pointcloud(point_cloud)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Point cloud must be Nx3 array")

        mask_array = self._resolve_mask(mask)
        h, w = mask_array.shape

        # Erode the mask by 5px to avoid edge effects
        #mask_array = cv2.erode(mask_array.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool)

        fx = fy = focal_length_px
        cx, cy = w / 2.0, h / 2.0

        logger.debug(f"Point cloud shape={pts.shape}, dtype={pts.dtype}")
        logger.debug(f"Mask size (w,h)=({w},{h})  focal_length_px={focal_length_px}")
        logger.debug(f"Mask shape={mask_array.shape}, dtype={mask_array.dtype}")

        # Use OpenCV-frame points to project and apply the mask
        pixels = _project_points(pts, (h, w), fx, fy, cx, cy)
        u = pixels[:, 0].astype(int)
        v = pixels[:, 1].astype(int)
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        keep = np.zeros(len(pts), dtype=bool)
        keep[in_bounds] = mask_array[v[in_bounds], u[in_bounds]]
        asset_pts = pts[keep]
        if len(asset_pts) == 0:
            raise RuntimeError(
                f"Mask removed all points (total {len(pts)}, kept 0). Check intrinsics and mask alignment."
            )

        # Convert selected points to OpenGL camera frame for geometry/OBB
        asset_pts_gl = _opencv_to_opengl(asset_pts)

        # Compute OBB in OpenGL camera frame
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(asset_pts_gl))
        obb = pcd.get_minimal_oriented_bounding_box()

        # Project OBB corners to pixels for downstream use
        corners_gl = np.asarray(obb.get_box_points())
        corners_cv = _opengl_to_opencv(corners_gl)
        corners_px = _project_points(corners_cv, (h, w), fx, fy, cx, cy)
        corners_2d_norm = corners_px / (np.asarray([w, h])[None, :])

        edges = get_edges()

        value = {
            # Corners are returned in OpenGL camera coordinates
            "obb_corners_3d": corners_cv.tolist(),
            "extent": obb.extent.tolist(),
            "edges": edges,
            "obb_corners_2d": corners_2d_norm.tolist(),
        }
        text = (
            f"Oriented bounding box computed from {len(asset_pts)}/{len(pts)} points."
            f"Corners in 3D: {np.around(corners_cv, 3).tolist()}\n"
            f"Corners in normalized image coordinates: {np.around(corners_2d_norm, 3).tolist()}\n"
            f"Edges: {edges}\n"
            f"Extent: {np.around(obb.extent, 3).tolist()}\n"
            f"Volume: {obb.volume():.3f}\n"
        )

        # Prepare variables if not suppressed
        variables = {} if self.no_output_vars else {
            "obb_corners_3d": value["obb_corners_3d"],
            "obb_corners_2d": value["obb_corners_2d"],
            "edges": value["edges"],
        }

        return ToolResult(value, text=text, variables=variables)

    #@tool_method
    # def compute_bbox_pca(self, point_cloud, mask, focal_length_px: float) -> ToolResult:
    #     """
    #     Compute an oriented bounding box for a masked subset of a point cloud using PCA.

    #     [[if:text]]Text output: Summary containing number of input points, the point coordinates in 3d and 2d,
    #     mask shape, box extents, and edges.[[/if:text]]
    #     [[if:vars]]Stored variables: 
    #         - $obb_corners_3d (8×3 list of lists, meters in opencv camera frame)
    #         - $obb_corners_2d (8×2 list of lists, normalized image coordinates)
    #         - $extent (3-element ndarray, extent of the bounding box in meters)
    #         - $edges (list of pairs of integers, edges of the bounding box defined by the indices of the corners)[[/if:vars]]

    #     Args:
    #         point_cloud: np.ndarray of shape (N, 3) with float values.
    #         mask: np.ndarray of shape (H, W) with boolean values. Dimensions are used for camera projection.
    #         focal_length_px: Camera focal length in pixels (square pixels assumed). Required for projection.

    #     Returns:
    #         Dict containing:
    #         - obb_corners_3d (8×3 list of lists, meters in opencv camera frame)
    #         - obb_corners_2d (8×2 list of lists, normalized image coordinates)
    #         - extent (3-element list of floats, extent of the bounding box in meters)
    #         - edges (list of tuples of integers, edges of the bounding box defined by the indices of the corners)
    #     """
    #     pts = self._resolve_pointcloud(point_cloud)
    #     if pts.ndim != 2 or pts.shape[1] != 3:
    #         raise ValueError("Point cloud must be Nx3 array")

    #     mask_array = self._resolve_mask(mask)
    #     # Erode the mask by 5px to avoid edge effects
    #     mask_array = cv2.erode(mask_array.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    #     h, w = mask_array.shape

    #     fx = fy = focal_length_px
    #     cx, cy = w / 2.0, h / 2.0

    #     logger.debug(f"Point cloud shape={pts.shape}, dtype={pts.dtype}")
    #     logger.debug(f"Mask size (w,h)=({w},{h})  focal_length_px={focal_length_px}")
    #     logger.debug(f"Mask shape={mask_array.shape}, dtype={mask_array.dtype}")

    #     # Project points and apply mask (points remain in OpenCV frame)
    #     pixels = _project_points(pts, (h, w), fx, fy, cx, cy)
    #     u = pixels[:, 0].astype(int)
    #     v = pixels[:, 1].astype(int)
    #     in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    #     keep = np.zeros(len(pts), dtype=bool)
    #     keep[in_bounds] = mask_array[v[in_bounds], u[in_bounds]]
    #     asset_pts = pts[keep]
    #     if len(asset_pts) == 0:
    #         raise RuntimeError(
    #             f"Mask removed all points (total {len(pts)}, kept 0). Check intrinsics and mask alignment."
    #         )

    #     # Compute PCA to find principal axes
    #     centroid = np.mean(asset_pts, axis=0)
    #     centered_pts = asset_pts - centroid
        
    #     # Covariance matrix and eigendecomposition
    #     cov = np.cov(centered_pts.T)
    #     eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
    #     # Sort by eigenvalues in descending order
    #     idx = eigenvalues.argsort()[::-1]
    #     eigenvalues = eigenvalues[idx]
    #     eigenvectors = eigenvectors[:, idx]
        
    #     # Transform points to PCA coordinate system
    #     pts_pca = centered_pts @ eigenvectors
        
    #     # Find extents along each principal axis
    #     min_vals = np.min(pts_pca, axis=0)
    #     max_vals = np.max(pts_pca, axis=0)
    #     extent = max_vals - min_vals
        
    #     # Generate 8 corners of the bounding box in PCA space
    #     # Ordering matches Open3D's OrientedBoundingBox::GetBoxPoints()
    #     # where x_axis, y_axis, z_axis are half-extents along principal directions
    #     corners_pca = np.array([
    #         [min_vals[0], min_vals[1], min_vals[2]],  # 0: center - x - y - z
    #         [max_vals[0], min_vals[1], min_vals[2]],  # 1: center + x - y - z
    #         [min_vals[0], max_vals[1], min_vals[2]],  # 2: center - x + y - z
    #         [min_vals[0], min_vals[1], max_vals[2]],  # 3: center - x - y + z
    #         [max_vals[0], max_vals[1], max_vals[2]],  # 4: center + x + y + z
    #         [min_vals[0], max_vals[1], max_vals[2]],  # 5: center - x + y + z
    #         [max_vals[0], min_vals[1], max_vals[2]],  # 6: center + x - y + z
    #         [max_vals[0], max_vals[1], min_vals[2]],  # 7: center + x + y - z
    #     ])
        
    #     # Transform corners back to original OpenCV camera frame
    #     corners_cv = corners_pca @ eigenvectors.T + centroid
        
    #     # Project corners to pixels
    #     corners_px = _project_points(corners_cv, (h, w), fx, fy, cx, cy)

    #     corners_2d_norm = corners_px / (np.asarray([h, w])[None, :])

    #     edges = get_edges()

    #     value = {
    #         # Corners are returned in OpenCV camera coordinates
    #         "obb_corners_3d": corners_cv.tolist(),
    #         "extent": extent.tolist(),
    #         "edges": edges,
    #         "obb_corners_2d": corners_2d_norm.tolist(),
    #     }
    #     text = (
    #         f"PCA-based oriented bounding box computed from {len(asset_pts)}/{len(pts)} points.\n"
    #         f"Corners in 3D: {np.around(corners_cv, 3).tolist()}\n"
    #         f"Corners in normalized image coordinates: {np.around(corners_2d_norm, 3).tolist()}\n"
    #         f"Edges: {edges}\n"
    #         f"Extent: {np.around(extent, 3).tolist()}\n"
    #     )
    #     self._stats["total_calls"] += 1
    #     self._stats["bbox_calls"] += 1

    #     # Prepare variables if not suppressed
    #     variables = {} if self.no_output_vars else {
    #         "obb_corners_3d": value["obb_corners_3d"],
    #         "obb_corners_2d": value["obb_corners_2d"],
    #         "edges": value["edges"],
    #     }

    #     return ToolResult(value, text=text, variables=variables)



    # ------------------------------------------------------------------
    # Helpers to mimic conventions in other tools
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_pointcloud(pc):
        """Accept numpy array, ObjectRef, or path to .npy."""
        if isinstance(pc, ray.ObjectRef):
            pc = ray.get(pc)
        elif isinstance(pc, str):
            raise TypeError("point_cloud must be Nx3 numpy array, got str. Did you use the $ syntax to access the variable?")
        elif isinstance(pc, np.ndarray):
            if pc.ndim != 2 or pc.shape[1] != 3:
                raise TypeError("point_cloud must be Nx3 numpy array, got array with shape {pc.shape}")
            return pc
        else:
            raise ValueError(f"point_cloud got an unexpected type: {type(pc)}. Expected numpy ndarray.")

    @staticmethod
    def _resolve_mask(mask):
        if isinstance(mask, ray.ObjectRef):
            mask = ray.get(mask)
        elif isinstance(mask, str):
            raise TypeError("mask must be numpy array, got str. Did you use the $ syntax to access the variable?")
        #if isinstance(mask, str):
        #    from PIL import Image
        #    mask = np.array(Image.open(mask).convert("L")) > 127
        elif not isinstance(mask, np.ndarray):
            raise ValueError(f"mask got an unexpected type: {type(mask)}. Expected numpy ndarray.")
        return mask.astype(bool)
