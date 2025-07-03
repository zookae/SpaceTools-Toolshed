# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

# Use non-GUI backend for headless environments
import matplotlib
matplotlib.use("Agg")

import os
import logging
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

from toolshed.tools.base import BaseTool, tool_method
from toolshed.tool_result import ToolResult

# GraspGen imports (optional - only required when tool is instantiated)
try:
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    from grasp_gen.robot import get_gripper_info
    from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
    GRASPGEN_AVAILABLE = True
except ImportError:
    GRASPGEN_AVAILABLE = False
    # Create dummy placeholders for type hints
    GraspGenSampler = None
    load_grasp_cfg = None
    get_gripper_info = None
    filter_colliding_grasps = None

logger = logging.getLogger(__name__)


def _find_toolshed_root() -> str:
    """Find toolshed root from package location."""
    import toolshed
    return os.path.dirname(os.path.dirname(os.path.abspath(toolshed.__file__)))


# -----------------------------------------------------------------------------
# Top-down filtering constants
# -----------------------------------------------------------------------------
ENABLE_TOPDOWN_FILTERING = True
TOPDOWN_GRAVITY_VECTOR = np.array([0, 0.7, 0.3])
TOPDOWN_GRAVITY_VECTOR = TOPDOWN_GRAVITY_VECTOR / np.linalg.norm(TOPDOWN_GRAVITY_VECTOR)  # Normalize
TOPDOWN_SCORE_THRESHOLD = 0.5

# -----------------------------------------------------------------------------
# Helper functions (copied from bounding_box)
# -----------------------------------------------------------------------------

def _project_points(
    xyz: np.ndarray, image_size: Tuple[int, int], fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Project 3-D points to pixel coordinates."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return np.stack([u, v], axis=1)


def _resolve_pointcloud(pc):
    """Accept numpy array, or path to .npy."""
    if isinstance(pc, str):
        pc = np.load(pc)
    if not (isinstance(pc, np.ndarray) and pc.ndim == 2 and pc.shape[1] == 3):
        raise TypeError("point_cloud must be Nx3 numpy array or path to .npy")
    return pc


def _resolve_image(img):
    if img is None:
        raise ValueError("image is None")
    if isinstance(img, str):
        img = Image.open(img)
    if not isinstance(img, Image.Image):
        raise TypeError("Unsupported image type")
    return img.convert("RGB")


def _resolve_mask(mask):
    if isinstance(mask, str):
        mask = np.array(Image.open(mask).convert("L")) > 127
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask should be numpy array or path")
    return mask.astype(bool)


def _filter_topdown_grasps(
    grasp_poses: np.ndarray,
    grasp_confidences: np.ndarray,
    gravity_vector: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter grasps based on their orientation relative to gravity vector.

    Args:
        grasp_poses: Nx4x4 array of grasp poses (homogeneous transforms)
        grasp_confidences: N-length array of confidence scores
        gravity_vector: 3-element normalized gravity vector
        threshold: Minimum dot product between gravity and grasp z-axis

    Returns:
        Tuple of (filtered_poses, filtered_confidences)
    """
    # Extract z-axis (third column) from rotation matrix for each grasp
    # grasp_poses[:, :3, 2] gives us the z-axis of each grasp
    grasp_z_axes = grasp_poses[:, :3, 2]  # Shape: (N, 3)

    # Compute dot product with gravity vector for each grasp
    topdown_scores = np.dot(grasp_z_axes, gravity_vector)  # Shape: (N,)

    # Keep grasps with score >= threshold
    keep_mask = topdown_scores >= threshold

    logger.info(
        f"Top-down filtering: {len(grasp_poses)} → {keep_mask.sum()} grasps "
        f"(threshold={threshold:.2f}, score range=[{topdown_scores.min():.3f}, {topdown_scores.max():.3f}])"
    )

    return grasp_poses[keep_mask], grasp_confidences[keep_mask]


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def _create_grasp_overlay(
    image: Image.Image,
    grasp_pose: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    axis_len: float = 0.05,
) -> Image.Image:
    """Draw RGB, G, B axes of the gripper origin projected into the image."""
    import matplotlib.pyplot as plt

    # Build 3-D axis points in gripper frame
    axes_points = np.array(
        [
            [0, 0, 0],  # origin
            [axis_len, 0, 0],  # +x
            [0, axis_len, 0],  # +y
            [0, 0, axis_len],  # +z
        ]
    )
    ones = np.ones((4, 1))
    axes_points_h = np.concatenate([axes_points, ones], axis=1)  # 4x4

    # Transform into camera (OpenCV) frame
    cam_pts = (grasp_pose @ axes_points_h.T).T[:, :3]

    # Project
    pts_px = _project_points(cam_pts, image.size[::-1], fx, fy, cx, cy)

    fig, ax = plt.subplots(figsize=(6, 6 * image.height / image.width))
    ax.imshow(image)
    origin = pts_px[0]
    colors = ["r", "g", "b"]
    for i in range(3):
        ax.plot([origin[0], pts_px[i + 1, 0]], [origin[1], pts_px[i + 1, 1]], color=colors[i], linewidth=2)
    ax.set_axis_off()
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(height, width, 4)[..., :3]
    overlay = Image.fromarray(buf.copy())
    plt.close(fig)
    return overlay


def _create_grasp_overlay_gripper(
    image: Image.Image,
    grasp_pose: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    gripper_width: float = 0.16,
    finger_length: float = 0.10,
) -> Image.Image:
    """Draw gripper gizmo with finger geometry projected into the image."""
    import matplotlib.pyplot as plt

    # Define gripper points in gripper frame (same as grasp_to_2d_points)
    gripper_points = np.array([
        [0, 0, 0],  # center
        [gripper_width / 2, 0, 0],  # left finger base
        [-gripper_width / 2, 0, 0],  # right finger base
        [gripper_width / 2, 0, finger_length],  # left finger tip
        [-gripper_width / 2, 0, finger_length],  # right finger tip
    ])

    # Transform to homogeneous coordinates
    gripper_points_h = np.concatenate([gripper_points, np.ones((5, 1))], axis=1)

    # Transform into camera (OpenCV) frame
    cam_pts = (grasp_pose @ gripper_points_h.T).T[:, :3]

    # Project to image coordinates
    pts_px = _project_points(cam_pts, image.size[::-1], fx, fy, cx, cy)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6 * image.height / image.width))
    ax.imshow(image)

    # Extract individual points
    center = pts_px[0]
    left_base = pts_px[1]
    right_base = pts_px[2]
    left_tip = pts_px[3]
    right_tip = pts_px[4]

    # Draw lines connecting the gripper structure
    # Center to finger bases
    ax.plot([center[0], left_base[0]], [center[1], left_base[1]], color='cyan', linewidth=3)
    ax.plot([center[0], right_base[0]], [center[1], right_base[1]], color='cyan', linewidth=3)

    # Finger bases to tips
    ax.plot([left_base[0], left_tip[0]], [left_base[1], left_tip[1]], color='cyan', linewidth=3)
    ax.plot([right_base[0], right_tip[0]], [right_base[1], right_tip[1]], color='cyan', linewidth=3)

    #ax.plot([center[0], left_tip[0]], [center[1], left_tip[1]], color='orange', linewidth=3)
    #ax.plot([center[0], right_tip[0]], [center[1], right_tip[1]], color='orange', linewidth=3)


    # Draw points for emphasis
    #ax.scatter(*center, color='red', s=100, zorder=5, marker='o')
    #ax.scatter([left_base[0], right_base[0]], [left_base[1], right_base[1]],
    #           color='blue', s=80, zorder=5, marker='s')
    #ax.scatter([left_tip[0], right_tip[0]], [left_tip[1], right_tip[1]],
    #           color='green', s=80, zorder=5, marker='^')

    ax.set_axis_off()
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(height, width, 4)[..., :3]
    overlay = Image.fromarray(buf.copy())
    plt.close(fig)
    return overlay


# -----------------------------------------------------------------------------
# Main tool class
# -----------------------------------------------------------------------------

class GraspGeneratorTool(BaseTool):
    """Generate a grasp using GraspGen and project it into the input RGB image."""

    def __init__(
        self,
        gripper_config: str | None = None,
        no_output_image: bool = False,
        no_output_vars: bool = False,
        exclude_methods: list[str] | None = None,
        exclude_behavior: str = "warn",
    ) -> None:
        super().__init__(
            no_output_image=no_output_image,
            no_output_vars=no_output_vars,
            exclude_methods=exclude_methods,
            exclude_behavior=exclude_behavior,
        )

        # Validate that GraspGen is available when instantiating
        if not GRASPGEN_AVAILABLE:
            raise ImportError(
                "GraspGen is not installed. This tool requires graspgen to be available. "
                "Please install it or ensure it's in PYTHONPATH."
            )

        # If gripper_config not provided, look for default in same directory as this file
        if gripper_config is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_config = os.path.join(script_dir, "graspgen_franka_panda.yml")
            if os.path.exists(default_config):
                gripper_config = default_config
            else:
                raise FileNotFoundError(
                    f"Gripper config not provided and default config not found at {default_config}"
                )
        else:
            # If gripper_config is provided but is relative, resolve relative to toolshed root
            gripper_config_expanded = os.path.expanduser(gripper_config)
            if not os.path.isabs(gripper_config_expanded):
                toolshed_root = _find_toolshed_root()
                gripper_config = os.path.join(toolshed_root, gripper_config_expanded)
            else:
                gripper_config = gripper_config_expanded
            
            if not os.path.exists(gripper_config):
                raise FileNotFoundError(f"Gripper config {gripper_config} not found")
        
        self._gripper_config = gripper_config
        self._sampler = self._get_sampler(gripper_config)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    @lru_cache(maxsize=4)
    def _get_sampler(gripper_config: str) -> GraspGenSampler:
        cfg = load_grasp_cfg(gripper_config)
        return GraspGenSampler(cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "grasp_generator"

    # ------------------------------------------------------------------
    # Tool method
    # ------------------------------------------------------------------

    def grasp_to_2d_points(self, grasp_pose: np.ndarray, image: Image.Image, focal_length_px: float) -> np.ndarray:
        """
        Convert a grasp pose to 2D points in the image.
        """
        h, w = image.height, image.width
        fx = fy = focal_length_px
        cx, cy = w / 2.0, h / 2.0

        # Define a set of 3D points in the gripper frame
        gripper_width = 0.16
        finger_length = 0.08
        # Example answer: Grasp center: [0.664, 0.720], Left finger base: [0.681, 0.728], Right finger base: [0.663, 0.754], Left finger tip: [0.646, 0.712], Right finger tip: [0.628, 0.739]
        # The 5 points are: gripper center, left finger base, right finger base, left finger tip, right finger tip
        gripper_points = np.array([
            [0, 0, 0],
            [gripper_width / 2, 0, 0],
            [-gripper_width / 2, 0, 0 ],
            [gripper_width / 2, 0, finger_length],
            [-gripper_width / 2, 0, finger_length],
        ])

        # Transform the gripper points to the camera frame
        gripper_points_h = np.concatenate([gripper_points, np.ones((5, 1))], axis=1)
        gripper_points_cam = (grasp_pose @ gripper_points_h.T).T[:, :3]
        gripper_points_px = _project_points(gripper_points_cam, (h, w), fx, fy, cx, cy)
        gripper_points_normalized = gripper_points_px / np.array([w, h])[None, :]
        return gripper_points_normalized

    @tool_method
    def compute_grasp(
        self,
        point_cloud,
        mask,
        image,
        focal_length_px: float
    ) -> ToolResult:
        """
        Generate a single grasp pose for a masked subset of a point cloud.

        [[if:text]]Text output: Confidence score, number of masked points used, projected 2D gripper points in normalize image coordinates.[[/if:text]]
        [[if:image]]Image output: Index 0 – RGB overlay with projected X-(red), Y-(green), Z-(blue) gripper axes.[[/if:image]]
        [[if:vars]]Stored variables: $grasp_pose (4×4 ndarray, OpenCV camera frame).[[/if:vars]]

        Args:
            point_cloud: Nx3 numpy float array. Full scene point cloud.
            mask: Boolean mask aligning with *image*; accepts ndarray. Indicates object points.
            image: RGB image matching the point cloud frame (PIL / path).
            focal_length_px: Camera focal length in pixels (square pixels assumed).

        Returns:
            ToolResult: value dict with ``grasp_pose`` + ``grasp_confidence``; images and variables as noted above.
        """

        # grasp_threshold: Threshold for valid grasps; if -1 returns highest-confidence grasp.
        # num_grasps: Number of candidate grasps to sample before selection.
        # collision_threshold: Distance threshold for collision detection in meters (default 0.02 = 2cm).
        # max_scene_points: Maximum scene points for collision checking. Downsamples if exceeded (default 8192).

        grasp_threshold: float = -1.0
        num_grasps: int = 200
        collision_threshold: float = 0.01
        max_scene_points: int = 8192

        # Resolve inputs
        pts = _resolve_pointcloud(point_cloud)
        img = _resolve_image(image)
        mask_array = _resolve_mask(mask)

        h, w = img.size[1], img.size[0]
        fx = fy = focal_length_px
        cx, cy = w / 2.0, h / 2.0

        # --------------------------------------------------------------
        # Apply mask to point cloud
        # --------------------------------------------------------------
        pixels = _project_points(pts, (h, w), fx, fy, cx, cy)
        u = pixels[:, 0].astype(int)
        v = pixels[:, 1].astype(int)
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        keep = np.zeros(len(pts), dtype=bool)
        keep[in_bounds] = mask_array[v[in_bounds], u[in_bounds]]
        obj_pts = pts[keep]
        if len(obj_pts) == 0:
            raise RuntimeError("Mask removed all points; cannot generate grasp.")

        # --------------------------------------------------------------
        # Run GraspGen inference
        # --------------------------------------------------------------
        grasps, grasp_conf = GraspGenSampler.run_inference(
            obj_pts,
            self._sampler,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=100,
        )
        if len(grasps) == 0:
            raise RuntimeError("Grasp generation produced zero grasps")

        # --------------------------------------------------------------
        # Top-down filtering (optional)
        # --------------------------------------------------------------
        # Convert to numpy for filtering
        grasps_np = grasps.cpu().numpy()
        grasp_conf_np = grasp_conf.cpu().numpy()

        # Ensure homogeneous coordinates are correct
        grasps_np[:, 3, 3] = 1

        if ENABLE_TOPDOWN_FILTERING:
            grasps_np, grasp_conf_np = _filter_topdown_grasps(
                grasps_np,
                grasp_conf_np,
                TOPDOWN_GRAVITY_VECTOR,
                TOPDOWN_SCORE_THRESHOLD,
            )
            if len(grasps_np) == 0:
                raise RuntimeError(
                    f"Top-down filtering removed all grasps. "
                    f"No grasps have z-axis aligned with gravity vector (threshold={TOPDOWN_SCORE_THRESHOLD})."
                )

        # --------------------------------------------------------------
        # Filter out colliding grasps
        # --------------------------------------------------------------
        # Load gripper collision mesh
        cfg = load_grasp_cfg(self._gripper_config)
        gripper_info = get_gripper_info(cfg.data.gripper_name)

        # Print statistics before filtering
        logger.info(f"Generated {len(grasps_np)} grasps from {len(obj_pts)} object points")
        logger.info(f"Confidence range: [{grasp_conf_np.min():.3f}, {grasp_conf_np.max():.3f}]")

        # Downsample scene point cloud for faster collision checking
        if len(pts) > max_scene_points:
            logger.info(
                f"Downsampling scene point cloud: {len(pts)} → {max_scene_points} points "
                f"({100.0 * max_scene_points / len(pts):.1f}% retained)"
            )
            indices = np.random.choice(len(pts), max_scene_points, replace=False)
            scene_pc_for_collision = pts[indices]
        else:
            scene_pc_for_collision = pts
            logger.info(f"Scene point cloud: {len(pts)} points (no downsampling needed)")

        # Filter colliding grasps
        logger.info(f"Filtering grasps with collision threshold: {collision_threshold:.4f}m")
        collision_free_mask = filter_colliding_grasps(
            scene_pc=scene_pc_for_collision,
            grasp_poses=grasps_np,
            gripper_collision_mesh=gripper_info.collision_mesh,
            collision_threshold=collision_threshold,
        )

        # Keep only collision-free grasps
        collision_free_grasps = grasps_np[collision_free_mask]
        collision_free_conf = grasp_conf_np[collision_free_mask]

        # Print filtering statistics
        num_colliding = len(grasps_np) - len(collision_free_grasps)
        collision_rate = 100.0 * num_colliding / len(grasps_np)
        logger.info(
            f"Collision filtering: {len(grasps_np)} → {len(collision_free_grasps)} "
            f"({num_colliding} colliding, {collision_rate:.1f}% filtered)"
        )

        if len(collision_free_grasps) == 0:
            logger.error(
                f"No collision-free grasps found! All {len(grasps_np)} grasps collide with scene."
            )
            raise RuntimeError(
                f"No collision-free grasps found. All grasps collide with other objects."
            )

        logger.info(
            f"Collision-free confidence range: [{collision_free_conf.min():.3f}, {collision_free_conf.max():.3f}]"
        )

        # Take best collision-free grasp (highest confidence)
        best_idx = int(np.argmax(collision_free_conf))
        best_grasp = collision_free_grasps[best_idx]  # 4x4
        best_conf = float(collision_free_conf[best_idx])
        logger.info(f"Selected best grasp with confidence: {best_conf:.3f}")

        # --------------------------------------------------------------
        # Visualization
        # --------------------------------------------------------------
        images: Optional[List[Image.Image]] = None
        projected_axes: Optional[np.ndarray] = None
        if not self.no_output_image:
            overlay = _create_grasp_overlay_gripper(img, best_grasp, fx, fy, cx, cy)
            images = [overlay]
            # Return projected axes for downstream
            axes_points = np.array(
                [[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]
            )
            axes_points_h = np.concatenate([axes_points, np.ones((4, 1))], axis=1)
            cam_pts = (best_grasp @ axes_points_h.T).T[:, :3]
            projected_axes = _project_points(cam_pts, (h, w), fx, fy, cx, cy)

        value = {
            "grasp_pose": best_grasp.tolist(),
            "grasp_confidence": best_conf,
            "projected_axes_px": projected_axes.tolist() if projected_axes is not None else None,
        }

        # Gripper points are in normalized image coordinates to 3 decimal places
        projected_gripper_points = self.grasp_to_2d_points(best_grasp, img, focal_length_px).tolist()
        projected_gripper_points_str = "[" + ", ".join([f"({x:.3f}, {y:.3f})" for x, y in projected_gripper_points]) + "]"
        text = (
            f"Generated collision-free grasp with confidence {best_conf:.3f} from {len(obj_pts)}/{len(pts)} masked points. "
            f"Filtered {len(grasps)} → {len(collision_free_grasps)} collision-free grasps. "
            f"Projected 2D gripper points: {projected_gripper_points_str}"
        )

        variables = {} if self.no_output_vars else {"grasp_pose": best_grasp.tolist()}
        images = None if self.no_output_image else images

        return ToolResult(value, text=text, image=images, variables=variables)
