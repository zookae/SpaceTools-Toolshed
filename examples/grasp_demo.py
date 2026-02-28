#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Complete grasp generation demo - from image and text query to grasp pose.

This script demonstrates the full pipeline:
1. Detect object using RoboRefer (VLM-based detection)
2. Segment object using SAM2
3. Estimate depth and generate point cloud
4. Compute grasp pose using GraspGen
5. Visualize results (2D overlays and 3D point cloud with grasp)

Usage:
    python examples/grasp_demo.py --image path/to/image.jpg --object "bok choy"
    
    # With custom model paths:
    python examples/grasp_demo.py --image path/to/image.jpg --object "apple" \
        --roborefer-model /path/to/model --gripper-config /path/to/gripper.yml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ray
from PIL import Image

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from toolshed import start_toolkit, get_toolkit, shutdown_toolkit
from toolshed.tools.grasp_generator import _project_points

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', force=True)
logger = logging.getLogger(__name__)



def main(args) -> None:
    """Run the complete grasp generation pipeline."""

    # Setup paths
    if args.image is None:
        img_path = ROOT_DIR / "examples" / "media" / "example_image.jpg"
    else:
        img_path = args.image
    
    if not img_path.exists():
        logger.error(f"Image not found: {img_path}")
        sys.exit(1)

    if not args.gripper_config.exists():
        logger.error(f"Gripper config not found: {args.gripper_config}")
        sys.exit(1)

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    logger.info(f"Loading image: {img_path}")
    image = Image.open(img_path).convert("RGB")
    image.save(output_dir / "input_image.png")

    # Configure all tools
    tool_configs = {
        "roborefer": {
            "num_actors": 1,
            "resources": {"num_gpus": 1},
            "conda_env": "tool_roborefer",
            "args": {
                "model_path": str(args.roborefer_model),
                "no_output_image": False,
                "no_output_vars": True,
                "exclude_methods": ["general_query"],
                "exclude_behavior": "error"
            },
        },
        "sam2": {
            "num_actors": 1,
            "resources": {"num_gpus": 1},
            "conda_env": "tool_sam2",
            "timeout": 120,
        },
        "depth_estimator": {
            "num_actors": 1,
            "conda_env": "tool_depth",
            "resources": {"num_gpus": 1},
            "timeout": 100,
            "args": {
                "checkpoint_path": str(args.depth_checkpoint)
            }
        },
        "grasp_generator": {
            "num_actors": 1,
            "resources": {"num_gpus": 1},
            "conda_env": "tool_graspgen",
            "timeout": 180,  # Grasp generation can take a while
            "args": {
                "gripper_config": str(args.gripper_config),
                "no_output_image": False,
                "no_output_vars": False,
            }
        }
    }

    # Override with JSON config if provided
    if args.config is not None:
        import json
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
        logger.info(f"Loading tool configs from {config_path}")
        with open(config_path, "r") as f:
            tool_configs = json.load(f)

    # Start toolkit
    logger.info("="*60)
    logger.info("Starting toolkit with all tools...")
    logger.info("="*60)
    handle = start_toolkit(tool_configs)

    try:
        toolkit = get_toolkit()

        # Step 1: Detect object with RoboRefer
        logger.info(f"\n[1/4] Detecting '{args.object}' with RoboRefer...")
        detect_result = toolkit.roborefer.detect_one(image, args.object)
        
        # Check for errors
        if detect_result.is_error:
            logger.error(f"Object detection failed: {detect_result.text}")
            sys.exit(1)
        
        points = detect_result.value
        logger.info(f"Detected point: {points}")
        
        if detect_result.image:
            overlay = detect_result.image[0] if isinstance(detect_result.image, list) else detect_result.image
            overlay.save(output_dir / "01_detection.png")
            logger.info(f"Saved detection overlay: {output_dir / '01_detection.png'}")
        
        if not points or len(points) == 0:
            logger.error("No object detected!")
            sys.exit(1)

        # Step 2: Segment with SAM2
        logger.info(f"\n[2/4] Segmenting object with SAM2...")
        seg_result = toolkit.sam2.segment_from_point(image, x=points[0][0], y=points[0][1])
        
        # Check for errors
        if seg_result.is_error:
            logger.error(f"Segmentation failed: {seg_result.text}")
            sys.exit(1)
        
        seg_mask = seg_result.value["mask"]
        logger.info(f"Segmentation complete. Mask shape: {seg_mask.shape}")
        
        if seg_result.image:
            overlay = seg_result.image[0]
            overlay.save(output_dir / "02_segmentation.png")
            logger.info(f"Saved segmentation overlay: {output_dir / '02_segmentation.png'}")

        # Step 3: Estimate depth and generate point cloud
        logger.info(f"\n[3/4] Estimating depth and generating point cloud...")
        depth_result = toolkit.depth_estimator.estimate_depth_with_pointcloud(image)
        
        # Check for errors
        if depth_result.is_error:
            logger.error(f"Depth estimation failed: {depth_result.text}")
            sys.exit(1)
        
        depth_map = depth_result.value["depth_map"]
        focal_length_px = depth_result.value["focal_length_px"]
        pointcloud = depth_result.value["point_cloud"]
        
        logger.info(f"Point cloud shape: {pointcloud.shape}")
        logger.info(f"Focal length: {focal_length_px:.2f} px")
        
        if depth_result.image:
            overlay = depth_result.image[0]
            overlay.save(output_dir / "03_depth_map.png")
            logger.info(f"Saved depth map: {output_dir / '03_depth_map.png'}")

        # Step 4: Compute grasp with GraspGen
        logger.info(f"\n[4/4] Computing grasp pose with GraspGen...")
        
        grasp_result = toolkit.grasp_generator.compute_grasp(
            point_cloud=pointcloud,
            mask=seg_mask,
            image=image,
            focal_length_px=focal_length_px,
        )
        
        # Check if tool returned an error
        if grasp_result.is_error:
            logger.error(f"Grasp generation failed: {grasp_result.text}")
            sys.exit(1)

        # Display results
        logger.info("\n" + "="*60)
        logger.info("GRASP GENERATION COMPLETE")
        logger.info("="*60)
        logger.info(f"\n{grasp_result.text}")
        
        grasp_pose = np.array(grasp_result.value['grasp_pose'])
        grasp_confidence = grasp_result.value['grasp_confidence']
        
        logger.info(f"\nGrasp confidence: {grasp_confidence:.4f}")
        logger.info(f"Grasp pose (4x4 matrix):\n{grasp_pose}")

        # Save grasp overlay
        if grasp_result.image:
            overlay_image = grasp_result.image[0] if isinstance(grasp_result.image, list) else grasp_result.image
            overlay_image.save(output_dir / "04_grasp_overlay.png")
            logger.info(f"Saved grasp overlay: {output_dir / '04_grasp_overlay.png'}")

        # Save grasp pose
        np.save(output_dir / "grasp_pose.npy", grasp_pose)
        logger.info(f"Saved grasp pose: {output_dir / 'grasp_pose.npy'}")

        # Summary
        logger.info("\n" + "="*60)
        logger.info("✓ DEMO COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"\nAll outputs saved to: {output_dir}/")
        logger.info("\nGenerated files:")
        logger.info(f"  - input_image.png       : Original input image")
        logger.info(f"  - 01_detection.png      : Object detection result")
        logger.info(f"  - 02_segmentation.png   : SAM2 segmentation mask")
        logger.info(f"  - 03_depth_map.png      : Depth estimation")
        logger.info(f"  - 04_grasp_overlay.png  : Grasp pose on image")
        logger.info(f"  - grasp_pose.npy        : Grasp pose matrix (4x4)")

    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}", exc_info=True)
        raise

    finally:
        logger.info("\nShutting down toolkit...")
        shutdown_toolkit()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Complete grasp generation demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--image", type=Path, help="Path to input image (default: examples/media/example_image.jpg)")
    p.add_argument("--object", type=str, default="alarm clock", help="Object name to detect and grasp (default: 'alarm clock')")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/grasp_demo"), help="Directory for output files (default: outputs/grasp_demo)")
    p.add_argument("--roborefer-model", type=Path, default=ROOT_DIR / "checkpoints" / "RoboRefer-8B-SFT", help="Path to RoboRefer checkpoint directory")
    p.add_argument("--gripper-config", type=Path, default=ROOT_DIR / "toolshed" / "tools" / "graspgen_franka_panda.yml", help="Path to gripper config file")
    p.add_argument("--depth-checkpoint", type=Path, default=ROOT_DIR / "checkpoints" / "depth_pro.pt", help="Path to depth estimator checkpoint")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config file for tool configurations. If provided, overrides the inline tool configs (including conda_env names).")
    args = p.parse_args()
    main(args)
