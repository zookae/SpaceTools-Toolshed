#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Web server for viewing multimodal validation logs with conversation history and images.
Usage: python view_multimodal_server.py <validation_outputs_dir> [--port PORT]
"""

import argparse
import json
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import base64
from io import BytesIO
import html
import re
import ast
from PIL import Image, ImageDraw
import numpy as np
from scipy.spatial import ConvexHull


def load_jsonl(filepath):
    """Load a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def parse_predicted_point(output_text):
    """Parse predicted point from model output text.
    
    Args:
        output_text: String containing model output with <answer>[(x, y)]</answer> format
        or any text containing a (x, y) coordinate pattern
        
    Returns:
        tuple: (x, y) coordinates as floats, or None if not found
    """
    if not output_text:
        return None
    
    # First, try to extract content between <answer> and </answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        
        try:
            # Try to parse as Python literal (list format)
            coordinates = ast.literal_eval(answer_content)
            if isinstance(coordinates, list) and len(coordinates) > 0:
                point = coordinates[0]
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    return (float(point[0]), float(point[1]))
        except (ValueError, SyntaxError):
            pass
        
        # If literal_eval failed, try regex pattern matching on answer content
        content_to_search = answer_content
    else:
        # No <answer> tags found, search the entire text
        content_to_search = output_text
    
    # Fallback: Find first (x, y) pattern in the text
    # Pattern matches: (number, number) where numbers can be int or float
    pattern = r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)'
    match = re.search(pattern, content_to_search)
    
    if match:
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            return (x, y)
        except ValueError:
            pass
    
    return None


def parse_ground_truth_points(ground_truth_text):
    """Parse ground truth points from string.
    
    Args:
        ground_truth_text: String containing list of coordinate tuples
        
    Returns:
        list: List of (x, y) tuples, or empty list if parsing fails
    """
    if not ground_truth_text:
        return []
    
    # First, try to parse as Python literal (list format)
    try:
        points = ast.literal_eval(ground_truth_text)
        if isinstance(points, list):
            result = []
            for point in points:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    result.append((float(point[0]), float(point[1])))
            if result:  # Only return if we successfully parsed points
                return result
    except (ValueError, SyntaxError):
        pass
    
    # Fallback: Find all (x, y) patterns in the text
    # Pattern matches: (number, number) where numbers can be int or float
    pattern = r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)'
    matches = re.findall(pattern, ground_truth_text)
    
    if matches:
        result = []
        for x_str, y_str in matches:
            try:
                result.append((float(x_str), float(y_str)))
            except ValueError:
                continue
        return result
    
    return []


def parse_predicted_points(output_text):
    """Parse predicted list of points from model output text.
    
    Args:
        output_text: String containing model output with <answer>[(x1, y1), (x2, y2), ...]</answer> format
        or any text containing (x, y) coordinate patterns
        
    Returns:
        list: List of (x, y) tuples, or empty list if parsing fails
    """
    if not output_text:
        return []
    
    # First, try to extract content between <answer> and </answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        
        try:
            # Try to parse as Python literal (list format)
            coordinates = ast.literal_eval(answer_content)
            if isinstance(coordinates, list):
                result = []
                for point in coordinates:
                    if isinstance(point, (list, tuple)) and len(point) == 2:
                        result.append((float(point[0]), float(point[1])))
                if result:  # Only return if we successfully parsed points
                    return result
        except (ValueError, SyntaxError):
            pass
        
        # If literal_eval failed, try regex pattern matching on answer content
        content_to_search = answer_content
    else:
        # No <answer> tags found, search the entire text
        content_to_search = output_text
    
    # Fallback: Find all (x, y) patterns in the text
    # Pattern matches: (number, number) where numbers can be int or float
    pattern = r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)'
    matches = re.findall(pattern, content_to_search)
    
    if matches:
        result = []
        for x_str, y_str in matches:
            try:
                result.append((float(x_str), float(y_str)))
            except ValueError:
                continue
        if result:
            print(f"[parse_predicted_points] Using regex fallback, found {len(result)} points")
        return result
    
    return []


def parse_grasp(text, is_from_answer_tag=False):
    """Parse grasp annotation from text.
    
    Args:
        text: String containing grasp in fixed order:
              Grasp center, Left finger base, Right finger base, Left finger tip, Right finger tip
        is_from_answer_tag: If True, extract from <answer></answer> tags first
        
    Returns:
        dict: Dictionary with keys 'center', 'left_base', 'right_base', 'left_tip', 'right_tip',
              each containing (x, y) tuple, or None if parsing fails
    """
    if not text:
        return None
    
    content = text
    
    # Extract from <answer> tags if needed
    if is_from_answer_tag:
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_match:
            content = answer_match.group(1).strip()
    
    # Find all coordinate pairs in order
    # Pattern matches: [x, y] or (x, y)
    pattern = r'[\[\(]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\]\)]'
    matches = re.findall(pattern, content)
    
    # We need exactly 5 coordinate pairs in the fixed order
    if len(matches) != 5:
        return None
    
    try:
        grasp = {
            'center': (float(matches[0][0]), float(matches[0][1])),
            'left_base': (float(matches[1][0]), float(matches[1][1])),
            'right_base': (float(matches[2][0]), float(matches[2][1])),
            'left_tip': (float(matches[3][0]), float(matches[3][1])),
            'right_tip': (float(matches[4][0]), float(matches[4][1]))
        }
        return grasp
    except (ValueError, IndexError):
        return None


def parse_predicted_grasp(output_text):
    """Parse predicted grasp from model output text."""
    return parse_grasp(output_text, is_from_answer_tag=True)


def parse_ground_truth_grasp(ground_truth_text):
    """Parse ground truth grasp from string."""
    return parse_grasp(ground_truth_text, is_from_answer_tag=False)


def parse_bboxes(text, num_corners, source_name, is_from_answer_tag=False):
    """Parse bounding boxes from text.
    
    Args:
        text: String containing bounding boxes (either in <answer> tags or direct)
        num_corners: Expected number of corners (4 for 2D, 8 for 3D)
        source_name: Name for logging (e.g., "Predicted 2D BBox", "GT 3D BBox")
        is_from_answer_tag: If True, extract from <answer></answer> tags first
        
    Returns:
        list: List of bounding boxes, where each bbox is a list of num_corners (x, y) tuples,
              or empty list if parsing fails
    """
    if not text:
        print(f"[{source_name}] No text provided")
        return []
    
    content = text
    
    # Extract from <answer> tags if needed
    if is_from_answer_tag:
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if not answer_match:
            print(f"\n{'='*80}")
            print(f"WARNING [{source_name}]: No <answer> tags found in output!")
            print(f"{'='*80}\n")
            return []
        content = answer_match.group(1).strip()
        print(f"[{source_name}] Extracted answer content: {content[:200]}...")
    else:
        print(f"[{source_name}] Parsing text: {content[:200]}...")
    
    try:
        # Parse as Python literal
        bboxes = ast.literal_eval(content)
        
        if not isinstance(bboxes, list):
            print(f"\n{'='*80}")
            print(f"WARNING [{source_name}]: Parsed content is not a list! Type: {type(bboxes)}")
            print(f"{'='*80}\n")
            return []
        
        # Check if this is a single bbox or multiple bboxes
        # Single bbox format: [(x1, y1), (x2, y2), ..., (xN, yN)]
        # Multiple bboxes format: [[(x1, y1), ..., (xN, yN)], [(x1, y1), ..., (xN, yN)]]
        
        if len(bboxes) > 0 and isinstance(bboxes[0], (list, tuple)) and len(bboxes[0]) == 2:
            # First element is a coordinate tuple, so this is a single bbox
            if len(bboxes) == num_corners:
                # This is one bbox with num_corners corners
                parsed_bbox = []
                for corner in bboxes:
                    if isinstance(corner, (list, tuple)) and len(corner) == 2:
                        parsed_bbox.append((float(corner[0]), float(corner[1])))
                
                if len(parsed_bbox) == num_corners:
                    print(f"[{source_name}] Successfully parsed 1 bbox with {num_corners} corners")
                    return [parsed_bbox]
                else:
                    print(f"\n{'='*80}")
                    print(f"WARNING [{source_name}]: Single bbox had {len(parsed_bbox)} valid corners instead of {num_corners}")
                    print(f"Bbox data: {bboxes}")
                    print(f"{'='*80}\n")
                    return []
            else:
                print(f"\n{'='*80}")
                print(f"WARNING [{source_name}]: Single bbox format but has {len(bboxes)} corners instead of {num_corners}")
                print(f"Bbox data: {bboxes}")
                print(f"{'='*80}\n")
                return []
        else:
            # This is multiple bboxes format
            result = []
            for idx, bbox in enumerate(bboxes):
                if isinstance(bbox, list) and len(bbox) == num_corners:
                    # Each bbox should have num_corners corners
                    parsed_bbox = []
                    for corner in bbox:
                        if isinstance(corner, (list, tuple)) and len(corner) == 2:
                            parsed_bbox.append((float(corner[0]), float(corner[1])))
                    if len(parsed_bbox) == num_corners:
                        result.append(parsed_bbox)
                        print(f"[{source_name}] Successfully parsed bbox {idx}")
                    else:
                        print(f"\n{'='*80}")
                        print(f"WARNING [{source_name}]: bbox {idx} had {len(parsed_bbox)} valid corners instead of {num_corners}")
                        print(f"Bbox data: {bbox}")
                        print(f"{'='*80}\n")
                else:
                    print(f"\n{'='*80}")
                    print(f"WARNING [{source_name}]: bbox {idx} is not a list of {num_corners} corners!")
                    print(f"Type: {type(bbox)}, Len: {len(bbox) if isinstance(bbox, list) else 'N/A'}")
                    print(f"Bbox data: {bbox}")
                    print(f"{'='*80}\n")
            
            print(f"[{source_name}] Successfully parsed {len(result)} bbox(es)")
            return result
        
    except (ValueError, SyntaxError) as e:
        print(f"\n{'='*80}")
        print(f"ERROR [{source_name}]: Failed to parse bboxes!")
        print(f"Error: {e}")
        print(f"Content: {content[:500]}")
        print(f"{'='*80}\n")
        return []


def parse_predicted_bboxes(output_text):
    """Parse predicted 2D bounding boxes from model output text."""
    return parse_bboxes(output_text, 4, "Predicted 2D BBox", is_from_answer_tag=True)


def parse_ground_truth_bboxes(ground_truth_text):
    """Parse ground truth 2D bounding boxes from string."""
    return parse_bboxes(ground_truth_text, 4, "GT 2D BBox", is_from_answer_tag=False)


def parse_predicted_bboxes_3d(output_text):
    """Parse predicted 3D bounding boxes from model output text."""
    return parse_bboxes(output_text, 8, "Predicted 3D BBox", is_from_answer_tag=True)


def parse_ground_truth_bboxes_3d(ground_truth_text):
    """Parse ground truth 3D bounding boxes from string."""
    return parse_bboxes(ground_truth_text, 8, "GT 3D BBox", is_from_answer_tag=False)


def create_annotated_image(image_path, predicted_point, ground_truth_points):
    """Create an annotated image with predicted point and ground truth convex hull.
    
    Args:
        image_path: Path to the original image
        predicted_point: Tuple (x, y) of predicted coordinates (normalized 0-1)
        ground_truth_points: List of (x, y) tuples of ground truth coordinates (normalized 0-1)
        
    Returns:
        PIL.Image: Annotated image
    """
    # Load the original image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    width, height = image.size
    
    # Draw ground truth convex hull if we have enough points
    if len(ground_truth_points) >= 3:
        # Convert normalized coordinates to pixel coordinates
        gt_pixels = [(x * width, y * height) for x, y in ground_truth_points]
        gt_array = np.array(gt_pixels)
        
        try:
            # Calculate convex hull
            hull = ConvexHull(gt_array)
            hull_points = [gt_pixels[i] for i in hull.vertices]
            
            # Draw convex hull as green polygon
            draw.polygon(hull_points, outline='green', fill=None, width=3)
            # Add semi-transparent fill
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.polygon(hull_points, fill=(0, 255, 0, 50))
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(image)
            
        except Exception:
            # If convex hull fails, just draw the points
            for point in gt_pixels:
                x, y = point
                draw.ellipse([x-3, y-3, x+3, y+3], fill='green', outline='darkgreen', width=2)
    
    elif len(ground_truth_points) > 0:
        # If we have fewer than 3 points, just draw them as circles
        for x, y in ground_truth_points:
            px, py = x * width, y * height
            draw.ellipse([px-3, py-3, px+3, py+3], fill='green', outline='darkgreen', width=2)
    
    # Draw predicted point as red circle
    if predicted_point:
        px, py = predicted_point[0] * width, predicted_point[1] * height
        radius = 8
        draw.ellipse([px-radius, py-radius, px+radius, py+radius], 
                    fill='red', outline='white', width=2)
        # Add a small cross in the center
        cross_size = 3
        draw.line([px-cross_size, py, px+cross_size, py], fill='white', width=2)
        draw.line([px, py-cross_size, px, py+cross_size], fill='white', width=2)
    
    return image


def create_annotated_image_with_points(image_path, predicted_points, ground_truth_points):
    """Create an annotated image with predicted and ground truth point lists.
    
    Args:
        image_path: Path to the original image
        predicted_points: List of (x, y) tuples of predicted coordinates (normalized 0-1)
        ground_truth_points: List of (x, y) tuples of ground truth coordinates (normalized 0-1)
        
    Returns:
        PIL.Image: Annotated image
    """
    # Load the original image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Create overlay for semi-transparent fills
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(image)
    
    # Draw ground truth points
    if len(ground_truth_points) >= 3:
        # Convert normalized coordinates to pixel coordinates
        gt_pixels = [(x * width, y * height) for x, y in ground_truth_points]
        gt_array = np.array(gt_pixels)
        
        try:
            # Calculate and draw convex hull
            hull = ConvexHull(gt_array)
            hull_points = [gt_pixels[i] for i in hull.vertices]
            
            # Draw as green polygon with semi-transparent fill
            overlay_draw.polygon(hull_points, outline=(0, 255, 0, 255), fill=(0, 255, 0, 50), width=3)
            
        except Exception:
            # If convex hull fails, draw polygon connecting all points
            overlay_draw.polygon(gt_pixels, outline=(0, 255, 0, 255), fill=(0, 255, 0, 50), width=3)
    
    elif len(ground_truth_points) > 0:
        # Draw individual points as circles
        for x, y in ground_truth_points:
            px, py = x * width, y * height
            overlay_draw.ellipse([px-4, py-4, px+4, py+4], fill=(0, 255, 0, 200), outline=(0, 200, 0, 255))
    
    # Draw predicted points
    if len(predicted_points) >= 3:
        # Convert normalized coordinates to pixel coordinates
        pred_pixels = [(x * width, y * height) for x, y in predicted_points]
        
        # Draw as red polygon with semi-transparent fill (connecting the points)
        overlay_draw.polygon(pred_pixels, outline=(255, 0, 0, 255), fill=(255, 0, 0, 50), width=3)
        
    elif len(predicted_points) > 0:
        # Draw individual points as circles
        for x, y in predicted_points:
            px, py = x * width, y * height
            overlay_draw.ellipse([px-5, py-5, px+5, py+5], fill=(255, 0, 0, 200), outline=(255, 255, 255, 255))
    
    # Composite overlay onto image
    image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
    
    # Add markers for individual points on top
    draw = ImageDraw.Draw(image)
    
    # Mark ground truth points
    for x, y in ground_truth_points:
        px, py = x * width, y * height
        draw.ellipse([px-2, py-2, px+2, py+2], fill='darkgreen', outline='darkgreen')
    
    # Mark predicted points
    for x, y in predicted_points:
        px, py = x * width, y * height
        draw.ellipse([px-3, py-3, px+3, py+3], fill='red', outline='white', width=1)
    
    return image


def draw_bboxes_on_image(image_path, predicted_bboxes, ground_truth_bboxes, is_3d=False):
    """Draw bounding boxes on an image.
    
    Args:
        image_path: Path to the original image
        predicted_bboxes: List of predicted bboxes (4 or 8 corners each)
        ground_truth_bboxes: List of ground truth bboxes (4 or 8 corners each)
        is_3d: If True, draw as 3D wireframe; if False, draw as 2D polygons with fill
        
    Returns:
        PIL.Image: Annotated image
    """
    # Load the original image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    if is_3d:
        # Draw 3D bounding boxes as wireframes
        draw = ImageDraw.Draw(image)
        
        # Define the edges of a 3D bounding box
        # Bottom face: 0-1, 1-2, 2-3, 3-0
        # Top face: 4-5, 5-6, 6-7, 7-4
        # Vertical edges: 0-4, 1-5, 2-6, 3-7
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        # Draw ground truth 3D bounding boxes in green
        for bbox in ground_truth_bboxes:
            pixel_coords = [(x * width, y * height) for x, y in bbox]
            for start_idx, end_idx in edges:
                draw.line([pixel_coords[start_idx], pixel_coords[end_idx]], fill='green', width=3)
        
        # Draw predicted 3D bounding boxes in red
        for bbox in predicted_bboxes:
            pixel_coords = [(x * width, y * height) for x, y in bbox]
            for start_idx, end_idx in edges:
                draw.line([pixel_coords[start_idx], pixel_coords[end_idx]], fill='red', width=3)
    else:
        # Draw 2D bounding boxes as polygons with semi-transparent fills
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Draw ground truth bounding boxes in green
        for bbox in ground_truth_bboxes:
            pixel_coords = [(x * width, y * height) for x, y in bbox]
            overlay_draw.polygon(pixel_coords, outline=(0, 255, 0, 255), fill=(0, 255, 0, 50), width=3)
        
        # Draw predicted bounding boxes in red
        for bbox in predicted_bboxes:
            pixel_coords = [(x * width, y * height) for x, y in bbox]
            overlay_draw.polygon(pixel_coords, outline=(255, 0, 0, 255), fill=(255, 0, 0, 50), width=3)
        
        # Composite the overlay onto the original image
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
    
    return image


def create_annotated_image_with_bboxes(image_path, predicted_bboxes, ground_truth_bboxes):
    """Create an annotated image with predicted and ground truth 2D bounding boxes."""
    return draw_bboxes_on_image(image_path, predicted_bboxes, ground_truth_bboxes, is_3d=False)


def create_annotated_image_with_3d_bboxes(image_path, predicted_bboxes, ground_truth_bboxes):
    """Create an annotated image with predicted and ground truth 3D bounding boxes."""
    return draw_bboxes_on_image(image_path, predicted_bboxes, ground_truth_bboxes, is_3d=True)


def create_annotated_image_with_grasps(image_path, predicted_grasp, ground_truth_grasp):
    """Create an annotated image with predicted and ground truth grasps.
    
    Args:
        image_path: Path to the original image
        predicted_grasp: Dict with keys 'center', 'left_base', 'right_base', 'left_tip', 'right_tip'
                        (normalized 0-1 coordinates), or None
        ground_truth_grasp: Dict with same structure as predicted_grasp, or None
        
    Returns:
        PIL.Image: Annotated image
    """
    # Load the original image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    draw = ImageDraw.Draw(image)
    
    def draw_grasp(grasp, color, line_width):
        """Draw a single grasp annotation."""
        if not grasp:
            return
        
        # Convert normalized coordinates to pixel coordinates
        center = (grasp['center'][0] * width, grasp['center'][1] * height)
        left_base = (grasp['left_base'][0] * width, grasp['left_base'][1] * height)
        right_base = (grasp['right_base'][0] * width, grasp['right_base'][1] * height)
        left_tip = (grasp['left_tip'][0] * width, grasp['left_tip'][1] * height)
        right_tip = (grasp['right_tip'][0] * width, grasp['right_tip'][1] * height)
        
        # Draw connections
        # Left finger: base to tip
        draw.line([left_base, left_tip], fill=color, width=line_width)
        # Right finger: base to tip
        draw.line([right_base, right_tip], fill=color, width=line_width)
        # Left base to center
        draw.line([left_base, center], fill=color, width=line_width)
        # Right base to center
        draw.line([right_base, center], fill=color, width=line_width)
        
        # Draw points
        point_radius = 5
        for point, label in [(center, 'C'), (left_base, 'LB'), (right_base, 'RB'), 
                              (left_tip, 'LT'), (right_tip, 'RT')]:
            # Draw circle
            draw.ellipse([point[0]-point_radius, point[1]-point_radius, 
                         point[0]+point_radius, point[1]+point_radius], 
                        fill=color, outline='white', width=2)
    
    # Draw ground truth grasp in green (drawn first so prediction is on top)
    if ground_truth_grasp:
        draw_grasp(ground_truth_grasp, 'green', 4)
    
    # Draw predicted grasp in red
    if predicted_grasp:
        draw_grasp(predicted_grasp, 'red', 4)
    
    return image


def create_annotated_image_with_mask(image_path, predicted_point, ground_truth_mask_b64):
    """Create an annotated image with predicted point and ground truth mask overlay.
    
    Args:
        image_path: Path to the original image
        predicted_point: Tuple (x, y) of predicted coordinates (normalized 0-1), or None
        ground_truth_mask_b64: Base64-encoded PNG mask string
        
    Returns:
        PIL.Image: Annotated image
    """
    # Load the original image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Decode and overlay the ground truth mask
    if ground_truth_mask_b64:
        try:
            # Decode base64 mask
            mask_bytes = base64.b64decode(ground_truth_mask_b64)
            mask_image = Image.open(BytesIO(mask_bytes))
            
            # Convert mask to grayscale if needed
            if mask_image.mode != 'L':
                mask_image = mask_image.convert('L')
            
            # Resize mask to match image dimensions if needed
            if mask_image.size != image.size:
                mask_image = mask_image.resize(image.size, Image.NEAREST)
            
            # Create a green overlay for the mask using numpy for efficiency
            mask_array = np.array(mask_image)
            
            # Create RGBA overlay array
            overlay_array = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.uint8)
            
            # Set green overlay where mask is non-zero (RGBA: green with alpha 50)
            mask_nonzero = mask_array > 0
            overlay_array[mask_nonzero] = [0, 255, 0, 50]  # Semi-transparent green
            
            # Convert to PIL image and composite
            overlay = Image.fromarray(overlay_array, mode='RGBA')
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            
        except Exception as e:
            print(f"[Mask Overlay] Error decoding/overlaying mask: {e}")
    
    # Draw predicted point as red circle
    draw = ImageDraw.Draw(image)
    if predicted_point:
        px, py = predicted_point[0] * width, predicted_point[1] * height
        radius = 8
        draw.ellipse([px-radius, py-radius, px+radius, py+radius], 
                    fill='red', outline='white', width=2)
        # Add a small cross in the center
        cross_size = 3
        draw.line([px-cross_size, py, px+cross_size, py], fill='white', width=2)
        draw.line([px, py-cross_size, px, py+cross_size], fill='white', width=2)
    
    return image


class MultimodalLogHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, base_dir=None, **kwargs):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)
        
        if path == '/':
            self.serve_index()
        elif path == '/view':
            step = query.get('step', [None])[0]
            sample = query.get('sample', [None])[0]
            self.serve_sample(step, sample)
        elif path.startswith('/image/'):
            # Serve image files
            img_path = path[7:]  # Remove '/image/' prefix
            self.serve_image(img_path)
        elif path.startswith('/annotated_image/'):
            # Serve annotated image files
            parts = path[17:].split('/')  # Remove '/annotated_image/' prefix
            if len(parts) == 2:
                step, sample = parts
                self.serve_annotated_image(step, sample)
            else:
                self.send_error(400, "Invalid annotated image path")
        else:
            self.send_error(404)
    
    def serve_index(self):
        """Serve the main index page."""
        # Find all JSONL files
        jsonl_files = sorted(self.base_dir.glob("*.jsonl"))
        
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal Conversation Logs</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px;
            background: #fafafa;
            color: #333;
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        .base-dir {{
            background: #ecf0f1;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 30px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }}
        .step-section {{ 
            margin: 15px 0; 
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: box-shadow 0.2s;
        }}
        .step-section:hover {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .step-header {{
            padding: 20px;
            background: #e3f2fd; 
            color: #0d47a1; 
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }}
        .step-header:hover {{
            background: #bbdefb;
        }}
        .step-header h2 {{
            margin: 0;
            font-size: 20px;
        }}
        .toggle-icon {{
            font-size: 24px;
            transition: transform 0.3s;
        }}
        .toggle-icon.expanded {{
            transform: rotate(180deg);
        }}
        .step-content {{
            padding: 20px;
            display: none;
            background: #f8f9fa;
        }}
        .step-content.expanded {{
            display: block;
        }}
        .sample-link {{ 
            display: inline-block; 
            margin: 5px; 
            padding: 8px 12px; 
            background: white; 
            border: 1px solid #ddd; 
            border-radius: 3px; 
            text-decoration: none; 
            color: #333;
        }}
        .sample-link:hover {{ 
            background: #e0e0e0; 
        }}
        .has-messages {{ 
            background: #e8f5e9; 
            font-weight: bold; 
        }}
        .stats {{ 
            color: #666; 
            margin: 10px 0; 
        }}
        .expand-all-btn {{
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            margin-bottom: 20px;
            transition: background 0.2s;
        }}
        .expand-all-btn:hover {{
            background: #2980b9;
        }}
    </style>
    <script>
        function toggleStep(stepId) {{
            const content = document.getElementById('content-' + stepId);
            const icon = document.getElementById('icon-' + stepId);
            content.classList.toggle('expanded');
            icon.classList.toggle('expanded');
        }}
        
        function toggleAll() {{
            const contents = document.querySelectorAll('.step-content');
            const icons = document.querySelectorAll('.toggle-icon');
            const allExpanded = Array.from(contents).every(c => c.classList.contains('expanded'));
            
            contents.forEach(content => {{
                if (allExpanded) {{
                    content.classList.remove('expanded');
                }} else {{
                    content.classList.add('expanded');
                }}
            }});
            
            icons.forEach(icon => {{
                if (allExpanded) {{
                    icon.classList.remove('expanded');
                }} else {{
                    icon.classList.add('expanded');
                }}
            }});
            
            document.getElementById('expand-all-btn').textContent = allExpanded ? 'Expand All' : 'Collapse All';
        }}
    </script>
</head>
<body>
    <h1>Multimodal Conversation Logs</h1>
    <p>Base directory: {base_dir}</p>
    <button id="expand-all-btn" class="expand-all-btn" onclick="toggleAll()">Expand All</button>
"""
        
        if not jsonl_files:
            html += "<p>No JSONL files found.</p>"
        else:
            for idx, jsonl_file in enumerate(jsonl_files):
                step = jsonl_file.stem
                samples = load_jsonl(jsonl_file)
                samples_with_messages = sum(1 for s in samples if 'messages' in s and s['messages'])
                
                html += f"""
    <div class="step-section">
        <div class="step-header" onclick="toggleStep('{idx}')">
            <div>
                <h2>Step {step}</h2>
                <div class="stats">
                    Total samples: {len(samples)} | 
                    With conversations: {samples_with_messages}
                </div>
            </div>
            <span id="icon-{idx}" class="toggle-icon">&#9660;</span>
        </div>
        <div id="content-{idx}" class="step-content">
"""
                for i, sample in enumerate(samples):
                    has_messages = 'messages' in sample and sample['messages']
                    css_class = "sample-link has-messages" if has_messages else "sample-link"
                    html += f'<a href="/view?step={step}&sample={i}" class="{css_class}">Sample {i}</a>\n'
                
                html += """
        </div>
    </div>
"""
        
        html += """
</body>
</html>"""
        
        html = html.format(base_dir=self.base_dir)
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def serve_sample(self, step, sample_idx):
        """Serve a specific sample view."""
        # Import html as html_lib to avoid conflict with html string variable
        import html as html_lib
        
        if not step or not sample_idx:
            self.send_error(400, "Missing step or sample parameter")
            return
        
        try:
            sample_idx = int(sample_idx)
        except ValueError:
            self.send_error(400, "Invalid sample index")
            return
        
        jsonl_file = self.base_dir / f"{step}.jsonl"
        if not jsonl_file.exists():
            self.send_error(404, f"Step {step} not found")
            return
        
        samples = load_jsonl(jsonl_file)
        if sample_idx < 0 or sample_idx >= len(samples):
            self.send_error(404, f"Sample {sample_idx} not found")
            return
        
        sample = samples[sample_idx]
        
        # Debug: Print question type
        question_type = sample.get('question_type', 'NOT SET')
        print(f"\n{'='*80}")
        print(f"[Sample View] Step: {step}, Sample: {sample_idx}")
        print(f"[Sample View] Question Type: {question_type}")
        print(f"[Sample View] Has output: {bool(sample.get('output'))}")
        print(f"[Sample View] Has ground_truth: {bool(sample.get('ground_truth'))}")
        print(f"{'='*80}\n")
        
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Step {step} - Sample {sample_idx}</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 20px;
        }}
        .nav {{ 
            margin-bottom: 30px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .nav a {{ 
            margin-right: 10px; 
            padding: 5px 10px; 
            background: #2196F3; 
            color: white; 
            text-decoration: none; 
            border-radius: 3px; 
        }}
        .nav a:hover {{ 
            background: #1976D2; 
        }}
        .metadata {{ 
            background: #f0f0f0; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 20px 0; 
        }}
        .metadata h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .metadata p {{
            margin: 8px 0;
            line-height: 1.6;
        }}
        .metadata strong {{
            color: #34495e;
        }}
        .conversation-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 30px;
            overflow: hidden;
        }}
        .conversation-header {{
            background: #e3f2fd;
            padding: 20px;
            color: #0d47a1;
            border-bottom: 1px solid #bbdefb;
        }}
        .conversation-header h2 {{
            margin: 0;
            font-size: 1.3em;
        }}
        .messages-container {{
            padding: 20px;
            background: #fafafa;
        }}
        .messages-container::-webkit-scrollbar {{
            width: 8px;
        }}
        .messages-container::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}
        .messages-container::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 4px;
        }}
        .messages-container::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
        .message-wrapper {{
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }}
        .message-wrapper.user {{
            flex-direction: row-reverse;
        }}
        .message-avatar {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            flex-shrink: 0;
            font-size: 14px;
        }}
        .message-wrapper.user .message-avatar {{
            background: #3498db;
        }}
        .message-wrapper.assistant .message-avatar {{
            background: #2ecc71;
        }}
        .message-wrapper.tool .message-avatar {{
            background: #f39c12;
        }}
        .message-wrapper.unknown .message-avatar {{
            background: #95a5a6;
        }}
        .message {{ 
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            position: relative;
        }}
        .message-wrapper.user .message {{
            background: #3498db;
            color: white;
            border-bottom-right-radius: 4px;
            margin-left: auto;
        }}
        .message-wrapper.assistant .message {{
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 4px;
        }}
        .message-wrapper.tool .message {{
            background: #fff8e1;
            color: #333;
            border: 1px solid #ffe0b2;
            border-bottom-left-radius: 4px;
            max-width: 85%;
        }}
        .message-wrapper.unknown .message {{
            background: #ecf0f1;
            color: #333;
            border: 1px solid #bdc3c7;
            border-bottom-left-radius: 4px;
        }}
        .message-role {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
            opacity: 0.8;
        }}
        .message-wrapper.user .message-role {{
            color: rgba(255,255,255,0.9);
        }}
        .tool-call {{ 
            background: #e8f5e9;
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }}
        .tool-call strong {{
            color: #2e7d32;
        }}
        .image-container {{ 
            margin: 15px 0;
            text-align: center;
        }}
        .image-container img {{ 
            max-width: 600px;
            max-height: 600px;
            border: 2px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .image-container small {{
            display: block;
            margin-top: 8px;
            color: #666;
            font-style: italic;
        }}
        pre {{ 
            background: #263238;
            color: #aed581;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.4;
        }}
        .content-text {{ 
            margin: 5px 0;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .message-wrapper.user .content-text {{
            color: white;
        }}
        .thinking-block {{
            background: #fff9e6;
            border-left: 4px solid #ffc107;
            padding: 12px 16px;
            margin: 10px 0;
            border-radius: 4px;
            font-style: italic;
            color: #5d4037;
        }}
        .thinking-label {{
            font-weight: bold;
            font-style: normal;
            color: #f57f17;
            margin-bottom: 8px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metadata-container {{
            margin: 20px 0;
            border: 1px solid #cfe8ff;
            border-radius: 6px;
            overflow: hidden;
            background: #f8fbff;
        }}
        .metadata-header {{
            padding: 12px 16px;
            background: #e3f2fd;
            color: #0d47a1;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .metadata-content {{
            display: none;
            padding: 15px;
            background: #f9fbff;
        }}
        .metadata-content.expanded {{
            display: block;
        }}
        .metadata-toggle-icon {{
            font-size: 18px;
            transition: transform 0.3s;
        }}
        .metadata-toggle-icon.expanded {{
            transform: rotate(180deg);
        }}
        .annotation-legend {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
            font-size: 14px;
        }}
        .annotation-legend h4 {{
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 16px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}
        .legend-symbol {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border-radius: 50%;
            border: 2px solid white;
        }}
        .legend-symbol.predicted {{
            background: red;
        }}
        .legend-symbol.ground-truth {{
            background: green;
            border-radius: 3px;
        }}
    </style>
    <script>
        function toggleStep(stepId) {{
            const content = document.getElementById('content-' + stepId);
            const icon = document.getElementById('icon-' + stepId);
            content.classList.toggle('expanded');
            icon.classList.toggle('expanded');
        }}
        
        function toggleAll() {{
            const contents = document.querySelectorAll('.step-content');
            const icons = document.querySelectorAll('.toggle-icon');
            const allExpanded = Array.from(contents).every(c => c.classList.contains('expanded'));
            
            contents.forEach(content => {{
                if (allExpanded) {{
                    content.classList.remove('expanded');
                }} else {{
                    content.classList.add('expanded');
                }}
            }});
            
            icons.forEach(icon => {{
                if (allExpanded) {{
                    icon.classList.remove('expanded');
                }} else {{
                    icon.classList.add('expanded');
                }}
            }});
            
            document.getElementById('expand-all-btn').textContent = allExpanded ? 'Expand All' : 'Collapse All';
        }}

        function toggleMetadata() {{
            const content = document.getElementById('metadata-content');
            const icon = document.getElementById('metadata-icon');
            content.classList.toggle('expanded');
            icon.classList.toggle('expanded');
        }}
    </script>
</head>
<body>
    <div class="nav">
        <a href="/">← Back to Index</a>
"""
        html = html.format(step=step, sample_idx=sample_idx)
        
        # Add previous/next navigation
        if sample_idx > 0:
            html += f'<a href="/view?step={step}&sample={sample_idx-1}">← Previous</a>'
        if sample_idx < len(samples) - 1:
            html += f'<a href="/view?step={step}&sample={sample_idx+1}">Next →</a>'
        
        html += f"""
    </div>
    
    <h1>Step {step} - Sample {sample_idx}</h1>
    
    <div class="metadata-container">
        <div class="metadata-header" onclick="toggleMetadata()">
            <h3>Metadata</h3>
            <span id="metadata-icon" class="metadata-toggle-icon">&#9660;</span>
        </div>
        <div id="metadata-content" class="metadata-content">
        <p><strong>Score:</strong> {html_lib.escape(str(sample.get('score', 'N/A')))}</p>
        <p><strong>Input:</strong> {html_lib.escape(str(sample.get('input', 'N/A')))}</p>
        <p><strong>Output:</strong> {html_lib.escape(str(sample.get('output', 'N/A')))}</p>
"""
        
        # Add any extra info
        for key in sample:
            if key not in ['input', 'output', 'score', 'step', 'messages']:
                html += f"<p><strong>{html_lib.escape(key)}:</strong> {html_lib.escape(str(sample[key]))}</p>"
        
        html += "</div></div>"
        
        # Add annotation legend for point questions
        if sample.get('question_type') in ['point', 'unary_spatial_context', 'bench_point']:
            html += '''
    <div class="annotation-legend">
        <h4>Annotation Legend</h4>
        <div class="legend-item">
            <div class="legend-symbol predicted"></div>
            <span>Predicted Point (Red Circle)</span>
        </div>
        <div class="legend-item">
            <div class="legend-symbol ground-truth"></div>
            <span>Ground Truth Region (Green Polygon/Mask)</span>
        </div>
    </div>
'''
        
        # Add annotation legend for coordinate list questions
        if sample.get('question_type') in ['coordinates', 'polygon', 'trajectory', 'points', 'spatial', 'object', 'vacant']:
            html += '''
    <div class="annotation-legend">
        <h4>Annotation Legend</h4>
        <div class="legend-item">
            <div class="legend-symbol predicted"></div>
            <span>Predicted Points/Polygon (Red)</span>
        </div>
        <div class="legend-item">
            <div class="legend-symbol ground-truth"></div>
            <span>Ground Truth Points/Polygon (Green)</span>
        </div>
    </div>
'''
        
        # Add annotation legend for bbox questions
        if sample.get('question_type') in ['bbox', 'bounding_box']:
            html += '''
    <div class="annotation-legend">
        <h4>Annotation Legend</h4>
        <div class="legend-item">
            <div class="legend-symbol predicted"></div>
            <span>Predicted Bounding Box (Red Rectangle)</span>
        </div>
        <div class="legend-item">
            <div class="legend-symbol ground-truth"></div>
            <span>Ground Truth Bounding Box (Green Rectangle)</span>
        </div>
    </div>
'''
        
        # Add annotation legend for 3D bbox questions
        if sample.get('question_type') in ['bbox_3d', 'bounding_box_3d', '3d_bbox', 'pose', 'unknown']:
            html += '''
    <div class="annotation-legend">
        <h4>Annotation Legend</h4>
        <div class="legend-item">
            <div class="legend-symbol predicted"></div>
            <span>Predicted 3D Bounding Box (Red Wireframe)</span>
        </div>
        <div class="legend-item">
            <div class="legend-symbol ground-truth"></div>
            <span>Ground Truth 3D Bounding Box (Green Wireframe)</span>
        </div>
    </div>
'''
        
        # Add annotation legend for grasp questions
        if sample.get('question_type') == 'grasp':
            html += '''
    <div class="annotation-legend">
        <h4>Annotation Legend</h4>
        <div class="legend-item">
            <div class="legend-symbol predicted"></div>
            <span>Predicted Grasp (Red Lines and Points)</span>
        </div>
        <div class="legend-item">
            <div class="legend-symbol ground-truth"></div>
            <span>Ground Truth Grasp (Green Lines and Points)</span>
        </div>
        <p style="margin-top: 10px; font-size: 13px; color: #666;">
            Grasp visualization: Lines connect finger bases to tips, and both bases to center.
        </p>
    </div>
'''
        
        # Display conversation history
        if 'messages' in sample and sample['messages']:
            html += '''
    <div class="conversation-container">
        <div class="conversation-header">
            <h2>Conversation History</h2>
        </div>
        <div class="messages-container">
'''
            
            # Handle the structure where messages is a dict with "messages" key inside
            messages_data = sample['messages']
            if isinstance(messages_data, dict) and 'messages' in messages_data:
                messages_list = messages_data['messages']
            elif isinstance(messages_data, list):
                messages_list = messages_data
            else:
                messages_list = []
            
            # Track global image counter for mapping to files
            img_counter = 0
            
            for msg_idx, msg in enumerate(messages_list):
                if msg is None:
                    continue
                
                # Handle case where msg is a string representation
                if isinstance(msg, str):
                    # Parse the string representation
                    # Format: "role='role_name' content=... tool_calls=..."
                    import re
                    import ast
                    
                    # Extract role
                    role_match = re.search(r"role='([^']+)'", msg)
                    role = role_match.group(1) if role_match else 'unknown'
                    
                    # Extract content - more robust parsing
                    content = None
                    
                    # Try to find content between content= and tool_calls=
                    content_match = re.search(r"content=(.*?)(?:\s+tool_calls=|$)", msg, re.DOTALL)
                    if content_match:
                        content_str = content_match.group(1).strip()
                        try:
                            # Try to parse as Python literal
                            content = ast.literal_eval(content_str)
                        except:
                            # If it fails, treat as string
                            if content_str.startswith(("'", '"')) and content_str.endswith(("'", '"')):
                                content = content_str[1:-1]
                            else:
                                content = content_str
                    
                    msg = {'role': role, 'content': content}
                
                # Handle case where msg might be a non-dict type
                if not isinstance(msg, dict):
                    continue
                    
                role = msg.get('role', 'unknown')
                
                # Get avatar initial
                avatar_initial = role[0].upper() if role else '?'
                
                html += f'<div class="message-wrapper {role}">'
                html += f'<div class="message-avatar">{avatar_initial}</div>'
                html += f'<div class="message">'
                html += f'<div class="message-role">{role}</div>'
                
                # Handle tool calls
                if 'tool_calls' in msg and msg['tool_calls']:
                    for tc in msg['tool_calls']:
                        html += '<div class="tool-call">'
                        html += f'<strong>Tool Call:</strong> {html_lib.escape(tc["name"])}<br>'
                        html += f'<strong>Arguments:</strong><br>'
                        html += f'<pre>{html_lib.escape(json.dumps(tc["args"], indent=2))}</pre>'
                        html += '</div>'
                
                # Handle content
                content = msg.get('content', '')
                
                # Track image counter for this message
                msg_image_counter = 0
                
                if isinstance(content, list):
                    # Handle list of content items (multimodal format)
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'thinking':
                                thinking_text = item.get('thinking', '')
                                # Escape HTML to prevent XML tags from being interpreted
                                escaped_thinking = html_lib.escape(thinking_text)
                                html += '<div class="thinking-block">'
                                html += '<div class="thinking-label">💭 Model Thinking</div>'
                                html += f'<div>{escaped_thinking}</div>'
                                html += '</div>'
                            elif item.get('type') == 'text':
                                text = item.get('text', '')
                                # Escape HTML to prevent XML tags from being interpreted
                                escaped_text = html_lib.escape(text)
                                html += f'<div class="content-text">{escaped_text}</div>'
                            elif item.get('type') == 'image':
                                # Find the corresponding image file
                                images_dir = f"images_{step}"
                                img_filename = f"s{sample_idx:03d}_img{img_counter:04d}.png"
                                img_path = f"{images_dir}/{img_filename}"
                                
                                # Always use un-annotated image in conversation
                                img_url = f"/image/{img_path}"
                                
                                html += '<div class="image-container">'
                                html += f'<img src="{img_url}" alt="Image" />'
                                html += f'<br><small>{img_filename}</small>'
                                html += '</div>'
                                
                                img_counter += 1
                                msg_image_counter += 1
                elif isinstance(content, dict):
                    if 'text' in content and content['text']:
                        escaped_text = html_lib.escape(content['text'])
                        html += f'<div class="content-text">{escaped_text}</div>'
                    if 'images' in content:
                        for img_path in content['images']:
                            # Always use un-annotated image in conversation
                            img_url = f"/image/{img_path}"
                                
                            html += '<div class="image-container">'
                            html += f'<img src="{img_url}" alt="Image" />'
                            html += f'<br><small>{img_path}</small>'
                            html += '</div>'
                elif isinstance(content, str) and content:
                    escaped_text = html_lib.escape(content)
                    html += f'<div class="content-text">{escaped_text}</div>'
                
                html += '</div>'  # Close message div
                html += '</div>'  # Close message-wrapper div
            
            html += """
        </div>  <!-- Close messages-container -->
    </div>  <!-- Close conversation-container -->
"""
        else:
            html += "<p>No conversation history available for this sample.</p>"
        
        # Add annotated image section for point, bbox, coordinates, 3D bbox, grasp, or bench_point questions
        if sample.get('question_type') in ['point', 'bbox', 'bounding_box', 'unary_spatial_context', 'coordinates', 'polygon', 'trajectory', 'points', 'spatial', 'object', 'vacant', 'bbox_3d', 'bounding_box_3d', '3d_bbox', 'pose', 'unknown', 'grasp', 'bench_point']:
            html += '''
    <div class="conversation-container" style="margin-top: 20px;">
        <div class="conversation-header">
            <h2>Annotated Image</h2>
        </div>
        <div class="messages-container" style="text-align: center;">
            <div class="image-container">
'''
            html += f'<img src="/annotated_image/{step}/{sample_idx}" alt="Annotated Image" />'
            html += '''
            </div>
        </div>
    </div>
'''
        
        html += """
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def serve_image(self, img_path):
        """Serve an image file."""
        full_path = self.base_dir / img_path
        
        if not full_path.exists() or not full_path.is_file():
            self.send_error(404, f"Image not found: {img_path}")
            return
        
        # Determine content type
        ext = full_path.suffix.lower()
        content_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
        }
        content_type = content_types.get(ext, 'application/octet-stream')
        
        # Read and serve the image
        try:
            with open(full_path, 'rb') as f:
                image_data = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Content-length', str(len(image_data)))
            self.end_headers()
            self.wfile.write(image_data)
        except Exception as e:
            self.send_error(500, f"Error reading image: {e}")
    
    def serve_annotated_image(self, step, sample_idx):
        """Serve an annotated image with predicted point and ground truth convex hull, or bounding boxes."""
        try:
            sample_idx = int(sample_idx)
        except ValueError:
            self.send_error(400, "Invalid sample index")
            return
        
        # Load the sample data to get annotations
        jsonl_file = self.base_dir / f"{step}.jsonl"
        if not jsonl_file.exists():
            self.send_error(404, f"Step {step} not found")
            return
        
        samples = load_jsonl(jsonl_file)
        if sample_idx < 0 or sample_idx >= len(samples):
            self.send_error(404, f"Sample {sample_idx} not found")
            return
        
        sample = samples[sample_idx]
        question_type = sample.get('question_type')
        
        print(f"[Annotated Image] Step: {step}, Sample: {sample_idx}, Question Type: {question_type}")
        
        # Check if this is a point, bbox, coordinates, 3D bbox, grasp, or bench_point question type
        if question_type not in ['point', 'bbox', 'bounding_box', 'unary_spatial_context', 'coordinates', 'polygon', 'trajectory', 'points', 'spatial', 'object', 'vacant', 'bbox_3d', 'bounding_box_3d', '3d_bbox', 'pose', 'unknown', 'grasp', 'bench_point']:
            print(f"[Annotated Image] Question type '{question_type}' not in supported types, serving original image")
            # For other question types, just serve the original image
            if 'image_path' in sample:
                img_path = sample['image_path']
                if img_path.startswith('images_'):
                    # Remove the images_X/ prefix to match the expected path format
                    img_path = img_path
                self.serve_image(img_path)
                return
            else:
                self.send_error(404, "No image found for this sample")
                return
        
        # Find the original image
        image_path = None
        if 'image_path' in sample:
            image_path = self.base_dir / sample['image_path']
            print(f"[Annotated Image] Image path: {image_path} from sample")
        else:
            # Fallback: construct image path
            images_dir = f"images_{step}"
            img_filename = f"s{sample_idx:03d}_img0000.png"
            image_path = self.base_dir / images_dir / img_filename
            print(f"[Annotated Image] Image path: {image_path} from fallback")
        if not image_path.exists():
            print(f"[Annotated Image] ERROR: Image not found: {image_path}")
            self.send_error(404, f"Image not found: {image_path}")
            return
        
        try:
            # Create annotated image based on question type
            if question_type in ['point', 'unary_spatial_context']:
                # Parse coordinates
                predicted_point = parse_predicted_point(sample.get('output', ''))
                ground_truth_points = parse_ground_truth_points(sample.get('ground_truth', ''))
                
                print(f"[Annotated Image] Point task - Predicted: {predicted_point}, GT points: {len(ground_truth_points)}")
                
                # Create annotated image with points
                annotated_image = create_annotated_image(image_path, predicted_point, ground_truth_points)
            
            elif question_type in ['coordinates', 'polygon', 'trajectory', 'points', 'spatial', 'object', 'vacant']:
                # Parse lists of coordinates
                output_text = sample.get('output', '')
                gt_text = sample.get('ground_truth', '')
                
                print(f"[Annotated Image] Coordinates task")
                print(f"[Annotated Image] Output text (first 200 chars): {output_text[:200]}")
                print(f"[Annotated Image] GT text (first 200 chars): {gt_text[:200]}")
                
                predicted_points = parse_predicted_points(output_text)
                ground_truth_points = parse_ground_truth_points(gt_text)
                
                print(f"[Annotated Image] Parsed - Predicted points: {len(predicted_points)}, GT points: {len(ground_truth_points)}")
                if predicted_points:
                    print(f"[Annotated Image] First predicted point: {predicted_points[0]}")
                if ground_truth_points:
                    print(f"[Annotated Image] First GT point: {ground_truth_points[0]}")
                
                # Create annotated image with point lists
                annotated_image = create_annotated_image_with_points(image_path, predicted_points, ground_truth_points)
            
            elif question_type in ['bbox', 'bounding_box']:
                # Parse bounding boxes
                predicted_bboxes = parse_predicted_bboxes(sample.get('output', ''))
                ground_truth_bboxes = parse_ground_truth_bboxes(sample.get('ground_truth', ''))
                
                print(f"[Annotated Image] BBox task - Predicted bboxes: {len(predicted_bboxes)}, GT bboxes: {len(ground_truth_bboxes)}")
                if predicted_bboxes:
                    print(f"[Annotated Image] First predicted bbox: {predicted_bboxes[0]}")
                if ground_truth_bboxes:
                    print(f"[Annotated Image] First GT bbox: {ground_truth_bboxes[0]}")
                
                # Create annotated image with bounding boxes
                annotated_image = create_annotated_image_with_bboxes(image_path, predicted_bboxes, ground_truth_bboxes)
            
            elif question_type in ['bbox_3d', 'bounding_box_3d', '3d_bbox', 'pose', 'unknown']:
                # Parse 3D bounding boxes
                predicted_bboxes = parse_predicted_bboxes_3d(sample.get('output', ''))
                ground_truth_bboxes = parse_ground_truth_bboxes_3d(sample.get('ground_truth', ''))
                
                print(f"[Annotated Image] 3D BBox task - Predicted bboxes: {len(predicted_bboxes)}, GT bboxes: {len(ground_truth_bboxes)}")
                if predicted_bboxes:
                    print(f"[Annotated Image] First predicted 3D bbox: {predicted_bboxes[0]}")
                if ground_truth_bboxes:
                    print(f"[Annotated Image] First GT 3D bbox: {ground_truth_bboxes[0]}")
                
                # Create annotated image with 3D bounding boxes
                annotated_image = create_annotated_image_with_3d_bboxes(image_path, predicted_bboxes, ground_truth_bboxes)
            
            elif question_type == 'grasp':
                # Parse grasps
                predicted_grasp = parse_predicted_grasp(sample.get('output', ''))
                ground_truth_grasp = parse_ground_truth_grasp(sample.get('ground_truth', ''))
                
                print(f"[Annotated Image] Grasp task - Predicted: {predicted_grasp is not None}, GT: {ground_truth_grasp is not None}")
                if predicted_grasp:
                    print(f"[Annotated Image] Predicted grasp center: {predicted_grasp['center']}")
                if ground_truth_grasp:
                    print(f"[Annotated Image] GT grasp center: {ground_truth_grasp['center']}")
                
                # Create annotated image with grasps
                annotated_image = create_annotated_image_with_grasps(image_path, predicted_grasp, ground_truth_grasp)
            
            elif question_type == 'bench_point':
                # Parse predicted point and ground truth mask
                predicted_point = parse_predicted_point(sample.get('output', ''))
                ground_truth_mask = sample.get('ground_truth', '')
                
                print(f"[Annotated Image] Bench point task - Predicted: {predicted_point}, GT mask length: {len(ground_truth_mask) if ground_truth_mask else 0}")
                
                # Create annotated image with mask overlay
                annotated_image = create_annotated_image_with_mask(image_path, predicted_point, ground_truth_mask)
            
            # Convert PIL image to bytes
            img_buffer = BytesIO()
            annotated_image.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            print(f"[Annotated Image] Successfully created annotated image ({len(img_data)} bytes)")
            
            # Send the annotated image
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.send_header('Content-length', str(len(img_data)))
            self.end_headers()
            self.wfile.write(img_data)
            
        except Exception as e:
            print(f"[Annotated Image] ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.send_error(500, f"Error creating annotated image: {e}")


def main():
    parser = argparse.ArgumentParser(description="Web server for viewing multimodal validation logs")
    parser.add_argument("dir", help="Path to validation_outputs directory")
    parser.add_argument("--port", type=int, default=8080, help="Port to run server on (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()
    
    base_dir = Path(args.dir)
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Create handler class with base_dir
    handler = lambda *args, **kwargs: MultimodalLogHandler(*args, base_dir=base_dir, **kwargs)
    
    # Start server
    server = HTTPServer((args.host, args.port), handler)
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"Serving files from: {base_dir}")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()
