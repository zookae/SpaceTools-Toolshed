# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import ast
from PIL import Image
import re
from pathlib import Path

def convert_answer(pixel_answer_str, image_width, image_height):
    """Convert pixel coordinates to normalized coordinates with 4 corners."""
    # Parse the string as a Python list
    pixel_boxes = ast.literal_eval(pixel_answer_str)
    
    normalized_boxes = []
    for x1, y1, x2, y2 in pixel_boxes:
        # Normalize coordinates
        nx1 = x1 / image_width
        ny1 = y1 / image_height
        nx2 = x2 / image_width
        ny2 = y2 / image_height
        
        # Create 4 corners: top-left, top-right, bottom-right, bottom-left
        corners = [
            (nx1, ny1),  # top-left
            (nx2, ny1),  # top-right
            (nx2, ny2),  # bottom-right
            (nx1, ny2)   # bottom-left
        ]
        normalized_boxes.append(corners)
    
    return str(normalized_boxes)

def extract_scene_and_filename(image_path):
    """Extract scene_id and filename from the image path."""
    # Pattern: .../scannet/posed_images/scene####_##/filename.jpg
    match = re.search(r'scannet/posed_images/(scene\d+_\d+)/(\d+\.jpg)', image_path)
    if match:
        scene_id = match.group(1)
        filename = match.group(2)
        return scene_id, filename
    else:
        raise ValueError(f"Could not extract scene_id and filename from: {image_path}")

def convert_dataset(input_json_path, output_json_path):
    """Convert the entire dataset from pixel to normalized coordinates."""
    # Load the input JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    converted_data = []
    
    for i, example in enumerate(data):
        print(f"Processing example {i+1}/{len(data)}...")
        
        # Extract scene_id and filename
        scene_id, filename = extract_scene_and_filename(example['image_path'])
        
        # Construct the actual full path to load the image
        full_image_path = f"/lustre/fsw/portfolios/nvr/users/vblukis/datasets/RoboSpatial/robospatial-train-full/images/scannet/posed_images/{scene_id}/{filename}"
        
        # Construct the relative path for the output
        relative_image_path = f"scannet/posed_images/{scene_id}/{filename}"
        
        # Load the image to get dimensions
        try:
            with Image.open(full_image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error loading image {full_image_path}: {e}")
            continue
        
        # Convert the question
        converted_question = example['question'].replace(
            "Your final answer should be a list of bounding boxes in the format [(x1, y1, x2, y2), ...] with image pixel coordinates.",
            "Format the result as a list of bounding boxes, each a tuple containing the four corner coordinates: [ [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], ... ]. Here, x and y are normalized pixel location of the point with values between 0 and 1."
        )
        
        # Convert the answer
        converted_answer = convert_answer(example['answer'], width, height)
        
        # Create the new example
        converted_example = {
            'image_path': relative_image_path,
            'question': converted_question,
            'answer': converted_answer,
            'qa_type': example['qa_type']
        }
        
        converted_data.append(converted_example)
    
    # Save the output
    with open(output_json_path, 'w') as f:
        json.dump(converted_data, f, indent=4)
    
    print(f"Conversion complete! Saved to {output_json_path}")
    print(f"Converted {len(converted_data)} examples")

if __name__ == "__main__":
    input_path = "/lustre/fsw/portfolios/nvr/users/siyic/projects/robospatial_scannet_updated/grounding.json"
    output_path = "/lustre/fsw/portfolios/nvr/users/vblukis/code/toolshed/data/grounding_normalized.json"
    
    convert_dataset(input_path, output_path)