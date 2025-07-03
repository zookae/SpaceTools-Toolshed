# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from toolshed.tools.base import BaseTool, tool_method
from toolshed.tool_result import ToolResult
import json
import os
from pathlib import Path
from PIL import Image
import ray


class SaveDataTool(BaseTool):
    """Save annotated data to disk in JSON format."""
    
    def __init__(self, save_path: str = "outputs/annotation", **kwargs):
        super().__init__(**kwargs)
        self.save_path = Path(save_path).resolve()  # Convert to absolute path
        self.save_path.mkdir(parents=True, exist_ok=True)
        print(f"[save_data] Initialized with save_path: {self.save_path}")
    
    def get_name(self) -> str:
        return "save_data"
    
    @tool_method
    def save_annotation(self, image, name: str, data: dict, explanation: str) -> ToolResult:
        """
        Save annotated data to a JSON file.
        
        Args:
            image: Input image (PIL Image, numpy array, or Ray ObjectRef)
            name: Unique name identifier for this annotation
            data: Python dictionary containing the annotation data
            explanation: Explanation listing all the tools you used and how you solved the task

        Returns:
            Confirmation message with file path
        """        
        # Convert numpy array to PIL Image if needed
        if not isinstance(image, Image.Image):
            import numpy as np
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
        
        # Save image
        image_filename = f"{name}.png"
        image_path = self.save_path / image_filename
        image.save(image_path)
        print(f"[save_data] Saved image: {image_path.absolute()}")
        
        # Handle case where data is passed as a JSON string instead of a dict
        if isinstance(data, str):
            data = json.loads(data)
        
        # Save JSON data
        json_filename = f"{name}.json"
        json_path = self.save_path / json_filename
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[save_data] Saved JSON: {json_path.absolute()}")
        print(f"[save_data] Data: {json.dumps(data, indent=2)}")
        
        # Save explanation
        explanation_filename = f"{name}.txt"
        explanation_path = self.save_path / explanation_filename
        with open(explanation_path, 'w') as f:
            f.write(explanation)
        print(f"[save_data] Saved explanation: {explanation_path.absolute()}")

        return ToolResult(
            value={"image_path": str(image_path), "json_path": str(json_path)},
            text=f"Saved annotation '{name}': image to {image_path}, data to {json_path}"
        )

