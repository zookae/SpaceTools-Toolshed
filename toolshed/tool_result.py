# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
ToolResult wrapper for standardized tool outputs.

A simple data container that carries both the return value and metadata
for tool outputs, without any magic proxy behavior.
"""

from typing import Any, Optional, List, Dict, TypeVar, Generic, Union

T = TypeVar('T')


class ToolResult(Generic[T]):
    """
    Simple data container for tool outputs with metadata.
    
    This is a plain data class that holds:
    - value: The actual return value
    - text: Text description for display
    - image/video: Optional media outputs
    - variables: Variables to store in context
    
    The execution wrapper explicitly unwraps the value when needed,
    and the Verl integration uses to_dict() to get structured data.
    
    Example:
        # In a tool:
        depth_map = compute_depth(image)
        return ToolResult(
            depth_map,
            text="Computed depth map", 
            variables={"depth_map": depth_map}
        )
        
        # In execution context (with wrapper):
        result = wrapped_tool.estimate_depth(image)
        # result is the raw depth_map, not ToolResult
        
        # In Verl integration:
        if isinstance(result, ToolResult):
            normalized = result.to_dict()
    """
    
    def __init__(
        self, 
        value: T,
        *,
        text: Optional[str] = None,
        image: Optional[Union[Any, List[Any]]] = None,
        video: Optional[Union[Any, List[Any]]] = None,
        variables: Optional[Dict[str, Any]] = None,
        is_error: bool = False
    ) -> None:
        """
        Initialize a ToolResult.
        
        Args:
            value: The actual return value that will be used in code execution
            text: Optional text description or message
            image: Optional image or list of images to display inline
            video: Optional video or list of videos to display inline  
            variables: Optional dict of {name: value} to store as context variables
            is_error: Whether this result represents an error condition
        """
        self.value = value
        self.text = text if text is not None else str(value)
        self.is_error = is_error
        
        # Normalize images and videos to lists
        if image is None:
            self.image = []
        elif isinstance(image, list):
            self.image = image
        else:
            self.image = [image]
            
        if video is None:
            self.video = []
        elif isinstance(video, list):
            self.video = video
        else:
            self.video = [video]
            
        self.variables = variables or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to the normalized dictionary format for Verl."""
        result: Dict[str, Any] = {"text": self.text}
        
        if self.is_error:
            result["is_error"] = True
        if self.image:
            result["image"] = self.image
        if self.video:
            result["video"] = self.video
        if self.variables:
            result["variables"] = [
                {"name": k, "value": v} 
                for k, v in self.variables.items()
            ]
            
        return result
    
    def __repr__(self) -> str:
        """Simple representation showing the value and metadata presence."""
        extras = []
        if self.is_error:
            extras.append("is_error=True")
        if self.image:
            extras.append(f"images={len(self.image)}")
        if self.video:
            extras.append(f"videos={len(self.video)}")
        if self.variables:
            extras.append(f"vars={list(self.variables.keys())}")
        
        extra_str = f", {', '.join(extras)}" if extras else ""
        return f"ToolResult(value={repr(self.value)}, text={repr(self.text[:50] + '...' if len(self.text) > 50 else self.text)}{extra_str})"
    
    def __str__(self) -> str:
        """String representation shows the text message."""
        return self.text
