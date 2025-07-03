# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This template is a guide for creating new tools for the toolshed toolkit.
"""

from types import EllipsisType


from .base import BaseTool, tool_method
from toolshed.tool_result import ToolResult

from typing import Union
import time

class ToolTemplate(BaseTool):
    """
    High level description of the tool.
    
    More detailed description of the tool.
    """
    
    def __init__(self, no_output_image: bool = False, no_output_vars: bool = False,
                 exclude_methods: list[str] = None, exclude_behavior: str = "warn"):
        """Initialize the tool.
        
        Note: BaseTool automatically tracks call statistics for all @tool_method decorated methods.
        No need to manually implement get_stats() unless you need custom statistics.
        """
        super().__init__(no_output_image, no_output_vars, exclude_methods, exclude_behavior)
    
    def get_name(self) -> str:
        return "example_tool"

    @tool_method
    def your_method(self, a: float, b: float) -> ToolResult[float]:
        """
        High level description of tool method.
        
        Detailed description of tool method
        over multiple lines.
        [[if:text]]Text output: Describe the text that this tool adds to the conversation.[[/if:text]]
        [[if:image]]Image output: Describe an image (if any) that this tool adds to the conversation.[[/if:image]]
        [[if:vars]]Variable output: Describe any variables that this tool saves in the conversation context.[[/if:vars]]

        Args:
            a: First argument
            b: Second argument
            
        Returns:
            Description of return value for a code execution context.
        """
        result = None
        return ToolResult[EllipsisType](result,text=f"tool output text", image=..., variables={})

# Register the tool

