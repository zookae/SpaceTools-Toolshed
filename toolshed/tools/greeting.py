# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Greeting tool - a simple example tool for the toolshed toolkit.
"""

from .base import BaseTool, tool_method
from toolshed.tool_result import ToolResult
import time


class GreetingTool(BaseTool):
    """
    A simple greeting tool that demonstrates the toolkit architecture.
    
    This tool takes a name and returns a greeting message.
    """
    
    def __init__(self, no_output_image: bool = False, no_output_vars: bool = False,
                 exclude_methods: list[str] = None, exclude_behavior: str = "warn"):
        """Initialize the greeting tool."""
        super().__init__(no_output_image, no_output_vars, exclude_methods, exclude_behavior)
    
    def get_name(self) -> str:
        """Get the name of the tool."""
        return "greeting"

    @tool_method
    def greet(self, name: str) -> ToolResult[str]:
        """
        Generate a greeting message.
        
        [[if:text]]Text output: A friendly greeting message with the provided name and greeting count.[[/if:text]]
        
        Args:
            name: Name of the person to greet
            
        Returns:
            str: Greeting message
        """
        # Get current count (will be incremented by decorator after this returns)
        current_count = self._method_call_stats.get('greet', 0) + 1
        message = f"Hello {name}! (Greeting #{current_count})"
        return ToolResult(message, text=message)
    
    @tool_method
    def formal_greet(self, name: str, title: str = "Mr./Ms.") -> ToolResult[str]:
        """
        Generate a formal greeting message.
        
        [[if:text]]Text output: A formal greeting message with title, name, and greeting count.[[/if:text]]
        
        Args:
            name: Name of the person to greet
            title: Title to use (default: "Mr./Ms.")
            
        Returns:
            str: Formal greeting message
        """
        # Get current count (will be incremented by decorator after this returns)
        current_count = self._method_call_stats.get('formal_greet', 0) + 1
        message = f"Good day, {title} {name}! How may I assist you today? (Greeting #{current_count})"
        return ToolResult(message, text=message)
    
    @tool_method
    def slow_greet(self, name: str, delay: float = 1.0) -> ToolResult[str]:
        """
        Generate a greeting with a simulated delay (for testing concurrency).
        
        [[if:text]]Text output: A greeting message with the provided name, delay duration, and greeting count.[[/if:text]]
        
        Args:
            name: Name of the person to greet
            delay: Delay in seconds
            
        Returns:
            str: Greeting message after delay
        """
        time.sleep(delay)
        # Get current count (will be incremented by decorator after this returns)
        current_count = self._method_call_stats.get('slow_greet', 0) + 1
        message = f"Hello {name}! (After {delay}s delay, Greeting #{current_count})"
        return ToolResult(message, text=message)


# Register the tool 