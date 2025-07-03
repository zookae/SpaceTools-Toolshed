# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Calculator tool - provides basic arithmetic operations for the toolshed toolkit.
This is just a a simple example of a tool.
"""

from .base import BaseTool, tool_method
from toolshed.tool_result import ToolResult

from typing import Union
import time


class CalculatorTool(BaseTool):
    """
    A calculator tool that performs basic arithmetic operations.
    
    This tool provides addition, subtraction, multiplication, and division
    operations with proper error handling.
    """
    
    def __init__(self, no_output_image: bool = False, no_output_vars: bool = False,
                 exclude_methods: list[str] = None, exclude_behavior: str = "warn"):
        """Initialize the calculator tool."""
        super().__init__(no_output_image, no_output_vars, exclude_methods, exclude_behavior)
    
    def get_name(self) -> str:
        """Get the name of the tool."""
        return "calculator"
    
    @tool_method
    def add(self, a: float, b: float) -> ToolResult[float]:
        """
        Add two numbers.
        
        [[if:text]]Text output: The sum calculation result with the format "a + b = result".[[/if:text]]
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            float: Sum of a and b
        """
        result = a + b
        time.sleep(0.2)
        return ToolResult(result, text=f"{a} + {b} = {result}")
    
    @tool_method
    def subtract(self, a: float, b: float) -> ToolResult[float]:
        """
        Subtract the second number from the first.
        
        [[if:text]]Text output: The subtraction calculation result with the format "a - b = result".[[/if:text]]
        
        Args:
            a: Number to subtract from
            b: Number to subtract
            
        Returns:
            float: Difference of a and b (a - b)
        """
        result = a - b
        time.sleep(0.2)
        return ToolResult(result, text=f"{a} - {b} = {result}")
    
    @tool_method
    def multiply(self, a: float, b: float) -> ToolResult[float]:
        """
        Multiply two numbers.
        
        [[if:text]]Text output: The multiplication calculation result with the format "a × b = result".[[/if:text]]
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            float: Product of a and b
        """
        result = a * b
        time.sleep(0.2)
        return ToolResult(result, text=f"{a} × {b} = {result}")
    
    @tool_method
    def divide(self, a: float, b: float) -> ToolResult[float]:
        """
        Divide two numbers.
        
        [[if:text]]Text output: The division calculation result with the format "a ÷ b = result".[[/if:text]]
        
        Args:
            a: Dividend
            b: Divisor (cannot be zero)
            
        Returns:
            float: Result of a / b
            
        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        time.sleep(0.2)
        return ToolResult(result, text=f"{a} ÷ {b} = {result}")
    
    @tool_method
    def power(self, base: float, exponent: float) -> ToolResult[float]:
        """
        Raise a number to a power.
        
        [[if:text]]Text output: The power calculation result with the format "base^exponent = result".[[/if:text]]
        
        Args:
            base: The base number
            exponent: The exponent
            
        Returns:
            float: Result of base raised to the power of exponent
        """
        result = base ** exponent
        time.sleep(0.2)
        return ToolResult(result, text=f"{base}^{exponent} = {result}")
    
    @tool_method
    def modulo(self, a: float, b: float) -> ToolResult[float]:
        """
        Get the remainder of division.
        
        [[if:text]]Text output: The modulo calculation result with the format "a mod b = result".[[/if:text]]
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            float: Remainder of a divided by b
            
        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            raise ValueError("Cannot perform modulo with zero divisor")
        
        result = a % b
        time.sleep(0.2)
        return ToolResult(result, text=f"{a} mod {b} = {result}")
    


# Register the tool

