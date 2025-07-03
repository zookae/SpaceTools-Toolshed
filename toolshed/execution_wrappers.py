# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Execution wrappers for unwrapping ToolResult in code execution context.

These wrappers provide a clean separation between the Verl integration
(which needs ToolResult for rich outputs) and code execution context
(which needs raw values for type checking and operations).
"""

from typing import Any, List, Optional
from toolshed.tool_result import ToolResult


class ToolkitExecutionWrapper:
    """
    Wrapper that provides execution-safe access to tools.
    
    This wrapper ensures that all tool method calls return raw values
    instead of ToolResult objects, making them safe for use in generated
    code where type checks and operations expect actual types.
    """
    
    def __init__(self, toolkit):
        """
        Initialize the wrapper with a toolkit.
        
        Args:
            toolkit: The underlying ToolkitClient to wrap
        """
        self._toolkit = toolkit
    
    def get_tool(self, name: str):
        """
        Get a wrapped tool that automatically unwraps ToolResult.
        
        Args:
            name: Name of the tool to get
            
        Returns:
            ToolExecutionWrapper that unwraps ToolResult from all methods
        """
        tool = self._toolkit.get_tool(name)
        return ToolExecutionWrapper(tool)
    
    def list_tools(self) -> List[str]:
        """List available tools."""
        return self._toolkit.list_tools()
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return self._toolkit.get_available_tools()
    
    def get_tool_status(self) -> dict:
        """Get status of all tools."""
        return self._toolkit.get_tool_status()
    
    def __getattr__(self, name: str):
        """
        Direct attribute access for tools (e.g., toolkit.calculator).
        
        This allows both:
        - toolkit.get_tool('calculator')
        - toolkit.calculator
        """
        # Check if it's a tool name
        if hasattr(self._toolkit, name):
            attr = getattr(self._toolkit, name)
            # If it's a tool, wrap it
            if hasattr(attr, '__call__'):
                # It's a method, just forward it
                return attr
            else:
                # It might be a tool attribute access
                return ToolExecutionWrapper(attr)
        
        # Try to get it as a tool
        try:
            return self.get_tool(name)
        except:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class ToolExecutionWrapper:
    """
    Wraps a single tool to automatically unwrap ToolResult from method calls.
    
    This wrapper intercepts all method calls on a tool and ensures that
    if the method returns a ToolResult, only the wrapped value is returned
    to the caller.
    """
    
    def __init__(self, tool):
        """
        Initialize the wrapper with a tool.
        
        Args:
            tool: The tool instance to wrap
        """
        self._tool = tool
        # Preserve tool metadata for introspection
        self.name = getattr(tool, 'name', tool.__class__.__name__)
        self.__doc__ = getattr(tool, '__doc__', None)
    
    def __getattr__(self, name: str):
        """
        Intercept attribute access to wrap method calls.
        
        Args:
            name: Name of the attribute being accessed
            
        Returns:
            For methods: A wrapper function that unwraps ToolResult
            For other attributes: The attribute value directly
        """
        attr = getattr(self._tool, name)
        
        # If it's not callable, return it directly
        if not callable(attr):
            return attr
        
        # Create a wrapper that unwraps ToolResult
        def unwrapping_call(*args, **kwargs):
            """Execute the method and unwrap ToolResult if present."""
            result = attr(*args, **kwargs)
            
            # If the result is a ToolResult, return just the value
            if isinstance(result, ToolResult):
                return result.value
            
            # Otherwise return the result as-is
            return result
        
        # Preserve method metadata for introspection
        unwrapping_call.__name__ = getattr(attr, '__name__', name)
        unwrapping_call.__doc__ = getattr(attr, '__doc__', None)
        
        # Check if the original method has the _is_tool_method marker
        if hasattr(attr, '_is_tool_method'):
            unwrapping_call._is_tool_method = True
        
        return unwrapping_call
    
    def __dir__(self):
        """Return list of available attributes for introspection."""
        return dir(self._tool)
    
    def __repr__(self):
        """String representation for debugging."""
        return f"ToolExecutionWrapper({self._tool!r})"


def create_execution_toolkit(toolkit):
    """
    Create an execution-safe version of a toolkit.
    
    This is a convenience function that creates a ToolkitExecutionWrapper
    for use in code execution contexts.
    
    Args:
        toolkit: The toolkit to wrap
        
    Returns:
        ToolkitExecutionWrapper that provides execution-safe tool access
    
    Example:
        >>> toolkit = get_toolkit()
        >>> exec_toolkit = create_execution_toolkit(toolkit)
        >>> # Now tool methods return raw values, not ToolResult
        >>> result = exec_toolkit.get_tool('calculator').add(5, 3)
        >>> assert isinstance(result, float)  # True! Not ToolResult
    """
    return ToolkitExecutionWrapper(toolkit)
