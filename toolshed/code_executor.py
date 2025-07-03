# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Code executor for executing Python code with access to the shed3d toolkit.

This module provides a CodeExecutor class that can execute Python code
in an environment where the shed3d toolkit is available, supporting both
eval (return value) and exec (console output) modes.
"""

import io
import re
import sys
import traceback
from typing import Any, Tuple, Optional, Dict
from contextlib import redirect_stdout, redirect_stderr
from .tool_client import get_toolkit, ToolkitClient
from .execution_wrappers import create_execution_toolkit
from .documentation import generate_tool_documentation
from .doc_utils import process_conditional_docs


class CodeExecutor:
    """
    Execute Python code with access to the shed3d toolkit.
    
    This class provides a sandboxed environment for executing Python code
    where the shed3d toolkit is available in the execution context.
    
    Example:
        executor = CodeExecutor()
        result, stdout, stderr = executor.exec("toolkit.greeting.greet('Alice')")
    """
    
    def __init__(self, 
                 router_name: str = "toolshed_router", 
                 namespace: str = "toolshed",
                 auto_connect: bool = True):
        """
        Initialize the code executor.
        
        Args:
            router_name: Name of the ToolRouterActor to connect to
            namespace: Ray namespace where the router actor is located
            auto_connect: Whether to automatically connect to the toolkit on init
        """
        self.router_name = router_name
        self.namespace = namespace
        self.toolkit: Optional[ToolkitClient] = None
        self.execution_context: Dict[str, Any] = {}
        
        if auto_connect:
            self._connect_toolkit()
        
        # Initialize the execution context with useful imports and toolkit
        self._setup_execution_context()
    
    def _connect_toolkit(self):
        """Connect to the shed3d toolkit."""
        try:
            self.toolkit = get_toolkit(self.router_name, self.namespace)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to toolkit: {e}")
    
    def _setup_execution_context(self):
        """Setup the execution context with common imports and toolkit access."""
        # Standard imports that are commonly useful
        import_context = {
            # Standard library
            'json': __import__('json'),
            'time': __import__('time'),
            'math': __import__('math'),
            'datetime': __import__('datetime'),
            'random': __import__('random'),
            're': __import__('re'),
            
            # Common third-party (if available)
            'np': None,
            'pd': None,
        }
        
        # Try to import numpy if available
        try:
            import numpy as np
            import_context['np'] = np
        except ImportError:
            pass
        
        # Add toolkit access (wrapped for execution safety)
        if self.toolkit:
            import_context['toolkit'] = create_execution_toolkit(self.toolkit)
        
        # Add useful functions
        import_context['print'] = print
        import_context['len'] = len
        import_context['range'] = range
        import_context['list'] = list
        import_context['dict'] = dict
        import_context['str'] = str
        import_context['int'] = int
        import_context['float'] = float
        import_context['bool'] = bool
        
        self.execution_context = import_context
    
    def get_toolkit(self) -> ToolkitClient:
        """Get the connected toolkit client."""
        if self.toolkit is None:
            self._connect_toolkit()
        return self.toolkit
    
    def extract_python_code_from_markdown(self, code: str) -> Optional[str]:
        """
        Extract Python code from a markdown-formatted string.
        
        Looks for the latest ```python code block in the string and returns
        just the Python code content.
        
        Args:
            code: String that may contain markdown-formatted Python code blocks
            
        Returns:
            The Python code from the latest ```python block, or None
            if no such block is found
        """
        # Pattern to match ```python ... ``` blocks
        # This pattern handles:
        # - Optional whitespace after ```python
        # - The code content (captured group)
        # - Optional whitespace before closing backticks
        pattern = r'```python\s*\n(.*?)\n\s*```'
        
        # Find all matches
        matches = re.findall(pattern, code, re.DOTALL)
        
        if matches:
            # Return the last (latest) match, stripped of leading/trailing whitespace
            return matches[-1].strip()
        else:
            # If no python code blocks found, return None
            return None

    def exec(self, code: str) -> Tuple[Any, str, str]:
        """
        Execute code and return the result value along with stdout/stderr.
        
        This method evaluates the code and returns the result of the last expression.
        
        Args:
            code: Python code to execute
            
        Returns:
            Tuple of (result, stdout, stderr)
            - result: The return value of the executed code (if it's a simple expression) or value of the "result" variable
            - stdout: Captured stdout during execution
            - stderr: Captured stderr during execution
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = None
        
        try:
            # Update toolkit in context (wrapped for execution safety)
            self.execution_context['toolkit'] = create_execution_toolkit(self.get_toolkit())
            
            # Execute with captured output
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try to evaluate as expression first
                try:
                    # For single expressions, use eval
                    compiled_code = compile(code, '<string>', 'eval')
                    result = eval(compiled_code, self.execution_context)
                except SyntaxError:
                    # If it's not a single expression, use exec
                    compiled_code = compile(code, '<string>', 'exec')
                    exec(compiled_code, self.execution_context)
                    
                    # Assume a result may be stored in a result variable
                    if "result" in self.execution_context:
                        result = self.execution_context["result"]
                    else:
                        result = None
                                
        except Exception as e:
            # Capture the exception in stderr using standard Python format
            # but only include frames from user code (filename == '<string>')
            exc_type, exc_value, exc_tb = sys.exc_info()
            
            # Extract traceback frames and filter to only user code
            tb_list = traceback.extract_tb(exc_tb)
            user_frames = [frame for frame in tb_list if frame.filename == '<string>']
            
            if user_frames:
                # Use standard Python traceback format
                stderr_capture.write("Traceback (most recent call last):\n")
                stderr_capture.write(''.join(traceback.format_list(user_frames)))
                stderr_capture.write(f"{exc_type.__name__}: {exc_value}\n")
            else:
                # If no user frames (error in internal code), show minimal info
                stderr_capture.write(f"{exc_type.__name__}: {exc_value}\n")
            
            result = None
        
        return result, stdout_capture.getvalue(), stderr_capture.getvalue()
    
    def update_context(self, variables: Dict[str, Any]):
        """
        Update the execution context with additional variables.
        
        Args:
            variables: Dictionary of variable names and values to add to context
        """
        self.execution_context.update(variables)
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current execution context.
        
        Returns:
            Dictionary of current context variables
        """
        return self.execution_context.copy()
    
    def reset_context(self):
        """Reset the execution context to initial state."""
        self.execution_context.clear()
        self._setup_execution_context()
    
    def get_available_tools(self) -> list:
        """
        Get list of available tools in the toolkit.
        
        Returns:
            List of available tool names
        """
        if self.toolkit is None:
            self._connect_toolkit()
        return self.toolkit.get_available_tools()

    def get_documentation(self) -> str:
        """Return documentation for code execution context (no conditional blocks).
        
        Only documents tools that are currently running (active tools) to avoid
        importing heavy dependencies of unused tools.
        """
        if self.toolkit is None:
            self._connect_toolkit()
        # Use active_only=True to avoid importing all tools
        docs = self.toolkit.get_documentation(active_only=True, for_code_execution=True)
        return docs
    
    def get_tool_status(self) -> dict:
        """
        Get status of all tools in the toolkit.
        
        Returns:
            Dictionary with tool status information
        """
        if self.toolkit is None:
            self._connect_toolkit()
        return self.toolkit.get_tool_status() 