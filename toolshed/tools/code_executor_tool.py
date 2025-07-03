# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Code Executor Tool
==================

Expose the existing ``toolshed.code_executor.CodeExecutor`` as a regular
Toolshed tool so that it can be managed by the router and therefore be
invoked by an LLM like any other atomic tool.  The tool provides two main
methods:

``exec(code: str)`` – execute an arbitrary Python block and return
              (result, stdout, stderr)
``eval(expression: str)`` – evaluate a single Python expression and return
              (result, stdout, stderr)

The class also forwards ``extract_python_code_from_markdown`` to help the
LLM run ```python fenced blocks directly.

NOTE: Executing arbitrary code obviously has security implications.  Only
use this tool in trusted environments.
"""

import inspect
from typing import Any, Dict, Tuple

from .base import BaseTool, tool_method
from ..code_executor import CodeExecutor
from toolshed.tool_result import ToolResult

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class CodeExecutorTool(BaseTool):
    """Execute Python code with access to the Toolshed toolkit."""

    def __init__(self, no_output_image: bool = False, no_output_vars: bool = False,
                 exclude_methods: list[str] = None, exclude_behavior: str = "warn",
                 router_name: str = "toolshed_router", namespace: str = "toolshed"):
        super().__init__(no_output_image, no_output_vars, exclude_methods, exclude_behavior)
        # Keep a single CodeExecutor instance per actor instance; this gives us
        # stateful execution context between successive calls if desired.
        self._executor = CodeExecutor(router_name=router_name, namespace=namespace, auto_connect=True)

    # ------------------------------------------------------------------
    # Metadata helpers required by BaseTool
    # ------------------------------------------------------------------
    def get_name(self) -> str:
        return "code_executor"

    # ------------------------------------------------------------------
    # Public methods available to the LLM/client
    # ------------------------------------------------------------------
    @tool_method
    def exec(self, code: str, variables: Dict[str, Any] = None) -> ToolResult[Tuple[Any, str, str]]:
        """Execute a multi-line block of Python code. You are allowed to import math and numpy modules. No other imports are allowed.

        To return data from your code, assign it to a variable named 'result'. This will be captured and 
        made available as output. For example: result = [1, 2, 3] or result = processed_image.

        [[if:text]]Text output: The execution result (if any), along with captured stdout and stderr output. If you assign 
        a value to 'result' in your code, it will be included in the output.[[/if:text]]
        [[if:image]]Visual output: If result contains a PIL image, it will be displayed inline.[[/if:image]]
        [[if:vars]]Stored variables: If you assign to 'result', that value is stored as $result for future use.[[/if:vars]]

        Args:
            code (str): The Python code block to execute. Can include multiple statements and expressions. 
                To return data, assign it to a variable named 'result' (e.g., result = my_calculation).
            variables (Dict[str, Any], optional): Dictionary with keys=variable names that will become python variables, 
                and values their corresponding values. For example: {"threshold": 0.5} will make 
                `threshold` available as a variable in the executed code.
                [[if:vars]]Previously stored variables can be accessed using the $ syntax in the variables parameter ONLY, 
                e.g. to pass a stored variable called stored_var_name, use {"var_name": $stored_var_name}. 
                DO NOT USE THE $ VARIABLE LOOKUP SYNTAX INSIDE THE PYTHON CODE ITSELF, ONLY IN THE TOOL CALL ARGUMENTS.[[/if:vars]]
        """
        # Reset context to start fresh (state is managed via session variables)
        self._executor.reset_context()
        
        # Merge input variables into the execution context if provided
        if variables:
            self._executor.update_context(variables)
            
        result, stdout, stderr = self._executor.exec(code)
        
        print("--------------------------------")
        print(f"Code Executor Tool: Code Executed: {code}")
        print("--------------------------------")
        print(f"Result: {result}")
        print("--------------------------------")
        print(f"Stdout: {stdout}")
        print(f"Stderr: {stderr}")
        print("--------------------------------")

        # Format the text output
        text_parts = []
        if result is not None:
            # For large arrays/objects, show a truncated representation
            result_str = str(result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "... (truncated)"
            text_parts.append(f"Result: {result_str}")
        if stdout:
            text_parts.append(f"Stdout: {stdout}")
        if stderr:
            text_parts.append(f"Stderr: {stderr}")
        
        text = "\n".join(text_parts) if text_parts else "Code executed successfully (no output)"
        
        # Store the execution result as variables for potential reuse (if not suppressed)
        variables = {}
        result_image = None
        
        if not self.no_output_vars and result is not None:
            # Also store as 'result' for consistency
            variables["result"] = result
            
            # Check if result is an image for display
            if _PIL_AVAILABLE and isinstance(result, Image.Image):
                result_image = result
                text += " The result is an image that will be displayed."
            
            # Add reference to variables in text
            text += " Use $result to reference the result."
        
        return ToolResult(
            (result, stdout, stderr),
            text=text,
            variables=variables,
            image=result_image if not self.no_output_image else None
        )

    @tool_method
    def eval(self, expression: str, variables: Dict[str, Any] = None) -> ToolResult[Tuple[Any, str, str]]:
        """Evaluate a single Python expression.

        [[if:text]]Text output: The evaluated result, along with any captured stdout and stderr output.[[/if:text]]
        [[if:image]]Visual output: If result contains a PIL image, it will be displayed inline.[[/if:image]]
        [[if:vars]]Stored variables: The evaluation result in $result variables for use in subsequent operations.[[/if:vars]]

        Args:
            expression (str): The Python expression to evaluate. Must be a single expression that returns a value.
            variables (Dict[str, Any], optional): Dictionary with keys=variable names that will become python variables, 
                and values their corresponding values. For example: {"threshold": 0.5} will make 
                `threshold` available as a variable in the evaluated expression.
                [[if:vars]]Previously stored variables can be accessed using the $ syntax in the variables argument ONLY, 
                e.g. to pass a stored variable called stored_var_name, use {"var_name": $stored_var_name}. 
                DO NOT USE THE $ VARIABLE LOOKUP SYNTAX INSIDE THE PYTHON CODE ITSELF, ONLY IN THE TOOL CALL ARGUMENTS.[[/if:vars]]
        """
        # Reset context to start fresh (state is managed via session variables)
        self._executor.reset_context()
        
        # Merge input variables into the execution context if provided
        if variables:
            self._executor.update_context(variables)
            
        result, stdout, stderr = self._executor.exec(expression)
        
        # Format the text output
        text_parts = []
        if result is not None:
            # For large arrays/objects, show a truncated representation
            result_str = str(result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "... (truncated)"
            text_parts.append(f"Result: {result_str}")
        if stdout:
            text_parts.append(f"Stdout: {stdout}")
        if stderr:
            text_parts.append(f"Stderr: {stderr}")
        
        text = "\n".join(text_parts) if text_parts else "Expression evaluated (no output)"
        
        # Store the evaluation result as variables for potential reuse (if not suppressed)
        variables = {}
        result_image = None
        
        if not self.no_output_vars and result is not None:
            variables["result"] = result
            
            # Check if result is an image for display
            if _PIL_AVAILABLE and isinstance(result, Image.Image):
                result_image = result
                text += " The result is an image that will be displayed."
            
            # Add reference to variables in text
            text += " Use $result to reference the result."
        
        return ToolResult(
            (result, stdout, stderr),
            text=text,
            variables=variables,
            image=result_image if not self.no_output_image else None
        )

    def extract_python_code_from_markdown(self, text: str) -> str | None:
        """Extract the last ```python code block from markdown text.
        
        Args:
            text: The markdown text to parse
        """
        return self._executor.extract_python_code_from_markdown(text) 