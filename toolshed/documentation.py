# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Simple documentation generator for shed3d toolkit tools.

This module provides functionality to automatically generate documentation
by concatenating tool docstrings and method signatures.
"""

import inspect
from typing import get_type_hints, Dict, Any, List, Optional
from .registry import list_available_tools, get_tool_class
from .tools.base import BaseTool
from .doc_utils import process_conditional_docs


def _get_base_tool_methods() -> set:
    """Get all method names defined in the BaseTool base class."""
    base_methods = set()
    
    # Get all methods from BaseTool and its parent classes
    for method_name, method in inspect.getmembers(BaseTool, inspect.isfunction):
        if not method_name.startswith('_'):  # Only public methods
            base_methods.add(method_name)
    
    # Also check for abstract methods that might be implemented
    for method_name, method in inspect.getmembers(BaseTool, inspect.ismethod):
        if not method_name.startswith('_'):  # Only public methods
            base_methods.add(method_name)
    
    return base_methods


def _get_tool_description(tool_class) -> str:
    """Extract tool description from the class docstring (first line)."""
    doc = tool_class.__doc__
    if not doc:
        return "No description available"
    return doc.strip().split('\n')[0].strip()


def _format_type(type_obj) -> str:
    """Format a type object into a readable string."""
    if type_obj is None:
        return ''
    
    if hasattr(type_obj, '__name__'):
        return type_obj.__name__
    
    # Handle typing module types
    type_str = str(type_obj)
    
    # Clean up common typing patterns
    type_str = type_str.replace('typing.', '')
    type_str = type_str.replace('<class \'', '').replace('\'>', '')
    
    return type_str


def _get_method_signature(method) -> str:
    """Get a clean method signature string."""
    try:
        sig = inspect.signature(method)
        hints = get_type_hints(method)
        
        # Build parameter list
        params = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_str = param_name
            
            # Add type hint if available
            if param_name in hints:
                param_str += f": {_format_type(hints[param_name])}"
            elif param.annotation != inspect.Parameter.empty:
                param_str += f": {_format_type(param.annotation)}"
            
            # Add default value if available
            if param.default != inspect.Parameter.empty:
                param_str += f" = {repr(param.default)}"
            
            params.append(param_str)
        
        # Build full signature
        signature = f"({', '.join(params)})"
        
        # Add return type if available
        if 'return' in hints:
            signature += f" -> {_format_type(hints['return'])}"
        elif sig.return_annotation != inspect.Signature.empty:
            signature += f" -> {_format_type(sig.return_annotation)}"
        
        return signature
        
    except Exception:
        # If signature inspection fails, return empty
        return "()"


def generate_tool_documentation(tool_names: Optional[List[str]] = None, for_code_execution=True) -> str:
    """
    Generate documentation for the specified *tool_names*.

    If *tool_names* is ``None`` (default) the full list from
    ``list_available_tools()`` is used, preserving the original behaviour.

    Parameters
    ----------
    tool_names
        Optional list of tool names to document.  Pass only the *active* tool
        names to avoid importing heavy dependencies of unused tools.
    for_code_execution
        If *True*, exclude conditional documentation blocks.  Defaults to *True*

    Returns
    -------
    str
        A formatted documentation string.
    """

    # Back-compat: fall back to all available tools when no explicit list is
    # provided.  This keeps existing callers unchanged.
    if tool_names is None:
        tool_names = list_available_tools()

    if not tool_names:
        return "No tools are currently registered."
    
    doc_lines = [
        "# Available Tools Documentation",
        "",
        "This document describes all available tools and their methods.",
        "Use these tools by calling: toolkit.tool_name.method_name(parameters)",
        "",
        "=" * 80,
        ""
    ]
    
    for tool_name in tool_names:
        try:
            tool_class = get_tool_class(tool_name, allow_refresh=True)
            
            # Add tool header
            doc_lines.extend([
                f"## Tool: {tool_name}",
                f"Description: {_get_tool_description(tool_class)}",
                ""
            ])
            
            # Get all public methods (excluding BaseTool methods)
            base_methods = _get_base_tool_methods()
            methods = []
            
            for method_name, method in inspect.getmembers(tool_class, inspect.isfunction):
                if not method_name.startswith('_') and method_name not in base_methods:
                    methods.append((method_name, method))
            
            if not methods:
                doc_lines.extend([
                    "*No public methods available*",
                    ""
                ])
                continue
            
            # Add methods section
            doc_lines.append("### Methods:")
            doc_lines.append("")
            
            for method_name, method in methods:
                # Get method signature
                signature = _get_method_signature(method)
                
                # Add method header
                doc_lines.extend([
                    f"**toolkit.{tool_name}.{method_name}{signature}**",
                    ""
                ])
                
                # Add full docstring if available
                method_doc = inspect.getdoc(method)
                if method_doc:
                    # For code exeuction docs, remove all conditional documentation
                    processed = process_conditional_docs(
                        method_doc, 
                        include_image=not for_code_execution, 
                        include_vars=not for_code_execution, 
                        include_text=not for_code_execution)
                    for line in processed.split('\n'):
                        doc_lines.append(f"  {line}")
                else:
                    doc_lines.append("  No documentation available.")
                
                doc_lines.extend([
                    "",
                    "-" * 40,
                    ""
                ])
            
        except Exception as e:
            doc_lines.extend([
                f"*Error generating documentation for {tool_name}: {e}*",
                ""
            ])
    
    doc_lines.extend([
        "=" * 80,
        "",
        "End of documentation."
    ])
    
    return '\n'.join(doc_lines)


def get_tool_summary() -> str:
    """
    Get a quick summary of available tools and their methods.
    
    Returns:
        Brief summary of tools and methods
    """
    tool_names = list_available_tools()
    if not tool_names:
        return "No tools are currently registered."
    
    summary_lines = ["Available Tools Summary:"]
    
    for tool_name in tool_names:
        try:
            tool_class = get_tool_class(tool_name, allow_refresh=True)
            summary_lines.append(f"- {tool_name}: {_get_tool_description(tool_class)}")
            
            # Get method names
            base_methods = _get_base_tool_methods()
            methods = [name for name, _ in inspect.getmembers(tool_class, inspect.isfunction)
                      if not name.startswith('_') and name not in base_methods]
            
            if methods:
                summary_lines.append(f"  Methods: {', '.join(methods)}")
            else:
                summary_lines.append("  No public methods")
            
        except Exception as e:
            summary_lines.append(f"- {tool_name}: Error - {e}")
    
    return '\n'.join(summary_lines) 