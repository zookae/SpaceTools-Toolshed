# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Base classes for shed3d toolkit tools.
"""

from abc import ABC, abstractmethod
from enum import Enum
import warnings

# ---------------------------------------------------------------------------
# Decorator to mark LLM-exposed tool methods
# ---------------------------------------------------------------------------

def tool_method(func=None, *, requires_image_output=False, requires_var_output=False):
    """Decorator to mark a method as an LLM-exposed *tool method*.

    Only methods decorated with ``@tool_method`` will be exposed to external
    callers (e.g. the Verl integration / OpenAI function tools).  The decorator
    automatically wraps methods to track call statistics.
    
    Args:
        requires_image_output: If True, only include in schema if no_output_image=False
        requires_var_output: If True, only include in schema if no_output_vars=False
    """
    from functools import wraps
    
    def decorator(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            # Track method call statistics
            if hasattr(self, '_method_call_stats'):
                method_name = f.__name__
                self._method_call_stats[method_name] = self._method_call_stats.get(method_name, 0) + 1
                self._method_call_stats['_total_calls'] = self._method_call_stats.get('_total_calls', 0) + 1
            
            # Call the actual method
            return f(self, *args, **kwargs)
        
        # Mark the wrapper with tool_method attributes
        wrapper._is_tool_method = True  # type: ignore[attr-defined]
        wrapper._requires_image_output = requires_image_output  # type: ignore[attr-defined]
        wrapper._requires_var_output = requires_var_output  # type: ignore[attr-defined]
        return wrapper
    
    if func is None:
        # Called with arguments: @tool_method(requires_image_output=True)
        return decorator
    else:
        # Called without arguments: @tool_method
        return decorator(func)


class BaseTool(ABC):
    """
    Base class for all tools in the shed3d toolkit.
    
    All tools should inherit from this class and implement the required methods.
    
    Tools should accept and respect the no_output_* flags when populating ToolResult.

    Warning:
        Tools should not implement __call__ on the tool instance itself. The runtime wraps
        tools and their methods for execution safety; defining a callable tool instance could
        bypass wrappers (e.g., ToolResult unwrapping) in some access paths. Stick to
        explicit @tool_method-decorated methods.
    """
    
    def __init__(self, no_output_image: bool = False, no_output_vars: bool = False, 
                 exclude_methods: list[str] = None, exclude_behavior: str = "warn"):
        """Initialize the tool.
        
        Args:
            no_output_image: Whether to suppress image outputs
            no_output_vars: Whether to suppress variable outputs  
            exclude_methods: List of method names to exclude from schema and optionally block at runtime
            exclude_behavior: How to handle calls to excluded methods:
                - "warn": Log warning but allow execution (default)
                - "error": Raise RuntimeError to block execution
                - "silent": Allow execution without warning (schema-only exclusion)
        """
        self.no_output_image = no_output_image
        self.no_output_vars = no_output_vars
        self.exclude_methods = set(exclude_methods or [])
        self.exclude_behavior = exclude_behavior
        self._method_call_stats = {}
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the tool."""
        pass
    
    def get_stats(self) -> dict:
        """Get statistics about tool usage.
        
        Default implementation returns method call counts. Subclasses can override
        to provide additional custom statistics.
        
        Returns:
            Dictionary with tool name, total calls, and per-method call counts
        """
        stats = {
            "tool_name": self.get_name(),
            "total_calls": self._method_call_stats.get('_total_calls', 0),
        }
        # Add per-method stats (excluding the internal _total_calls key)
        for method_name, count in self._method_call_stats.items():
            if not method_name.startswith('_'):
                stats[f"{method_name}_calls"] = count
        return stats
    
    def reset_stats(self) -> None:
        """Reset call statistics to zero."""
        self._method_call_stats.clear()
    
    def get_version(self) -> str:
        """Get the version of the tool."""
        return "1.0.0"
    
    def _check_method_excluded(self, method_name: str) -> None:
        """Check if method is excluded and handle according to exclude_behavior."""
        if method_name not in self.exclude_methods:
            return
            
        if self.exclude_behavior == "error":
            raise RuntimeError(
                f"Method '{method_name}' is excluded from tool '{self.get_name()}' "
                f"and cannot be called. Available methods: {self.get_available_methods()}"
            )
        elif self.exclude_behavior == "warn":
            warnings.warn(
                f"Method '{method_name}' is excluded from tool '{self.get_name()}' schema "
                f"but was called directly. Consider using available methods: {self.get_available_methods()}",
                UserWarning,
                stacklevel=3
            )
        # "silent" behavior: do nothing, just proceed
    
    def get_available_methods(self) -> list[str]:
        """Get list of available (non-excluded) tool methods."""
        import inspect
        
        available = []
        for m_name, fn in inspect.getmembers(self.__class__, inspect.isfunction):
            if not getattr(fn, "_is_tool_method", False):
                continue
            if m_name in self.exclude_methods:
                continue
            # Skip if method requires image output but it's disabled
            if getattr(fn, "_requires_image_output", False) and self.no_output_image:
                continue
            # Skip if method requires var output but it's disabled
            if getattr(fn, "_requires_var_output", False) and self.no_output_vars:
                continue
            available.append(m_name)
        return available

    # ------------------------------------------------------------------
    # Runtime-debug helper (optional for users, handy for troubleshooting)
    # ------------------------------------------------------------------

    def _runtime_info(self) -> dict:  # noqa: D401 – simple diagnostic
        """Return interpreter and environment diagnostics.

        This helper is **not** exposed as a public tool method by default, but
        users can still call it directly through the toolkit for quick
        inspection, e.g. ``toolkit.vlm._runtime_info()``.
        """

        import os, sys, platform  # local import keeps base module lightweight

        return {
            "tool": self.get_name(),
            "python": sys.executable,
            "version": platform.python_version(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "(none)"),
            "cwd": os.getcwd(),
        }

    # ------------------------------------------------------------------
    # Automatic JSON-schema generation *inside* the tool
    # actor so that controller processes do **not** have to import the tool
    # module to discover its API.
    # ------------------------------------------------------------------

    # --- helpers -------------------------------------------------------

    @staticmethod
    def _python_type_to_json(t) -> str:  # noqa: C901 – acceptable complexity
        """Map a Python/typing type annotation to a JSON-schema *type* string."""

        from typing import Union, get_origin, get_args  # local import

        if t is None or t is type(None):
            return "null"
        if t in (str,):
            return "string"
        if t in (int,):
            return "integer"
        if t in (float,):
            return "number"
        if t in (bool,):
            return "boolean"

        origin = get_origin(t)
        if origin is list:
            return "array"
        if origin is dict:
            return "object"

        if origin is Union:  # Optional or general Union
            args = [a for a in get_args(t) if a is not type(None)]
            if len(args) == 1:
                return BaseTool._python_type_to_json(args[0])
            # mixed union – choose generic string
            return "string"

        # default fallback
        return "string"

    @staticmethod
    def _build_type_schema(annotation) -> dict:
        """
        Build a complete JSON schema from a Python type annotation.
        
        Returns a dict with 'type' and potentially 'items' (for arrays) or other schema fields.
        """
        from typing import Union, Tuple, get_origin, get_args  # local import
        
        # Handle None/Optional
        if annotation is None or annotation is type(None):
            return {"type": "null"}
        
        # Handle simple types
        if annotation is str or annotation == str:
            return {"type": "string"}
        if annotation is int or annotation == int:
            return {"type": "integer"}
        if annotation is float or annotation == float:
            return {"type": "number"}
        if annotation is bool or annotation == bool:
            return {"type": "boolean"}
        
        # Handle generic types
        origin = get_origin(annotation)
        args = get_args(annotation)
        
        # Handle List types
        if origin is list:
            schema = {"type": "array"}
            if args:
                # Extract the item type (first argument)
                item_type = args[0]
                # Recursively build schema for items
                item_schema = BaseTool._build_type_schema(item_type)
                schema["items"] = item_schema
            else:
                # List without type args - default to array of anything (use object as fallback)
                schema["items"] = {"type": "string"}
            return schema
        
        # Handle Tuple types (treated as arrays in JSON schema)
        if origin is tuple:
            schema = {"type": "array"}
            if args:
                # For tuples, use the first type and add length constraints
                # This handles Tuple[float, float] -> array of numbers with minItems/maxItems
                item_type = args[0]
                item_schema = BaseTool._build_type_schema(item_type)
                schema["items"] = item_schema
                # Add tuple length constraints
                # Check if all args are the same type (using identity check for built-ins)
                if len(args) > 1:
                    first_arg = args[0]
                    all_same = all(arg is first_arg or arg == first_arg for arg in args)
                    if all_same:
                        schema["minItems"] = len(args)
                        schema["maxItems"] = len(args)
            else:
                schema["items"] = {"type": "string"}
            return schema
        
        # Handle Union/Optional
        if origin is Union:
            # Filter out None for Optional types
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                # Optional[T] -> just use T
                return BaseTool._build_type_schema(non_none_args[0])
            # Multiple types - default to string (could be enhanced)
            return {"type": "string"}
        
        # Handle dict
        if origin is dict:
            return {"type": "object"}
        
        # Default fallback
        return {"type": "string"}

    @staticmethod
    def _build_param_schema(name: str, annotation, has_default: bool) -> tuple[dict, bool]:
        """Return (property_schema, required_flag)."""
        
        # Build complete schema including items for arrays
        prop = BaseTool._build_type_schema(annotation if annotation is not None else str)
        # Minimal description – callers (router) may enrich later if desired.
        prop["description"] = f"Parameter {name}"
        return prop, not has_default

    # --- public helper -------------------------------------------------

    def _process_conditional_docs(self, text: str) -> str:
        """Process conditional documentation blocks based on instance config."""
        from ..doc_utils import process_conditional_docs
        return process_conditional_docs(
            text,
            include_image=not self.no_output_image,
            include_vars=not self.no_output_vars,
            include_text=True,
        )

    def get_openai_schemas(self) -> list[dict]:  # noqa: D401 – simple helper
        """Return OpenAI/Anthropic function-tool schemas for *this* instance.

        The method inspects only those methods explicitly decorated with
        ``@tool_method`` to keep the external surface exactly identical to the
        controller-side logic, but without requiring the module import in the
        controller process.
        """

        import inspect, typing, logging

        # Local logger for schema generation diagnostics
        _logger = logging.getLogger(__name__)

        schemas: list[dict] = []

        for m_name, fn in inspect.getmembers(self.__class__, inspect.isfunction):
            if not getattr(fn, "_is_tool_method", False):
                continue
            
            # Skip excluded methods
            if m_name in self.exclude_methods:
                continue
            
            # Skip if method requires image output but it's disabled
            if getattr(fn, "_requires_image_output", False) and self.no_output_image:
                continue
            
            # Skip if method requires var output but it's disabled
            if getattr(fn, "_requires_var_output", False) and self.no_output_vars:
                continue

            sig = inspect.signature(fn)
            type_hints = typing.get_type_hints(fn)

            # ----------------------------------------------------------
            # Parse parameter descriptions from the docstring (if the
            # optional ``docstring_parser`` package is available). We fall
            # back to the generic description otherwise.
            # ----------------------------------------------------------
            param_docs: dict[str, str] = {}
            doc = inspect.getdoc(fn) or ""
            
            # Process conditional documentation before parsing
            doc = self._process_conditional_docs(doc)

            # ----------------------------------------------------------
            # Make docstring parsing a *hard* requirement for schema
            # generation – missing dependency or parse failure will now
            # raise so that users notice and can fix their environment.
            # ----------------------------------------------------------
            try:
                from docstring_parser import parse as _parse_doc  # type: ignore
            except ImportError as _imp_exc:  # pragma: no cover – explicit crash
                raise ImportError(
                    "docstring_parser is required to extract parameter descriptions. "
                    "Install it with 'pip install docstring-parser'."
                ) from _imp_exc

            try:
                _parsed = _parse_doc(doc)
                for _p in _parsed.params:
                    if _p.arg_name:
                        _desc = (_p.description or "").split("\n")[0].strip()
                        if _desc:
                            param_docs[_p.arg_name] = _desc
            except Exception as _parse_exc:
                # Crash with a clear message – avoids silent behavioural
                # changes that could impact reproducibility.
                raise RuntimeError(
                    f"Failed to parse docstring for {self.get_name()}.{m_name}: {_parse_exc}"
                ) from _parse_exc

            properties: dict = {}
            required: list[str] = []

            for p_name, param in sig.parameters.items():
                if p_name == "self":
                    continue
                
                # Skip parameters starting with _ (internal/reserved parameters like _mock_data)
                if p_name.startswith("_"):
                    continue

                annotation = type_hints.get(p_name, param.annotation)
                prop_schema, is_required = self._build_param_schema(
                    p_name,
                    annotation if annotation is not inspect._empty else str,
                    param.default is not inspect.Parameter.empty,
                )
                # Override placeholder description with the one parsed from
                # the docstring, if available.
                if p_name in param_docs:
                    prop_schema["description"] = param_docs[p_name]

                properties[p_name] = prop_schema
                if is_required:
                    required.append(p_name)

            # Build description from parsed docstring parts: short + long, trimmed
            short_desc = (_parsed.short_description or "").strip()
            long_desc = (_parsed.long_description or "").strip()
            description_parts = [p for p in (short_desc, long_desc) if p]
            if description_parts:
                # Preserve internal newlines between short and long sections; trim outer blanks
                description = "\n".join(description_parts).strip()
            else:
                description = f"Call {m_name} on {self.get_name()}"
            
            schema = {
                "type": "function",
                "function": {
                    "name": f"{self.get_name()}.{m_name}",
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }

            schemas.append(schema)

        return schemas 