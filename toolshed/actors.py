# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Ray actors for running tools in isolated environments.
"""

import ray
import sys
import os
from typing import Any, Dict
# get_tool_registry removed – no direct use in this module
import logging

logger = logging.getLogger(__name__)


@ray.remote(max_concurrency=1, num_cpus=0)
class ToolActor:
    """
    Ray actor that wraps a tool and provides isolated execution.
    
    Each actor runs in its own process/environment and can handle
    one request at a time.
    """
    
    def __init__(
        self,
        tool_name: str,
        tool_args: Dict[str, Any] = None,
        import_path: str = None,
    ):
        """
        Initialize the tool actor.
        
        Args:
            tool_name: Name of the tool to load
            tool_args: Arguments to pass to the tool constructor
            import_path: Explicit import path for external tools (e.g. "mymodule:MyTool")
        """
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.import_path = import_path
        self.tool_instance = None

        logger.info(
            "ToolActor for %s running under Python %s  (exe=%s, CONDA_DEFAULT_ENV=%s)",
            self.tool_name,
            sys.version.split()[0],
            sys.executable,
            os.environ.get("CONDA_DEFAULT_ENV"),
        )
        self._load_tool()
    
    def _load_tool(self):
        """Load the tool instance."""
        try:
            from .registry import get_tool_class

            tool_class = get_tool_class(
                self.tool_name,
                import_path=self.import_path,
                allow_refresh=True,
            )
            self.tool_instance = tool_class(**self.tool_args)
            logger.info(
                "Loaded tool '%s' in actor with args: %s (import_path=%s)",
                self.tool_name,
                self.tool_args,
                self.import_path,
            )

        except Exception as e:
            logger.error("Failed to load tool '%s': %s", self.tool_name, e)
            raise
    
    def call_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call a method on the tool instance.
        
        ObjectRefs in arguments are automatically resolved to their values,
        allowing tool implementations to work with actual objects without
        needing to understand Ray internals.
        
        Args:
            method_name: Name of the method to call
            *args: Positional arguments for the method (ObjectRefs auto-resolved)
            **kwargs: Keyword arguments for the method (ObjectRefs auto-resolved)
            
        Returns:
            Result from the method call
        """
        if not self.tool_instance:
            raise RuntimeError(f"Tool '{self.tool_name}' not initialized")
        
        method = getattr(self.tool_instance, method_name, None)
        if not method:
            raise AttributeError(f"Tool '{self.tool_name}' has no method '{method_name}'")
        
        # Check if method is excluded and handle accordingly
        if hasattr(self.tool_instance, '_check_method_excluded'):
            self.tool_instance._check_method_excluded(method_name)
        
        # Automatically resolve any ObjectRefs to their actual values
        from .ray_utils import resolve_object_refs
        args, kwargs = resolve_object_refs(args, kwargs)
        
        return method(*args, **kwargs)
    
    def get_tool_info(self) -> dict:
        """Get information about the tool instance."""
        if not self.tool_instance:
            return {"error": "Tool not loaded"}
        
        return {
            "tool_name": self.tool_name,
            "tool_class": self.tool_instance.__class__.__name__,
            "available_methods": [method for method in dir(self.tool_instance) 
                                if not method.startswith('_') and callable(getattr(self.tool_instance, method))],
            "status": "ready"
        } 

    # ------------------------------------------------------------------
    # NEW: expose schema generation to the router so it can introspect the
    # tool without importing the module itself.
    # ------------------------------------------------------------------

    def get_openai_schemas(self) -> list:
        """Return the OpenAI function-tool schemas for this tool instance."""

        if self.tool_instance is None:
            raise RuntimeError("Tool instance not initialised")

        try:
            return self.tool_instance.get_openai_schemas()
        except Exception as exc:
            logger.error("Failed to generate schemas in actor for %s – %s", self.tool_name, exc)
            raise 