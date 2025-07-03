# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Verl integration for Toolshed
=============================

This module provides integration between Toolshed and Verl, creating individual
tool schemas and wrappers for each Toolshed method, making them appear as native
tools to LLMs.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Union, get_origin, get_args, Tuple
from importlib import import_module
from uuid import uuid4

from toolshed.tool_client import get_toolkit
# get_tool_registry deprecated – use runtime helpers instead
from toolshed.tools.base import tool_method
from toolshed.registry import get_tool_class, list_available_tools
from toolshed.tool_result import ToolResult

try:
    from docstring_parser import parse as parse_docstring
except ImportError:
    parse_docstring = None

# Verl imports (optional - only needed if using as Verl tools)
try:
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from verl.utils.rollout_trace import rollout_trace_op
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    # Create dummy classes for type hints
    class BaseTool:
        pass
    class OpenAIFunctionToolSchema:
        pass
    def rollout_trace_op(func):
        return func

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TOOLSHED_LOGGING_LEVEL", "WARN"))

# Shared state for Weave initialization
_weave_initialized = False


def _ensure_weave_initialized(config: Optional[dict] = None):
    """Ensure Weave tracing is initialized for this process.
    
    Args:
        config: Optional configuration dict. Can contain:
                - 'project_name' and 'experiment_name' (from VERL trainer config)
                - 'trace_project' and 'trace_experiment' (explicit overrides)
                If not provided, falls back to environment variables.
    """
    global _weave_initialized
    if _weave_initialized:
        return
        
    from verl.utils.rollout_trace import RolloutTraceConfig
    backend = RolloutTraceConfig.get_backend()
    
    project_name = os.getenv("WEAVE_PROJECT", "toolshed_default")
    experiment_name = os.getenv("WEAVE_EXPERIMENT", "rollout_worker")

    if config and isinstance(config, dict):
        # First check for explicit trace configuration
        project_name = config.get('project_name')
        experiment_name = config.get('experiment_name')
        print("found config: ", "project_name: ", project_name, "experiment_name: ", experiment_name)

    try:
        if backend is None:
            # Not initialized yet → initialize with provided names
            import weave
            weave.init(project_name)
            RolloutTraceConfig.init(
                project_name=project_name,
                experiment_name=experiment_name,
                backend="weave",
                token2text=False,
            )
            _weave_initialized = True
            logger.info(
                f"Initialized Weave tracing in worker (project: {project_name}, experiment: {experiment_name})"
            )
        elif backend == "weave":
            # Already initialized with Weave → reinitialize to update configuration
            import weave
            weave.init(project_name)
            RolloutTraceConfig.init(
                project_name=project_name,
                experiment_name=experiment_name,
                backend="weave",
                token2text=False,
            )
            _weave_initialized = True
            logger.info(
                f"Updated Weave tracing config in worker (project: {project_name}, experiment: {experiment_name})"
            )
        else:
            # Some other backend is already configured – leave as-is
            _weave_initialized = True
            logger.info(f"Rollout tracing already initialized with backend '{backend}', leaving as-is")
    except Exception as e:
        logger.warning(f"Failed to initialize or update Weave tracing: {e}")


__all__ = [
    "get_toolshed_tool_wrappers", 
    "ToolshedMethodTool",
    "ToolshedCodeTool",
]

# ------------------------------------------------------------------
# Deprecation stubs – legacy helpers are no longer required now that
# schemas are generated inside each tool actor.  The stubs raise at
# runtime to alert any residual call-sites while preventing accidental
# imports of heavy dependencies (e.g. *docstring_parser*).
# ------------------------------------------------------------------


def _deprecated(*_a, **_kw):  # pragma: no cover – helper
    raise RuntimeError(
        "This helper has been removed. The router's export_openai_schemas() "
        "method should be used instead."
    )


# Map old public symbols → stub
get_toolshed_tool_schemas = _deprecated  # type: ignore
create_verl_tool_configs = _deprecated  # type: ignore
get_verl_tool_schemas = _deprecated  # type: ignore

# Internal parsing helpers likewise removed
_parse_docstring = _deprecated  # type: ignore
_python_type_to_json_schema_type = _deprecated  # type: ignore
_get_method_json_schema = _deprecated  # type: ignore


def get_toolshed_tool_wrappers(
    router_name: str = "toolshed_router",
    namespace: str = "toolshed"
) -> Dict[str, Callable]:
    """Get callable wrappers for all Toolshed tool methods.
    
    Args:
        router_name: Name of the Toolshed router
        namespace: Ray namespace for Toolshed
        
    Returns:
        Dictionary mapping function names to callable wrappers
    """
    from toolshed.tools.base import BaseTool
    
    toolkit = get_toolkit(router_name=router_name, namespace=namespace)
    base_methods = {name for name, _ in inspect.getmembers(BaseTool, inspect.isfunction)}

    try:
        active_tools = toolkit.get_available_tools()
    except Exception as e:
        logger.error(f"Failed to get available tools from Toolshed router: {e}")
        raise  # Fail early - if we can't connect to router, tool calls won't work anyway

    for _tname in active_tools:
        try:
            get_tool_class(_tname, allow_refresh=True)
        except Exception as _exc:  # pragma: no cover – ignore load failures
            logger.debug("Deferred load of tool '%s' failed – %s", _tname, _exc)

    wrappers = {}

    # Only iterate through tools that are actually running in the cluster
    for tool_name in active_tools:
        try:
            tool_cls = get_tool_class(tool_name, allow_refresh=True)
        except Exception as e:
            logger.warning(f"Failed to load tool class '{tool_name}': {e}")
            continue

        tool_proxy = getattr(toolkit, tool_name)
        
        # Determine whether the class defines any decorated methods (cache once per class)
        decorated_methods = {
            name for name, m in inspect.getmembers(tool_cls, inspect.isfunction)
            if getattr(m, "_is_tool_method", False)
        }

        for method_name, method in inspect.getmembers(tool_cls, inspect.isfunction):
            if method_name.startswith("_") or method_name in base_methods:
                continue
            if decorated_methods and method_name not in decorated_methods:
                continue
            if decorated_methods and not getattr(method, "_is_tool_method", False):
                continue
                
            function_name = f"{tool_name}.{method_name}"
            
            # Capture current values via default arguments to avoid late-binding issues
            # (omg this was soo hard to debug)
            def _make_wrapper(_proxy=tool_proxy, _meth=method_name):
                def _wrapper(**kwargs):  # noqa: D401 – simple passthrough
                    return getattr(_proxy, _meth)(**kwargs)
                return _wrapper

            wrappers[function_name] = _make_wrapper()
    
    logger.warning(f"Created {len(wrappers)} function wrappers: {list(wrappers.keys())}")
    return wrappers


# Verl compatibility functions (if Verl is available)
def _try_import_verl():
    """Try to import Verl schemas, return None if not available."""
    try:
        from verl.tools.schemas import OpenAIFunctionToolSchema
        return OpenAIFunctionToolSchema
    except ImportError:
        return None


# Verl Tool Wrapper Classes (only available if Verl is installed)
class ToolshedMethodTool(BaseTool):
    """Wrapper for individual Toolshed tool methods."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        if not VERL_AVAILABLE:
            raise ImportError("Verl is not available. Install Verl to use ToolshedMethodTool.")
        
        super().__init__(config, tool_schema)
        print("ToolshedMethodTool: ", "config: ", config)

        self._instance_dict = {}
        self._toolkit = None
        self._function_wrapper = None
    
    def _get_toolkit(self):
        """Lazy initialization of toolkit connection."""
        if self._toolkit is None:
            try:
                self._toolkit = get_toolkit(
                    router_name=self.config.get("router_name", "toolshed_router"),
                    namespace=self.config.get("namespace", "toolshed")
                )
            except Exception as e:
                logger.error(f"Failed to connect to Toolshed: {e}")
                raise
        return self._toolkit
    
    def _get_function_wrapper(self):
        """Get the function wrapper for this tool."""
        if self._function_wrapper is None:
            try:
                wrappers = get_toolshed_tool_wrappers(
                    router_name=self.config.get("router_name", "toolshed_router"),
                    namespace=self.config.get("namespace", "toolshed")
                )
                function_name = self.config.get("function_name")
                if function_name not in wrappers:
                    raise ValueError(f"Function {function_name} not found in Toolshed")
                self._function_wrapper = wrappers[function_name]
            except Exception as e:
                logger.error(f"Failed to get function wrapper: {e}")
                raise
        return self._function_wrapper

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Initialize instance state
        self._instance_dict[instance_id] = {
            "calls": 0,
            "last_result": None,
            "total_reward": 0.0
        }
        
        # Ensure toolkit connection
        self._get_toolkit()
        self._get_function_wrapper()
        
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str|dict, float, dict]:
        """Execute the Toolshed tool method.

        Supports structured returns for multi-modal and variables in addition to plain text.
        Tool methods should return a ToolResult, which will be converted to a dict:
            {
              "text": str,
              "image": list[Any] | Any,
              "video": list[Any] | Any,
              "variables": [ {"name": str, "value": Any}, ... ]
            }
        """

        # Ensure Weave is initialized (will be a no-op if already done in create)
        _ensure_weave_initialized(self.config)
        
        if instance_id not in self._instance_dict:
            raise ValueError(f"Instance {instance_id} not found")
        
        try:
            # Get the function wrapper
            func = self._get_function_wrapper()
            
            # Execute the function (run in thread pool since it's synchronous)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: func(**parameters))

            # Check if result is an error ToolResult
            if isinstance(result, ToolResult) and result.is_error:
                raise RuntimeError(result.text or "Unknown Toolshed error")
            
            # Update instance state
            self._instance_dict[instance_id]["calls"] += 1
            self._instance_dict[instance_id]["last_result"] = result
            
            # All tools MUST return ToolResult
            if not isinstance(result, ToolResult):
                raise TypeError(
                    f"Tool '{self.config.get('function_name', 'unknown')}' must return ToolResult, "
                    f"got {type(result).__name__} instead. "
                    f"Please update the tool to use: return ToolResult(value, text=..., ...)"
                )
            
            # Import ray once at the beginning
            import ray
            ray_initialized = ray.is_initialized()
            
            # Extract structured data from ToolResult
            normalized = result.to_dict()
            
            # Apply ray.put to images, videos, and variables for efficiency
            if ray_initialized:
                if "image" in normalized:
                    normalized["image"] = [ray.put(img) for img in normalized["image"]]
                if "video" in normalized:
                    normalized["video"] = [ray.put(vid) for vid in normalized["video"]]
                if "variables" in normalized:
                    normalized["variables"] = [
                        {"name": v["name"], "value": ray.put(v["value"])}
                        for v in normalized["variables"]
                    ]
            
            # Calculate step reward (successful execution)
            step_reward = 0.1  # Small positive reward for successful tool use
            
            metrics = {
                "toolshed_calls": self._instance_dict[instance_id]["calls"],
                "function_name": self.config.get("function_name", "unknown")
            }
            print("ToolshedMethodTool: ", "step_reward:", step_reward, "metrics:", metrics)
            return normalized, step_reward, metrics
            
        except Exception as e:
            logger.error(f"Error executing Toolshed function: {e}")
            # Do not penalize with a negative reward; return zero reward instead
            return f"Error: {str(e)}", 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate cumulative reward for this tool instance."""
        if instance_id not in self._instance_dict:
            return 0.0
        
        # Basic reward based on successful usage
        calls = self._instance_dict[instance_id]["calls"]
        return min(calls * 0.1, 1.0)  # Cap at 1.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class ToolshedCodeTool(BaseTool):
    """Wrapper for Toolshed code execution tool."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        if not VERL_AVAILABLE:
            raise ImportError("Verl is not available. Install Verl to use ToolshedCodeTool.")
        
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self._toolkit = None
        
    def _get_toolkit(self):
        """Lazy initialization of toolkit connection."""
        if self._toolkit is None:
            try:
                self._toolkit = get_toolkit(
                    router_name=self.config.get("router_name", "toolshed_router"),
                    namespace=self.config.get("namespace", "toolshed")
                )
            except Exception as e:
                logger.error(f"Failed to connect to Toolshed: {e}")
                raise
        return self._toolkit

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Initialize instance state
        self._instance_dict[instance_id] = {
            "executions": 0,
            "successful_executions": 0,
            "last_result": None
        }
        
        # Ensure toolkit connection
        self._get_toolkit()
        
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute Python code using Toolshed."""
        
        # Ensure Weave is initialized (will be a no-op if already done in create)
        _ensure_weave_initialized(self.config)
        
        if instance_id not in self._instance_dict:
            raise ValueError(f"Instance {instance_id} not found")
        
        try:
            toolkit = self._get_toolkit()
            code = parameters.get("code", "")
            
            if not code:
                return "Error: No code provided", 0.0, {"error": "no_code"}
            
            # Execute code using Toolshed's code executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: toolkit.code_executor.exec(code))
            
            # Update instance state
            self._instance_dict[instance_id]["executions"] += 1
            
            # Parse result
            if isinstance(result, (list, tuple)) and len(result) >= 3:
                actual_result, stdout, stderr = result[:3]
                
                # Check if execution was successful
                success = stderr == "" or stderr is None
                if success:
                    self._instance_dict[instance_id]["successful_executions"] += 1
                
                self._instance_dict[instance_id]["last_result"] = actual_result
                
                # Format response
                response = f"Execution result: {actual_result}"
                if stdout:
                    response += f"\nOutput: {stdout}"
                if stderr:
                    response += f"\nError: {stderr}"
                
                # Calculate step reward
                # Remove negative penalty for failed executions
                step_reward = 0.2 if success else 0.0
                
                metrics = {
                    "code_executions": self._instance_dict[instance_id]["executions"],
                    "successful_executions": self._instance_dict[instance_id]["successful_executions"],
                    "success": success
                }

                print("ToolshedCodeTool: \n", "response: ", response, "\n", "step_reward: ", step_reward, "\n", "metrics: ", metrics)
                
                return response, step_reward, metrics
            else:
                # Return zero reward for unexpected format instead of negative
                return f"Unexpected result format: {result}", 0.0, {"error": "format_error"}
                
        except Exception as e:
            # No negative penalty on execution error
            return f"Execution error: {str(e)}", 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate cumulative reward for this tool instance."""
        if instance_id not in self._instance_dict:
            return 0.0
        
        # Reward based on successful code executions
        successful = self._instance_dict[instance_id]["successful_executions"]
        total = self._instance_dict[instance_id]["executions"]
        
        if total == 0:
            return 0.0
        
        success_rate = successful / total
        return success_rate * min(total * 0.2, 2.0)  # Cap at 2.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]