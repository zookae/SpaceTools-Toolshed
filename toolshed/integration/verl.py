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
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
    from verl.utils.rollout_trace import rollout_trace_op
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    # Create dummy classes for type hints
    class BaseTool:
        pass
    class OpenAIFunctionToolSchema:
        pass
    class ToolResponse:
        def __init__(self, **kwargs):
            self.text = kwargs.get("text")
            self.image = kwargs.get("image")
            self.video = kwargs.get("video")
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
        """Get the function wrapper for this tool.

        Builds the wrapper directly from ``function_name`` (e.g.
        ``"roborefer.detect_one"``) and the toolkit proxy, without importing
        the tool's Python class locally.  This avoids import failures when the
        RL worker environment doesn't have the tool's heavy dependencies
        (e.g. ``llava``, ``cv2``, ``depth_pro``).
        """
        if self._function_wrapper is None:
            try:
                function_name = self.config.get("function_name")
                if not function_name or "." not in function_name:
                    raise ValueError(
                        f"Invalid function_name '{function_name}': "
                        "expected 'tool_name.method_name'"
                    )
                tool_name, method_name = function_name.split(".", 1)
                toolkit = self._get_toolkit()
                tool_proxy = getattr(toolkit, tool_name)

                def _wrapper(_proxy=tool_proxy, _meth=method_name, **kwargs):
                    return getattr(_proxy, _meth)(**kwargs)

                self._function_wrapper = _wrapper
            except Exception as e:
                logger.error(f"Failed to get function wrapper: {e}")
                raise
        return self._function_wrapper

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
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
        
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, dict]:
        """Execute the Toolshed tool method.

        Supports structured returns for multi-modal content.
        Tool methods should return a ToolResult, which will be converted to a ToolResponse
        with text, image, and video fields.
        """

        # Ensure Weave is initialized (will be a no-op if already done in create)
        _ensure_weave_initialized(self.config)
        
        if instance_id not in self._instance_dict:
            raise ValueError(f"Instance {instance_id} not found")
        
        try:
            # Get the function wrapper
            func = self._get_function_wrapper()

            # Resolve image_index → PIL Image.  The schema uses
            # use_image_by_index=True so the model must pass
            # {"image_index": N}.  Any other format (e.g. "image": "<string>")
            # is a schema violation and should fail — we must not silently
            # fix malformed calls during RL or the model won't learn the
            # correct format.
            if "image_index" in parameters:
                agent_data = kwargs.get("agent_data")
                images = getattr(agent_data, "image_data", None) or []
                if not isinstance(images, list):
                    images = list(images)
                idx = parameters.pop("image_index")
                if isinstance(idx, int) and 0 <= idx < len(images):
                    parameters["image"] = images[idx]
                else:
                    raise ValueError(
                        f"image_index={idx} but only {len(images)} "
                        "images available in conversation"
                    )

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
            
            # Convert to verl ToolResponse.  Only pass image/video when they
            # are actual lists — ToolResponse validates that image/video must
            # be lists and rejects None.  Tools with no_output_image=True
            # return None for these fields.
            tool_response_kwargs = {"text": normalized.get("text")}
            if normalized.get("image"):
                tool_response_kwargs["image"] = normalized["image"]
            if normalized.get("video"):
                tool_response_kwargs["video"] = normalized["video"]
            tool_response = ToolResponse(**tool_response_kwargs)
            
            # Calculate step reward (successful execution)
            step_reward = 0.1  # Small positive reward for successful tool use
            
            metrics = {
                "toolshed_calls": self._instance_dict[instance_id]["calls"],
                "function_name": self.config.get("function_name", "unknown")
            }

            # Include variables so the agent loop can store them for
            # cross-tool $variable resolution.
            if normalized.get("variables"):
                metrics["variables"] = normalized["variables"]

            logger.info(f"ToolshedMethodTool: step_reward={step_reward}, metrics={metrics}")
            return tool_response, step_reward, metrics
            
        except Exception as e:
            logger.error(f"Error executing Toolshed function: {e}")
            # Do not penalize with a negative reward; return zero reward instead
            return ToolResponse(text=f"Error: {str(e)}"), 0.0, {"error": str(e)}

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

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
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
        
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, dict]:
        """Execute Python code using Toolshed."""
        
        # Ensure Weave is initialized (will be a no-op if already done in create)
        _ensure_weave_initialized(self.config)
        
        if instance_id not in self._instance_dict:
            raise ValueError(f"Instance {instance_id} not found")
        
        try:
            toolkit = self._get_toolkit()
            code = parameters.get("code", "")
            
            if not code:
                return ToolResponse(text="Error: No code provided"), 0.0, {"error": "no_code"}
            
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
                step_reward = 0.2 if success else 0.0
                
                metrics = {
                    "code_executions": self._instance_dict[instance_id]["executions"],
                    "successful_executions": self._instance_dict[instance_id]["successful_executions"],
                    "success": success
                }

                logger.info(f"ToolshedCodeTool: response={response}, step_reward={step_reward}")
                
                return ToolResponse(text=response), step_reward, metrics
            else:
                return ToolResponse(text=f"Unexpected result format: {result}"), 0.0, {"error": "format_error"}
                
        except Exception as e:
            return ToolResponse(text=f"Execution error: {str(e)}"), 0.0, {"error": str(e)}

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