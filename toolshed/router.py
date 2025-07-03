# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Internal router actor for the shed3d toolkit.

The ToolRouterActor is the centralized coordinator that manages tool actors
and handles load balancing.
"""

import ray
from collections import deque
import threading
import time
from typing import Dict, Any, Optional, List, Deque, Tuple
from .actors import ToolActor
from .tool_result import ToolResult
# get_tool_registry removed – not required here
import logging

logger = logging.getLogger(__name__)


@ray.remote(max_concurrency=512, num_cpus=0)  # Allow many concurrent calls without reserving CPUs
class ToolRouterActor:
    """
    Internal Ray actor that routes tool calls to appropriate tool actors.
    
    This actor manages tool actors and handles load balancing by tracking
    which actors are available vs busy. Multiple Ray workers can access this 
    actor concurrently, and it will queue requests when all actors are busy.
    
    Users don't interact with this directly - they use ToolkitClient instead.
    """
    
    def __init__(self):
        """Initialize the tool router actor."""
        # Core data structures
        self.actors: Dict[str, List[ray.actor.ActorHandle]] = {}  # All actors for each tool
        # Queue of idle actors ready for work
        self.available_actors: Dict[str, Deque[ray.actor.ActorHandle]] = {}

        # Actors that have been spawned but are *still* running the tool's
        # heavy-weight start-up logic (e.g. checkpoint load). These actors
        # will be promoted to ``available_actors`` once a lightweight
        # readiness probe succeeds.
        self.initializing_actors: Dict[str, Deque[ray.actor.ActorHandle]] = {}

        self.busy_actors: Dict[str, Dict[ray.actor.ActorHandle, Dict[str, Any]]] = {}  # Busy actors and their state
        
        # Thread safety
        self.actor_locks: Dict[str, threading.Lock] = {}  # Per-tool locks
        self.actor_conditions: Dict[str, threading.Condition] = {}  # For waiting on available actors
        
        self.tool_configs: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        # tool_name -> list[schemas]
        self._tool_schemas: Dict[str, List[dict]] = {}
        # Map from schema name (what LLM sees) to config key (how tool is registered)
        # This allows tools like "mock_robot" to present as "robot" in schemas
        self._schema_name_to_config_key: Dict[str, str] = {}
        # Optional placement group for scheduling constraints
        self.placement_group: Optional[Any] = None
    
    def initialize(self, tool_configs: Dict[str, Dict[str, Any]], placement_group: Optional[Any] = None):
        """
        Initialize the router with tool configurations.
        
        Args:
            tool_configs: Dictionary mapping tool names to their configurations.
                         Each config should include:
                         - 'num_actors': Number of actors to create
                         - 'conda_env': Conda environment name (optional)
                         - 'resources': Resource requirements (optional)
                         - 'args': Arguments to pass to tool constructor (optional)
            placement_group: Optional Ray placement group to constrain where tool actors are scheduled.
        """
        self.tool_configs = tool_configs
        self.placement_group = placement_group
        
        # ------------------------------------------------------------------
        # 1. Spawn actors for *all* tools first, marking them as
        #    "initializing". At the same time collect one readiness probe
        #    (Ray ObjectRef) per actor so that we can poll their completion
        #    in one central loop.
        # ------------------------------------------------------------------
        pending_readiness: List[Tuple[str, ray.actor.ActorHandle, ray.ObjectRef, bool]] = []

        for tool_name, config in tool_configs.items():
            # ``_create_tool_actors`` now returns the list of readiness refs
            # together with bookkeeping info.
            pending_readiness.extend(self._create_tool_actors(tool_name, config))

        # ------------------------------------------------------------------
        # 2. Poll until *all* actors have finished their start-up work. The
        #    router actor has high max_concurrency so this loop will not
        #    starve other RPC calls (``ray.wait`` returns frequently).
        # ------------------------------------------------------------------
        while pending_readiness:
            # Wait for the *next* actor to become ready.
            ready_refs, _ = ray.wait([item[2] for item in pending_readiness], num_returns=1, timeout=0.2)
            if not ready_refs:
                # Nothing ready yet – loop again.
                continue

            # Promote every actor whose readiness ref completed in this tick.
            for ready_ref in ready_refs:
                # Find associated tuple
                idx = next(i for i, t in enumerate(pending_readiness) if t[2] == ready_ref)
                tool_name, actor_handle, ref, has_schema = pending_readiness.pop(idx)

                # Fetch result (ensure exceptions propagate early)
                result = ray.get(ref)

                # Cache schemas only once – they come from the *first* actor.
                if has_schema:
                    self._tool_schemas[tool_name] = result
                    
                    # Extract schema name from the first schema and build mapping
                    # This allows tools to present themselves with a different name than their config key
                    # (e.g., "mock_robot" config key presenting as "robot" in schemas)
                    if result and len(result) > 0:
                        # Get the function name from first schema (format: "toolname.method")
                        first_func_name = result[0].get("function", {}).get("name", "")
                        if "." in first_func_name:
                            schema_name = first_func_name.split(".", 1)[0]
                            # Store bidirectional mapping: schema_name -> config_key
                            self._schema_name_to_config_key[schema_name] = tool_name
                            logger.info(
                                "Tool '%s' presents as '%s' in schemas%s",
                                tool_name,
                                schema_name,
                                "" if schema_name == tool_name else f" (aliased)"
                            )

                # Move actor from initializing -> available
                with self.actor_conditions[tool_name]:
                    if actor_handle in self.initializing_actors[tool_name]:
                        self.initializing_actors[tool_name].remove(actor_handle)
                        self.available_actors[tool_name].append(actor_handle)
                        self.actor_conditions[tool_name].notify()

        self._initialized = True
        logger.info(
            "Initialized shed3d router with tools: %s (total actors = %d)",
            list(tool_configs.keys()),
            sum(len(v) for v in self.actors.values()),
        )
    
    def _create_tool_actors(self, tool_name: str, config: Dict[str, Any]):
        """Create Ray actors for a specific tool."""
        num_actors = config.get('num_actors', 1)
        conda_env = config.get('conda_env')
        resources = config.get('resources', {})

        # Build a single options dictionary – avoid chaining .options on ActorOptionWrapper
        actor_options: Dict[str, Any] = {}
        if conda_env:
            actor_options["runtime_env"] = {"conda": conda_env}
        actor_options.update(resources)  # num_cpus / num_gpus / custom resources

        # Add placement group scheduling strategy if provided
        if self.placement_group is not None:
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            # Place all actors in the first bundle (index 0) of the placement group
            actor_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=self.placement_group,
                placement_group_bundle_index=0
            )

        # Create the Ray actor class with combined options (if any)
        actor_class = ToolActor.options(**actor_options) if actor_options else ToolActor

        # Create multiple actors for this tool
        actors = []
        tool_args = config.get('args', {})  # Extract tool-specific arguments
        import_path = config.get('import_path')  # For external tools
        for _ in range(num_actors):
            actor = actor_class.remote(tool_name, tool_args, import_path)
            actors.append(actor)

        # --------------------------------------------------------------
        # Initialize tracking structures – all actors start in the
        # *initializing* bucket. They will be promoted to *available* once
        # their readiness probe completes.
        # --------------------------------------------------------------
        self.actors[tool_name] = actors
        self.initializing_actors[tool_name] = deque(actors)
        self.available_actors[tool_name] = deque()
        self.busy_actors[tool_name] = {}
        self.actor_locks[tool_name] = threading.Lock()
        self.actor_conditions[tool_name] = threading.Condition(self.actor_locks[tool_name])

        # --------------------------------------------------------------
        # Prepare readiness probes
        #
        # * First actor – call ``get_openai_schemas`` which both confirms
        #   readiness *and* returns the schema list we need to cache.
        # * Remaining actors – a lightweight ``get_tool_info`` is enough to
        #   signal completion.
        #
        # Return a list of tuples to the caller so that the central polling
        # loop can manage promotion.
        # --------------------------------------------------------------
        pending: List[Tuple[str, ray.actor.ActorHandle, ray.ObjectRef, bool]] = []

        # First actor → schema call
        schema_ref = actors[0].get_openai_schemas.remote()
        pending.append((tool_name, actors[0], schema_ref, True))

        # Remaining actors → simple info call
        for actor in actors[1:]:
            info_ref = actor.get_tool_info.remote()
            pending.append((tool_name, actor, info_ref, False))

        logger.info("Created %d actors for tool '%s' (awaiting readiness)", num_actors, tool_name)

        return pending
    
    def _get_available_actor(self, tool_name: str, method_name: str, timeout: float = None) -> ray.actor.ActorHandle:
        """
        Get an available actor for the tool, waiting if necessary.
        
        Args:
            tool_name: Name of the tool
            method_name: Method being called (for state tracking)
            timeout: Maximum time to wait for an available actor
            
        Returns:
            An available actor handle
            
        Raises:
            TimeoutError: If timeout is reached while waiting
        """
        start_time = time.time()
        
        with self.actor_conditions[tool_name]:
            while True:
                if self.available_actors[tool_name]:
                    # Get an available actor
                    actor = self.available_actors[tool_name].popleft()
                    
                    # Mark it as busy
                    self.busy_actors[tool_name][actor] = {
                        'method': method_name,
                        'start_time': time.time()
                    }
                    
                    return actor
                
                # No actors available, wait
                if timeout is not None:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        raise TimeoutError(f"No available actors for tool '{tool_name}' after {timeout}s")
                    if not self.actor_conditions[tool_name].wait(timeout=remaining):
                        raise TimeoutError(f"No available actors for tool '{tool_name}' after {timeout}s")
                else:
                    self.actor_conditions[tool_name].wait()
    
    def _return_actor_to_pool(self, tool_name: str, actor: ray.actor.ActorHandle):
        """Return an actor to the available pool."""
        with self.actor_conditions[tool_name]:
            # Remove from busy
            if actor in self.busy_actors[tool_name]:
                del self.busy_actors[tool_name][actor]
            
            # Add to available
            self.available_actors[tool_name].append(actor)
            
            # Notify waiting threads
            self.actor_conditions[tool_name].notify()
    
    def _sanitize_error_message(self, error: Exception) -> str:
        """
        Extract clean error message from Ray exception, removing system information.
        
        Args:
            error: Exception from Ray actor call
            
        Returns:
            Sanitized error message with just the exception type and message
        """
        import re
        
        error_str = str(error)
        
        # Strip ANSI escape codes
        error_str = re.sub(r'\x1b\[[0-9;]*m', '', error_str)
        
        # Ray exceptions have format:
        # ray::ToolActor.call_method() (pid=..., ip=..., actor_id=..., repr=...)
        #   File "...", line ..., in ...
        #     ...
        # ExceptionType: actual message
        
        # Extract just the final exception line (ExceptionType: message)
        lines = error_str.strip().split('\n')
        
        # Find the last line that looks like an exception (contains ':')
        # and doesn't start with 'File' or whitespace
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('File') and ':' in stripped:
                # This is likely the exception line
                return stripped
        
        # Fallback: return the last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        return str(error)
    
    def list_active_tools(self) -> List[str]:
        """Return the set of tool names that currently have actors."""
        return list(self.actors.keys())
    
    def call_tool(self, tool_name: str, method_name: str, *args, **kwargs) -> Any:
        """
        Route a tool call to an available actor.
        
        Args:
            tool_name: Name of the tool to call (schema name from LLM)
            method_name: Name of the method to call on the tool
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            Result from the tool method
            
        Raises:
            ValueError: If tool is not available
            RuntimeError: If router is not initialized
            TimeoutError: If no actor becomes available within timeout
        """
        if not self._initialized:
            raise RuntimeError("Router not initialized. Call initialize() first.")
        
        # Map schema name (what LLM uses) to config key (how tool is stored)
        # For backward compatibility: if not in mapping, assume schema_name == config_key
        config_key = self._schema_name_to_config_key.get(tool_name, tool_name)
        
        if config_key not in self.actors:
            # Provide helpful error message showing both schema name and available tools
            available_schema_names = list(self._schema_name_to_config_key.keys())
            available_config_keys = list(self.actors.keys())
            raise ValueError(
                f"Tool '{tool_name}' not available. "
                f"Available schema names: {available_schema_names}, "
                f"Available config keys: {available_config_keys}"
            )
        
        # Get timeout from config (use config_key to look up the actual tool config)
        timeout = self.tool_configs.get(config_key, {}).get("timeout", 30)
        
        # Get an available actor (may wait) - use config_key for actor pool lookup
        # TimeoutError here is an infrastructure issue (no actors available),
        # not a tool execution error. We re-raise with a clear message rather than
        # returning ToolResult, since this can't be fixed by the LLM.
        try:
            actor = self._get_available_actor(config_key, method_name, timeout)
        except TimeoutError as e:
            raise RuntimeError(
                f"Tool Router Error: No available actors for tool '{tool_name}' "
                f"(config: {config_key}) after {timeout}s timeout. "
                f"This may indicate actors have died, are overloaded, or misconfigured."
            ) from e
        
        try:
            # Make the call
            ref = actor.call_method.remote(method_name, *args, **kwargs)
            result = ray.get(ref, timeout=timeout)
            return result
        except TimeoutError:
            err_msg = f"Timeout (>{timeout}s) calling {tool_name}.{method_name} with args={args} kwargs={kwargs}"
            logger.error(err_msg)
            return ToolResult(None, text=err_msg, is_error=True)
        except Exception as e:
            logger.error(f"Error executing {tool_name}.{method_name}: {e}")
            sanitized_error = self._sanitize_error_message(e)
            return ToolResult(None, text=sanitized_error, is_error=True)
        finally:
            # Always return the actor to the pool (use config_key)
            self._return_actor_to_pool(config_key, actor)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools (returns schema names as seen by LLM).
        
        Returns schema names rather than config keys for consistency with how
        the LLM sees tools. For example, if mock_robot is configured but presents
        as "robot" in schemas, this will return ["robot"].
        """
        return list(self._schema_name_to_config_key.keys())
    
    def get_tool_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all tools."""
        status = {}
        for tool_name in self.actors.keys():
            with self.actor_locks[tool_name]:
                status[tool_name] = {
                    'total_actors': len(self.actors[tool_name]),
                    'available_actors': len(self.available_actors[tool_name]),
                    'busy_actors': len(self.busy_actors[tool_name]),
                    'tool_config': self.tool_configs.get(tool_name, {}),
                    'initialized': True
                }
        return status
    
    def get_per_actor_stats(self, tool_name: str) -> List[Dict[str, Any]]:
        """
        Get individual statistics from each actor of a tool.
        
        Args:
            tool_name: Name of the tool (schema name from LLM)
            
        Returns:
            List of statistics dictionaries, one per actor
            
        Raises:
            ValueError: If tool is not available
        """
        # Map schema name to config key
        config_key = self._schema_name_to_config_key.get(tool_name, tool_name)
        
        if config_key not in self.actors:
            available_schema_names = list(self._schema_name_to_config_key.keys())
            raise ValueError(
                f"Tool '{tool_name}' not available. "
                f"Available tools: {available_schema_names}"
            )
        
        # Collect stats from all actors
        stats_refs = [actor.call_method.remote("get_stats") for actor in self.actors[config_key]]
        all_stats = ray.get(stats_refs)
        
        # Add actor index to each stat dict for identification
        for i, stats in enumerate(all_stats):
            stats['actor_index'] = i
        
        return all_stats
    
    def get_actors(self) -> Dict[str, List[ray.actor.ActorHandle]]:
        """Get actor handles for all tools. Used by dashboard."""
        return self.actors
    
    def get_actor_states(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get detailed state of all actors for dashboard visibility.
        
        Returns:
            Dictionary mapping tool names to lists of actor states
        """
        states = {}
        
        for tool_name in self.actors.keys():
            tool_states = []
            
            with self.actor_locks[tool_name]:
                # Check each actor
                for i, actor in enumerate(self.actors[tool_name]):
                    actor_id = f"{tool_name}_{i}"
                    # Always reset per-iteration
                    current_method = None
                    
                    if actor in self.available_actors[tool_name]:
                        state = 'available'
                    elif actor in self.initializing_actors[tool_name]:
                        state = 'initializing'
                    elif actor in self.busy_actors[tool_name]:
                        state = 'working'
                        info = self.busy_actors[tool_name][actor]
                        current_method = info.get('method')
                    else:
                        state = 'unknown'
                    
                    tool_states.append({
                        'id': actor_id,
                        'name': actor_id,
                        'tool': tool_name,
                        'state': state,
                        'current_method': current_method,
                        'actor_handle': actor
                    })
            
            states[tool_name] = tool_states

        # ------------------------------------------------------------------
        # Ensure every tool that *will* be created appears in the list, even
        # if its actors haven't been spawned yet (happens during the early
        # phase of ``initialize`` when dashboard may query us).
        # ------------------------------------------------------------------
        for tool_name in self.tool_configs.keys():
            if tool_name not in states:
                states[tool_name] = [
                    {
                        'id': f'{tool_name}_0',
                        'name': f'{tool_name}_0',
                        'tool': tool_name,
                        'state': 'offline',
                        'current_method': None,
                        'actor_handle': None,
                    }
                ]

        return states
    
    def shutdown(self):
        """Shutdown the router and cleanup resources."""
        if self._initialized:
            # Kill all actors
            for tool_name, actors in self.actors.items():
                for actor in actors:
                    ray.kill(actor)
            
            self.actors.clear()
            self.available_actors.clear()
            self.busy_actors.clear()
            self._initialized = False
            logger.info("Shutdown shed3d router")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    # ------------------------------------------------------------------
    # One-shot schema export
    # ------------------------------------------------------------------
    def export_openai_schemas(
        self,
        timeout: int = 30,
        include_code_executor: bool = False,
        use_image_by_index: bool = False,
    ):
        """Return OpenAI/Anthropic function-tool schemas for *currently
        running* tools.

        This re-computes the full schema list and then filters it against the
        router's live actors in order to avoid the self-call dead-lock
        that would occur if we tried to discover the running tools through a
        secondary Toolkit client from inside this very actor.

        Args:
            timeout: Default per-call timeout stored in the generated configs.
            include_code_executor: Whether to include the special execute_python tool.
            use_image_by_index: If True, replace heavy "image" params with lightweight
                integer "image_index" so the model points to images already in the
                conversation.
        """
        configs: List[dict] = []

        for tool_name, schemas in self._tool_schemas.items():
            for schema in schemas:
                # ------------------------------------------------------
                # Optional rewrite: expose lightweight "image_index" arg
                # instead of heavy binary "image".  This is purely a
                # schema-level change – the runtime still expects the
                # original parameter.  Roll-out code (on verl side) will 
                # inject the real image just before the remote call.
                # ------------------------------------------------------
                # ------------------------------------------------------
                # Optional rewrite for image inputs
                # - Replace heavyweight binary "image" with lightweight index reference if requested
                # ------------------------------------------------------
                if use_image_by_index:
                    try:
                        _params = schema["function"]["parameters"]
                        _props = _params.get("properties", {})
                        # whether to remove heavy param only if present
                        had_heavy_image = False
                        if "image" in _props:
                            had_heavy_image = True

                            if "robot" in schema["function"]["name"]:
                                print(f"DEBUG: schema: \n\n\n{schema}\n\n\n\n")
                                print(f"DEBUG: Had heavy image: {had_heavy_image}")

                            _props.pop("image", None)
                            _props["image_index"] = {
                                "type": "integer",
                                "description": (
                                    "0-based index of an image already present in the conversation. "
                                    "Use this to reference an inline image by its order (0 = first, 1 = second, ...)."
                                ),
                            }

                        # Update required list: drop hard requirement on "image" and require the alternative
                        if had_heavy_image:
                            _req = _params.get("required", [])
                            if "image" in _req:
                                _req = [r for r in _req if r != "image"]
                                _req.append("image_index")
                                _params["required"] = _req
                    except Exception:
                        # Schema shape unexpected – skip rewrite silently.
                        print(f"Schema shape unexpected – skip rewrite silently. {schema}")

                function_name = schema["function"]["name"]

                configs.append(
                    {
                        "class_name": "toolshed.integration.verl.ToolshedMethodTool",
                        "config": {
                            "type": "native",
                            "router_name": "toolshed_router",
                            "namespace": "toolshed",
                            "function_name": function_name,
                            "timeout": timeout,
                        },
                        "tool_schema": schema,
                    }
                )

        # ------------------------------------------------------------------
        # Include code executor tool if requested
        # ------------------------------------------------------------------
        if include_code_executor:
            configs.append(
                {
                    "class_name": "toolshed.integration.verl.ToolshedCodeTool",
                    "config": {"timeout": timeout},
                    "tool_schema": {
                        "type": "function",
                        "function": {
                            "name": "execute_python",
                            "description": (
                                "Execute a python code snippet. "
                                "The last line of output will be captured and returned. "
                                "You need to print the final result if you want to see it. "
                                "Don't create files unless it's absolutely necessary."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "code": {
                                        "type": "string",
                                        "description": "The python code to execute.",
                                    }
                                },
                                "required": ["code"],
                            },
                        },
                    },
                }
            )

        return configs 

    def is_ready(self) -> bool:
        """Return True when the router has finished \*initialize\* and at least one tool is available.

        A router is considered *ready* once `initialize()` has run to completion
        **and** it has registered at least one tool actor.  External helpers
        (e.g. the YAML generator) can poll this flag instead of guessing how
        long to sleep.
        """
        return self._initialized