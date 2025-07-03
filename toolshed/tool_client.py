# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
User-facing client for the shed3d toolkit.

This module provides the main API that users interact with, including:
- ToolkitClient: Main class for pythonic tool access
- ToolProxy: Enables toolkit.tool.method() syntax
- start_toolkit(): Convenience function to start the distributed toolkit
- get_toolkit(): Convenience function to get a client
"""

import ray
from typing import Dict, Any, Optional, List, Union
from .router import ToolRouterActor
from .documentation import generate_tool_documentation
import logging

logger = logging.getLogger(__name__)


class ToolProxy:
    """
    Proxy class for pythonic tool access with automatic Ray optimization.
    
    Enables syntax like: toolkit.greeting.greet("TestUser")
    Each ToolProxy represents one tool and dynamically creates method calls.
    
    Large objects (images, numpy arrays, etc.) are automatically placed in
    the Ray object store to avoid double serialization when passing through
    the router to tool actors.
    """
    
    def __init__(self, router_actor: ray.actor.ActorHandle, tool_name: str, 
                 auto_put_threshold: int = 100_000):
        """
        Initialize tool proxy.
        
        Args:
            router_actor: Handle to the ToolRouterActor
            tool_name: Name of the tool this proxy represents
            auto_put_threshold: Size threshold (bytes) for automatic ray.put().
                               Objects larger than this are automatically placed
                               in the object store. Default: 100KB
        """
        self._router_actor = router_actor
        self._tool_name = tool_name
        self._auto_put_threshold = auto_put_threshold
    
    def __getattr__(self, method_name: str):
        """
        Dynamically create method calls.
        
        When user calls toolkit.greeting.greet("TestUser"), this creates
        a function that routes the call through the ToolRouterActor.
        
        Large objects are automatically placed in the Ray object store
        before being passed to the router.
        """
        def method(*args, **kwargs):
            from .ray_utils import prepare_args_for_ray
            
            # Automatically put large objects in the object store
            args, kwargs = prepare_args_for_ray(args, kwargs, self._auto_put_threshold)
            
            return ray.get(self._router_actor.call_tool.remote(self._tool_name, method_name, *args, **kwargs))
        
        # Set a helpful name for debugging
        method.__name__ = f"{self._tool_name}.{method_name}"
        return method


class ToolkitClient:
    """
    Main client for accessing the toolshed toolkit with pythonic syntax.
    
    This class provides the interface that Ray workers use to access tools.
    It connects to the ToolRouterActor and provides clean pythonic access.
    
    Usage:
        toolkit = get_toolkit()  # or ToolkitClient()
        result = toolkit.greeting.greet("TestUser")
    """
    
    def __init__(self, router_name: str = "toolshed_router", namespace: str = "toolshed"):
        """
        Initialize client to connect to ToolRouterActor.
        
        Args:
            router_name: Name of the ToolRouterActor to connect to
            namespace: Ray namespace where the router actor is located
        """
        self._router_name = router_name
        self._namespace = namespace
        self._router_actor = None
        self._tool_proxies = {}
    
    def _get_router_actor(self):
        """Get reference to the ToolRouterActor."""
        if self._router_actor is None:
            try:
                self._router_actor = ray.get_actor(self._router_name, namespace=self._namespace)
            except ValueError as e:
                raise RuntimeError(f"Could not find ToolRouterActor '{self._router_name}' in namespace '{self._namespace}'. "
                                 f"Make sure it's been started first with start_toolkit(). Error: {e}")
        return self._router_actor
    
    def __getattr__(self, tool_name: str) -> ToolProxy:
        """
        Get a tool proxy for pythonic access.
        
        When user accesses toolkit.greeting, this returns a ToolProxy
        that handles method calls like .greet("TestUser").
        """
        if tool_name.startswith('_'):
            # Avoid infinite recursion for private attributes
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{tool_name}'")
        
        if tool_name not in self._tool_proxies:
            router_actor = self._get_router_actor()
            self._tool_proxies[tool_name] = ToolProxy(router_actor, tool_name)
        
        return self._tool_proxies[tool_name]
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        router_actor = self._get_router_actor()
        return ray.get(router_actor.get_available_tools.remote())
    
    def get_tool_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all tools."""
        router_actor = self._get_router_actor()
        return ray.get(router_actor.get_tool_status.remote())
    
    def get_per_actor_stats(self, tool_name: str) -> List[Dict[str, Any]]:
        """
        Get individual statistics from each actor of a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            List of statistics dictionaries, one per actor
        """
        router_actor = self._get_router_actor()
        return ray.get(router_actor.get_per_actor_stats.remote(tool_name))
    
    def get_documentation(self, active_only: bool = True, for_code_execution: bool = True) -> str:
        """
        Generate documentation for the toolkit.

        By default this returns *all* registered tools, matching the previous
        behaviour.  If ``active_only`` is *True* the list is restricted to the
        tools that currently have actors running ("active" tools).  This avoids
        importing heavy dependencies of unused tools when generating docs in a
        lightweight driver process.

        Parameters
        ----------
        active_only
            If *True*, include only active tools.  Defaults to *True*
        for_code_execution
            If *True*, exclude conditional documentation blocks.  Defaults to *True*
        """

        tool_names = None
        if active_only:
            # Use router query rather than list_available_tools() to avoid
            # importing every tool module.
            tool_names = self.get_active_tools()

        return generate_tool_documentation(tool_names, for_code_execution)

    # ------------------------------------------------------------------
    # NEW: Active-tool discovery bound to the current client context
    # ------------------------------------------------------------------
    def get_active_tools(self) -> List[str]:
        """Return the list of tools that currently have actors running.

        The query is executed against *this* client's router and namespace so
        the result is always consistent with subsequent calls made through
        the same client instance.
        """
        router_actor = self._get_router_actor()
        return ray.get(router_actor.list_active_tools.remote())

    def call_tool(self, tool_name: str, method_name: str, *args, **kwargs):
        """Convenience wrapper matching the old ToolkitClient API.

        This lets existing code that used ``toolkit.call_tool("greeting", "greet", ...)``
        continue to work after the migration to the pythonic access style.
        
        Large objects are automatically placed in the Ray object store.
        """
        from .ray_utils import prepare_args_for_ray
        
        router_actor = self._get_router_actor()
        
        # Automatically put large objects in the object store
        args, kwargs = prepare_args_for_ray(args, kwargs)
        
        return ray.get(router_actor.call_tool.remote(tool_name, method_name, *args, **kwargs))
    
    def export_openai_schemas(self, 
                             timeout: int = 30,
                             include_code_executor: bool = False,
                             use_image_by_index: bool = False):
        """Export OpenAI/Anthropic function-tool schemas for currently running tools.
        
        Args:
            timeout: Default per-call timeout stored in the generated configs.
            include_code_executor: Whether to include the special execute_python tool.
            use_image_by_index: If True, replace heavy "image" params with lightweight
                integer "image_index" so the model points to images already in the
                conversation.
                
        Returns:
            List of schema configuration dictionaries
        """
        router_actor = self._get_router_actor()
        return ray.get(router_actor.export_openai_schemas.remote(
            timeout=timeout,
            include_code_executor=include_code_executor,
            use_image_by_index=use_image_by_index
        ))


def _start_or_connect_dashboard(router_name: str,
                               namespace: str,
                               port: int) -> None:
    """
    Start a new dashboard or connect to an existing one.
    
    The dashboard is always created with detached lifetime since we don't
    return its handle and it needs to be discoverable by name.
    
    Args:
        router_name: Name of the router actor
        namespace: Ray namespace
        port: Port for the dashboard
    """
    from toolshed.dashboard.dashboard_actor import DashboardActor
    
    dashboard_name = f"{router_name}_dashboard"
    
    # Try to get existing dashboard first
    try:
        dashboard_actor = ray.get_actor(dashboard_name, namespace=namespace)
        logger.info(f"Found existing dashboard actor '{dashboard_name}' in namespace '{namespace}'")
        
        # Check if it's actually running
        status = ray.get(dashboard_actor.get_status.remote())
        logger.info(f"Dashboard status: {status}")
        
        if not status['running']:
            logger.info("Existing dashboard not running, starting it...")
            dashboard_actor.start.remote()
        else:
            logger.info(f"Dashboard already running at {status['url']}")
            
    except ValueError as e:
        # Dashboard doesn't exist, create it
        logger.info(f"Creating new dashboard actor '{dashboard_name}' (not found: {e})")
        
        # Dashboard must always be detached since we don't return its handle
        dashboard_options = {
            "name": dashboard_name,
            "namespace": namespace,
            "lifetime": "detached"
        }
            
        dashboard_actor = DashboardActor.options(**dashboard_options).remote(
            router_name=router_name,
            namespace=namespace,
            port=port
        )
        
        # Start the dashboard - DON'T BLOCK WITH ray.get()!
        # Just fire and forget the start command
        dashboard_actor.start.remote()
        logger.info(f"Started dashboard actor on port {port}")
        
        # Give the dashboard a moment to start up
        import time
        time.sleep(2)


def start_toolkit(tool_configs: Dict[str, Dict[str, Any]], 
                  router_name: str = "toolshed_router",
                  namespace: str = "toolshed",
                  ray_address: Optional[str] = None,
                  detached: bool = False,
                  dashboard: bool = False,
                  dashboard_port: int = 7001,
                  placement_group: Optional[Any] = None,
                  placement_group_size: Optional[Union[int, str]] = None) -> ray.actor.ActorHandle:
    """
    Start the distributed toolshed toolkit.
    
    This creates the ToolRouterActor that manages all tool actors and handles
    load balancing. Call this once per Ray cluster.
    
    Args:
        tool_configs: Tool configurations. Example:
                     {"greeting": {"num_actors": 2, "conda_env": None, "resources": {}}}
        router_name: Name for the router actor (for ray.get_actor)
        namespace: Ray namespace for cluster-wide discovery (default: "toolshed")
        ray_address: Ray cluster address (None for local)
        detached: Whether actors should be detached from parent process (default: True)
                 If False, actors will be cleaned up when parent process dies
        dashboard: Whether to start the web dashboard (default: False)
        dashboard_port: Port for the dashboard web server (default: 7001)
        placement_group: Optional Ray placement group to constrain where tool actors are scheduled.
                        All tool actors will be placed in the first bundle (index 0) of the group.
                        Use this to constrain tools to specific nodes.
        placement_group_size: Optional convenience parameter to auto-create a placement group.
                            Can be an integer (number of GPUs) or "auto" to calculate from tool_configs.
                            If provided, a placement group will be created with STRICT_PACK strategy.
                            Cannot be used together with placement_group parameter.
        
    Returns:
        Handle to the ToolRouterActor
        
    Example:
        tool_configs = {"greeting": {"num_actors": 2}}
        router = start_toolkit(tool_configs)
        
        # For training scripts that want automatic cleanup on crash:
        router = start_toolkit(tool_configs, detached=False)
        
        # With dashboard:
        router = start_toolkit(tool_configs, dashboard=True, dashboard_port=7001)
        
        # With manual placement group to constrain to one node:
        pg = ray.util.placement_group([{"GPU": 8}], strategy="STRICT_PACK")
        ray.get(pg.ready())
        router = start_toolkit(tool_configs, placement_group=pg)
        
        # With auto-created placement group (8 GPUs):
        router = start_toolkit(tool_configs, placement_group_size=8)
        
        # With auto-calculated placement group size:
        router = start_toolkit(tool_configs, placement_group_size="auto")
    """
    if not ray.is_initialized():
        if ray_address:
            ray.init(address=ray_address, log_to_driver=True)
        else:
            ray.init(log_to_driver=True)
    
    # Handle placement group creation if placement_group_size is provided
    if placement_group_size is not None:
        if placement_group is not None:
            raise ValueError("Cannot specify both 'placement_group' and 'placement_group_size'. "
                           "Use 'placement_group' to pass a pre-created group, or 'placement_group_size' "
                           "to auto-create one.")
        
        # Calculate GPU requirement
        if placement_group_size == "auto":
            # Sum up GPU requirements from all tool configs
            total_gpus = 0.0
            for tool_name, config in tool_configs.items():
                resources = config.get('resources', {})
                num_actors = config.get('num_actors', 1)
                gpus_per_actor = resources.get('num_gpus', 0.0)
                total_gpus += num_actors * gpus_per_actor
            
            # Round up to nearest integer
            import math
            num_gpus = math.ceil(total_gpus)
            
            logger.warning(
                f"Auto-calculated placement group size: {num_gpus} GPUs "
                f"(total requirement: {total_gpus:.2f} GPUs across all tools). "
                f"Creating placement group with STRICT_PACK strategy."
            )
        elif isinstance(placement_group_size, int):
            num_gpus = placement_group_size
            logger.info(f"Creating placement group with {num_gpus} GPUs using STRICT_PACK strategy.")
        else:
            raise ValueError(f"placement_group_size must be an integer or 'auto', got: {placement_group_size}")
        
        # Create the placement group
        placement_group = ray.util.placement_group([{"GPU": num_gpus}], strategy="STRICT_PACK")
        ray.get(placement_group.ready())
        logger.info("Placement group is ready.")
    
    # Start dashboard if requested
    if dashboard:
        _start_or_connect_dashboard(router_name, namespace, dashboard_port)
    
    # Create named router actor with optional detached lifetime
    router_options = {
        "name": router_name,
        "namespace": namespace
    }
    if detached:
        router_options["lifetime"] = "detached"
    
    router_actor = ToolRouterActor.options(**router_options).remote()
    
    # Initialize it with tool configurations
    ray.get(router_actor.initialize.remote(tool_configs, placement_group=placement_group))
    
    logger.info(f"Started ToolRouterActor as '{router_name}' in namespace '{namespace}' with tools: {list(tool_configs.keys())}")
    
    return router_actor


def get_toolkit(router_name: str = "toolshed_router", namespace: str = "toolshed") -> ToolkitClient:
    """
    Get a client to access the distributed toolshed toolkit.
    
    This connects to an existing ToolRouterActor and provides pythonic access
    to tools. Call this from any Ray worker that needs to use tools.
    
    Args:
        router_name: Name of the ToolRouterActor to connect to
        namespace: Ray namespace where the router actor is located (default: "toolshed")
        
    Returns:
        ToolkitClient for pythonic access
        
    Example:
        toolkit = get_toolkit()
        result = toolkit.greeting.greet("TestUser")
    """
    return ToolkitClient(router_name, namespace)


def shutdown_toolkit(router_name: str = "toolshed_router", namespace: str = "toolshed"):
    """
    Shutdown the distributed toolkit.
    
    This shuts down the ToolRouterActor and all managed tool actors.
    
    Args:
        router_name: Name of the ToolRouterActor to shutdown
        namespace: Ray namespace where the router actor is located (default: "toolshed")
    """
    try:
        router_actor = ray.get_actor(router_name, namespace=namespace)
        ray.kill(router_actor)
        #ray.get(router_actor.shutdown.remote())
        logger.info(f"Shutdown ToolRouterActor '{router_name}' in namespace '{namespace}'")
    except ValueError:
        logger.warning(f"ToolRouterActor '{router_name}' not found in namespace '{namespace}' - may already be shut down")
    
    # Also shutdown dashboard if it exists
    dashboard_name = f"{router_name}_dashboard"
    try:
        dashboard_actor = ray.get_actor(dashboard_name, namespace=namespace)
        # Just kill it immediately - no graceful shutdown
        ray.kill(dashboard_actor)
        logger.info(f"Killed DashboardActor '{dashboard_name}' in namespace '{namespace}'")
    except ValueError:
        # Dashboard might not have been started, which is fine
        pass 
