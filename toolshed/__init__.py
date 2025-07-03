# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
toolshed: A distributed toolkit for AI agents.

Main user-facing exports:
- ToolkitClient: Main client class for pythonic tool access
- start_toolkit(): Start the distributed toolkit 
- get_toolkit(): Get a client to access tools
- shutdown_toolkit(): Shutdown the toolkit

Example usage:
    from toolshed import start_toolkit, get_toolkit
    
    # Start the toolkit (once per cluster) - uses "toolshed" namespace by default
    tool_configs = {"greeting": {"num_actors": 2}}
    start_toolkit(tool_configs)
    
    # Get a client and use tools - connects to "toolshed" namespace by default
    toolkit = get_toolkit()
    result = toolkit.greeting.greet("TestUser")
    
    # Or use custom namespaces for isolation:
    start_toolkit(tool_configs, router_name="my_router", namespace="my_project")
    toolkit = get_toolkit(router_name="my_router", namespace="my_project")
"""

# Import user-facing classes and functions
from .tool_client import (
    ToolkitClient,
    start_toolkit,
    get_toolkit,
    shutdown_toolkit
)

# Import tool-related utilities that users might need
from .registry import list_available_tools, get_tool_class

# Import ToolResult for tool developers
from .tool_result import ToolResult

# Import code executor
from .code_executor import CodeExecutor

# Import execution wrappers for code execution context
from .execution_wrappers import (
    ToolkitExecutionWrapper,
    ToolExecutionWrapper,
    create_execution_toolkit
)

# Import documentation generator
from .documentation import generate_tool_documentation

# Import dashboard actor for advanced users
from .dashboard.dashboard_actor import DashboardActor

# ---------------------------------------------------------------------------
# Version info
# ---------------------------------------------------------------------------
__version__ = "0.1.0"

# Public API - these are what users should import
__all__ = [
    # Main client API
    "ToolkitClient",
    "start_toolkit", 
    "get_toolkit",
    "shutdown_toolkit",
    
    # Tool discovery utilities
    "list_available_tools",
    "get_tool_class",
    
    # Tool development
    "ToolResult",
    
    # Code execution
    "CodeExecutor",
    "ToolkitExecutionWrapper",
    "ToolExecutionWrapper",
    "create_execution_toolkit",
    
    # Documentation
    "generate_tool_documentation",
    
    # Dashboard (for advanced users)
    "DashboardActor",
]

# Optional Weave tracing – explicitly enable if available (opt-in).
try:
    # Expose helpers for Weave tracing
    from .integration.weave_trace import (
        enable_weave_tracing,
        initialize_ray_worker_tracing
    )  # type: ignore

    __all__.extend(["enable_weave_tracing", "initialize_ray_worker_tracing"])
except Exception:
    # Weave or Verl might be missing – ignore.
    pass 