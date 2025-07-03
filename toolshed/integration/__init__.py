# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Integration helpers that allow external frameworks (e.g. Verl, OpenAI, Anthropic) to
plug Toolshed into an LLM/tool-calling workflow.

This sub-package exposes helpers in :pyfile:`verl.py` to build
JSON schemas and callable wrappers compatible with OpenAI/Verl style
function-calling.

For the agentic workflow, use toolshed.agent.ToolAgent instead.
"""

from .verl import (
    get_toolshed_tool_schemas,
    get_toolshed_tool_wrappers,
    create_verl_tool_configs,
    get_verl_tool_schemas,  # Optional, returns None if Verl not available
)

# Legacy OpenAI-specific integration (will be deprecated)
try:
    from .openai import (
        OpenAIToolshedIntegration,
        create_openai_integration,
    )
    __all__ = [
        "get_toolshed_tool_schemas",
        "get_toolshed_tool_wrappers", 
        "create_verl_tool_configs",
        "get_verl_tool_schemas",
        "OpenAIToolshedIntegration",
        "create_openai_integration",
    ]
except ImportError:
    # OpenAI not available
    __all__ = [
        "get_toolshed_tool_schemas",
        "get_toolshed_tool_wrappers",
        "create_verl_tool_configs", 
        "get_verl_tool_schemas",
    ] 