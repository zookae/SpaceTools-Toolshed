# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Step types emitted during agent execution.

These factory functions create step dictionaries that are passed to step callbacks
during `get_response_async()`. Each function documents the fields it produces.

Usage:
    from toolshed.step_types import make_reasoning_step, make_tool_result_step
    
    # In your step callback:
    def handle_step(step):
        if step["type"] == "reasoning":
            print(step["message"])  # Full LLM message
        elif step["type"] == "tool_result":
            print(step["full_result"])  # Complete tool output
"""

from typing import Dict, Any, List, Optional


def make_reasoning_step(
    iteration: int,
    message: str,
    timestamp: str
) -> Dict[str, Any]:
    """
    Emitted when the LLM responds with text.
    
    Fields:
        type: "reasoning"
        iteration: Current iteration number
        status: Truncated preview of the message
        message: Full LLM message content
        word_count: Number of words in message
        char_count: Number of characters in message
        timestamp: ISO timestamp
    """
    return {
        "type": "reasoning",
        "iteration": iteration,
        "status": f"💭 AI reasoning: {message[:300]}{'...' if len(message) > 300 else ''}",
        "message": message,
        "word_count": len(message.split()),
        "char_count": len(message),
        "timestamp": timestamp
    }


def make_tool_decision_step(
    iteration: int,
    tool_calls: List[Dict[str, Any]],
    timestamp: str
) -> Dict[str, Any]:
    """
    Emitted when the LLM decides to call tools.
    
    Fields:
        type: "tool_decision"
        iteration: Current iteration number
        status: Summary of how many tools will be called
        tool_calls: List of {name, arguments} for each tool
        timestamp: ISO timestamp
    """
    return {
        "type": "tool_decision",
        "iteration": iteration,
        "status": f"🛠️ Decided to use {len(tool_calls)} tool(s)",
        "tool_calls": [
            {
                "name": tc["name"],
                "arguments": tc["arguments"],
            } for tc in tool_calls
        ],
        "timestamp": timestamp
    }


def make_tool_executing_step(
    iteration: int,
    tool_index: int,
    tool_name: str,
    arguments: Dict[str, Any],
    timestamp: str
) -> Dict[str, Any]:
    """
    Emitted when a tool starts executing.
    
    Fields:
        type: "tool_executing"
        iteration: Current iteration number
        tool_index: 1-based index of this tool in the batch
        tool_name: Name of the tool being executed
        status: Execution status message
        arguments: Arguments passed to the tool
        timestamp: ISO timestamp
    """
    return {
        "type": "tool_executing",
        "iteration": iteration,
        "tool_index": tool_index,
        "tool_name": tool_name,
        "status": f"⚡ Executing {tool_name}...",
        "arguments": arguments,
        "timestamp": timestamp
    }


def make_tool_result_step(
    iteration: int,
    tool_index: int,
    tool_name: str,
    result: str,
    has_image: bool,
    timestamp: str,
    image_data: Optional[str] = None
) -> Dict[str, Any]:
    """
    Emitted when a tool completes execution.
    
    Fields:
        type: "tool_result"
        iteration: Current iteration number
        tool_index: 1-based index of this tool in the batch
        tool_name: Name of the tool that executed
        status: Completion status message
        result: Truncated result (first 200 chars)
        full_result: Complete result text
        has_image: Whether the tool generated an image
        image_data: Base64-encoded image (only if has_image and tracking enabled)
        timestamp: ISO timestamp
    """
    step = {
        "type": "tool_result",
        "iteration": iteration,
        "tool_index": tool_index,
        "tool_name": tool_name,
        "status": f"✅ {tool_name} completed" + (" (with generated image)" if has_image else ""),
        "result": result[:200] + "..." if len(result) > 200 else result,
        "full_result": result,
        "has_image": has_image,
        "timestamp": timestamp
    }
    if image_data is not None:
        step["image_data"] = image_data
    return step


def make_synthesizing_step(
    iteration: int,
    num_tools: int,
    timestamp: str
) -> Dict[str, Any]:
    """
    Emitted after tool execution, before calling LLM again.
    
    Fields:
        type: "synthesizing"
        iteration: Current iteration number
        status: Synthesis status message
        timestamp: ISO timestamp
    """
    return {
        "type": "synthesizing",
        "iteration": iteration,
        "status": f"🧠 Synthesizing results from {num_tools} tool(s)...",
        "timestamp": timestamp
    }


def make_complete_step(
    iteration: int,
    message: str,
    timestamp: str
) -> Dict[str, Any]:
    """
    Emitted when the agent finishes (no more tool calls).
    
    Fields:
        type: "complete"
        iteration: Current iteration number
        status: Completion status message
        message: Final LLM response content
        timestamp: ISO timestamp
    """
    return {
        "type": "complete",
        "iteration": iteration,
        "status": "🎯 Analysis complete, formulating final response",
        "message": message,
        "timestamp": timestamp
    }

