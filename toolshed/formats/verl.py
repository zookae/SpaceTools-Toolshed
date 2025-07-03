# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
VeRL format conversion utilities.

Converts conversation messages from various LLM provider formats (OpenAI, Anthropic, Bedrock)
into VeRL's standardized format.
"""

import json
import re
from typing import Dict, List, Optional


def convert_content_item_to_verl(item: Dict, verl_msg: Dict) -> Optional[Dict]:
    """
    Convert a single content item from LLM provider format to VeRL format.
    
    Provider formats (OpenAI, Anthropic, Bedrock) → VeRL format
    - text: kept as-is
    - image_url: converted to generic "image" type (removes base64 data)
    - tool_use (Anthropic): extracted to message-level tool_calls
    - thinking/redacted_thinking: kept as-is for extended thinking support
    
    Args:
        item: Content item dict with 'type' field
        verl_msg: Message being built (modified in-place for tool_calls)
        
    Returns:
        Content item to add to message, or None if handled separately (e.g., tool_use)
    """
    item_type = item.get("type")
    
    if item_type == "text":
        return item
    elif item_type == "image_url":
        # Strip base64 data, just reference that an image exists
        return {"type": "image"}
    elif item_type == "tool_use":
        # Anthropic/Bedrock format: move to message-level tool_calls
        if "tool_calls" not in verl_msg:
            verl_msg["tool_calls"] = []
        verl_msg["tool_calls"].append({
            "name": item["name"],
            "args": item["input"]
        })
        return None  # Don't add to content
    elif item_type == "tool_result":
        # Anthropic/Bedrock tool response: extract to message-level tool_call_id
        if "tool_use_id" in item:
            verl_msg["tool_call_id"] = item["tool_use_id"]
        # Keep the result content
        return {"type": "text", "text": item.get("content", "")}
    elif item_type in ("thinking", "redacted_thinking", "image"):
        return item
    else:
        raise ValueError(f"Unknown content item type: {item_type}. Full item: {item}")


def convert_message_content_to_verl(msg: Dict) -> Dict:
    """
    Convert message content from LLM provider format to VeRL format.
    
    Handles three content formats:
    1. List (multimodal): [{type: "text", ...}, {type: "image_url", ...}]
    2. String with embedded base64: "text data:image/jpeg;base64,..."
    3. Plain string: "simple text"
    
    Args:
        msg: Original message dict with 'content' field
        
    Returns:
        VeRL-formatted message dict with converted content
    """
    verl_msg = {"role": msg["role"]}
    content = msg["content"]
    
    if isinstance(content, list):
        # Multimodal format: convert each content item
        content_items = []
        for item in content:
            converted = convert_content_item_to_verl(item, verl_msg)
            if converted is not None:
                content_items.append(converted)
        verl_msg["content"] = content_items
        
    elif isinstance(content, str) and "data:image" in content and "base64," in content:
        # Legacy string format with embedded base64 images - parse and convert
        content_items = []
        parts = content.split("data:image")
        
        # Add text before first image
        if parts[0].strip():
            content_items.append({"type": "text", "text": parts[0].strip()})
        
        # Add image placeholders for each embedded image
        for part in parts[1:]:
            content_items.append({"type": "image"})
            
            # Check for text after the base64 data
            if "base64," in part:
                after_base64 = part.split("base64,", 1)[1]
                text_match = re.search(r'["\s]*([^"]*?)(?:\s*["\]}]|$)', after_base64)
                if text_match and text_match.group(1).strip():
                    content_items.append({"type": "text", "text": text_match.group(1).strip()})
        
        verl_msg["content"] = content_items if content_items else content
        
    else:
        # Plain string content
        verl_msg["content"] = content
    
    return verl_msg


def convert_messages_to_verl(messages: List[Dict]) -> List[Dict]:
    """
    Convert a complete message list from LLM provider format to VeRL format.
    
    Handles:
    - Content conversion (text, images, tool use blocks)
    - OpenAI tool_calls format → VeRL tool_calls format
    - Tool response metadata (tool_call_id)
    
    Args:
        messages: Original messages from LLM provider
        
    Returns:
        VeRL-formatted messages
    """
    verl_messages = []
    
    for msg in messages:
        # Convert content
        verl_msg = convert_message_content_to_verl(msg)
        
        # Convert OpenAI-style tool_calls (if not already handled by content conversion)
        if "tool_calls" in msg and msg["tool_calls"] and "tool_calls" not in verl_msg:
            verl_msg["tool_calls"] = []
            for tc in msg["tool_calls"]:
                verl_msg["tool_calls"].append({
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"])
                })
        
        # Preserve tool response metadata
        if "tool_call_id" in msg:
            verl_msg["tool_call_id"] = msg["tool_call_id"]
        
        verl_messages.append(verl_msg)
    
    return verl_messages

