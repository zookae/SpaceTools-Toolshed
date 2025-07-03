# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
ShareGPT format conversion utilities.

Converts conversation messages into ShareGPT format compatible with LLaMA-Factory
and other fine-tuning frameworks.
"""

import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def extract_text_content(content) -> str:
    """
    Extract plain text from content (handles both string and list formats).
    
    Args:
        content: Content which may be string or list of content items
        
    Returns:
        Plain text string
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Extract text from content items
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get('type')
                if item_type == 'text':
                    text_parts.append(item.get('text', ''))
                elif item_type == 'image':
                    text_parts.append('<image>')  # Standard multimodal token
                elif item_type == 'image_url':
                    text_parts.append('<image>')  # OpenAI format → standard token
                elif item_type == 'tool_result':
                    # Extract content from tool result (Anthropic/Bedrock format)
                    tool_content = item.get('content', '')
                    if isinstance(tool_content, str):
                        text_parts.append(tool_content)
                    else:
                        # If content is structured (dict/list), format it
                        text_parts.append(str(tool_content))
                elif item_type == 'tool_use':
                    # Tool calls are handled at message level as separate function_call messages
                    # Don't include them in the text content
                    pass
                elif item_type == 'thinking':
                    # Extended thinking content (Anthropic format)
                    thinking_text = item.get('thinking', item.get('text', ''))
                    text_parts.append(f"{thinking_text}")
                elif item_type == 'redacted_thinking':
                    # Redacted thinking content (Anthropic format)
                    pass
                else:
                    # Unknown content item type - fail explicitly
                    raise ValueError(f"Unknown content item type '{item_type}' in message content. Item: {item}")
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                # Unknown content item format - fail explicitly
                raise ValueError(f"Unknown content item format (type: {type(item)}). Item: {item}")
        return ' '.join(text_parts).strip()
    return str(content)


def convert_messages_to_sharegpt(messages: List[Dict]) -> List[Dict]:
    """
    Convert messages to ShareGPT format.
    
    ShareGPT format for LLaMA-Factory:
    - role: system/user/assistant/function_call/observation
    - content: message text
    - For tool calls: role='function_call', content=JSON string with name and arguments
    - For tool responses: role='observation', content=tool output
    
    Args:
        messages: Conversation messages
        
    Returns:
        List of ShareGPT formatted messages
    """
    sharegpt_messages = []
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'system':
            sharegpt_messages.append({
                "role": "system",
                "content": extract_text_content(content)
            })
        
        elif role == 'user':
            # Check if this is a tool result in Anthropic/Bedrock format
            # (tool results come as user messages with tool_result content blocks)
            is_tool_result = False
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'tool_result':
                        is_tool_result = True
                        break
            
            if is_tool_result:
                # Convert to observation role for ShareGPT format
                sharegpt_messages.append({
                    "role": "observation",
                    "content": extract_text_content(content)
                })
            else:
                # Regular user message (may contain images)
                user_msg = {
                    "role": "user",
                    "content": extract_text_content(content)
                }
                sharegpt_messages.append(user_msg)
        
        elif role == 'assistant':
            # Check for tool calls - they can be at message level OR in content blocks
            tool_calls_list = []
            
            # Option 1: Message-level tool_calls (OpenAI format)
            if 'tool_calls' in msg and msg['tool_calls']:
                tool_calls_list = msg['tool_calls']
            
            # Option 2: Content-level tool_use blocks (Anthropic/Bedrock format)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'tool_use':
                        tool_calls_list.append({
                            'name': item.get('name', ''),
                            'args': item.get('input', {})
                        })
            
            if tool_calls_list:
                # Add assistant message if it has non-tool content
                extracted_content = extract_text_content(content)
                if extracted_content:
                    sharegpt_messages.append({
                        "role": "assistant",
                        "content": extracted_content
                    })
                
                # Add each tool call as function_call
                for tool_call in tool_calls_list:
                    function_call_content = json.dumps({
                        "name": tool_call.get('name', ''),
                        "arguments": tool_call.get('args', {})
                    })
                    sharegpt_messages.append({
                        "role": "function_call",
                        "content": function_call_content
                    })
            else:
                # Regular assistant message
                sharegpt_messages.append({
                    "role": "assistant",
                    "content": extract_text_content(content)
                })
        
        elif role == 'tool':
            # Tool response becomes observation
            sharegpt_messages.append({
                "role": "observation",
                "content": extract_text_content(content)
            })
        
        elif role == 'thinking':
            # Extended thinking blocks (e.g., Claude extended thinking)
            # Preserve as a separate message for training data
            sharegpt_messages.append({
                "role": "thinking",
                "content": extract_text_content(content)
            })
        
        else:
            # Unknown role - fail explicitly instead of silently skipping
            raise ValueError(f"Unknown message role '{role}' in conversation. Message: {msg}")
    
    return sharegpt_messages

