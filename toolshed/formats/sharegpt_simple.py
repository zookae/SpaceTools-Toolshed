# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Simplified ShareGPT format conversion utilities.

Converts conversation messages into a simplified ShareGPT format with:
- Only two roles: user and assistant
- Tool calls/responses embedded with XML-like tags
- System message, images, and tools as separate fields
"""

import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def extract_image_path_from_message(message: Dict) -> Optional[str]:
    """
    Extract image path from a message's content structure.
    
    Args:
        message: Message dict with 'content' field
        
    Returns:
        Image path string or None if not found
    """
    content = message.get('content', '')
    
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'image':
                image_path = item.get('image_path')
                if image_path:
                    return image_path
    
    return None


def extract_all_image_paths_from_messages(messages: List[Dict]) -> List[str]:
    """
    Extract all unique image paths from messages.
    
    Args:
        messages: List of message dicts
        
    Returns:
        List of image path strings (in order of first appearance)
    """
    image_paths = []
    seen = set()
    
    for message in messages:
        path = extract_image_path_from_message(message)
        if path and path not in seen:
            image_paths.append(path)
            seen.add(path)
    
    return image_paths


def wrap_plain_text_in_think_tags(text: str) -> str:
    """
    Wrap plain text parts in <think> tags while preserving existing XML tags.
    
    Parses XML structure and only wraps text that is NOT already inside a tag pair.
    Existing tags like <answer>, <tool_call>, <think>, etc. are preserved as-is.
    
    Args:
        text: Text that may contain XML tags
        
    Returns:
        Text with plain parts wrapped in <think> tags
    """
    import re
    
    # Pattern to match XML-like tags (opening or closing)
    # Captures: opening tags, closing tags, or self-closing tags
    tag_pattern = r'<(/?)([a-zA-Z_][a-zA-Z0-9_]*)(?:\s[^>]*)?>|([^<]+)'
    
    result = []
    tag_stack = []  # Track which tags we're currently inside
    
    for match in re.finditer(tag_pattern, text):
        if match.group(1) is not None or match.group(2) is not None:
            # This is a tag
            is_closing = match.group(1) == '/'
            tag_name = match.group(2)
            full_tag = match.group(0)
            
            if is_closing:
                # Closing tag
                if tag_stack and tag_stack[-1] == tag_name:
                    tag_stack.pop()
                result.append(full_tag)
            else:
                # Opening tag
                tag_stack.append(tag_name)
                result.append(full_tag)
        elif match.group(3) is not None:
            # This is plain text
            plain_text = match.group(3).strip()
            if plain_text:
                if not tag_stack:
                    # Not inside any tags - wrap in <think>
                    result.append(f"<think> {plain_text} </think>")
                else:
                    # Inside a tag - keep as-is
                    result.append(plain_text)
    
    return '\n'.join(result)


def extract_text_from_content(content) -> str:
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
        text_parts = []
        num_image_tokens = 0
        
        # First pass: count existing <image> tokens in text content
        for item in content:
            if isinstance(item, dict):
                item_type = item.get('type')
                if item_type == 'text':
                    text = item.get('text', '')
                    num_image_tokens += text.count('<image>')
            elif isinstance(item, str):
                num_image_tokens += item.count('<image>')
        
        # Second pass: process content and handle image tokens
        for item in content:
            if isinstance(item, dict):
                item_type = item.get('type')
                if item_type == 'text':
                    text_parts.append(item.get('text', ''))
                elif item_type == 'image':
                    # If we already have image tokens, skip and decrement
                    if num_image_tokens > 0:
                        num_image_tokens -= 1
                    else:
                        # Need to add <image> token
                        text_parts.append('<image>')
                elif item_type == 'image_url':
                    # If we already have image tokens, skip and decrement
                    if num_image_tokens > 0:
                        num_image_tokens -= 1
                    else:
                        # Need to add <image> token
                        text_parts.append('<image>')
                elif item_type == 'tool_result':
                    tool_content = item.get('content', '')
                    if isinstance(tool_content, str):
                        text_parts.append(tool_content)
                    else:
                        text_parts.append(str(tool_content))
                elif item_type == 'tool_use':
                    # Skip - handled at message level
                    pass
                elif item_type == 'thinking':
                    thinking_text = item.get('thinking', item.get('text', ''))
                    text_parts.append(thinking_text)
                elif item_type == 'redacted_thinking':
                    pass
                else:
                    raise ValueError(f"Unknown content item type '{item_type}' in message content. Item: {item}")
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                raise ValueError(f"Unknown content item format (type: {type(item)}). Item: {item}")
        return ' '.join(text_parts).strip()
    return str(content)


def format_tool_call_as_xml(tool_call: Dict) -> str:
    """
    Format a tool call as XML-like string.
    
    Converts tool name from internal format to training data format.
    
    Args:
        tool_call: Tool call dict with 'name' and 'args' keys
        
    Returns:
        XML-like string: <tool_call>\n{json}\n</tool_call>
    """
    tool_name = tool_call.get('name', '')
    tool_json = json.dumps({
        "name": convert_tool_name(tool_name),
        "arguments": tool_call.get('args', {})
    })
    return f"<tool_call>\n{tool_json}\n</tool_call>"


def convert_messages_to_sharegpt_simple(messages: List[Dict]) -> Dict[str, Any]:
    """
    Convert messages to simplified ShareGPT format.
    
    Format:
    - conversations: list of messages with only 'user' and 'assistant' roles
    - system: system message content (must have exactly 1)
    - images: list of image paths
    - tools: tool schemas (if available)
    
    Rules:
    - Tool calls are embedded in assistant messages as <tool_call>...</tool_call>
    - Tool responses are embedded in user messages as <tool_response>...</tool_response>
    - Regular assistant text is wrapped in <think>...</think>
    - Final answers are wrapped in <answer>...</answer>
    - Content order within messages is preserved
    
    Args:
        messages: Conversation messages
        
    Returns:
        Dictionary with 'conversations', 'system', 'images', 'tools' keys
        
    Raises:
        ValueError: If number of system messages is not exactly 1
    """
    conversations = []
    system_messages = []
    
    # Extract system messages
    for msg in messages:
        if msg.get('role') == 'system':
            system_messages.append(extract_text_from_content(msg.get('content', '')))
    
    # Validate exactly 1 system message
    if len(system_messages) != 1:
        raise ValueError(f"Expected exactly 1 system message, found {len(system_messages)}")
    
    system_content = system_messages[0]
    
    # Extract image paths
    image_paths = extract_all_image_paths_from_messages(messages)
    
    # Process messages into conversations
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'system':
            # Skip - already handled
            continue
        
        elif role == 'user':
            # Check if this is a tool result
            is_tool_result = False
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'tool_result':
                        is_tool_result = True
                        break
            
            if is_tool_result:
                # Wrap tool result in <tool_response> tags
                result_text = extract_text_from_content(content)
                conversations.append({
                    "role": "user",
                    "content": f"<tool_response>\n{result_text}\n</tool_response>"
                })
            else:
                # Regular user message
                user_text = extract_text_from_content(content)
                conversations.append({
                    "role": "user",
                    "content": user_text
                })
        
        elif role == 'assistant':
            # Build assistant message content preserving order
            content_parts = []
            
            # Extract tool calls from message level or content blocks
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
            
            # Process content based on type
            if isinstance(content, str):
                # Simple string content - wrap in <think> tags
                if content.strip():
                    content_parts.append(f"<think> {content} </think>")
            elif isinstance(content, list):
                # Process content items in order
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get('type')
                        if item_type == 'text':
                            text = item.get('text', '').strip()
                            if text:
                                # Wrap plain text parts in <think> tags while preserving existing tags
                                content_parts.append(wrap_plain_text_in_think_tags(text))
                        elif item_type == 'thinking':
                            thinking_text = item.get('thinking', item.get('text', '')).strip()
                            if thinking_text:
                                content_parts.append(f"<think> {thinking_text} </think>")
                        elif item_type == 'tool_use':
                            # Tool use will be added after regular content
                            pass
                        elif item_type in ['image', 'image_url']:
                            # Skip images in assistant messages
                            pass
                        elif item_type == 'redacted_thinking':
                            # Skip redacted content
                            pass
                        # Other types are ignored or handled elsewhere
                    elif isinstance(item, str):
                        if item.strip():
                            content_parts.append(f"<think> {item} </think>")
            
            # Add tool calls after content
            if tool_calls_list:
                for tool_call in tool_calls_list:
                    content_parts.append(format_tool_call_as_xml(tool_call))
            
            # Join all parts
            if content_parts:
                conversations.append({
                    "role": "assistant",
                    "content": "\n".join(content_parts)
                })
        
        elif role == 'tool':
            # Tool response becomes a user message with <tool_response> tags
            result_text = extract_text_from_content(content)
            conversations.append({
                "role": "user",
                "content": f"<tool_response>\n{result_text}\n</tool_response>"
            })
        
        elif role == 'thinking':
            # Extended thinking blocks - wrap in <think> tags as assistant message
            thinking_text = extract_text_from_content(content)
            conversations.append({
                "role": "assistant",
                "content": f"<think> {thinking_text} </think>"
            })
        
        else:
            raise ValueError(f"Unknown message role '{role}' in conversation. Message: {msg}")
    
    # Merge consecutive user messages to maintain alternating roles
    merged_conversations = []
    for msg in conversations:
        if merged_conversations and merged_conversations[-1]["role"] == "user" and msg["role"] == "user":
            # Append to previous user message on a new line
            merged_conversations[-1]["content"] += "\n" + msg["content"]
        else:
            merged_conversations.append(msg)
    
    return {
        "conversations": merged_conversations,
        "system": system_content,
        "images": image_paths,
        "tools": "[tool descriptions here]"  # Placeholder - will be populated by caller
    }


def convert_tool_name(tool_name: str) -> str:
    """
    Convert tool name from internal format to training data format.
    
    Converts "__" back to "." for tool names (e.g., "vlm__detect_one" -> "vlm.detect_one").
    This reverses the conversion done in LLM integration for compatibility.
    
    Args:
        tool_name: Tool name in internal format
        
    Returns:
        Tool name in training data format
    """
    return tool_name.replace("__", ".")


def convert_tool_schemas_to_simple_format(tool_schemas: Optional[List[Dict]]) -> str:
    """
    Convert tool schemas to simplified format.
    
    Converts tool names from internal format (e.g., "vlm__detect_one") to 
    training data format (e.g., "vlm.detect_one").
    
    Args:
        tool_schemas: List of tool schema dictionaries (or None)
        
    Returns:
        JSON string representation of tool schemas with converted names
    """
    if tool_schemas is None:
        return "[tool descriptions here]"
    
    # Deep copy and convert tool names
    converted_schemas = []
    for schema in tool_schemas:
        schema_copy = schema.copy()
        if 'function' in schema_copy and 'name' in schema_copy['function']:
            # Convert the function name
            schema_copy = json.loads(json.dumps(schema_copy))  # Deep copy
            schema_copy['function']['name'] = convert_tool_name(schema_copy['function']['name'])
        converted_schemas.append(schema_copy)
    
    return json.dumps(converted_schemas, indent=2)


def build_training_example(
    messages: List[Dict],
    task_id: str,
    dataset_index: int,
    score: float,
    image_paths: Optional[List[str]] = None,
    tool_schemas: Optional[List[Dict]] = None,
    system_prompt_override: Optional[str] = None,
    **extra_fields
) -> Dict:
    """
    Build a complete training example in sharegpt_simple format.
    
    Args:
        messages: Conversation messages to convert
        task_id: Unique task identifier
        dataset_index: Index in the original dataset
        score: Evaluation score
        image_paths: List of image paths (overrides any from messages)
        tool_schemas: Tool schema definitions
        system_prompt_override: If provided, replaces the system message from messages
        **extra_fields: Additional fields to include (e.g., ground_truth, question_type)
    
    Returns:
        Complete dict ready to write to JSONL with keys:
        - task_id, dataset_index, score
        - conversations, system, images, tools
        - any extra_fields
    """
    # Convert messages to simple format
    converted = convert_messages_to_sharegpt_simple(messages)
    
    # Build complete example
    example = {
        "task_id": task_id,
        "dataset_index": dataset_index,
        "score": score,
        "conversations": converted["conversations"],
        "system": system_prompt_override if system_prompt_override is not None else converted["system"],
        "images": image_paths if image_paths is not None else converted["images"],
        "tools": convert_tool_schemas_to_simple_format(tool_schemas)
    }
    
    # Add any extra fields (filter out None values)
    for key, value in extra_fields.items():
        if value is not None:
            example[key] = value
    
    return example

