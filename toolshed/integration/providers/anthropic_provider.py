# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Anthropic Claude provider implementation for Toolshed LLM integration.
"""

import json
import base64
import io
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

from ..base_provider import BaseLLMProvider

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    anthropic = None


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude-specific implementation of the LLM provider interface."""
    
    def __init__(self, model: str = "claude-opus-4-1-20250805"):
        """
        Initialize Anthropic provider.
        
        Args:
            model: Anthropic model to use (e.g., "claude-opus-4-1-20250805", "claude-4-20250805")
        """
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            )
        self.model = model
        self.client = None
        self._last_response = None  # Store full response for message history
        
    def create_client(self, **kwargs):
        """Create Anthropic client."""
        self.client = anthropic.Anthropic(**kwargs)
        return self.client
    
    def format_tools(self, tool_schemas: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        
        for schema in tool_schemas:
            if schema.get("type") == "function":
                func = schema["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func["parameters"]
                })
        
        return anthropic_tools
    
    def format_messages(self, messages: List[Dict], images: List[Image.Image]) -> List[Dict]:
        """
        Format messages for Anthropic API.
        
        Anthropic has specific requirements:
        - No system role in messages (handled separately)
        - Content can be string or list of content blocks
        - Tool results are sent as user messages with tool_result blocks
        """
        formatted = []
        
        for msg in messages:
            role = msg.get("role")
            
            # Skip system messages (handled separately in API call)
            if role == "system":
                continue
            
            # Handle tool results (OpenAI format -> Anthropic format)
            if role == "tool":
                # Convert OpenAI tool result to Anthropic format
                formatted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id"),
                            "content": msg.get("content", "")
                        }
                    ]
                })
            else:
                # Regular user/assistant messages
                formatted.append({
                    "role": role,
                    "content": msg.get("content")
                })
        
        return formatted
    
    async def call_api_async(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Any:
        """Call Anthropic API asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Extract system message from original messages if provided
        system_message = kwargs.get('system')  # Still support direct system parameter
        if not system_message:
            original_messages = kwargs.get('original_messages', [])
            for msg in original_messages:
                if msg.get("role") == "system":
                    system_message = msg.get("content")
                    break
        
        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', 4096)
        }
        
        # Add tools if provided
        if tools:
            api_params["tools"] = tools
        
        # Add system message if provided
        if system_message:
            api_params["system"] = system_message
        
        # Make async API call
        response = await loop.run_in_executor(
            None,
            lambda: self.client.messages.create(**api_params)
        )
        
        # Store response for later use
        self._last_response = response
        return response
    
    def extract_message_content(self, response: Any) -> Optional[str]:
        """Extract text content from Anthropic response."""
        # Anthropic responses have content as a list of content blocks
        if hasattr(response, 'content') and response.content:
            # Extract text from content blocks
            text_parts = []
            for block in response.content:
                if hasattr(block, 'type') and block.type == 'text':
                    text_parts.append(block.text)
            
            return '\n'.join(text_parts) if text_parts else None
        
        return None
    
    def has_tool_calls(self, response: Any) -> bool:
        """Check if Anthropic response has tool calls."""
        if hasattr(response, 'content') and response.content:
            return any(
                hasattr(block, 'type') and block.type == 'tool_use'
                for block in response.content
            )
        return False
    
    def extract_tool_calls(self, response: Any) -> List[Dict]:
        """Extract tool calls from Anthropic response."""
        tool_calls = []
        
        if hasattr(response, 'content') and response.content:
            for block in response.content:
                if hasattr(block, 'type') and block.type == 'tool_use':
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": json.dumps(block.input)  # Anthropic uses 'input' not 'arguments'
                    })
        
        return tool_calls
    
    def format_tool_result_message(self, tool_call_id: str, result: str, role: str = None) -> Dict:
        """Format tool result for Anthropic."""
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": result
                }
            ]
        }
    
    def format_image_message(self, text: str, image: Image.Image, role: str = "user") -> Dict:
        """Format an image message for Anthropic."""
        # Convert image to base64
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Anthropic uses a different format for images
        content = []
        
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_base64
            }
        })
        
        return {
            "role": role,
            "content": content
        }
    
    def add_message_to_history(self, messages: List[Dict], response: Any) -> List[Dict]:
        """Add Anthropic response to message history."""
        # Build assistant message from response
        content = []
        
        if hasattr(response, 'content'):
            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == 'text':
                        content.append({
                            "type": "text",
                            "text": block.text
                        })
                    elif block.type == 'tool_use':
                        content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                    elif block.type == 'thinking':
                        # Preserve thinking blocks for thinking mode compatibility
                        thinking_block = {
                            "type": "thinking",
                            "thinking": block.thinking
                        }
                        # Include signature if present
                        if hasattr(block, 'signature') and block.signature:
                            thinking_block["signature"] = block.signature
                        content.append(thinking_block)
                    elif block.type == 'redacted_thinking':
                        print("--------------------------------")
                        print("WARNING: Redacted thinking block found in response")
                        print(block)
                        print("--------------------------------")
                        # Preserve redacted thinking blocks for thinking mode compatibility
                        content.append({
                            "type": "redacted_thinking"
                        })
        
        messages.append({
            "role": "assistant",
            "content": content
        })
        
        return messages
    
    @property
    def supports_system_messages(self) -> bool:
        """Anthropic supports system messages but as a separate parameter."""
        return True
    
    @property
    def max_image_size(self) -> Optional[Tuple[int, int]]:
        """Anthropic has image size limits."""
        # Claude typically supports images up to 5MB and reasonable dimensions
        return (8192, 8192)  # Conservative limit
