# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
OpenAI provider implementation for Toolshed LLM integration.
"""

import json
import base64
import io
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

from ..base_provider import BaseLLMProvider

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = None


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-specific implementation of the LLM provider interface."""
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize OpenAI provider.
        
        Args:
            model: OpenAI model to use (e.g., "gpt-4o", "gpt-4o-mini")
        """
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                "Install it with: pip install openai"
            )
        self.model = model
        self.client = None
        
    def create_client(self, **kwargs):
        """Create OpenAI client."""
        self.client = OpenAI(**kwargs)
        return self.client
    
    def format_tools(self, tool_schemas: List[Dict]) -> List[Dict]:
        """OpenAI tools are already in the correct format."""
        return tool_schemas
    
    def format_messages(self, messages: List[Dict], images: List[Image.Image]) -> List[Dict]:
        """Format messages for OpenAI API."""
        # OpenAI messages are already in the correct format
        return messages
    
    async def call_api_async(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Any:
        """Call OpenAI API asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Different token limits for different models
        default_max_tokens = {
            "gpt-4o": 4096,
            "gpt-4o-mini": 4096,
            "gpt-4": 4096,
            "gpt-4-turbo": 4096,
            "gpt-3.5-turbo": 4096,
            "gpt-5": 8192,  # Assuming GPT-5 has higher limits
            "gpt-5-mini": 4096
        }
        
        # Use model-specific default or provided value
        max_tokens = kwargs.get('max_completion_tokens', default_max_tokens.get(self.model, 4096))
        
        # Build base API call kwargs
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": tools if tools else None,
            "tool_choice": "auto" if tools else None,
            "max_completion_tokens": max_tokens,
        }
        
        # Only add reasoning_effort for models that support it (gpt-5 and later)
        if self.model.startswith("gpt-5"):
            print(f"OpenAI provider: using reasoning effort for model {self.model}")
            api_kwargs["reasoning_effort"] = "low"

        # Use standard chat completions API only - no fallbacks
        return await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(**api_kwargs)
        )
    
    def extract_message_content(self, response: Any) -> Optional[str]:
        """Extract text content from OpenAI response."""
        message = response.choices[0].message
        return message.content
    
    def has_tool_calls(self, response: Any) -> bool:
        """Check if OpenAI response has tool calls."""
        message = response.choices[0].message
        return hasattr(message, 'tool_calls') and message.tool_calls is not None
    
    def extract_tool_calls(self, response: Any) -> List[Dict]:
        """Extract tool calls from OpenAI response."""
        message = response.choices[0].message
        if not self.has_tool_calls(response):
            return []
        
        return [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
            for tc in message.tool_calls
        ]
    
    def format_tool_result_message(self, tool_call_id: str, result: str, role: str = None) -> Dict:
        """Format tool result for OpenAI."""
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        }
        # DEBUG: Log tool result message formatting
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"OpenAI provider formatted tool result: role={tool_msg['role']}, tool_call_id={tool_call_id}, content_length={len(result)}")
        return tool_msg
    
    def format_image_message(self, text: str, image: Image.Image, role: str = "user") -> Dict:
        """Format an image message for OpenAI."""
        # Convert image to base64
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "role": role,
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    
    def add_message_to_history(self, messages: List[Dict], response: Any) -> List[Dict]:
        """Add OpenAI response to message history."""
        message = response.choices[0].message
        messages.append(message.model_dump())
        return messages
    
    @property
    def supports_system_messages(self) -> bool:
        """OpenAI supports system messages."""
        return True
    
    @property
    def max_image_size(self) -> Optional[Tuple[int, int]]:
        """OpenAI doesn't have strict image size limits."""
        return None
