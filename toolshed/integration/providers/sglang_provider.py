# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
SGLang provider implementation for Toolshed LLM integration.

This provider connects to a running sglang server hosting fine-tuned vision-language
models (e.g., Qwen2.5-VL) that have been trained with tool calling capabilities.
"""

import json
import base64
import io
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import logging

from ..base_provider import BaseLLMProvider

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    httpx = None

logger = logging.getLogger(__name__)


class SglangProvider(BaseLLMProvider):
    """SGLang server provider for fine-tuned VLMs with tool calling."""
    
    def __init__(self, 
                 model: str = "Qwen2.5-VL-3B-Instruct",
                 base_url: str = "http://localhost:30000/v1",
                 **kwargs):
        """
        Initialize SGLang provider.
        
        Args:
            model: Model identifier (informational, server already has model loaded)
            base_url: Base URL for sglang server (default: http://localhost:30000/v1)
            **kwargs: Additional configuration
        """
        if not _HTTPX_AVAILABLE:
            raise ImportError(
                "SGLang provider requires the 'httpx' package. "
                "Install it with: pip install httpx"
            )
        
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.client = None
        self._last_response = None
        
        # Tool call parsing patterns
        # The model outputs: <tool_call>\n{json}\n</tool_call>
        self.tool_call_pattern = re.compile(
            r'<tool_call>\s*(.*?)\s*</tool_call>',
            re.DOTALL
        )
        
        logger.info(f"Initialized SGLang provider for model {model} at {base_url}")
    
    def create_client(self, **kwargs):
        """Create HTTP client for sglang server."""
        timeout = kwargs.get('timeout', 300.0)  # 5 min default for VLM inference
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout
        )
        return self.client
    
    def format_tools(self, tool_schemas: List[Dict]) -> List[Dict]:
        """
        Convert tool schemas to sglang format.
        
        Sglang uses OpenAI-compatible format, so we pass through as-is.
        """
        return tool_schemas
    
    def format_messages(self, messages: List[Dict], images: List[Image.Image]) -> List[Dict]:
        """
        Format messages for sglang API.
        
        SGLang/Qwen uses OpenAI-compatible message format with base64 images.
        """
        formatted = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            # Skip system messages - will be handled separately
            if role == "system":
                continue
            
            # Handle tool result messages
            if role == "tool":
                # Convert to user message with <tool_response> wrapper
                tool_content = msg.get("content", "")
                formatted.append({
                    "role": "user",
                    "content": f"<tool_response>\n{tool_content}\n</tool_response>"
                })
                continue
            
            # Handle regular messages
            formatted_msg = {
                "role": role,
                "content": content
            }
            
            formatted.append(formatted_msg)
        
        return formatted
    
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _parse_tool_calls_from_text(self, text: str) -> List[Dict]:
        """
        Parse tool calls from model output.
        
        The model outputs tool calls in this format:
        <tool_call>
        {"name": "tool.method", "arguments": {...}}
        </tool_call>
        
        Returns:
            List of dicts with keys: id, name, arguments
        """
        tool_calls = []
        

        # If there are more <tool_call> tags than </tool_call> tags, append a </tool_call> tag to the end of the text
        print(f"DEBUG: text: {text}")
        
        text = text.replace("\\n", "\n").strip()
        if text.count("<tool_call>") > text.count("</tool_call>"):
            text += "</tool_call>"

        # Find all tool call blocks
        matches = self.tool_call_pattern.findall(text)
        
        for idx, match in enumerate(matches):
            try:
                # Clean the match: strip whitespace and replace literal \n strings
                cleaned = match.replace('\\n', '\n').strip()
                
                # Skip if empty after stripping
                if not cleaned:
                    logger.warning(f"Skipping empty tool call block at index {idx}")
                    continue
                
                # Parse the JSON inside the tool_call tags
                tool_data = json.loads(cleaned)
                
                tool_calls.append({
                    "id": f"call_{idx}",
                    "name": tool_data.get("name", ""),
                    "arguments": json.dumps(tool_data.get("arguments", {}))
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON at index {idx}. "
                             f"Content (first 200 chars): {repr(match[:200])}. Error: {e}")
                continue
        
        return tool_calls
    
    def _remove_tool_call_markers(self, text: str) -> str:
        """Remove <tool_call>...</tool_call> blocks from text."""
        return self.tool_call_pattern.sub('', text).strip()
    
    def _format_tools_as_text(self, tools: List[Dict]) -> str:
        """
        Format tool schemas as text for inclusion in the prompt.
        
        The model was trained with tool descriptions in text format, not as API payload.
        """
        if not tools:
            return ""
        
        tool_descriptions = ["Available tools:"]
        
        for tool in tools:
            # Extract tool info from OpenAI format
            if "function" in tool:
                func = tool["function"]
                name = func.get("name", "")
                description = func.get("description", "")
                parameters = func.get("parameters", {})
                
                tool_desc = f"\n- {name}: {description}"
                
                # Add parameter info
                if parameters and "properties" in parameters:
                    props = parameters["properties"]
                    required = parameters.get("required", [])
                    
                    param_list = []
                    for param_name, param_info in props.items():
                        param_type = param_info.get("type", "")
                        param_desc = param_info.get("description", "")
                        req_marker = " (required)" if param_name in required else ""
                        param_list.append(f"    - {param_name} ({param_type}){req_marker}: {param_desc}")
                    
                    if param_list:
                        tool_desc += "\n  Parameters:\n" + "\n".join(param_list)
                
                tool_descriptions.append(tool_desc)
        
        tool_descriptions.append("\nTo use a tool, output: <tool_call>\\n{\"name\": \"tool.name\", \"arguments\": {...}}\\n</tool_call>")
        
        return "\n".join(tool_descriptions)
    
    async def call_api_async(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Any:
        """
        Call sglang server API asynchronously (NON-STREAMING).
        
        Uses the OpenAI-compatible /chat/completions endpoint.
        
        Note: We do NOT send tools in the API payload because the model was trained
        to generate tool calls as text with <tool_call> tags. Tool schemas should be
        included in the system message instead.
        """
        if self.client is None:
            self.create_client()
        
        # Extract system message if present
        system_message = kwargs.get('system')
        if not system_message:
            original_messages = kwargs.get('original_messages', [])
            for msg in original_messages:
                if msg.get("role") == "system":
                    system_message = msg.get("content")
                    break
        
        # Build tool descriptions text if tools are provided
        tool_descriptions = ""
        if tools:
            tool_descriptions = self._format_tools_as_text(tools)
        
        # Build request payload
        # Note: Qwen2.5-VL-3B has 8192 token context limit
        # Use conservative max_tokens default to prevent runaway generation
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get('temperature', 0.0),
            "max_tokens": kwargs.get('max_tokens', 512),  # Conservative default
            "stream": False,  # Non-streaming for simplicity
        }
        
        # Add stop sequences to prevent infinite generation
        default_stops = [
            "   ", # three spaces
            "\n\n",  # Two newlines (common whitespace loop)
            "\\n\\n", 
            "</answer>",  # Stop after final answer tag
            "<|im_end|>",  # Qwen chat template end token
            "<|endoftext|>"  # Standard end token
        ]
        stop_sequences = kwargs.get('stop', default_stops)
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        # Add system message with tool descriptions if present
        if system_message or tool_descriptions:
            combined_system = system_message or ""
            if tool_descriptions:
                combined_system = f"{combined_system}\n\n{tool_descriptions}".strip()
            
            # Prepend to first user message
            if messages and messages[0]["role"] == "user":
                messages[0]["content"] = f"{combined_system}\n\n{messages[0]['content']}"
            else:
                # Insert as first message
                messages.insert(0, {"role": "user", "content": combined_system})
        
        logger.info(f"Calling sglang API with {len(messages)} messages (tools in prompt: {len(tools)})")
        logger.info(f"Generation params: temperature={payload['temperature']}, max_tokens={payload['max_tokens']}")
        logger.info(f"Stop sequences: {payload.get('stop', [])}")
        
        # Log payload for debugging (truncate long content)
        payload_str = json.dumps(payload, indent=2, default=str)
        if len(payload_str) > 2000:
            logger.debug(f"Request payload (truncated): {payload_str[:2000]}...")
        else:
            logger.debug(f"Request payload: {payload_str}")
        
        try:
            # Make async API call (non-streaming)
            logger.info("🤖 Calling SGLang API (non-streaming)...")
            
            response = await self.client.post(
                "/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Log result summary
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
                finish_reason = result["choices"][0].get("finish_reason", "unknown")
                
                logger.info(f"✅ Response received: {len(content)} chars, finish_reason={finish_reason}")
                
                # Warn if hit max_tokens
                if finish_reason == "length":
                    logger.warning(f"⚠️ Generation hit max_tokens limit ({payload['max_tokens']})")
                    logger.warning(f"   Consider increasing max_tokens if response seems truncated")
            
            # Store response for later use
            self._last_response = result
            
            return result
            
        except httpx.HTTPStatusError as e:
            # Try to get detailed error message
            error_details = e.response.text
            try:
                error_json = e.response.json()
                if "error" in error_json:
                    error_details = json.dumps(error_json["error"], indent=2)
                elif "detail" in error_json:
                    error_details = error_json["detail"]
                else:
                    error_details = json.dumps(error_json, indent=2)
            except:
                pass  # Use text if JSON parsing fails
            
            logger.error(f"SGLang API error: {e.response.status_code}")
            logger.error(f"Error details: {error_details}")
            logger.error(f"Request URL: {e.request.url}")
            
            # Re-raise with more context
            raise ValueError(f"SGLang API error ({e.response.status_code}): {error_details}") from e
            
        except Exception as e:
            logger.error(f"SGLang API call failed: {type(e).__name__}: {e}")
            raise
    
    def extract_message_content(self, response: Any) -> Optional[str]:
        """Extract text content from sglang response."""
        try:
            # SGLang returns OpenAI-compatible format
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    
                    # Remove tool call markers from the content
                    # Keep <think> tags but remove <tool_call> tags
                    clean_content = self._remove_tool_call_markers(content)
                    
                    return clean_content if clean_content else None
            
            return None
        except Exception as e:
            logger.error(f"Error extracting message content: {e}")
            return None
    
    def has_tool_calls(self, response: Any) -> bool:
        """Check if sglang response has tool calls."""
        try:
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    
                    # Check for <tool_call> tags in content
                    return "<tool_call>" in content#bool(self.tool_call_pattern.search(content))
            
            return False
        except Exception as e:
            logger.error(f"Error checking for tool calls: {e}")
            return False
    
    def extract_tool_calls(self, response: Any) -> List[Dict]:
        """Extract tool calls from sglang response."""
        try:
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    
                    return self._parse_tool_calls_from_text(content)
            
            return []
        except Exception as e:
            logger.error(f"Error extracting tool calls: {e}")
            return []
    
    def format_tool_result_message(self, tool_call_id: str, result: str, role: str = None) -> Dict:
        """
        Format tool result for sglang.
        
        Tool results are formatted as user messages with <tool_response> wrapper.
        """
        return {
            "role": "user",
            "content": f"<tool_response>\n{result}\n</tool_response>",
            "tool_call_id": tool_call_id  # Keep for tracking
        }
    
    def format_image_message(self, text: str, image: Image.Image, role: str = "user") -> Dict:
        """
        Format an image message for sglang/Qwen.
        
        Qwen2.5-VL uses OpenAI-style image format with base64 encoding.
        """
        # Encode image to base64
        image_base64 = self._encode_image_to_base64(image)
        
        # Format as OpenAI-style content with image
        content = [
            {
                "type": "text",
                "text": text
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
        ]
        
        return {
            "role": role,
            "content": content
        }
    
    def add_message_to_history(self, messages: List[Dict], response: Any) -> List[Dict]:
        """Add sglang response to message history."""
        try:
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    
                    # Add assistant message with full content (including tool calls)
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })
            
            return messages
        except Exception as e:
            logger.error(f"Error adding message to history: {e}")
            return messages
    
    @property
    def supports_system_messages(self) -> bool:
        """SGLang/Qwen supports system messages (via prepending to first user message)."""
        return True
    
    @property
    def max_image_size(self) -> Optional[Tuple[int, int]]:
        """Qwen2.5-VL supports images up to reasonable dimensions."""
        return (4096, 4096)  # Conservative limit

