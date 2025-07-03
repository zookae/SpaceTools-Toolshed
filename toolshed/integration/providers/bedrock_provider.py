# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
AWS Bedrock provider implementation for Toolshed LLM integration.

This provider supports the NVIDIA LLM Gateway that uses AWS Bedrock as an interface
with simple Bearer token authentication.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
import logging

import httpx

from .anthropic_provider import AnthropicProvider

logger = logging.getLogger(__name__)

from anthropic import AnthropicBedrock
_ANTHROPIC_BEDROCK_AVAILABLE = True


class NvidiaLLMGatewayBedrockClient(AnthropicBedrock):
    """
    Custom AnthropicBedrock client for NVIDIA LLM Gateway with Bearer token auth.
    """
    
    def __init__(
        self,
        *,
        base_url: str | httpx.URL | None = None,
        bearer_token: str | None = None,
        default_headers: Dict[str, str] | None = None,
        **kwargs
    ):
        # Initialize with dummy AWS credentials since they're required
        # but won't be used with our custom authentication
        super().__init__(
            aws_secret_key="dummy",
            aws_access_key="dummy",
            aws_region="us-west-2",
            base_url=base_url,
            default_headers=default_headers,
            **kwargs
        )
        self.bearer_token = bearer_token
    
    def _prepare_request(self, request: httpx.Request) -> None:
        """
        Override the AWS auth preparation to use Bearer token instead.
        
        Args:
            request: The HTTP request to prepare
        """
        # Add our custom headers
        if self.bearer_token:
            request.headers["Authorization"] = f"Bearer {self.bearer_token}"
            request.headers["dataClassification"] = "confidential"
            request.headers["correlationId"] = str(os.urandom(16).hex())


class BedrockProvider(AnthropicProvider):
    """
    AWS Bedrock provider with NVIDIA LLM Gateway support using static Bearer token.
    """
    
    def __init__(self, 
                 model: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
                 **kwargs):
        """
        Initialize Bedrock provider.
        
        Args:
            model: Bedrock model ID (e.g., "us.anthropic.claude-sonnet-4-20250514-v1:0")
            **kwargs: Additional configuration passed from create_client
        """
        if not _ANTHROPIC_BEDROCK_AVAILABLE:
            raise ImportError(
                "Bedrock provider requires the 'anthropic' package with bedrock support. "
                "Install it with: pip install anthropic[bedrock]"
            )
        
        self.model = model
        self.client = None
        self.bearer_token = None
        self._config = kwargs  # Store config for later use in create_client
        
        # Store the last response for message history
        self._last_response = None
    
    def create_client(self, **kwargs):
        """
        Create custom Bedrock client with bearer token auth.
        
        Args:
            **kwargs: Client initialization parameters including:
                - bearer_token: The static bearer token for authentication (defaults to LLM_GATEWAY_TOKEN env var)
                - base_url: LLM Gateway base URL (default: https://prod.api.nvidia.com/llm/v1/aws)
                - enable_thinking: Whether to enable thinking mode by default (default: True)
                - thinking_budget_tokens: Token budget for thinking mode (default: 12000)
            
        Returns:
            Configured NvidiaLLMGatewayBedrockClient instance
        """
        # Extract configuration from kwargs and stored config
        config = {**self._config, **kwargs}
        
        # Extract bearer token - check argument first, then env var
        self.bearer_token = config.get('bearer_token') or os.getenv('LLM_GATEWAY_TOKEN')
        if not self.bearer_token:
            raise ValueError(
                "bearer_token is required for Bedrock authentication. "
                "Pass it as an argument or set the LLM_GATEWAY_TOKEN environment variable."
            )
        
        # Extract other configuration
        self.base_url = config.get('base_url', 'https://prod.api.nvidia.com/llm/v1/aws')
        self.enable_thinking = config.get('enable_thinking', True)
        self.thinking_budget_tokens = config.get('thinking_budget_tokens', 16000)
        
        # Create custom client (filter out our custom config keys)
        client_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['bearer_token', 'base_url', 'enable_thinking', 'thinking_budget_tokens']}
        
        self.client = NvidiaLLMGatewayBedrockClient(
            base_url=self.base_url,
            bearer_token=self.bearer_token,
            **client_kwargs
        )
        
        logger.info(f"Created Bedrock client with base URL: {self.base_url}")
        logger.info(f"Thinking mode: {'enabled' if self.enable_thinking else 'disabled'} "
                   f"(budget: {self.thinking_budget_tokens} tokens)")
        return self.client
    
    def _sanitize_for_logging(self, data: Any, max_length: int = 100) -> Any:
        """
        Sanitize data for logging by truncating large strings (like base64 images).
        
        Args:
            data: Data to sanitize
            max_length: Maximum string length before truncation
            
        Returns:
            Sanitized copy of data
        """
        if isinstance(data, str):
            if len(data) > max_length:
                return f"<TRUNCATED length={len(data)}>{data[:max_length]}..."
            return data
        elif isinstance(data, dict):
            return {k: self._sanitize_for_logging(v, max_length) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_for_logging(item, max_length) for item in data]
        else:
            return data
    
    async def call_api_async(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Any:
        """
        Call Bedrock API asynchronously with thinking mode support.
        
        Args:
            messages: Formatted messages
            tools: Formatted tools
            **kwargs: Additional parameters including thinking mode config
            
        Returns:
            Bedrock response object
        """
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
            "max_tokens": kwargs.get('max_tokens', 24000)  # Must be > thinking budget_tokens (16000)
        }
        
        # Add tools if provided
        if tools:
            api_params["tools"] = tools
        
        # Add system message if provided
        if system_message:
            api_params["system"] = system_message
        
        # Handle thinking mode
        thinking_config = kwargs.get('thinking_mode')
        if thinking_config is None and self.enable_thinking:
            # Use default thinking configuration
            thinking_config = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens
            }
        
        if thinking_config:
            # NVIDIA LLM Gateway uses standard Anthropic API format for thinking mode
            # Add thinking configuration directly to the API parameters
            api_params["thinking"] = thinking_config
            logger.debug(f"Enabling thinking mode with config: {thinking_config}")
        
        # Log sanitized request parameters (truncate large strings like images)
        sanitized_params = self._sanitize_for_logging(api_params)
        #logger.info(f"Bedrock API request parameters: {json.dumps(sanitized_params, indent=2)}")
        
        # Retry mechanism: try up to 3 times with 5 second waits
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # TODO: This was commented out to try without thinking api.
                # Use streaming if thinking mode is enabled to avoid timeout issues
                if thinking_config:
                    # Make streaming API call for thinking mode
                    stream = await loop.run_in_executor(
                        None,
                        lambda: self.client.messages.stream(**api_params)
                    )
                    
                    # Collect the full response from the stream
                    response = await loop.run_in_executor(None, self._collect_stream_response, stream)
                else:
                    # Make regular async API call for non-thinking requests
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.client.messages.create(**api_params)
                    )
                
                # Make regular async API call for non-thinking requests
                # response = await loop.run_in_executor(
                #     None,
                #     lambda: self.client.messages.create(**api_params)
                # )

                # Store response for later use
                self._last_response = response
                return response
                
            except Exception as e:
                # Log detailed error information
                is_last_attempt = (attempt == max_attempts - 1)
                logger.error(f"Bedrock API call failed (attempt {attempt + 1}/{max_attempts})")
                logger.error(f"Exception type: {type(e).__name__}")
                # Truncate exception message to avoid dumping huge request data with images
                exception_str = str(e)
                if len(exception_str) > 500:
                    logger.error(f"Exception message (truncated): {exception_str[:500]}...")
                else:
                    logger.error(f"Exception message: {exception_str}")
                
                # Log all available exception attributes
                exception_attrs = [attr for attr in dir(e) if not attr.startswith('_')]
                logger.error(f"Available exception attributes: {exception_attrs}")
                
                # Try to extract detailed error info from common attributes
                if hasattr(e, 'response'):
                    logger.error(f"Response status: {getattr(e.response, 'status_code', 'N/A')}")
                    response_text = getattr(e.response, 'text', None)
                    if response_text:
                        # Truncate response body if too long
                        if len(response_text) > 1000:
                            logger.error(f"Response body (truncated): {response_text[:1000]}...")
                        else:
                            logger.error(f"Response body: {response_text}")
                
                if hasattr(e, 'status_code'):
                    logger.error(f"Status code: {e.status_code}")
                
                if hasattr(e, 'request_id'):
                    logger.error(f"Request ID: {e.request_id}")
                
                #if hasattr(e, 'message'):
                #    logger.error(f"Error message attribute: {e.message}")
                #if hasattr(e, 'body'):
                #    logger.error(f"Error body: {e.body}")
                
                if is_last_attempt:
                    # No more retries, create a sanitized exception to avoid dumping huge request data
                    logger.error(f"Exceeded max retries. Raising sanitized exception.")
                    
                    # Create a clean error message without the full request/response data
                    error_msg = (
                        f"Bedrock API call failed after {max_attempts} attempts. "
                        f"Exception: {type(e).__name__}: {str(e)[:200]}"
                    )
                    
                    # Add request ID if available
                    if hasattr(e, 'request_id'):
                        error_msg += f" (request_id: {e.request_id})"
                    
                    # Raise a new RuntimeError instead of the original exception
                    # This prevents the anthropic exception from dumping full request data
                    raise RuntimeError(error_msg)
                else:
                    # Wait 5 seconds before retrying
                    logger.info(f"Retrying in 5 seconds...")
                    await asyncio.sleep(5)
    
    def _collect_stream_response(self, stream):
        """
        Collect the full response from a streaming API call.
        
        Args:
            stream: The streaming response from the client
            
        Returns:
            Complete response object
        """
        # Use the stream context manager to get the final message
        with stream as stream_context:
            # The stream will automatically collect all chunks
            return stream_context.get_final_message()
    
    def extract_message_content(self, response: Any) -> Optional[str]:
        """
        Extract text content from Bedrock response, handling thinking blocks.
        
        Args:
            response: Bedrock response object
            
        Returns:
            Text content, with thinking content as fallback if no text blocks present
        """
        # Bedrock responses have the same structure as Anthropic
        if hasattr(response, 'content') and response.content:
            # Extract text from content blocks, and collect thinking blocks as fallback
            text_parts = []
            thinking_parts = []
            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == 'text':
                        text_parts.append(block.text)
                    elif block.type == 'thinking':
                        # Collect thinking content as fallback
                        thinking_text = getattr(block, 'thinking', getattr(block, 'text', None))
                        if thinking_text:
                            thinking_parts.append(thinking_text)
                            logger.debug(f"Thinking block: {thinking_text[:200]}...")
            
            # Return text content if available, otherwise return thinking content as fallback
            if text_parts:
                return '\n'.join(text_parts)
            elif thinking_parts:
                logger.warning("Response contains only thinking blocks, no text content. Returning thinking content as fallback.")
                return '\n'.join(thinking_parts)
            else:
                return None
        
        return None
    
    @property
    def supports_thinking_mode(self) -> bool:
        """Whether this provider supports thinking mode."""
        return True
    
    def get_thinking_content(self, response: Any) -> Optional[str]:
        """
        Extract thinking content from response if available.
        
        Args:
            response: Bedrock response object
            
        Returns:
            Thinking content or None
        """
        if hasattr(response, 'content') and response.content:
            thinking_parts = []
            for block in response.content:
                if hasattr(block, 'type') and block.type == 'thinking':
                    if hasattr(block, 'text'):
                        thinking_parts.append(block.text)
            
            return '\n'.join(thinking_parts) if thinking_parts else None
        
        return None
