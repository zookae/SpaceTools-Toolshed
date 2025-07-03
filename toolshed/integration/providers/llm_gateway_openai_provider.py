# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
NVIDIA LLM Gateway provider implementation for OpenAI models.

This provider supports OpenAI models served through the NVIDIA LLM Gateway
with Bearer token authentication.
"""

import os
import logging
from typing import Dict, Any

from .openai_provider import OpenAIProvider

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = None

logger = logging.getLogger(__name__)


class LLMGatewayOpenAIProvider(OpenAIProvider):
    """
    NVIDIA LLM Gateway provider for OpenAI models with Bearer token authentication.
    
    This provider extends OpenAIProvider to work with the NVIDIA LLM Gateway,
    which serves OpenAI models through a custom gateway endpoint.
    """
    
    def __init__(self, model: str = "gpt-4o", **kwargs):
        """
        Initialize LLM Gateway OpenAI provider.
        
        Args:
            model: OpenAI model to use (e.g., "gpt-4o", "gpt-4o-mini", "gpt-4.1-nano")
            **kwargs: Additional configuration passed from create_client
        """
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "LLM Gateway OpenAI provider requires the 'openai' package. "
                "Install it with: pip install openai"
            )
        
        super().__init__(model=model)
        self._config = kwargs  # Store config for later use in create_client
    
    def create_client(self, **kwargs):
        """
        Create OpenAI client configured for NVIDIA LLM Gateway.
        
        Args:
            **kwargs: Client initialization parameters including:
                - api_key: The Bearer token for authentication (defaults to LLM_GATEWAY_TOKEN env var)
                - base_url: LLM Gateway base URL (default: https://prod.api.nvidia.com/llm/v1/azure/v1)
                - default_query: Query parameters to include (default: {"api-version": "preview"})
        
        Returns:
            Configured OpenAI client instance
        """
        # Extract configuration from kwargs and stored config
        config = {**self._config, **kwargs}
        
        # Extract API key - check argument first, then env var
        api_key = config.get('api_key') or os.getenv('LLM_GATEWAY_TOKEN')
        if not api_key:
            raise ValueError(
                "api_key is required for LLM Gateway authentication. "
                "Pass it as an argument or set the LLM_GATEWAY_TOKEN environment variable."
            )
        
        # Extract other configuration
        base_url = config.get('base_url', 'https://prod.api.nvidia.com/llm/v1/azure/v1')
        default_query = config.get('default_query', {"api-version": "preview"})
        
        # Filter out our custom config keys before passing to OpenAI client
        client_kwargs = {k: v for k, v in config.items() 
                        if k not in ['api_key', 'base_url', 'default_query']}
        
        # Create OpenAI client with LLM Gateway configuration
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_query=default_query,
            **client_kwargs
        )
        
        logger.info(f"Created LLM Gateway OpenAI client with base URL: {base_url}")
        logger.info(f"Model: {self.model}")
        return self.client

