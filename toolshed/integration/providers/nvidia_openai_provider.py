# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""NVIDIA Inference API provider using the OpenAI-compatible chat API."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List

from .openai_provider import OpenAIProvider

try:
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = None

logger = logging.getLogger(__name__)


class NvidiaOpenAIProvider(OpenAIProvider):
    """OpenAI-compatible provider for https://inference-api.nvidia.com."""

    def __init__(self, model: str = "openai/openai/gpt-5.5", **kwargs: Any):
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "NVIDIA OpenAI provider requires the 'openai' package. "
                "Install it with: pip install openai"
            )

        super().__init__(model=model)
        self._config = kwargs

    def create_client(self, **kwargs: Any):
        """Create an OpenAI client configured for NVIDIA Inference API."""
        config = {**self._config, **kwargs}
        api_key = (
            config.get("api_key")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("LLM_GATEWAY_TOKEN")
        )
        if not api_key:
            raise ValueError(
                "api_key is required for NVIDIA Inference API authentication. "
                "Pass it as an argument or set NVIDIA_API_KEY."
            )

        base_url = config.get("base_url", "https://inference-api.nvidia.com/v1")
        client_kwargs = {
            key: value for key, value in config.items() if key not in {"api_key", "base_url"}
        }
        self.client = OpenAI(api_key=api_key, base_url=base_url, **client_kwargs)
        logger.info("Created NVIDIA Inference API OpenAI client with base URL: %s", base_url)
        logger.info("Model: %s", self.model)
        return self.client

    async def call_api_async(self, messages: List[Dict], tools: List[Dict], **kwargs: Any) -> Any:
        """Call NVIDIA's OpenAI-compatible chat completions endpoint."""
        loop = asyncio.get_event_loop()
        max_tokens = kwargs.get("max_tokens", kwargs.get("max_completion_tokens", 4096))
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            api_kwargs["tools"] = tools
            api_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")
        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            api_kwargs["top_p"] = kwargs["top_p"]

        return await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(**api_kwargs),
        )
