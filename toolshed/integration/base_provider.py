# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Abstract base class for LLM provider implementations.

This module defines the interface that all LLM providers must implement
to work with the Toolshed integration system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image


class BaseLLMProvider(ABC):
    """Abstract base class for LLM provider implementations."""
    
    @abstractmethod
    def create_client(self, **kwargs):
        """Create and return the provider-specific client."""
        pass
    
    @abstractmethod
    def format_tools(self, tool_schemas: List[Dict]) -> List[Dict]:
        """
        Convert toolshed schemas to provider-specific format.
        
        Args:
            tool_schemas: List of tool schemas in OpenAI format
            
        Returns:
            List of tool schemas in provider-specific format
        """
        pass
    
    @abstractmethod
    def format_messages(self, messages: List[Dict], images: List[Image.Image]) -> List[Dict]:
        """
        Format messages for the provider's API.
        
        Args:
            messages: List of message dictionaries
            images: List of PIL images available in conversation
            
        Returns:
            List of formatted messages for the provider
        """
        pass
    
    @abstractmethod
    async def call_api_async(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Any:
        """
        Make an async API call to the provider.
        
        Args:
            messages: Formatted messages
            tools: Formatted tools
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Provider's response object
        """
        pass
    
    @abstractmethod
    def extract_message_content(self, response: Any) -> Optional[str]:
        """
        Extract text content from the response.
        
        Args:
            response: Provider's response object
            
        Returns:
            Text content or None
        """
        pass
    
    @abstractmethod
    def has_tool_calls(self, response: Any) -> bool:
        """
        Check if the response contains tool calls.
        
        Args:
            response: Provider's response object
            
        Returns:
            True if response has tool calls
        """
        pass
    
    @abstractmethod
    def extract_tool_calls(self, response: Any) -> List[Dict]:
        """
        Extract tool calls from the response.
        
        Args:
            response: Provider's response object
            
        Returns:
            List of tool call dictionaries with keys: id, name, arguments
        """
        pass
    
    @abstractmethod
    def format_tool_result_message(self, tool_call_id: str, result: str, role: str = None) -> Dict:
        """
        Format a tool result message.
        
        Args:
            tool_call_id: ID of the tool call
            result: Result string from tool execution
            role: Message role (provider-specific)
            
        Returns:
            Formatted message dictionary
        """
        pass
    
    @abstractmethod
    def format_image_message(self, text: str, image: Image.Image, role: str = "user") -> Dict:
        """
        Format a message containing an image.
        
        Args:
            text: Text content
            image: PIL Image
            role: Message role
            
        Returns:
            Formatted message dictionary
        """
        pass
    
    @abstractmethod
    def add_message_to_history(self, messages: List[Dict], response: Any) -> List[Dict]:
        """
        Add the provider's response to message history.
        
        Args:
            messages: Current message history
            response: Provider's response object
            
        Returns:
            Updated message history
        """
        pass
    
    @property
    @abstractmethod
    def supports_system_messages(self) -> bool:
        """Whether the provider supports system messages."""
        pass
    
    @property
    @abstractmethod
    def max_image_size(self) -> Optional[Tuple[int, int]]:
        """Maximum image dimensions supported by the provider."""
        pass

