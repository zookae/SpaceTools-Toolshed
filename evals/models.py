# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Model wrappers for evaluation framework.

This module provides a unified interface for different model implementations,
allowing easy evaluation across various model types.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for models to be evaluated."""
    
    @abstractmethod
    def predict(self, text: str, images: Optional[List[Image.Image]] = None, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response for the given input.
        
        Args:
            text: The input text/prompt
            images: Optional list of PIL images
            system_prompt: Optional system prompt to use
            
        Returns:
            The model's text response
        """
        pass
    
    def get_conversation_history(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get the conversation history in message format.
        
        Returns:
            List of messages or None if not tracked
        """
        return None


class UnifiedModel(BaseModel):
    """Unified model wrapper around ToolAgent for all providers."""
    
    def __init__(self, 
                 toolkit=None,
                 provider: str = "openai", 
                 model: Optional[str] = None,
                 enable_variables: bool = True,
                 max_iterations: int = 5,
                 use_web_demo_prompt: bool = False,
                 **provider_kwargs):
        """
        Initialize unified model with LLM integration.
        
        Args:
            toolkit: Optional pre-initialized toolkit. If None, will get the default toolkit.
            provider: LLM provider ("openai", "anthropic", "bedrock")
            model: Model name (provider-specific default if None)
            enable_variables: Whether to enable variable handling
            max_iterations: Maximum number of tool call iterations
            use_web_demo_prompt: Whether to use the same system prompt as the web demo
            **provider_kwargs: Additional provider-specific parameters
        """
        from toolshed.agent import create_tool_agent
        from toolshed import get_toolkit
        
        self.provider = provider
        self.model_name = model
        self.max_iterations = max_iterations
        self.use_web_demo_prompt = use_web_demo_prompt
        
        # Handle toolkit initialization
        if toolkit is None:
            # For "no tools" mode, we'll pass None to the integration
            # The integration will handle this gracefully
            self.toolkit = None
        else:
            self.toolkit = toolkit
            
        # Initialize LLM integration
        self.integration = create_tool_agent(
            toolkit=self.toolkit,
            provider=provider,
            model=model,
            enable_variables=enable_variables,
            **provider_kwargs
        )
        
        # Prepare system prompt if using web demo prompt
        self.system_prompt = None
        if use_web_demo_prompt:
            from toolshed.prompts.coordinate_conventions import COORDINATE_CONVENTIONS_PROMPT
            self.system_prompt = f"""You are an AI assistant with access to powerful computer vision tools. Your role is to help users understand and analyze images by using the available tools to examine spatial relationships, detect objects, estimate depth, segment regions, and execute code when needed.

When a user asks questions about images, you should:
1. Analyze what information is needed to answer their question
2. Before calling tools, explain to the user your thought process so far and how you are approaching the problem.
3. Use the appropriate vision tools to gather spatial and visual information
4. Combine the results from multiple tools if needed to provide comprehensive answers
5. Explain your findings clearly, referencing specific locations and relationships in the scene
6. After you explanation, write "Final Answer:" and provie a concise answer to the user's question.

{COORDINATE_CONVENTIONS_PROMPT}

Select the appropriate frame to do your reasoning (2D or 3D). For example, 3D coordinates are perspective invariant and may better capture object sizes and locations in space. 2D coordinates may be sufficient to answer simpler questions about objects at a similar distance.
Always strive to provide accurate, detailed analysis of the spatial relationships and visual content in the images."""
        
        logger.info(f"Initialized unified model: {provider}/{model or 'default'} (tools={'enabled' if self.toolkit else 'disabled'})")
        
        # Store the current session_id to track conversation
        self.current_session_id = None
    
    def predict(
        self, 
        text: str, 
        images: Optional[List[Image.Image]] = None, 
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response using the LLM integration.
        
        Args:
            text: Input text/question for the model
            images: Optional list of images to include in the conversation
            system_prompt: Optional system prompt to set the model's behavior
        
        Returns:
            The model's text response
        """
        # Create a unique session for this prediction to track conversation history
        import uuid
        self.current_session_id = f"eval_{uuid.uuid4()}"
        
        try:
            # Create session
            session = self.integration.create_session(
                self.current_session_id, 
                initial_images=[]
            )
            
            # Add system prompt if provided
            if system_prompt:
                session.add_system_message(system_prompt)
            elif self.system_prompt:
                session.add_system_message(self.system_prompt)
            
            # Add user message with images
            session.add_user_message(text, images)
            
            # Get response using the unified API
            response_data = self.integration.get_response(self.current_session_id, max_iterations=self.max_iterations)
            
            return response_data["response"]
        except Exception as e:
            # Clean up on error
            if self.current_session_id and self.current_session_id in self.integration.sessions:
                self.integration.clear_session(self.current_session_id)
            self.current_session_id = None
            raise
    
    def get_conversation_history(self) -> Optional[List[Dict[str, Any]]]:
        """Get the conversation history from the integration's session."""
        if self.current_session_id and self.current_session_id in self.integration.sessions:
            session_history = self.integration.get_session_history(self.current_session_id)
            messages = session_history["messages"]
            logger.debug(f"Retrieved conversation history: {len(messages)} messages")
            for i, msg in enumerate(messages):
                logger.debug(f"  Message {i}: role={msg.get('role', 'unknown')}, content_type={type(msg.get('content', ''))}")
            # Don't clear the session here - let the integration handle cleanup
            # The session will be cleared when the next prediction starts
            return messages
        return None


# Legacy model classes for backward compatibility
class OpenAIToolshedModel(UnifiedModel):
    """Legacy wrapper for OpenAI models using Toolshed integration."""
    
    def __init__(self, model_name: str = "gpt-4o", toolkit=None, use_web_demo_prompt: bool = False, max_iterations: int = 5):
        """Initialize OpenAI model with Toolshed integration (legacy interface)."""
        super().__init__(
            toolkit=toolkit,
            provider="openai",
            model=model_name,
            enable_variables=True,
            max_iterations=max_iterations,
            use_web_demo_prompt=use_web_demo_prompt
        )


class SimpleOpenAIModel(UnifiedModel):
    """Legacy wrapper for OpenAI models without tools (for comparison)."""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """Initialize OpenAI model without Toolshed (legacy interface)."""
        super().__init__(
            toolkit=None,  # No toolkit = no tools
            provider="openai",
            model=model_name,
            enable_variables=False,
            max_iterations=1  # No tool calls needed
        )
