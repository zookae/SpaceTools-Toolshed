# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Unified LLM Integration for Toolshed

This module provides a unified interface for integrating multiple LLM providers
(OpenAI, Anthropic, etc.) with Toolshed's computer vision tools.
"""

import json
import logging
import base64
import io
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path

import ray
from PIL import Image

from toolshed.tool_result import ToolResult
from toolshed.variable_engine import VariableEngine
from toolshed.integration.base_provider import BaseLLMProvider
from toolshed.prompts.variables import get_variable_prompt
from toolshed import agent_step_types as ast

logger = logging.getLogger(__name__)


class ConversationSession:
    """
    Self-contained conversation session with all state and operations.
    
    Manages messages, images, tool-generated images, and variables in one place.
    """
    
    def __init__(self, 
                 session_id: str,
                 initial_images: Optional[List[Image.Image]] = None,
                 enable_variables: bool = True,
                 variable_instructions: Optional[str] = None,
                 provider: Optional[BaseLLMProvider] = None):
        """
        Initialize a conversation session.
        
        Args:
            session_id: Unique identifier for the session
            initial_images: Optional list of initial images for the session
            enable_variables: Whether to enable variable handling (default: True)
            variable_instructions: Optional variable handling prompt to append to system messages.
                If None and enable_variables is True, no variable instructions are added.
            provider: LLM provider instance for message formatting (optional)
        """
        self.session_id = session_id
        self.messages: List[Dict[str, Any]] = []
        self.images: List[Image.Image] = initial_images or []
        self.tool_generated_images: List[Dict[str, Any]] = []
        self.created_at = datetime.datetime.now().isoformat()
        self.provider = provider
        self.variable_instructions = variable_instructions if enable_variables else None
        
        # Embed variable engine directly in session
        if enable_variables:
            self.variable_engine = VariableEngine()
        else:
            self.variable_engine = None
    
    def add_user_message(self, text: str, images: Optional[List[Image.Image]] = None):
        """Add a user message to this session."""
        # Add new images to session
        if images:
            self.images.extend(images)
        
        # Create message content
        if images:
            # Use provider's format_image_message for the first image, then handle additional images
            if len(images) == 1 and self.provider:
                # Single image case - use provider's format_image_message
                message_dict = self.provider.format_image_message(text, images[0], role="user")
                content = message_dict["content"]
            else:
                # Multiple images case - we'll add them as separate messages after this one
                # For now, just add the text
                content = text
        else:
            content = text
        
        # Debug logging
        logger.debug(f"Adding user message to session {self.session_id}")
        logger.debug(f"Message text: {text[:200]}...")
        if images:
            logger.debug(f"Including {len(images)} image(s)")
        
        self.messages.append({
            "role": "user",
            "content": content
        })
        
        # Handle additional images if there were multiple
        if images and len(images) > 1 and self.provider:
            for i, image in enumerate(images[1:], 2):  # Start from image 2
                image_message = self.provider.format_image_message(
                    f"Additional image {i}/{len(images)}:",
                    image,
                    role="user"
                )
                self.messages.append(image_message)
    
    def add_system_message(self, text: str):
        """Add a system message to this session.
        
        If variable_instructions is set, automatically appends the variable
        handling prompt to teach the LLM how to use stored variables.
        """
        if self.variable_instructions:
            text = text + "\n\n" + self.variable_instructions
        
        if self.provider and self.provider.supports_system_messages:
            self.messages.insert(0, {
                "role": "system",
                "content": text
            })
    
    def get_history(self) -> Dict[str, Any]:
        """Get the full conversation history for this session."""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "num_images": len(self.images)
        }
    
    def add_tool_generated_image(self, image: Image.Image, tool_name: str):
        """
        Add a tool-generated image to this session.
        
        Args:
            image: Generated image
            tool_name: Name of the tool that generated the image
        """
        self.tool_generated_images.append({
            "image": image,
            "tool_name": tool_name,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def get_tool_generated_images(self) -> List[Dict[str, Any]]:
        """Get all tool-generated images for this session."""
        return self.tool_generated_images
    
    def store_variables(self, variables: Dict[str, Any]):
        """
        Store variables in this session.
        
        Args:
            variables: Dictionary of variable names and values to store
        """
        if self.variable_engine:
            self.variable_engine.store_variables(variables)
    
    def resolve_variables(self, data: Any) -> Any:
        """
        Resolve variable references in data structures.
        
        Args:
            data: Data structure that may contain variable references
            
        Returns:
            Data structure with variables resolved
        """
        if self.variable_engine:
            return self.variable_engine.resolve_variables(data)
        return data
    
    def get_variables(self) -> Dict[str, Any]:
        """
        Get all variables for this session.
        
        Returns:
            Dictionary of variable name -> value
        """
        if self.variable_engine:
            return self.variable_engine.export_variables()
        return {}
    
    def get_variable_stats(self) -> Dict[str, Any]:
        """
        Get variable statistics for this session.
        
        Returns:
            Dictionary with variable statistics
        """
        if self.variable_engine:
            return self.variable_engine.get_stats()
        return {"total_variables": 0, "variable_names": [], "variable_types": {}}
    
    def clear(self):
        """Clear all session data (messages, images, variables)."""
        self.messages.clear()
        self.images.clear()
        self.tool_generated_images.clear()
        if self.variable_engine:
            self.variable_engine.clear_variables()


class ToolAgent:
    """Unified integration for multiple LLM providers with Toolshed vision tools."""
    
    def __init__(self, 
                 toolkit,
                 provider: str = "openai",
                 model: Optional[str] = None,
                 client_kwargs: Optional[Dict] = None,
                 enable_variables: bool = True,
                 inject_variable_instructions: bool = True,
                 hide_tool_images: bool = False,
                 keep_dots_in_tool_names: bool = False):
        """
        Initialize the integration.
        
        Args:
            toolkit: ToolkitClient instance (should already be connected)
            provider: Provider name ("openai", "llm_gateway_openai", "anthropic", "bedrock", or "sglang")
            model: Model to use (provider-specific default if None)
            client_kwargs: Additional kwargs for provider client initialization
            enable_variables: Whether to enable variable handling (default: True)
            inject_variable_instructions: Whether to auto-append variable handling instructions
                to system messages (default: True). Set to False if you manage your own
                system prompt that already includes variable instructions.
            hide_tool_images: Whether to hide tool output images from LLM (default: False)
            keep_dots_in_tool_names: Whether to keep dots in tool names (e.g., "vlm.detect_one")
                                    instead of converting to double underscores ("vlm__detect_one").
                                    Default: False (convert to double underscores for OpenAI compatibility)
        """
        self.toolkit = toolkit
        self.provider_name = provider.lower()
        self.enable_variables = enable_variables
        self.inject_variable_instructions = inject_variable_instructions
        self.hide_tool_images = hide_tool_images
        self.keep_dots_in_tool_names = keep_dots_in_tool_names
        
        # Initialize provider
        if self.provider_name == "openai":
            from toolshed.integration.providers.openai_provider import OpenAIProvider
            default_model = "gpt-4o"
            self.provider = OpenAIProvider(model or default_model)
        elif self.provider_name == "llm_gateway_openai":
            from toolshed.integration.providers.llm_gateway_openai_provider import LLMGatewayOpenAIProvider
            default_model = "gpt-4o"
            self.provider = LLMGatewayOpenAIProvider(model or default_model)
        elif self.provider_name == "anthropic":
            from toolshed.integration.providers.anthropic_provider import AnthropicProvider
            default_model = "claude-opus-4-1-20250805"
            self.provider = AnthropicProvider(model or default_model)
        elif self.provider_name == "bedrock":
            from toolshed.integration.providers.bedrock_provider import BedrockProvider
            default_model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
            self.provider = BedrockProvider(model or default_model)
        elif self.provider_name == "sglang":
            from toolshed.integration.providers.sglang_provider import SglangProvider
            default_model = "Qwen2.5-VL-3B-Instruct"
            # Extract base_url from client_kwargs if provided
            base_url = (client_kwargs or {}).get('base_url', 'http://localhost:30000/v1')
            self.provider = SglangProvider(model or default_model, base_url=base_url)
        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported providers: openai, llm_gateway_openai, anthropic, bedrock, sglang"
            )
        
        # Create client
        self.provider.create_client(**(client_kwargs or {}))
        
        # Tool schemas and mappings - only store raw format
        self.raw_tool_schemas = []      # Current filtered raw schemas
        self.all_raw_tool_schemas = []  # All raw schemas before filtering
        self.function_map = {}
        self.tool_filter = None  # Optional function to filter tools
        
        # Session management
        self.sessions: Dict[str, ConversationSession] = {}
        
        # Generate schemas immediately after initialization
        self.generate_schemas()
        
        # Determine appropriate variable prompt based on available tools
        self._variable_prompt = self._compute_variable_prompt()
        
        logger.info(f"Initialized {self.provider_name} integration with model {model or default_model}")
    
    def _get_provider_tools(self) -> List[Dict]:
        """Convert current raw schemas to provider format."""
        return self.provider.format_tools(self.raw_tool_schemas)
    
    def _convert_to_provider_format(self, raw_schemas: List[Dict]) -> List[Dict]:
        """Convert given raw schemas to provider format."""
        return self.provider.format_tools(raw_schemas)
        
    def generate_schemas(self):
        """Generate schemas from toolshed tools."""
        # Handle no-tools mode
        if self.toolkit is None:
            logger.info("No toolkit provided, skipping schema generation (no-tools mode)")
            self.all_raw_tool_schemas = []
            self.raw_tool_schemas = []
            self.function_map = {}
            return
        
        logger.info("Generating schemas from toolshed tools...")
        
        # Get schemas from toolshed (in OpenAI format)
        schema_configs = self.toolkit.export_openai_schemas(
            use_image_by_index=True  # Use lightweight image references
        )
        
        # Process schemas and build function map
        processed_schemas = []
        self.function_map = {}
        
        for config in schema_configs:
            schema = config.get("tool_schema", {})
            if schema.get("type") == "function":
                # Get original function name from toolshed
                original_func_name = schema["function"]["name"]
                
                # Optionally convert dots to double underscores for compatibility
                if self.keep_dots_in_tool_names:
                    safe_func_name = original_func_name  # Keep as-is: "vlm.detect_one"
                else:
                    safe_func_name = original_func_name.replace(".", "__")  # Convert to: "vlm__detect_one"
                
                # Update schema with safe name
                schema["function"]["name"] = safe_func_name
                
                # Add to processed schemas
                processed_schemas.append(schema)
                
                # Map safe function name to toolshed method
                if "." in original_func_name:
                    tool_name, method_name = original_func_name.split(".", 1)
                    self.function_map[safe_func_name] = {
                        "tool_name": tool_name,
                        "method_name": method_name,
                        "original_name": original_func_name
                    }
        
        # Store only raw schemas
        self.all_raw_tool_schemas = processed_schemas.copy()
        self.raw_tool_schemas = processed_schemas.copy()  # Start with all tools
        
        # Apply filter if set
        self._apply_tool_filter()
        
        logger.debug("Generated raw_tool_schemas: %s", self.raw_tool_schemas)

        logger.info("Generated %d tool schemas (filtered: %d)", 
                   len(self.all_raw_tool_schemas), len(self.raw_tool_schemas))
        
        return self._get_provider_tools()  # Return provider format for backward compatibility
    
    def _has_code_executor(self) -> bool:
        """Check if code_executor tool is available."""
        return any("code_executor" in name for name in self.get_available_tools())
    
    def _compute_variable_prompt(self) -> Optional[str]:
        """Compute the appropriate variable prompt based on available tools."""
        if not self.inject_variable_instructions:
            return None
        return get_variable_prompt(has_code_executor=self._has_code_executor())
    
    def execute_function_call(self, function_call: Dict, images: List[Image.Image], 
                            session_id: Optional[str] = None, 
                            return_images: bool = False) -> Union[str, Tuple[str, Optional[Image.Image]]]:
        """
        Execute a function call through toolshed with variable resolution.
        
        Args:
            function_call: Function call dict with keys: name, arguments
            images: List of images from the conversation (for image_index resolution)
            session_id: Session ID for variable resolution (optional)
            return_images: If True, return tuple of (text_result, generated_image_or_none)
            
        Returns:
            String result from the function call, or tuple if return_images=True
        """

        logger.debug("Executing tool_call: %s", function_call)
        
        func_name = function_call["name"]
        arguments = json.loads(function_call["arguments"])
        
        # Log function call without full arguments (might contain base64 images)
        arg_keys = list(arguments.keys()) if isinstance(arguments, dict) else []
        logger.debug("Executing function call: %s with args: %s", func_name, arg_keys)
        
        # Handle potential mismatch between dots and double underscores
        if func_name not in self.function_map:
            # Try converting dots/underscores in case of format mismatch
            alt_func_name = func_name.replace(".", "__") if "." in func_name else func_name.replace("__", ".")
            if alt_func_name in self.function_map:
                logger.warning(f"Tool name format mismatch: model used '{func_name}' but expected '{alt_func_name}'. Using '{alt_func_name}'.")
                func_name = alt_func_name
            else:
                error_msg = f"Error: Unknown function {func_name}"
                return (error_msg, None) if return_images else error_msg
        
        mapping = self.function_map[func_name]
        tool_name = mapping["tool_name"]
        method_name = mapping["method_name"]
        
        # Get session if available
        session = None
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Resolve variable references if enabled
            if session.variable_engine:
                try:
                    logger.debug("Resolving variables in arguments: %s", arguments)
                    arguments = session.resolve_variables(arguments)
                    logger.debug("Arguments after variable resolution: %s", arguments)
                except KeyError as e:
                    error_msg = f"Variable resolution error: {e}"
                    logger.error(error_msg)
                    return (error_msg, None) if return_images else error_msg
        
        # Handle image_index parameter - convert to actual image
        if "image_index" in arguments:
            image_index = arguments.pop("image_index")
            if 0 <= image_index < len(images):
                # Put image in Ray object store for efficient passing
                arguments["image"] = ray.put(images[image_index])
            else:
                error_msg = f"Error: image_index {image_index} out of range (0-{len(images)-1})"
                return (error_msg, None) if return_images else error_msg
        
        try:
            result = self.toolkit.call_tool(tool_name, method_name, **arguments)

            # Store variables from result if enabled
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                if session.variable_engine and isinstance(result, ToolResult) and result.variables:
                    session.store_variables(result.variables)

            # Extract text and image results
            text_result, generated_image = self._extract_result_content(result)

            if return_images:
                return text_result, generated_image
            else:
                return text_result

        except Exception as e:
            logger.error("Error executing function %s.%s: %s", tool_name, method_name, e)
            logger.error("Arguments were: %s", arguments)
            error_msg = f"Error executing {tool_name}.{method_name}: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}"
            if return_images:
                return error_msg, None
            else:
                return error_msg
    
    async def get_response_async(self,
                                session_id: str,
                                max_iterations: int = 5,
                                track_steps: bool = False,
                                step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                                **provider_kwargs) -> Dict[str, Any]:
        """
        Unified async method for getting LLM responses with tool execution.
        
        Args:
            session_id: Session identifier
            max_iterations: Maximum number of tool call iterations
            track_steps: Whether to include detailed steps in the response
            step_callback: Optional callback for real-time step updates
            **provider_kwargs: Provider-specific parameters (e.g., thinking_mode for bedrock)
            
        Returns:
            Dictionary with:
                - response: The final text response
                - iterations: Number of iterations taken
                - tool_calls_made: Total number of tool calls
                - session_id: Session identifier
                - steps: (if track_steps=True) Array of step details
                - reasoning: (if track_steps=True) Array of reasoning content
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        # Auto-inject variable handling prompt if enabled and no system message exists
        if session.variable_instructions:
            has_system_message = any(msg.get("role") == "system" for msg in session.messages)
            if not has_system_message:
                # Add a minimal system message that will get the variable prompt appended
                session.add_system_message("You are a helpful assistant.")
        
        messages = session.messages
        images = session.images
        
        iteration = 0
        tool_calls_made = 0
        steps = []
        reasoning = []
        
        def emit_step(step):
            """Emit a step through callback and/or collect it."""
            if track_steps:
                steps.append(step)
            if step_callback:
                step_callback(step)
        
        while iteration < max_iterations:
            iteration += 1
            logger.debug("Response iteration %d/%d", iteration, max_iterations)
            
            # Format messages for provider
            provider_messages = self.provider.format_messages(messages, images)
            
            # Call provider API asynchronously
            # Pass original messages so provider can extract system message if needed
            provider_tools = self._get_provider_tools()
            response = await self.provider.call_api_async(
                provider_messages, 
                provider_tools,
                original_messages=messages,  # Let provider extract system message if needed
                **provider_kwargs
            )
            
            # Add response to message history
            messages = self.provider.add_message_to_history(messages, response)
            
            # Update session messages with the updated conversation
            session.messages = messages
            
            # Extract and log reasoning if present
            message_content = self.provider.extract_message_content(response)
            if message_content:
                logger.debug(f"Assistant response: {message_content[:300]}...")
                
                if track_steps or step_callback:
                    reasoning.append({
                        "iteration": iteration,
                        "content": message_content,
                        "timestamp": self._get_timestamp()
                    })
                    
                    emit_step(ast.make_reasoning_step(
                        iteration=iteration,
                        message=message_content,
                        timestamp=self._get_timestamp()
                    ))
            
            # Check if model wants to use tools
            if self.provider.has_tool_calls(response):
                tool_calls = self.provider.extract_tool_calls(response)
                logger.info(f"Model requested {len(tool_calls)} tool calls")
                logger.debug(f"Tool calls: {[tc['name'] for tc in tool_calls]}")
                tool_calls_made += len(tool_calls)
                
                # DEBUG: Log message history before tool execution
                logger.debug(f"Message history before tool execution ({len(messages)} messages):")
                for idx, msg in enumerate(messages[-3:]):  # Show last 3 messages
                    logger.debug(f"  [{idx-3}] Role: {msg.get('role', 'unknown')}, Content preview: {str(msg.get('content', ''))[:100]}...")
                
                # Emit tool decision step
                emit_step(ast.make_tool_decision_step(
                    iteration=iteration,
                    tool_calls=tool_calls,
                    timestamp=self._get_timestamp()
                ))
                
                # Execute each tool call and collect results
                tool_generated_images = []
                for i, tool_call in enumerate(tool_calls):
                    # Emit tool execution start
                    emit_step(ast.make_tool_executing_step(
                        iteration=iteration,
                        tool_index=i + 1,
                        tool_name=tool_call["name"],
                        arguments=tool_call["arguments"],
                        timestamp=self._get_timestamp()
                    ))
                    
                    # Execute tool (in executor to not block) with image support
                    text_result, generated_image = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.execute_function_call,
                        tool_call,
                        images,
                        session_id,
                        True  # Always get images
                    )
                    
                    # Handle generated image
                    if generated_image:
                        # Add image to session
                        session.add_tool_generated_image(generated_image, tool_call["name"])
                        # Add to conversation images for subsequent tool calls
                        images.append(generated_image)
                        # Collect for later addition to conversation
                        tool_generated_images.append(generated_image)
                    
                    # Log and emit tool result
                    logger.debug(f"Tool result: {text_result[:200]}...")
                    
                    # Build image_data if needed
                    image_data = None
                    if generated_image and (track_steps or step_callback):
                        image_data = self._image_to_base64(generated_image)
                    
                    emit_step(ast.make_tool_result_step(
                        iteration=iteration,
                        tool_index=i + 1,
                        tool_name=tool_call["name"],
                        result=text_result,
                        has_image=generated_image is not None,
                        timestamp=self._get_timestamp(),
                        image_data=image_data
                    ))
                    
                    # Add tool result to conversation
                    tool_result_msg = self.provider.format_tool_result_message(
                        tool_call["id"],
                        text_result
                    )
                    messages.append(tool_result_msg)
                    
                    # DEBUG: Log tool result message insertion
                    logger.info(f"Added tool result message for call {tool_call['id']}: {tool_result_msg}")
                    logger.debug(f"Current message count: {len(messages)}, last message role: {messages[-1].get('role', 'unknown')}")
                
                # After all tool results are added, add generated images back to the conversation
                # (unless hide_tool_images is enabled)
                if tool_generated_images and not self.hide_tool_images:
                    # For multiple images, we'll add them as separate messages for cleaner handling
                    for i, image in enumerate(tool_generated_images):
                        # Use provider's format_image_message method
                        image_message = self.provider.format_image_message(
                            f"Tool-generated image {i+1}/{len(tool_generated_images)}:",
                            image,
                            role="user"
                        )
                        messages.append(image_message)
                    
                    logger.debug(f"Added single user message with {len(tool_generated_images)} generated images")
                elif tool_generated_images and self.hide_tool_images:
                    logger.debug(f"Skipped adding {len(tool_generated_images)} generated images to conversation (hide_tool_images=True)")
                
                # DEBUG: Log message history after all tool results added
                logger.debug(f"Message history after tool execution ({len(messages)} messages):")
                for idx, msg in enumerate(messages):  # Show all messages
                    role = msg.get('role', 'unknown')
                    content_preview = str(msg.get('content', ''))[:100] + "..." if len(str(msg.get('content', ''))) > 100 else str(msg.get('content', ''))
                    tool_call_id = msg.get('tool_call_id', 'N/A')
                    logger.debug(f"  [{idx-5}] Role: {role}, Tool Call ID: {tool_call_id}, Content: {content_preview}")
                
                # Emit synthesis step
                emit_step(ast.make_synthesizing_step(
                    iteration=iteration,
                    num_tools=len(tool_calls),
                    timestamp=self._get_timestamp()
                ))
                
                # Update session messages with all tool calls and results
                session.messages = messages
                logger.debug(f"Updated session messages after tool execution: {len(session.messages)} messages")
                for i, msg in enumerate(session.messages):  # Show all messages
                    logger.debug(f"  Session message {i}: role={msg.get('role', 'unknown')}, content_type={type(msg.get('content', ''))}")
                
                # Continue the conversation
                continue
            else:
                # No more tool calls, we're done
                emit_step(ast.make_complete_step(
                    iteration=iteration,
                    message=message_content,
                    timestamp=self._get_timestamp()
                ))
                break
        
        # Update session messages one final time
        session.messages = messages
        
        # Build response
        result = {
            "response": message_content,
            "iterations": iteration,
            "tool_calls_made": tool_calls_made,
            "session_id": session_id
        }
        
        # Add steps and reasoning if requested
        if track_steps:
            result["steps"] = steps
            result["reasoning"] = reasoning
        
        return result
    
    def get_session(self, session_id: str) -> ConversationSession:
        """
        Get or create a conversation session.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            ConversationSession instance
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                enable_variables=self.enable_variables,
                variable_instructions=self._variable_prompt,
                provider=self.provider
            )
        return self.sessions[session_id]
    
    def create_session(
        self, 
        session_id: str, 
        initial_images: Optional[List[Image.Image]] = None
    ) -> ConversationSession:
        """
        Create a new conversation session (or return existing).
        
        Args:
            session_id: Unique identifier for the session
            initial_images: Optional list of initial images for the session
            
        Returns:
            ConversationSession instance
        """
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        session = ConversationSession(
            session_id=session_id,
            initial_images=initial_images,
            enable_variables=self.enable_variables,
            variable_instructions=self._variable_prompt,
            provider=self.provider
        )
        self.sessions[session_id] = session
        return session
    
    def add_user_message(self, session_id: str, text: str, images: Optional[List[Image.Image]] = None):
        """
        Add a user message to a session.
        
        This is a convenience wrapper. Prefer using:
            integration.get_session(session_id).add_user_message(text, images)
        """
        session = self.get_session(session_id)
        session.add_user_message(text, images)
    
    def add_system_message(self, session_id: str, text: str):
        """
        Add a system message to a session.
        
        This is a convenience wrapper. Prefer using:
            integration.get_session(session_id).add_system_message(text)
        """
        session = self.get_session(session_id)
        session.add_system_message(text)
    
    def get_response(self, session_id: str, max_iterations: int = 5, **provider_kwargs) -> Dict[str, Any]:
        """
        Get LLM response for a session, handling tool calls.
        
        This is a synchronous wrapper around get_response_async().
        
        Args:
            session_id: Session identifier
            max_iterations: Maximum number of tool call iterations
            **provider_kwargs: Provider-specific parameters (e.g., thinking_mode for bedrock)
            
        Returns:
            Dictionary with:
                - response: The final text response
                - iterations: Number of iterations taken
                - tool_calls_made: Total number of tool calls
                - session_id: Session identifier
        """
        return asyncio.run(self.get_response_async(
            session_id=session_id,
            max_iterations=max_iterations,
            step_callback=None,
            **provider_kwargs
        ))
    
    def quick_query(
        self,
        text: str,
        images: Optional[List[Image.Image]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Convenience method for one-shot queries without manual session management.
        
        Creates a temporary session, sends the query, and cleans up automatically.
        
        Args:
            text: The question/prompt text
            images: Optional list of images
            system_prompt: Optional system prompt to use
            
        Returns:
            The model's text response
            
        Example:
            agent = create_tool_agent(toolkit, provider="openai")
            response = agent.quick_query("What's in this image?", images=[my_image])
        """
        import uuid
        session_id = f"query_{uuid.uuid4()}"
        
        try:
            session = self.create_session(session_id, images)
            
            if system_prompt:
                session.add_system_message(system_prompt)
            
            session.add_user_message(text, images)
            response = self.get_response(session_id)
            return response["response"]
        finally:
            self.clear_session(session_id)
    
    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """
        Get the full conversation history for a session.
        
        This is a convenience wrapper. Prefer using:
            integration.get_session(session_id).get_history()
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        return self.sessions[session_id].get_history()
    
    def add_tool_generated_image(self, session_id: str, image: Image.Image, tool_name: str):
        """
        Add a tool-generated image to a session.
        
        This is a convenience wrapper. Prefer using:
            integration.get_session(session_id).add_tool_generated_image(image, tool_name)
        """
        if session_id in self.sessions:
            self.sessions[session_id].add_tool_generated_image(image, tool_name)
    
    def get_tool_generated_images(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all tool-generated images for a session.
        
        This is a convenience wrapper. Prefer using:
            integration.get_session(session_id).get_tool_generated_images()
        """
        if session_id in self.sessions:
            return self.sessions[session_id].get_tool_generated_images()
        return []
    
    def clear_session(self, session_id: str):
        """Clear a session's conversation history and variables."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_available_tools(self) -> List[str]:
        """Get list of available function names."""
        return [tool["function"]["name"] for tool in self.raw_tool_schemas]
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get mapping of tool names to descriptions."""
        return {
            tool["function"]["name"]: tool["function"]["description"] 
            for tool in self.raw_tool_schemas
        }
    
    def get_raw_schemas(self) -> List[Dict]:
        """
        Get tool schemas in raw format (OpenAI format).
        
        Returns:
            List of tool schema dictionaries in raw OpenAI format
        """
        return self.raw_tool_schemas.copy()
    
    def set_tool_filter(self, filter_func: Optional[callable]):
        """
        Set a filter function to control which tools are exposed to the LLM.
        
        Args:
            filter_func: Function that takes a tool schema dict and returns True to include it.
                        Set to None to include all tools.
        """
        self.tool_filter = filter_func
        self._apply_tool_filter()
    
    def _apply_tool_filter(self):
        """Apply the current tool filter to update available tools."""
        if self.tool_filter is None:
            self.raw_tool_schemas = self.all_raw_tool_schemas.copy()
        else:
            # Filter by converting each tool to provider format for the filter function
            filtered_tools = []
            for raw_tool in self.all_raw_tool_schemas:
                provider_tool = self._convert_to_provider_format([raw_tool])[0]
                if self.tool_filter(provider_tool):
                    filtered_tools.append(raw_tool)
            self.raw_tool_schemas = filtered_tools
    
    def get_all_available_tools(self) -> List[str]:
        """Get list of all available function names (before filtering)."""
        return [tool["function"]["name"] for tool in self.all_raw_tool_schemas]
    
    def get_session_variables(self, session_id: str) -> Dict[str, Any]:
        """
        Get all variables for a session.
        
        This is a convenience wrapper. Prefer using:
            integration.get_session(session_id).get_variables()
        """
        if session_id not in self.sessions:
            return {}
        return self.sessions[session_id].get_variables()
    
    def get_variable_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get variable statistics for a session.
        
        This is a convenience wrapper. Prefer using:
            integration.get_session(session_id).get_variable_stats()
        """
        if session_id not in self.sessions:
            return {"total_variables": 0, "variable_names": [], "variable_types": {}}
        return self.sessions[session_id].get_variable_stats()
    
    def _get_timestamp(self):
        """Get current timestamp for step tracking."""
        return datetime.datetime.now().isoformat()
    
    def _extract_result_content(self, result) -> Tuple[str, Optional[Image.Image]]:
        """
        Extract text and image content from a tool result.
        
        Args:
            result: Tool execution result (ToolResult object, dict, or other)
            
        Returns:
            Tuple of (text_result, generated_image_or_none)
        """
        text_result = None
        generated_image = None
        
        # Handle ToolResult objects
        text_result = result.text or str(result.value)
        
        # Check for images in the ToolResult
        if result.image:
            if isinstance(result.image, list) and len(result.image) > 0:
                # Take the first image if multiple
                generated_image = result.image[0]
            elif isinstance(result.image, Image.Image):
                generated_image = result.image
        
        return text_result, generated_image
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        
        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode()
    
def create_tool_agent(toolkit, provider: str = "openai", model: Optional[str] = None, 
                      enable_variables: bool = True, 
                      inject_variable_instructions: bool = True,
                      hide_tool_images: bool = False,
                      keep_dots_in_tool_names: bool = False,
                      **kwargs) -> ToolAgent:
    """
    Convenience function to create and initialize a ToolAgent.
    
    Args:
        toolkit: ToolkitClient instance
        provider: LLM provider ("openai", "llm_gateway_openai", "anthropic", "bedrock", or "sglang")
        model: Model to use (provider-specific default if None)
        enable_variables: Whether to enable variable handling
        inject_variable_instructions: Whether to auto-append variable handling instructions
            to system messages (default: True). Set to False if you manage your own
            system prompt that already includes variable instructions.
        hide_tool_images: Whether to hide tool output images from LLM (but still compute/display them)
        keep_dots_in_tool_names: Whether to keep dots in tool names instead of converting to double underscores.
                                Default: False (convert to double underscores for OpenAI compatibility)
        **kwargs: Additional client initialization parameters (e.g., api_key, bearer_token, base_url)
        
    Returns:
        Initialized ToolAgent
    """
    integration = ToolAgent(
        toolkit, 
        provider=provider, 
        model=model, 
        client_kwargs=kwargs,
        enable_variables=enable_variables,
        inject_variable_instructions=inject_variable_instructions,
        hide_tool_images=hide_tool_images,
        keep_dots_in_tool_names=keep_dots_in_tool_names
    )
    return integration
