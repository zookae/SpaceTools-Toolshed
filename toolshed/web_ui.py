#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
LLM Responses API + Toolshed Web Demo

A web interface for interacting with multiple LLM providers (OpenAI, Anthropic, Bedrock, SGLang) with Toolshed vision tools.
This version uses the unified LLM integration to support multiple provider APIs.
Users can upload images and have conversations about them, with the AI able to use
computer vision tools and display generated images directly in the chat.

Features:
- Support for OpenAI, Anthropic, Bedrock, and SGLang (fine-tuned models) providers
- Upload multiple images per conversation
- Continuous back-and-forth dialog with image support
- Tool-generated images displayed in conversation
- Real-time tool execution visualization
- Session management for multiple conversations
- Toolshed stays alive for efficient tool reuse

Usage:
    python llm_responses_web_demo.py [--port 8001] [--provider openai] [--model gpt-4o]

    # For SGLang provider with fine-tuned models:
    python llm_responses_web_demo.py --provider sglang --sglang-url http://localhost:30000/v1

Then open http://localhost:8001 in your browser.
"""

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import base64
import io
import os
import subprocess
import time
from datetime import datetime

# Web framework imports
try:
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    _WEB_DEPS_AVAILABLE = True
except ImportError:
    _WEB_DEPS_AVAILABLE = False

from PIL import Image
import ray
import numpy as np

# Toolshed imports
from toolshed import start_toolkit, get_toolkit
from toolshed.agent import create_tool_agent
from toolshed.prompts.coordinate_conventions import COORDINATE_CONVENTIONS_PROMPT
from toolshed.prompts.variables import VARIABLE_HANDLING_PROMPT
from toolshed.formats.verl import convert_messages_to_verl

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set specific loggers to appropriate levels
logging.getLogger('toolshed.agent').setLevel(logging.DEBUG)
logging.getLogger('toolshed.integration.providers.openai_provider').setLevel(logging.DEBUG)


# System prompt for the vision assistant
SYSTEM_PROMPT = f"""You are an AI assistant with access to powerful computer vision tools. Your role is to help users understand and analyze images by using the available tools to examine spatial relationships, detect objects, estimate depth, segment regions, and execute code when needed.

When a user asks questions about images, you should:
1. Analyze what information is needed to answer their question
2. Before calling tools, explain to the user your thought process so far and how you are approaching the problem.
3. Use the appropriate vision tools to gather spatial and visual information
4. Combine the results from multiple tools if needed to provide comprehensive answers
5. Explain your findings clearly, referencing specific locations and relationships in the scene
6. After you explanation, write "Final Answer:" and provie a concise answer to the user's question.

{COORDINATE_CONVENTIONS_PROMPT}

Select the appropriate frame to do your reasoning (2D or 3D). For example, 3D coordinates are perspective invariant and may better capture object sizes and locations in space. 2D coordinates may be sufficient to answer simpler questions about objects at a similar distance.
Always strive to provide accurate, detailed analysis of the spatial relationships and visual content in the images.
"""

SYSTEM_PROMPT_SGLANG = (
    "You are an expert in 3D spatial reasoning for robotics. "
    "Given an image and a spatial reasoning question, follow this process:\n\n"
    "1. First, think about the reasoning process as an internal monologue the first time you receive the question, and every time you receive new information.\n"
    "Your reasoning process MUST be enclosed within <think> </think> tags.\n"
    "2. After thinking, if you need additional information to answer the question, such as specific object location in the image, call the appropriate vision tool exactly once.\n"
    "3. When you receive a tool response that contains reasonable information, use that information to continue your analysis on the question.\n"
    "4. Once no further visual analysis or tool calls are needed, you MUST provide your final answer inside "
    "<answer> and </answer> tags without detailed illustrations.\n\n"
    "Example answer format: <answer> <Your final answer here> </answer>."
)


class LLMResponsesWebDemo:
    """Web demo for unified LLM API + Toolshed integration."""

    def __init__(self,
                 tool_configs: Dict[str, Dict[str, Any]],
                 provider: str = "bedrock",
                 model: Optional[str] = None,
                 port: int = 8001,
                 enable_variables: bool = True,
                 enable_images: bool = True,
                 hide_tool_images: bool = False,
                 keep_dots_in_tool_names: bool = False,
                 logs_dir: str = "logs/unsorted",
                 saved_conversations_dir: str = "saved_conversations",
                 custom_system_prompt: str = "",
                 dashboard: bool = False,
                 dashboard_port: int = 7001):
        """
        Initialize the web demo.

        Args:
            tool_configs: Toolshed tool configurations
            provider: LLM provider ("openai", "anthropic", "bedrock", "sglang")
            model: Model to use (provider-specific default if None)
            port: Port for the web server
            enable_variables: Whether to enable variable output
            enable_images: Whether to enable image output
            hide_tool_images: Whether to hide tool output images from LLM (but still compute/display them)
            keep_dots_in_tool_names: Whether to keep dots in tool names (e.g., "vlm.detect_one") instead
                                    of converting to double underscores (e.g., "vlm__detect_one")
            logs_dir: Directory for auto-saved conversation logs
            saved_conversations_dir: Directory for manually saved conversations
            custom_system_prompt: Additional text to append to the system prompt
            dashboard: Whether to start the toolshed dashboard in the background
            dashboard_port: Port for the toolshed dashboard web server
        """
        if not _WEB_DEPS_AVAILABLE:
            raise ImportError(
                "Web demo requires FastAPI and uvicorn. Install with: "
                "pip install fastapi uvicorn python-multipart"
            )

        self.tool_configs = tool_configs
        self.provider = provider
        self.model = model
        self.port = port
        self.enable_variables = enable_variables
        self.enable_images = enable_images
        self.hide_tool_images = hide_tool_images
        self.keep_dots_in_tool_names = keep_dots_in_tool_names
        self.logs_dir = logs_dir
        self.saved_conversations_dir = saved_conversations_dir
        self.custom_system_prompt = custom_system_prompt
        self.dashboard = dashboard
        self.dashboard_port = dashboard_port

        # Build initial system prompt (will be editable from UI)
        self.system_prompt = self._build_default_system_prompt()

        # Create conversation directories
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.saved_conversations_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Auto-save logs directory: {self.logs_dir}")
        logger.info(f"Manual save directory: {self.saved_conversations_dir}")

        # Apply configuration to tool configs
        for tool_name in self.tool_configs:
            self.tool_configs[tool_name]["args"]["no_output_vars"] = not enable_variables
            self.tool_configs[tool_name]["args"]["no_output_image"] = not enable_images

        # Will be initialized when starting
        self.toolkit = None
        self.router = None
        self.llm_integration = None

        # Task tracking for real-time updates
        self.chat_tasks = {}

        # Track stopped tasks
        self.stopped_tasks = set()

        # Track running asyncio tasks for cancellation
        self.running_tasks = {}

        # Track enabled/disabled tools (all enabled by default)
        self.enabled_tools = {}

        # Conversation enumeration counters - initialize by scanning existing files
        self.logs_enum_counter = self._get_next_enum_index(self.logs_dir)
        self.saved_enum_counter = self._get_next_enum_index(self.saved_conversations_dir)
        logger.info(f"Initialized conversation counters - logs: {self.logs_enum_counter}, saved: {self.saved_enum_counter}")

        # Available models by provider
        self.available_models = {
            "openai": [
                {"id": "gpt-5", "name": "GPT-5", "description": "Latest and most capable model"},
                {"id": "gpt-5-mini", "name": "GPT-5 Mini", "description": "Efficient GPT-5 variant"},
                {"id": "gpt-4o", "name": "GPT-4o", "description": "Optimized GPT-4 model"},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "description": "Faster, cost-effective"},
                {"id": "gpt-4", "name": "GPT-4", "description": "Previous generation flagship"},
                {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "description": "Optimized for speed"},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "Fast and efficient"},
            ],
            "llm_gateway_openai": [
                {"id": "gpt-5", "name": "GPT-5", "description": "Latest and most capable model via NVIDIA LLM Gateway"},
                {"id": "gpt-5-mini", "name": "GPT-5 Mini", "description": "Efficient GPT-5 variant via NVIDIA LLM Gateway"},
                {"id": "gpt-4o", "name": "GPT-4o", "description": "Optimized GPT-4 model via NVIDIA LLM Gateway"},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "description": "Faster, cost-effective via NVIDIA LLM Gateway"},
                {"id": "gpt-4", "name": "GPT-4", "description": "Previous generation flagship via NVIDIA LLM Gateway"},
                {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "description": "Optimized for speed via NVIDIA LLM Gateway"},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "Fast and efficient via NVIDIA LLM Gateway"},
                {"id": "gpt-4.1-nano", "name": "GPT-4.1 Nano", "description": "NVIDIA LLM Gateway specific model"},
            ],
            "anthropic": [
                {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "description": "Latest Claude Sonnet model"},
                {"id": "claude-opus-4-1-20250805", "name": "Claude 4.1 Opus", "description": "Most capable Claude model"},
                {"id": "claude-opus-4-20250514", "name": "Claude Opus 4", "description": "Latest Claude Opus generation"},
                {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "description": "Previous Opus model"},
                {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "description": "Fast and efficient"},
            ],
            "bedrock": [
                {"id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "name": "Claude 4.5 Sonnet", "description": "Latest Bedrock Claude model"},
                {"id": "us.anthropic.claude-opus-4-20250514-v1:0", "name": "Claude Opus 4", "description": "Most capable Bedrock Claude model"},
                {"id": "us.anthropic.claude-sonnet-4-20250514-v1:0", "name": "Claude Sonnet 4", "description": "Balanced performance on Bedrock"},
                {"id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0", "name": "Claude 3.7 Sonnet", "description": "Updated Claude 3 Sonnet"},
            ],
            "sglang": [
                {"id": "Qwen2.5-VL-3B-Instruct", "name": "Qwen 2.5 VL 3B (Fine-tuned)", "description": "Fine-tuned Qwen model for pick and place"},
                {"id": "Qwen2.5-VL-3B-Base", "name": "Qwen 2.5 VL 3B (Base)", "description": "Non-finetuned base Qwen model"},
                {"id": "custom", "name": "Custom Model", "description": "Custom fine-tuned model on sglang server"},
            ]
        }

        # Map sglang model IDs to their server URLs
        self.sglang_model_urls = {
            "Qwen2.5-VL-3B-Instruct": "http://localhost:34567/v1",  # Fine-tuned model
            "Qwen2.5-VL-3B-Base": "http://localhost:34568/v1",      # Base model
            "custom": "http://localhost:34567/v1",                  # Default to fine-tuned port
        }

        # Create FastAPI app
        self.app = FastAPI(title="LLM Responses API + Toolshed Demo")

        # Add exception handler for request validation errors
        @self.app.exception_handler(Exception)
        async def generic_exception_handler(request, exc):
            logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": f"{type(exc).__name__}: {str(exc)}"}
            )

        # Setup static file serving
        resources_dir = Path(__file__).parent / "resources"
        self.app.mount("/static", StaticFiles(directory=str(resources_dir)), name="static")

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/favicon.ico")
        async def favicon():
            """Serve the toolbox icon as favicon."""
            resources_dir = Path(__file__).parent / "resources"
            icon_path = resources_dir / "toolbox.png"
            return FileResponse(str(icon_path), media_type="image/png")

        @self.app.get("/", response_class=HTMLResponse)
        async def home():
            resources_dir = Path(__file__).parent / "resources"
            html_file = resources_dir / "llm_index.html"

            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
                # Inject favicon link if not already present
                if '<link rel="icon"' not in html_content and '<link rel="shortcut icon"' not in html_content:
                    # Insert favicon link in the head section
                    html_content = html_content.replace(
                        '</head>',
                        '<link rel="icon" type="image/png" href="/favicon.ico">\n</head>',
                        1
                    )
                return html_content

        @self.app.post("/api/upload")
        async def upload_images(files: List[UploadFile] = File(...)):
            """Upload images and return image IDs."""
            try:
                image_ids = []
                for file in files:
                    # Read and validate image
                    contents = await file.read()
                    image = Image.open(io.BytesIO(contents))

                    # Generate unique ID and store
                    image_id = str(uuid.uuid4())
                    # In a real app, you'd store this in a database or cache
                    # For demo, we'll return the base64 data directly
                    image_data = self._image_to_base64(image)

                    image_ids.append({
                        "id": image_id,
                        "filename": file.filename,
                        "data": image_data,
                        "width": image.width,
                        "height": image.height
                    })

                return {"images": image_ids}

            except Exception as e:
                logger.error("Error uploading images: %s", e)
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/api/chat/start")
        async def start_chat(
            session_id: str = Form(...),
            message: str = Form(...),
            image_data: str = Form(None)  # JSON string of image data
        ):
            """Start a chat session and return a task ID for polling."""
            logger.info(f"Received chat start request for session {session_id}")
            try:
                if not self.llm_integration:
                    raise HTTPException(status_code=500, detail="Integration not initialized")

                # Parse image data if provided
                images = []
                if image_data:
                    image_list = json.loads(image_data)
                    for img_data in image_list:
                        # Decode base64 image
                        image_bytes = base64.b64decode(img_data["data"])
                        image = Image.open(io.BytesIO(image_bytes))
                        images.append(image)

                # Create session with system message if it doesn't exist
                if session_id not in self.llm_integration.sessions:
                    logger.info(f"Creating new session {session_id} with {len(images)} images")
                    session = self.llm_integration.create_session(session_id, images)
                    # Add system message
                    session.add_system_message(self._build_system_prompt())
                    logger.debug(f"Added system message to session {session_id}")

                    # Store images as $input_image variable if provided
                    if images and session.variable_engine:
                        if len(images) == 1:
                            session.store_variables({"input_image": images[0]})
                            logger.debug(f"Stored single input image as $input_image variable")
                        else:
                            session.store_variables({"input_image": images})
                            logger.debug(f"Stored {len(images)} input images as $input_image variable")
                else:
                    logger.debug(f"Using existing session {session_id}")

                # Add user message to session
                session = self.llm_integration.get_session(session_id)
                session.add_user_message(message, images)

                # Generate a task ID for this request
                task_id = str(uuid.uuid4())

                # Start the processing in the background and track it
                task = asyncio.create_task(self._process_chat_async(session_id, task_id))
                self.running_tasks[task_id] = task

                return {
                    "task_id": task_id,
                    "session_id": session_id,
                    "status": "started"
                }

            except Exception as e:
                logger.error(f"Error starting chat: {type(e).__name__}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

        @self.app.get("/api/chat/status/{task_id}")
        async def get_chat_status(task_id: str):
            """Get the current status and steps for a chat task."""
            try:
                # Get status from our task tracker
                if task_id in self.chat_tasks:
                    return self.chat_tasks[task_id]
                else:
                    raise HTTPException(status_code=404, detail="Task not found")

            except Exception as e:
                logger.error("Error getting chat status: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/chat/stop/{task_id}")
        async def stop_chat(task_id: str):
            """Stop a running chat task."""
            try:
                if task_id not in self.chat_tasks:
                    raise HTTPException(status_code=404, detail="Task not found")

                # Mark task as stopped
                self.stopped_tasks.add(task_id)

                # Cancel the running asyncio task if it exists
                if task_id in self.running_tasks:
                    self.running_tasks[task_id].cancel()
                    logger.info(f"Task {task_id} cancelled")
                else:
                    logger.warning(f"Task {task_id} marked for stopping but no running task found")

                return {"message": "Stop signal sent", "task_id": task_id}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error stopping chat: {type(e).__name__}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/session/{session_id}/history")
        async def get_session_history(session_id: str):
            """Get conversation history for a session."""
            try:
                if not self.llm_integration:
                    raise HTTPException(status_code=500, detail="Integration not initialized")

                history = self.llm_integration.get_session_history(session_id)
                return history

            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error("Error getting history: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/session/{session_id}")
        async def clear_session(session_id: str):
            """Clear a conversation session."""
            try:
                if not self.llm_integration:
                    raise HTTPException(status_code=500, detail="Integration not initialized")

                self.llm_integration.clear_session(session_id)
                return {"message": "Session cleared"}

            except Exception as e:
                logger.error("Error clearing session: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/session/{session_id}/save")
        async def save_conversation(session_id: str, name: str = Form(...)):
            """Manually save a conversation with a name."""
            try:
                if not self.llm_integration:
                    raise HTTPException(status_code=500, detail="Integration not initialized")

                # Save to saved_conversations_dir with name
                result = self._save_conversation(session_id, self.saved_conversations_dir, name=name)

                if result.get("success"):
                    return {
                        "message": f"Conversation saved as '{name}'",
                        "file": result["file"],
                        "enum_idx": result["enum_idx"]
                    }
                else:
                    raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

            except HTTPException:
                raise
            except Exception as e:
                logger.error("Error saving conversation: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/tools")
        async def get_available_tools():
            """Get list of available tools."""
            try:
                if not self.llm_integration:
                    raise HTTPException(status_code=500, detail="Integration not initialized")

                # Get all tools (before filtering)
                all_tools = self.llm_integration.get_all_available_tools()
                active_tools = self.llm_integration.get_available_tools()

                # Get tool descriptions
                all_descriptions = self.llm_integration.get_tool_descriptions()

                # Build tool list with enabled status
                tool_list = []
                for tool in all_tools:
                    tool_list.append({
                        "name": tool,
                        "description": all_descriptions[tool],  # Let it raise KeyError if missing
                        "enabled": tool in active_tools and self.enabled_tools.get(tool, True)
                    })

                return {
                    "tools": tool_list,
                    "provider": self.provider,
                    "model": self.model,
                    "variables_enabled": self.llm_integration.enable_variables
                }

            except Exception as e:
                logger.error("Error getting tools: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/session/{session_id}/variables")
        async def get_session_variables(session_id: str):
            """Get variables for a session."""
            try:
                if not self.llm_integration:
                    raise HTTPException(status_code=500, detail="Integration not initialized")

                if not self.llm_integration.enable_variables:
                    return {"variables": {}, "stats": {"total_variables": 0, "variable_names": [], "variable_types": {}}}

                variables = self.llm_integration.get_session_variables(session_id)
                stats = self.llm_integration.get_variable_stats(session_id)

                # Make variables JSON-serializable
                serializable_variables = self._make_variables_serializable(variables)

                return {
                    "variables": serializable_variables,
                    "stats": stats
                }

            except Exception as e:
                logger.error("Error getting session variables: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/providers")
        async def get_available_providers():
            """Get list of available LLM providers."""
            try:
                return {
                    "providers": [
                        {"id": "openai", "name": "OpenAI", "description": "GPT models"},
                        {"id": "llm_gateway_openai", "name": "NVIDIA LLM Gateway (OpenAI)", "description": "OpenAI models via NVIDIA LLM Gateway"},
                        {"id": "anthropic", "name": "Anthropic", "description": "Claude models"},
                        {"id": "bedrock", "name": "AWS Bedrock", "description": "Claude models via NVIDIA LLM Gateway"},
                        {"id": "sglang", "name": "SGLang", "description": "Fine-tuned models via sglang server"}
                    ],
                    "current_provider": self.provider
                }

            except Exception as e:
                logger.error("Error getting providers: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/models")
        async def get_available_models():
            """Get list of available models for current provider."""
            try:
                return {
                    "models": self.available_models.get(self.provider, []),
                    "current_model": self.model,
                    "current_provider": self.provider
                }

            except Exception as e:
                logger.error("Error getting available models: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/provider/change")
        async def change_provider(provider: str = Form(...), model: str = Form(None)):
            """Change the LLM provider and optionally the model."""
            try:
                # Validate provider
                if provider not in ["openai", "llm_gateway_openai", "anthropic", "bedrock", "sglang"]:
                    raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

                # If no model specified, use default for provider
                if not model:
                    if provider == "openai":
                        model = "gpt-5"
                    elif provider == "llm_gateway_openai":
                        model = "gpt-4o"
                    elif provider == "anthropic":
                        model = "claude-opus-4-1-20250805"
                    elif provider == "bedrock":
                        model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
                    else:  # sglang
                        model = "Qwen2.5-VL-3B-Instruct"

                # Update provider and model
                await self.update_provider_and_model(provider, model)

                return {
                    "message": f"Provider changed to {provider} with model {model}",
                    "provider": provider,
                    "model": model
                }

            except Exception as e:
                logger.error("Error changing provider: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/model/change")
        async def change_model(model: str = Form(...)):
            """Change the model for current provider."""
            try:
                # Validate model for current provider
                provider_models = [m["id"] for m in self.available_models.get(self.provider, [])]
                if model not in provider_models:
                    raise HTTPException(status_code=400, detail=f"Model {model} not available for provider {self.provider}")

                await self.update_provider_and_model(self.provider, model)
                return {"message": f"Model changed to {model}", "model": model}

            except Exception as e:
                logger.error("Error changing model: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/config/current")
        async def get_current_configuration():
            """Get current configuration settings."""
            try:
                # Return instance variables which are the source of truth
                return {
                    "enable_variables": self.enable_variables,
                    "enable_images": self.enable_images,
                    "hide_tool_images": self.hide_tool_images,
                    "provider": self.provider,
                    "model": self.model,
                    "custom_system_prompt": self.custom_system_prompt
                }

            except Exception as e:
                logger.error("Error getting configuration: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/config/update")
        async def update_configuration(
            enable_variables: bool = Form(...),
            enable_images: bool = Form(...),
            hide_tool_images: bool = Form(False),
            custom_system_prompt: str = Form("")
        ):
            """Update tool configuration and restart toolshed."""
            try:
                # Update instance variables
                self.enable_variables = enable_variables
                self.enable_images = enable_images
                self.hide_tool_images = hide_tool_images
                self.custom_system_prompt = custom_system_prompt

                # Rebuild system prompt with new settings
                self.system_prompt = self._build_default_system_prompt()

                # Update tool configs
                for tool_name in self.tool_configs:
                    self.tool_configs[tool_name]["args"]["no_output_vars"] = not enable_variables
                    self.tool_configs[tool_name]["args"]["no_output_image"] = not enable_images

                # Shutdown existing toolshed
                if self.router:
                    logger.info("Shutting down existing toolshed...")
                    ray.kill(self.router)
                    self.router = None
                    self.toolkit = None
                    self.llm_integration = None

                # Restart with new configuration
                await self.start_toolshed()

                # Clear sessions
                logger.info("Sessions cleared after toolshed restart")

                return {
                    "message": "Configuration updated successfully",
                    "enable_variables": enable_variables,
                    "enable_images": enable_images,
                    "hide_tool_images": hide_tool_images,
                    "custom_system_prompt": custom_system_prompt,
                    "sessions_cleared": True
                }

            except Exception as e:
                logger.error("Error updating configuration: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/tools/update")
        async def update_tool_configuration(enabled_tools: str = Form(...)):
            """Update which tools are enabled without restarting toolshed."""
            try:
                if not self.llm_integration:
                    raise HTTPException(status_code=500, detail="Integration not initialized")

                # Parse JSON string
                enabled_tools_dict = json.loads(enabled_tools)

                # Update enabled tools
                self.enabled_tools = enabled_tools_dict

                # Create filter function
                def tool_filter(tool_schema):
                    if self.provider in ["openai", "llm_gateway_openai", "sglang"]:
                        tool_name = tool_schema["function"]["name"]
                    else:  # anthropic, bedrock
                        tool_name = tool_schema["name"]
                    return self.enabled_tools.get(tool_name, True)

                # Apply filter to LLM integration
                self.llm_integration.set_tool_filter(tool_filter)

                logger.info("Updated tool configuration: %d tools enabled out of %d total",
                           sum(enabled_tools_dict.values()),
                           len(self.llm_integration.get_all_available_tools()))

                return {
                    "message": "Tool configuration updated successfully",
                    "enabled_count": sum(enabled_tools_dict.values()),
                    "total_count": len(self.llm_integration.get_all_available_tools())
                }

            except Exception as e:
                logger.error("Error updating tool configuration: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/system-prompt")
        async def get_system_prompt():
            """Get the current system prompt."""
            try:
                return {
                    "system_prompt": self.system_prompt,
                    "default_system_prompt": self._build_default_system_prompt()
                }

            except Exception as e:
                logger.error("Error getting system prompt: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/system-prompt/update")
        async def update_system_prompt(system_prompt: str = Form(...)):
            """Update the system prompt."""
            try:
                self.system_prompt = system_prompt
                logger.info("System prompt updated (length: %d characters)", len(system_prompt))

                return {
                    "message": "System prompt updated successfully",
                    "length": len(system_prompt)
                }

            except Exception as e:
                logger.error("Error updating system prompt: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/system-prompt/reset")
        async def reset_system_prompt():
            """Reset the system prompt to default."""
            try:
                self.system_prompt = self._build_default_system_prompt()
                logger.info("System prompt reset to default")

                return {
                    "message": "System prompt reset to default",
                    "system_prompt": self.system_prompt
                }

            except Exception as e:
                logger.error("Error resetting system prompt: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

    def _build_default_system_prompt(self) -> str:
        """Build the default system prompt based on provider and settings."""
        if self.provider == "sglang":
            print("Using SGLang system prompt")
            prompt = SYSTEM_PROMPT_SGLANG
        else:
            print("Using default system prompt")
            prompt = SYSTEM_PROMPT

            # Add variable handling instructions if variables are enabled
            if self.enable_variables:
                prompt = prompt + "\n\n" + VARIABLE_HANDLING_PROMPT

        # Append custom system prompt if provided
        if self.custom_system_prompt:
            prompt = prompt + "\n\n" + self.custom_system_prompt

        return prompt

    def _build_system_prompt(self) -> str:
        """Return the current system prompt (uses stored editable version)."""
        return self.system_prompt

    def _get_next_enum_index(self, directory: str) -> int:
        """
        Scan a directory to find the maximum enumeration index used in saved conversations.
        Returns the next available index (max + 1, or 0 if no files exist).

        Args:
            directory: Directory to scan for existing conversation files

        Returns:
            Next available enumeration index

        Raises:
            Exception: If there's an error scanning the directory (to prevent data loss from overwriting)
        """
        import re

        max_idx = -1

        # Check if directory exists
        if not os.path.exists(directory):
            return 0

        # Scan all subdirectories for image filenames with pattern s{enum_idx:03d}_*
        for root, dirs, files in os.walk(directory):
            for filename in files:
                # Look for patterns like s000_img0001.png or s123_tool0042.png
                match = re.match(r's(\d+)_', filename)
                if match:
                    idx = int(match.group(1))
                    max_idx = max(max_idx, idx)

        # Return next available index
        return max_idx + 1

    def _get_local_ip(self) -> str:
        """Get the local IP address accessible from other machines on the network."""
        try:
            # Use hostname -I which returns all IPs, first one is typically the primary interface
            result = subprocess.run(
                ["hostname", "-I"],
                check=True,
                capture_output=True,
                text=True
            )
            ips = result.stdout.strip().split()
            if ips:
                return ips[0]  # First IP is usually the primary network interface
        except Exception:
            pass
        return "localhost"

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)

        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode()

    def _make_variables_serializable(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert variables to JSON-serializable format for API responses.

        Args:
            variables: Dictionary of variable name -> value

        Returns:
            Dictionary with JSON-serializable values
        """
        serializable = {}

        for name, value in variables.items():
            try:
                # Handle PIL Images
                if hasattr(value, 'save') and hasattr(value, 'mode'):  # PIL Image
                    serializable[name] = {
                        "_type": "PIL_Image",
                        "_summary": f"PIL Image ({value.width}x{value.height}, {value.mode})",
                        "width": value.width,
                        "height": value.height,
                        "mode": value.mode
                    }
                # Handle numpy arrays
                elif hasattr(value, 'shape') and hasattr(value, 'dtype'):  # numpy array
                    serializable[name] = {
                        "_type": "numpy_array",
                        "_summary": f"numpy array {value.shape} {value.dtype}",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "size": int(value.size)
                    }
                # Handle Ray object references
                elif hasattr(value, '__module__') and value.__module__ and 'ray' in value.__module__:
                    serializable[name] = {
                        "_type": "ray_object",
                        "_summary": f"Ray object reference ({type(value).__name__})"
                    }
                # Handle other complex objects that might not be JSON serializable
                else:
                    # Try to serialize directly first
                    import json
                    try:
                        json.dumps(value)  # Test if it's JSON serializable
                        serializable[name] = value
                    except (TypeError, ValueError):
                        # If not serializable, provide a summary
                        serializable[name] = {
                            "_type": type(value).__name__,
                            "_summary": f"{type(value).__name__}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}",
                            "_length": len(value) if hasattr(value, '__len__') else None
                        }

            except Exception as e:
                # Fallback for any errors
                logger.warning(f"Error serializing variable '{name}': {e}")
                serializable[name] = {
                    "_type": "error",
                    "_summary": f"Serialization error: {str(e)[:100]}"
                }

        return serializable

    def _convert_variables_to_numpy(self, variables: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert session variables to numpy arrays for saving to npz.

        Args:
            variables: Dictionary of variable name -> value

        Returns:
            Dictionary with numpy-compatible values (PIL Images converted, etc.)
        """
        numpy_vars = {}

        for name, value in variables.items():
            try:
                # Handle PIL Images
                if hasattr(value, 'save') and hasattr(value, 'mode'):  # PIL Image
                    numpy_vars[name] = np.array(value)
                    logger.debug(f"Converted variable '{name}' (PIL Image) to numpy array {numpy_vars[name].shape}")

                # Handle numpy arrays - keep as is
                elif isinstance(value, np.ndarray):
                    numpy_vars[name] = value
                    logger.debug(f"Kept variable '{name}' as numpy array {value.shape}")

                # Handle lists of PIL Images
                elif isinstance(value, (list, tuple)) and len(value) > 0 and hasattr(value[0], 'save'):
                    # Convert list of images to array
                    img_arrays = [np.array(img) for img in value]
                    numpy_vars[name] = np.stack(img_arrays)
                    logger.debug(f"Converted variable '{name}' (list of {len(value)} images) to numpy array {numpy_vars[name].shape}")

                # Handle numeric types (int, float)
                elif isinstance(value, (int, float, np.number)):
                    numpy_vars[name] = np.array(value)
                    logger.debug(f"Converted variable '{name}' (numeric) to numpy scalar")

                # Handle lists/tuples of numbers
                elif isinstance(value, (list, tuple)):
                    try:
                        arr = np.array(value)
                        # Only save if it's a numeric array
                        if np.issubdtype(arr.dtype, np.number):
                            numpy_vars[name] = arr
                            logger.debug(f"Converted variable '{name}' (list) to numpy array {arr.shape}")
                        else:
                            logger.debug(f"Skipping variable '{name}' (non-numeric list)")
                    except (ValueError, TypeError):
                        logger.debug(f"Skipping variable '{name}' (could not convert list to array)")

                # Handle strings - save as numpy string array
                elif isinstance(value, str):
                    numpy_vars[name] = np.array(value, dtype=object)
                    logger.debug(f"Saved variable '{name}' as string")

                # Skip unsupported types
                else:
                    logger.debug(f"Skipping variable '{name}' (unsupported type: {type(value).__name__})")

            except Exception as e:
                logger.warning(f"Error converting variable '{name}' to numpy: {e}")
                continue

        return numpy_vars

    def _save_conversation(self, session_id: str, output_dir: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Save a conversation to disk in VeRL format.

        For unsorted logs (auto-save): Appends to a single JSONL with shared images directory
        For manual saves: Creates a separate folder per conversation with its own JSON/JSONL and images

        Args:
            session_id: Session identifier
            output_dir: Directory to save to (logs_dir or saved_conversations_dir)
            name: Optional name for the conversation (for manual saves)

        Returns:
            Dict with save status and file paths
        """
        try:
            logger.info(f"Attempting to save conversation {session_id} to {output_dir} (name={name})")

            # Get session history
            if not self.llm_integration:
                logger.error("LLM integration not initialized")
                return {"success": False, "error": "Integration not initialized"}

            if session_id not in self.llm_integration.sessions:
                logger.error(f"Session {session_id} not found in sessions: {list(self.llm_integration.sessions.keys())}")
                return {"success": False, "error": "Session not found"}

            session_history = self.llm_integration.get_session_history(session_id)
            messages = session_history["messages"]

            if not messages:
                return {"success": False, "error": "No messages to save"}

            # Determine if this is logs or saved_conversations
            is_logs = output_dir == self.logs_dir

            # Get enumeration counter
            if is_logs:
                enum_idx = self.logs_enum_counter
                self.logs_enum_counter += 1
            else:
                enum_idx = self.saved_enum_counter
                self.saved_enum_counter += 1

            # Extract first user message and last assistant message
            user_messages = [m for m in messages if m.get("role") == "user"]
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]

            first_input = ""
            if user_messages:
                content = user_messages[0].get("content", "")
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    first_input = " ".join(text_parts)
                else:
                    first_input = str(content)

            last_output = ""
            if assistant_messages:
                content = assistant_messages[-1].get("content", "")
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    last_output = " ".join(text_parts)
                else:
                    last_output = str(content)

            # Setup directories based on whether this is unsorted or manual save
            if is_logs:
                # Unsorted: use shared structure (existing behavior)
                step = 0
                os.makedirs(output_dir, exist_ok=True)
                images_dir = os.path.join(output_dir, f"images_{step}")
                os.makedirs(images_dir, exist_ok=True)
                images_relative_path = f"images_{step}"
            else:
                # Manual save: create folder per conversation
                if not name:
                    return {"success": False, "error": "Name required for manual save"}

                # Sanitize folder name
                safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
                convo_dir = os.path.join(output_dir, safe_name)

                # Check if conversation folder already exists
                if os.path.exists(convo_dir):
                    return {"success": False, "error": f"Conversation '{name}' already exists. Please choose a different name to avoid overwriting data."}

                os.makedirs(convo_dir, exist_ok=True)
                images_dir = os.path.join(convo_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                images_relative_path = "images"

            # Save input images from session
            image_paths = []
            session = self.llm_integration.get_session(session_id)
            initial_images = session.images

            for img_idx, image in enumerate(initial_images):
                img_filename = f"s{enum_idx:03d}_img{img_idx:04d}.png"
                img_path = os.path.join(images_dir, img_filename)
                image.save(img_path)
                image_paths.append(f"{images_relative_path}/{img_filename}")

            # Save tool-generated images
            tool_images = self.llm_integration.get_tool_generated_images(session_id)
            for tool_img_idx, tool_img_data in enumerate(tool_images):
                img_filename = f"s{enum_idx:03d}_tool{tool_img_idx:04d}.png"
                img_path = os.path.join(images_dir, img_filename)
                tool_img_data["image"].save(img_path)
                image_paths.append(f"{images_relative_path}/{img_filename}")

            # Convert messages to VeRL format (if available, otherwise use original format)
            verl_messages = convert_messages_to_verl(messages)

            # Update image references in messages
            if image_paths:
                image_path_idx = 0
                for msg in verl_messages:
                    if 'content' in msg and isinstance(msg['content'], list):
                        for content_item in msg['content']:
                            if content_item.get('type') == 'image':
                                # Assign images sequentially in chronological order
                                if image_path_idx < len(image_paths):
                                    content_item['image_path'] = image_paths[image_path_idx]
                                    image_path_idx += 1

            # Build entry
            entry = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "input": first_input,
                "output": last_output,
                "messages": verl_messages,
                "image_paths": image_paths,
                "provider": self.provider,
                "model": self.model,
                "num_messages": len(messages),
                "num_images": len(initial_images),
                "num_tool_images": len(tool_images)
            }

            # Add name if provided (for manual saves)
            if name:
                entry["name"] = name

            # Save files based on type
            if is_logs:
                # Unsorted: append to single JSONL
                filename = os.path.join(output_dir, "0.jsonl")
                with open(filename, "a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                json_filename = None
                vars_dir = output_dir
            else:
                # Manual save: create both JSONL and JSON in conversation folder
                jsonl_filename = os.path.join(convo_dir, "conversation.jsonl")
                json_filename = os.path.join(convo_dir, "conversation.json")

                # Write JSONL (1 line)
                with open(jsonl_filename, "w") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

                # Write JSON (formatted)
                with open(json_filename, "w") as f:
                    json.dump(entry, f, ensure_ascii=False, indent=2, default=str)

                filename = jsonl_filename
                vars_dir = convo_dir

            # Save session variables to npz if variables are enabled
            vars_filename = None
            saved_var_images = []
            if self.llm_integration.enable_variables:
                try:
                    variables = self.llm_integration.get_session_variables(session_id)
                    if variables:
                        vars_filename = os.path.join(vars_dir, f"vars_s{enum_idx:03d}.npz" if is_logs else "vars.npz")

                        # Save PIL images as individual PNG files before converting to numpy
                        for var_name, var_value in variables.items():
                            # Handle single PIL Image
                            if hasattr(var_value, 'save') and hasattr(var_value, 'mode'):
                                img_filename = os.path.join(vars_dir, f"{var_name}.png")
                                var_value.save(img_filename)
                                saved_var_images.append(img_filename)
                                logger.debug(f"Saved variable image '{var_name}' to {img_filename}")

                            # Handle list of PIL Images
                            elif isinstance(var_value, (list, tuple)) and len(var_value) > 0 and hasattr(var_value[0], 'save'):
                                for idx, img in enumerate(var_value):
                                    img_filename = os.path.join(vars_dir, f"{var_name}_{idx:04d}.png")
                                    img.save(img_filename)
                                    saved_var_images.append(img_filename)
                                logger.debug(f"Saved variable image list '{var_name}' ({len(var_value)} images)")

                        # Convert to numpy and save npz
                        numpy_vars = self._convert_variables_to_numpy(variables)
                        if numpy_vars:
                            np.savez(vars_filename, **numpy_vars)
                            logger.info(f"Saved {len(numpy_vars)} variables to {vars_filename}")
                            if saved_var_images:
                                logger.info(f"Saved {len(saved_var_images)} variable images as PNG files")
                except Exception as e:
                    logger.warning(f"Failed to save variables: {e}")
                    # Don't fail the entire save if variable saving fails
                    vars_filename = None

            logger.info(f"Saved conversation {session_id} to {filename} (enum_idx={enum_idx})")

            return {
                "success": True,
                "file": filename,
                "json_file": json_filename if not is_logs else None,
                "vars_file": vars_filename,
                "var_images": saved_var_images,
                "image_paths": image_paths,
                "enum_idx": enum_idx
            }

        except Exception as e:
            logger.error(f"Error saving conversation: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _process_chat_async(self, session_id: str, task_id: str):
        """Process chat request asynchronously with real-time step updates."""
        try:
            # Initialize task status
            self.chat_tasks[task_id] = {
                "status": "processing",
                "steps": [],
                "response": None,
                "error": None,
                "completed": False
            }

            # Initialize timing tracker
            timings = self._init_timing_tracker()

            # Define step callback that updates the task
            def step_callback(step):
                # Check if task was stopped
                if task_id in self.stopped_tasks:
                    logger.info(f"Task {task_id} stop detected in step callback")
                    raise asyncio.CancelledError("Task stopped by user")

                # Update timing
                self._update_timing_from_step(timings, step)

                # DEBUG: Log step callback details
                step_type = step.get('type', 'unknown')
                logger.debug(f"Step callback received: {step_type} - {step.get('status', 'no status')}")
                if step_type == 'tool_result':
                    logger.info(f"Tool result step: {step.get('tool_name', 'unknown')} completed with result length {len(step.get('full_result', ''))}")
                self._add_step_to_task(task_id, step)

            # Use the integration's unified async method
            result = await self.llm_integration.get_response_async(
                session_id=session_id,
                max_iterations=20,  # Increased from default 5 for complex tasks
                track_steps=False,  # We're using callback instead
                step_callback=step_callback,
                temperature=0.0
            )

            # Check if stopped after completion
            if task_id in self.stopped_tasks:
                logger.info(f"Task {task_id} was stopped")
                self.chat_tasks[task_id].update({
                    "status": "stopped",
                    "response": "Processing stopped by user.",
                    "completed": True
                })
                return

            # DEBUG: Log final result details
            logger.info(f"Chat processing completed for task {task_id}: {result['iterations']} iterations, {result['tool_calls_made']} tool calls")
            logger.debug(f"Final response length: {len(result['response'])} characters")

            # Print timing summary
            self._print_timing_summary(timings, session_id)

            # Update task with final result
            self.chat_tasks[task_id].update({
                "status": "completed",
                "response": result["response"],
                "iterations": result["iterations"],
                "tool_calls_made": result["tool_calls_made"],
                "completed": True
            })

            # Auto-save conversation to logs
            logger.info(f"Starting auto-save for session {session_id}")
            save_result = self._save_conversation(session_id, self.logs_dir)
            if save_result.get("success"):
                logger.info(f"✅ Auto-saved conversation {session_id} to logs: {save_result.get('file')}")
                self.chat_tasks[task_id]["auto_saved"] = True
            else:
                logger.error(f"❌ Failed to auto-save conversation {session_id}: {save_result.get('error')}")
                self.chat_tasks[task_id]["auto_saved"] = False

            # Schedule cleanup after 10 minutes
            asyncio.create_task(self._cleanup_task_after_delay(task_id, 600))

        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was cancelled")

            # Print timing summary even when stopped
            self._print_timing_summary(timings, session_id)

            self.chat_tasks[task_id].update({
                "status": "stopped",
                "response": "Processing stopped by user.",
                "completed": True
            })

        except Exception as e:
            logger.error("Error in async chat processing: %s", e)
            self.chat_tasks[task_id].update({
                "status": "error",
                "error": str(e),
                "completed": True
            })

        finally:
            # Clean up tracking
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            if task_id in self.stopped_tasks:
                self.stopped_tasks.remove(task_id)

    def _add_step_to_task(self, task_id: str, step: dict):
        """Add a step to the task's step list."""
        if task_id in self.chat_tasks:
            self.chat_tasks[task_id]["steps"].append(step)

    def _init_timing_tracker(self) -> Dict[str, Any]:
        """Initialize a timing tracker for measuring operation times."""
        return {
            "assistant_times": [],  # List of individual assistant response times
            "tool_times": [],  # List of (tool_name, time) tuples
            "current_assistant_start": None,
            "current_tool_start": None,
            "current_tool_name": None
        }

    def _update_timing_from_step(self, timings: Dict[str, Any], step: Dict[str, Any]):
        """Update timing tracker based on step type."""
        step_type = step.get('type', 'unknown')

        if step_type == 'synthesizing':
            # After tool execution, before next LLM call - start timing
            timings['current_assistant_start'] = time.time()

        elif step_type == 'reasoning':
            # End of assistant response generation
            if timings['current_assistant_start'] is not None:
                assistant_time = time.time() - timings['current_assistant_start']
                timings['assistant_times'].append(assistant_time)
                timings['current_assistant_start'] = None

        elif step_type == 'tool_executing':
            # Start of tool execution
            timings['current_tool_start'] = time.time()
            timings['current_tool_name'] = step.get('tool_name', 'unknown')

        elif step_type == 'tool_result':
            # End of tool execution
            if timings['current_tool_start'] is not None:
                tool_time = time.time() - timings['current_tool_start']
                tool_name = timings['current_tool_name']
                timings['tool_times'].append((tool_name, tool_time))
                timings['current_tool_start'] = None
                timings['current_tool_name'] = None

    def _print_timing_summary(self, timings: Dict[str, Any], session_id: str):
        """Print timing summary for debugging."""
        total_assistant_time = sum(timings['assistant_times'])
        total_tool_time = sum(t for _, t in timings['tool_times'])

        print("\n" + "="*80)
        print(f"TIMING SUMMARY - Session {session_id}")
        print("="*80)
        print(f"\n1. ASSISTANT RESPONSE GENERATION:")
        print(f"   Total time: {total_assistant_time:.2f}s")
        print(f"   Number of calls: {len(timings['assistant_times'])}")
        if timings['assistant_times']:
            for i, t in enumerate(timings['assistant_times'], 1):
                print(f"   - Call {i}: {t:.2f}s")

        print(f"\n2. TOOL EXECUTIONS:")
        print(f"   Total time: {total_tool_time:.2f}s")
        print(f"   Number of tool calls: {len(timings['tool_times'])}")
        if timings['tool_times']:
            cumulative_tools = 0.0
            cumulative_assistant = 0.0
            assistant_idx = 0

            for i, (tool_name, t) in enumerate(timings['tool_times'], 1):
                # Update cumulative assistant time (tools happen after assistant calls)
                if assistant_idx < len(timings['assistant_times']):
                    cumulative_assistant += timings['assistant_times'][assistant_idx]
                    assistant_idx += 1

                # Update cumulative tool time
                cumulative_tools += t

                print(f"   - Tool {i} ({tool_name}): {t:.2f}s [cumulative: assistant {cumulative_assistant:.2f}s, tools {cumulative_tools:.2f}s]")

        print(f"\n3. OVERALL:")
        print(f"   Total assistant time: {total_assistant_time:.2f}s")
        print(f"   Total tool time: {total_tool_time:.2f}s")
        print(f"   Combined time: {total_assistant_time + total_tool_time:.2f}s")
        print("="*80 + "\n")

    async def _cleanup_task_after_delay(self, task_id: str, delay_seconds: int):
        """Clean up a task after a delay to prevent memory leaks."""
        await asyncio.sleep(delay_seconds)
        if task_id in self.chat_tasks:
            del self.chat_tasks[task_id]
            logger.debug("Cleaned up task %s", task_id)

    async def update_provider_and_model(self, provider: str, model: str):
        """Update the provider and model, reinitializing the integration."""
        logger.info("Updating from %s/%s to %s/%s", self.provider, self.model, provider, model)

        self.provider = provider
        self.model = model

        # Rebuild system prompt for new provider (sglang has different default)
        self.system_prompt = self._build_default_system_prompt()

        # Shutdown existing integration
        if self.llm_integration:
            # Clear all sessions
            for session_id in list(self.llm_integration.sessions.keys()):
                self.llm_integration.clear_session(session_id)
            self.llm_integration = None

        # Recreate integration with new provider/model
        await self.start_llm_integration()

        logger.info("Provider and model updated successfully")

    async def start_llm_integration(self):
        """Start or restart the LLM integration with current settings."""
        # Determine authentication based on provider
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self.llm_integration = create_tool_agent(
                self.toolkit,
                provider=self.provider,
                model=self.model,
                enable_variables=self.enable_variables,
                inject_variable_instructions=False,  # web_ui manages its own system prompt
                hide_tool_images=self.hide_tool_images,
                keep_dots_in_tool_names=self.keep_dots_in_tool_names,
                api_key=api_key
            )

        elif self.provider == "llm_gateway_openai":
            api_key = os.getenv("LLM_GATEWAY_TOKEN")
            if not api_key:
                raise ValueError("LLM_GATEWAY_TOKEN environment variable not set for NVIDIA LLM Gateway")

            self.llm_integration = create_tool_agent(
                self.toolkit,
                provider=self.provider,
                model=self.model,
                enable_variables=self.enable_variables,
                inject_variable_instructions=False,  # web_ui manages its own system prompt
                hide_tool_images=self.hide_tool_images,
                keep_dots_in_tool_names=self.keep_dots_in_tool_names,
                api_key=api_key
            )

        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")

            self.llm_integration = create_tool_agent(
                self.toolkit,
                provider=self.provider,
                model=self.model,
                enable_variables=self.enable_variables,
                inject_variable_instructions=False,  # web_ui manages its own system prompt
                hide_tool_images=self.hide_tool_images,
                keep_dots_in_tool_names=self.keep_dots_in_tool_names,
                api_key=api_key
            )

        elif self.provider == "bedrock":
            # Bedrock uses bearer token, not API key
            bearer_token = os.getenv("LLM_GATEWAY_TOKEN")
            if not bearer_token:
                raise ValueError("LLM_GATEWAY_TOKEN environment variable not set for Bedrock provider")

            self.llm_integration = create_tool_agent(
                self.toolkit,
                provider=self.provider,
                model=self.model,
                enable_variables=self.enable_variables,
                inject_variable_instructions=False,  # web_ui manages its own system prompt
                hide_tool_images=self.hide_tool_images,
                keep_dots_in_tool_names=self.keep_dots_in_tool_names,
                bearer_token=bearer_token
            )

        else:  # sglang
            # SGLang uses a base URL to connect to a running server, no authentication
            # Use model-specific URL if available, otherwise fall back to environment variable
            if self.model in self.sglang_model_urls:
                base_url = self.sglang_model_urls[self.model]
            else:
                base_url = os.getenv("SGLANG_BASE_URL", "http://localhost:34567/v1")

            logger.info(f"Connecting to sglang server at {base_url} for model {self.model}")

            self.llm_integration = create_tool_agent(
                self.toolkit,
                provider=self.provider,
                model=self.model,
                enable_variables=self.enable_variables,
                inject_variable_instructions=False,  # web_ui manages its own system prompt
                hide_tool_images=self.hide_tool_images,
                keep_dots_in_tool_names=self.keep_dots_in_tool_names,
                base_url=base_url
            )

        # Initialize enabled tools (all enabled by default)
        all_tools = self.llm_integration.get_all_available_tools()
        if not self.enabled_tools:  # Only initialize if empty
            self.enabled_tools = {tool: True for tool in all_tools}

        # Apply current tool filter
        def tool_filter(tool_schema):
            if self.provider in ["openai", "llm_gateway_openai", "sglang"]:
                # OpenAI, LLM Gateway OpenAI, and sglang use function format
                tool_name = tool_schema["function"]["name"]
            else:  # anthropic, bedrock
                tool_name = tool_schema["name"]
            return self.enabled_tools.get(tool_name, True)

        self.llm_integration.set_tool_filter(tool_filter)

        logger.info("LLM integration initialized with %s/%s", self.provider, self.model)
        logger.info("Variables enabled: %s, Images enabled: %s", self.enable_variables, self.enable_images)
        logger.info("Tools enabled: %d out of %d", sum(self.enabled_tools.values()), len(all_tools))

    async def start_toolshed(self):
        """Start toolshed and initialize the LLM integration."""
        logger.info("Starting toolshed with tools: %s", list(self.tool_configs.keys()))

        # Start the toolkit
        self.router = start_toolkit(
            self.tool_configs,
            router_name="llm_responses_demo_router",
            namespace="llm_responses_demo",
            dashboard=self.dashboard,
            dashboard_port=self.dashboard_port
        )

        # Get toolkit client
        self.toolkit = get_toolkit(
            router_name="llm_responses_demo_router",
            namespace="llm_responses_demo"
        )

        # Create LLM integration
        await self.start_llm_integration()

        logger.info("Toolshed and LLM integration initialized successfully")

    async def run(self):
        """Run the web server."""
        # Start toolshed first
        await self.start_toolshed()

        # Start web server
        print(f"Starting web server on port {self.port}")
        print("-" * 80)
        print(f"Open http://{self._get_local_ip()}:{self.port} in your browser")
        print("-" * 80)

        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLM Responses API + Toolshed Web Demo")
    parser.add_argument("--port", type=int, default=8001, help="Port for web server")
    parser.add_argument("--provider", choices=["openai", "llm_gateway_openai", "anthropic", "bedrock", "sglang"], default="bedrock", help="LLM provider to use")
    parser.add_argument("--model", help="Model to use (provider-specific default if not specified)")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file for tool configurations (relative to toolshed/configs/ or absolute path)")
    parser.add_argument("--exclude-tools", type=str, default="", help="Comma-separated list of tools to exclude (default: empty)")
    parser.add_argument("--no-variables", action="store_true", help="Disable variable output")
    parser.add_argument("--no-images", action="store_true", help="Disable image output")
    parser.add_argument("--hide-tool-images", action="store_true", help="Hide tool output images from LLM (but still compute and display them)")
    parser.add_argument("--logs-dir", type=str, default="logs/unsorted", help="Directory for auto-saved conversation logs (default: logs/unsorted)")
    parser.add_argument("--saved-conversations-dir", type=str, default="saved_conversations", help="Directory for manually saved conversations (default: saved_conversations)")
    parser.add_argument("--sglang-url", type=str, default="http://localhost:30000/v1", help="SGLang server URL (only for sglang provider)")
    parser.add_argument("--custom-system-prompt", type=str, default="", help="Additional text to append to the system prompt")
    parser.add_argument("--keep-dots-in-tool-names", action="store_true", help="Keep dots in tool names (e.g., 'vlm.detect_one') instead of converting to double underscores (e.g., 'vlm__detect_one'). Useful for fine-tuned models that expect dot notation.")
    parser.add_argument("--dashboard", action="store_true", help="Start the toolshed dashboard in the background")
    parser.add_argument("--dashboard-port", type=int, default=7001, help="Port for the toolshed dashboard (default: 7001)")

    args = parser.parse_args()

    # Check authentication requirements
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    elif args.provider == "llm_gateway_openai" and not os.getenv("LLM_GATEWAY_TOKEN"):
        print("Error: LLM_GATEWAY_TOKEN environment variable not set")
        print("Please set your NVIDIA LLM Gateway bearer token:")
        print("export LLM_GATEWAY_TOKEN='your-bearer-token-here'")
        exit(1)
    elif args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set your Anthropic API key:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        exit(1)
    elif args.provider == "bedrock" and not os.getenv("LLM_GATEWAY_TOKEN"):
        print("Error: LLM_GATEWAY_TOKEN environment variable not set")
        print("Please set your NVIDIA LLM Gateway bearer token:")
        print("export LLM_GATEWAY_TOKEN='your-bearer-token-here'")
        exit(1)
    elif args.provider == "sglang":
        # For sglang, set base URL from args if provided
        if args.sglang_url:
            os.environ["SGLANG_BASE_URL"] = args.sglang_url
        print(f"Using SGLang server at: {os.getenv('SGLANG_BASE_URL', 'http://localhost:30000/v1')}")
        print("Make sure the sglang server is running. See examples/SGLANG_PROVIDER_README.md for details.")

    print(f"Starting LLM demo with {args.provider} provider...")
    
    # Load tool configs from file or use default
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found at {config_path}")
            exit(1)
        
        print(f"Loading tool config from {config_path}")
        with open(config_path, 'r') as f:
            tool_configs = json.load(f)

    keep_tools = list(tool_configs.keys())
    all_tools = list(tool_configs.keys())
    if args.exclude_tools:
        exclude_tools = args.exclude_tools.split(",")
    else:
        exclude_tools = []
    keep_tools = [tool for tool in keep_tools if tool not in exclude_tools]
    tool_configs = {tool: tool_configs[tool] for tool in keep_tools}

    print(f"Keeping tools: {keep_tools}")
    print(f"All tools: {all_tools}")
    print(f"Excluding tools: {exclude_tools}")

    # Create and run demo
    demo = LLMResponsesWebDemo(
        tool_configs=tool_configs,
        provider=args.provider,
        model=args.model,
        port=args.port,
        enable_variables=not args.no_variables,
        enable_images=not args.no_images,
        hide_tool_images=args.hide_tool_images,
        keep_dots_in_tool_names=args.keep_dots_in_tool_names,
        logs_dir=args.logs_dir,
        saved_conversations_dir=args.saved_conversations_dir,
        custom_system_prompt=args.custom_system_prompt,
        dashboard=args.dashboard,
        dashboard_port=args.dashboard_port
    )

    await demo.run()


if __name__ == "__main__":
    # Check if web dependencies are available
    if not _WEB_DEPS_AVAILABLE:
        print("Error: Web demo requires FastAPI and uvicorn")
        print("Install with: pip install fastapi uvicorn python-multipart")
        exit(1)

    asyncio.run(main())
