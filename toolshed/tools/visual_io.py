# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Visual IO tool
==============

Utility tool that lets the model move images between the conversation context
and the variable store in a consistent, minimal way:

- store_image(image, name?): stores the provided image under a variable name
  (returns variables=[{name, value}]) so it can be referenced later.

- view_image(image): returns the image inline so it is injected into the
  conversation by the rollout (handled via add_tool_response_messages), letting
  the model "see" it.

Both methods accept the heavyweight "image" argument. The schema exporter may
optionally expose a lightweight integer "image_index" argument that the rollout
will resolve to the heavyweight image before the tool call.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from toolshed.tools.base import BaseTool, tool_method
from toolshed.tool_result import ToolResult


class VisualIO(BaseTool):
    """Store and view images in a conversation."""

    def __init__(self, no_output_image: bool = False, no_output_vars: bool = False,
                 exclude_methods: list[str] = None, exclude_behavior: str = "warn"):
        super().__init__(no_output_image, no_output_vars, exclude_methods, exclude_behavior)
        if no_output_image or no_output_vars:
            print(f"WARNING! VisualIO tool initialized with no_output_image: {no_output_image} and no_output_vars: {no_output_vars}\n"
                   "This likely makes no sense and is probably a misconfiguration.")

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------
    def get_name(self) -> str:
        return "visual_io"

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------
    @tool_method
    def store_image(self, image: Any, name: Optional[str] = None) -> ToolResult[str]:
        """Store the given image into the variable context.

        [[if:text]]Text output: The variable name where the image was stored for future reference.[[/if:text]]
        [[if:vars]]Stored variables: The image stored with the specified name (or auto-generated name like 'var_image_abc123') for use in subsequent operations.[[/if:vars]]

        Args:
            image (Image): Image to store for later use.
            name (str, optional): Variable name to use. If omitted, a unique name will be generated.

        Returns:
            str: The variable name where the image was stored (e.g., 'var_image_abc123').
        """
        var_name = name if name else f"var_image_{uuid.uuid4().hex[:8]}"
        msg = f"Stored image as '{var_name}'. Reference it with ${var_name} in tool arguments."
        
        return ToolResult(
            var_name,  # The actual value for code execution
            text=msg,
            variables={var_name: image}
        )

    @tool_method
    def view_image(self, image: Any) -> ToolResult[Any]:
        """Return the image inline so it is visible to the model.

        [[if:text]]Text output: Confirmation that the image is being displayed.[[/if:text]]
        [[if:image]]Visual output: The requested image displayed in the conversation.[[/if:image]]

        Args:
            image (Image): Image to display. Can be a stored variable referenced with $var_name syntax (e.g., $my_image).

        Returns:
            Image: The image object that was passed in, now visible in the conversation.
        """
        return ToolResult(
            image,  # The actual value for code execution
            text="Displaying image.",
            image=image
        )


