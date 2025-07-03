# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Documentation processing utilities for toolshed.

This module provides utilities for processing documentation strings,
including conditional documentation blocks that adapt based on tool configuration.
"""


def process_conditional_docs(
    text: str,
    *,
    include_image: bool = True,
    include_vars: bool = True,
    include_text: bool = True,
) -> str:
    """Process [[if:*]] conditional blocks in documentation text.
    
    Args:
        text: The documentation text to process
        include_image: Whether to include [[if:image]] blocks
        include_vars: Whether to include [[if:vars]] blocks  
        include_text: Whether to include [[if:text]] blocks
        
    Returns:
        Processed text with conditional blocks resolved
    """
    import re

    def _apply(block: str, include: bool) -> None:
        nonlocal text
        if include:
            text = re.sub(fr'\[\[if:{block}\]\](.*?)\[\[/if:{block}\]\]', r'\1', text, flags=re.DOTALL)
        else:
            # Handle conditionals with optional leading whitespace and optional trailing whitespace + newline
            # This catches cases where the conditional is on its own line or at end of line
            text = re.sub(fr'[ \t]*\[\[if:{block}\]\].*?\[\[/if:{block}\]\][ \t]*\n?', '', text, flags=re.DOTALL)

    _apply('image', include_image)
    _apply('vars', include_vars)
    _apply('text', include_text)

    text = re.sub(r' +', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    return text.strip()
