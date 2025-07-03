# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Output format converters for evaluation results.

This module provides functions to convert conversation histories from various
LLM provider formats into standardized training data formats.
"""

from .verl import convert_messages_to_verl
from .sharegpt import convert_messages_to_sharegpt
from .sharegpt_simple import (
    convert_messages_to_sharegpt_simple,
    convert_tool_schemas_to_simple_format,
    convert_tool_name,
    extract_image_path_from_message,
    extract_all_image_paths_from_messages,
    build_training_example
)

__all__ = [
    'convert_messages_to_verl',
    'convert_messages_to_sharegpt',
    'convert_messages_to_sharegpt_simple',
    'convert_tool_schemas_to_simple_format',
    'convert_tool_name',
    'extract_image_path_from_message',
    'extract_all_image_paths_from_messages',
    'build_training_example'
]

