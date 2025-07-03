# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Toolshed Evaluation Framework

This module provides tools for evaluating language models on various datasets,
with support for models that use Toolshed's vision tools.
"""

from .models import BaseModel, UnifiedModel, OpenAIToolshedModel, SimpleOpenAIModel
from .common import BaseEvaluator
from .robospatial import RoboSpatialEvaluator
from .refspatial import RefSpatialEvaluator
from .bop_ask import BopAskEvaluator
from toolshed.formats import (
    convert_messages_to_verl,
    convert_messages_to_sharegpt,
    convert_messages_to_sharegpt_simple,
    convert_tool_schemas_to_simple_format,
    convert_tool_name,
    extract_image_path_from_message,
    extract_all_image_paths_from_messages,
    build_training_example
)

# Re-export formats for backward compatibility
import toolshed.formats as formats

__all__ = [
    'BaseModel',
    'UnifiedModel',
    'OpenAIToolshedModel', 
    'SimpleOpenAIModel',
    'BaseEvaluator',
    'RoboSpatialEvaluator',
    'RefSpatialEvaluator',
    'BopAskEvaluator',
    'formats'
]
