# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Ray utility functions for automatic object store optimization.

This module provides utilities for transparently handling ray.put() and ray.get()
to simplify the developer experience while maintaining optimal performance.
"""

import sys
import logging
from typing import Any, Tuple, Dict

import ray

logger = logging.getLogger(__name__)

# Default threshold for automatic ray.put() (100KB)
DEFAULT_PUT_THRESHOLD = 100_000

# Default max depth for recursive size estimation
DEFAULT_MAX_DEPTH = 3

# Max list elements to sample for size estimation
MAX_LIST_SAMPLE_SIZE = 100


def _estimate_iterable_size(
    iterable, 
    length: int,
    max_depth: int, 
    _seen: set
) -> int:
    """Estimate size of an iterable, sampling if too large."""
    if length == 0:
        return 0
    
    # For large iterables, sample first N elements and extrapolate
    if length <= MAX_LIST_SAMPLE_SIZE:
        return sum(
            estimate_object_size(v, max_depth - 1, _seen) 
            for v in iterable
        )
    else:
        # Sample first N elements
        sample = list(iterable)[:MAX_LIST_SAMPLE_SIZE] if not isinstance(iterable, (list, tuple)) else iterable[:MAX_LIST_SAMPLE_SIZE]
        sample_size = sum(
            estimate_object_size(v, max_depth - 1, _seen) 
            for v in sample
        )
        avg_element_size = sample_size / MAX_LIST_SAMPLE_SIZE
        return int(avg_element_size * length)


def estimate_object_size(
    obj: Any, 
    max_depth: int = DEFAULT_MAX_DEPTH, 
    _seen: set = None
) -> int:
    """
    Recursively estimate the size of an object including nested contents.
    
    Handles numpy arrays, PIL Images, dicts, lists, and other common types.
    Includes infinite recursion detection and list sampling for performance.
    
    Args:
        obj: The object to estimate size for
        max_depth: Maximum recursion depth (default: 3)
        _seen: Internal set for tracking visited objects (infinite recursion detection)
        
    Returns:
        Estimated size in bytes
    """
    # Initialize seen set for tracking visited objects
    if _seen is None:
        _seen = set()
    
    # Check for max depth
    if max_depth <= 0:
        return 0
    
    # Check for infinite recursion (circular references)
    obj_id = id(obj)
    if obj_id in _seen:
        return 0  # Already counted this object
    
    # Add to seen set (only for mutable container types that could have circular refs)
    if isinstance(obj, (dict, list, set)):
        _seen.add(obj_id)
    
    try:
        # NumPy arrays - accurate size from nbytes
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.nbytes
        except ImportError:
            pass
        
        # PIL Images - always assume large (they almost always are)
        try:
            from PIL import Image
            if isinstance(obj, Image.Image):
                return DEFAULT_PUT_THRESHOLD + 1
        except ImportError:
            pass
        
        # Ray ObjectRef - already in object store
        if isinstance(obj, ray.ObjectRef):
            return 0
        
        # Dictionaries - recurse into values
        if isinstance(obj, dict):
            return sum(
                estimate_object_size(v, max_depth - 1, _seen) 
                for v in obj.values()
            )
        
        # Lists, tuples, sets - unified handling with sampling
        if isinstance(obj, (list, tuple, set)):
            return _estimate_iterable_size(obj, len(obj), max_depth, _seen)
        
        # Scalars and small types
        if isinstance(obj, (int, float, bool)):
            return 8
        
        if isinstance(obj, str):
            return len(obj.encode('utf-8'))
        
        if isinstance(obj, bytes):
            return len(obj)
        
        # Unknown types - fall back to sys.getsizeof
        try:
            return sys.getsizeof(obj)
        except:
            return 0
            
    finally:
        # Remove from seen set when done
        _seen.discard(obj_id)


def should_put(obj: Any, threshold: int = DEFAULT_PUT_THRESHOLD) -> bool:
    """
    Determine if an object should be placed in the Ray object store.
    
    Args:
        obj: The object to check
        threshold: Size threshold in bytes (default: 100KB)
        
    Returns:
        True if object should be put in object store
    """
    # Already a ref - skip
    if isinstance(obj, ray.ObjectRef):
        return False
    
    return estimate_object_size(obj) > threshold


def prepare_args_for_ray(
    args: tuple, 
    kwargs: dict, 
    threshold: int = DEFAULT_PUT_THRESHOLD
) -> Tuple[tuple, dict]:
    """
    Prepare arguments for Ray remote call by automatically putting large objects.
    
    Objects larger than the threshold are automatically placed in the Ray object
    store to avoid double serialization when passing through router → tool actor.
    
    Args:
        args: Positional arguments
        kwargs: Keyword arguments
        threshold: Size threshold in bytes (default: 100KB)
        
    Returns:
        Tuple of (processed_args, processed_kwargs)
    """
    if not ray.is_initialized():
        return args, kwargs
    
    def maybe_put(obj: Any) -> Any:
        if should_put(obj, threshold):
            return ray.put(obj)
        return obj
    
    new_args = tuple(maybe_put(arg) for arg in args)
    new_kwargs = {k: maybe_put(v) for k, v in kwargs.items()}
    
    return new_args, new_kwargs


def resolve_object_refs(args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
    """
    Resolve any Ray ObjectRefs in arguments to their actual values.
    
    This allows tool implementations to work with actual objects without
    needing to understand Ray internals.
    
    Args:
        args: Positional arguments (may contain ObjectRefs)
        kwargs: Keyword arguments (may contain ObjectRefs)
        
    Returns:
        Tuple of (resolved_args, resolved_kwargs)
    """
    def maybe_get(obj: Any) -> Any:
        if isinstance(obj, ray.ObjectRef):
            return ray.get(obj)
        return obj
    
    new_args = tuple(maybe_get(arg) for arg in args)
    new_kwargs = {k: maybe_get(v) for k, v in kwargs.items()}
    
    return new_args, new_kwargs

