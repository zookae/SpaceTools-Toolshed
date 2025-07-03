# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Runtime tool registry built on top of a static *manifest*.

Key points:
•  Lightweight – drivers don't import heavy ML packages.
•  Deterministic – one authoritative mapping from tool name ➜ class path.
•  Extensible – users can specify ``import_path`` in tool config for external tools.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, List, Optional, Type

from .tool_manifest import TOOL_MANIFEST  # default static manifest
from .tools.base import BaseTool

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# In-memory caches
# -----------------------------------------------------------------------------

# Resolved classes that have already been imported (name ➜ class)
_CLASS_CACHE: Dict[str, Type[BaseTool]] = {}


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------


def get_tool_class(
    tool_name: str,
    *,
    import_path: Optional[str] = None,
    allow_refresh: bool = True,
) -> Type[BaseTool]:
    """Return the concrete class for *tool_name*.

    Parameters
    ----------
    tool_name
        Symbolic name such as ``"vlm"``.
    import_path
        Explicit import path in ``"module.path:ClassName"`` format.
        If provided, this takes precedence over the static manifest.
        Use this for external tools not in the built-in manifest.
    allow_refresh
        If *True* and the manifest entry fails to resolve (e.g. the module
        isn't on *PYTHONPATH* yet) we make a best-effort attempt to import a
        module named ``toolshed.tools.<tool_name>`` and retry once.  This helps
        during local development when new tool files are added but the static
        manifest hasn't been updated yet.
    """

    # 1. Fast path – already loaded (only if no explicit import_path override)
    if import_path is None and tool_name in _CLASS_CACHE:
        return _CLASS_CACHE[tool_name]

    # 2. Determine import target: explicit import_path > static manifest
    if import_path is not None:
        target = import_path
    else:
        try:
            target = TOOL_MANIFEST[tool_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown tool '{tool_name}'. Available: {sorted(list_available_tools())}. "
                f"For external tools, specify 'import_path' in the tool config."
            ) from exc

    module_path, _, attr = target.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if allow_refresh:
            # Development convenience: attempt to import toolshed.tools.<name>
            # then retry.  If it still fails we propagate the original error.
            try:
                importlib.import_module(f"toolshed.tools.{tool_name}")
                module = importlib.import_module(module_path)
            except Exception:  # pragma: no cover – propagate original
                raise exc
        else:
            raise

    try:
        cls = getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover – manifest typo
        raise ImportError(
            f"Manifest entry for '{tool_name}' points to missing attribute '{attr}'"
        ) from exc

    if not issubclass(cls, BaseTool):  # pragma: no cover – manifest error
        raise TypeError(
            f"Resolved class for '{tool_name}' is not a subclass of BaseTool"
        )

    _CLASS_CACHE[tool_name] = cls
    return cls


def list_available_tools() -> List[str]:
    """Return the list of built-in tool names from the manifest."""
    return sorted(TOOL_MANIFEST.keys())


def clear_registry() -> None:  # pragma: no cover – mainly for tests
    """Clear all caches."""
    _CLASS_CACHE.clear() 