# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

"""Optional Weave tracing for Toolshed‒Verl integration.

Call ``enable_weave_tracing()`` once at program start (after importing
``toolshed.integration.verl``) to decorate the async lifecycle methods
of ``ToolshedMethodTool`` and ``ToolshedCodeTool`` with ``@weave.op`` so
that every call is automatically captured by Weave.

Importing this module has *no* side-effects – tracing is only enabled
when the helper is invoked, which makes behaviour explicit and
predictable.
"""

import inspect
import os
from typing import TYPE_CHECKING

# -----------------------------------------------------------------------------
# Optional dependency handling – Weave is not required to use Toolshed.
# -----------------------------------------------------------------------------
try:
    import weave  # type: ignore
except ImportError:  # pragma: no cover – weave not installed
    weave = None  # type: ignore

__all__ = [
    "enable_weave_tracing",
    "initialize_ray_worker_tracing",
]


def _is_already_wrapped(func):
    """Check if a function has already been wrapped by weave.op."""
    return getattr(func, "__weave_wrapped__", False)


def enable_weave_tracing(
    project_name: str | None = None,
    experiment_name: str = "toolshed_weave",
) -> None:  # noqa: C901 – keep simple signature
    """Enable Weave tracing for Toolshed‐Verl integration.

    Configures Verl's tracing backend.  For calls that go through Verl's
    Toolshed wrappers this is sufficient because their lifecycle methods are
    already decorated with ``@rollout_trace_op`` inside the Verl codebase.

    Parameters
    ----------
    project_name: str | None, optional
        Weave project name to initialise.  If *None* we read the environment
        variable ``WEAVE_PROJECT`` and fall back to "toolshed".
    experiment_name: str, optional
        Label propagated to Weave traces via ``rollout_trace_attr`` so you can
        group multiple runs under the same logical experiment.
    """

    # Early exit if tracing explicitly disabled.
    if os.getenv("TOOLSHED_DISABLE_WEAVE", "false").lower() == "true":
        return

    # Configure Verl's rollout tracing to forward to Weave (if not already).
    try:
        from verl.utils.rollout_trace import RolloutTraceConfig  # type: ignore

        if RolloutTraceConfig.get_backend() is None:
            _proj = project_name or os.getenv("WEAVE_PROJECT", "toolshed")
            RolloutTraceConfig.init(
                project_name=_proj,
                experiment_name=experiment_name,
                backend="weave",
                token2text=False,
            )
    except Exception:
        pass


def initialize_ray_worker_tracing(project_name: str | None = None, experiment_name: str | None = None) -> None:
    """Initialize Weave tracing configuration for Ray workers.
    
    This function sets environment variables that will be read by Ray workers
    when they lazily initialize Weave tracing on first tool call.
    
    Parameters
    ----------
    project_name: str | None, optional
        Weave project name. If None, uses existing WEAVE_PROJECT env var.
    experiment_name: str | None, optional  
        Experiment name for grouping traces. If None, uses existing WEAVE_EXPERIMENT
        env var or defaults to "rollout_worker".
    """
    if project_name:
        os.environ["WEAVE_PROJECT"] = project_name
    if experiment_name:
        os.environ["WEAVE_EXPERIMENT"] = experiment_name
    
    # Also initialize in the current process if not already done
    enable_weave_tracing(project_name, experiment_name or "main_process") 