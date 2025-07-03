# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Clean demo: Weave-traced calls to Toolshed tools via Verl wrappers.

This demonstrates the clean approach to enabling Weave tracing without
sitecustomize.py or other hacks. The tracing is initialized explicitly
and Ray workers will automatically initialize on first tool call.
"""

from __future__ import annotations

import os
import pprint

import ray
from toolshed import start_toolkit, shutdown_toolkit
from toolshed.integration.weave_trace import initialize_ray_worker_tracing
from toolshed.integration.verl import get_toolshed_tool_wrappers


def main() -> None:
    # 1) Initialize Weave tracing configuration
    # This sets up environment variables and initializes the main process
    project_name = "toolshed_weave_demo"
    initialize_ray_worker_tracing(
        project_name=project_name,
        experiment_name="demo_run"
    )
    print(f"Initialized Weave tracing for project: {project_name}")

    # 2) Connect to Ray cluster
    ray.init()

    # 3) Start toolkit
    tool_configs = {
        'calculator': {'num_actors': 2},
        'greeting': {'num_actors': 1},
        'code_executor': {'num_actors': 1},
    }
    
    print("Starting Toolshed toolkit...")
    handle = start_toolkit(tool_configs, detached=False)

    # 4) Get tool wrappers
    wrappers = get_toolshed_tool_wrappers()
    print("\nAvailable tools:")
    pprint.pp(sorted(wrappers.keys()))

    # 5) Call some tools - Weave tracing will be initialized on first call
    calc_add = wrappers["calculator.add"]
    greet_tool = wrappers["greeting.greet"]
    
    print("\nCalling calculator.add(5, 7)...")
    result = calc_add(a=5, b=7)
    print(f"Result: {result}")
    
    print("\nCalling greeting.greet('World')...")
    msg = greet_tool(name="World")
    print(f"Result: {msg}")

    # 6) Cleanup
    shutdown_toolkit()
    print("\nDone! Check your Weave project for traces.")


if __name__ == "__main__":
    main() 