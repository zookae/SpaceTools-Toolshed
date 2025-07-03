# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from toolshed import start_toolkit
from time import sleep

if __name__ == "__main__":
    # Configure and start the toolkit
    tool_configs = {
        "greeting": {"num_actors": 2, "resources": {"num_cpus": 1, "num_gpus": 0}},
        "calculator": {"num_actors": 2, "resources": {"num_cpus": 1, "num_gpus": 0}}
    }
    actor_handle = start_toolkit(tool_configs)

    while True:
        # Sleep forever to keep actor_handle alive.
        sleep(1)
