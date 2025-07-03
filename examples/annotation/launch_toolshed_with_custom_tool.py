#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Launch toolshed with a custom save_data tool.

This script starts toolshed separately so the annotation pipeline can be run
multiple times without waiting for tools to launch each time.

The save_data tool is an external tool defined in examples/annotation/tool_save_data.py.
It is configured in annotation_config.json with an import_path that tells toolshed
where to find the tool class.
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from toolshed import start_toolkit


def main():
    parser = argparse.ArgumentParser(description="Launch toolshed with custom save_data tool")
    parser.add_argument(
        "--config",
        default="examples/annotation/annotation_config.json",
        help="Path to tool config JSON file"
    )
    parser.add_argument(
        "--save-path",
        default="outputs/annotation",
        help="Path to save annotations"
    )
    
    args = parser.parse_args()
    
    # Load tool config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return 1
    
    with open(config_path, 'r') as f:
        tool_configs = json.load(f)
    
    # Update save_path in config (use absolute path for Ray workers)
    tool_configs["save_data"]["args"]["save_path"] = str(Path(args.save_path).resolve())
    
    print(f"Starting toolkit with {len(tool_configs)} tools...")
    handle = start_toolkit(tool_configs, detached=False, dashboard=True, dashboard_port=7001)
    
    print("Toolshed is running. Press Ctrl+C to stop.")
    print("You can now run the annotation pipeline multiple times.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down toolshed...")
        return 0


if __name__ == "__main__":
    exit(main())
