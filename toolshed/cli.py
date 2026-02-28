#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Toolshed CLI tools.

This module provides CLI commands for:
- toolshed-launch: Launch the Toolshed toolkit with tools configured via JSON
- toolshed-export-schemas: Export OpenAI-compatible tool schemas from running toolkit
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import ray


def load_config(config_path: str) -> dict:
    """Load tool configuration from JSON file.
    
    Expected format:
    {
        "tool_name": {
            "num_actors": 32,
            "conda_env": null,
            "resources": {},
            "timeout": 5
        },
        ...
    }
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        Dictionary of tool configurations
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a JSON object with tool configurations")
    
    return config


def main():
    """Main entry point for toolshed-launch CLI."""
    parser = argparse.ArgumentParser(
        description="Launch Toolshed toolkit with configured tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example config file format (JSON):
{
    "calculator": {
        "num_actors": 32,
        "conda_env": null,
        "resources": {},
        "timeout": 5
    },
    "greeting": {
        "num_actors": 4,
        "timeout": 10
    }
}

Example usage:
  toolshed-launch --config tools.json --dashboard
  toolshed-launch --config tools.json --dashboard-port 8080
  toolshed-launch --config tools.json --namespace my_namespace --router-name my_router
  toolshed-launch --config tools.json --detached
  toolshed-launch --config tools.json --placement-group-size 8  # Constrain to one 8-GPU node
  toolshed-launch --config tools.json --placement-group-size auto  # Auto-calculate from config
  toolshed-launch --config tools.json --ray-address auto  # Connect to existing Ray cluster
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to JSON config file with tool configurations'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Enable Toolshed dashboard'
    )
    
    parser.add_argument(
        '--dashboard-port',
        type=int,
        default=7001,
        help='Port for dashboard server (default: 7001)'
    )
    
    parser.add_argument(
        '--detached',
        action='store_true',
        help='Run toolkit in detached mode (actors persist after script exits)'
    )
    
    parser.add_argument(
        '--ray-address',
        type=str,
        default=None,
        help='Ray cluster address. Use "auto" to connect to an existing cluster. '
             'If not specified, a local Ray instance is started automatically (matching Python API behavior).'
    )
    
    parser.add_argument(
        '--namespace',
        type=str,
        default='toolshed',
        help='Ray namespace for cluster-wide discovery (default: "toolshed")'
    )
    
    parser.add_argument(
        '--router-name',
        type=str,
        default='toolshed_router',
        help='Name for the router actor (default: "toolshed_router")'
    )
    
    parser.add_argument(
        '--placement-group-size',
        type=str,
        default=None,
        help='Size for automatic placement group creation. Can be an integer (number of GPUs) '
             'or "auto" to calculate from tool configs. Use this to constrain tools to a single node.'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"Loading config from: {args.config}")
        tool_configs = load_config(args.config)
        print(f"Found {len(tool_configs)} tool(s) to launch: {list(tool_configs.keys())}")
        
        # Initialize Ray — match Python API behavior:
        # if --ray-address is given, connect to that cluster;
        # otherwise auto-start a local Ray instance.
        if not ray.is_initialized():
            if args.ray_address:
                print(f"Connecting to Ray cluster at '{args.ray_address}'...")
                ray.init(address=args.ray_address, log_to_driver=True)
            else:
                print("Starting local Ray instance...")
                ray.init(log_to_driver=True)
        print("Connected to Ray successfully")
        
        # Import here to ensure Ray is initialized first
        from toolshed import start_toolkit
        
        # Parse placement_group_size
        placement_group_size = None
        if args.placement_group_size is not None:
            if args.placement_group_size.lower() == 'auto':
                placement_group_size = 'auto'
            else:
                try:
                    placement_group_size = int(args.placement_group_size)
                except ValueError:
                    print(f"ERROR: --placement-group-size must be an integer or 'auto', got: {args.placement_group_size}", file=sys.stderr)
                    sys.exit(1)
        
        # Start the toolkit
        print("Launching configured tools...")
        router = start_toolkit(
            tool_configs,
            router_name=args.router_name,
            namespace=args.namespace,
            detached=args.detached,
            dashboard=args.dashboard,
            dashboard_port=args.dashboard_port,
            placement_group_size=placement_group_size
        )
        print(f"✓ Toolshed started successfully with {len(tool_configs)} tool(s)")
        print(f"  Router name: {args.router_name}")
        print(f"  Namespace: {args.namespace}")
        if placement_group_size is not None:
            print(f"  Placement group size: {placement_group_size}")
        
        if args.dashboard:
            print(f"✓ Dashboard available at http://localhost:{args.dashboard_port}")
        
        if args.detached:
            print("✓ Toolkit running in detached mode - actors will persist after exit")
            print("  Use ray.kill() or restart Ray to cleanup actors")
        else:
            print("✓ Toolkit running in attached mode")
            print("  Press Ctrl+C to stop and cleanup actors")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                print("\nShutting down...")
                
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to start toolkit: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def export_schemas():
    """Main entry point for toolshed-export-schemas CLI."""
    parser = argparse.ArgumentParser(
        description="Generate OpenAI-compatible tool schemas from running Toolshed actors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This command queries a running Toolshed router and exports tool schemas to YAML format.
The schemas are compatible with OpenAI tool calling and can be used by LLM frameworks.

Example usage:
  toolshed-export-schemas -o tools.yaml
  toolshed-export-schemas -o tools.yaml --include-code-executor
  toolshed-export-schemas -o tools.yaml --trace-project my_project --trace-experiment exp1
        """
    )
    
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=Path('toolshed_tool_config.yaml'),
        help='Destination YAML file (default: toolshed_tool_config.yaml)'
    )
    
    parser.add_argument(
        '--router-name',
        default='toolshed_router',
        help='Router actor name (default: toolshed_router)'
    )
    
    parser.add_argument(
        '--namespace',
        default='toolshed',
        help='Ray namespace (default: toolshed)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout value stored in generated tool configs (default: 30)'
    )
    
    parser.add_argument(
        '--include-code-executor',
        action='store_true',
        help='Include the special execute_python tool'
    )
    
    parser.add_argument(
        '--use-image-by-index',
        action='store_true',
        help='Expose image_index integer instead of heavy image parameter'
    )
    
    parser.add_argument(
        '--trace-project',
        help='Weave project name for tracing (passed to tool configs)'
    )
    
    parser.add_argument(
        '--trace-experiment',
        help='Weave experiment name for tracing (passed to tool configs)'
    )
    
    args = parser.parse_args()
    
    try:
        # Connect to Ray cluster if not already connected
        if not ray.is_initialized():
            print("Connecting to Ray cluster...")
            ray.init(address='auto', ignore_reinit_error=True)
            print("Connected to Ray successfully")
        
        # Import yaml here to avoid requiring it for toolshed-launch
        try:
            import yaml
        except ImportError:
            print("ERROR: PyYAML is required for this command", file=sys.stderr)
            print("Install it with: pip install pyyaml", file=sys.stderr)
            sys.exit(1)
        
        # Wait for router to be available
        router = None
        first_wait_msg_printed = False
        last_print_time = 0.0
        
        print(f"Looking for router '{args.router_name}' in namespace '{args.namespace}'...")
        while router is None:
            try:
                router = ray.get_actor(args.router_name, namespace=args.namespace)
            except ValueError:
                # Actor not found yet – wait and inform user periodically
                current_time = time.monotonic()
                if not first_wait_msg_printed or (current_time - last_print_time) > 30:
                    print(
                        f"Router '{args.router_name}' not found – waiting for it to become available..."
                    )
                    first_wait_msg_printed = True
                    last_print_time = current_time
                time.sleep(5)
                continue
        
        print(f"Found router '{args.router_name}'")
        
        # Wait for router to be ready
        first_ready_msg_printed = False
        last_ready_print_time = 0.0
        while True:
            try:
                ready = ray.get(router.is_ready.remote())
            except Exception:
                ready = False
            
            if ready:
                break
            
            current_time = time.monotonic()
            if not first_ready_msg_printed or (current_time - last_ready_print_time) > 30:
                print("Router found but not ready – waiting for initialization to complete...")
                first_ready_msg_printed = True
                last_ready_print_time = current_time
            time.sleep(5)
        
        # Get available tools
        available_tools = ray.get(router.get_available_tools.remote())
        print(f"Found available tools: {available_tools}")
        time.sleep(1)
        
        # Export schemas
        print("Exporting tool schemas...")
        schemas: List[dict] = ray.get(
            router.export_openai_schemas.remote(
                timeout=args.timeout,
                include_code_executor=args.include_code_executor,
                use_image_by_index=args.use_image_by_index,
            )
        )
        
        # Add trace configuration to each tool if provided
        if args.trace_project or args.trace_experiment:
            for schema in schemas:
                if "config" not in schema:
                    schema["config"] = {}
                if args.trace_project:
                    schema["config"]["project_name"] = args.trace_project
                if args.trace_experiment:
                    schema["config"]["experiment_name"] = args.trace_experiment
        
        # Write YAML output
        data = {"tools": schemas}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fp:
            yaml.safe_dump(data, fp, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Wrote {len(schemas)} tool schemas to {args.output}")
        
    except Exception as e:
        print(f"ERROR: Failed to export schemas: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
