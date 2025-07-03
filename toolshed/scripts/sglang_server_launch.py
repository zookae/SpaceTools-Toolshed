#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Launch sglang server for fine-tuned vision-language models.

This script starts an sglang server that hosts a fine-tuned VLM model
(e.g., Qwen2.5-VL) with tool calling capabilities. The server provides
an OpenAI-compatible API endpoint for inference.

Usage:
    # Start server with default settings
    python sglang_server_launch.py \\
        --model-path /path/to/qwen2_5vl-3b-alltasks-graspfulltools

    # Start with custom port and multiple GPUs
    python sglang_server_launch.py \\
        --model-path /path/to/model \\
        --port 30000 \\
        --tp-size 2 \\
        --host 0.0.0.0

    # With memory optimization
    python sglang_server_launch.py \\
        --model-path /path/to/model \\
        --mem-fraction-static 0.8

Example model paths:
    - /lustre/fsw/portfolios/nvr/users/siyic/projects/LLaMA-Factory/saves_v4-51-1/qwen2_5vl-3b-alltasks-graspfulltools/full/sft
    - /lustre/fsw/portfolios/nvr/users/siyic/projects/LLaMA-Factory/saves_v4-51-1/qwen2_5vl-3b-robos-bopaskpose-refspatial-rebalance-fulltools/full/sft

Server endpoints:
    - Health: http://localhost:30000/health
    - Chat: http://localhost:30000/v1/chat/completions (OpenAI-compatible)
    - Models: http://localhost:30000/v1/models
"""

import argparse
import subprocess
import sys
import os
import time
import requests
from pathlib import Path


def check_model_path(model_path: str) -> bool:
    """Verify that the model path exists and contains required files."""
    path = Path(model_path)
    
    if not path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return False
    
    # Check for config file (either config.json or configuration.json)
    if not (path / "config.json").exists() and not (path / "configuration.json").exists():
        print(f"Warning: No config.json found in {model_path}")
        print("This might not be a valid model directory")
    
    return True


def wait_for_server(host: str, port: int, timeout: int = 120) -> bool:
    """Wait for sglang server to become ready."""
    url = f"http://{host}:{port}/health"
    start_time = time.time()
    
    print(f"Waiting for server to become ready at {url}...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("✓ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
        print(".", end="", flush=True)
    
    print(f"\n✗ Server did not become ready within {timeout} seconds")
    return False


def launch_server(
    model_path: str,
    port: int = 34567,
    host: str = "127.0.0.1",
    tp_size: int = 1,
    mem_fraction_static: float = 0.9,
    context_length: int = 8192,
    trust_remote_code: bool = True,
    disable_radix_cache: bool = False,
    enable_torch_compile: bool = False,
    wait: bool = True,
):
    """
    Launch sglang server with the specified configuration.
    
    Args:
        model_path: Path to the fine-tuned model directory
        port: Port to run the server on
        host: Host address to bind to
        tp_size: Tensor parallelism size (number of GPUs)
        mem_fraction_static: Fraction of GPU memory to use for static allocation
        context_length: Maximum context length
        trust_remote_code: Whether to trust remote code in model
        disable_radix_cache: Disable RadixAttention cache (may improve compatibility)
        enable_torch_compile: Enable torch.compile for optimization
        wait: Wait for server to become ready before returning
    """
    
    # Check model path
    if not check_model_path(model_path):
        sys.exit(1)
    
    # Build sglang command
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--host", host,
        "--log-level", "debug",
        "--tp-size", str(tp_size),
        "--mem-fraction-static", str(mem_fraction_static),
        "--context-length", str(context_length),
    ]
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if disable_radix_cache:
        cmd.append("--disable-radix-cache")
    
    if enable_torch_compile:
        cmd.append("--enable-torch-compile")
    
    # Print launch info
    print("=" * 80)
    print("SGLang Server Configuration")
    print("=" * 80)
    print(f"Model path:        {model_path}")
    print(f"Server address:    http://{host}:{port}")
    print(f"Tensor parallel:   {tp_size} GPU(s)")
    print(f"Memory fraction:   {mem_fraction_static}")
    print(f"Context length:    {context_length}")
    print(f"Trust remote code: {trust_remote_code}")
    print("=" * 80)
    print(f"\nLaunching: {' '.join(cmd)}\n")
    print("=" * 80)
    
    # Set environment variables for better stability
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
    
    try:
        # Launch the server
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        if wait:
            # Wait for server to be ready
            if wait_for_server(host, port):
                print("\n" + "=" * 80)
                print("Server is ready for requests!")
                print("=" * 80)
                print(f"\nAPI endpoint: http://{host}:{port}/v1/chat/completions")
                print(f"Health check: http://{host}:{port}/health")
                print(f"List models:  http://{host}:{port}/v1/models")
                print("\nPress Ctrl+C to stop the server")
                print("=" * 80 + "\n")
        
        # Wait for process to complete (or be interrupted)
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        process.terminate()
        process.wait()
        print("Server stopped.")
    except Exception as e:
        print(f"\nError launching server: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Launch sglang server for fine-tuned VLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=34567,
        help="Port to run the server on (default: 34567)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1, use 0.0.0.0 for external access)"
    )
    
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size (number of GPUs, default: 1)"
    )
    
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.9,
        help="Fraction of GPU memory for static allocation (default: 0.9)"
    )
    
    parser.add_argument(
        "--context-length",
        type=int,
        default=12000,
        help="Maximum context length (default: 12000)"
    )
    
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Do not trust remote code in model (default: trust remote code)"
    )
    
    parser.add_argument(
        "--disable-radix-cache",
        action="store_true",
        help="Disable RadixAttention cache (may improve compatibility)"
    )
    
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="Enable torch.compile for optimization (may increase startup time)"
    )
    
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for server to become ready"
    )
    
    args = parser.parse_args()
    
    launch_server(
        model_path=args.model_path,
        port=args.port,
        host=args.host,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        trust_remote_code=not args.no_trust_remote_code,
        disable_radix_cache=args.disable_radix_cache,
        enable_torch_compile=args.enable_torch_compile,
        wait=not args.no_wait,
    )


if __name__ == "__main__":
    main()
