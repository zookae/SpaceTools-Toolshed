# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Simple usage example for the CodeExecutor.

This demonstrates the exact API pattern requested by the user:
    from toolshed import CodeExecutor
    executor = CodeExecutor()
    result, stdout, stderr = executor.eval(code_block)
"""

import argparse
import json
import sys
from pathlib import Path

from toolshed import start_toolkit, CodeExecutor, shutdown_toolkit


def main():
    """Demonstrate the CodeExecutor usage pattern."""

    parser = argparse.ArgumentParser(description="CodeExecutor usage example")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file for tool configurations. "
                             "If provided, overrides the inline tool configs (including conda_env names).")
    args = parser.parse_args()

    # Setup the toolkit first (this would typically be done once per cluster)
    tool_configs = {
        "greeting": {"num_actors": 2, "resources": {"num_cpus": 0, "num_gpus": 0}},
        "calculator": {"num_actors": 2, "resources": {"num_cpus": 0, "num_gpus": 0}}
    }

    # Override with JSON config if provided
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found at {config_path}")
            sys.exit(1)
        with open(config_path, "r") as f:
            tool_configs = json.load(f)

    actor = start_toolkit(tool_configs)
    
    try:
        # Create a code executor:
        executor = CodeExecutor()
        # Retrieve the toolkit documentation (this should be given to the LLM):
        docs = executor.get_toolkit().get_documentation()
        print("\n\nToolkit documentation:")
        print(docs)
        print("\n\n")

        # Example 1: Eval mode - returns result value
        print("=== EXAMPLE 1 ===")
        
        code_block = "toolkit.greeting.greet('World')"
        result, stdout, stderr = executor.exec(code_block)
        print(f"Code: {code_block}")
        print(f"Result: {result}")
        print(f"Stdout: {repr(stdout)}")
        print(f"Stderr: {repr(stderr)}")
        print("\n")
        
        # Example 2: More complex eval ("result" variable is returned as the result)
        print("=== EXAMPLE 2 ===")

        example_llm_response = """I will now greet the user and perform some calculations.
        ```python
# Greet the user
name = "Alice"
greeting = toolkit.greeting.greet(name)
print(f"Greeting: {greeting}")

# Use calculator to perform some calculations
x = 15
y = 3
sum_result = toolkit.calculator.add(x, y)
product = toolkit.calculator.multiply(x, y)
quotient = toolkit.calculator.divide(x, y)

print(f"Calculations for {x} and {y}:")
print(f"  Sum: {sum_result}")
print(f"  Product: {product}")
print(f"  Quotient: {quotient}")

# Return a summary as the result
result = {
    'calculations': {
        'sum': sum_result,
        'product': product,
        'quotient': quotient
    }
}
        ```
        """

        code_block = executor.extract_python_code_from_markdown(example_llm_response)

        result, stdout, stderr = executor.exec(code_block)
        print(f"Code: {repr(code_block)}")
        print(f"Result: {result}")
        print(f"Stdout: {repr(stdout)}")
        print(f"Stderr: {repr(stderr)}")
        print()
        
        print("\n✅ All examples completed successfully!")
        
    finally:
        # Clean up
        shutdown_toolkit()


if __name__ == "__main__":
    main() 