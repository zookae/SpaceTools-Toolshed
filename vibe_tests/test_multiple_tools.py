# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Test multiple tools working together - greeting and calculator.

NOTE: Tools now return ToolResult objects. When accessing tools via get_toolkit(),
the ToolResult is returned directly. Use result.value to get the underlying value.
The CodeExecutor automatically unwraps ToolResult, so code executed via exec() 
works with raw values.
"""

from toolshed import start_toolkit, get_toolkit, shutdown_toolkit, CodeExecutor
from toolshed.tool_result import ToolResult


def test_multiple_tools():
    """Test multiple tools working together."""
    print("=== Testing Multiple Tools ===\n")

    # Configuration for both tools
    tool_configs = {
        "greeting": {
            "num_actors": 1,
            "conda_env": None,
            "resources": {}
        },
        "calculator": {
            "num_actors": 2,
            "conda_env": None,
            "resources": {}
        }
    }

    # Initialize the toolkit
    print("Initializing toolkit with multiple tools...")
    router_actor = start_toolkit(tool_configs, router_name="multi_test", namespace="shed3d_multi_test")

    try:
        # Get toolkit client and code executor
        toolkit = get_toolkit(router_name="multi_test", namespace="shed3d_multi_test")
        executor = CodeExecutor(router_name="multi_test", namespace="shed3d_multi_test")

        print("Available tools:", toolkit.get_available_tools())
        print()

        # Test 1: Basic tool usage - tools return ToolResult via toolkit
        print("=== Test 1: Basic Tool Usage ===")
        greeting_result = toolkit.greeting.greet("Calculator User")
        assert isinstance(greeting_result, ToolResult), f"Expected ToolResult, got {type(greeting_result)}"
        print(f"Greeting: {greeting_result.value}")

        calc_result = toolkit.calculator.add(10, 5)
        assert isinstance(calc_result, ToolResult), f"Expected ToolResult, got {type(calc_result)}"
        print(f"10 + 5 = {calc_result.value}")

        calc_result = toolkit.calculator.multiply(3, 7)
        assert isinstance(calc_result, ToolResult), f"Expected ToolResult, got {type(calc_result)}"
        print(f"3 * 7 = {calc_result.value}")
        print()

        # Test 2: Code executor with multiple tools
        # Note: CodeExecutor unwraps ToolResult automatically
        print("=== Test 2: Code Executor with Multiple Tools ===")
        code = """
# Use both tools in a single code block
name = "Alice"
greeting = toolkit.greeting.greet(name)
print(f"Got greeting: {greeting}")

# Do some calculations
x = 15
y = 3
sum_result = toolkit.calculator.add(x, y)
product = toolkit.calculator.multiply(x, y)
quotient = toolkit.calculator.divide(x, y)

result = {
    'greeting': greeting,
    'calculations': {
        'sum': sum_result,
        'product': product,
        'quotient': quotient
    }
}
"""
        result, stdout, stderr = executor.exec(code)
        print("Code executed:")
        print(f"Result: {result}")
        print(f"Output: {stdout}")
        if stderr:
            print(f"Errors: {stderr}")
        print()

        # Test 3: Error handling
        print("=== Test 3: Error Handling ===")
        code = """
try:
    result = toolkit.calculator.divide(10, 0)
except ValueError as e:
    result = f"Caught error: {e}"
"""
        result, stdout, stderr = executor.exec(code)
        print(f"Division by zero handling: {result}")
        print()

        # Test 4: Tool statistics - get_stats returns dict directly
        print("=== Test 4: Tool Statistics ===")
        greeting_stats = toolkit.greeting.get_stats()
        calc_stats = toolkit.calculator.get_stats()
        print(f"Greeting stats: {greeting_stats}")
        print(f"Calculator stats: {calc_stats}")
        print()

        # Test 5: Documentation
        print("=== Test 5: Documentation ===")
        documentation = toolkit.get_documentation()
        print("Generated Documentation:")
        print("=" * 60)
        print(documentation)
        print("=" * 60)

        print("\n✅ All multi-tool tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up
        print("\nShutting down toolkit...")
        _ = router_actor
        shutdown_toolkit(router_name="multi_test", namespace="shed3d_multi_test")
        print("Done!")


if __name__ == "__main__":
    test_multiple_tools()
