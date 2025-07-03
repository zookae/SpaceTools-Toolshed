# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Test and demonstration of the CodeExecutor functionality.

This script demonstrates how to use the CodeExecutor to execute Python code
with access to the toolshed toolkit in both eval and exec modes.
"""

from toolshed import start_toolkit, CodeExecutor, shutdown_toolkit


def test_code_executor():
    """Test the CodeExecutor with various code examples."""
    print("=== toolshed CodeExecutor Test ===\n")

    # Configuration for tools
    tool_configs = {
        "greeting": {
            "num_actors": 2,
            "conda_env": None,
            "resources": {}
        }
    }

    # Initialize the toolkit
    print("Initializing toolshed toolkit...")
    router_actor = start_toolkit(tool_configs, router_name="code_executor_test", namespace="shed3d_test")

    try:
        # Create the code executor
        print("Creating CodeExecutor...")
        executor = CodeExecutor(router_name="code_executor_test", namespace="shed3d_test")

        print(f"Available tools: {executor.get_available_tools()}")
        print()

        # Test 1: Simple expression
        print("1. Testing simple expression:")
        code = "toolkit.greeting.greet('Alice')"
        result, stdout, stderr = executor.exec(code)
        print(f"   Code: {code}")
        print(f"   Result: {result}")
        print(f"   Stdout: {repr(stdout)}")
        print(f"   Stderr: {repr(stderr)}")
        print()

        # Test 2: Multiple statements with result variable
        print("2. Testing multiple statements with result variable:")
        code = """
name = 'Bob'
greeting = toolkit.greeting.greet(name)
result = f"Processed: {greeting}"
"""
        result, stdout, stderr = executor.exec(code)
        print(f"   Code: {repr(code)}")
        print(f"   Result: {result}")
        print(f"   Stdout: {repr(stdout)}")
        print(f"   Stderr: {repr(stderr)}")
        print()

        # Test 3: Code with print statements (no result variable)
        print("3. Testing code with print statements:")
        code = """
print("Starting greeting test...")
for name in ['Charlie', 'Diana']:
    greeting = toolkit.greeting.greet(name)
    print(f"Greeting for {name}: {greeting}")
print("Test completed!")
"""
        result, stdout, stderr = executor.exec(code)
        print(f"   Code: {repr(code)}")
        print(f"   Result: {result}")
        print(f"   Stdout: {repr(stdout)}")
        print(f"   Stderr: {repr(stderr)}")
        print()

        # Test 4: Math calculations with toolkit
        print("4. Testing math calculations with toolkit:")
        code = """
import math
names = ['Eve', 'Frank', 'Grace']
greetings = [toolkit.greeting.greet(name) for name in names]
result = {
    'total_greetings': len(greetings),
    'greetings': greetings,
    'calculation': math.pi * len(greetings)
}
"""
        result, stdout, stderr = executor.exec(code)
        print(f"   Code: {repr(code)}")
        print(f"   Result: {result}")
        print(f"   Stdout: {repr(stdout)}")
        print(f"   Stderr: {repr(stderr)}")
        print()

        # Test 5: Error handling
        print("5. Testing error handling:")
        code = "toolkit.nonexistent_tool.fake_method('test')"
        result, stdout, stderr = executor.exec(code)
        print(f"   Code: {code}")
        print(f"   Result: {result}")
        print(f"   Stdout: {repr(stdout)}")
        print(f"   Stderr: {repr(stderr)}")
        print()

        # Test 6: Using context variables
        print("6. Testing context variables:")
        executor.update_context({"custom_name": "Henry", "multiplier": 3})
        code = """
results = []
for i in range(multiplier):
    greeting = toolkit.greeting.greet(f"{custom_name}_{i+1}")
    results.append(greeting)
result = results
"""
        result, stdout, stderr = executor.exec(code)
        print(f"   Code: {repr(code)}")
        print(f"   Result: {result}")
        print(f"   Stdout: {repr(stdout)}")
        print(f"   Stderr: {repr(stderr)}")
        print()

        # Test 7: Tool statistics
        print("7. Testing tool statistics:")
        code = "toolkit.greeting.get_stats()"
        result, stdout, stderr = executor.exec(code)
        print(f"   Code: {code}")
        print(f"   Result: {result}")
        print(f"   Stdout: {repr(stdout)}")
        print(f"   Stderr: {repr(stderr)}")
        print()

        # Test 8: Complex code with functions
        print("8. Testing complex code with function definition:")
        code = """
def batch_greet(names, formal=False):
    results = []
    for name in names:
        if formal:
            greeting = toolkit.greeting.formal_greet(name, "Dr.")
        else:
            greeting = toolkit.greeting.greet(name)
        results.append(greeting)
    return results

test_names = ['Isaac', 'Julia']
informal = batch_greet(test_names, formal=False)
formal = batch_greet(test_names, formal=True)

result = {
    'informal': informal,
    'formal': formal
}
"""
        result, stdout, stderr = executor.exec(code)
        print(f"   Code: {repr(code)}")
        print(f"   Result: {result}")
        print(f"   Stdout: {repr(stdout)}")
        print(f"   Stderr: {repr(stderr)}")
        print()

        print("=== All tests completed successfully! ===")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Always clean up
        print("\nShutting down toolkit...")
        _ = router_actor
        shutdown_toolkit(router_name="code_executor_test", namespace="shed3d_test")
        print("Done!")


if __name__ == "__main__":
    test_code_executor()
