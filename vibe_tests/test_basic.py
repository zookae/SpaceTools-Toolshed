#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Simple test script to verify the toolshed toolkit is working correctly.
Tests both the new pythonic API and backward compatibility.

NOTE: Tools now return ToolResult objects. When accessing tools via get_toolkit(),
the ToolResult is returned directly. Use result.value to get the underlying value.
The CodeExecutor automatically unwraps ToolResult, so code executed via exec() 
works with raw values.
"""

from toolshed import start_toolkit, get_toolkit, shutdown_toolkit
from toolshed.tool_result import ToolResult


def test_pythonic_api():
    """Test the pythonic API with toolkit.tool.method() syntax."""
    print("=== Testing Pythonic API ===")

    # Configuration
    tool_configs = {
        "greeting": {
            "num_actors": 1,
            "conda_env": None,
            "resources": {}
        }
    }

    # Start the distributed toolkit
    print("1. Starting distributed toolkit...")
    router_actor = start_toolkit(tool_configs, router_name="test_router", namespace="shed3d_test")

    try:
        # Get a client
        print("2. Getting toolkit client...")
        toolkit = get_toolkit(router_name="test_router", namespace="shed3d_test")

        # Test pythonic syntax - now returns ToolResult
        print("3. Testing pythonic syntax...")
        result = toolkit.greeting.greet("TestUser")
        assert isinstance(result, ToolResult), f"Expected ToolResult, got {type(result)}"
        print(f"   toolkit.greeting.greet(): {result.value}")
        assert "Hello TestUser!" in result.value

        # Test with parameters
        print("4. Testing with parameters...")
        result = toolkit.greeting.formal_greet("TestUser", title="Dr.")
        assert isinstance(result, ToolResult), f"Expected ToolResult, got {type(result)}"
        print(f"   toolkit.greeting.formal_greet(): {result.value}")
        assert "Dr. TestUser" in result.value

        # Test stats - get_stats returns dict directly, not ToolResult
        print("5. Testing stats...")
        stats = toolkit.greeting.get_stats()
        print(f"   Stats: {stats}")
        assert stats["total_greetings"] >= 2

        # Test status via client
        print("6. Testing status via client...")
        status = toolkit.get_tool_status()
        print(f"   Status: {status}")
        assert "greeting" in status

        # Test alternative usage pattern
        print("7. Testing alternative usage pattern...")
        result = toolkit.greeting.greet("AnotherUser")
        assert isinstance(result, ToolResult), f"Expected ToolResult, got {type(result)}"
        print(f"   Alternative call: {result.value}")
        assert "Hello AnotherUser!" in result.value

        print("   ✅ Pythonic API tests passed!")

    finally:
        print("8. Shutting down distributed toolkit...")
        _ = router_actor  # keep handle alive until shutdown
        shutdown_toolkit(router_name="test_router", namespace="shed3d_test")


def test_multiple_clients():
    """Test that multiple clients can access the same toolkit."""
    print("\n=== Testing Multiple Clients ===")

    # Configuration
    tool_configs = {
        "greeting": {
            "num_actors": 2,  # Multiple actors for load balancing
            "conda_env": None,
            "resources": {}
        }
    }

    # Start the distributed toolkit
    print("1. Starting toolkit with 2 actors...")
    router_actor = start_toolkit(tool_configs, router_name="multi_router", namespace="shed3d_multi")

    try:
        # Create multiple clients
        print("2. Creating multiple clients...")
        client1 = get_toolkit(router_name="multi_router", namespace="shed3d_multi")
        client2 = get_toolkit(router_name="multi_router", namespace="shed3d_multi")

        # Test that both clients work
        print("3. Testing client 1...")
        result1 = client1.greeting.greet("Client1User")
        assert isinstance(result1, ToolResult), f"Expected ToolResult, got {type(result1)}"
        print(f"   Client 1 result: {result1.value}")
        assert "Hello Client1User!" in result1.value

        print("4. Testing client 2...")
        result2 = client2.greeting.greet("Client2User")
        assert isinstance(result2, ToolResult), f"Expected ToolResult, got {type(result2)}"
        print(f"   Client 2 result: {result2.value}")
        assert "Hello Client2User!" in result2.value

        # Test load balancing by making multiple calls
        print("5. Testing load balancing with multiple calls...")
        for i in range(4):
            result = client1.greeting.greet(f"User{i}")
            assert isinstance(result, ToolResult), f"Expected ToolResult, got {type(result)}"
            print(f"   Call {i+1}: {result.value}")

        # Check final stats
        print("6. Checking final stats...")
        stats = client2.greeting.get_stats()  # Use different client
        print(f"   Final stats: {stats}")
        assert stats["total_greetings"] >= 6  # At least 6 calls made

        print("   ✅ Multiple clients test passed!")

    finally:
        print("7. Shutting down toolkit...")
        _ = router_actor
        shutdown_toolkit(router_name="multi_router", namespace="shed3d_multi")


def test_error_handling():
    """Test error handling."""
    print("\n=== Testing Error Handling ===")

    tool_configs = {
        "greeting": {
            "num_actors": 1,
            "conda_env": None,
            "resources": {}
        }
    }

    # Start toolkit
    router_actor = start_toolkit(tool_configs, router_name="error_router", namespace="shed3d_error")

    try:
        toolkit = get_toolkit(router_name="error_router", namespace="shed3d_error")

        # Test accessing non-existent tool
        print("1. Testing non-existent tool...")
        try:
            toolkit.nonexistent.some_method("test")
            print("   ❌ Should have failed!")
            assert False, "Expected error for non-existent tool"
        except Exception as e:
            print(f"   ✅ Correctly caught error: {e}")

        # Test accessing non-existent method
        print("2. Testing non-existent method...")
        try:
            toolkit.greeting.nonexistent_method("test")
            print("   ❌ Should have failed!")
            assert False, "Expected error for non-existent method"
        except Exception as e:
            print(f"   ✅ Correctly caught error: {e}")

        print("   ✅ Error handling tests passed!")

    finally:
        _ = router_actor
        shutdown_toolkit(router_name="error_router", namespace="shed3d_error")


def test_default_namespace():
    """Test that the default namespace works correctly."""
    print("\n=== Testing Default Namespace ===")

    tool_configs = {
        "greeting": {
            "num_actors": 1,
            "conda_env": None,
            "resources": {}
        }
    }

    # Start toolkit with default namespace
    print("1. Starting toolkit with default namespace...")
    router_actor = start_toolkit(tool_configs)  # Uses default "toolshed" namespace

    try:
        # Get client with default namespace
        print("2. Getting client with default namespace...")
        toolkit = get_toolkit()  # Uses default "toolshed" namespace

        # Test basic functionality
        print("3. Testing basic functionality...")
        result = toolkit.greeting.greet("DefaultNamespaceUser")
        assert isinstance(result, ToolResult), f"Expected ToolResult, got {type(result)}"
        print(f"   Result: {result.value}")
        assert "Hello DefaultNamespaceUser!" in result.value

        print("   ✅ Default namespace test passed!")

    finally:
        print("4. Shutting down default namespace toolkit...")
        _ = router_actor
        shutdown_toolkit()  # Uses default "toolshed" namespace


def main():
    """Run all tests."""
    print("Testing toolshed toolkit - New Module Structure with Namespaces...")

    try:
        # Test main pythonic API
        test_pythonic_api()

        # Test multiple clients
        test_multiple_clients()

        # Test error handling
        test_error_handling()

        # Test default namespace
        test_default_namespace()

        print("\n🎉 All tests passed! New module structure with namespaces works correctly.")

    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        raise


if __name__ == "__main__":
    main()
