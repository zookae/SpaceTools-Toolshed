# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Test the documentation generation functionality.
"""

from toolshed import start_toolkit, get_toolkit, shutdown_toolkit


def test_documentation():
    """Test the documentation generation."""
    print("=== Testing Documentation Generation ===\n")

    # Configuration for tools
    tool_configs = {
        "greeting": {
            "num_actors": 1,
            "conda_env": None,
            "resources": {}
        }
    }

    # Initialize the toolkit
    print("Initializing toolkit...")
    router_actor = start_toolkit(tool_configs, router_name="doc_test", namespace="shed3d_doc_test")

    try:
        # Get toolkit client
        toolkit = get_toolkit(router_name="doc_test", namespace="shed3d_doc_test")

        # Generate documentation
        print("Generating documentation...\n")
        documentation = toolkit.get_documentation()

        print("=" * 80)
        print("GENERATED DOCUMENTATION:")
        print("=" * 80)
        print(documentation)
        print("=" * 80)

        print("\n✅ Documentation generation test completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up
        print("\nShutting down toolkit...")
        _ = router_actor
        shutdown_toolkit(router_name="doc_test", namespace="shed3d_doc_test")
        print("Done!")


if __name__ == "__main__":
    test_documentation()
