# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from toolshed import get_toolkit


if __name__ == "__main__":
    # Retrieve the toolkit client - a class that provides access to the tools
    toolkit = get_toolkit()

    # Print available tools:
    print(f"Available tools: {toolkit.get_available_tools()}")

    # Call a tool via the pythonic interface
    result = toolkit.calculator.add(4, 5)

    print(f"Tool result value: {result.value}")
    print(f"Tool result text: {result.text}")
    print(f"Tool result images: {result.image}")
    print(f"Tool result variables: {result.variables}")

    # Call a tool via function interface
    result = toolkit.call_tool("calculator", "add", 4, 5)
    result = toolkit.call_tool("calculator", "add", a=4, b=5)