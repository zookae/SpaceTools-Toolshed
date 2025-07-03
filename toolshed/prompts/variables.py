# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Shared variable handling instructions for LLM integration.

This module defines how LLMs should interact with the variable system
that tracks outputs between tool calls.
"""

VARIABLE_HANDLING_BASE_PROMPT = """
## Variable Handling

**IMPORTANT - How to Use Stored Variables:**

Tools can store outputs as variables (like $depth_map, $focal_length_px, etc.) that persist across tool calls.

**The $ Syntax Rule:**
- Use $ prefix when passing variables between tools (in tool call arguments)
- Example: If a tool stored $depth_map, pass it to another tool as: {"input": "$depth_map"}

**Example Workflow:**

Step 1 - depth_estimator returns:
  - $depth_map (numpy array)
  - $focal_length_px (float)

Step 2 - Pass to another tool:
  - Use $depth_map or $focal_length_px in tool arguments to reference the stored values

**Remember:** The $ syntax is the bridge between tools. Use it in tool arguments to reference stored variables.
"""

VARIABLE_HANDLING_CODE_EXECUTOR_PROMPT = """
## Variable Handling

**IMPORTANT - How to Use Stored Variables:**

Tools can store outputs as variables (like $depth_map, $focal_length_px, etc.) that persist across tool calls.
When using the code_executor tool, you MUST understand how to properly pass and use these variables:

**The $ Syntax Rule:**
- Use $ prefix ONLY when passing variables between tools (in tool call arguments)
- NEVER use $ prefix inside Python code itself
- Inside code, use plain variable names

**Correct Usage Pattern:**

1. A tool returns stored variables (e.g., depth_estimator stores $depth_map, $focal_length_px)
2. To use these in code_executor:
   - Pass them via the 'variables' parameter WITH $ prefix:
     ```
     variables: {"depth": "$depth_map", "focal": "$focal_length_px"}
     ```
   - Then use them in code WITHOUT $ prefix:
     ```
     code: "result = depth[100, 200] * focal"
     ```

**Example Workflow:**

Step 1 - depth_estimator returns:
  - $depth_map (numpy array)
  - $focal_length_px (float)

Step 2 - code_executor__exec call:
  ```
  {
    "variables": {
      "depth": "$depth_map",
      "focal": "$focal_length_px"
    },
    "code": "import numpy as np\\nresult = depth.mean() * focal"
  }
  ```
  ✓ Correct: $ in variables parameter, plain names in code

**Common Mistakes to Avoid:**

❌ Using $ inside Python code:
  ```
  code: "result = $depth_map.mean()"  # WRONG!
  ```

❌ Forgetting $ in variables parameter:
  ```
  variables: {"depth": "depth_map"}  # WRONG! Missing $
  ```

❌ Not passing variables at all:
  ```
  code: "result = depth_map.mean()"  # WRONG! Variable not defined
  variables: {}
  ```

**Remember:** The $ syntax is the bridge between tools. Use it in tool arguments to reference stored variables, but omit it inside Python code where you're working with the actual values.
"""


def get_variable_prompt(has_code_executor: bool) -> str:
    """
    Get the appropriate variable handling prompt based on available tools.
    
    Args:
        has_code_executor: Whether the code_executor tool is available
        
    Returns:
        The appropriate variable handling prompt string
    """
    if has_code_executor:
        return VARIABLE_HANDLING_CODE_EXECUTOR_PROMPT
    return VARIABLE_HANDLING_BASE_PROMPT


# Legacy alias for backward compatibility
VARIABLE_HANDLING_PROMPT = VARIABLE_HANDLING_CODE_EXECUTOR_PROMPT
