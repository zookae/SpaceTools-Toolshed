# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Static manifest mapping of tool names to their import paths.

End-users may extend this by specifying ``import_path`` in the tool config
for external tools.
"""

TOOL_MANIFEST = {
    # Example tools
    "calculator": "toolshed.tools.calculator:CalculatorTool",
    "greeting": "toolshed.tools.greeting:GreetingTool",

    # Code execution
    "code_executor": "toolshed.tools.code_executor_tool:CodeExecutorTool",

    # Vision tools
    "vlm": "toolshed.tools.vlm:VLMTool",
    "roborefer": "toolshed.tools.roborefer:RoboreferTool",
    "sam2": "toolshed.tools.sam2:Sam2SegmentationTool",
    "depth_estimator": "toolshed.tools.depth_estimator:DepthEstimatorTool",
    "real_depth_estimator": "toolshed.tools.real_depth:DepthEstimatorTool",  # HTTP-based depth from robot
    "vision_ops": "toolshed.tools.vision_ops:VisionOpsTool",
    "visual_io": "toolshed.tools.visual_io:VisualIO",
    "bounding_box": "toolshed.tools.bounding_box:BoundingBoxTool",
    "grasp_generator": "toolshed.tools.grasp_generator:GraspGeneratorTool",

    # Robot control
    "robot": "toolshed.tools.robot:RobotTool",
    "mock_robot": "toolshed.tools.mock_robot:MockRobotTool",
}