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

    # Isaac ROS manipulation primitives
    "isaac_ros_graph": "toolshed.tools.isaac_ros:IsaacRosGraphTool",
    "isaac_ros_launch": "toolshed.tools.isaac_ros:IsaacRosLaunchTool",
    "isaac_ros_topic": "toolshed.tools.isaac_ros:IsaacRosTopicTool",
    "isaac_ros_action": "toolshed.tools.isaac_ros:IsaacRosActionTool",
    "isaac_ros_service": "toolshed.tools.isaac_ros:IsaacRosServiceTool",
    "isaac_ros_param": "toolshed.tools.isaac_ros:IsaacRosParamTool",
    "isaac_ros_tf": "toolshed.tools.isaac_ros:IsaacRosTfTool",
    "isaac_ros_config": "toolshed.tools.isaac_ros:IsaacRosConfigTool",
    "isaac_ros_scene": "toolshed.tools.isaac_ros:IsaacRosSceneTool",
    "isaac_perception": "toolshed.tools.isaac_ros:IsaacPerceptionTool",
    "isaac_object_info": "toolshed.tools.isaac_ros:IsaacObjectInfoTool",
    "isaac_segmentation": "toolshed.tools.isaac_ros:IsaacSegmentationTool",
    "isaac_pick_place": "toolshed.tools.isaac_ros:IsaacPickPlaceTool",
    "isaac_gripper": "toolshed.tools.isaac_ros:IsaacGripperTool",

    # Isaac ROS manipulation launch include facades
    "isaac_cumotion": "toolshed.tools.isaac_ros:IsaacCumotionTool",
    "isaac_dope": "toolshed.tools.isaac_ros:IsaacDopeTool",
    "isaac_ess": "toolshed.tools.isaac_ros:IsaacEssTool",
    "isaac_foundationpose": "toolshed.tools.isaac_ros:IsaacFoundationposeTool",
    "isaac_foundationstereo": "toolshed.tools.isaac_ros:IsaacFoundationstereoTool",
    "isaac_grounding_dino": "toolshed.tools.isaac_ros:IsaacGroundingDinoTool",
    "isaac_nvblox": "toolshed.tools.isaac_ros:IsaacNvbloxTool",
    "isaac_object_following": "toolshed.tools.isaac_ros:IsaacObjectFollowingTool",
    "isaac_pose_to_pose": "toolshed.tools.isaac_ros:IsaacPoseToPoseTool",
    "isaac_realsense": "toolshed.tools.isaac_ros:IsaacRealsenseTool",
    "isaac_rtdetr": "toolshed.tools.isaac_ros:IsaacRtdetrTool",
    "isaac_segment_anything": "toolshed.tools.isaac_ros:IsaacSegmentAnythingTool",
    "isaac_segment_anything2": "toolshed.tools.isaac_ros:IsaacSegmentAnything2Tool",
    "isaac_static_transforms": "toolshed.tools.isaac_ros:IsaacStaticTransformsTool",
}
