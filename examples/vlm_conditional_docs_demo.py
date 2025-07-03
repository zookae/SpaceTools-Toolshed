#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Demonstration of conditional documentation for the VLM tool.

This script shows how the documentation for VLM.detect_all changes
based on the no_output_image and no_output_vars configuration flags.
"""

from toolshed.tools.base import BaseTool, tool_method


# Mock VLM class for demonstration (doesn't require GPU)
class MockVLMTool(BaseTool):
    """Mock VLM tool that demonstrates conditional documentation."""
    
    def get_name(self) -> str:
        return "vlm"
    
    def get_description(self) -> str:
        return "Vision-Language model for object detection and image understanding"
    
    def get_stats(self) -> dict:
        return {}
    
    @tool_method
    def detect_all(self, image, obj_name: str):
        """Detect *all* instances of *obj_name* in *image*.

        Text output: coordinates of a single point for the first instance of the object, in normalized pixel space in range [0, 1].
        [[if:image]]Visual output: an overlay image with detected objects marked as red points.[[/if:image]]
        [[if:vars]]Stored variables: Coordinates in ${obj_name}_detections variable of list of (x, y) point coordinates in normalized pixel space [0, 1] for use in subsequent operations.[[/if:vars]]

        Args:
            image (Image): image in which to detect objects.
            obj_name (str): Name or description of the object to detect.
        """
        pass


def demonstrate_conditional_docs():
    """Show how documentation changes with different configurations."""
    
    print("="*80)
    print("VLM Tool Conditional Documentation Demonstration")
    print("="*80)
    print("\nThis shows how the detect_all method documentation adapts based on configuration.\n")
    
    configurations = [
        {
            "name": "Full Features (default)",
            "no_output_image": False,
            "no_output_vars": False,
            "expected": "Should include both visual output and stored variables sections"
        },
        {
            "name": "Text Only Mode",
            "no_output_image": True,
            "no_output_vars": True,
            "expected": "Should only show text output, no visual or variables sections"
        },
        {
            "name": "With Images, No Variables",
            "no_output_image": False,
            "no_output_vars": True,
            "expected": "Should show visual output but no stored variables"
        },
        {
            "name": "With Variables, No Images",
            "no_output_image": True,
            "no_output_vars": False,
            "expected": "Should show stored variables but no visual output"
        }
    ]
    
    for config in configurations:
        print(f"\n{'='*60}")
        print(f"Configuration: {config['name']}")
        print(f"  no_output_image: {config['no_output_image']}")
        print(f"  no_output_vars: {config['no_output_vars']}")
        print(f"Expected: {config['expected']}")
        print("-"*60)
        
        # Create tool with specific configuration
        tool = MockVLMTool(
            no_output_image=config['no_output_image'],
            no_output_vars=config['no_output_vars']
        )
        
        # Get schemas
        schemas = tool.get_openai_schemas()
        
        # Find and display detect_all documentation
        for schema in schemas:
            if "detect_all" in schema["function"]["name"]:
                description = schema["function"]["description"]
                print(f"\nGenerated description:")
                print(f'"{description}"')
                
                # Analyze what's included
                print(f"\nAnalysis:")
                has_text = "coordinates of a single point" in description
                has_visual = "overlay image" in description
                has_vars = "${obj_name}_detections" in description or "Stored variables:" in description
                
                print(f"  ✓ Has text output description: {has_text}")
                print(f"  {'✓' if has_visual == (not config['no_output_image']) else '✗'} "
                      f"Has visual output: {has_visual} "
                      f"(expected: {not config['no_output_image']})")
                print(f"  {'✓' if has_vars == (not config['no_output_vars']) else '✗'} "
                      f"Has variables output: {has_vars} "
                      f"(expected: {not config['no_output_vars']})")
                break
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("\nThe conditional documentation system allows tools to:")
    print("1. Always show what text/data output is returned")
    print("2. Conditionally show visual output documentation based on no_output_image flag")
    print("3. Conditionally show variable storage documentation based on no_output_vars flag")
    print("\nThis helps LLMs understand what outputs are available in the current configuration.")


if __name__ == "__main__":
    demonstrate_conditional_docs()
