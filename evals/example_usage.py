#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Example usage of the Toolshed evaluation framework.

This script demonstrates how to:
1. Evaluate models programmatically (without CLI)
2. Create custom model wrappers
3. Process and analyze results
"""

import os
import sys
from pathlib import Path

# Add toolshed to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from toolshed import start_toolkit
from evals import UnifiedModel, RoboSpatialEvaluator


def evaluate_with_tools():
    """Example: Evaluate a model with Toolshed tools."""
    print("=== Evaluating GPT-4o with Toolshed Tools ===\n")
    
    # Start toolkit with vision tools
    print("Starting Toolshed...")
    toolkit = start_toolkit({
        'vlm': {
            'num_actors': 1,
            'resources': {'num_gpus': 1},
            'conda_env': 'tool_vlm',
            'timeout': 600,
        }
    })
    
    # Initialize unified model
    model = UnifiedModel(
        toolkit=toolkit,
        provider="openai",
        model="gpt-4o",
        enable_variables=True,
        max_iterations=5
    )
    
    # Create evaluator
    evaluator = RoboSpatialEvaluator(model)
    
    # Run evaluation on a small sample
    dataset_path = "data/robospatial_home/test.parquet"
    if not Path(dataset_path).exists():
        print(f"Dataset not found at {dataset_path}")
        print("Please download the RoboSpatial-Home dataset first.")
        return
    
    results = evaluator.evaluate(dataset_path, limit=5)
    
    # Print results
    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Average Score: {results['average_score']:.3f}")
    
    # Save results
    evaluator.save_results(results, "example_results_with_tools.json")
    print("\nResults saved to example_results_with_tools.json")


def evaluate_without_tools():
    """Example: Evaluate a model without tools (baseline)."""
    print("\n=== Evaluating GPT-4o without Tools (Baseline) ===\n")
    
    # Initialize unified model without toolkit
    model = UnifiedModel(
        toolkit=None,  # No toolkit = no tools
        provider="openai",
        model="gpt-4o",
        enable_variables=False,
        max_iterations=1  # No tool calls needed
    )
    
    # Create evaluator
    evaluator = RoboSpatialEvaluator(model)
    
    # Run evaluation
    dataset_path = "data/robospatial_home/test.parquet"
    results = evaluator.evaluate(dataset_path, limit=5)
    
    # Print results
    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Average Score: {results['average_score']:.3f}")
    
    # Save results
    evaluator.save_results(results, "example_results_no_tools.json")
    print("\nResults saved to example_results_no_tools.json")


def evaluate_different_providers():
    """Example: Evaluate models from different providers."""
    print("\n=== Evaluating Different Providers ===\n")
    
    # Example with Anthropic Claude (if available)
    print("Anthropic Claude example:")
    try:
        model = UnifiedModel(
            toolkit=None,  # No tools for this example
            provider="anthropic",
            model="claude-opus-4-1-20250805",
            enable_variables=False,
            max_iterations=1
        )
        print(f"  Initialized: {model.provider}/{model.model_name}")
    except Exception as e:
        print(f"  Failed to initialize Anthropic: {e}")
    
    # Example with AWS Bedrock (if available)
    print("\nAWS Bedrock example:")
    try:
        model = UnifiedModel(
            toolkit=None,  # No tools for this example
            provider="bedrock",
            model="us.anthropic.claude-sonnet-4-20250514-v1:0",
            enable_variables=False,
            max_iterations=1
        )
        print(f"  Initialized: {model.provider}/{model.model_name}")
    except Exception as e:
        print(f"  Failed to initialize Bedrock: {e}")


def custom_model_example():
    """Example: Create a custom model wrapper."""
    from evals.models import BaseModel
    
    class DummyModel(BaseModel):
        """A dummy model that always returns 'yes' for testing."""
        
        def predict(self, text: str, images=None, system_prompt=None) -> str:
            # Extract question type from text
            if "yes or no" in text.lower():
                return "#### yes"
            elif "pinpoint" in text.lower():
                return "#### [(0.5, 0.5)]"
            else:
                return "#### [[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]]"
    
    print("\n=== Evaluating Dummy Model ===\n")
    
    model = DummyModel()
    evaluator = RoboSpatialEvaluator(model)
    
    # This would evaluate the dummy model
    # results = evaluator.evaluate("data/robospatial_home/test.parquet", limit=5)


def analyze_results():
    """Example: Analyze saved results programmatically."""
    import json
    
    print("\n=== Analyzing Results ===\n")
    
    # Load results
    try:
        with open("example_results_with_tools.json", 'r') as f:
            results_with_tools = json.load(f)
        
        with open("example_results_no_tools.json", 'r') as f:
            results_no_tools = json.load(f)
    except FileNotFoundError:
        print("Run the evaluation examples first to generate result files.")
        return
    
    # Compare accuracies
    acc_with = results_with_tools['summary']['accuracy']
    acc_without = results_no_tools['summary']['accuracy']
    
    print(f"Accuracy with tools: {acc_with:.2%}")
    print(f"Accuracy without tools: {acc_without:.2%}")
    print(f"Improvement: {(acc_with - acc_without):.2%}")
    
    # Analyze by question type
    if 'metrics_by_type' in results_with_tools['summary']:
        print("\nPerformance by question type (with tools):")
        for qtype, metrics in results_with_tools['summary']['metrics_by_type'].items():
            print(f"  {qtype}: {metrics['accuracy']:.2%} ({metrics['count']} examples)")


def main():
    """Run all examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Run examples
    try:
        # Note: Comment out evaluate_with_tools() if you don't have GPU
        # evaluate_with_tools()
        evaluate_without_tools()
        evaluate_different_providers()
        custom_model_example()
        analyze_results()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
