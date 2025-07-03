#!/usr/bin/env python3

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Demo script to test the toolshed dashboard.

This script starts the toolkit with multiple tools and enables the dashboard
to visualize actor states in real-time.
"""

import time
import logging
import ray
from toolshed import start_toolkit, get_toolkit, shutdown_toolkit

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
    force=True  # Force reconfiguration even if logging was already configured
)
logger = logging.getLogger(__name__)

def main():
    # Configure multiple tools with different numbers of actors
    tool_configs = {
        "greeting": {
            "num_actors": 3,
            "conda_env": None,
            "resources": {}
        },
        "calculator": {
            "num_actors": 2,
            "conda_env": None,
            "resources": {}
        }
    }
    
    # Initialize Ray with log_to_driver if not already initialized
    if not ray.is_initialized():
        ray.init(log_to_driver=True)
    
    # Start toolkit with dashboard enabled
    logger.info("Starting toolkit with dashboard...")
    router_handle = start_toolkit(
        tool_configs, 
        dashboard=True, 
        dashboard_port=7001,
        detached=False  # Clean up on exit
    )
    
    logger.info("Dashboard is running at http://localhost:7001")
    logger.info("Open this URL in your browser to see the actor states")
    
    # Get toolkit client
    toolkit = get_toolkit()
    
    # Simulate some work to see actors in action
    logger.info("Starting simulated workload...")
    
    try:
        while True:
            # Submit multiple concurrent requests to see actors working
            logger.info("Submitting batch of concurrent requests...")
            
            # Submit greeting requests without waiting (get ray ObjectRefs)
            greeting_refs = []
            for i in range(10):  # More requests than actors to see queuing
                # Get the ToolProxy and router actor directly
                greeting_proxy = toolkit.greeting
                ref = greeting_proxy._router_actor.call_tool.remote(
                    greeting_proxy._tool_name, "slow_greet", 
                    name=f"User{i}", delay=3  # Longer delay to see working state
                )
                greeting_refs.append(ref)
            
            # Submit calculator requests too
            calc_refs = []
            for i in range(5):
                calc_proxy = toolkit.calculator
                ref = calc_proxy._router_actor.call_tool.remote(
                    calc_proxy._tool_name, "slow_add",  # Let's add a slow_add method
                    a=i, b=i+1, delay=2
                )
                calc_refs.append(ref)
            
            logger.info("Submitted all requests, now actors should be working...")
            
            # Wait a bit to observe working state
            time.sleep(1)
            logger.info("Check the dashboard now - actors should show as 'working'!")
            
            # Now collect all results
            logger.info("Collecting results...")
            greeting_results = ray.get(greeting_refs)
            for i, result in enumerate(greeting_results):
                logger.info(f"Greeting result {i}: {result}")
                
            calc_results = ray.get(calc_refs)
            for i, result in enumerate(calc_results):
                logger.info(f"Calculator result {i}: {result}")
            
            # Brief pause before next batch
            logger.info("Batch complete, pausing before next batch...")
            time.sleep(3)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        shutdown_toolkit()

if __name__ == "__main__":
    main() 
