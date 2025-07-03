# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Distributed usage example for the toolshed toolkit.

This example demonstrates the new pythonic API that works across Ray workers:

1. Start the toolkit actor (done once per cluster)
2. Multiple Ray workers can access it with pythonic syntax
3. Automatic queuing when more workers than tool actors
4. Clean separation between toolkit setup and usage
"""

import time
import ray
from toolshed import start_toolkit, get_toolkit, shutdown_toolkit


@ray.remote
def worker_task(worker_id: int, num_calls: int = 3):
    """
    Simulate a Ray worker that needs to use tools.
    This could be running on any machine in the cluster.
    """
    # Get a toolkit client (connects to the centralized ToolRouterActor)
    toolkit = get_toolkit()
    
    results = []
    print(f"Worker {worker_id} starting {num_calls} calls...")
    
    for i in range(num_calls):
        # Use pythonic syntax - this goes through the distributed actor!
        result = toolkit.greeting.slow_greet(f"Worker{worker_id}_Call{i+1}", 1.0)
        results.append(result)
        print(f"  Worker {worker_id} Call {i+1}: {result}")
        
        # Small delay to show interleaving
        time.sleep(0.1)
    
    # Try different methods
    formal_result = toolkit.greeting.formal_greet(f"Worker{worker_id}", title="Dr.")
    results.append(formal_result)
    print(f"  Worker {worker_id} Formal: {formal_result}")
    
    return results


@ray.remote  
def analytics_worker():
    """A worker that periodically checks tool status."""
    toolkit = get_toolkit()
    
    for i in range(3):
        time.sleep(1.0)
        
        # Check available tools
        tools = toolkit.get_available_tools()
        print(f"Analytics: Available tools: {tools}")
        
        # Get tool statistics
        stats = toolkit.greeting.get_stats()
        print(f"Analytics: Tool stats: {stats}")
        
        # Get tool status
        status = toolkit.get_tool_status()
        print(f"Analytics: Tool status: {status}")


def main():
    """Main demonstration function."""
    print("=== toolshed Distributed Toolkit Example ===\n")
    
    # Configuration - only 2 actors for the greeting tool
    tool_configs = {
        "greeting": {
            "num_actors": 2,
            "conda_env": None,
            "resources": {"num_cpus": 1, "num_gpus": 0}
        }
    }
    
    # Step 1: Start the toolkit actor (done once per cluster)
    print("1. Starting distributed toolkit...")
    router_actor = start_toolkit(tool_configs)
    print("   Toolkit started and ready for workers!")
    
    try:
        # Step 2: Test basic pythonic access
        print("\n2. Testing basic pythonic access:")
        toolkit = get_toolkit()
        
        result = toolkit.greeting.greet("DirectUser")
        print(f"   Direct call result: {result}")

        # Step 3: Launch multiple workers (more workers than tool actors!)
        print("\n3. Launching 4 workers with only 2 tool actors (demonstrates queuing):")
        
        # Start workers - these will compete for the 2 greeting tool actors
        worker_futures = []
        for worker_id in range(8):
            future = worker_task.remote(worker_id, num_calls=2)
            worker_futures.append(future)
        
        # Also start an analytics worker
        analytics_future = analytics_worker.remote()
        
        print("   Workers launched, watch the interleaved output...")
        print("   (4 workers sharing 2 tool actors)")
        
        # Wait for all workers to complete
        start_time = time.time()
        worker_results = ray.get(worker_futures)
        ray.get(analytics_future)  # Wait for analytics too
        end_time = time.time()
        
        time.sleep(3)
        print(f"\n   All workers completed in {end_time - start_time:.2f}s")
        
        # Step 4: Show final statistics
        print("\n4. Final statistics:")
        
        # Get per-actor stats to see how calls were distributed
        per_actor_stats = toolkit.get_per_actor_stats("greeting")
        print(f"   Per-actor statistics ({len(per_actor_stats)} actors):")
        total_calls_across_actors = 0
        for actor_stats in per_actor_stats:
            actor_idx = actor_stats.get('actor_index', '?')
            actor_total = actor_stats.get('total_calls', 0)
            total_calls_across_actors += actor_total
            print(f"     Actor {actor_idx}: {actor_total} total calls - {actor_stats}")
        
        print(f"\n   Total calls across all actors: {total_calls_across_actors}")
        print(f"   Total calls made by workers: {sum(len(results) for results in worker_results) + 1}")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        raise

    finally:
        # Clean up
        print("\n5. Cleaning up...")
        shutdown_toolkit()
        print("   Toolkit shut down")


if __name__ == "__main__":
    main()
    
    # print("\n" + "="*60)
    # demonstrate_distributed_workers() 