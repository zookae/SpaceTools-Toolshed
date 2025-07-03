# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Ray actor for the Toolshed dashboard.

This actor runs the dashboard web server independently, communicating with
the ToolRouterActor via Ray RPC calls.
"""

import ray
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0)
class DashboardActor:
    """
    Ray actor that runs the dashboard web server.
    
    This actor is independent and communicates with the ToolRouterActor
    via Ray RPC calls, following Ray's distributed architecture patterns.
    """
    
    def __init__(self, 
                 router_name: str,
                 namespace: str = "toolshed",
                 host: str = '0.0.0.0',
                 port: int = 7001):
        """
        Initialize the dashboard actor.
        
        Args:
            router_name: Name of the router actor to monitor
            namespace: Ray namespace
            host: Host to bind the web server to
            port: Port for the web server
        """
        self.router_name = router_name
        self.namespace = namespace
        self.host = host
        self.port = port
        self._running = False
        self._server_task = None
        self._server = None  # Store the server instance
        
    async def start(self):
        """Start the dashboard server asynchronously."""
        if self._running:
            logger.warning("Dashboard already running")
            return
            
        try:
            # Import here to avoid circular imports
            from .dashboard import ActorStateMonitor, DashboardServer
            
            # Create monitor with router name for auto-reconnection
            monitor = ActorStateMonitor(self.router_name, self.namespace)
            monitor.connect()
            
            # Create server
            self._server = DashboardServer(monitor, self.host, self.port)
            
            # Start server in background task
            self._server_task = asyncio.create_task(self._server.start())
            self._running = True
            
            logger.info(f"Dashboard actor started, serving at http://{self.host}:{self.port}")
            
            # Print prominent connection message
            from .dashboard import _get_local_ip
            local_ip = _get_local_ip()
            print("-" * 80)
            print(f"Dashboard running at http://{local_ip}:{self.port}")
            print("-" * 80)
            
            # Don't await the server task - let it run in background
            await asyncio.sleep(0.1)  # Small delay to ensure startup
            
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            self._running = False
            raise
            
    def is_running(self) -> bool:
        """Check if the dashboard is running."""
        return self._running
        
    async def stop(self):
        """Stop the dashboard server."""
        if not self._running:
            return
            
        self._running = False
        
        # Shutdown the server properly
        if self._server:
            await self._server.shutdown()
        
        # Cancel the server task
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Dashboard actor stopped")
        
    def get_status(self) -> dict:
        """Get dashboard status."""
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "url": f"http://{self.host}:{self.port}" if self._running else None
        } 