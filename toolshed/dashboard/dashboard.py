# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Real-time web dashboard for monitoring toolshed actor states.

This module provides:
1. Actor state discovery and monitoring
2. Web server to host the dashboard
3. WebSocket support for real-time updates
"""

import asyncio
import json
import logging
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import sys

import ray
from aiohttp import web
import aiohttp_cors

# Handle different Python versions for resource access
if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources

logger = logging.getLogger(__name__)


def _get_local_ip() -> str:
    """Get the local IP address accessible from other machines on the network."""
    try:
        result = subprocess.run(
            ["hostname", "-I"],
            check=True,
            capture_output=True,
            text=True
        )
        ips = result.stdout.strip().split()
        if ips:
            return ips[0]
    except Exception:
        pass
    return "localhost"


class ActorStateMonitor:
    """Monitor and track the state of toolshed actors."""
    
    def __init__(self, router_name: str = "toolshed_router", namespace: str = "toolshed"):
        self.router_name = router_name
        self.namespace = namespace
        self.router_actor = None
        
        # State tracking
        self.actor_states: Dict[str, Dict[str, any]] = {}
        self.state_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.last_update = time.time()
        
    def connect(self):
        """Connect to the router actor."""
        try:
            self.router_actor = ray.get_actor(self.router_name, namespace=self.namespace)
            logger.info(f"Connected to router actor: {self.router_name}")
        except Exception as e:
            logger.error(f"Failed to connect to router actor: {e}")
            raise
            
    def ensure_connected_sync(self):
        """Sync version of ensure_connected for use in synchronous methods."""
        if self.router_actor is None:
            self.connect()
            return
            
        # Check if the router is still alive
        try:
            # Try a quick operation to test if router is alive
            ready, _ = ray.wait([self.router_actor.get_available_tools.remote()], timeout=0.5)
            if not ready:
                raise Exception("Router not responding")
        except Exception as e:
            logger.info(f"Router appears to be dead ({e}), attempting to reconnect...")
            try:
                self.connect()
                logger.info("Successfully reconnected to router")
            except Exception as e:
                logger.error(f"Failed to reconnect to router: {e}")
                self.router_actor = None
                raise
                
    async def ensure_connected(self):
        """Ensure we have a valid connection to the router, reconnecting if needed."""
        if self.router_actor is None:
            self.connect()
            return
            
        # Check if the router is still alive
        try:
            # Try a quick operation to test if router is alive
            await asyncio.wait_for(
                self.router_actor.get_available_tools.remote(), 
                timeout=0.5
            )
        except Exception as e:
            logger.info(f"Router appears to be dead ({e}), attempting to reconnect...")
            try:
                self.connect()
                logger.info("Successfully reconnected to router")
            except Exception as e:
                logger.error(f"Failed to reconnect to router: {e}")
                self.router_actor = None
                raise
            
    async def get_actor_states_async(self) -> Dict[str, List[Dict]]:
        """
        Get current state of all actors (async version).
        
        Returns:
            Dictionary mapping tool names to list of actor states
        """
        # Ensure we have a valid router connection
        await self.ensure_connected()
        
        if not self.router_actor:
            return {}
            
        try:
            # Get actor states directly from the router
            states = await self.router_actor.get_actor_states.remote()
            
            # Get tool status once for conda_env info
            tool_status = await self.router_actor.get_tool_status.remote()
            
            # Add any additional info we need for the dashboard
            for tool_name, tool_states in states.items():
                conda_env = tool_status.get(tool_name, {}).get('tool_config', {}).get('conda_env', 'base')
                for actor_state in tool_states:
                    # Remove actor_handle as it's not needed for display
                    actor_state.pop('actor_handle', None)
                    # Add conda_env
                    actor_state['conda_env'] = conda_env
                    
            return states
        except Exception as e:
            logger.error(f"Error getting actor states: {e}")
            return {}
            
    def get_actor_states(self) -> Dict[str, List[Dict]]:
        """
        Get current state of all actors (sync version for backwards compatibility).
        
        Returns:
            Dictionary mapping tool names to list of actor states
        """
        # Ensure we have a valid router connection
        self.ensure_connected_sync()
        
        if not self.router_actor:
            return {}
            
        try:
            # Get actor states directly from the router
            states = ray.get(self.router_actor.get_actor_states.remote())
            
            # Get tool status once for conda_env info
            tool_status = ray.get(self.router_actor.get_tool_status.remote())
            
            # Add any additional info we need for the dashboard
            for tool_name, tool_states in states.items():
                conda_env = tool_status.get(tool_name, {}).get('tool_config', {}).get('conda_env', 'base')
                for actor_state in tool_states:
                    # Remove actor_handle as it's not needed for display
                    actor_state.pop('actor_handle', None)
                    # Add conda_env
                    actor_state['conda_env'] = conda_env
                    
            return states
        except Exception as e:
            logger.error(f"Error getting actor states: {e}")
            return {}
    
    async def update_history_async(self):
        """Update state history for all actors (async version)."""
        current_states = await self.get_actor_states_async()
        timestamp = time.time()
        
        for tool_name, actors in current_states.items():
            for actor in actors:
                actor_id = actor['id']
                self.state_history[actor_id].append({
                    'timestamp': timestamp,
                    'state': actor['state']
                })
                
        self.actor_states = current_states
        self.last_update = timestamp
        



class DashboardServer:
    """Web server for the toolshed dashboard."""
    
    def __init__(self, monitor: ActorStateMonitor, host: str = '0.0.0.0', port: int = 8080):
        self.monitor = monitor
        self.host = host
        self.port = port
        self.app = web.Application()
        self.websockets = set()
        self.runner = None  # Store runner for proper shutdown
        self.site = None    # Store site for proper shutdown
        
        # Setup routes
        self._setup_routes()
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Configure CORS on all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
        
    def _setup_routes(self):
        """Setup web routes."""
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/api/states', self.get_states)
        self.app.router.add_get('/api/history/{actor_id}', self.get_history)
        self.app.router.add_get('/ws', self.websocket_handler)
        
        # Add static file serving
        import pathlib
        import toolshed
        static_dir = str(pathlib.Path(toolshed.__file__).parent / 'dashboard' / 'static')
        self.app.router.add_static('/static', static_dir)
        
    async def index(self, request):
        """Serve the main dashboard page."""
        import pathlib
        import toolshed
        
        # Get the path to the static file
        static_path = str(pathlib.Path(toolshed.__file__).parent / 'dashboard' / 'static' / 'index.html')
        return web.FileResponse(static_path)
        
    async def get_states(self, request):
        """Get current actor states."""
        states = await self.monitor.get_actor_states_async()
        return web.json_response(states)
        
    async def get_history(self, request):
        """Get state history for a specific actor."""
        actor_id = request.match_info['actor_id']
        history = list(self.monitor.state_history.get(actor_id, []))
        return web.json_response(history)
        
    async def websocket_handler(self, request):
        """Handle WebSocket connections for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    # Handle incoming messages if needed
                    pass
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
        finally:
            self.websockets.discard(ws)
            
        return ws
        
    async def broadcast_update(self):
        """Broadcast state updates to all connected WebSocket clients."""
        if not self.websockets:
            return
            
        states = await self.monitor.get_actor_states_async()
        
        message = json.dumps({
            'type': 'state_update',
            'data': states,
            'timestamp': time.time()
        })
        
        # Send to all connected clients
        disconnected = set()
        for ws in self.websockets:
            try:
                await ws.send_str(message)
            except ConnectionResetError:
                disconnected.add(ws)
                
        # Remove disconnected clients
        self.websockets -= disconnected
        
    async def update_loop(self):
        """Continuously update actor states and broadcast to clients."""
        while True:
            try:
                await self.monitor.update_history_async()
                await self.broadcast_update()
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                
            await asyncio.sleep(0.1)
            
    async def start(self):
        """Start the dashboard server."""
        # Start update loop
        self.update_task = asyncio.create_task(self.update_loop())
        
        # Start web server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
        
        # Print prominent connection message
        local_ip = _get_local_ip()
        print("-" * 80)
        print(f"Dashboard running at http://{local_ip}:{self.port}")
        print("-" * 80)
        
        # Keep server running
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()
        
    async def shutdown(self):
        """Properly shutdown the dashboard server."""
        logger.info("Shutting down dashboard server...")
        
        # Stop the update loop
        if hasattr(self, 'update_task'):
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Close all websockets
        for ws in list(self.websockets):
            await ws.close()
        self.websockets.clear()
        
        # Stop the web server
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
            
        # Signal the start method to exit
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
            
        logger.info("Dashboard server shutdown complete")


def start_dashboard(router_name: str = "toolshed_router", 
                   namespace: str = "toolshed",
                   host: str = '0.0.0.0',
                   port: int = 8080):
    """
    Start the toolshed dashboard server.
    
    Args:
        router_name: Name of the router actor
        namespace: Ray namespace
        host: Host to bind to (0.0.0.0 for all interfaces)
        port: Port to bind to
    """
    # Create monitor and connect to router
    monitor = ActorStateMonitor(router_name, namespace)
    monitor.connect()
    
    # Create and start server
    server = DashboardServer(monitor, host, port)
    
    # Run the async server
    asyncio.run(server.start()) 