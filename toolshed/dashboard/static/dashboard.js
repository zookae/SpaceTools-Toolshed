/**
 * Toolshed Dashboard JavaScript
 * Handles WebSocket connection and real-time visualization of actor states
 */

class ActorDashboard {
    constructor() {
        this.ws = null;
        this.actors = {};
        this.actorHistory = {};
        this.plotWidth = 1000;  // Width in pixels for the timeline
        this.timeWindow = 60;   // Show last 60 seconds
        this.pixelsPerSecond = this.plotWidth / this.timeWindow;
        this.startTime = Date.now() / 1000;
        this.canvases = {};
        this.isConnected = false;  // Track connection status
        this.lastUpdateTime = Date.now() / 1000;  // Track last update time
        
        this.stateColors = {
            'offline': '#9e9e9e',
            'initializing': '#ffc107', // Amber for in-progress start-up
            'available': '#4caf50',
            'working': '#ff5722',
            'unknown': '#cccccc',
            'disconnected': '#cccccc'  // Gray color for disconnected state
        };
        
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.setupActivityPlot();
        this.startRenderLoop();
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus(true);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus(false);
            this.handleDisconnection();
            // Reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.isConnected = false;
            this.handleDisconnection();
        };
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'state_update') {
                this.lastUpdateTime = Date.now() / 1000;
                this.updateActorStates(message.data);
            }
        };
    }
    
    handleDisconnection() {
        // Add a disconnected state to all actors' history
        const disconnectTime = Date.now() / 1000;
        // Freeze the timeline at the moment of disconnection so the
        // newly-added "disconnected" block is visible when rendering.
        this.lastUpdateTime = disconnectTime;
        
        for (const actorId of Object.keys(this.actors)) {
            if (this.actorHistory[actorId]) {
                // Add disconnected state to history
                this.actorHistory[actorId].push({
                    time: disconnectTime,
                    state: 'disconnected'
                });
            }
        }
        
        // Update stats to show all as disconnected
        this.updateStatsForDisconnection();
    }
    
    updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        if (connected) {
            statusEl.textContent = 'Connected';
            statusEl.className = 'connection-status status-connected';
        } else {
            statusEl.textContent = 'Disconnected';
            statusEl.className = 'connection-status status-disconnected';
        }
    }
    
    setupActivityPlot() {
        const plotContainer = document.getElementById('activity-plot');
        plotContainer.innerHTML = ''; // Clear existing content
    }
    
    updateActorStates(states) {
        const currentTime = Date.now() / 1000;
        
        // Flatten actor states
        const flatActors = {};
        for (const [toolName, actors] of Object.entries(states)) {
            for (const actor of actors) {
                flatActors[actor.id] = actor;
                
                // Initialize history if needed
                if (!this.actorHistory[actor.id]) {
                    this.actorHistory[actor.id] = [];
                }
                
                // Add to history
                this.actorHistory[actor.id].push({
                    time: currentTime,
                    state: actor.state
                });
                
                // Keep only recent history
                const cutoffTime = currentTime - this.timeWindow * 2;
                this.actorHistory[actor.id] = this.actorHistory[actor.id].filter(
                    entry => entry.time > cutoffTime
                );
            }
        }
        
        this.actors = flatActors;
        this.updateStats();
        this.updateActivityPlot();
    }
    
    updateStats() {
        let total = 0;
        let available = 0;
        let working = 0;
        let initializing = 0;
        let offline = 0;
        
        for (const actor of Object.values(this.actors)) {
            total++;
            switch (actor.state) {
                case 'available':
                    available++;
                    break;
                case 'working':
                    working++;
                    break;
                case 'initializing':
                    initializing++;
                    break;
                case 'offline':
                case 'disconnected':
                    offline++;
                    break;
            }
        }
        
        document.getElementById('total-actors').textContent = total;
        document.getElementById('available-actors').textContent = available;
        document.getElementById('working-actors').textContent = working;
        document.getElementById('initializing-actors').textContent = initializing;
        document.getElementById('offline-actors').textContent = offline;
    }
    
    updateStatsForDisconnection() {
        const total = Object.keys(this.actors).length;
        document.getElementById('total-actors').textContent = total;
        document.getElementById('available-actors').textContent = 0;
        document.getElementById('working-actors').textContent = 0;
        document.getElementById('initializing-actors').textContent = 0;
        document.getElementById('offline-actors').textContent = total;
    }
    
    updateActivityPlot() {
        const plotContainer = document.getElementById('activity-plot');
        const actorIds = Object.keys(this.actors).sort();
        
        // Calculate dynamic height based on number of actors
        const rowHeight = 30;
        const minHeight = 200; // Minimum height for empty state
        const calculatedHeight = Math.max(minHeight, actorIds.length * rowHeight);
        plotContainer.style.height = `${calculatedHeight}px`;
        
        // Create rows for new actors
        actorIds.forEach((actorId, index) => {
            if (!document.getElementById(`actor-row-${actorId}`)) {
                const row = document.createElement('div');
                row.className = 'actor-row';
                row.id = `actor-row-${actorId}`;
                row.style.height = '30px';
                
                const label = document.createElement('div');
                label.className = 'actor-label';
                label.textContent = actorId;
                row.appendChild(label);
                
                const canvas = document.createElement('canvas');
                canvas.className = 'activity-canvas';
                canvas.width = this.plotWidth;
                canvas.height = 30;
                canvas.style.width = `${this.plotWidth}px`;
                canvas.style.height = '30px';
                row.appendChild(canvas);
                
                plotContainer.appendChild(row);
                this.canvases[actorId] = canvas;
            }
        });
        
        // Remove rows for actors that no longer exist
        const existingRows = plotContainer.querySelectorAll('.actor-row');
        existingRows.forEach(row => {
            const actorId = row.id.replace('actor-row-', '');
            if (!this.actors[actorId]) {
                row.remove();
                delete this.canvases[actorId];
            }
        });
    }
    
    renderActivityTimelines() {
        // Use lastUpdateTime when disconnected to pause scrolling
        const currentTime = this.isConnected ? Date.now() / 1000 : this.lastUpdateTime;
        const startTime = currentTime - this.timeWindow;
        
        for (const [actorId, canvas] of Object.entries(this.canvases)) {
            const ctx = canvas.getContext('2d');
            const history = this.actorHistory[actorId] || [];
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw background
            ctx.fillStyle = '#f5f5f5';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            if (history.length === 0) continue;
            
            // Draw state blocks
            for (let i = 0; i < history.length; i++) {
                const entry = history[i];
                const nextEntry = history[i + 1];
                
                const startX = Math.max(0, (entry.time - startTime) * this.pixelsPerSecond);
                const endTime = nextEntry ? nextEntry.time : currentTime;
                const endX = Math.min(canvas.width, (endTime - startTime) * this.pixelsPerSecond);
                
                if (endX > startX) {
                    ctx.fillStyle = this.stateColors[entry.state] || this.stateColors.unknown;
                    ctx.fillRect(startX, 0, endX - startX, canvas.height);
                }
            }
            
            // Draw current time indicator
            ctx.strokeStyle = this.isConnected ? '#333' : '#999';  // Dim when disconnected
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(canvas.width - 1, 0);
            ctx.lineTo(canvas.width - 1, canvas.height);
            ctx.stroke();
            
            // If disconnected, add an overlay to dim the entire canvas
            if (!this.isConnected) {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
            }
        }
    }
    
    startRenderLoop() {
        const render = () => {
            this.renderActivityTimelines();
            requestAnimationFrame(render);
        };
        render();
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new ActorDashboard();
}); 