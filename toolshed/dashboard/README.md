# Toolshed Dashboard

The Toolshed Dashboard provides real-time visualization of actor states in your distributed toolkit.

## Features

- **Real-time Activity Timeline**: Multi-row plot showing the state of each actor over time
  - Gray: Offline actors
  - Green: Available actors (ready to process requests)
  - Red: Working actors (currently processing)
  
- **System Overview**: Live statistics showing total actors and their current states

- **WebSocket Updates**: Real-time updates without page refresh

## Usage

### Starting the Dashboard

Enable the dashboard when starting your toolkit:

```python
from toolshed import start_toolkit

tool_configs = {
    "greeting": {"num_actors": 3},
    "calculator": {"num_actors": 2}
}

# Start toolkit with dashboard on port 8080
start_toolkit(tool_configs, dashboard=True, dashboard_port=7001)
```

### Accessing the Dashboard

Once started, open your browser and navigate to:
- Local: `http://localhost:7001`
- Remote: `http://<server-ip>:7001`

## Architecture

The dashboard consists of three main components:

1. **ActorStateMonitor**: Monitors Ray actors and tracks their states
2. **DashboardServer**: Serves the web interface and WebSocket connections
3. **Web Frontend**: HTML/JavaScript interface with real-time visualization

## How It Works

1. The monitor polls actor states every second using Ray's API
2. States are broadcast to all connected WebSocket clients
3. The frontend renders a timeline showing state transitions
4. The timeline automatically scrolls and scales to show recent activity

## Requirements

- `aiohttp`: For the web server
- `aiohttp-cors`: For CORS support
- Modern web browser with WebSocket support

## Future Enhancements

- Actor details table (planned for part 2)
- Export to monitoring systems (e.g., Weights & Biases)
- Historical data persistence
- Performance metrics and statistics 