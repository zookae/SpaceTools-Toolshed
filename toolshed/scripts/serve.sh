#!/bin/bash

# Function to cleanup all processes
cleanup() {
    echo "Shutting down all processes..."
    kill $PID1 $PID2 $PID3 2>/dev/null
    wait $PID1 $PID2 $PID3 2>/dev/null
    echo "All processes stopped."
    exit 0
}

# Set up trap to catch CTRL+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

UNSORTED_DIR="logs/unsorted"
SAVED_CONVERSATIONS_DIR="logs/saved_conversations"

mkdir -p $UNSORTED_DIR
mkdir -p $SAVED_CONVERSATIONS_DIR

# Start all processes in the background and capture their PIDs
python -m toolshed.web_ui --config configs/vision_full.json --logs-dir $UNSORTED_DIR --saved-conversations-dir $SAVED_CONVERSATIONS_DIR &
PID1=$!

python -m toolshed.scripts.view_multimodal_server --port 9001 $UNSORTED_DIR/ &
PID2=$!

python -m toolshed.scripts.view_multimodal_server --port 9002 $SAVED_CONVERSATIONS_DIR/ &
PID3=$!

echo "Started processes with PIDs: $PID1, $PID2, $PID3"
echo "Press CTRL+C to stop all processes"

# Wait for all background processes
wait $PID1 $PID2 $PID3