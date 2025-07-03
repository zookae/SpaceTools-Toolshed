#!/bin/bash

# Function to cleanup all processes
cleanup() {
    echo "Shutting down all processes..."
    kill $PID1 $PID2 $PID3 $PID4 $PID5 2>/dev/null
    wait $PID1 $PID2 $PID3 $PID4 $PID5 2>/dev/null
    echo "All processes stopped."
    exit 0
}

# Set up trap to catch CTRL+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

UNSORTED_DIR="robot_logs/unsorted"
SAVED_CONVERSATIONS_DIR="robot_logs/saved_conversations"

mkdir -p $UNSORTED_DIR
mkdir -p $SAVED_CONVERSATIONS_DIR

# Start all processes in the background and capture their PIDs
python -m toolshed.web_ui --config configs/vision_full.json --exclude-tools vlm,code_executor,vision_ops,bounding_box,depth_estimator --hide-tool-images --logs-dir $UNSORTED_DIR --saved-conversations-dir $SAVED_CONVERSATIONS_DIR &
PID1=$!

python -m toolshed.scripts.view_multimodal_server --port 9001 $UNSORTED_DIR/ &
PID2=$!

python -m toolshed.scripts.view_multimodal_server --port 9002 $SAVED_CONVERSATIONS_DIR/ &
PID3=$!

# NEW PICK AND PLACE (Fine-tuned model on port 34567):
CUDA_VISIBLE_DEVICES=7 python -m toolshed.scripts.sglang_server_launch --model-path /lustre/fsw/portfolios/nvr/users/siyic/projects/LLaMA-Factory/saves_v4-51-1/qwen2_5vl-3b-alltasks-pickplacefulltools-2/full/sft &

PID4=$!

# Base Qwen2.5-VL-3B model (non-finetuned) on port 34568:
CUDA_VISIBLE_DEVICES=6 python -m toolshed.scripts.sglang_server_launch --model-path /lustre/fsw/portfolios/nvr/users/siyic/projects/basemodels/qwen2_5_vl_3b --port 34568 &

PID5=$!

echo "Started processes with PIDs: $PID1, $PID2, $PID3, $PID4, $PID5"
echo "Press CTRL+C to stop all processes"

# Wait for all background processes
wait $PID1 $PID2 $PID3 $PID4 $PID5
