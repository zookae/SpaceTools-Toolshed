#!/bin/bash
#
# Install dependencies for the Depth Estimator tool
#
# Usage:
#   ./install_tools/tool_scripts/install_depth.sh
#
# Must be run from an activated tool environment (e.g., tool_depth)
#
# This script will:
#   1. Install pip requirements
#   2. Prompt for checkpoint directory and download depth_pro.pt
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLSHED_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REQUIREMENTS_FILE="$TOOLSHED_ROOT/install_tools/requirements/tool-depth.txt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Installing Depth Estimator tool dependencies...${NC}"

# Prompt for checkpoint directory at the start
echo ""
echo -e "${YELLOW}Depth Pro requires a model checkpoint (~370MB).${NC}"
echo -n "Enter directory to save checkpoint [default: $TOOLSHED_ROOT/checkpoints]: "
read -r CHECKPOINT_DIR

if [ -z "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_DIR="$TOOLSHED_ROOT/checkpoints"
fi

# Expand ~ if present
CHECKPOINT_DIR="${CHECKPOINT_DIR/#\~/$HOME}"

# Check requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${RED}Error: Requirements file not found: $REQUIREMENTS_FILE${NC}"
    exit 1
fi

# Install pip requirements
echo -e "${YELLOW}Installing pip requirements...${NC}"
pip install -r "$REQUIREMENTS_FILE"

# Create directory
mkdir -p "$CHECKPOINT_DIR"

# Download checkpoint if not already present
CHECKPOINT_FILE="$CHECKPOINT_DIR/depth_pro.pt"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo -e "${GREEN}Checkpoint already exists: $CHECKPOINT_FILE${NC}"
else
    echo -e "${YELLOW}Downloading depth_pro.pt to $CHECKPOINT_DIR...${NC}"
    wget -q --show-progress https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -O "$CHECKPOINT_FILE"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Depth Estimator tool installation complete!"
echo "==========================================${NC}"
echo ""
echo "Checkpoint saved to: $CHECKPOINT_FILE"
echo ""
echo "To use the tool, configure the checkpoint path:"
echo "  tool_configs = {"
echo "      \"depth_estimator\": {"
echo "          \"args\": {\"checkpoint_path\": \"$CHECKPOINT_FILE\"}"
echo "      }"
echo "  }"

