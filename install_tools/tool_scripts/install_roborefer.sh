#!/bin/bash
#
# Install dependencies for the RoboRefer tool
#
# Usage:
#   ./install_tools/tool_scripts/install_roborefer.sh
#
# Must be run from an activated tool environment (e.g., tool_roborefer)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLSHED_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REQUIREMENTS_FILE="$TOOLSHED_ROOT/install_tools/requirements/tool-roborefer.txt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Installing RoboRefer tool dependencies...${NC}"

# Prompt for checkpoint directory at the start
echo ""
echo -e "${YELLOW}RoboRefer requires a model checkpoint (~16GB).${NC}"
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
CHECKPOINT_SUBDIR="$CHECKPOINT_DIR/RoboRefer-8B-SFT"
if [ -d "$CHECKPOINT_SUBDIR" ] && [ -f "$CHECKPOINT_SUBDIR/config.json" ]; then
    echo -e "${GREEN}Checkpoint already exists: $CHECKPOINT_SUBDIR${NC}"
else
    echo -e "${YELLOW}Downloading RoboRefer-8B-SFT from HuggingFace to $CHECKPOINT_DIR...${NC}"
    
    # Check if huggingface-cli is available
    if ! command -v huggingface-cli &> /dev/null; then
        echo -e "${YELLOW}huggingface-cli not found. Installing huggingface-hub...${NC}"
        pip install huggingface-hub
    fi
    
    # Download the model
    huggingface-cli download Zhoues/RoboRefer-8B-SFT --local-dir "$CHECKPOINT_SUBDIR" --local-dir-use-symlinks False
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to download RoboRefer checkpoint${NC}"
        exit 1
    fi
fi

# Clone RoboRefer repository if it doesn't exist
ROBOREFER_DIR="$TOOLSHED_ROOT/RoboRefer"
if [ -d "$ROBOREFER_DIR" ]; then
    echo -e "${YELLOW}RoboRefer directory already exists at $ROBOREFER_DIR${NC}"
    echo -e "${YELLOW}Skipping clone. To reinstall, remove the directory first.${NC}"
else
    echo -e "${YELLOW}Cloning RoboRefer repository...${NC}"
    cd "$TOOLSHED_ROOT"
    git clone git@github.com:nvalts/RoboRefer.git
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to clone RoboRefer repository${NC}"
        exit 1
    fi
fi

# Run RoboRefer setup script
echo -e "${YELLOW}Running RoboRefer env_setup.sh...${NC}"
cd "$ROBOREFER_DIR"
bash env_setup.sh

# Install pillow=12.0.0 (RoboRefer overwrites this with version 11, but 12 is required for compatibility with base environment)
echo "Installing pillow==12.0.0..."
pip install pillow==12.0.0

echo ""
echo -e "${GREEN}=========================================="
echo "RoboRefer tool installation complete!"
echo "==========================================${NC}"
echo ""
echo "Checkpoint saved to: $CHECKPOINT_SUBDIR"
echo ""
echo "If you used a different checkpoint directory, configure the checkpoint path:"
echo "  tool_configs = {"
echo "      \"roborefer\": {"
echo "          \"init_args\": {\"model_path\": \"$CHECKPOINT_SUBDIR\"}"
echo "      }"
echo "  }"

