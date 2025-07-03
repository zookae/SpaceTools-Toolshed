#!/bin/bash
#
# Install dependencies for the GraspGen tool
#
# Usage:
#   ./install_tools/tool_scripts/install_graspgen.sh
#
# Must be run from an activated tool environment (e.g., tool_graspgen)
#
# This script will:
#   1. Install pip requirements (if any)
#   2. Clone GraspGen from gitlab
#   3. Install GraspGen and pointnet2_ops
#   4. Download model weights from HuggingFace
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLSHED_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REQUIREMENTS_FILE="$TOOLSHED_ROOT/install_tools/requirements/tool-graspgen.txt"

GRASPGEN_REPO="https://github.com/NVlabs/GraspGen.git"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Installing GraspGen tool dependencies...${NC}"

# Prompt for locations at the start
echo ""
echo -e "${YELLOW}GraspGen needs to be cloned and installed.${NC}"
echo -e "${YELLOW}(This can be wherever you want)${NC}"
echo -n "Enter directory to clone GraspGen into [default: $TOOLSHED_ROOT/GraspGen]: "
read -r GRASPGEN_DIR

if [ -z "$GRASPGEN_DIR" ]; then
    GRASPGEN_DIR="$TOOLSHED_ROOT"
fi

# Expand ~ if present
GRASPGEN_DIR="${GRASPGEN_DIR/#\~/$HOME}"

echo ""
echo -e "${YELLOW}GraspGen requires model weights from HuggingFace.${NC}"
echo -e "${YELLOW}(This can be wherever you want)${NC}"
echo -n "Enter directory to save models [default: $TOOLSHED_ROOT/checkpoints]: "
read -r MODELS_DIR

if [ -z "$MODELS_DIR" ]; then
    MODELS_DIR="$TOOLSHED_ROOT/checkpoints"
fi

# Expand ~ if present
MODELS_DIR="${MODELS_DIR/#\~/$HOME}"

# Install pip requirements if file exists and has content
if [ -f "$REQUIREMENTS_FILE" ]; then
    # Check if file has any non-comment, non-empty lines
    if grep -q '^[^#]' "$REQUIREMENTS_FILE" 2>/dev/null; then
        echo -e "${YELLOW}Installing pip requirements...${NC}"
        pip install -r "$REQUIREMENTS_FILE"
    fi
fi

cd "$GRASPGEN_DIR"

# Clone or update GraspGen
if [ -d "GraspGen" ]; then
    echo -e "${YELLOW}GraspGen directory already exists. Pulling latest...${NC}"
    cd GraspGen
    git pull
else
    echo -e "${YELLOW}Cloning GraspGen...${NC}"
    git clone "$GRASPGEN_REPO"
    cd GraspGen
fi

# Install GraspGen
echo -e "${YELLOW}Installing GraspGen...${NC}"
pip install -e .

# Install pointnet2_ops
echo -e "${YELLOW}Installing pointnet2_ops (this may take a while)...${NC}"
cd pointnet2_ops
pip install --no-build-isolation .
cd ..

# Test import
echo -e "${YELLOW}Testing import...${NC}"
python -c "from grasp_gen.grasp_server import GraspGenSampler" && echo -e "${GREEN}Import successful!${NC}"

mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# Clone model weights if not present
if [ -d "GraspGenModels" ]; then
    echo -e "${GREEN}GraspGenModels already exists: $MODELS_DIR/GraspGenModels${NC}"
else
    echo -e "${YELLOW}Cloning GraspGenModels from HuggingFace...${NC}"
    git clone https://huggingface.co/adithyamurali/GraspGenModels
fi

echo ""
echo -e "${GREEN}=========================================="
echo "GraspGen tool installation complete!"
echo "==========================================${NC}"
echo ""
echo "GraspGen installed at: $GRASPGEN_DIR/GraspGen"
echo "Models saved to: $MODELS_DIR/GraspGenModels"
echo ""
echo "If you used a different checkpoint directory, configure the gripper config path:"
echo "  tool_configs = {"
echo "      \"grasp_generator\": {"
echo "          \"args\": {\"gripper_config\": \"path/to/graspgen_franka_panda.yml\"}"
echo "      }"
echo "  }"

