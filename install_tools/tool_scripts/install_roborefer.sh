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

# Prompt for checkpoint directory at the start (can be pre-set via env var for non-interactive use)
echo ""
echo -e "${YELLOW}RoboRefer requires a model checkpoint (~16GB).${NC}"
if [ -z "$CHECKPOINT_DIR" ]; then
    echo -n "Enter directory to save checkpoint [default: $TOOLSHED_ROOT/checkpoints]: "
    read -r CHECKPOINT_DIR
fi

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
    
    # Determine which HuggingFace CLI command is available
    # Newer huggingface_hub (>=1.0) uses 'hf', older versions use 'huggingface-cli'
    if command -v huggingface-cli &> /dev/null; then
        HF_CLI="huggingface-cli"
    elif command -v hf &> /dev/null; then
        HF_CLI="hf"
    else
        echo -e "${YELLOW}No HuggingFace CLI found. Installing huggingface-hub[cli]...${NC}"
        pip install "huggingface-hub[cli]"
        if command -v huggingface-cli &> /dev/null; then
            HF_CLI="huggingface-cli"
        elif command -v hf &> /dev/null; then
            HF_CLI="hf"
        else
            echo -e "${RED}Error: Could not find HuggingFace CLI after installation${NC}"
            exit 1
        fi
    fi
    echo -e "${GREEN}Using HuggingFace CLI: $HF_CLI${NC}"
    
    # Download the model
    $HF_CLI download Zhoues/RoboRefer-8B-SFT --local-dir "$CHECKPOINT_SUBDIR"
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
    git clone https://github.com/Zhoues/RoboRefer.git
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to clone RoboRefer repository${NC}"
        exit 1
    fi
fi

# Patch RoboRefer pyproject.toml: comment out lighteval dependency
# (it hangs during installation and is not needed for inference)
cd "$ROBOREFER_DIR"
if grep -q '^    "lighteval @' pyproject.toml 2>/dev/null; then
    echo -e "${YELLOW}Patching RoboRefer pyproject.toml: disabling lighteval dependency (hangs during install)...${NC}"
    sed -i 's|^    "lighteval @ git+https://github.com/huggingface/lighteval.git@[^"]*",|    #"lighteval (disabled - hangs during install)",|' pyproject.toml
fi

# Install cuda-nvcc so PyTorch can resolve CUDA_HOME via `which nvcc`
# (needed by deepspeed import check at runtime)
echo -e "${YELLOW}Installing cuda-nvcc (for CUDA_HOME resolution)...${NC}"
conda install -c nvidia cuda-nvcc=12.4 -y

# Pre-install CLIP to work around setuptools >= 78 removing pkg_resources.
# RoboRefer's env_setup.sh runs `pip install --upgrade setuptools` which pulls
# setuptools >= 78, and then `pip install -e ".[train,eval]"` tries to build
# CLIP from source. CLIP's setup.py uses `import pkg_resources` which no longer
# exists in setuptools >= 78 (it was split out but the standalone package is not
# on PyPI). We temporarily ensure setuptools < 78, pre-install CLIP, then let
# env_setup.sh upgrade setuptools freely — pip will see CLIP is already installed
# and skip the rebuild.
echo -e "${YELLOW}Pre-installing CLIP (workaround for setuptools >= 78 removing pkg_resources)...${NC}"
pip install "setuptools>=75,<78" 2>/dev/null
pip install --no-build-isolation "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"

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

