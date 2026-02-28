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

# Prompt for locations at the start (can be pre-set via env vars for non-interactive use)
echo ""
echo -e "${YELLOW}GraspGen needs to be cloned and installed.${NC}"
echo -e "${YELLOW}(This can be wherever you want)${NC}"
if [ -z "$GRASPGEN_DIR" ]; then
    echo -n "Enter directory to clone GraspGen into [default: $TOOLSHED_ROOT/GraspGen]: "
    read -r GRASPGEN_DIR
fi

if [ -z "$GRASPGEN_DIR" ]; then
    GRASPGEN_DIR="$TOOLSHED_ROOT"
fi

# Expand ~ if present
GRASPGEN_DIR="${GRASPGEN_DIR/#\~/$HOME}"

echo ""
echo -e "${YELLOW}GraspGen requires model weights from HuggingFace.${NC}"
echo -e "${YELLOW}(This can be wherever you want)${NC}"
if [ -z "$MODELS_DIR" ]; then
    echo -n "Enter directory to save models [default: $TOOLSHED_ROOT/checkpoints]: "
    read -r MODELS_DIR
fi

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

# ---------------------------------------------------------------------------
# Patch GraspGen's pyproject.toml to allow torch 2.3+ and numpy 2.x
# GraspGen upstream pins torch==2.1.0 and numpy==1.26.4, but torch 2.1
# cannot interoperate with numpy 2.x (torch.from_numpy / .numpy() crash).
# torch >=2.3 restores numpy 2.x compatibility. The GraspGen code itself
# uses no torch-2.1-specific APIs, so this upgrade is safe.
# ---------------------------------------------------------------------------
echo -e "${YELLOW}Patching GraspGen pyproject.toml for torch 2.3 + numpy 2.x...${NC}"
sed -i 's/"torch==2.1.0"/"torch>=2.3.0,<2.4"/' pyproject.toml
sed -i 's/"torchvision==0.16.0"/"torchvision>=0.18.0,<0.19"/' pyproject.toml
sed -i 's/"numpy==1.26.4"/"numpy>=2.0"/' pyproject.toml
# Switch from spconv-cu120 to spconv-cu121 to match torch's CUDA version
sed -i 's/"spconv-cu120"/"spconv-cu121"/' pyproject.toml

# Verify the patches took effect. If GraspGen upstream changed their pins,
# the sed commands above would silently no-op and we'd end up with torch 2.1
# + numpy 1.x, which causes runtime serialisation failures with numpy 2.x envs.
if grep -q '"torch==2.1' pyproject.toml; then
    echo -e "${RED}ERROR: Failed to patch torch version in pyproject.toml.${NC}"
    echo -e "${RED}GraspGen may have changed its version pins. Please update the sed patterns in this script.${NC}"
    exit 1
fi
if grep -q '"numpy==1\.' pyproject.toml; then
    echo -e "${RED}ERROR: Failed to patch numpy version in pyproject.toml.${NC}"
    echo -e "${RED}GraspGen may have changed its version pins. Please update the sed patterns in this script.${NC}"
    exit 1
fi
if grep -q '"spconv-cu120"' pyproject.toml; then
    echo -e "${RED}ERROR: Failed to patch spconv-cu120 → spconv-cu121 in pyproject.toml.${NC}"
    echo -e "${RED}GraspGen may have changed its version pins. Please update the sed patterns in this script.${NC}"
    exit 1
fi
echo -e "${GREEN}Patches applied successfully.${NC}"

# Install CUDA toolkit so PyTorch can resolve CUDA_HOME via `which nvcc`
# and the compiler can find CUDA headers (cuda_runtime.h, cusparse.h, etc.)
# and libraries (libcudart) needed to build pointnet2_ops.
echo -e "${YELLOW}Installing CUDA toolkit (for CUDA_HOME, headers, and libs)...${NC}"
conda install -c nvidia cuda-toolkit=12.1 -y

# Install GraspGen
# --find-links provides pre-built PyG wheels (torch-cluster, torch-scatter)
# so pip doesn't try to compile them from source in an isolated build env
echo -e "${YELLOW}Installing GraspGen...${NC}"
pip install -e . --find-links https://data.pyg.org/whl/torch-2.3.0+cu121.html

# Install pointnet2_ops with explicit CUDA architectures
# (needed when building on nodes without a GPU; defaults cover V100 through H100)
#
# NOTE: Must be built on a node with the same (or older) glibc as runtime nodes.
# The cuda-toolkit conda package places some headers under targets/x86_64-linux/include/
# and some under $CONDA_PREFIX/include, so we add both to CPATH.
# We also set CUDA_HOME explicitly and LIBRARY_PATH for the linker.
echo -e "${YELLOW}Installing pointnet2_ops (this may take a while)...${NC}"
cd pointnet2_ops
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH:-}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9;9.0}" \
    pip install --no-build-isolation .
cd ..

# Ensure numpy 2.x is installed (pip may have resolved a 1.x version
# to satisfy a transitive dependency; override it).
echo -e "${YELLOW}Ensuring numpy 2.x...${NC}"
pip install "numpy>=2.0" --force-reinstall --no-deps

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

