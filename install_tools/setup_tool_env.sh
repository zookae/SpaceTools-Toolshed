#!/bin/bash
#
# Set up a complete tool environment with toolshed and tool-specific dependencies
#
# This script:
#   1. Creates a conda env with matching Python + Ray (via create_tool_env.sh)
#   2. Installs toolshed
#   3. Runs the tool-specific install script
#
# Usage:
#   source scripts/setup_tool_env.sh <env_name> <tool_name>
#
# Examples:
#   source scripts/setup_tool_env.sh tool_vlm vlm
#   source scripts/setup_tool_env.sh tool_depth depth
#   source scripts/setup_tool_env.sh tool_sam2 sam2
#
# For a blank environment (no toolshed or tool deps), use:
#   source scripts/create_tool_env.sh <env_name>
#
# The script must be sourced (not executed) to activate the new environment.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo -e "${RED}Error: Please provide environment name and tool name${NC}"
    echo "Usage: source scripts/setup_tool_env.sh <env_name> <tool_name>"
    echo ""
    echo "Available tools:"
    echo "  vlm       - Vision-Language Model (Molmo)"
    echo "  depth     - Depth Estimator (DepthPro)"
    echo "  sam2      - SAM2 Segmentation"
    echo "  bbox      - Bounding Box tool"
    echo "  roborefer - RoboRefer"
    echo "  graspgen  - GraspGen"
    echo ""
    echo "For a blank environment without tool deps, use:"
    echo "  source scripts/create_tool_env.sh <env_name>"
    return 1 2>/dev/null || exit 1
fi

ENV_NAME="$1"
TOOL_NAME="$2"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TOOLSHED_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check if install script exists for the tool
INSTALL_SCRIPT="$SCRIPT_DIR/tool_scripts/install_${TOOL_NAME}.sh"
if [ ! -f "$INSTALL_SCRIPT" ]; then
    echo -e "${RED}Error: Install script not found: $INSTALL_SCRIPT${NC}"
    echo ""
    echo "Available tool install scripts:"
    ls -1 "$SCRIPT_DIR/tool_scripts/" 2>/dev/null | grep "^install_" | sed 's/install_/  /g' | sed 's/.sh//g'
    return 1 2>/dev/null || exit 1
fi

echo -e "${GREEN}Setting up environment for tool: $TOOL_NAME${NC}"
echo ""

# Step 1: Create blank environment with Python + Ray
source "$SCRIPT_DIR/create_tool_env.sh" "$ENV_NAME"

# Step 2: Install toolshed
echo ""
echo -e "${YELLOW}Installing toolshed...${NC}"
cd "$TOOLSHED_DIR"
pip install -e .

# Step 3: Run tool-specific install script
echo ""
echo -e "${YELLOW}Running tool install script: $INSTALL_SCRIPT${NC}"
"$INSTALL_SCRIPT"

# Print success message
echo ""
echo -e "${GREEN}=========================================="
echo "Tool environment setup complete!"
echo "==========================================${NC}"
echo ""
echo "Environment '$ENV_NAME' includes:"
echo "  - Python + Ray (matching base environment)"
echo "  - toolshed (editable install)"
echo "  - $TOOL_NAME tool dependencies"
echo ""
echo -e "${GREEN}Environment '$ENV_NAME' is now active.${NC}"
