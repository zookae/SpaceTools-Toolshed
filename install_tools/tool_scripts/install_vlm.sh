#!/bin/bash
#
# Install dependencies for the VLM (Vision-Language Model) tool
#
# Usage:
#   ./install_tools/tool_scripts/install_vlm.sh
#
# Must be run from an activated tool environment (e.g., tool_vlm)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLSHED_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REQUIREMENTS_FILE="$TOOLSHED_ROOT/install_tools/requirements/tool-vlm.txt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Installing VLM tool dependencies...${NC}"

# Check requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${RED}Error: Requirements file not found: $REQUIREMENTS_FILE${NC}"
    exit 1
fi

# Install pip requirements
echo -e "${YELLOW}Installing pip requirements...${NC}"
pip install -r "$REQUIREMENTS_FILE"

echo ""
echo -e "${GREEN}=========================================="
echo "VLM tool installation complete!"
echo "==========================================${NC}"
echo ""
echo "Model weights will be downloaded automatically from HuggingFace"
echo "when the tool is first used."

