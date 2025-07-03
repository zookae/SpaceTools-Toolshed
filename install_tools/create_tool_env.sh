#!/bin/bash
#
# Create a blank conda environment with matching Python and Ray versions
#
# This script creates a minimal environment suitable for developing new tools
# or experimenting. It only installs Python and Ray - nothing else.
#
# Usage:
#   source scripts/create_tool_env.sh <env_name>
#
# Example:
#   source scripts/create_tool_env.sh tool_my_new_tool
#
# After running, you can install toolshed and any dependencies you need:
#   pip install -e .
#   pip install <your-dependencies>
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
if [ -z "$1" ]; then
    echo -e "${RED}Error: Please provide an environment name${NC}"
    echo "Usage: source scripts/create_tool_env.sh <env_name>"
    echo ""
    echo "Example:"
    echo "  source scripts/create_tool_env.sh tool_my_new_tool"
    return 1 2>/dev/null || exit 1
fi

ENV_NAME="$1"

# Check if we're in a conda environment with Ray
if ! python -c "import ray" 2>/dev/null; then
    echo -e "${RED}Error: Ray not found in current environment.${NC}"
    echo "Please activate a toolshed environment first:"
    echo "  conda activate toolshed"
    return 1 2>/dev/null || exit 1
fi

# Capture current Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}Detected Python version: $PYTHON_VERSION${NC}"

# Capture current Ray version
RAY_VERSION=$(python -c "import ray; print(ray.__version__)")
echo -e "${GREEN}Detected Ray version: $RAY_VERSION${NC}"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Warning: Environment '$ENV_NAME' already exists.${NC}"
    echo -n "Continue with existing (c), recreate (r), or abort (a)? [c/r/a] "
    read -r response
    if [ "$response" = "r" ] || [ "$response" = "R" ]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
        # Create new conda environment
        echo ""
        echo -e "${YELLOW}Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION...${NC}"
        conda create -n "$ENV_NAME" python=="$PYTHON_VERSION" -y
        
        # Activate the new environment
        echo ""
        echo -e "${YELLOW}Activating environment: $ENV_NAME...${NC}"
        eval "$(conda shell.bash hook)"
        conda activate "$ENV_NAME"
        
        # Install Ray with the matching version
        echo ""
        echo -e "${YELLOW}Installing ray==$RAY_VERSION...${NC}"
        pip install "ray[default]==$RAY_VERSION"
    elif [ "$response" = "c" ] || [ "$response" = "C" ]; then
        echo "Continuing with existing environment..."
        # Activate the existing environment
        echo ""
        echo -e "${YELLOW}Activating environment: $ENV_NAME...${NC}"
        eval "$(conda shell.bash hook)"
        conda activate "$ENV_NAME"
    else
        echo "Aborting."
        return 1 2>/dev/null || exit 1
    fi
else
    # Create new conda environment
    echo ""
    echo -e "${YELLOW}Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION...${NC}"
    conda create -n "$ENV_NAME" python=="$PYTHON_VERSION" -y

    # Activate the new environment
    echo ""
    echo -e "${YELLOW}Activating environment: $ENV_NAME...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    # Install Ray with the matching version
    echo ""
    echo -e "${YELLOW}Installing ray==$RAY_VERSION...${NC}"
    pip install "ray[default]==$RAY_VERSION"
fi

# Print success message
echo ""
echo -e "${GREEN}=========================================="
echo "Success!"
echo "==========================================${NC}"
echo ""
echo "Created conda environment '$ENV_NAME' with:"
echo "  - Python $PYTHON_VERSION"
echo "  - Ray $RAY_VERSION"
echo ""
echo -e "${GREEN}Environment '$ENV_NAME' is now active.${NC}"
echo ""
echo "Next steps:"
echo "  cd toolshed"
echo "  pip install -e .                              # Install toolshed"
echo "  pip install -r requirements/tool-<name>.txt   # Install tool deps (optional)"
echo ""
