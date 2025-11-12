#!/bin/bash
# Nomic-Embed-text-v1.5 Environment Activation Script

# Get the directory of this script
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$(dirname "$CONFIG_DIR")")/scripts/nomic-embed-text-v1.5"
VENV_PATH="$SCRIPTS_DIR/venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "Please run the setup script first: bash $SCRIPTS_DIR/setup_nomic_embed_env.sh"
    return 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" = "$VENV_PATH" ]; then
    echo "✅ Nomic-Embed-text-v1.5 environment activated: $VIRTUAL_ENV"
else
    echo "❌ Failed to activate Nomic-Embed-text-v1.5 environment"
    return 1
fi