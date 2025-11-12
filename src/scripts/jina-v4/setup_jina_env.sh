#!/bin/bash

# Jina-v4 Environment Setup Script
# Sets up isolated virtual environment for Jina-v4 embeddings evaluation

set -e  # Exit on any error

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../../config/jina-v4"
VENV_DIR="$SCRIPT_DIR/venv"
REQUIREMENTS_FILE="$CONFIG_DIR/requirements.txt"

echo "🔧 Setting up Jina-v4 environment..."
echo "📁 Config directory: $CONFIG_DIR"
echo "🐍 Virtual environment: $VENV_DIR"

# Create virtual environment if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
    echo "🔨 Creating virtual environment for Jina-v4..."
    python3 -m venv "$VENV_DIR"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📦 Installing Jina-v4 requirements..."
echo "🔍 Requirements file: $REQUIREMENTS_FILE"

if [[ -f "$REQUIREMENTS_FILE" ]]; then
    pip install -r "$REQUIREMENTS_FILE"
    echo "✅ Requirements installed successfully"
else
    echo "❌ Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

# Verify installation
echo "🔍 Verifying Jina-v4 installation..."
python -c "
import torch
import transformers
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA devices: {torch.cuda.device_count()}')
"

echo ""
echo "🎉 Jina-v4 environment setup completed successfully!"
echo ""

# Ask user if they want to download the model
echo "📥 Would you like to download the Jina-v4 model to local storage? (y/n)"
read -r download_choice

if [[ "$download_choice" =~ ^[Yy]$ ]]; then
    echo "📥 Downloading Jina-v4 model..."
    python "$CONFIG_DIR/download_jina_model.py"

    if [[ $? -eq 0 ]]; then
        echo "✅ Model download completed!"
    else
        echo "❌ Model download failed - model will be downloaded from HuggingFace during evaluation"
    fi
else
    echo "⏭️  Skipping model download - model will be downloaded from HuggingFace during evaluation"
fi

echo ""
echo "💡 To activate this environment manually:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "🚀 To run Jina-v4 evaluation:"
echo "   cd $SCRIPT_DIR"
echo "   source ./activate_env.sh"
echo "   python run_jina-v4_evaluation.py"
echo ""
echo "📥 To download model manually:"
echo "   python $CONFIG_DIR/download_jina_model.py"
echo ""
echo "🛑 To deactivate:"
echo "   deactivate"