#!/bin/bash
# Jina-v4 Environment Activation Script

echo "⚡ Activating Jina-v4 environment..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Virtual environment activation
source "$VENV_DIR/bin/activate"

echo "✅ Jina-v4 environment activated"
echo "📁 Virtual environment: $VENV_DIR"

# Show Python and key package versions
python -c "
import sys
import torch
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Set environment variables for better performance
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

echo ""
echo "🚀 Ready to run Jina-v4 evaluation!"
echo "   Run: python run_jina-v4_evaluation.py"
echo "   Deactivate: deactivate"