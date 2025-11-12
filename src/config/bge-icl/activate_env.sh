#!/bin/bash

# BGE-ICL Environment Activation Script
# Activates the isolated BGE-ICL virtual environment for evaluation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/venv"

echo "⚡ Activating BGE-ICL environment..."

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "   Please run setup_bge-icl_env.sh first"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ BGE-ICL environment activated"
    echo "📁 Virtual environment: $VIRTUAL_ENV"
    echo "Python: $(python --version 2>&1)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
    echo ""
    echo "🚀 Ready to run BGE-ICL evaluation!"
    echo "   Run: python /mmfs2/home/jacks.local/mali9292/aaai_student_abstract/clip-cc-bench/src/scripts/bge-icl/run_bge-icl_evaluation.py"
    echo "   Deactivate: deactivate"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi