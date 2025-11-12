#!/bin/bash

# BGE-ICL Environment Setup Script
# Creates isolated virtual environment with proper FlashAttention dependency handling

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

echo "🔧 Setting up BGE-ICL isolated environment..."

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

# Remove existing virtual environment if it exists
if [ -d "$VENV_PATH" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf "$VENV_PATH"
fi

# Create new virtual environment
echo "📦 Creating new virtual environment..."
python3 -m venv "$VENV_PATH"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" = "" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment created and activated"
echo "📁 Location: $VIRTUAL_ENV"

# Upgrade pip, setuptools, and wheel first
echo "⬆️  Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install PyTorch and related packages first (required for FlashAttention compilation)
echo "🔥 Installing PyTorch and core dependencies..."
pip install torch==2.4.0 transformers==4.44.2 tokenizers==0.19.1

# Install FlashAttention separately after PyTorch is available
echo "⚡ Installing FlashAttention (this may take a while)..."
if pip install flash-attn>=2.5.0 --no-build-isolation; then
    echo "✅ FlashAttention installed successfully"
else
    echo "⚠️  FlashAttention installation failed, proceeding without it"
    echo "    BGE-ICL will work but may be slower for long contexts"
fi

# Install remaining requirements
echo "📚 Installing remaining dependencies..."
pip install sentence-transformers==3.0.1
pip install numpy>=1.21.0
pip install datasets>=2.19.0
pip install accelerate>=0.20.1
pip install sentencepiece
pip install protobuf
pip install peft==0.12.0
pip install FlagEmbedding>=1.2.10
pip install scikit-learn
pip install scipy
pip install Pillow
pip install "PyYAML>=6.0"
pip install tqdm
pip install pandas
pip install psutil
pip install "huggingface-hub>=0.21.0"
pip install "safetensors>=0.4.3"

# Verify installation
echo "🔍 Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from FlagEmbedding import FlagICLModel; print('FlagEmbedding: OK')"

# Check FlashAttention
if python -c "import flash_attn; print(f'FlashAttention: {flash_attn.__version__}')" 2>/dev/null; then
    echo "✅ FlashAttention is available"
else
    echo "⚠️  FlashAttention is not available (will impact performance)"
fi

echo ""
echo "🎉 BGE-ICL environment setup completed!"
echo ""
echo "To activate the environment:"
echo "  source $SCRIPT_DIR/activate_env.sh"
echo ""
echo "To run BGE-ICL evaluation:"
echo "  cd /mmfs2/home/jacks.local/mali9292/aaai_student_abstract/clip-cc-bench/src/scripts/bge-icl"
echo "  source /mmfs2/home/jacks.local/mali9292/aaai_student_abstract/clip-cc-bench/src/config/bge-icl/activate_env.sh"
echo "  python run_bge-icl_evaluation.py"
echo ""