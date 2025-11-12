#!/bin/bash
# E5-Mistral-7B-Instruct Environment Setup
# Memory-optimized installation for 7B parameter model

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "🚀 Setting up E5-Mistral-7B-Instruct environment..."
echo "Location: $SCRIPT_DIR"

# Check available memory
echo "🔍 Checking system resources..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

    # Get total GPU memory in MB
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    echo "Total GPU Memory: ${GPU_MEMORY}MB"

    if [ "$GPU_MEMORY" -lt 14000 ]; then
        echo "⚠️  WARNING: E5-Mistral-7B requires ~14GB GPU memory (float16)"
        echo "   Your GPU has ${GPU_MEMORY}MB. Consider using quantization or smaller batch sizes."
    fi
else
    echo "⚠️  No NVIDIA GPU detected. CPU inference will be very slow for 7B model."
fi

# Check available RAM
if command -v free &> /dev/null; then
    RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    echo "System RAM: ${RAM_GB}GB"
    if [ "$RAM_GB" -lt 32 ]; then
        echo "⚠️  WARNING: Recommended 32GB+ RAM for comfortable 7B model loading"
    fi
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip for better dependency resolution
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Set memory-optimized environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Install PyTorch with CUDA support first (if available)
echo "🔥 Installing PyTorch with CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    # Install CUDA version of PyTorch
    pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    # Fallback to CPU version
    echo "No CUDA detected, installing CPU version..."
    pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core requirements
echo "📚 Installing core dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Test installation
echo "🧪 Testing E5-Mistral installation..."
python -c "
import torch
import transformers
from sentence_transformers import SentenceTransformer
print('✅ Basic imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'Current CUDA device: {torch.cuda.current_device()}')
    print(f'CUDA device name: {torch.cuda.get_device_name()}')
"

# Pre-download model for faster first-time usage (optional, requires internet)
echo "🌐 Checking E5-Mistral-7B-Instruct model availability..."
python -c "
try:
    from huggingface_hub import snapshot_download, list_repo_files
    import os

    model_id = 'intfloat/e5-mistral-7b-instruct'

    # Check if model exists on HuggingFace
    try:
        files = list_repo_files(model_id)
        print(f'✅ Model {model_id} found on HuggingFace')
        print(f'   Files available: {len(files)}')

        # Optionally download model metadata (small files)
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        print(f'   HuggingFace cache: {cache_dir}')

    except Exception as e:
        print(f'⚠️  Could not access model {model_id}: {e}')

except ImportError as e:
    print(f'⚠️  Could not import huggingface_hub: {e}')
"

# Create activation shortcut
echo "🔗 Creating activation script..."
cat > "$SCRIPT_DIR/activate_env.sh" << 'EOF'
#!/bin/bash
# Activate E5-Mistral-7B-Instruct environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

echo "✅ E5-Mistral-7B-Instruct environment activated"
echo "Python: $(which python)"
echo "Pip: $(which pip)"

# Show GPU memory if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory Status:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "  Used: %sMB / %sMB (%.1f%%)\n", $1, $2, ($1/$2)*100}'
fi

echo ""
echo "🔥 Ready for E5-Mistral-7B-Instruct evaluation!"
echo "   Model: intfloat/e5-mistral-7b-instruct"
echo "   Parameters: 7B (32 layers)"
echo "   Embedding dim: 4096"
echo "   Max tokens: 4096"
echo ""
echo "💡 Memory tips for 7B model:"
echo "   - Use batch_size=1-2 for 16GB GPU"
echo "   - Use batch_size=2-4 for 24GB GPU"
echo "   - Monitor GPU memory with: watch nvidia-smi"
EOF

chmod +x "$SCRIPT_DIR/activate_env.sh"

echo ""
echo "🎉 E5-Mistral-7B-Instruct environment setup complete!"
echo ""
echo "📍 Location: $SCRIPT_DIR"
echo "🔥 Activation: source $SCRIPT_DIR/activate_env.sh"
echo ""
echo "📊 Model Specifications:"
echo "   - Model: intfloat/e5-mistral-7b-instruct"
echo "   - Parameters: 7B (32 layers)"
echo "   - Embedding dimension: 4096"
echo "   - Max sequence length: 4096 tokens"
echo "   - Memory requirement: ~14GB GPU (float16)"
echo ""
echo "⚠️  Important notes for 7B model:"
echo "   - Use conservative batch sizes (1-4)"
echo "   - Monitor GPU memory usage"
echo "   - Consider gradient checkpointing for training"
echo "   - Use torch.float16 for memory efficiency"
echo ""
echo "🚀 Ready to run E5-Mistral-7B-Instruct evaluations!"