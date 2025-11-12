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
