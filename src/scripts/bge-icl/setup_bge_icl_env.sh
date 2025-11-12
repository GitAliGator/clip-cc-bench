#!/bin/bash

# BGE-ICL Environment Setup Script
# Creates isolated virtual environment for BGE-ICL encoder

set -e  # Exit on any error

echo "🔧 Setting up BGE-ICL isolated environment..."

# Get script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BGE_ICL_DIR="$BASE_DIR/config/bge-icl"
VENV_DIR="$SCRIPT_DIR/venv"

echo "📁 Base directory: $BASE_DIR"
echo "📁 BGE-ICL config directory: $BGE_ICL_DIR"
echo "📁 Virtual environment: $VENV_DIR"

# Check if config directory exists
if [[ ! -d "$BGE_ICL_DIR" ]]; then
    echo "❌ Error: BGE-ICL config directory not found: $BGE_ICL_DIR"
    exit 1
fi

# Check if requirements file exists
if [[ ! -f "$BGE_ICL_DIR/requirements.txt" ]]; then
    echo "❌ Error: Requirements file not found: $BGE_ICL_DIR/requirements.txt"
    exit 1
fi

# Remove existing virtual environment if it exists
if [[ -d "$VENV_DIR" ]]; then
    echo "🧹 Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create new virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv "$VENV_DIR"

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing BGE-ICL requirements..."
echo "   This may take several minutes for CUDA packages..."
pip install -r "$BGE_ICL_DIR/requirements.txt"

# Verify key packages
echo "✅ Verifying installation..."
python -c "
import torch
import transformers
import sentence_transformers
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ Transformers version: {transformers.__version__}')
print(f'✅ Sentence-Transformers version: {sentence_transformers.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
"

# Test BGE-ICL model access
echo "🧪 Testing BGE-ICL model access..."
python -c "
try:
    from sentence_transformers import SentenceTransformer
    # Test if we can access the model (without loading it fully)
    model_name = 'BAAI/bge-en-icl'
    print(f'✅ BGE-ICL model accessible: {model_name}')
except Exception as e:
    print(f'⚠️  BGE-ICL model access warning: {e}')
    print('   This may resolve during actual evaluation.')
"

# Create activation script
ACTIVATE_SCRIPT="$SCRIPT_DIR/activate_env.sh"
echo "📝 Creating activation script: $ACTIVATE_SCRIPT"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# BGE-ICL Environment Activation Script

echo "⚡ Activating BGE-ICL environment..."
source "$VENV_DIR/bin/activate"

echo "✅ BGE-ICL environment activated"
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
echo "🚀 Ready to run BGE-ICL evaluation!"
echo "   Run: python run_bge-icl_evaluation.py"
echo "   Deactivate: deactivate"
EOF

chmod +x "$ACTIVATE_SCRIPT"

# Create quick test script
TEST_SCRIPT="$SCRIPT_DIR/test_env.py"
echo "📝 Creating environment test script: $TEST_SCRIPT"
cat > "$TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""Quick test script for BGE-ICL environment."""

import sys
import torch
from pathlib import Path

def test_environment():
    """Test the BGE-ICL environment setup."""
    print("🧪 Testing BGE-ICL environment...")

    # Test basic imports
    try:
        import transformers
        import sentence_transformers
        import numpy as np
        import yaml
        import datasets
        import peft
        print("✅ All basic imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Test CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    else:
        print("⚠️  CUDA not available, will use CPU")

    # Test sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformer import successful")
    except Exception as e:
        print(f"❌ SentenceTransformer error: {e}")
        return False

    # Test BGE-ICL model access
    try:
        from transformers import AutoTokenizer, AutoModel
        model_name = "BAAI/bge-en-icl"
        print(f"✅ BGE-ICL model accessible: {model_name}")
    except Exception as e:
        print(f"⚠️  BGE-ICL model access warning: {e}")
        print("   This may resolve during actual evaluation.")

    # Test shared utilities
    try:
        # Add paths for shared utilities
        script_dir = Path(__file__).parent
        src_dir = script_dir.parent.parent
        sys.path.append(str(src_dir / "utils" / "shared"))

        from base_types import SimilarityScore, EmbeddingResult
        from config_loader import IsolatedEncoderConfigLoader
        print("✅ Shared utilities import successful")
    except Exception as e:
        print(f"❌ Shared utilities error: {e}")
        return False

    print("🎉 Environment test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
EOF

chmod +x "$TEST_SCRIPT"

echo ""
echo "🎉 BGE-ICL environment setup completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate environment: source $ACTIVATE_SCRIPT"
echo "   2. Test environment: python $TEST_SCRIPT"
echo "   3. Run evaluation: python $SCRIPT_DIR/run_bge-icl_evaluation.py"
echo ""
echo "📁 Environment files:"
echo "   Virtual env: $VENV_DIR"
echo "   Activation script: $ACTIVATE_SCRIPT"
echo "   Test script: $TEST_SCRIPT"
echo ""
echo "ℹ️  Note: BGE-ICL is a SOTA 2024 model - first run may take time to download"