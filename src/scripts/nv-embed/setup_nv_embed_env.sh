#!/bin/bash

# NV-Embed Environment Setup Script
# Creates isolated virtual environment for NV-Embed encoder

set -e  # Exit on any error

echo "🔧 Setting up NV-Embed isolated environment..."

# Get script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
NV_EMBED_DIR="$BASE_DIR/config/nv-embed"
VENV_DIR="$SCRIPT_DIR/venv"

echo "📁 Base directory: $BASE_DIR"
echo "📁 NV-Embed config directory: $NV_EMBED_DIR"
echo "📁 Virtual environment: $VENV_DIR"

# Check if config directory exists
if [[ ! -d "$NV_EMBED_DIR" ]]; then
    echo "❌ Error: NV-Embed config directory not found: $NV_EMBED_DIR"
    exit 1
fi

# Check if requirements file exists
if [[ ! -f "$NV_EMBED_DIR/requirements.txt" ]]; then
    echo "❌ Error: Requirements file not found: $NV_EMBED_DIR/requirements.txt"
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
echo "📦 Installing NV-Embed requirements..."
echo "   This may take several minutes for CUDA packages..."
pip install -r "$NV_EMBED_DIR/requirements.txt"

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

# Create activation script
ACTIVATE_SCRIPT="$SCRIPT_DIR/activate_env.sh"
echo "📝 Creating activation script: $ACTIVATE_SCRIPT"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# NV-Embed Environment Activation Script

echo "⚡ Activating NV-Embed environment..."
source "$VENV_DIR/bin/activate"

echo "✅ NV-Embed environment activated"
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
echo "🚀 Ready to run NV-Embed evaluation!"
echo "   Run: python run_nv-embed_evaluation.py"
echo "   Deactivate: deactivate"
EOF

chmod +x "$ACTIVATE_SCRIPT"

# Create quick test script
TEST_SCRIPT="$SCRIPT_DIR/test_env.py"
echo "📝 Creating environment test script: $TEST_SCRIPT"
cat > "$TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""Quick test script for NV-Embed environment."""

import sys
import torch
from pathlib import Path

def test_environment():
    """Test the NV-Embed environment setup."""
    print("🧪 Testing NV-Embed environment...")

    # Test basic imports
    try:
        import transformers
        import sentence_transformers
        import numpy as np
        import yaml
        import datasets
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
echo "🎉 NV-Embed environment setup completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate environment: source $ACTIVATE_SCRIPT"
echo "   2. Test environment: python $TEST_SCRIPT"
echo "   3. Run evaluation: python run_nv-embed_evaluation.py"
echo ""
echo "📁 Environment files:"
echo "   Virtual env: $VENV_DIR"
echo "   Activation script: $ACTIVATE_SCRIPT"
echo "   Test script: $TEST_SCRIPT"