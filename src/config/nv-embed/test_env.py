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
