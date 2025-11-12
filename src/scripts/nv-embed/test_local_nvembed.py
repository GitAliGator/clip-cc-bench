#!/usr/bin/env python3
"""
Simple test script for local NV-Embed implementation
"""

import sys
import torch
from pathlib import Path

# Add local NV-Embed implementation path
sys.path.append("/mmfs2/home/jacks.local/mali9292/scratch/VAD-LLM2Vec/nv-embed")

try:
    from modeling_nvembed import NVEmbedModel as LocalNVEmbedModel
    print("✅ Successfully imported LocalNVEmbedModel")

    # Test loading the model
    model_path = "/mmfs2/home/jacks.local/mali9292/scratch/VAD-LLM2Vec/nv-embed"
    print(f"🔄 Attempting to load model from: {model_path}")

    model = LocalNVEmbedModel.from_pretrained(
        model_path,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    print("✅ Model loaded successfully!")
    print(f"   Model type: {type(model)}")
    print(f"   Device: {next(model.parameters()).device}")

    # Test encoding
    test_texts = ["This is a test sentence.", "Another test sentence."]
    print(f"🔄 Testing encoding of {len(test_texts)} texts...")

    with torch.no_grad():
        embeddings = model.encode(test_texts, instruction="", max_length=4096)
        print(f"✅ Encoding successful!")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Embeddings dtype: {embeddings.dtype}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()