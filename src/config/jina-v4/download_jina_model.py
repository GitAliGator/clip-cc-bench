#!/usr/bin/env python3
"""
Download Jina-v4 Model Script

Downloads the Jina-v4 model from HuggingFace to the local encoder_models directory.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_jina_v4_model():
    """Download Jina-v4 model to local directory."""
    # Get paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent.parent  # Go up to clip-cc-bench root
    local_model_dir = base_dir.parent / "encoder_models" / "jina-embeddings-v4"  # Centralized location

    logger.info(f"Downloading Jina-v4 model to: {local_model_dir}")

    # Create directory if it doesn't exist
    local_model_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download the model
        logger.info("Starting download from HuggingFace...")
        snapshot_download(
            repo_id="jinaai/jina-embeddings-v4",
            local_dir=str(local_model_dir),
            local_dir_use_symlinks=False,  # Create actual files, not symlinks
            resume_download=True,  # Resume if partially downloaded
            token=None,  # No auth token needed for public models
        )

        logger.info("✅ Jina-v4 model downloaded successfully!")

        # Verify download
        config_file = local_model_dir / "config.json"
        if config_file.exists():
            logger.info(f"✅ Verified: config.json exists")
        else:
            logger.warning("⚠️  config.json not found - download may be incomplete")

        # Check for model weights
        model_files = list(local_model_dir.glob("*.safetensors")) + list(local_model_dir.glob("*.bin"))
        if model_files:
            logger.info(f"✅ Verified: Found {len(model_files)} model weight files")
            for model_file in model_files:
                logger.info(f"   - {model_file.name}")
        else:
            logger.warning("⚠️  No model weight files found - download may be incomplete")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to download Jina-v4 model: {e}")
        return False

if __name__ == "__main__":
    success = download_jina_v4_model()
    sys.exit(0 if success else 1)