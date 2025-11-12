#!/usr/bin/env python3
"""
Download Nomic-Embed-text-v1.5 model weights from HuggingFace Hub.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger('download_model')

def main():
    logger = setup_logging()

    # Determine paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent.parent
    model_dir = base_dir.parent / "encoder_models" / "nomic-embed-text-v1.5"  # Centralized location

    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Model will be saved to: {model_dir}")

    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Starting download of nomic-ai/nomic-embed-text-v1.5...")
        logger.info("This may take several minutes depending on your internet connection...")

        # Download the model using snapshot_download
        downloaded_path = snapshot_download(
            repo_id="nomic-ai/nomic-embed-text-v1.5",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,  # Use actual files instead of symlinks
            resume_download=True,  # Resume if partially downloaded
            allow_patterns=None,  # Download all files
            ignore_patterns=[".git*", "*.md", "*.txt", "*.py~"]  # Skip some unnecessary files
        )

        logger.info(f"Model downloaded successfully to: {downloaded_path}")

        # Verify key files exist
        key_files = ["config.json", "modules.json", "pytorch_model.bin"]
        missing_files = []

        for file_name in key_files:
            file_path = model_dir / file_name
            if file_path.exists():
                logger.info(f"✓ Found {file_name}")
            else:
                # Check for safetensors alternative
                if file_name == "pytorch_model.bin":
                    safetensors_files = list(model_dir.glob("*.safetensors"))
                    if safetensors_files:
                        logger.info(f"✓ Found safetensors files: {[f.name for f in safetensors_files]}")
                        continue

                missing_files.append(file_name)
                logger.warning(f"✗ Missing {file_name}")

        if missing_files:
            logger.error(f"Some required files are missing: {missing_files}")
            logger.error("The download may be incomplete. Please try running the script again.")
            return 1

        logger.info("✅ All required model files are present!")
        logger.info("✅ Nomic-Embed-text-v1.5 model download completed successfully!")

        # Show directory contents
        logger.info("\nDownloaded files:")
        for file_path in sorted(model_dir.iterdir()):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  {file_path.name} ({size_mb:.1f} MB)")

        return 0

    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)