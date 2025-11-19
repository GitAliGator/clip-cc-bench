# Decoder Models Directory

This directory contains the decoder model weights used for CLIP-CC-Bench evaluation.

## Directory Structure

```
decoder_models/
├── gte-qwen2-7b/       # GTE-Qwen2-7B-Instruct model weights
├── qwen3-8b/           # Qwen3-Embedding-8B model weights
├── nemo/               # NVIDIA Llama-Embed-Nemotron-8B model weights
├── nv-embed/           # NV-Embed model weights
└── kalm/               # KaLM-Embedding-Gemma3-12B model weights
```

## Setup Instructions

Download the required model weights from HuggingFace and place them in their respective directories:

### 1. GTE-Qwen2-7B-Instruct
```bash
# Download from HuggingFace
huggingface-cli download Alibaba-NLP/gte-Qwen2-7B-instruct --local-dir decoder_models/gte-qwen2-7b
```

### 2. Qwen3-Embedding-8B
```bash
# Download from HuggingFace
huggingface-cli download Alibaba-NLP/gte-Qwen2-7B-instruct --local-dir decoder_models/qwen3-8b
```

### 3. NVIDIA Nemotron
```bash
# Download from HuggingFace
huggingface-cli download nvidia/Llama-3.1-Nemotron-70B-Instruct-HF --local-dir decoder_models/nemo
```

### 4. NV-Embed
```bash
# Download from HuggingFace
huggingface-cli download nvidia/NV-Embed-v2 --local-dir decoder_models/nv-embed
```

### 5. KaLM-Embedding-Gemma3-12B
```bash
# Download from HuggingFace
huggingface-cli download pgfoundation/KaLM-Embedding-Gemma3-12B-2511 --local-dir decoder_models/kalm
```

## Storage Requirements

Each model requires approximately 7-15 GB of disk space:
- GTE-Qwen2-7B: ~14 GB
- Qwen3-8B: ~16 GB
- Nemotron: ~140 GB (70B model)
- NV-Embed: ~14 GB
- KaLM: ~24 GB

Total: ~208 GB

## Notes

- Model weights are NOT tracked in git (see .gitignore)
- Only the directory structure and setup instructions are version controlled
- Make sure you have sufficient disk space before downloading
- Downloaded models will be automatically detected by the evaluation scripts
