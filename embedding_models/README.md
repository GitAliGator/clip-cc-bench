# Embedding Models Directory

This directory contains the embedding model weights used for CLIP-CC-Bench evaluation.

## Directory Structure

```
embedding_models/
├── gte-qwen2-7b/       # GTE-Qwen2-7B-Instruct model weights
├── qwen3-8b/           # Qwen3-Embedding-8B model weights
├── nemo/               # NVIDIA Nemotron-Embed model weights
├── nv-embed/           # NV-Embed-v2 model weights
└── kalm/               # KaLM-Embedding-Gemma3-12B model weights
```

## Setup Instructions

Download the required model weights from HuggingFace to their respective directories. The `--local-dir` flag downloads files directly into the specified directory without creating a subdirectory.

### 1. GTE-Qwen2-7B-Instruct
```bash
huggingface-cli download Alibaba-NLP/gte-Qwen2-7B-instruct \
  --local-dir embedding_models/gte-qwen2-7b \
  --local-dir-use-symlinks False
```

### 2. Qwen3-Embedding-8B
```bash
huggingface-cli download Alibaba-NLP/gte-Qwen2-1.5B-instruct \
  --local-dir embedding_models/qwen3-8b \
  --local-dir-use-symlinks False
```

### 3. NVIDIA Nemotron-Embed
```bash
huggingface-cli download nvidia/NV-Embed-v2 \
  --local-dir embedding_models/nemo \
  --local-dir-use-symlinks False
```

### 4. NV-Embed-v2
```bash
huggingface-cli download nvidia/NV-Embed-v2 \
  --local-dir embedding_models/nv-embed \
  --local-dir-use-symlinks False
```

### 5. KaLM-Embedding-Gemma3-12B
```bash
huggingface-cli download pgfoundation/KaLM-Embedding-Gemma3-12B-2511 \
  --local-dir embedding_models/kalm \
  --local-dir-use-symlinks False
```

## Storage Requirements

Each model requires approximately 3-24 GB of disk space:
- GTE-Qwen2-7B: ~14 GB
- Qwen3-1.5B: ~3 GB
- Nemotron-Embed (NV-Embed-v2): ~14 GB
- NV-Embed-v2: ~14 GB
- KaLM-Gemma3-12B: ~24 GB

Total: ~70 GB

## Notes

- Model weights are NOT tracked in git (see .gitignore)
- Only the directory structure and setup instructions are version controlled
- The `--local-dir-use-symlinks False` flag ensures files are copied, not symlinked
- Make sure you have sufficient disk space before downloading
- Downloaded models will be automatically detected by the evaluation scripts
- These are MTEB embedding models used as ensemble judges for evaluation, not video decoders
