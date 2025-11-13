# CLIP-CC-Bench: Text Embedding-Based Evaluation for Video Captioning Models

An evaluation framework that uses state-of-the-art text embedding models to measure the quality of video captions by computing semantic similarity between model predictions and ground truth.

## Overview

CLIP-CC-Bench is a text embedding evaluation framework that:
- Evaluates 17 video language models (VLMs) using 4 state-of-the-art text embedding models
- Measures caption quality through embedding similarity (cosine similarity, fine-grained metrics)
- Supports both coarse-grained (full caption) and fine-grained (sentence-level) evaluation
- Provides isolated implementations for each embedding model for reproducibility
- Evaluates models on the CLIP-CC dataset with reference captions

**Important**: This is not a video captioning system - it's an evaluation tool that measures how well video captioning models perform by comparing their outputs to ground truth using text embeddings.

## Project Structure

```
clip-cc-bench/
├── src/
│   ├── configs/              # Embedding model configurations (YAML)
│   │   ├── nv-embed.yaml     # NV-Embed-v2 configuration
│   │   ├── nemo.yaml         # Llama-Embed-Nemotron configuration
│   │   ├── gte.yaml          # GTE-Qwen2-7B configuration
│   │   ├── qwen.yaml         # Qwen3-Embedding configuration
│   │   └── requirements/     # Python dependencies per model
│   ├── scripts/              # Evaluation scripts (one per embedding model)
│   │   ├── run_nv_embed_evaluation.py
│   │   ├── run_nemo_evaluation.py
│   │   ├── run_gte_evaluation.py
│   │   └── run_qwen_evaluation.py
│   └── utils/                # Model implementations and utilities
│       ├── nv_embed_model.py # NV-Embed model wrapper
│       ├── nemo_model.py     # Nemotron model wrapper
│       ├── gte_model.py      # GTE model wrapper
│       ├── qwen_model.py     # Qwen3 model wrapper
│       ├── base_types.py     # Shared data types
│       ├── config_loader.py  # Configuration loading
│       ├── result_manager.py # Results management
│       └── text_chunking.py  # Sentence chunking for fine-grained eval
├── data/
│   ├── ground_truth/         # CLIP-CC dataset reference captions
│   │   └── clip_cc_dataset.json (463KB)
│   └── models/               # Video model predictions (17 models)
│       ├── internvl.json, llava_next_video.json, etc.
│       └── (predictions from 17 VLMs)
├── results/                  # Evaluation outputs (JSON/CSV)
└── README.md
```

## Text Embedding Models

The framework evaluates captions using 4 state-of-the-art text embedding models:

| Model | Embedding Dim | Max Length | Implementation | Status |
|-------|---------------|------------|----------------|---------|
| **NV-Embed-v2** (NVIDIA) | 4,096 | 32,768 | Custom local model | ✅ Verified |
| **Llama-Embed-Nemotron-8B** (NVIDIA) | 4,096 | 4,096 | SentenceTransformer | ✅ Verified |
| **GTE-Qwen2-7B-instruct** (Alibaba) | 3,584 | 8,192 | SentenceTransformer | ✅ Verified |
| **Qwen3-Embedding-8B** (Qwen) | Variable | 32,768 | SentenceTransformer | ✅ Verified |

**Implementation Verification**: All 4 models use the official/recommended implementations from their respective HuggingFace model cards.

## Video Language Models Evaluated

The framework evaluates predictions from **17 video language models**:

- internvl, llava_next_video, llava_one_vision
- longva, longvu, minicpm, mplug, oryx
- Qwen2.5-32B, Qwen2.5-72B
- sharegpt4, timechat, ts_llava
- videochatflash, videollama3, video_xl, vilamp

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (24GB+ VRAM recommended for large models)
- ~50-60GB disk space for embedding models
- PyTorch 2.0+
- transformers>=4.42.0
- sentence-transformers>=2.7.0

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/clip-cc-bench.git
   cd clip-cc-bench
   ```

2. **Install dependencies**

   Each embedding model has its own requirements file:
   ```bash
   # For NV-Embed-v2
   pip install -r src/configs/requirements/nv-embed.txt

   # For Llama-Embed-Nemotron-8B
   pip install -r src/configs/requirements/nemo.txt

   # For GTE-Qwen2-7B-instruct
   pip install -r src/configs/requirements/gte.txt

   # For Qwen3-Embedding-8B
   pip install -r src/configs/requirements/qwen.txt
   ```

3. **Download embedding models**

   The models will be automatically downloaded from HuggingFace on first run, or you can pre-download them:
   ```bash
   # Models will be stored in decoder_models/ directory
   # NV-Embed-v2: nvidia/NV-Embed-v2
   # Nemotron: nvidia/llama-embed-nemotron-8b
   # GTE: Alibaba-NLP/gte-Qwen2-7B-instruct
   # Qwen3: Qwen/Qwen3-Embedding-8B
   ```

4. **Prepare your data** (Optional - sample data included)

   The repository includes sample ground truth and model predictions. To use your own:
   ```bash
   # Place ground truth in data/ground_truth/clip_cc_dataset.json
   # Place model predictions in data/models/{model_name}.json
   ```

### Quick Start

Run evaluation using any of the 4 embedding models:

```bash
# 1. NV-Embed-v2
python src/scripts/run_nv_embed_evaluation.py

# 2. Llama-Embed-Nemotron-8B
python src/scripts/run_nemo_evaluation.py

# 3. GTE-Qwen2-7B-instruct
python src/scripts/run_gte_evaluation.py

# 4. Qwen3-Embedding-8B
python src/scripts/run_qwen_evaluation.py
```

Each script will:
1. Load the embedding model
2. Load ground truth captions from `data/ground_truth/clip_cc_dataset.json`
3. Load predictions from all models in `data/models/`
4. Compute similarity scores (coarse + fine-grained)
5. Save results to `results/decoders/` directory

## Configuration

Each embedding model has a YAML configuration file in `src/configs/`:

**Example: `src/configs/nv-embed.yaml`**
```yaml
decoder:
  name: "nv-embed"
  path: "decoder_models/nv-embed"
  type: "nvembed"
  batch_size: 8
  max_length: 32768
  trust_remote_code: true

  # Fine-grained evaluation settings
  fine_grained:
    enabled: true           # Enable sentence-level similarity
    chunking_method: "nltk" # NLTK sentence tokenizer

processing:
  device: "cuda:0"
  clear_cache_interval: 25  # Clear GPU cache every N samples
  progress_interval: 10     # Log progress every N samples

data_paths:
  ground_truth_file: "data/ground_truth/clip_cc_dataset.json"
  predictions_dir: "data/models"
  results_base_dir: "results"

# Specify which models to evaluate
models_to_evaluate:
  - "internvl"
  - "llava_next_video"
  # ... (17 models total)

logging:
  level: "INFO"
  log_dir: "results/decoders/logs/nv-embed"
```

You can customize batch size, max length, device, and which models to evaluate by editing these config files.

## Usage

### Basic Usage

Run evaluation with default settings (all 17 models):

```bash
# Evaluate using NV-Embed-v2
python src/scripts/run_nv_embed_evaluation.py

# Evaluate using Llama-Embed-Nemotron-8B
python src/scripts/run_nemo_evaluation.py

# Evaluate using GTE-Qwen2-7B-instruct
python src/scripts/run_gte_evaluation.py

# Evaluate using Qwen3-Embedding-8B
python src/scripts/run_qwen_evaluation.py
```

### Evaluate Specific Models Only

```bash
# Evaluate only specific video models
python src/scripts/run_nv_embed_evaluation.py --models internvl llava_next_video longvu

# Use custom base directory
python src/scripts/run_nemo_evaluation.py --base-dir /path/to/project
```

### Command-Line Options

All evaluation scripts support:
- `--config PATH`: Path to custom config file (optional)
- `--models MODEL1 MODEL2 ...`: Specific models to evaluate (optional, defaults to all 17)
- `--base-dir PATH`: Custom base directory (optional, auto-detected by default)

## Evaluation Metrics

The framework computes multiple similarity metrics:

### Coarse-Grained Metrics
- **Cosine Similarity**: Raw cosine similarity between full caption embeddings (range: [-1, 1])
- **Normalized Cosine**: Same as cosine similarity (for compatibility)

### Fine-Grained Metrics (Sentence-Level)
- **Precision**: For each prediction sentence, find best matching ground truth sentence
- **Recall**: For each ground truth sentence, find best matching prediction sentence
- **F1 Score**: Harmonic mean of precision and recall
- **HM-CF**: Harmonic mean of coarse similarity and fine-grained F1

### GAS-Style Implementation
All models use "GAS-style" (raw cosine similarity without [0,1] normalization), matching the original GAS paper implementation.

## Results

Results are saved in JSON and CSV formats:

```
results/
└── decoders/
    ├── individual_results/
    │   ├── json/
    │   │   ├── nv-embed_internvl_results.json
    │   │   ├── nv-embed_llava_next_video_results.json
    │   │   └── ... (per-model results)
    │   └── csv/
    │       └── ... (CSV versions)
    └── logs/
        ├── nv-embed/
        ├── nemo/
        ├── gte/
        └── qwen/
```

### Result Format

Each result file contains:
```json
{
  "video_id": "xyz123",
  "model_name": "internvl",
  "ground_truth_text": "A person walks across a bridge...",
  "prediction_text": "A man walking on a bridge...",
  "decoder_similarities": {
    "nv-embed": {
      "cosine_similarity": 0.85,
      "fine_grained_precision": 0.82,
      "fine_grained_recall": 0.88,
      "fine_grained_f1": 0.85,
      "hm_cf": 0.85
    }
  },
  "success": true,
  "timestamp": "2025-11-13T12:34:56"
}
```

## Data Format

### Ground Truth (`data/ground_truth/clip_cc_dataset.json`)
```json
[
  {
    "id": "video_001",
    "summary": "A person walks across a bridge while the sun sets..."
  },
  {
    "id": "video_002",
    "summary": "A cat jumps onto a table and knocks over a vase..."
  }
]
```

### Model Predictions (`data/models/{model_name}.json`)
```json
{
  "video_001": "A man walking on a bridge during sunset...",
  "video_002": "Cat jumping on table, vase falls down..."
}
```

**Format Notes:**
- Ground truth is a list of objects with `"id"` and `"summary"` fields
- Predictions are dictionaries mapping video IDs to predicted captions
- Video IDs must match between ground truth and predictions

## Implementation Details

### Model-Specific Implementation Notes

**1. NV-Embed-v2**
- Uses custom local model from `modeling_nvembed.py`
- Loads with `trust_remote_code=True`
- Max sequence length: 32,768 tokens
- Implementation: Custom NVEmbedModel class

**2. Llama-Embed-Nemotron-8B**
- Uses SentenceTransformer library
- Special methods: `encode_query()` and `encode_document()`
- Falls back to standard `encode()` if methods unavailable
- Attention: "eager" or "flash_attention_2"
- Padding: Left-side padding required

**3. GTE-Qwen2-7B-instruct**
- Uses SentenceTransformer library
- Supports prompt_name="query" for query encoding
- Documents encoded without prompts (query-side instruction tuning)
- Max sequence length: 8,192 tokens

**4. Qwen3-Embedding-8B**
- Uses SentenceTransformer library
- Supports prompt_name="query" for queries
- Flash attention 2 support for better performance
- Max sequence length: 32,768 tokens

### Adding a New Embedding Model

1. **Create configuration file**
   ```bash
   # Create src/configs/new_model.yaml
   # Add requirements to src/configs/requirements/new_model.txt
   ```

2. **Implement model wrapper**
   ```bash
   # Create src/utils/new_model_model.py
   # Implement loading, encoding, and similarity computation
   ```

3. **Create evaluation script**
   ```bash
   # Create src/scripts/run_new_model_evaluation.py
   # Follow pattern from existing scripts
   ```

4. **Test the implementation**
   ```bash
   python src/scripts/run_new_model_evaluation.py --models internvl
   ```

## Key Features

✅ **Verified Implementations**: All 4 embedding models use official/recommended implementations from HuggingFace
✅ **Fine-Grained Evaluation**: Supports both full-caption and sentence-level similarity metrics
✅ **Flexible Configuration**: YAML-based configs for easy customization
✅ **Production Ready**: Includes logging, error handling, GPU memory management
✅ **Comprehensive Evaluation**: Tests 17 state-of-the-art video language models

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@inproceedings{clip-cc-bench-2025,
  title={Text Embedding-Based Evaluation Framework for Video Captioning Models},
  author={Your Name},
  booktitle={Conference Name},
  year={2025}
}
```

## License

[Specify your license here - MIT, Apache 2.0, etc.]

## Acknowledgments

- **NVIDIA** for NV-Embed-v2 and Llama-Embed-Nemotron-8B models
- **Alibaba NLP** for GTE-Qwen2-7B-instruct model
- **Qwen Team** for Qwen3-Embedding-8B model
- **HuggingFace** for model hosting and transformers library
- **Sentence-Transformers** for the excellent embedding library

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Solution 1: Reduce batch size in config
# Edit src/configs/{model}.yaml and set batch_size: 4 or lower

# Solution 2: Clear GPU cache manually
import torch; torch.cuda.empty_cache()

# Solution 3: Use smaller models (GTE-Qwen2-7B or Qwen3)
```

**2. Model download fails**
```bash
# Ensure you have HuggingFace access
huggingface-cli login

# For NV-Embed-v2, you may need to request access on HuggingFace
# Visit: https://huggingface.co/nvidia/NV-Embed-v2
```

**3. Import errors (missing dependencies)**
```bash
# Install requirements for specific model
pip install -r src/configs/requirements/{model}.txt

# Or install all dependencies
pip install torch transformers sentence-transformers nltk numpy
```

**4. Ground truth/prediction file not found**
```bash
# Verify files exist
ls data/ground_truth/clip_cc_dataset.json
ls data/models/

# Check file paths in config
cat src/configs/{model}.yaml
```

**5. Sentence chunking errors**
```bash
# Download NLTK punkt tokenizer
python -c "import nltk; nltk.download('punkt')"
```

## FAQ

**Q: Can I use my own dataset?**
A: Yes! Replace `data/ground_truth/clip_cc_dataset.json` and add predictions to `data/models/`. Follow the data format shown in this README.

**Q: Can I evaluate just one video model instead of all 17?**
A: Yes! Use `--models` flag: `python src/scripts/run_nv_embed_evaluation.py --models internvl`

**Q: Which embedding model should I use?**
A: It depends on your needs:
- **Longest context**: NV-Embed-v2 (32K) or Qwen3 (32K)
- **Fastest**: GTE-Qwen2-7B (smaller 7B model)
- **Most features**: Llama-Embed-Nemotron-8B (separate query/document encoding)

**Q: How long does evaluation take?**
A: Depends on dataset size and model. For 17 models with ~1000 videos each, expect 1-4 hours per embedding model on a single GPU.

---

**Note**: This is a research evaluation framework. Model implementations follow official HuggingFace recommendations as of November 2025.
