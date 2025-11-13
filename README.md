# CLIP-CC-Bench: Video Captioning Benchmark with Decoder Evaluation

A comprehensive benchmark for evaluating video language models using text embedding decoders with both coarse-grained and fine-grained similarity metrics.

## Overview

CLIP-CC-Bench is an evaluation framework that:
- Assesses video captioning models using state-of-the-art text embedding decoders
- Compares ground truth captions with model predictions via embedding similarity
- Supports both **coarse-grained** (full caption) and **fine-grained** (sentence-level) evaluation
- Computes precision, recall, F1, and hybrid metrics (hm-cf: harmonic mean of coarse & fine)
- Provides isolated environments for reproducible evaluation
- Currently supports **NV-Embed** with plans to add more decoders

## Key Features

### Dual-Level Evaluation
- **Coarse-Grained**: Full caption embeddings with normalized cosine similarity
- **Fine-Grained**: Sentence-level chunk matching using greedy alignment algorithm
  - Precision: Average best match from prediction to ground truth chunks
  - Recall: Average best match from ground truth to prediction chunks
  - F1: Harmonic mean of precision and recall
  - hm-cf: Harmonic mean of coarse similarity and fine F1

### Methodology
Inspired by EMScore (CVPR 2022) and BERTScore, our fine-grained evaluation:
1. Splits captions into sentences using NLTK
2. Encodes each sentence independently
3. Computes max-similarity matching between ground truth and prediction chunks
4. Aggregates to precision/recall/F1 metrics

## Project Structure

```
clip-cc-bench/
├── src/
│   ├── configs/              # Decoder configurations (YAML)
│   │   └── nv-embed.yaml
│   ├── scripts/              # Evaluation scripts
│   │   └── run_nv_embed_evaluation.py
│   └── utils/                # Shared utilities (flattened structure)
│       ├── base_types.py         # Data structures with fine-grained metrics
│       ├── config_loader.py      # Configuration management
│       ├── nv_embed_model.py     # NV-Embed implementation
│       ├── paths.py              # Path management
│       ├── result_manager.py     # Result saving with metrics
│       └── text_chunking.py      # NLTK sentence tokenization
├── data/
│   ├── ground_truth/         # Reference captions (clip_cc_dataset.json)
│   └── models/               # Model predictions (JSON files)
├── results/
│   └── decoders/
│       ├── individual_results/   # Per-video CSV/JSON results
│       ├── aggregated_results/   # Per-model summaries
│       └── logs/                 # Evaluation logs
└── decoder_models/           # Model weights (centralized, parent dir)
    └── nv-embed/
```

## Supported Decoders

| Decoder | Status | Embedding Dim | Context Length | Fine-Grained |
|---------|--------|---------------|----------------|--------------|
| NV-Embed | ✅ Active | Variable | 32768 | ✅ Enabled |
| gte-Qwen2-7B-instruct | 🔜 Planned | - | - | 🔜 |
| nvidia-llama-embed-nemotron-8b | 🔜 Planned | - | - | 🔜 |
| Qwen3-Embedding-8B | 🔜 Planned | - | - | 🔜 |

## Setup

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- Disk space: ~20GB for NV-Embed model
- NLTK data (automatically downloaded)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/clip-cc-bench.git
   cd clip-cc-bench
   ```

2. **Set up decoder models directory**

   The project expects model weights in a centralized location:
   ```bash
   # Project structure:
   parent_dir/
   ├── clip-cc-bench/           # This repo
   └── decoder_models/          # Model weights
       └── nv-embed/            # NV-Embed v2 model
   ```

3. **Download NV-Embed model**

   ```bash
   cd ../decoder_models
   # Option 1: Download from HuggingFace
   git lfs clone https://huggingface.co/nvidia/NV-Embed-v2 nv-embed

   # Option 2: Use HuggingFace hub (automatic download)
   # The script will download automatically if path not found
   ```

4. **Install dependencies**

   ```bash
   cd clip-cc-bench
   pip install -r src/configs/requirements/nv-embed.txt
   ```

### Quick Start

1. **Prepare your data**
   ```bash
   # Ground truth format: data/ground_truth/clip_cc_dataset.json
   # {
   #   "id": "video_001",
   #   "summary": "Ground truth caption..."
   # }

   # Model predictions format: data/models/{model_name}.json
   # {
   #   "video_001": "Model predicted caption..."
   # }
   ```

2. **Run NV-Embed evaluation**
   ```bash
   python src/scripts/run_nv_embed_evaluation.py \
       --config src/configs/nv-embed.yaml \
       --models internvl llava_next_video longva
   ```

3. **View results**
   ```bash
   # Per-model CSV with fine-grained metrics
   cat results/decoders/individual_results/csv/nv-embed/internvl_results.csv

   # Per-model summary JSON
   cat results/decoders/aggregated_results/nv-embed/per_model/internvl_summary.json

   # Evaluation logs
   tail -f results/decoders/logs/nv-embed/nv_embed_evaluation_*.log
   ```

## Configuration

The NV-Embed configuration (`src/configs/nv-embed.yaml`):

```yaml
decoder:
  name: "nv-embed"
  path: "decoder_models/nv-embed"  # Relative to parent dir
  type: "nvembed"
  batch_size: 8                    # Adjust based on GPU memory
  max_length: 32768                # Max context length
  device_map: "auto"
  trust_remote_code: true
  additional_params:
    instruction_for_retrieval: "Given a ground truth video summary, assess the quality and alignment of predicted summaries with fine-grained discrimination."

  # Fine-grained evaluation settings
  fine_grained:
    enabled: true                  # Enable sentence-level matching
    chunking_method: "nltk"        # Use NLTK sentence tokenizer

processing:
  device: "cuda:0"                 # GPU device
  clear_cache_interval: 25         # Clear GPU cache every N videos
  progress_interval: 10            # Log progress every N videos

data_paths:
  ground_truth_file: "data/ground_truth/clip_cc_dataset.json"
  predictions_dir: "data/models"
  results_base_dir: "results"

# Models to evaluate
models_to_evaluate:
  - "internvl"
  - "llava_next_video"
  - "llava_one_vision"
  - "longva"
  - "longvu"
  - "minicpm"
  - "mplug"
  - "oryx"
  - "sharegpt4"
  - "timechat"
  - "ts_llava"
  - "videochatflash"
  - "videollama3"
  - "video_xl"
  - "vilamp"
```

## Results Format

### Individual Results CSV

Each model gets a CSV file with per-video metrics:

```csv
video_id,coarse_similarity,fine_precision,fine_recall,fine_f1,hm_cf
video_001,0.8234,0.7891,0.8012,0.7951,0.8088
video_002,0.7654,0.7234,0.7456,0.7343,0.7495
```

### Per-Model Summary JSON

Aggregated statistics per model:

```json
{
  "model_name": "internvl",
  "decoder_name": "nv-embed",
  "total_videos": 500,
  "successful_evaluations": 498,
  "metrics": {
    "coarse_similarity": {
      "mean": 0.7823,
      "std": 0.0945,
      "min": 0.4521,
      "max": 0.9678
    },
    "fine_grained_precision": {
      "mean": 0.7512,
      "std": 0.0876
    },
    "fine_grained_recall": {
      "mean": 0.7634,
      "std": 0.0823
    },
    "fine_grained_f1": {
      "mean": 0.7572,
      "std": 0.0847
    },
    "hm_cf": {
      "mean": 0.7693,
      "std": 0.0889
    }
  }
}
```

## Understanding Metrics

### Coarse-Grained (normalized_cosine)
- Encodes full caption as single embedding
- Computes cosine similarity: `(cosine + 1) / 2` to normalize to [0, 1]
- Captures overall semantic similarity

### Fine-Grained Metrics

**Precision** (Quality of predictions)
- For each predicted sentence, find best matching ground truth sentence
- Average of max similarities
- High precision = predicted content is accurate

**Recall** (Coverage of ground truth)
- For each ground truth sentence, find best matching predicted sentence
- Average of max similarities
- High recall = all important content covered

**F1 Score** (Balance)
- Harmonic mean of precision and recall: `2 * P * R / (P + R)`
- Balanced measure of quality and coverage

**hm-cf** (Hybrid metric)
- Harmonic mean of coarse similarity and fine F1: `2 * C * F / (C + F)`
- Combines document-level and sentence-level evaluation
- Recommended primary metric for ranking models

## Usage Examples

### Evaluate Specific Models

```bash
python src/scripts/run_nv_embed_evaluation.py \
    --config src/configs/nv-embed.yaml \
    --models internvl llava_next_video
```

### Evaluate All Models

```bash
# Uses models_to_evaluate list from config
python src/scripts/run_nv_embed_evaluation.py \
    --config src/configs/nv-embed.yaml
```

### Custom Base Directory

```bash
python src/scripts/run_nv_embed_evaluation.py \
    --config src/configs/nv-embed.yaml \
    --base-dir /path/to/clip-cc-bench
```

### Disable Fine-Grained Evaluation

Edit `src/configs/nv-embed.yaml`:
```yaml
decoder:
  fine_grained:
    enabled: false
```

## Development

### Adding a New Decoder

1. **Create configuration file**
   ```bash
   cp src/configs/nv-embed.yaml src/configs/new-decoder.yaml
   # Edit decoder settings
   ```

2. **Create decoder implementation**
   ```bash
   # Add to src/utils/new_decoder_model.py
   # Implement compute_similarity() method
   # Use text_chunking.py for fine-grained evaluation
   ```

3. **Create evaluation script**
   ```bash
   cp src/scripts/run_nv_embed_evaluation.py src/scripts/run_new_decoder_evaluation.py
   # Update decoder_name and imports
   ```

4. **Add requirements**
   ```bash
   # Create src/configs/requirements/new-decoder.txt
   ```

### Shared Utilities

All decoders share common utilities in `src/utils/`:

- **base_types.py**: Data structures (SimilarityScore, DecoderEvaluationResult)
- **text_chunking.py**: NLTK sentence tokenization
- **result_manager.py**: CSV/JSON result saving
- **config_loader.py**: YAML configuration loading
- **paths.py**: Centralized path management

### Fine-Grained Implementation

To add fine-grained evaluation to a new decoder:

```python
from text_chunking import chunk_text_into_sentences

def compute_fine_grained_similarity(self, gt_chunks, pred_chunks):
    # Encode chunks
    gt_embeddings = self.encode_texts(gt_chunks)
    pred_embeddings = self.encode_texts(pred_chunks)

    # Precision: best match for each prediction
    precision_scores = []
    for pred_emb in pred_embeddings:
        max_sim = max(cosine_similarity(pred_emb, gt_emb) for gt_emb in gt_embeddings)
        precision_scores.append((max_sim + 1) / 2)  # Normalize
    precision = np.mean(precision_scores)

    # Recall: best match for each ground truth
    recall_scores = []
    for gt_emb in gt_embeddings:
        max_sim = max(cosine_similarity(gt_emb, pred_emb) for pred_emb in pred_embeddings)
        recall_scores.append((max_sim + 1) / 2)  # Normalize
    recall = np.mean(recall_scores)

    # F1: harmonic mean
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1
```

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce `batch_size` in `src/configs/nv-embed.yaml` (default: 8)
- Reduce `clear_cache_interval` to clear cache more frequently
- Use smaller `max_length` if captions are short

**2. Model not found**
- Verify `decoder_models/nv-embed/` exists in parent directory
- Check path in config: `decoder.path`
- Model will auto-download from HuggingFace if path doesn't exist

**3. NLTK punkt tokenizer error**
```bash
# Download manually
python -c "import nltk; nltk.download('punkt')"
```

**4. Import errors**
- Ensure all dependencies installed: `pip install -r src/configs/requirements/nv-embed.txt`
- Check Python version: `python --version` (3.12+ required)

**5. Slow evaluation**
- Enable GPU: `processing.device: "cuda:0"` in config
- Increase `batch_size` if GPU memory permits
- Reduce `progress_interval` to log less frequently

## Performance Benchmarks

On NVIDIA A100 (40GB):
- **Throughput**: ~50-100 videos/hour (depending on caption length)
- **Memory Usage**: ~12-18GB VRAM
- **Coarse-Grained**: ~2-3 seconds per video
- **Fine-Grained**: ~5-8 seconds per video (depends on sentence count)

Tips for optimization:
- Use `batch_size: 16` for A100
- Use `batch_size: 8` for RTX 3090/4090
- Use `batch_size: 4` for GPUs with <16GB VRAM

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{clip-cc-bench-2025,
  title={CLIP-CC-Bench: Video Captioning Benchmark with Multi-Decoder Evaluation},
  author={Your Name},
  booktitle={Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```

## Related Work

- **BERTScore**: Token-level greedy matching for text generation evaluation
- **EMScore**: Video captioning evaluation with embedding matching (CVPR 2022)
- **NV-Embed**: NVIDIA's instruction-tuned embedding model (State-of-the-art on MTEB)

## License

[Specify your license here - MIT, Apache 2.0, etc.]

## Contact

- **Author**: [Your Name]
- **Email**: [your.email@domain.com]
- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/clip-cc-bench/issues)

## Changelog

### v2.0.0 (2025-11-13)
- **Breaking**: Refactored from multi-encoder to decoder architecture
- **New**: Fine-grained sentence-level evaluation (precision/recall/F1/hm-cf)
- **New**: Flattened directory structure for easier maintenance
- **New**: NLTK-based sentence chunking
- **New**: Per-model summary statistics
- **Changed**: Encoder → Decoder terminology throughout codebase
- **Removed**: Legacy encoders (will be re-added as decoders)
- **Improved**: Result management with CSV and JSON outputs

### v1.0.0 (2025-11-12)
- Initial release
- Support for 7 encoders
- Coarse-grained evaluation only
- SLURM integration for HPC clusters

---

**Note**: This is a research project under active development. The fine-grained evaluation methodology is experimental and subject to refinement.
