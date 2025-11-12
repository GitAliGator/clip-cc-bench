# CLIP-CC-Bench: Video Captioning Benchmark with Encoder Evaluation

A comprehensive benchmark for evaluating video language models using multiple text embedding encoders to measure caption quality.

## Overview

CLIP-CC-Bench is an evaluation framework that:
- Assesses video captioning models using state-of-the-art text encoders
- Compares ground truth captions with model predictions via embedding similarity
- Supports multiple encoder architectures (BGE-ICL, E5-Mistral, Jina-v4, etc.)
- Provides isolated environments for reproducible evaluation
- Includes SLURM scripts for HPC cluster deployment

## Project Structure

```
clip-cc-bench/
├── src/
│   ├── config/          # Encoder-specific configurations
│   ├── scripts/         # Evaluation scripts for each encoder
│   └── utils/           # Shared utilities and encoder modules
├── data/
│   ├── ground_truth/    # Reference captions
│   └── models/          # Model predictions
├── results/             # Evaluation outputs
├── slurm_scripts/       # HPC job submission scripts
└── WACV/                # Conference paper materials
```

## Supported Encoders

| Encoder | Embedding Dim | Context Length | Type |
|---------|---------------|----------------|------|
| BGE-EN-ICL | 1024 | 8192 | In-Context Learning |
| E5-Mistral-7B | Variable | 32768 | Instruction-tuned |
| Jina-v4 | 2048 | 8192 | Long context |
| Envision | Variable | 512 | Vision-language |
| NV-Embed | Variable | 4096 | NVIDIA optimized |
| Nomic-Embed-v1.5 | 768 | 8192 | Efficient |
| Stella-EN-1.5B | Variable | 512 | Lightweight |

## Setup

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)
- 150GB+ disk space for models
- HPC cluster access (optional, for SLURM)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/clip-cc-bench.git
   cd clip-cc-bench
   ```

2. **Set up centralized directories**

   The project uses centralized storage for large files:
   ```bash
   # Project expects this structure:
   parent_dir/
   ├── clip-cc-bench/           # This repo
   ├── encoder_models/          # 119GB of model weights
   ├── venv_configs/            # Virtual environment setup
   └── venv_*/                  # Isolated environments per encoder
   ```

3. **Download encoder models**

   Models should be placed in `../encoder_models/` relative to the repo:
   ```bash
   mkdir -p ../encoder_models
   # Download models from HuggingFace or use provided download scripts
   ```

4. **Create virtual environments**

   Each encoder has its own isolated environment:
   ```bash
   cd ../venv_configs

   # Setup all environments (1-3 hours)
   ./setup_all_environments.sh

   # Or setup individual encoder
   ./setup_bge-icl_env.sh
   ```

### Quick Start

1. **Prepare your data**
   ```bash
   # Place ground truth in data/ground_truth/clip_cc_dataset.json
   # Place model predictions in data/models/{model_name}.json
   ```

2. **Run evaluation for a specific encoder**
   ```bash
   # Activate encoder environment
   source ../venv_configs/activate_bge-icl_env.sh

   # Run evaluation
   python src/scripts/bge-icl/run_bge-icl_evaluation.py \
       --config src/config/bge-icl/encoders_config.yaml

   deactivate
   ```

3. **Run on SLURM cluster**
   ```bash
   # Submit job
   sbatch slurm_scripts/run_bge_icl_evaluation.sh

   # Check logs
   tail -f slurm_logs/bge-icl-*.out
   ```

## Configuration

Each encoder has a YAML configuration file in `src/config/{encoder_name}/encoders_config.yaml`:

```yaml
encoder:
  name: "encoder-name"
  path: "encoder_models/encoder-dir"  # Relative to parent
  type: "encoder_type"
  batch_size: 16
  max_length: 4096
  device_map: "auto"
  trust_remote_code: true

processing:
  clear_cache_interval: 25
  progress_interval: 10

data_paths:
  ground_truth_file: "data/ground_truth/clip_cc_dataset.json"
  predictions_dir: "data/models"
  results_base_dir: "results"
```

## Usage

### Running Individual Encoder Evaluation

```bash
# 1. Activate environment
source ../venv_configs/activate_{encoder}_env.sh

# 2. Run evaluation
python src/scripts/{encoder}/run_{encoder}_evaluation.py \
    --config src/config/{encoder}/encoders_config.yaml \
    --models model1 model2 model3

# 3. View results
cat results/encoders/individual_results/json/{encoder}_{model}_results.json
```

### Running Multiple Encoders

```bash
# Use provided batch script
./run_all_encoders.sh
```

### SLURM Cluster Deployment

```bash
# Submit all encoder evaluations
cd slurm_scripts
for script in run_*_evaluation.sh; do
    sbatch $script
done

# Monitor jobs
squeue -u $USER
```

## Results

Results are organized by encoder and model:

```
results/
├── encoders/
│   ├── individual_results/
│   │   ├── csv/          # Per-model CSV files
│   │   └── json/         # Per-model JSON files
│   ├── aggregated_results/
│   │   └── summary.csv   # Cross-encoder comparison
│   └── logs/
│       └── {encoder}/    # Evaluation logs
```

## Data Format

### Ground Truth
```json
{
  "video_id_1": {
    "summary": "Ground truth caption text..."
  },
  "video_id_2": {
    "summary": "Another caption..."
  }
}
```

### Model Predictions
```json
{
  "video_id_1": "Model predicted caption...",
  "video_id_2": "Another prediction..."
}
```

## Development

### Adding a New Encoder

1. **Create configuration**
   ```bash
   mkdir -p src/config/new-encoder
   cp src/config/bge-icl/encoders_config.yaml src/config/new-encoder/
   # Edit configuration
   ```

2. **Create encoder module**
   ```bash
   mkdir -p src/utils/new-encoder
   # Implement embedding_models.py
   ```

3. **Create evaluation script**
   ```bash
   mkdir -p src/scripts/new-encoder
   # Implement run_new-encoder_evaluation.py
   ```

4. **Add virtual environment setup**
   ```bash
   # Create requirements.txt in src/config/new-encoder/
   # Create setup script in ../venv_configs/
   ```

### Code Structure

- **`src/utils/shared/`**: Common utilities (paths, config loading, result management)
- **`src/utils/{encoder}/`**: Encoder-specific implementations
- **`src/config/{encoder}/`**: Encoder configurations and requirements
- **`src/scripts/{encoder}/`**: Standalone evaluation scripts

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{clip-cc-bench-2025,
  title={CLIP-CC-Bench: Video Captioning Benchmark with Multi-Encoder Evaluation},
  author={Your Name},
  booktitle={Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```

## License

[Specify your license here - MIT, Apache 2.0, etc.]

## Acknowledgments

- HuggingFace for model hosting
- NVIDIA for GPU support
- [Other acknowledgments]

## Contact

- **Author**: [Your Name]
- **Email**: [your.email@domain.com]
- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/clip-cc-bench/issues)

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce `batch_size` in encoder config
- Use `device_map: "auto"` for automatic device allocation
- Clear GPU cache: `torch.cuda.empty_cache()`

**2. Encoder model not found**
- Verify encoder_models/ directory structure
- Check symlink: `ls -l encoder_models`
- Download missing models

**3. Import errors**
- Ensure correct virtual environment activated
- Run setup script: `./setup_{encoder}_env.sh`
- Check `pip list` for missing dependencies

**4. SLURM job failures**
- Check logs: `slurm_logs/*.out`
- Verify GPU allocation in SLURM script
- Ensure sufficient memory requested

## Changelog

### v1.0.0 (2025-11-12)
- Initial release
- Support for 7 encoders
- Centralized model and environment management
- SLURM integration for HPC clusters
- Comprehensive evaluation pipeline

---

**Note**: This is a research project under active development. Features and APIs may change.
