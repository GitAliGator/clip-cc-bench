# CLIP-CC-Bench Setup Guide

Detailed setup instructions for CLIP-CC-Bench evaluation framework.

## Directory Structure Overview

CLIP-CC-Bench uses a **centralized structure** where large files (models, virtual environments) are stored outside the main repository:

```
your_workspace/
├── clip-cc-bench/              # This Git repository
│   ├── src/                    # Source code
│   ├── data/                   # Dataset structure (data files gitignored)
│   ├── slurm_scripts/          # HPC job scripts
│   └── README.md
├── encoder_models/             # 119GB - Encoder model weights
│   ├── bge-en-icl/
│   ├── e5-mistral-7b-instruct/
│   ├── jina-embeddings-v4/
│   ├── ...
├── venv_configs/               # Virtual environment setup
│   ├── setup_*.sh              # Setup scripts for each encoder
│   ├── activate_*.sh           # Activation scripts
│   ├── *-requirements.txt      # Frozen requirements
│   └── README.md
└── venv_*/                     # Isolated venvs (~57GB total)
    ├── venv_bge-icl/
    ├── venv_jina-v4/
    └── ...
```

## Step-by-Step Setup

### 1. Clone the Repository

```bash
# Create workspace directory
mkdir -p ~/clip-cc-workspace
cd ~/clip-cc-workspace

# Clone repository
git clone https://github.com/YOUR_USERNAME/clip-cc-bench.git
cd clip-cc-bench
```

### 2. Download Encoder Models

You have two options:

#### Option A: Download from HuggingFace

```bash
cd ..  # Go to workspace root
mkdir -p encoder_models

# Download each model
cd encoder_models

# Example: Download BGE-EN-ICL
git clone https://huggingface.co/BAAI/bge-en-icl bge-en-icl

# Example: Download Jina-v4
git clone https://huggingface.co/jinaai/jina-embeddings-v4 jina-embeddings-v4

# Continue for other models...
```

#### Option B: Use Download Scripts

```bash
# Some models have download scripts in the repo
cd clip-cc-bench

# Activate a basic Python environment
python3 -m venv temp_venv
source temp_venv/bin/activate
pip install huggingface-hub

# Run download scripts
python src/config/jina-v4/download_jina_model.py
python src/scripts/nomic-embed-text-v1.5/download_model.py

deactivate
rm -rf temp_venv
```

#### Option C: Copy from Existing Location

If you already have the models:
```bash
cd ..  # Go to workspace root
cp -r /path/to/existing/encoder_models .
```

### 3. Set Up Virtual Environments

The `venv_configs/` directory should be in your workspace root (created during migration):

```bash
cd ~/clip-cc-workspace/venv_configs

# Setup all environments (recommended - takes 1-3 hours)
./setup_all_environments.sh

# Or setup individual environments
./setup_bge-icl_env.sh
./setup_jina-v4_env.sh
# ... etc
```

**What this does:**
- Creates isolated Python environments for each encoder
- Installs exact package versions from frozen requirements
- Verifies installations and dependencies
- Places venvs in workspace root (outside git repo)

### 4. Prepare Data

#### Ground Truth Data

Place your ground truth captions in `data/ground_truth/clip_cc_dataset.json`:

```json
{
  "video_id_1": {
    "summary": "A person walking in the park..."
  },
  "video_id_2": {
    "summary": "A car driving on the highway..."
  }
}
```

#### Model Predictions

Place model predictions in `data/models/{model_name}.json`:

```bash
cd clip-cc-bench/data/models

# Each model should have its own JSON file
ls
# Expected:
# internvl.json
# llava_next_video.json
# minicpm.json
# ... etc
```

Format for each file:
```json
{
  "video_id_1": "Predicted caption for video 1...",
  "video_id_2": "Predicted caption for video 2..."
}
```

### 5. Configure Encoders

Each encoder has a config file in `src/config/{encoder}/encoders_config.yaml`.

Example (`src/config/bge-icl/encoders_config.yaml`):

```yaml
encoder:
  name: "bge-en-icl"
  path: "encoder_models/bge-en-icl"  # Relative to workspace root
  type: "bge_en_icl"
  batch_size: 8
  max_length: 8192
  device_map: "auto"
  trust_remote_code: true

processing:
  clear_cache_interval: 25
  progress_interval: 10

data_paths:
  ground_truth_file: "data/ground_truth/clip_cc_dataset.json"
  predictions_dir: "data/models"
  results_base_dir: "results"

models_to_evaluate:
  - "internvl"
  - "llava_next_video"
  - "minicpm"
  # ... add your models

logging:
  level: "INFO"
  log_dir: "results/encoders/logs/bge-icl"
  log_prefix: "bge_icl_evaluation"
```

**Key paths** (all relative to workspace/clip-cc-bench):
- `encoder.path`: Points to encoder model directory
- `data_paths.*`: Point to data and results directories

### 6. Verify Setup

```bash
cd ~/clip-cc-workspace/clip-cc-bench

# Check encoder models
ls -lh ../encoder_models/

# Check virtual environments
ls -lh ../venv_*/

# Test encoder activation
source ../venv_configs/activate_bge-icl_env.sh
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
deactivate
```

## HPC/SLURM Setup

If using an HPC cluster:

### 1. Update SLURM Scripts

Edit scripts in `slurm_scripts/` to match your cluster:

```bash
#!/bin/bash
#SBATCH --job-name=bge-icl-eval
#SBATCH --partition=gpu          # Your GPU partition
#SBATCH --gres=gpu:1             # GPU allocation
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/bge-icl-%j.out

# Load modules (if needed)
module load cuda/11.8
module load python/3.12

# Activate environment
source ../venv_configs/activate_bge-icl_env.sh

# Run evaluation
python src/scripts/bge-icl/run_bge-icl_evaluation.py \
    --config src/config/bge-icl/encoders_config.yaml

deactivate
```

### 2. Submit Jobs

```bash
cd clip-cc-bench/slurm_scripts

# Submit single job
sbatch run_bge_icl_evaluation.sh

# Submit all encoders
for script in run_*_evaluation.sh; do
    sbatch $script
done

# Monitor
squeue -u $USER
```

## Environment Variables (Optional)

Set base directory explicitly:

```bash
export CLIP_CC_BASE_DIR="/path/to/workspace/clip-cc-bench"
```

This overrides the default path resolution.

## Testing Your Setup

### Quick Test

```bash
# Activate environment
source ../venv_configs/activate_bge-icl_env.sh

# Test import
python -c "
import sys
sys.path.insert(0, 'src/utils/shared')
from paths import get_project_paths
paths = get_project_paths()
print(f'Base: {paths.get_base_dir()}')
print(f'Encoders: {paths.get_encoder_models_dir()}')
print(f'Models exist: {paths.get_encoder_models_dir().exists()}')
"

deactivate
```

### Run Small Test

```bash
source ../venv_configs/activate_bge-icl_env.sh

python src/scripts/bge-icl/run_bge-icl_evaluation.py \
    --config src/config/bge-icl/encoders_config.yaml \
    --max-samples 10  # Test with 10 samples only

deactivate
```

## Troubleshooting

### Issue: Models not found

```bash
# Check encoder_models location
ls -lh ../encoder_models/

# Verify symlink (if used)
ls -l encoder_models

# Check config paths
cat src/config/bge-icl/encoders_config.yaml | grep path
```

### Issue: Import errors

```bash
# Verify environment
which python
echo $VIRTUAL_ENV

# Check installed packages
pip list | grep -i transformers

# Reinstall if needed
cd ../venv_configs
./setup_bge-icl_env.sh  # Recreate environment
```

### Issue: CUDA not available

```bash
# Check CUDA
nvidia-smi

# Test in Python
python -c "import torch; print(torch.cuda.is_available())"

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

### Issue: Out of memory

1. Reduce batch size in config
2. Use smaller context length
3. Enable gradient checkpointing (if supported)
4. Request more GPU memory in SLURM

## Next Steps

After setup:
1. Review encoder configurations
2. Run test evaluation on small dataset
3. Monitor GPU usage and memory
4. Scale up to full dataset
5. Aggregate results across encoders

See [README.md](README.md) for usage examples and [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
