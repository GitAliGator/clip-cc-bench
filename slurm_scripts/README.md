# G-VEval SLURM Batch Evaluation System

Concurrent evaluation of 15 video captioning models using LLaMA-3.1-70B judge on SLURM cluster.

**Note:** Qwen2.5-72B and Qwen2.5-32B already evaluated - excluded from batches.

## Quick Start

```bash
# Step 1: Launch first batch (9 models)
cd /mmfs2/home/jacks.local/mali9292/aaai_student_abstract/clip-cc-bench
bash slurm_scripts/launch_batch1.sh

# Step 2: Monitor progress
bash slurm_scripts/monitor_all.sh

# Step 3: When batch 1 completes (~4 hours), launch batch 2 (8 models)
bash slurm_scripts/launch_batch2.sh

# Step 4: Monitor until all complete
watch -n 30 'bash slurm_scripts/monitor_all.sh'
```

## System Overview

### Architecture
- **Judge Model:** LLaMA-3.1-70B-Instruct (2x H100 GPUs per job)
- **Evaluation Method:** G-VEval ACCR rubric
- **Processing Rate:** ~30 sec/sample, ~1.7 hours/model
- **Batching Strategy:** 1 model per job for maximum parallelism

### Batches

**Batch 1 (8 models):**
1. internvl
2. llava_next_video
3. llava_one_vision
4. longva
5. longvu
6. minicpm
7. mplug
8. oryx

**Batch 2 (7 models):**
9. sharegpt4
10. timechat
11. ts_llava
12. videochatflash
13. videollama3
14. video_xl
15. vilamp

**Already Completed:**
- Qwen2.5-32B ✓
- Qwen2.5-72B ✓

### Resource Requirements per Job
- **GPUs:** 2x H100 (80GB each)
- **CPUs:** 16 cores
- **Memory:** 100GB
- **Time:** 4 hours (max)
- **Partition:** `all-gpu`
- **Account:** `eecs`

## Files

### Scripts
- `gveval_single.slurm` - SLURM template for single model evaluation
- `launch_batch1.sh` - Submit first 9 models
- `launch_batch2.sh` - Submit remaining 8 models
- `monitor_all.sh` - Display progress across all jobs

### Modified Python Script
- `../src/scripts/run_gveval_llama_evaluation.py` - Now accepts `--models` CLI argument

## Usage Details

### Launch Individual Batch

```bash
# Launch batch 1 (8 models, 16 GPUs)
bash slurm_scripts/launch_batch1.sh

# Launch batch 2 (7 models, 14 GPUs)
bash slurm_scripts/launch_batch2.sh
```

Each launcher:
1. Shows model list for confirmation
2. Prompts for y/n approval
3. Creates temporary SLURM scripts from template
4. Submits jobs with `sbatch`
5. Displays job IDs
6. Provides monitoring commands

### Monitor Progress

```bash
# One-time check
bash slurm_scripts/monitor_all.sh

# Auto-refresh every 30 seconds
watch -n 30 'bash slurm_scripts/monitor_all.sh'

# Check SLURM queue
squeue -u $USER | grep gveval

# View specific job output (replace JOB_ID)
tail -f slurm_logs/gveval_*_JOB_ID.out
```

### Manual Job Submission

```bash
# Submit single model manually
MODEL="internvl"
sed "s/MODEL_NAME/${MODEL}/g" slurm_scripts/gveval_single.slurm | sbatch

# Or use Python directly
python src/scripts/run_gveval_llama_evaluation.py --models internvl
```

### Cancel Jobs

```bash
# Cancel all your gveval jobs
scancel -u $USER --name=gveval_*

# Cancel specific job
scancel JOB_ID
```

## Results

### Location
```
results/g-veval/
├── individual_results/
│   ├── json/              # Per-sample scores (model_name.jsonl)
│   └── csv/               # Per-sample scores (model_name.csv)
├── aggregated_results/    # Summary statistics (model_name_summary.json)
└── logs/                  # Python evaluation logs
```

### SLURM Logs
```
slurm_logs/
├── gveval_gveval_internvl_12345.out    # stdout
└── gveval_gveval_internvl_12345.err    # stderr
```

## Timeline

### Sequential (baseline)
- 15 models × 1.7 hours = **~25.5 hours**
- (Plus 2 already completed: Qwen2.5-72B, Qwen2.5-32B)

### Batch Strategy (implemented)
- **Batch 1:** 8 jobs × 1.7 hours = **~1.7 hours** (wall time)
- **Batch 2:** 7 jobs × 1.7 hours = **~1.7 hours** (wall time)
- **Total:** **~3.4 hours** (wall time, not including queue wait)
- **Speedup:** 7.5x faster

### Full Parallelism (if 30 GPUs available)
- All 15 jobs concurrent = **~1.7 hours** (wall time)

## Troubleshooting

### Job Fails Immediately
```bash
# Check error log
tail slurm_logs/gveval_*_JOB_ID.err

# Common issues:
# - Model prediction file missing: Check data/models/MODEL_NAME.json exists
# - GPU out of memory: Should not happen with 2x H100
# - CUDA error: Check nvidia-smi on allocated node
```

### Job Stuck/Slow
```bash
# Check job status
scontrol show job JOB_ID

# SSH to compute node and check GPU usage
ssh NODE_NAME
nvidia-smi

# Check progress from log
grep "Processed" results/g-veval/logs/gveval_llama_evaluation_*.log | tail
```

### Results Missing
```bash
# Check if job completed successfully
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode

# Check SLURM output for errors
cat slurm_logs/gveval_*_JOB_ID.out
```

## Advanced Usage

### Evaluate Custom Model List
```bash
# Python directly (no SLURM)
python src/scripts/run_gveval_llama_evaluation.py --models model1 model2

# Submit custom SLURM job
MODEL="custom_model"
sed "s/MODEL_NAME/${MODEL}/g" slurm_scripts/gveval_single.slurm | sbatch
```

### Modify Resource Allocation
Edit `gveval_single.slurm`:
```bash
#SBATCH --gres=gpu:2        # Change GPU count
#SBATCH --mem=100G          # Change memory
#SBATCH --time=04:00:00     # Change time limit
```

### Re-evaluate Failed Models
```bash
# Check which models completed
ls results/g-veval/aggregated_results/*_summary.json | xargs -n1 basename | sed 's/_summary.json//'

# Find missing models
comm -23 <(ls data/models/*.json | xargs -n1 basename | sed 's/.json//' | sort) \
         <(ls results/g-veval/aggregated_results/*_summary.json | xargs -n1 basename | sed 's/_summary.json//' | sort)

# Re-submit only failed models
python src/scripts/run_gveval_llama_evaluation.py --models FAILED_MODEL
```

## Performance Metrics

Based on actual cluster runs:
- **Total models:** 15 remaining (17 total, 2 completed)
- **Inference rate:** 30.4 sec/sample
- **Samples per model:** ~199
- **Time per model:** 1.68 hours average
- **Safety margin:** 4-hour limit allows 2.4x buffer
- **Success rate:** Expect >95% samples evaluated per model

## Notes

- Jobs skip pre-flight checks (`--skip-checks`) to speed up startup
- Model loading takes ~2 minutes (included in 4-hour limit)
- Results are saved incrementally (checkpoint every 10 samples)
- GPU memory is cleared between models in same job
- SLURM logs persist even after job completion
