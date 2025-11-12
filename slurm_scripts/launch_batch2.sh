#!/bin/bash
# G-VEval Batch 2 Launcher - Remaining 7 Models
# Submits 7 concurrent SLURM jobs, each evaluating 1 model

set -e

echo "=================================================="
echo "G-VEval Batch 2 Launcher"
echo "=================================================="
echo "This will submit 7 SLURM jobs for the second batch of models"
echo "Each job will use 2x H100 GPUs and run for up to 4 hours"
echo ""
echo "NOTE: Qwen2.5-32B and Qwen2.5-72B already evaluated - excluded"
echo ""

# Set working directory
cd "$(dirname "$0")/.."
PROJ_DIR=$(pwd)

echo "Project directory: $PROJ_DIR"
echo ""

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

# Define batch 2 models (7 models - Qwen2.5-72B already done)
BATCH2_MODELS=(
    "sharegpt4"
    "timechat"
    "ts_llava"
    "videochatflash"
    "videollama3"
    "video_xl"
    "vilamp"
)

echo "Batch 2 Models (7 total):"
for model in "${BATCH2_MODELS[@]}"; do
    echo "  - $model"
done
echo ""

# Confirmation prompt
read -p "Submit ${#BATCH2_MODELS[@]} jobs to SLURM? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 1
fi

echo ""
echo "Submitting jobs..."
echo ""

# Submit jobs
JOB_IDS=()
for model in "${BATCH2_MODELS[@]}"; do
    # Create model-specific SLURM script from template
    SLURM_SCRIPT="slurm_scripts/gveval_${model}.slurm"

    # Replace MODEL_NAME placeholder in template
    sed "s/MODEL_NAME/${model}/g" slurm_scripts/gveval_single.slurm > "$SLURM_SCRIPT"

    # Submit job
    JOB_OUTPUT=$(sbatch "$SLURM_SCRIPT")
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
    JOB_IDS+=($JOB_ID)

    echo "✓ Submitted $model (Job ID: $JOB_ID)"

    # Clean up temporary script
    rm "$SLURM_SCRIPT"

    # Small delay to avoid overwhelming scheduler
    sleep 0.5
done

echo ""
echo "=================================================="
echo "Batch 2 Submission Complete!"
echo "=================================================="
echo "Submitted ${#JOB_IDS[@]} jobs:"
for i in "${!JOB_IDS[@]}"; do
    echo "  Job ${JOB_IDS[$i]}: ${BATCH2_MODELS[$i]}"
done
echo ""
echo "Monitor jobs with:"
echo "  watch -n 30 'squeue -u \$USER | grep gveval'"
echo ""
echo "Check individual logs in:"
echo "  slurm_logs/gveval_*.out"
echo ""
echo "Or use the monitor script:"
echo "  bash slurm_scripts/monitor_all.sh"
echo "=================================================="
