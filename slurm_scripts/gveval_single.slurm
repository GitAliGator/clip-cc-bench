#!/bin/bash
#SBATCH --job-name=gveval_MODEL_NAME
#SBATCH --partition=all-gpu
#SBATCH --account=eecs
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/gveval_%x_%j.out
#SBATCH --error=slurm_logs/gveval_%x_%j.err

# G-VEval LLaMA Evaluation - Single Model SLURM Script
# This script evaluates one model using the LLaMA-3.1-70B judge
# Generated for concurrent batch evaluation

echo "=================================================="
echo "G-VEval LLaMA Evaluation - SLURM Job"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Model: MODEL_NAME"
echo "Start time: $(date)"
echo "=================================================="

# Set working directory
cd /mmfs2/home/jacks.local/mali9292/aaai_student_abstract/clip-cc-bench

# Environment setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_CACHE=/home/jacks.local/mali9292/.cache/huggingface
export HF_HOME=/home/jacks.local/mali9292/.cache/huggingface

# Activate virtual environment
echo "Activating virtual environment..."
source /mmfs2/home/jacks.local/mali9292/aaai_student_abstract/venv/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Show GPU allocation
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
echo ""

# Run evaluation for specified model
echo "Starting evaluation for model: MODEL_NAME"
echo "Command: python src/scripts/run_gveval_llama_evaluation.py --skip-checks --models MODEL_NAME"
echo ""

python src/scripts/run_gveval_llama_evaluation.py --skip-checks --models MODEL_NAME

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
