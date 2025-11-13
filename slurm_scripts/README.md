# SLURM Scripts

This directory is for HPC cluster job submission scripts.

## Purpose

Store SLURM batch scripts (.slurm, .sbatch) and launcher scripts (.sh) for running evaluations on HPC clusters with SLURM workload manager.

## Directory Structure

```
slurm_scripts/
├── README.md           # This file
└── (Add your .slurm and .sh scripts here)
```

## Usage Example

### Creating a SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=my_evaluation
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=all-gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --account=eecs

# Activate virtual environment
source /mmfs2/home/jacks.local/mali9292/aaai_student_abstract/venv_<encoder>/bin/activate

# Run your evaluation script
python src/scripts/run_<encoder>_evaluation.py
```

### Submitting Jobs

```bash
# Submit single job
sbatch slurm_scripts/my_evaluation.slurm

# Check job status
squeue -u $USER

# View job output
tail -f slurm_logs/my_evaluation_<job_id>.out

# Cancel job
scancel <job_id>
```

## SLURM Logs

Output and error logs are stored in `../slurm_logs/`:

```
slurm_logs/
├── job_name_12345.out    # stdout
└── job_name_12345.err    # stderr
```

Make sure the `slurm_logs/` directory exists before submitting jobs.

## Common SLURM Commands

```bash
# View job details
scontrol show job <job_id>

# View job accounting
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed

# List available partitions
sinfo

# Check node resources
scontrol show partition <partition_name>

# Cancel all your jobs
scancel -u $USER

# View queue
squeue -u $USER
```

## Best Practices

1. **Test locally first** before submitting to SLURM
2. **Use appropriate time limits** to avoid unnecessary queue time
3. **Request only needed resources** (GPUs, CPUs, memory)
4. **Create logs directory** before job submission
5. **Use job arrays** for running multiple similar jobs
6. **Set up email notifications** for long-running jobs

## Resources

- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [SLURM Quick Start Guide](https://slurm.schedmd.com/quickstart.html)
- [SLURM Cheat Sheet](https://slurm.schedmd.com/pdfs/summary.pdf)
