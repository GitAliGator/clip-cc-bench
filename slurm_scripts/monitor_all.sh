#!/bin/bash
# G-VEval Progress Monitor
# Displays status of all G-VEval jobs and their progress

set -e

# Set working directory
cd "$(dirname "$0")/.."
PROJ_DIR=$(pwd)

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear
echo "=================================================="
echo "G-VEval LLaMA Evaluation - Progress Monitor"
echo "=================================================="
echo "Project: $PROJ_DIR"
echo "Time: $(date)"
echo ""

# Check SLURM queue
echo -e "${BLUE}SLURM Queue Status:${NC}"
echo "=================================================="
squeue -u $USER -o "%.10i %.12P %.18j %.8T %.10M %.6D %R" | grep -E "JOBID|gveval" || echo "No G-VEval jobs in queue"
echo ""

# Count job states
RUNNING=$(squeue -u $USER -h -o "%T" | grep -c "RUNNING" || echo "0")
PENDING=$(squeue -u $USER -h -o "%T" | grep -c "PENDING" || echo "0")
TOTAL_QUEUED=$((RUNNING + PENDING))

echo -e "${BLUE}Summary:${NC}"
echo "  Running: $RUNNING"
echo "  Pending: $PENDING"
echo "  Total in queue: $TOTAL_QUEUED"
echo ""

# Check completed results
echo -e "${BLUE}Completed Evaluations:${NC}"
echo "=================================================="

RESULTS_DIR="$PROJ_DIR/results/g-veval/aggregated_results"
if [ -d "$RESULTS_DIR" ]; then
    COMPLETED_COUNT=$(find "$RESULTS_DIR" -name "*_summary.json" 2>/dev/null | wc -l)
    # Subtract already completed models if they exist
    ALREADY_DONE_COUNT=0
    [ -f "$RESULTS_DIR/Qwen2.5-72B_summary.json" ] && ALREADY_DONE_COUNT=$((ALREADY_DONE_COUNT + 1))
    [ -f "$RESULTS_DIR/Qwen2.5-32B_summary.json" ] && ALREADY_DONE_COUNT=$((ALREADY_DONE_COUNT + 1))
    BATCH_COMPLETED=$((COMPLETED_COUNT - ALREADY_DONE_COUNT))

    echo "Total completed: $COMPLETED_COUNT/17 ($ALREADY_DONE_COUNT pre-existing, $BATCH_COMPLETED from batches)"
    echo ""

    if [ $COMPLETED_COUNT -gt 0 ]; then
        echo "Completed models:"
        for summary in "$RESULTS_DIR"/*_summary.json; do
            if [ -f "$summary" ]; then
                model=$(basename "$summary" _summary.json)
                # Extract average score if possible
                avg_score=$(python3 -c "import json; data=json.load(open('$summary')); print(f\"{data.get('average_gveval_score', 0):.2f}\")" 2>/dev/null || echo "N/A")
                success_rate=$(python3 -c "import json; data=json.load(open('$summary')); print(f\"{data.get('success_rate', 0)*100:.1f}\")" 2>/dev/null || echo "N/A")
                echo -e "  ${GREEN}✓${NC} $model (Score: $avg_score, Success: $success_rate%)"
            fi
        done
    fi
else
    echo "No results directory found yet"
fi
echo ""

# Check recent log activity
echo -e "${BLUE}Recent Log Activity:${NC}"
echo "=================================================="
LOGS_DIR="$PROJ_DIR/results/g-veval/logs"
if [ -d "$LOGS_DIR" ]; then
    RECENT_LOGS=$(find "$LOGS_DIR" -name "*.log" -mmin -60 2>/dev/null | wc -l)
    echo "Active logs (modified in last hour): $RECENT_LOGS"
    echo ""

    # Show latest progress from recent logs
    echo "Latest progress updates:"
    for log in $(find "$LOGS_DIR" -name "*.log" -mmin -10 2>/dev/null | head -5); do
        LATEST_PROGRESS=$(grep "Processed.*samples" "$log" 2>/dev/null | tail -1)
        if [ ! -z "$LATEST_PROGRESS" ]; then
            LOG_NAME=$(basename "$log")
            echo "  $LOG_NAME: $LATEST_PROGRESS"
        fi
    done
else
    echo "No logs directory found"
fi
echo ""

# Check SLURM output logs
echo -e "${BLUE}Latest SLURM Outputs:${NC}"
echo "=================================================="
SLURM_LOGS_DIR="$PROJ_DIR/slurm_logs"
if [ -d "$SLURM_LOGS_DIR" ]; then
    SLURM_COUNT=$(find "$SLURM_LOGS_DIR" -name "gveval_*.out" 2>/dev/null | wc -l)
    echo "Total SLURM log files: $SLURM_COUNT"

    # Show most recent SLURM logs
    RECENT_SLURM=$(find "$SLURM_LOGS_DIR" -name "gveval_*.out" -mmin -60 2>/dev/null | sort -t_ -k2 -n | tail -5)
    if [ ! -z "$RECENT_SLURM" ]; then
        echo ""
        echo "Recent SLURM jobs (last 5):"
        for slurm_log in $RECENT_SLURM; do
            JOB_NAME=$(basename "$slurm_log" .out | sed 's/gveval_//')
            LAST_LINE=$(tail -1 "$slurm_log" 2>/dev/null)
            echo "  $JOB_NAME: $LAST_LINE"
        done
    fi
else
    echo "No SLURM logs directory found"
fi
echo ""

# Overall progress
echo "=================================================="
echo -e "${BLUE}Overall Progress:${NC}"
TOTAL_MODELS=17
ALREADY_DONE_STATIC=2  # Qwen2.5-72B, Qwen2.5-32B
REMAINING_MODELS=$((TOTAL_MODELS - ALREADY_DONE_STATIC))
# Use the counts from earlier
COMPLETED_TOTAL=${COMPLETED_COUNT:-0}
BATCH_DONE=${BATCH_COMPLETED:-0}
PROGRESS_PCT=$((COMPLETED_TOTAL * 100 / TOTAL_MODELS))
echo "  Completed: $COMPLETED_TOTAL / $TOTAL_MODELS models ($PROGRESS_PCT%)"
echo "    - Already done: $ALREADY_DONE_STATIC (Qwen2.5-72B, Qwen2.5-32B)"
echo "    - From batches: $BATCH_DONE / $REMAINING_MODELS"
echo "  Running/Pending: $TOTAL_QUEUED jobs"
echo "  Remaining: $((REMAINING_MODELS - BATCH_DONE)) models"
echo "=================================================="

# Recommendations
echo ""
if [ $TOTAL_QUEUED -eq 0 ] && [ $BATCH_DONE -lt $REMAINING_MODELS ]; then
    echo -e "${YELLOW}⚠️  No jobs running. Consider launching next batch:${NC}"
    echo "  bash slurm_scripts/launch_batch2.sh"
elif [ $BATCH_DONE -eq $REMAINING_MODELS ]; then
    echo -e "${GREEN}🎉 All batch evaluations complete!${NC}"
fi

echo ""
echo "Refresh this view with: bash slurm_scripts/monitor_all.sh"
echo "Auto-refresh with: watch -n 30 'bash slurm_scripts/monitor_all.sh'"
