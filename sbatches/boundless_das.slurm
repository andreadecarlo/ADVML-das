#!/bin/bash
#SBATCH --job-name=boundless_das
#SBATCH --partition=edu-short 
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
##SBATCH --time=24:00:00
#SBATCH --output=.out/boundless_das_%j.out

# Boundless DAS (multiplication counterfactual): prepare then train. Qwen2-7B.
# Plan: (1) prepare_boundless_das_dataset → train/val/test with intervention positions from data
#       (2) train_boundless_das → intervene one token at a time, stream IIA, test IIA at end
# Usage (from project root):
#   sbatch sbatches/boundless_das.slurm
#   MAX_SAMPLES=5000 sbatch sbatches/boundless_das.slurm
#   MAX_SAMPLES=2000 sbatch sbatches/boundless_das.slurm   # default: 10k

set -e
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p .out datasets/boundless_das outputs/boundless_das
# SLURM writes to .out; create it so the job doesn't fail if sbatch doesn't

# Optional: load CUDA if required on your cluster
# module load CUDA/12.5.0

# 1) Prepare dataset (no GPU); --max-samples limits examples (pass via sbatch ... --max-samples N)
#    To include write_down counterfactuals, run generate_multiplication_dataset.py first, then add:
#    --counterfactual-dataset-write-down datasets/multiplication_write_down_counterfactual.json
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
echo "=== Preparing Boundless DAS dataset (max_samples=${MAX_SAMPLES}) ==="
uv run python scripts/prepare_boundless_das_dataset.py \
    --counterfactual-dataset datasets/multiplication_counterfactual.json \
    --counterfactual-dataset-write-down datasets/multiplication_write_down_counterfactual.json \
    --output-dir datasets/boundless_das \
    --tokenizer Qwen/Qwen2-7B \
    --max-samples "$MAX_SAMPLES"

# 2) Train Boundless DAS (GPU)
echo "=== Training Boundless DAS ==="
uv run python scripts/train_boundless_das.py \
    --data-dir datasets/boundless_das \
    --model-name Qwen/Qwen2-7B \
    --layer 25 \
    --epochs 1 \
    --batch-size 16 \
    --output-dir outputs/boundless_das \
    --intervention-type "carry_over" \
    --step 0 \
    "$@"

echo "End: $(date)"
