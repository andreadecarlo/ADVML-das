#!/bin/bash
#SBATCH --job-name=boundless_das
#SBATCH --partition=edu-short 
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
##SBATCH --time=24:00:00
# Write both stdout and stderr to the same log file so that
# tqdm.write-based debug lines from train_boundless_das.py
# (which go to stderr) appear in the .out/boundless_das_*.out logs.
#SBATCH --output=.out/boundless_das_%j.out
#SBATCH --error=.out/boundless_das_%j.out

# Boundless DAS (multiplication counterfactual): train on canonical dataset. Qwen2-7B.
# Plan:
#   (1) Run sbatches/prepare_boundless_das_dataset.slurm once to build datasets/boundless_das/*.jsonl
#   (2) Use this script and sbatches/boundless_das_array.slurm for training only (read-only on data).
# Usage (from project root):
#   sbatch sbatches/boundless_das.slurm

set -e
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p .out datasets/boundless_das outputs/boundless_das
# Reduce CUDA fragmentation (PyTorch recommendation when OOM)
export PYTORCH_ALLOC_CONF=expandable_segments:True
# SLURM writes to .out; create it so the job doesn't fail if sbatch doesn't

# Optional: load CUDA if required on your cluster
# module load CUDA/12.5.0

# 1) Train Boundless DAS (GPU) on the canonical dataset prepared once by
#    sbatches/prepare_boundless_das_dataset.slurm.
echo "=== Training Boundless DAS ==="
uv run python scripts/train_boundless_das.py \
    --data-dir datasets/boundless_das \
    --model-name Qwen/Qwen2-7B \
    --layer 20 \
    --epochs 1 \
    --batch-size 8 \
    --eval-batch-size 1 \
    --gradient-accumulation-steps 2 \
    --eval-steps 200 \
    --log-steps 50 \
    --output-dir outputs/boundless_das \
    --intervention-type "carry_over" \
    --step 0 \
    --wandb-project "boundless_das" \
    --wandb-run-name "layer20_carry_over_step0" \
    --max-train-samples 1000 \
    --iia-resultwise \
    "$@"

echo "End: $(date)"
