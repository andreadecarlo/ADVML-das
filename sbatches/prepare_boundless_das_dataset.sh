#!/bin/bash
#SBATCH --job-name=prep_boundless_das
#SBATCH --partition=edu-medium
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
##SBATCH --time=04:00:00
#SBATCH --output=.out/prepare_boundless_das_%j.out

# One-off canonical Boundless DAS dataset preparation.
# This job SHOULD be run only occasionally to (re)build the shared dataset:
#   sbatch sbatches/prepare_boundless_das_dataset.slurm
#
# After this finishes, all training jobs (including the array in
# sbatches/boundless_das_array.slurm) ONLY READ from datasets/boundless_das.

set -e
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p .out datasets/boundless_das

# Control dataset size via MAX_SAMPLES (optional; default: 30000)
MAX_SAMPLES="${MAX_SAMPLES:-30000}"
# Limit operand ranges to reduce combinatorial explosion (default: 2-digit x 1-digit).
X_DIGITS="${X_DIGITS:-2}"
Y_DIGITS="${Y_DIGITS:-1}"
echo "=== Preparing canonical Boundless DAS dataset ==="
uv run python scripts/prepare_boundless_das_dataset.py \
    --counterfactual-dataset datasets/multiplication_carry_counterfactual.json \
    --counterfactual-dataset-write-down datasets/multiplication_write_down_counterfactual.json \
    --output-dir datasets/boundless_das \
    --tokenizer Qwen/Qwen2-7B \
    --x-digits "$X_DIGITS" \
    --y-digits "$Y_DIGITS"
    # --max-samples "$MAX_SAMPLES" \

echo "End: $(date)"

