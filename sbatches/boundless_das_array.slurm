#!/bin/bash
#SBATCH --job-name=boundless_das_arr
#SBATCH --partition=edu-short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --array=0-29
##SBATCH --time=24:00:00
#SBATCH --output=.out/boundless_das_array_%A_%a.out

# Boundless DAS array: layers {5,10,15,20,25} x intervention-type {carry_over, write_down} x step {0,1,2}.
# Array index 0-29: task_id -> layer_idx=task_id/6, type_idx=(task_id/3)%2, step_idx=task_id%3.
# Usage (from project root):
#   sbatch sbatches/boundless_das_array.slurm
#   MAX_SAMPLES=5000 sbatch sbatches/boundless_das_array.slurm

set -e
LAYERS=(5 10 15 20 25)
TYPES=(carry_over write_down)
STEP_NAMES=(ones tens hundreds)
task_id=${SLURM_ARRAY_TASK_ID:-0}
layer_idx=$(( task_id / 6 ))
type_idx=$(( (task_id / 3) % 2 ))
step_idx=$(( task_id % 3 ))
LAYER=${LAYERS[$layer_idx]}
TYPE=${TYPES[$type_idx]}
STEP=$step_idx
STEP_NAME=${STEP_NAMES[$step_idx]}
OUTPUT_DIR="outputs/boundless_das_layer${LAYER}_${TYPE}_step${STEP}"

echo "Job ID: $SLURM_JOB_ID Array ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: layer=$LAYER intervention-type=$TYPE step=$STEP ($STEP_NAME)"
echo "Start: $(date)"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p .out datasets/boundless_das outputs/boundless_das "$OUTPUT_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 1) Prepare dataset (same for all array tasks; overwrites if multiple run)
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
echo "=== Preparing Boundless DAS dataset (max_samples=${MAX_SAMPLES}) ==="
uv run python scripts/prepare_boundless_das_dataset.py \
    --counterfactual-dataset datasets/multiplication_carry_counterfactual.json \
    --counterfactual-dataset-write-down datasets/multiplication_write_down_counterfactual.json \
    --output-dir datasets/boundless_das \
    --tokenizer Qwen/Qwen2-7B \
    # --max-samples "$MAX_SAMPLES"

# 2) Train for this (layer, intervention-type, step)
echo "=== Training Boundless DAS layer=$LAYER intervention-type=$TYPE step=$STEP ($STEP_NAME) ==="
uv run python scripts/train_boundless_das.py \
    --data-dir datasets/boundless_das \
    --model-name Qwen/Qwen2-7B \
    --layer "$LAYER" \
    --epochs 1 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --output-dir "$OUTPUT_DIR" \
    --intervention-type "$TYPE" \
    --step "$STEP" \
    "$@"

echo "End: $(date)"
