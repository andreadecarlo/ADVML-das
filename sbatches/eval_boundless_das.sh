#!/bin/bash
#SBATCH --job-name=eval_boundless_das_arr
#SBATCH --partition=edu-short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --array=0-29
##SBATCH --time=24:00:00
#SBATCH --output=.out/eval_boundless_das_array_%A_%a.out

# Boundless DAS array: layers {5,10,15,20,25} x intervention-type {carry_over, write_down} x step {0,1,2}.
# Array index 0-29: task_id -> layer_idx=task_id/6, type_idx=(task_id/3)%2, step_idx=task_id%3.
# Usage (from project root):
#   # 1) One-off canonical dataset preparation (no array):
#   #    sbatch sbatches/prepare_boundless_das_dataset.slurm
#   #
#   # 2) Many parallel training jobs reading the same canonical dataset:
#   #    sbatch sbatches/boundless_das_array.slurm

set -e
NUM_EPOCHS=3
LAYERS=(15 20 25 10 5)
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
OUTPUT_DIR="outputs/${NUM_EPOCHS}_epochs/boundless_das_layer${LAYER}_${TYPE}_step${STEP}/intervention"

echo "Job ID: $SLURM_JOB_ID Array ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: layer=$LAYER intervention-type=$TYPE step=$STEP ($STEP_NAME)"
echo "Start: $(date)"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 1) Evaluation for this (layer, intervention-type, step) using the canonical dataset
#    prepared once by sbatches/prepare_boundless_das_dataset.slurm.
echo "=== Evaluating Boundless DAS layer=$LAYER intervention-type=$TYPE step=$STEP ($STEP_NAME) ==="
uv run python scripts/eval_boundless_das.py \
    --data-dir datasets/boundless_das \
    --model-name Qwen/Qwen2-7B \
    --layer "$LAYER" \
    --step "$STEP" \
    --intervention-type "$TYPE" \
    --load-dir "$OUTPUT_DIR" \
    --batch-size-eval 1 \
    --iia-resultwise \
    --output-csv results/alignment_results_${NUM_EPOCHS}_epochs.csv
    "$@"

echo "End: $(date)"
