#!/bin/bash
#SBATCH --job-name=eval_pythia_mult
#SBATCH --partition=edu-long
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --output=.out/eval_pythia_multiplication_%j.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# Evaluate Pythia on multiplication scratchpad data.
# Usage (from project root):
#   sbatch sbatches/eval_pythia_multiplication.sbatch
#   sbatch sbatches/eval_pythia_multiplication.sbatch --model EleutherAI/pythia-410m --max-samples 1000
#   sbatch sbatches/eval_pythia_multiplication.sbatch --models EleutherAI/pythia-70m EleutherAI/pythia-410m --output results/pythia_mult.json
#   sbatch sbatches/eval_pythia_multiplication.sbatch --prompt-type scratchpad --max-samples 500

module load CUDA/12.5.0

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

uv run python scripts/eval_pythia_multiplication.py "$@" \
    --max-samples 200 \
    --max-examples-in-json 100 \
    --max-prompt-chars 500 \
    --prompt-type question   \
    --csv results/pythia1b_multiplication_first_ft_question/ \
    --output results/pythia1b_multiplication_first_ft_question/\
    --model EleutherAI/pythia-1b-multiplication-first-ft \

    # --model Qwen/Qwen2-7B \
    # --model models/finetune_pythia/without_column_prompt/final
    # --model Qwen/Qwen2.5-7B-Instruct
    # --all-models \
