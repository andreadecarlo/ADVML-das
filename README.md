# ADVML DAS: Boundless DAS on Multiplication Scratchpads

This project trains **Boundless DAS** (Disentangled Alignment via Steering) interventions on a causal language model (Qwen2-7B) using multiplication scratchpad counterfactuals. The goal is to steer the model at a single layer and token position so that, after intervention, it predicts the counterfactual outcome (e.g. a different "write down" or "carry over" value) and thus the correct counterfactual product.

## Overview

1. **Dataset generation**: Build factual and counterfactual multiplication scratchpad data (single-digit × multi-digit, step-by-step "Write down X and carry over Y").
2. **Data preparation**: Tokenize pairs (base scratchpad, counterfactual scratchpad), compute intervention positions, and create train/val/test splits for Boundless DAS.
3. **Prealignment (optional)**: Evaluate the base model on multiplication before any training to establish a baseline.
4. **Training**: Train Boundless DAS interventions (pyvene) at a chosen layer; base model is frozen.
5. **Evaluation**: Evaluate Interchange Intervention Accuracy (IIA) on val/test with the trained interventions.

## Requirements

- **Python** ≥ 3.12
- **uv** (recommended) or **pip** for dependency management
- **CUDA**-capable GPU for training and evaluation (Qwen2-7B)

## Installation

From the project root:

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e .
```

Dependencies include PyTorch, Transformers, [pyvene](https://github.com/stanfordnlp/pyvene) (from GitHub), `datasets`, `wandb`, and others (see `pyproject.toml`).

## Pipeline

### 1. Generate multiplication datasets

Produces factual scratchpads and two kinds of counterfactuals (carry-only, write-down–only). Saves JSON under `datasets/`.

```bash
uv run python scripts/generate_multiplication_dataset.py
```

Outputs (in `datasets/`):

- `multiplication_factual.json`
- `multiplication_carry_counterfactual.json`
- `multiplication_write_down_counterfactual.json`

Or submit the Slurm job:

```bash
sbatch sbatches/generate_dataset.sh
```

### 2. Prepare Boundless DAS dataset

Tokenizes base/source pairs, finds intervention positions (“Write down” / “carry over” tokens), and creates train/val/test splits. Writes to `datasets/boundless_das/`.

```bash
uv run python scripts/prepare_boundless_das_dataset.py \
  --counterfactual-dataset datasets/multiplication_carry_counterfactual.json \
  --counterfactual-dataset-write-down datasets/multiplication_write_down_counterfactual.json \
  --output-dir datasets/boundless_das \
  --tokenizer Qwen/Qwen2-7B \
  --x-digits 2 \
  --y-digits 1
```

Optional: `--max-samples`, `--train-ratio`, `--val-ratio`, `--seed`.

Or:

```bash
sbatch sbatches/prepare_boundless_das_dataset.sh
```

The script uses `MAX_SAMPLES`, `X_DIGITS`, `Y_DIGITS` from the environment when set.

### 3. (Optional) Prealignment

Evaluate the base model on multiplication (question-only or full scratchpad) before training:

```bash
uv run python scripts/prealign_multiplication.py \
  --model Qwen/Qwen2-7B \
  --data-dir datasets \
  --prompt-type scratchpad \
  --max-samples 400 \
  --output results/prealign/
  --csv results/prealign/
```

Or:

```bash
sbatch sbatches/prealign_multiplication.sh
```

### 4. Train Boundless DAS

Train a single intervention at one layer (and optionally one intervention type and step). Only the intervention parameters are trained; the base model is frozen.

```bash
uv run python scripts/train_boundless_das.py \
  --data-dir datasets/boundless_das \
  --model-name Qwen/Qwen2-7B \
  --layer 15 \
  --intervention-type carry_over \
  --step 0 \
  --epochs 5 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --lr 1e-3 \
  --boundary-lr 1e-2 \
  --max-train-samples 5000 \
  --output-dir outputs/boundless_das_layer15_carry_over_step0 \
  --iia-resultwise
```

Use `--wandb-project` and `--wandb-run-name` to log to Weights & Biases.

Single job:

```bash
sbatch sbatches/boundless_das.sh
```

Sweep over layers {5,10,15,20,25} × types {carry_over, write_down} × steps {0,1,2} (30 jobs):

```bash
sbatch sbatches/boundless_das_array.sh
```

### 5. Evaluate trained interventions

Compute val/test IIA for a given (layer, intervention-type, step) checkpoint:

```bash
uv run python scripts/eval_boundless_das.py \
  --data-dir datasets/boundless_das \
  --model-name Qwen/Qwen2-7B \
  --layer 15 \
  --intervention-type carry_over \
  --step 0 \
  --load-dir outputs/5_epochs/boundless_das_layer15_carry_over_step0/intervention \
  --iia-resultwise \
  --output-csv results/alignment_results.csv
```

Array evaluation over all 30 configurations (after training with `boundless_das_array.sh`):

```bash
sbatch sbatches/eval_boundless_das.sh
```

Results are appended to `results/alignment_results_3_epochs.csv` (or the path set in the script).

## Project layout

```
ADVML_das/
├── pyproject.toml           # Dependencies (uv/pip)
├── README.md
├── scripts/
│   ├── generate_multiplication_dataset.py   # Factual + counterfactual data
│   ├── multiplication_boundless_das_utils.py # Token positions, samplers
│   ├── prepare_boundless_das_dataset.py     # Train/val/test + intervention positions
│   ├── prealign_multiplication.py           # Base model eval on multiplication
│   ├── train_boundless_das.py               # Boundless DAS training
│   └── eval_boundless_das.py                # IIA evaluation
├── sbatches/                # Slurm job scripts
│   ├── generate_dataset.sh
│   ├── prepare_boundless_das_dataset.sh
│   ├── prealign_multiplication.sh
│   ├── boundless_das.sh
│   ├── boundless_das_array.sh
│   └── eval_boundless_das.sh
├── datasets/                # Generated and prepared data
│   ├── multiplication_*.json
│   └── boundless_das/
│       ├── boundless_das_{train,val,test}.json
│       └── boundless_das_{train,val,test}.jsonl
├── outputs/                 # Training checkpoints (interventions)
│   └── 5_epochs/boundless_das_layer{L}_{type}_step{S}/intervention/
├── results/                 # CSVs and eval outputs
└── notebooks/               # Exploratory notebooks
```

## Main training parameters

| Parameter            | Default  | Description                                      |
|---------------------|----------|--------------------------------------------------|
| `--model-name`      | Qwen/Qwen2-7B | Base causal LM (frozen).                    |
| `--layer`           | 15       | Layer at which the intervention is applied.   |
| `--intervention-type` | None  | `carry_over` or `write_down` (or both if None). |
| `--step`            | None     | 0, 1, or 2 (ones, tens, hundreds).              |
| `--epochs`          | 3        | Number of training epochs.                       |
| `--batch-size`      | 4        | Per-device batch size.                           |
| `--lr`              | 1e-3     | Learning rate for intervention rotation.       |
| `--boundary-lr`     | 1e-2     | Learning rate for intervention boundaries.     |
| `--warmup-ratio`    | 0.1      | Fraction of steps for linear warmup.             |
| `--max-train-samples` | None  | Cap on training examples.                        |
| `--iia-resultwise`  | off      | Evaluate IIA by decoded product match.           |

The base model is loaded in **bfloat16** and **eval** mode; only the pyvene intervention (rotation + boundaries) is trained. Loss is cross-entropy on the counterfactual labels plus L1 on the boundaries; temperature is annealed from 50 to 0.1 over training.

## References

- **pyvene**: [stanfordnlp/pyvene](https://github.com/stanfordnlp/pyvene)
- **Boundless DAS**: Disentangled Alignment via Steering (see pyvene docs and related work on representation interventions).
