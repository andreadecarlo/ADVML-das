"""
Prepare multiplication datasets for Boundless DAS experiments.
Creates datasets with intervention positions for write-down and carry-over values,
differentiated by step (ones, tens, hundreds). Supports both scratchpad and
prompt-only counterfactual formats. Outputs unified train/val/test splits with
positions from data; labels are counterfactual (source) for IIA training.
"""

import sys
from pathlib import Path

# Allow imports from scripts/ when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer
from multiplication_boundless_das_utils import (
    find_write_down_token_position,
    find_carry_over_token_position,
    compute_carries_from_prompt,
)
from generate_multiplication_dataset import digits, compute_write_down_values


def _normalize_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure original/counterfactual have scratchpad and write_down_values (from prompt/x/y if needed)."""
    for key in ("original", "counterfactual"):
        obj = example[key]
        if "scratchpad" not in obj and "prompt" in obj:
            lines = obj["prompt"].split("\n")
            obj["scratchpad"] = "\n".join(lines[1:]) if len(lines) > 1 else ""
        if "write_down_values" not in obj and "x" in obj and "y" in obj:
            x_digits = digits(obj["x"])
            obj["write_down_values"] = compute_write_down_values(x_digits, obj["y"])
        if "question" not in obj and "prompt" in obj:
            obj["question"] = obj["prompt"].split("\n")[0]
    return example


def _infer_intervention_type(example: Dict[str, Any]) -> str:
    """Infer intervention type from counterfactual keys."""
    if "old_write_down" in example or "new_write_down" in example:
        return "write_down"
    return "carry_over"


def prepare_boundless_das_dataset(
    counterfactual_dataset_path: str,
    output_dir: str = "datasets/boundless_das",
    tokenizer_name: str = "Qwen/Qwen2-7B",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    counterfactual_dataset_path_write_down: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """
    Prepare counterfactual dataset for Boundless DAS with intervention positions from data.
    Supports both carry and write_down; step and position come from each example.
    Labels are set to counterfactual (source) sequence for IIA training.

    Args:
        counterfactual_dataset_path: Path to counterfactual JSON (e.g. carry)
        output_dir: Output directory for processed datasets
        tokenizer_name: HuggingFace tokenizer name
        train_ratio: Fraction for train split
        val_ratio: Fraction for val split (test = 1 - train - val)
        seed: Random seed for split
        counterfactual_dataset_path_write_down: Optional path to write_down counterfactual JSON
        max_samples: If set, only process this many examples (for faster/smaller runs)
    """
    random.seed(seed)
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load counterfactual dataset(s)
    all_examples: List[Dict[str, Any]] = []
    paths = [counterfactual_dataset_path]
    if counterfactual_dataset_path_write_down:
        paths.append(counterfactual_dataset_path_write_down)
    for path in paths:
        print(f"Loading from {path}")
        with open(path) as f:
            data = json.load(f)
        if max_samples is not None and len(all_examples) + len(data) > max_samples:
            take = max_samples - len(all_examples)
            data = data[:take]
        for ex in data:
            ex = _normalize_example(ex)
            itype = _infer_intervention_type(ex)
            ex["_intervention_type"] = itype
            all_examples.append(ex)
        if max_samples is not None and len(all_examples) >= max_samples:
            break
    if max_samples is not None:
        all_examples = all_examples[:max_samples]
    print(f"Loaded {len(all_examples)} counterfactual examples")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed: List[Dict[str, Any]] = []
    for example in all_examples:
        original = example["original"]
        counterfactual = example["counterfactual"]
        step = example.get("step", 0)
        intervention_type = example["_intervention_type"]
        num_steps = len(original.get("write_down_values", []))
        if num_steps == 0:
            continue
        orig_carries = compute_carries_from_prompt(
            original["x"], original["y"], num_steps
        )
        if intervention_type == "write_down":
            intervention_pos = find_write_down_token_position(
                tokenizer,
                original["scratchpad"],
                step,
                original["write_down_values"][step],
            )
        else:
            intervention_pos = find_carry_over_token_position(
                tokenizer,
                original["scratchpad"],
                step,
                orig_carries[step],
            )
        if intervention_pos is None:
            continue
        base_prompt = original["question"] + "\n" + original["scratchpad"]
        source_prompt = counterfactual["question"] + "\n" + counterfactual["scratchpad"]
        base_input_ids = tokenizer.encode(
            base_prompt, add_special_tokens=True, return_tensors="pt"
        )[0].tolist()
        source_input_ids = tokenizer.encode(
            source_prompt, add_special_tokens=True, return_tensors="pt"
        )[0].tolist()
        question_tokens = tokenizer.encode(
            original["question"] + "\n", add_special_tokens=False
        )
        intervention_pos += len(question_tokens)
        if tokenizer.bos_token_id is not None:
            intervention_pos += 1
        # Labels = counterfactual (source) sequence for IIA: after intervention we predict source
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        labels = source_input_ids[1:] + [eos]
        step_name = ["ones", "tens", "hundreds"][step] if step < 3 else f"step_{step}"
        processed.append({
            "input_ids": base_input_ids,
            "source_input_ids": source_input_ids,
            "labels": labels,
            "intervention_ids": [intervention_pos],
            "step": step,
            "step_name": step_name,
            "intervention_type": intervention_type,
        })
    random.shuffle(processed)
    n = len(processed)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_data = processed[:n_train]
    val_data = processed[n_train : n_train + n_val]
    test_data = processed[n_train + n_val :]
    # Save per-type/per-step files for optional use
    for intervention_type in ["write_down", "carry_over"]:
        for step in [0, 1, 2]:
            step_name = ["ones", "tens", "hundreds"][step]
            subset = [
                ex
                for ex in processed
                if ex["intervention_type"] == intervention_type and ex["step"] == step
            ]
            if subset:
                out_file = output_path / f"multiplication_{intervention_type}_{step_name}.json"
                with open(out_file, "w") as f:
                    json.dump(subset, f, indent=2)
                print(f"  Saved {len(subset)} examples to {out_file.name}")
    # Save unified splits (no full original/counterfactual to keep size small)
    splits = {"train": train_data, "val": val_data, "test": test_data}
    for split_name, split_data in splits.items():
        out_file = output_path / f"boundless_das_{split_name}.json"
        with open(out_file, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"  Saved {len(split_data)} examples to {out_file.name}")
    print(f"\nAll datasets saved to {output_path}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Boundless DAS datasets")
    parser.add_argument(
        "--counterfactual-dataset",
        type=str,
        default="datasets/multiplication_counterfactual.json",
        help="Path to counterfactual dataset JSON (e.g. carry)",
    )
    parser.add_argument(
        "--counterfactual-dataset-write-down",
        type=str,
        default=None,
        help="Optional path to write_down counterfactual JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/boundless_das",
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2-7B",
        help="HuggingFace tokenizer name",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of counterfactual examples to process (default: all)",
    )
    args = parser.parse_args()

    prepare_boundless_das_dataset(
        args.counterfactual_dataset,
        args.output_dir,
        args.tokenizer,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        counterfactual_dataset_path_write_down=args.counterfactual_dataset_write_down,
        max_samples=args.max_samples,
    )
