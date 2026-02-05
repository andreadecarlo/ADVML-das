#!/usr/bin/env python3
"""
Evaluate Pythia family models on multiplication questions from scratchpad_zfill data.

Usage:
  uv run python scripts/eval_pythia_multiplication.py --model EleutherAI/pythia-70m
  uv run python scripts/eval_pythia_multiplication.py --model EleutherAI/pythia-410m --max-samples 200
  uv run python scripts/eval_pythia_multiplication.py --model EleutherAI/pythia-70m --prompt-type scratchpad
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import StoppingCriteria, StoppingCriteriaList


# Default Pythia model IDs (EleutherAI)
PYTHIA_MODELS = [
    # "EleutherAI/pythia-14m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b"
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "multiplication" / "scratchpad_zfill"

# System prompt wrapping question and optional scratchpad for multiplication
SYSTEM_PROMPT_BASE = """To multiply two numbers, start by multiplying the rightmost digit of the multiplicand by each digit of the multiplier, writing down the products and carrying over any remainders.

Always answer with only the final product as a number (no words, no explanation).

Question: {question}"""


def format_multiplication_prompt(question: str, scratchpad: str = "", include_answer_prefix: bool = False) -> str:
    """Build prompt: base + optional 'Scratchpad: {scratchpad}' or 'Answer: ' prefix.
    
    If include_answer_prefix=True and scratchpad is empty, adds '\n\nAnswer: ' to match training format.
    """
    base = SYSTEM_PROMPT_BASE.format(question=question)
    if not scratchpad:
        if include_answer_prefix:
            return base + "\n\nAnswer: "
        return base
    return base + "\n\nScratchpad: " + scratchpad


def model_short_name(model_id: str) -> str:
    """e.g. EleutherAI/pythia-70m -> pythia-70m"""
    return model_id.split("/")[-1]


def _model_digits_suffix(
    model_ids: list[str], digits: int | None, prompt_type: str = "question"
) -> str:
    """e.g. pythia-70m_4digit_scratchpad or all_alldigit_question"""
    model_part = model_short_name(model_ids[0]) if len(model_ids) == 1 else "all"
    digits_part = str(digits) if digits is not None else "all"
    return f"{model_part}_{digits_part}digit_{prompt_type}"


def format_output_path(
    path: Path, model_ids: list[str], digits: int | None, ext: str, prompt_type: str = "question"
) -> Path:
    """
    If path is a folder (no .json/.csv extension), append {model}_{digits}digit_{prompt_type}.{ext}.
    Otherwise replace {model}, {digits}, {prompt_type} in the path string.
    """
    p = Path(str(path).rstrip("/")).expanduser()
    is_folder = p.suffix.lower() not in (".json", ".csv")
    if is_folder:
        suffix = _model_digits_suffix(model_ids, digits, prompt_type)
        return p / f"{suffix}.{ext}"
    s = str(path)
    model_part = model_short_name(model_ids[0]) if len(model_ids) == 1 else "all"
    digits_part = str(digits) if digits is not None else "all"
    s = s.replace("{model}", model_part).replace("{digits}", digits_part).replace("{prompt_type}", prompt_type)
    return Path(s)


def load_prompts(
    data_dir: Path,
    glob: str = "scratchpad_*_prompts.json",
    max_samples: int | None = None,
    seed: int = 42,
):
    """Load all prompt JSON files from data_dir and yield (filename, items). When max_samples is set, shuffle first then take that many."""
    files = sorted(data_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f"No files matching {glob} in {data_dir}")
    rng = random.Random(seed)
    for path in files:
        with open(path) as f:
            items = json.load(f)
        if max_samples is not None:
            items = list(items)
            rng.shuffle(items)
            items = items[:max_samples]
        yield path.name, items


def extract_answer(text: str) -> int | None:
    """
    Extract the product from model output.

    We take the LAST integer in the FIRST sentence so that we:
    - ignore any follow‑up questions or continuations the model might generate,
    - still handle patterns like "The product is 0014." or "0014." correctly.
    Handles zfilled output like "0021" and "21" by comparing as int (both → 21).
    """
    # First, restrict to the first sentence (up to the first ., ?, or !)
    m = re.search(r"[.!?]", text)
    if m:
        first_sentence = text[: m.end()]
    else:
        first_sentence = text

    matches = re.findall(r"\b(\d+)\b", first_sentence)
    if not matches:
        return None
    # Last number in the first sentence is treated as the final product.
    return int(matches[-1])


class StopOnNewlinePair(StoppingCriteria):
    """Stop generation when \"\\n\\n\" appears in the generated text (avoids generating next question)."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] <= self.prompt_length:
            return False
        gen = self.tokenizer.decode(input_ids[0][self.prompt_length:], skip_special_tokens=True)
        return "\n\n" in gen


@torch.no_grad()
def run_eval(
    model_id: str,
    data_dir: Path = DATA_DIR,
    max_samples: int | None = None,
    max_new_tokens: int = 50,
    prompt_type: str = "question",
    device: str | None = None,
    batch_size: int = 1,
    max_examples_in_json: int = 0,
    max_prompt_chars: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Evaluate a single model on multiplication data.

    prompt_type: "question" = only the question; "scratchpad" = full scratchpad prompt.
    max_examples_in_json: cap on examples (prompt, answer, expected_answer) in JSON; 0 = none.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    results_by_file: dict[str, dict] = {}
    all_correct = 0
    all_total = 0
    examples: list[dict] = []

    for filename, items in load_prompts(data_dir, max_samples=max_samples, seed=seed):
        correct = 0
        total = len(items)
        pbar = tqdm(
            range(0, total, batch_size),
            desc=f"{model_id} {filename}",
            leave=False,
            unit="batch",
        )
        for i in pbar:
            batch = items[i : i + batch_size]
            for item in batch:
                if prompt_type == "scratchpad":
                    scratchpad_part = item["prompt"][:-6]
                    prompt = format_multiplication_prompt(item["question"], scratchpad_part, include_answer_prefix=False)
                else:
                    # For question-only mode, use Answer: prefix to match training format
                    prompt = format_multiplication_prompt(item["question"], "", include_answer_prefix=True)
                expected = item["expected_answer"]

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
                prompt_length = inputs["input_ids"].shape[1]
                stopping_criteria = StoppingCriteriaList([
                    StopOnNewlinePair(tokenizer, prompt_length),
                ])
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )
                # Decode only the generated part (exclude input)
                gen_len = out.shape[1] - prompt_length
                if gen_len <= 0:
                    gen = ""
                    pred = None
                else:
                    gen = tokenizer.decode(out[0][-gen_len:], skip_special_tokens=True)
                    # Use only text before first "\n\n" (like OLD script: one answer segment, no next question)
                    if "\n\n" in gen:
                        gen = gen.split("\n\n", 1)[0].strip()
                    pred = extract_answer(gen)

                if pred is not None and pred == expected:
                    correct += 1
                if max_examples_in_json > 0 and len(examples) < max_examples_in_json:
                    prompt_stored = prompt if len(prompt) <= max_prompt_chars else prompt[:max_prompt_chars] + "..."
                    examples.append({
                        "prompt": prompt_stored,
                        "answer": pred,
                        "expected_answer": expected,
                        "raw_output": gen,
                    })
            processed = min(i + batch_size, total)
            pbar.set_postfix(accuracy=f"{correct / processed:.2%}" if processed else "N/A")
        results_by_file[filename] = {"correct": correct, "total": total, "accuracy": correct / total if total else 0}
        all_correct += correct
        all_total += total

    result: dict = {
        "model": model_id,
        "prompt_type": prompt_type,
        "by_file": results_by_file,
        "overall": {
            "correct": all_correct,
            "total": all_total,
            "accuracy": all_correct / all_total if all_total else 0,
        },
    }
    if examples:
        result["examples"] = examples
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pythia on multiplication scratchpad data")
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m",
        help="HuggingFace model ID (default: EleutherAI/pythia-70m)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Evaluate multiple models (overrides --model)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help=f"Run all Pythia models: {', '.join(PYTHIA_MODELS)}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory with scratchpad JSON files (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=None,
        help="Number of digits (for output filename placeholder {digits})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per file (default: all); when set, data is shuffled before sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling when using --max-samples (default: 42)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Max new tokens per generation (default: 50)",
    )
    parser.add_argument(
        "--prompt-type",
        choices=("question", "scratchpad"),
        default="question",
        help="question = only 'What is the product...'; scratchpad = full step-by-step prompt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1)",
    )
    parser.add_argument(
        "--max-examples-in-json",
        type=int,
        default=0,
        help="Max examples to store in JSON (prompt, answer, expected_answer); 0 = none (default: 0)",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=2000,
        help="Truncate prompt to this many chars when storing in JSON (default: 2000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Folder for results JSON; filename is {model}_{digits}digit.json (or pass a full path with {model}/{digits})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Folder for stats CSV; filename is {model}_{digits}digit.csv (or pass a full path with {model}/{digits})",
    )
    args = parser.parse_args()

    if args.all_models:
        model_ids = PYTHIA_MODELS
        print(f"Evaluating all {len(model_ids)} Pythia models")
    elif args.models:
        model_ids = args.models
    else:
        model_ids = [args.model]
    all_results = []

    for model_id in model_ids:
        print(f"Evaluating {model_id} ...")
        res = run_eval(
            model_id=model_id,
            data_dir=args.data_dir,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            prompt_type=args.prompt_type,
            batch_size=args.batch_size,
            max_examples_in_json=args.max_examples_in_json,
            max_prompt_chars=args.max_prompt_chars,
            seed=args.seed,
        )
        all_results.append(res)
        print(f"  Overall accuracy: {res['overall']['accuracy']:.2%} ({res['overall']['correct']}/{res['overall']['total']})")
        for name, stats in res["by_file"].items():
            print(f"    {name}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    if args.output:
        out_path = format_output_path(args.output, model_ids, args.digits, "json", args.prompt_type)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results if len(all_results) > 1 else all_results[0], f, indent=2)
        print(f"Results written to {out_path}")

    if args.csv:
        csv_path = format_output_path(args.csv, model_ids, args.digits, "csv", args.prompt_type)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "prompt_type", "split", "correct", "total", "accuracy"])
            for res in all_results:
                for split, stats in res["by_file"].items():
                    w.writerow([
                        res["model"],
                        res["prompt_type"],
                        split,
                        stats["correct"],
                        stats["total"],
                        f"{stats['accuracy']:.4f}",
                    ])
                w.writerow([
                    res["model"],
                    res["prompt_type"],
                    "overall",
                    res["overall"]["correct"],
                    res["overall"]["total"],
                    f"{res['overall']['accuracy']:.4f}",
                ])
        print(f"Stats written to {csv_path}")


if __name__ == "__main__":
    main()
