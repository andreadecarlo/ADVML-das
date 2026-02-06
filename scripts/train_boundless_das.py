# Train Boundless DAS on the multiplication counterfactual dataset.
# Intervene on one token at a time; position from data. Supports layer, step, intervention_type. Evaluates IIA.

import csv
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Suppress verbose model/tokenizer loading logs
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

try:
    from datasets import load_dataset as hf_load_dataset
    _HF_DATASETS_AVAILABLE = True
except ImportError:
    _HF_DATASETS_AVAILABLE = False

from pyvene import (
    IntervenableModel,
    BoundlessRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene import set_seed, count_parameters


def simple_boundless_das_position_config(model_type, intervention_type: str, layer: int):
    # Config for Boundless DAS at a single layer (position-aligned).
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(layer, intervention_type),
        ],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    return config


def compute_iia(logits: torch.Tensor, labels: torch.Tensor, last_token_only: bool = True) -> float:
    # Interchange Intervention Accuracy: after intervention, does the model predict the counterfactual (source) next token?
    if last_token_only:
        pred = torch.argmax(logits[:, -1], dim=-1)
        actual = labels[:, -1]
    else:
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        pred = torch.argmax(shift_logits, dim=-1)
        actual = shift_labels
    valid = actual != -100
    if valid.sum() == 0:
        return 0.0
    return (pred[valid] == actual[valid]).float().mean().item()


def load_boundless_das_splits(
    data_dir: str,
    intervention_type: Optional[str] = None,
    step: Optional[int] = None,
) -> tuple:
    # Load train/val/test from prepared boundless_das JSONs; optional filter by type/step.
    data_path = Path(data_dir)
    splits = {}
    for name in ("train", "val", "test"):
        p = data_path / f"boundless_das_{name}.json"
        if not p.exists():
            raise FileNotFoundError(f"Run prepare_boundless_das_dataset first; missing {p}")
        with open(p) as f:
            data = json.load(f)
        if intervention_type is not None:
            data = [ex for ex in data if ex["intervention_type"] == intervention_type]
        if step is not None:
            data = [ex for ex in data if ex["step"] == step]
        splits[name] = data
    return splits["train"], splits["val"], splits["test"]


def load_boundless_das_hf(
    data_dir: str,
    intervention_type: Optional[str] = None,
    step: Optional[int] = None,
):
    """Load train/val/test using HuggingFace datasets (memory-mapped, efficient)."""
    data_path = Path(data_dir)
    jsonl_files = {
        "train": str(data_path / "boundless_das_train.jsonl"),
        "val": str(data_path / "boundless_das_val.jsonl"),
        "test": str(data_path / "boundless_das_test.jsonl"),
    }
    for name, p in jsonl_files.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"JSONL not found: {p}. Run prepare_boundless_das_dataset first.")
    ds = hf_load_dataset(
        "json",
        data_files=jsonl_files,
        split=None,
        cache_dir=str(data_path / ".hf_cache"),
    )
    train_ds, val_ds, test_ds = ds["train"], ds["val"], ds["test"]

    def _filter(ex):
        if intervention_type is not None and ex.get("intervention_type") != intervention_type:
            return False
        if step is not None and ex.get("step") != step:
            return False
        return True

    train_ds = train_ds.filter(_filter, num_proc=1, desc="Filter train")
    val_ds = val_ds.filter(_filter, num_proc=1, desc="Filter val")
    test_ds = test_ds.filter(_filter, num_proc=1, desc="Filter test")
    return train_ds, val_ds, test_ds


def _example_to_item(ex: Dict, pad_id: int, max_length: Optional[int]) -> Dict[str, torch.Tensor]:
    # Convert one example dict to dataset item (shared by in-memory and lazy).
    input_ids = ex["input_ids"]
    source_input_ids = ex["source_input_ids"]
    labels = ex["labels"]
    pos = ex["intervention_ids"][0]
    if max_length:
        input_ids = input_ids[:max_length]
        source_input_ids = source_input_ids[:max_length]
        labels = labels[:max_length]
        pos = min(pos, len(input_ids) - 1)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "source_input_ids": torch.tensor(source_input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "intervention_position": int(pos),
    }


class LazyBoundlessDASDataset(Dataset):
    # Reads one example per __getitem__ from JSONL; avoids loading full JSON into RAM.

    def __init__(
        self,
        jsonl_path: Path,
        pad_id: int,
        max_length: Optional[int] = None,
        intervention_type: Optional[str] = None,
        step: Optional[int] = None,
    ):
        self.jsonl_path = jsonl_path
        self.pad_id = pad_id
        self.max_length = max_length
        # One pass: build list of byte offsets for lines that pass the filter
        self._offsets: List[int] = []
        with open(jsonl_path) as f:
            offset = 0
            for line in f:
                start = offset
                offset += len(line.encode("utf-8"))
                if not line.strip():
                    continue
                if intervention_type is not None or step is not None:
                    ex = json.loads(line)
                    if intervention_type is not None and ex.get("intervention_type") != intervention_type:
                        continue
                    if step is not None and ex.get("step") != step:
                        continue
                self._offsets.append(start)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        with open(self.jsonl_path) as f:
            f.seek(self._offsets[i])
            line = f.readline()
        ex = json.loads(line)
        return _example_to_item(ex, self.pad_id, self.max_length)


class BoundlessDASDataset(Dataset):
    # Dataset of (input_ids, source_input_ids, labels, intervention_position).

    def __init__(self, examples: List[Dict], pad_id: int, max_length: Optional[int] = None):
        self.examples = examples
        self.pad_id = pad_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[i]
        return _example_to_item(ex, self.pad_id, self.max_length)


class HFDatasetAdapter(Dataset):
    """Wraps a HuggingFace Dataset for Boundless DAS; memory-mapped, no full load."""

    def __init__(self, hf_dataset, pad_id: int, max_length: Optional[int] = None):
        self.hf_dataset = hf_dataset
        self.pad_id = pad_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row = self.hf_dataset[i]
        ex = {
            "input_ids": row["input_ids"],
            "source_input_ids": row["source_input_ids"],
            "labels": row["labels"],
            "intervention_ids": row["intervention_ids"],
        }
        return _example_to_item(ex, self.pad_id, self.max_length)


def collate_boundless_das(batch: List[Dict], pad_id: int):
    # Pad to max length in batch; stack intervention_position.
    max_len = max(b["input_ids"].size(0) for b in batch)
    device = batch[0]["input_ids"].device

    def pad(tensor, length):
        if tensor.size(0) >= length:
            return tensor[:length]
        return torch.cat([tensor, torch.full((length - tensor.size(0),), pad_id, dtype=tensor.dtype, device=tensor.device)])

    input_ids = torch.stack([pad(b["input_ids"], max_len) for b in batch])
    source_input_ids = torch.stack([pad(b["source_input_ids"], max_len) for b in batch])
    labels_list = []
    for b in batch:
        l = b["labels"]
        if l.size(0) < max_len:
            l = torch.cat([l, torch.full((max_len - l.size(0),), -100, dtype=l.dtype, device=l.device)])
        else:
            l = l[:max_len]
        labels_list.append(l)
    labels = torch.stack(labels_list)
    intervention_positions = torch.tensor([b["intervention_position"] for b in batch], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "source_input_ids": source_input_ids,
        "labels": labels,
        "intervention_positions": intervention_positions,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Boundless DAS (multiplication counterfactual)")
    parser.add_argument("--data-dir", type=str, default="datasets/boundless_das")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-7B")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--step", type=int, default=None, help="Filter by step (0,1,2); None = all")
    parser.add_argument("--intervention-type", type=str, choices=["carry_over", "write_down"], default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size (default 4)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--boundary-lr", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="outputs/boundless_das")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--use-hf-dataset",
        action="store_true",
        default=True,
        help="Use HuggingFace datasets for memory-mapped loading (default: True when datasets installed)",
    )
    parser.add_argument("--no-use-hf-dataset", action="store_false", dest="use_hf_dataset")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data: use JSONL (lazy) when present to avoid loading full JSON into RAM
    data_path = Path(args.data_dir)
    jsonl_train = data_path / "boundless_das_train.jsonl"
    import os
    _prev_hf = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    step_tag = str(args.step) if args.step is not None else "all"
    log_tag = f"[layer={args.layer} intervention_type={args.intervention_type} step={step_tag}]"

    use_hf = args.use_hf_dataset and _HF_DATASETS_AVAILABLE and jsonl_train.exists()
    if use_hf:
        train_hf, val_hf, test_hf = load_boundless_das_hf(
            args.data_dir,
            intervention_type=args.intervention_type,
            step=args.step,
        )
        train_ds = HFDatasetAdapter(train_hf, pad_id, args.max_length)
        val_ds = HFDatasetAdapter(val_hf, pad_id, args.max_length)
        test_ds = HFDatasetAdapter(test_hf, pad_id, args.max_length)
        print(f"Train {len(train_ds)} val {len(val_ds)} test {len(test_ds)} (HuggingFace datasets) {log_tag}")
    elif jsonl_train.exists():
        train_ds = LazyBoundlessDASDataset(
            data_path / "boundless_das_train.jsonl",
            pad_id, args.max_length,
            intervention_type=args.intervention_type, step=args.step,
        )
        val_ds = LazyBoundlessDASDataset(
            data_path / "boundless_das_val.jsonl",
            pad_id, args.max_length,
            intervention_type=args.intervention_type, step=args.step,
        )
        test_ds = LazyBoundlessDASDataset(
            data_path / "boundless_das_test.jsonl",
            pad_id, args.max_length,
            intervention_type=args.intervention_type, step=args.step,
        )
        print(f"Train {len(train_ds)} val {len(val_ds)} test {len(test_ds)} (lazy from JSONL) {log_tag}")
    else:
        train_ex, val_ex, test_ex = load_boundless_das_splits(
            args.data_dir,
            intervention_type=args.intervention_type,
            step=args.step,
        )
        print(f"Train {len(train_ex)} val {len(val_ex)} test {len(test_ex)} {log_tag}")
        train_ds = BoundlessDASDataset(train_ex, pad_id, args.max_length)
        val_ds = BoundlessDASDataset(val_ex, pad_id, args.max_length)
        test_ds = BoundlessDASDataset(test_ex, pad_id, args.max_length)

    if len(train_ds) == 0:
        raise ValueError(
            "No training examples. Run prepare_boundless_das_dataset first with --max-samples (e.g. 10000). "
            "Ensure counterfactual JSON has scratchpad/prompt and step/intervention_type."
        )
    if len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            "Val or test split is empty (e.g. from very small --max-samples). Use at least ~100 samples so val/test get examples."
        )
    print(f"Config {log_tag}")
    collate = lambda b: collate_boundless_das(b, pad_id)
    # num_workers=0 and pin_memory=False to avoid extra GPU/CPU memory from prefetch
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=0, pin_memory=False)

    # Model (quiet load)
    print(f"Loading model {args.model_name} {log_tag}")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    if _prev_hf is None:
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    else:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = _prev_hf
    model.to(device)
    model.eval()

    # Register Qwen2 in pyvene if not already supported (e.g. older pyvene; git version may have it)
    # _register_qwen2_in_pyvene_if_needed(type(model))

    config = simple_boundless_das_position_config(type(model), "block_output", args.layer)
    intervenable = IntervenableModel(config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Optimizer & scheduler
    t_total = len(train_loader) * args.epochs
    warm_up_steps = int(args.warmup_ratio * t_total)
    optimizer_params = []
    for v in intervenable.interventions.values():
        optimizer_params += [{"params": v.rotate_layer.parameters()}]
        optimizer_params += [{"params": v.intervention_boundaries, "lr": args.boundary_lr}]
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total)

    temperature_schedule = torch.linspace(50.0, 0.1, t_total).to(torch.bfloat16).to(device)
    total_step = 0
    intervenable.set_temperature(temperature_schedule[0])

    def calculate_loss(logits, labels, intervenable_model):
        shift_logits = logits.contiguous()
        shift_labels = labels.contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        shift_logits_flat = shift_logits.view(-1, intervenable_model.model_config.vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        loss = loss_fct(shift_logits_flat, shift_labels_flat)
        for v in intervenable_model.interventions.values():
            loss = loss + 1.0 * v.intervention_boundaries.sum()
        return loss

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    intervenable.model.train()
    print(f"Model trainable params: {count_parameters(intervenable.model)} {log_tag}")
    print(f"Intervention trainable params: {intervenable.count_parameters()} {log_tag}")

    # Training loop with IIA streaming
    last_train_iia = None
    last_mean_loss = None
    for epoch in trange(args.epochs, desc="Epoch"):
        epoch_iia_sum = 0.0
        epoch_iia_count = 0
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            b_s = batch["input_ids"].shape[0]
            positions = batch["intervention_positions"]
            # With batch_size=1 we pass one position; pyvene accepts {"sources->base": [pos]}
            unit_locations = {"sources->base": positions.tolist()}
            _, counterfactual_outputs = intervenable(
                {"input_ids": batch["input_ids"]},
                [{"input_ids": batch["source_input_ids"]}],
                unit_locations,
            )
            logits = counterfactual_outputs.logits
            labels_b = batch["labels"]
            iia = compute_iia(logits, labels_b, last_token_only=True)
            epoch_iia_sum += iia * b_s
            epoch_iia_count += b_s
            loss = calculate_loss(logits, labels_b, intervenable)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            epoch_loss_sum += loss.item() * b_s
            epoch_loss_count += b_s
            loss.backward()
            # Free large tensors before next batch to reduce peak GPU memory
            del logits, counterfactual_outputs
            if device == "cuda":
                torch.cuda.empty_cache()
            if (total_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                intervenable.set_zero_grad()
                if device == "cuda":
                    torch.cuda.empty_cache()
            total_step += 1
            if total_step < len(temperature_schedule):
                intervenable.set_temperature(temperature_schedule[total_step])
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "IIA": f"{iia:.3f}"})
            if (step + 1) % args.eval_steps == 0:
                intervenable.eval()
                val_iia_sum, val_n = 0.0, 0
                with torch.no_grad():
                    for v_batch in val_loader:
                        for k, v in v_batch.items():
                            if isinstance(v, torch.Tensor):
                                v_batch[k] = v.to(device)
                        bv = v_batch["input_ids"].shape[0]
                        pos_v = v_batch["intervention_positions"].tolist()
                        _, out = intervenable(
                            {"input_ids": v_batch["input_ids"]},
                            [{"input_ids": v_batch["source_input_ids"]}],
                            {"sources->base": pos_v},
                        )
                        val_iia_sum += compute_iia(out.logits, v_batch["labels"], last_token_only=True) * bv
                        val_n += bv
                val_iia = val_iia_sum / val_n if val_n else 0.0
                pbar.write(f"  [Val] IIA: {val_iia:.4f} {log_tag}")
                intervenable.model.train()
        train_iia_epoch = epoch_iia_sum / epoch_iia_count if epoch_iia_count else 0.0
        mean_loss_epoch = epoch_loss_sum / epoch_loss_count if epoch_loss_count else 0.0
        last_train_iia = train_iia_epoch
        last_mean_loss = mean_loss_epoch
        print(f"Epoch {epoch} train IIA: {train_iia_epoch:.4f} mean_loss: {mean_loss_epoch:.4f} {log_tag}")

    # Test split IIA
    print(f"Evaluating IIA on test split {log_tag}...")
    intervenable.eval()
    test_iia_sum, test_n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            bn = batch["input_ids"].shape[0]
            pos_t = batch["intervention_positions"].tolist()
            _, out = intervenable(
                {"input_ids": batch["input_ids"]},
                [{"input_ids": batch["source_input_ids"]}],
                {"sources->base": pos_t},
            )
            test_iia_sum += compute_iia(out.logits, batch["labels"], last_token_only=True) * bn
            test_n += bn
    test_iia = test_iia_sum / test_n if test_n else 0.0
    print(f"Test IIA: {test_iia:.4f} {log_tag}")
    with open(Path(args.output_dir) / "test_iia.json", "w") as f:
        json.dump(
            {"test_iia": test_iia, "n_test": test_n, "layer": args.layer, "intervention_type": args.intervention_type, "step": args.step},
            f, indent=2,
        )
    intervenable.save(Path(args.output_dir) / "intervention")
    print(f"Saved intervention to {args.output_dir}/intervention {log_tag}")

    # Append configuration and results to CSV (under outputs/)
    n_train = len(train_ds)
    results_csv = Path(args.output_dir).parent / "boundless_das_results.csv"
    row = {
        "layer": args.layer,
        "intervention_type": args.intervention_type,
        "step": step_tag,
        "n_train": n_train,
        "train_iia": last_train_iia if last_train_iia is not None else "",
        "mean_loss": last_mean_loss if last_mean_loss is not None else "",
        "test_iia": test_iia,
        "output_dir": args.output_dir,
    }
    file_exists = results_csv.exists() and results_csv.stat().st_size > 0
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended results to {results_csv} {log_tag}")


if __name__ == "__main__":
    main()
