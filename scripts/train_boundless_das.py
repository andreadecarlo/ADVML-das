"""
Train Boundless DAS on the multiplication counterfactual dataset.
Intervene on one token at a time; position from data. Supports layer, step,
and intervention_type (carry / write_down). Evaluates IIA and streams it
during training; tests IIA on the test split at the end.
"""

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

from pyvene import (
    IntervenableModel,
    BoundlessRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene import set_seed, count_parameters


def simple_boundless_das_position_config(model_type, intervention_type: str, layer: int):
    """Config for Boundless DAS at a single layer (position-aligned)."""
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(layer, intervention_type),
        ],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    return config


def compute_iia(logits: torch.Tensor, labels: torch.Tensor, last_token_only: bool = True) -> float:
    """
    Interchange Intervention Accuracy: after intervention, does the model
    predict the counterfactual (source) next token? Labels are counterfactual.
    """
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
    """Load train/val/test from prepared boundless_das JSONs; optional filter by type/step."""
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


class BoundlessDASDataset(Dataset):
    """Dataset of (input_ids, source_input_ids, labels, intervention_position)."""

    def __init__(self, examples: List[Dict], pad_id: int, max_length: Optional[int] = None):
        self.examples = examples
        self.pad_id = pad_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[i]
        input_ids = ex["input_ids"]
        source_input_ids = ex["source_input_ids"]
        labels = ex["labels"]
        pos = ex["intervention_ids"][0]
        if self.max_length:
            input_ids = input_ids[: self.max_length]
            source_input_ids = source_input_ids[: self.max_length]
            labels = labels[: self.max_length]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "source_input_ids": torch.tensor(source_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "intervention_position": int(pos),
        }


def collate_boundless_das(batch: List[Dict], pad_id: int):
    """Pad to max length in batch; stack intervention_position."""
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
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_ex, val_ex, test_ex = load_boundless_das_splits(
        args.data_dir,
        intervention_type=args.intervention_type,
        step=args.step,
    )
    print(f"Train {len(train_ex)} val {len(val_ex)} test {len(test_ex)}")
    if len(train_ex) == 0:
        raise ValueError(
            "No training examples. Run prepare_boundless_das_dataset first with --max-samples (e.g. 10000). "
            "Ensure counterfactual JSON has scratchpad/prompt and step/intervention_type."
        )
    if len(val_ex) == 0 or len(test_ex) == 0:
        raise ValueError(
            "Val or test split is empty (e.g. from very small --max-samples). Use at least ~100 samples so val/test get examples."
        )

    import os
    _prev_hf = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    train_ds = BoundlessDASDataset(train_ex, pad_id, args.max_length)
    val_ds = BoundlessDASDataset(val_ex, pad_id, args.max_length)
    test_ds = BoundlessDASDataset(test_ex, pad_id, args.max_length)
    collate = lambda b: collate_boundless_das(b, pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate)

    # Model (quiet load)
    print(f"Loading model {args.model_name}")
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
    print("Model trainable params: ", count_parameters(intervenable.model))
    print("Intervention trainable params: ", intervenable.count_parameters())

    # Training loop with IIA streaming
    for epoch in trange(args.epochs, desc="Epoch"):
        epoch_iia_sum = 0.0
        epoch_iia_count = 0
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
            loss.backward()
            if (total_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                intervenable.set_zero_grad()
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
                pbar.write(f"  [Val] IIA: {val_iia:.4f}")
                intervenable.model.train()
        train_iia_epoch = epoch_iia_sum / epoch_iia_count if epoch_iia_count else 0.0
        print(f"Epoch {epoch} train IIA: {train_iia_epoch:.4f}")

    # Test split IIA
    print("Evaluating IIA on test split...")
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
    print(f"Test IIA: {test_iia:.4f}")
    with open(Path(args.output_dir) / "test_iia.json", "w") as f:
        json.dump({"test_iia": test_iia, "n_test": test_n}, f, indent=2)
    intervenable.save(Path(args.output_dir) / "intervention")
    print(f"Saved intervention to {args.output_dir}/intervention")


if __name__ == "__main__":
    main()
