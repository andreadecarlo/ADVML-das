import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from pyvene import IntervenableModel

from train_boundless_das import (
    compute_iia,
    compute_resultwise_iia,
    simple_boundless_das_position_config,
    load_boundless_das_hf,
    load_boundless_das_splits,
    BoundlessDASDataset,
    LazyBoundlessDASDataset,
    HFDatasetAdapter,
    collate_boundless_das,
    _HF_DATASETS_AVAILABLE,
)


def run_split_iia(
    intervenable: IntervenableModel,
    dataloader: DataLoader,
    tokenizer,
    device: str,
    iia_resultwise: bool,
    iia_examplewise: bool,
    split_name: str,
) -> float:
    """Evaluate IIA on a given dataloader. We keep this simple and robust:
    batch_size is expected to be 1 so that unit_locations can be a scalar."""
    intervenable.eval()
    iia_sum, n_ex = 0.0, 0
    debug_logged = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=split_name):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            bsz = batch["input_ids"].shape[0]

            # For now, we require batch_size == 1 for robustness.
            if bsz != 1:
                raise ValueError(
                    f"Expected batch_size == 1 for eval, got {bsz}. "
                    f"Run with --batch-size-eval 1."
                )

            interv_pos = int(batch["intervention_positions"][0].item())
            unit_locations = {"sources->base": interv_pos}

            _, cf_out = intervenable(
                {"input_ids": batch["input_ids"]},
                [{"input_ids": batch["source_input_ids"]}],
                unit_locations,
            )
            logits = cf_out.logits
            result_mask = batch.get("result_mask")
            if iia_resultwise and result_mask is not None:
                iia = compute_resultwise_iia(
                    logits,
                    batch["labels"],
                    result_mask,
                    tokenizer,
                )
            else:
                iia = compute_iia(
                    logits,
                    batch["labels"],
                    last_token_only=True,
                    result_mask=result_mask,
                    example_wise=iia_examplewise,
                )
            iia_sum += iia * bsz
            n_ex += bsz

            # Optional: brief debugging for first few batches.
            if debug_logged < 3:
                positions_t = batch["intervention_positions"]
                for ex_idx in range(bsz):
                    interv_pos_i = int(positions_t[ex_idx].item())
                    base_ids = batch["input_ids"][ex_idx].detach().cpu()
                    source_ids = batch["source_input_ids"][ex_idx].detach().cpu()

                    mask_i = None
                    result_indices: List[int] = []
                    if result_mask is not None:
                        mask_i = result_mask[ex_idx].bool()
                        if mask_i.any():
                            idx = mask_i.nonzero(as_tuple=False).view(-1)
                            result_indices = idx.tolist()
                            actual_ids = batch["labels"][ex_idx][idx].detach().cpu()
                            pred_ids = torch.argmax(
                                logits[ex_idx][idx], dim=-1
                            ).detach().cpu()
                        else:
                            mask_i = None

                    if mask_i is None:
                        valid = (batch["labels"][ex_idx] != -100).nonzero(as_tuple=False).view(-1)
                        if valid.numel() > 0:
                            idx = valid
                            result_indices = idx.tolist()
                            actual_ids = batch["labels"][ex_idx][idx].detach().cpu()
                            pred_ids = torch.argmax(
                                logits[ex_idx][idx], dim=-1
                            ).detach().cpu()
                        else:
                            last_idx = torch.tensor([logits.size(1) - 1], device=logits.device)
                            result_indices = [int(last_idx.item())]
                            actual_ids = batch["labels"][ex_idx][last_idx].detach().cpu()
                            pred_ids = torch.argmax(
                                logits[ex_idx][last_idx], dim=-1
                            ).detach().cpu()

                    base_text = tokenizer.decode(base_ids, skip_special_tokens=True)
                    source_text = tokenizer.decode(source_ids, skip_special_tokens=True)
                    actual_text = tokenizer.decode(actual_ids, skip_special_tokens=True)
                    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

                    print(
                        f"[DEBUG BoundlessDAS eval {split_name}] example={ex_idx} "
                        f"interv_pos={interv_pos_i} result_indices={result_indices}"
                    )
                    print("Base:", base_text[:256])
                    print("Source:", source_text[:256])
                    print("Labels:", actual_text)
                    print("Pred  :", pred_text)
                debug_logged += 1

            del logits, cf_out
            if device == "cuda":
                torch.cuda.empty_cache()

    return iia_sum / n_ex if n_ex else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Boundless DAS interventions (multiplication dataset).")
    parser.add_argument("--data-dir", type=str, default="datasets/boundless_das")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-7B")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--intervention-type", type=str, choices=["carry_over", "write_down"], default=None)
    parser.add_argument(
        "--load-dir",
        type=str,
        required=True,
        help="Directory containing a saved intervention (e.g. outputs/boundless_das_layer15_carry_over_step0/intervention).",
    )
    parser.add_argument(
        "--batch-size-eval",
        type=int,
        default=1,
        help="Evaluation batch size. Must be 1 for now (per-example unit locations).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional max sequence length; should match training if set.",
    )
    parser.add_argument(
        "--use-hf-dataset",
        action="store_true",
        default=True,
        help="Use HuggingFace datasets when JSONL files are available.",
    )
    parser.add_argument("--no-use-hf-dataset", action="store_false", dest="use_hf_dataset")
    parser.add_argument(
        "--iia-examplewise",
        action="store_true",
        help="If set, compute example-wise IIA over result_mask positions.",
    )
    parser.add_argument(
        "--iia-resultwise",
        action="store_true",
        help="If set, compute numeric-result-wise IIA over result_mask positions.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["val", "test"],
        choices=["val", "test"],
        help="Which splits to evaluate on.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write a JSON summary of IIA scores.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to append a CSV line of IIA scores (with header if file doesn't exist).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.batch_size_eval != 1:
        print(
            f"Overriding --batch-size-eval={args.batch_size_eval} to 1 for robust pyvene evaluation.",
            flush=True,
        )
        args.batch_size_eval = 1

    data_path = Path(args.data_dir)
    jsonl_train = data_path / "boundless_das_train.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    use_hf = args.use_hf_dataset and _HF_DATASETS_AVAILABLE and jsonl_train.exists()
    if use_hf:
        _, val_hf, test_hf = load_boundless_das_hf(
            args.data_dir,
            intervention_type=args.intervention_type,
            step=args.step,
        )
        val_ds = HFDatasetAdapter(val_hf, pad_id, args.max_length)
        test_ds = HFDatasetAdapter(test_hf, pad_id, args.max_length)
    elif jsonl_train.exists():
        val_ds = LazyBoundlessDASDataset(
            data_path / "boundless_das_val.jsonl",
            pad_id,
            args.max_length,
            intervention_type=args.intervention_type,
            step=args.step,
        )
        test_ds = LazyBoundlessDASDataset(
            data_path / "boundless_das_test.jsonl",
            pad_id,
            args.max_length,
            intervention_type=args.intervention_type,
            step=args.step,
        )
    else:
        _, val_ex, test_ex = load_boundless_das_splits(
            args.data_dir,
            intervention_type=args.intervention_type,
            step=args.step,
        )
        val_ds = BoundlessDASDataset(val_ex, pad_id, args.max_length)
        test_ds = BoundlessDASDataset(test_ex, pad_id, args.max_length)

    collate = lambda b: collate_boundless_das(b, pad_id)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size_eval,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size_eval,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Val {len(val_ds)} test {len(test_ds)} examples for evaluation.")

    print(f"Loading model {args.model_name} for evaluation...")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    config = simple_boundless_das_position_config(type(model), "block_output", args.layer)
    intervenable = IntervenableModel(config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()

    load_dir = Path(args.load_dir)
    print(f"Loading intervention from {load_dir} ...")
    intervenable.load_intervention(str(load_dir), include_model=False)

    results: Dict[str, float] = {}
    if "val" in args.splits:
        val_iia = run_split_iia(
            intervenable,
            val_loader,
            tokenizer,
            device,
            iia_resultwise=args.iia_resultwise,
            iia_examplewise=args.iia_examplewise,
            split_name="Val",
        )
        print(f"[Eval] Val IIA: {val_iia:.4f}")
        results["val_iia"] = float(val_iia)
        results["n_val"] = float(len(val_ds))
    if "test" in args.splits:
        test_iia = run_split_iia(
            intervenable,
            test_loader,
            tokenizer,
            device,
            iia_resultwise=args.iia_resultwise,
            iia_examplewise=args.iia_examplewise,
            split_name="Test",
        )
        print(f"[Eval] Test IIA: {test_iia:.4f}")
        results["test_iia"] = float(test_iia)
        results["n_test"] = float(len(test_ds))

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    **results,
                    "layer": args.layer,
                    "intervention_type": args.intervention_type,
                    "step": args.step,
                    "load_dir": args.load_dir,
                },
                f,
                indent=2,
            )
        print(f"Wrote eval results to {out_path}")

    if args.output_csv is not None:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_path.exists()
        with open(out_path, "a") as f:
            if write_header:
                f.write("layer,intervention_type,step,val_iia,test_iia,n_val,n_test,load_dir\n")
            f.write(
                f"{args.layer},{args.intervention_type},{args.step},"
                f"{results.get('val_iia', '')},{results.get('test_iia', '')},"
                f"{results.get('n_val', '')},{results.get('n_test', '')},"
                f"{args.load_dir}\n"
            )
        print(f"Wrote eval results to {out_path}")


if __name__ == "__main__":
    main()

