"""
Prepare multiplication datasets for Boundless DAS experiments.
Creates datasets with intervention positions for write-down and carry-over values,
differentiated by step (ones, tens, hundreds).
"""

import json
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer
from multiplication_boundless_das_utils import (
    find_write_down_token_position,
    find_carry_over_token_position,
    compute_carries_from_prompt,
)


def prepare_boundless_das_dataset(
    counterfactual_dataset_path: str,
    output_dir: str = "datasets/boundless_das",
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
):
    """
    Prepare counterfactual dataset for Boundless DAS with intervention positions.
    
    Args:
        counterfactual_dataset_path: Path to counterfactual JSON file
        output_dir: Output directory for processed datasets
        tokenizer_name: HuggingFace tokenizer name
    """
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load counterfactual dataset
    print(f"Loading counterfactual dataset from {counterfactual_dataset_path}")
    with open(counterfactual_dataset_path) as f:
        counterfactual_dataset = json.load(f)
    
    print(f"Loaded {len(counterfactual_dataset)} counterfactual examples")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process datasets by intervention type and step
    for intervention_type in ["write_down", "carry_over"]:
        for step in [0, 1, 2]:  # ones, tens, hundreds
            step_name = ["ones", "tens", "hundreds"][step]
            print(f"\nProcessing {intervention_type} interventions at {step_name} place...")
            
            processed_examples = []
            
            for example in counterfactual_dataset:
                # Filter by step if the example has a step field
                if 'step' in example and example['step'] != step:
                    continue
                
                original = example['original']
                counterfactual = example['counterfactual']
                
                # Compute carries
                orig_carries = compute_carries_from_prompt(
                    original['x'], original['y'], len(original['write_down_values'])
                )
                
                # Find intervention position
                if intervention_type == "write_down":
                    intervention_pos = find_write_down_token_position(
                        tokenizer,
                        original['scratchpad'],
                        step,
                        original['write_down_values'][step]
                    )
                else:  # carry_over
                    intervention_pos = find_carry_over_token_position(
                        tokenizer,
                        original['scratchpad'],
                        step,
                        orig_carries[step]
                    )
                
                if intervention_pos is not None:
                    # Tokenize prompts (combine question and scratchpad)
                    base_prompt = original['question'] + '\n' + original['scratchpad']
                    source_prompt = counterfactual['question'] + '\n' + counterfactual['scratchpad']
                    
                    base_input_ids = tokenizer.encode(
                        base_prompt, 
                        add_special_tokens=True,
                        return_tensors="pt"
                    )[0].tolist()
                    
                    source_input_ids = tokenizer.encode(
                        source_prompt,
                        add_special_tokens=True,
                        return_tensors="pt"
                    )[0].tolist()
                    
                    # Adjust intervention position for question + newline
                    question_tokens = tokenizer.encode(
                        original['question'] + '\n',
                        add_special_tokens=False
                    )
                    intervention_pos += len(question_tokens)
                    
                    # Create labels (shifted by 1 for next token prediction)
                    labels = base_input_ids[1:] + [tokenizer.eos_token_id]
                    
                    # Adjust intervention position for special tokens
                    # Account for BOS token if present
                    if tokenizer.bos_token_id is not None:
                        intervention_pos += 1
                    
                    processed_examples.append({
                        'input_ids': base_input_ids,
                        'source_input_ids': source_input_ids,
                        'labels': labels,
                        'intervention_ids': [intervention_pos],
                        'step': step,
                        'step_name': step_name,
                        'intervention_type': intervention_type,
                        'original': original,
                        'counterfactual': counterfactual,
                    })
            
            # Save dataset
            output_file = output_path / f"multiplication_{intervention_type}_{step_name}.json"
            with open(output_file, 'w') as f:
                json.dump(processed_examples, f, indent=2)
            
            print(f"  Saved {len(processed_examples)} examples to {output_file}")
    
    print(f"\nAll datasets saved to {output_path}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Boundless DAS datasets")
    parser.add_argument(
        "--counterfactual-dataset",
        type=str,
        default="datasets/multiplication_write_down_counterfactual.json",
        help="Path to counterfactual dataset JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/boundless_das",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace tokenizer name"
    )
    
    args = parser.parse_args()
    
    prepare_boundless_das_dataset(
        args.counterfactual_dataset,
        args.output_dir,
        args.tokenizer
    )
