"""
Utilities for creating Boundless DAS datasets from multiplication scratchpads.
Identifies token positions for "Write down X" and "Carry over Y" interventions.
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from transformers import PreTrainedTokenizer


def _token_position_from_char_offset(
    tokenizer: PreTrainedTokenizer, prompt: str, char_offset: int
) -> Optional[int]:
    """Return token index that contains the character at char_offset (0-based)."""
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offset_mapping = enc.get("offset_mapping")
    if not offset_mapping:
        return None
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_offset < end:
            return i
    return None


def find_write_down_token_position(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    step: int,
    write_down_value: int
) -> Optional[int]:
    """
    Find the token position of the write-down value in the prompt.
    
    Args:
        tokenizer: Tokenizer to use
        prompt: Full prompt text
        step: Step index (0=ones, 1=tens, 2=hundreds)
        write_down_value: The value that was written down at this step
    
    Returns:
        Token position of the write-down value, or None if not found
    """
    # Split prompt into lines. Scratchpad has intro line first ("Multiply X by Y step by step."),
    # then step 0 (ones) on line 1, step 1 (tens) on line 2, etc.
    lines = prompt.split('\n')
    step_line_idx = step + 1
    if step_line_idx >= len(lines):
        return None
    step_line = lines[step_line_idx]
    
    # Find "Write down X" in this line
    write_down_text = f"Write down {write_down_value}"
    if write_down_text not in step_line:
        return None
    
    # Tokenize the full prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    # Tokenize "Write down" and the value separately
    write_down_tokens = tokenizer.encode("Write down", add_special_tokens=False)
    value_tokens = tokenizer.encode(str(write_down_value), add_special_tokens=False)
    
    # Find all occurrences of "Write down" and check which one corresponds to our step
    # We need to count which occurrence this is
    occurrences = []
    for i in range(len(tokens) - len(write_down_tokens) + 1):
        if tokens[i:i+len(write_down_tokens)] == write_down_tokens:
            occurrences.append(i)
    
    # The step-th occurrence corresponds to our step
    if step < len(occurrences):
        write_down_pos = occurrences[step]
        value_start = write_down_pos + len(write_down_tokens)
        
        # Verify the value matches
        if value_start + len(value_tokens) <= len(tokens):
            if tokens[value_start:value_start+len(value_tokens)] == value_tokens:
                return value_start

    # Fallback: find by character offset (robust to tokenizer differences)
    prefix = f"Write down {write_down_value}"
    idx = step_line.find(prefix)
    if idx != -1:
        # Offset of first char of the value (after "Write down ")
        value_char_in_line = idx + len("Write down ")
        line_start = sum(len(line) + 1 for line in lines[:step_line_idx])  # +1 for newline
        char_offset_in_prompt = line_start + value_char_in_line
        return _token_position_from_char_offset(tokenizer, prompt, char_offset_in_prompt)
    return None


def find_carry_over_token_position(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    step: int,
    carry_value: int
) -> Optional[int]:
    """
    Find the token position of the carry-over value in the prompt.
    
    Args:
        tokenizer: Tokenizer to use
        prompt: Full prompt text
        step: Step index (0=ones, 1=tens, 2=hundreds)
        carry_value: The carry-over value at this step
    
    Returns:
        Token position of the carry-over value, or None if not found
    """
    # Split prompt into lines. Scratchpad has intro line first ("Multiply X by Y step by step."),
    # then step 0 (ones) on line 1, step 1 (tens) on line 2, etc.
    lines = prompt.split('\n')
    step_line_idx = step + 1
    if step_line_idx >= len(lines):
        return None
    step_line = lines[step_line_idx]
    
    # Find "carry over X" in this line
    carry_text = f"carry over {carry_value}"
    if carry_text not in step_line:
        return None
    
    # Tokenize the full prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    # Tokenize "carry over" and the value
    carry_over_tokens = tokenizer.encode("carry over", add_special_tokens=False)
    value_tokens = tokenizer.encode(str(carry_value), add_special_tokens=False)
    
    # Find all occurrences of "carry over" and check which one corresponds to our step
    occurrences = []
    for i in range(len(tokens) - len(carry_over_tokens) + 1):
        if tokens[i:i+len(carry_over_tokens)] == carry_over_tokens:
            occurrences.append(i)
    
    # The step-th occurrence corresponds to our step
    if step < len(occurrences):
        carry_over_pos = occurrences[step]
        value_start = carry_over_pos + len(carry_over_tokens)
        
        # Verify the value matches
        if value_start + len(value_tokens) <= len(tokens):
            if tokens[value_start:value_start+len(value_tokens)] == value_tokens:
                return value_start

    # Fallback: find by character offset (robust to tokenizer differences)
    prefix = f"carry over {carry_value}"
    idx = step_line.find(prefix)
    if idx != -1:
        value_char_offset = idx + len("carry over ")
        line_start = sum(len(line) + 1 for line in lines[:step_line_idx])
        char_offset_in_prompt = line_start + value_char_offset
        return _token_position_from_char_offset(tokenizer, prompt, char_offset_in_prompt)
    return None


def find_all_intervention_positions(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    write_down_values: List[int],
    carries: List[int],
    intervention_type: str = "write_down"
) -> Dict[int, int]:
    """
    Find all intervention positions for a given prompt.
    
    Args:
        tokenizer: Tokenizer to use
        prompt: Full prompt text
        write_down_values: List of write-down values for each step
        carries: List of carry values for each step
        intervention_type: "write_down" or "carry_over"
    
    Returns:
        Dictionary mapping step index to token position
    """
    positions = {}
    
    for step in range(len(write_down_values)):
        if intervention_type == "write_down":
            pos = find_write_down_token_position(
                tokenizer, prompt, step, write_down_values[step]
            )
        else:  # carry_over
            pos = find_carry_over_token_position(
                tokenizer, prompt, step, carries[step]
            )
        
        if pos is not None:
            positions[step] = pos
    
    return positions


def compute_carries_from_prompt(x: int, y: int, num_digits: int) -> List[int]:
    """
    Compute carry values for each step.
    
    Args:
        x: First number
        y: Second number (single digit)
        num_digits: Number of digits in x
    
    Returns:
        List of carry values for each step
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_multiplication_dataset import digits, compute_carry
    
    x_digits = digits(x, size=num_digits)
    carries = compute_carry(x_digits, y)
    return carries


def write_down_alignment_example_sampler(
    tokenizer: PreTrainedTokenizer,
    counterfactual_dataset: List[Dict],
    step: Optional[int] = None
):
    """
    Sample function for write-down alignment examples.
    Similar to lower_bound_alignment_example_sampler but for write-down values.
    
    Args:
        tokenizer: Tokenizer to use
        counterfactual_dataset: List of counterfactual examples
        step: Optional step filter (0=ones, 1=tens, 2=hundreds). If None, uses all steps.
    
    Yields:
        Tuples of (base_input_ids, source_input_ids, labels, intervention_ids)
    """
    # Filter by step if specified
    if step is not None:
        filtered_dataset = [ex for ex in counterfactual_dataset if ex.get('step') == step]
    else:
        filtered_dataset = counterfactual_dataset
    
    for example in filtered_dataset:
        original = example['original']
        counterfactual = example['counterfactual']
        
        # Get write-down values
        orig_write_down = original['write_down_values']
        
        # Tokenize prompts (with special tokens for consistency)
        # Combine question and scratchpad
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
        
        # Create labels (shifted by 1 for next token prediction)
        labels = base_input_ids[1:] + [tokenizer.eos_token_id if tokenizer.eos_token_id else base_input_ids[-1]]
        
        # Find intervention positions for the specific step
        step_idx = example.get('step', 0)
        # Use scratchpad for finding positions (it contains the actual computation)
        intervention_pos = find_write_down_token_position(
            tokenizer,
            original['scratchpad'],
            step_idx,
            orig_write_down[step_idx]
        )
        
        if intervention_pos is not None:
            # Adjust for question + newline + special tokens
            question_tokens = tokenizer.encode(
                original['question'] + '\n',
                add_special_tokens=False
            )
            intervention_pos += len(question_tokens)
            if tokenizer.bos_token_id is not None:
                intervention_pos += 1
            # intervention_ids marks where to intervene
            intervention_ids = [intervention_pos]
            yield base_input_ids, source_input_ids, labels, intervention_ids


def carry_over_alignment_example_sampler(
    tokenizer: PreTrainedTokenizer,
    counterfactual_dataset: List[Dict],
    step: Optional[int] = None
):
    """
    Sample function for carry-over alignment examples.
    
    Args:
        tokenizer: Tokenizer to use
        counterfactual_dataset: List of counterfactual examples
        step: Optional step filter (0=ones, 1=tens, 2=hundreds). If None, uses all steps.
    
    Yields:
        Tuples of (base_input_ids, source_input_ids, labels, intervention_ids)
    """
    # Filter by step if specified
    if step is not None:
        filtered_dataset = [ex for ex in counterfactual_dataset if ex.get('step') == step]
    else:
        filtered_dataset = counterfactual_dataset
    
    for example in filtered_dataset:
        original = example['original']
        counterfactual = example['counterfactual']
        
        # Compute carries
        orig_carries = compute_carries_from_prompt(
            original['x'], original['y'], len(original['write_down_values'])
        )
        
        # Tokenize prompts (with special tokens for consistency)
        # Combine question and scratchpad
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
        
        # Create labels (shifted by 1 for next token prediction)
        labels = base_input_ids[1:] + [tokenizer.eos_token_id if tokenizer.eos_token_id else base_input_ids[-1]]
        
        # Find intervention positions for the specific step
        step_idx = example.get('step', 0)
        # Use scratchpad for finding positions (it contains the actual computation)
        intervention_pos = find_carry_over_token_position(
            tokenizer,
            original['scratchpad'],
            step_idx,
            orig_carries[step_idx]
        )
        
        if intervention_pos is not None:
            # Adjust for question + newline + special tokens
            question_tokens = tokenizer.encode(
                original['question'] + '\n',
                add_special_tokens=False
            )
            intervention_pos += len(question_tokens)
            if tokenizer.bos_token_id is not None:
                intervention_pos += 1
            intervention_ids = [intervention_pos]
            yield base_input_ids, source_input_ids, labels, intervention_ids


def bound_alignment_sampler(
    tokenizer: PreTrainedTokenizer,
    num_samples: int,
    example_samplers: List,
    max_attempts: int = 100000
):
    """
    Sample alignment examples using multiple samplers.
    Similar to pyvene's bound_alignment_sampler but for multiplication.
    
    Args:
        tokenizer: Tokenizer to use
        num_samples: Number of samples to generate
        example_samplers: List of sampler functions
        max_attempts: Maximum attempts to generate samples
    
    Returns:
        Tuple of (input_ids_list, source_input_ids_list, labels_list, intervention_ids_list)
    """
    input_ids_list = []
    source_input_ids_list = []
    labels_list = []
    intervention_ids_list = []
    
    attempts = 0
    while len(input_ids_list) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Randomly select a sampler
        sampler = random.choice(example_samplers)
        
        try:
            for base_ids, source_ids, labels, interv_ids in sampler:
                input_ids_list.append(base_ids)
                source_input_ids_list.append(source_ids)
                labels_list.append(labels)
                intervention_ids_list.append(interv_ids)
                
                if len(input_ids_list) >= num_samples:
                    break
        except StopIteration:
            continue
    
    return (
        input_ids_list[:num_samples],
        source_input_ids_list[:num_samples],
        labels_list[:num_samples],
        intervention_ids_list[:num_samples]
    )
