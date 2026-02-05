import json
import random
from typing import List, Tuple, Dict
from pathlib import Path

# Helper functions from task_design.ipynb
magnitudes = [
    "ones",
    "tens",
    "hundreds",
    "thousands",
    "ten-thousands",
    "hundred-thousands",
    "millions",
    "ten-millions",
    "hundred-millions",
    "billions",
]


def _say_magnitude(i: int):
    assert i < len(magnitudes), f"magnitude not supported: {i}, max is {len(magnitudes)}"
    return magnitudes[i]


def digits(x: int, size: int = 0, sep=0) -> List[int]:
    """Extract digits from a number, optionally padding to a specific size."""
    n = x
    digits_list = []
    while n > 0:
        digits_list.append(n % 10)
        n //= 10

    while len(digits_list) < size:
        digits_list.append(sep)

    return digits_list


def generate_prompt_multiplication(x: int, y: int):
    """
    Generate a multiplication scratchpad prompt for x * y.
    Works up to 3 digits for x.
    Matches the format: "Multiply 23 by 9 step by step..."
    
    Returns: (prompt, question, expected_answer)
    """
    digits_x = digits(x)
    digits_y = digits(y)
    
    question = f"Multiply {x} by {y} step by step."
    steps = [question]

    dy = digits_y[0]  # Single digit multiplier
    carry_over = 0
    results = []

    # Iterate over x's digits from least to most significant
    # digits() returns digits in reverse order: [ones, tens, hundreds, ...]
    for i, dx in enumerate(digits_x):
        next_step = dx * dy + carry_over
        old_carry = carry_over

        if next_step >= 10:
            residual = next_step % 10
            carry_over = next_step // 10
        else:
            residual = next_step
            carry_over = 0

        if old_carry > 0:
            step = (
                f"Multiply {dy} by the digit in the {_say_magnitude(i)} place of {x}, which is {dx}. "
                f"Add the carryover {old_carry}. We get {next_step}. "
                f"Write down {residual} and carry over {carry_over}."
            )
        else:
            step = (
                f"Multiply {dy} by the digit in the {_say_magnitude(i)} place of {x}, which is {dx}. "
                f"We get {next_step}. "
                f"Write down {residual} and carry over {carry_over}."
            )

        steps.append(step)
        results.insert(0, residual)

    product = x * y
    steps.append(f"We finally get {product}.")

    final_prompt = "\n".join(steps)
    return final_prompt, question, product


def compute_carry(x_digits: List[int], k: int) -> List[int]:
    """
    Compute the carries for a number x_digits (as list of digits) and multiplier k.
    
    Args:
        x_digits: List of digits, least significant first (e.g., [3, 2, 1] for 123)
                  This matches the output of digits() function
        k: Multiplier (single digit)
    
    Returns:
        List of carries for each step, where index 0 = ones place, index 1 = tens place, etc.
    """
    carries = []
    carry = 0
    # Process from least significant (first in list) to most significant (last in list)
    for d in x_digits:
        prod = d * k + carry
        carry = prod // 10
        carries.append(carry)
    return carries


def find_carry_changers_no_propagation(x: int, k: int, max_digits: int = 3):
    """
    Find combinations of x (modifying one digit) and k that change the carry
    at a given step without causing propagation to more significant steps.
    Ensures the counterfactual number has the same number of digits as x.
    
    Args:
        x: The base number (up to max_digits)
        k: The multiplier (single digit)
        max_digits: Maximum number of digits for x (default 3)
    
    Returns:
        dict: each step has a list of dicts with:
            - 'new_number': the full number after change (same number of digits as x)
            - 'k': multiplier used
            - 'new_carry': carry at that step
            - 'old_carry': original carry at that step
            - 'changed_digit_idx': index of changed digit (None if k was changed)
    
    Note: Steps are indexed from 0 (ones place) to max_digits-1 (most significant place)
    """
    # Get the number of digits in x (without padding)
    num_digits_x = len(str(x))
    
    # Convert to digits using digits() function (least significant first: [ones, tens, hundreds, ...])
    x_digits = digits(x, size=max_digits)
    orig_carry = compute_carry(x_digits, k)

    results = {i: [] for i in range(max_digits)}

    # Modify each digit of x
    # x_digits is least-significant-first: [ones, tens, hundreds, ...]
    for idx in range(max_digits):
        for new_digit in range(10):
            if new_digit == x_digits[idx]:
                continue
            x_new_digits = x_digits.copy()
            x_new_digits[idx] = new_digit
            new_carry = compute_carry(x_new_digits, k)
            
            # Only keep if carry at this step changes AND all other steps unchanged
            # step 0 = ones place (least significant), step max_digits-1 = most significant
            for step in range(max_digits):
                if (new_carry[step] != orig_carry[step] and 
                    new_carry[step+1:] == orig_carry[step+1:] and 
                    new_carry[:step] == orig_carry[:step]):
                    # Convert digits back to number
                    x_new_digits_reversed = x_new_digits[::-1]
                    # Remove leading zeros
                    while len(x_new_digits_reversed) > 1 and x_new_digits_reversed[0] == 0:
                        x_new_digits_reversed = x_new_digits_reversed[1:]
                    x_new_number = int(''.join(map(str, x_new_digits_reversed))) if x_new_digits_reversed else 0
                    
                    # Only add if the new number has exactly the same number of digits as x
                    # This ensures the response length stays the same
                    if len(str(x_new_number)) == num_digits_x:
                        results[step].append({
                            'new_number': x_new_number,
                            'k': k,
                            'new_carry': new_carry[step],
                            'old_carry': orig_carry[step],
                            'changed_digit_idx': idx,
                        })

    # Modify k (multiplier) - x stays the same, so digit count is preserved
    for new_k in range(1, 10):  # Skip 0 as multiplier
        if new_k == k:
            continue
        new_carry = compute_carry(x_digits, new_k)
        for step in range(max_digits):
            if (new_carry[step] != orig_carry[step] and 
                new_carry[step+1:] == orig_carry[step+1:] and 
                new_carry[:step] == orig_carry[:step]):
                # x stays the same, so digit count is automatically preserved
                results[step].append({
                    'new_number': x,
                    'k': new_k,
                    'new_carry': new_carry[step],
                    'old_carry': orig_carry[step],
                    'changed_digit_idx': None,  # k was changed, not a digit
                })

    return results


def generate_factual_dataset(max_digits: int = 3, num_samples: int = None):
    """
    Generate a factual dataset of multiplication scratchpads.
    
    Args:
        max_digits: Maximum number of digits for the first number (1-3)
        num_samples: Number of samples to generate. If None, generates all combinations.
    
    Returns:
        List of dicts with 'question', 'prompt', 'expected_answer', 'x', 'y'
    """
    dataset = []
    
    for num_digits in range(1, max_digits + 1):
        start_x = 10 ** (num_digits - 1)
        end_x = 10 ** num_digits
        
        # For each x, try multipliers 1-9
        for x in range(start_x, end_x):
            for y in range(1, 10):
                prompt, question, expected_answer = generate_prompt_multiplication(x, y)
                dataset.append({
                    'question': question,
                    'prompt': prompt,
                    'expected_answer': expected_answer,
                    'x': x,
                    'y': y,
                })
    
    if num_samples and num_samples < len(dataset):
        dataset = random.sample(dataset, num_samples)
    
    return dataset


def generate_counterfactual_dataset(factual_dataset: List[Dict], max_digits: int = 3):
    """
    Generate counterfactual dataset by finding carryover changes without propagation.
    
    Args:
        factual_dataset: List of factual examples
        max_digits: Maximum number of digits (for consistency)
    
    Returns:
        List of counterfactual examples with original and modified prompts
    """
    counterfactual_dataset = []
    
    for example in factual_dataset:
        x = example['x']
        y = example['y']
        
        # Find all carryover changes without propagation
        changes = find_carry_changers_no_propagation(x, y, max_digits=max_digits)
        
        # Create counterfactual examples for each step that has changes
        for step, change_list in changes.items():
            for change in change_list:
                new_x = change['new_number']
                new_y = change['k']
                
                # Generate the modified prompt
                new_prompt, new_question, new_answer = generate_prompt_multiplication(new_x, new_y)
                
                counterfactual_dataset.append({
                    'original': {
                        'question': example['question'],
                        'prompt': example['prompt'],
                        'expected_answer': example['expected_answer'],
                        'x': x,
                        'y': y,
                    },
                    'counterfactual': {
                        'question': new_question,
                        'prompt': new_prompt,
                        'expected_answer': new_answer,
                        'x': new_x,
                        'y': new_y,
                    },
                    'step': step,
                    'old_carry': change['old_carry'],
                    'new_carry': change['new_carry'],
                    'changed_digit_idx': change['changed_digit_idx'],
                })
    
    return counterfactual_dataset


if __name__ == "__main__":
    # Generate factual dataset
    print("Generating factual dataset...")
    factual_dataset = generate_factual_dataset(max_digits=3)
    print(f"Generated {len(factual_dataset)} factual examples")
    
    # Show some samples
    print("\n=== Sample Factual Examples ===")
    for i, example in enumerate(factual_dataset[:5]):
        print(f"\nExample {i+1}:")
        print(f"Question: {example['question']}")
        print(f"Prompt:\n{example['prompt']}")
        print(f"Expected answer: {example['expected_answer']}")
        print("-" * 80)
    
    # Generate counterfactual dataset
    print("\n\nGenerating counterfactual dataset...")
    counterfactual_dataset = generate_counterfactual_dataset(factual_dataset, max_digits=3)
    print(f"Generated {len(counterfactual_dataset)} counterfactual examples")
    
    # Show some counterfactual samples
    print("\n=== Sample Counterfactual Examples ===")
    for i, example in enumerate(counterfactual_dataset[:5]):
        print(f"\nCounterfactual Example {i+1}:")
        print(f"Step: {example['step']}")
        print(f"Carry changed from {example['old_carry']} to {example['new_carry']}")
        print(f"\nOriginal:")
        print(f"  {example['original']['prompt']}")
        print(f"\nCounterfactual:")
        print(f"  {example['counterfactual']['prompt']}")
        print("-" * 80)
    
    # Save datasets
    output_dir = Path("datasets")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "multiplication_factual.json", "w") as f:
        json.dump(factual_dataset, f, indent=2)
    
    with open(output_dir / "multiplication_counterfactual.json", "w") as f:
        json.dump(counterfactual_dataset, f, indent=2)
    
    print(f"\n\nDatasets saved to {output_dir}/")
