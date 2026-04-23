"""
GPQA (Graduate-Level Google-Proof Q&A) evaluation utilities.

Extraction pattern from:
https://github.com/openai/simple-evals/blob/90e3e821cabba2aeb6be651dcb662b253df04225/common.py#L25
"""

import re

# Multi-choice answer extraction pattern (case-insensitive, matches A-D)
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Compute score for GPQA multi-choice questions.

    Extracts the answer letter (A-D) from the solution string using
    the pattern 'Answer: X' and compares it to the ground truth.

    Args:
        solution_str: The model's full response string.
        ground_truth: The correct answer letter (A, B, C, or D).

    Returns:
        1.0 if the extracted answer matches ground_truth, 0.0 otherwise.
    """
    match = re.search(ANSWER_PATTERN_MULTICHOICE, solution_str)
    extracted_answer = match.group(1) if match else None
    score = 1.0 if extracted_answer == ground_truth else 0.0
    return score
