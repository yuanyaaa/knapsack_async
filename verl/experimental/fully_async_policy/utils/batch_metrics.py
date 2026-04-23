"""
Batch-level training signal metrics for fully-async training.

Computes:
- Difficulty distribution (based on prompt buffer pass_rate)
- Non-zero-advantage prompt & sample statistics
- Response length statistics split by correct / incorrect
- Overlong ratio per correct / incorrect
"""

from typing import Any

import numpy as np
import torch

from verl import DataProto


def compute_difficulty_distribution(batch: DataProto) -> dict[str, Any]:
    """Compute difficulty distribution from prompt buffer pass_rates.

    Uses per-RolloutSample pass_rates stored in meta_info to classify prompts
    into five performance statuses based on success rate (p_i):
    - extremely_hard: p_i = 0 (all failures)
    - hard: 0 < p_i <= 0.2
    - medium: 0.2 < p_i < 0.8
    - easy: 0.8 <= p_i < 1.0
    - extremely_easy: p_i = 1.0 (all successes)
    - unknown: p_i < 0 (not yet observed)

    Returns:
        Dictionary with keys like "batch_analysis/difficulty/{bucket}" and
        "batch_analysis/difficulty/{bucket}_ratio".
    """
    pass_rates = batch.meta_info.get("pass_rates", [])
    if not pass_rates:
        return {}

    total = len(pass_rates)
    buckets = {
        "extremely_hard": 0,
        "hard": 0,
        "medium": 0,
        "easy": 0,
        "extremely_easy": 0,
        "unknown": 0,
    }

    for pr in pass_rates:
        if pr < 0:
            # -1.0 means not yet observed
            buckets["unknown"] += 1
        elif pr == 0.0:
            buckets["extremely_hard"] += 1
        elif pr <= 0.2:
            buckets["hard"] += 1
        elif pr < 0.8:
            buckets["medium"] += 1
        elif pr < 1.0:
            buckets["easy"] += 1
        else:
            buckets["extremely_easy"] += 1

    metrics: dict[str, Any] = {}
    for bucket_name, count in buckets.items():
        metrics[f"batch_analysis/difficulty/{bucket_name}"] = count
        metrics[f"batch_analysis/difficulty/{bucket_name}_ratio"] = count / total if total > 0 else 0.0

    return metrics


def compute_advantage_signal_metrics(batch: DataProto) -> dict[str, Any]:
    """Compute non-zero-advantage prompt & sample statistics.

    Groups responses by prompt (using ``non_tensor_batch["uid"]``), then checks
    whether each prompt's responses have any non-zero advantage tokens.

    A prompt with all-zero advantages provides no gradient signal — this metric
    tracks how many such "dead" prompts exist in the batch.

    Returns:
        Dictionary with keys:
        - batch_analysis/advantage/total_prompts
        - batch_analysis/advantage/nonzero_adv_prompts
        - batch_analysis/advantage/nonzero_adv_prompt_ratio
        - batch_analysis/advantage/total_samples
        - batch_analysis/advantage/nonzero_adv_samples
        - batch_analysis/advantage/nonzero_adv_sample_ratio
    """
    if "advantages" not in batch.batch:
        return {}

    advantages = batch.batch["advantages"]  # [batch_size, response_length]
    response_mask = batch.batch.get("response_mask", None)

    # Per-sample: check if any advantage token is non-zero (within masked region)
    if response_mask is not None:
        masked_adv = advantages * response_mask
    else:
        masked_adv = advantages

    # Per-sample non-zero check: a sample has non-zero advantage if any token != 0
    # Use a small epsilon to handle floating point
    eps = 1e-8
    sample_has_nonzero_adv = (masked_adv.abs() > eps).any(dim=-1)  # [batch_size]

    total_samples = len(sample_has_nonzero_adv)
    nonzero_adv_samples = int(sample_has_nonzero_adv.sum().item())

    # Per-prompt grouping using uid
    uids = batch.non_tensor_batch.get("uid", None)
    if uids is not None:
        unique_uids = list(set(uids))
        total_prompts = len(unique_uids)
        nonzero_adv_prompts = 0

        uid_to_indices: dict[str, list[int]] = {}
        for i, uid in enumerate(uids):
            uid_to_indices.setdefault(uid, []).append(i)

        for uid, indices in uid_to_indices.items():
            # A prompt has non-zero advantage if ANY of its responses has non-zero advantage
            if any(sample_has_nonzero_adv[i].item() for i in indices):
                nonzero_adv_prompts += 1
    else:
        # Fallback: treat each sample as a separate prompt
        total_prompts = total_samples
        nonzero_adv_prompts = nonzero_adv_samples

    return {
        "batch_analysis/advantage/total_prompts": total_prompts,
        "batch_analysis/advantage/nonzero_adv_prompts": nonzero_adv_prompts,
        "batch_analysis/advantage/nonzero_adv_prompt_ratio": (
            nonzero_adv_prompts / total_prompts if total_prompts > 0 else 0.0
        ),
        "batch_analysis/advantage/total_samples": total_samples,
        "batch_analysis/advantage/nonzero_adv_samples": nonzero_adv_samples,
        "batch_analysis/advantage/nonzero_adv_sample_ratio": (
            nonzero_adv_samples / total_samples if total_samples > 0 else 0.0
        ),
    }


def compute_response_length_by_correctness(batch: DataProto) -> dict[str, Any]:
    """Compute response length statistics split by correct / incorrect.

    Uses ``token_level_scores`` (summed per sequence) to determine correctness:
    - correct: sequence_score > 0
    - incorrect: sequence_score <= 0

    For each group, computes mean/max/min response length and overlong ratio
    (response_length >= max_response_length).

    Returns:
        Dictionary with keys like:
        - batch_analysis/response_length/correct/mean
        - batch_analysis/response_length/correct/overlong_ratio
        - batch_analysis/response_length/incorrect/mean
        - batch_analysis/response_length/incorrect/overlong_ratio
        etc.
    """
    if "token_level_scores" not in batch.batch or "responses" not in batch.batch:
        return {}

    # Compute per-sequence score
    sequence_score = batch.batch["token_level_scores"].sum(-1)  # [batch_size]

    # Compute response lengths
    max_response_length = batch.batch["responses"].shape[-1]
    response_attention = batch.batch["attention_mask"][:, -max_response_length:]
    response_length = response_attention.sum(-1).float()  # [batch_size]

    # Split by correctness
    correct_mask = (sequence_score > 0).bool()
    incorrect_mask = ~correct_mask

    metrics: dict[str, Any] = {}

    # Total counts
    metrics["batch_analysis/response_length/correct_count"] = int(correct_mask.sum().item())
    metrics["batch_analysis/response_length/incorrect_count"] = int(incorrect_mask.sum().item())
    metrics["batch_analysis/response_length/correct_ratio"] = (
        float(correct_mask.float().mean().item()) if len(correct_mask) > 0 else 0.0
    )

    for label, mask in [("correct", correct_mask), ("incorrect", incorrect_mask)]:
        lengths = response_length[mask]
        if lengths.numel() == 0:
            metrics[f"batch_analysis/response_length/{label}/mean"] = 0.0
            metrics[f"batch_analysis/response_length/{label}/max"] = 0.0
            metrics[f"batch_analysis/response_length/{label}/min"] = 0.0
            metrics[f"batch_analysis/response_length/{label}/overlong_count"] = 0
            metrics[f"batch_analysis/response_length/{label}/overlong_ratio"] = 0.0
            continue

        metrics[f"batch_analysis/response_length/{label}/mean"] = float(lengths.mean().item())
        metrics[f"batch_analysis/response_length/{label}/max"] = float(lengths.max().item())
        metrics[f"batch_analysis/response_length/{label}/min"] = float(lengths.min().item())

        # Overlong: response_length >= max_response_length
        overlong_mask = (lengths >= max_response_length).float()
        overlong_count = int(overlong_mask.sum().item())
        metrics[f"batch_analysis/response_length/{label}/overlong_count"] = overlong_count
        metrics[f"batch_analysis/response_length/{label}/overlong_ratio"] = float(overlong_mask.mean().item())

    return metrics


def compute_raw_acc_with_importance_sampling(batch: DataProto) -> dict[str, Any]:
    """Compute raw accuracy (including rejected samples) and importance-sampling-corrected accuracy.

    When rejection sampling is enabled, the training batch only contains accepted
    samples (those with non-zero reward variance). This function uses the full
    reward info (including rejected samples) stored in meta_info to compute:

    1. **raw_acc**: The unfiltered accuracy across ALL samples seen during
       collection (accepted + rejected). This reflects the true model performance
       before rejection sampling.

    2. **unweighted_acc** (only when priority sampling is active): When priority
       sampling assigns non-uniform sampling probabilities, the raw_acc is biased
       towards frequently-sampled prompts. We use importance sampling weights
       (1 / sampling_prob) to correct this bias and estimate the population-level
       accuracy under uniform sampling.

       Specifically:
         unweighted_acc = sum(w_i * acc_i) / sum(w_i)
       where w_i = 1 / (N * p_i), N = number of prompts, p_i = sampling probability.
       Since N is constant, it simplifies to:
         unweighted_acc = sum(acc_i / p_i) / sum(1 / p_i)

    The per-sample reward info is stored as a list of (correct_count, total_count,
    sampling_prob) tuples in ``batch.meta_info["rejection_sampling_reward_info"]``.

    Returns:
        Dictionary with keys:
        - batch_analysis/raw_acc/mean: raw accuracy across all seen samples
        - batch_analysis/raw_acc/total_correct: total correct responses
        - batch_analysis/raw_acc/total_responses: total responses
        - batch_analysis/raw_acc/total_prompts: total prompts seen
        - batch_analysis/unweighted_acc/mean: IS-corrected accuracy (only if priority sampling active)
    """
    reward_info = batch.meta_info.get("rejection_sampling_reward_info", [])
    if not reward_info:
        return {}

    total_correct = 0
    total_responses = 0
    total_prompts = len(reward_info)

    for correct_count, total_count, _ in reward_info:
        total_correct += correct_count
        total_responses += total_count

    metrics: dict[str, Any] = {}
    raw_acc = total_correct / total_responses if total_responses > 0 else 0.0
    metrics["batch_analysis/raw_acc/mean"] = raw_acc
    metrics["batch_analysis/raw_acc/total_correct"] = total_correct
    metrics["batch_analysis/raw_acc/total_responses"] = total_responses
    metrics["batch_analysis/raw_acc/total_prompts"] = total_prompts

    # Check if priority sampling is active (any sampling_prob > 0)
    has_priority_sampling = any(sp > 0 for _, _, sp in reward_info)
    if has_priority_sampling:
        # Importance sampling correction:
        # w_i = 1 / p_i (unnormalized importance weight)
        # unweighted_acc = sum(w_i * acc_i) / sum(w_i)
        # where acc_i = correct_count_i / total_count_i for each prompt
        weighted_acc_sum = 0.0
        weight_sum = 0.0
        for correct_count, total_count, sampling_prob in reward_info:
            if sampling_prob > 0 and total_count > 0:
                w = 1.0 / sampling_prob
                acc_i = correct_count / total_count
                weighted_acc_sum += w * acc_i
                weight_sum += w

        if weight_sum > 0:
            unweighted_acc = weighted_acc_sum / weight_sum
            metrics["batch_analysis/unweighted_acc/mean"] = unweighted_acc

    return metrics


def compute_batch_training_signal_metrics(batch: DataProto) -> dict[str, Any]:
    """Compute all batch-level training signal metrics.

    This is the main entry point that aggregates:
    1. Difficulty distribution (from prompt buffer pass_rates)
    2. Non-zero-advantage prompt & sample statistics
    3. Response length statistics split by correct / incorrect with overlong ratio

    Args:
        batch: DataProto batch after advantage computation.

    Returns:
        Combined dictionary of all metrics.
    """
    metrics: dict[str, Any] = {}
    metrics.update(compute_difficulty_distribution(batch))
    metrics.update(compute_advantage_signal_metrics(batch))
    metrics.update(compute_response_length_by_correctness(batch))
    metrics.update(compute_raw_acc_with_importance_sampling(batch))
    return metrics
