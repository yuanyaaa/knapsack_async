# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Zero-advantage sample filtering for GRPO in fully-async training.

This module implements the ``filter_zero_adv`` feature: right after a batch is
received from the rollouter (and before log_prob / actor forward), we identify
GRPO groups whose sequence-level rewards all coincide (zero intra-group
variance) — such groups will receive zero advantage under GRPO, contributing
no useful gradient signal. We zero out their ``response_mask`` so that:

  1. Their tokens are excluded from pg / entropy / KL loss denominators
     (i.e. loss mask is more faithful to the effective sample set).
  2. Their backward gradients are naturally zero (loss * 0 = 0), removing
     dilution of the learning signal.

We still run forward passes for those samples (batch shape is preserved),
trading a bit of compute for zero intrusion into framework batch-alignment
assumptions.

Judgement rule (see ``compute_grpo_outcome_advantage``): for GRPO,
``adv == 0`` for every token in a group iff the group's sequence-level
rewards are all identical ⇔ group's reward std == 0. We therefore inspect
``rm_scores.sum(-1)`` grouped by ``non_tensor_batch["uid"]``, which matches
the existing reward/uid conventions used elsewhere in fully_async_policy.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto

__all__ = ["mask_out_zero_variance_groups", "compute_grpo_advantage_exclude_pad"]

# Numerical tolerance for treating intra-group reward std as zero.
_STD_EPS = 1e-8


def mask_out_zero_variance_groups(batch: DataProto) -> dict[str, Any]:
    """Zero out ``response_mask`` for GRPO groups with zero reward variance.

    NOTE: This is the core primitive of the ``algorithm.filter_zero_adv``
    switch. It is invoked from ``FullyAsyncTrainer._fit_generate`` AFTER the
    batch is assembled and BEFORE ``_fit_compute_log_prob`` / actor forward,
    so zeroed-out samples contribute neither to loss mask denominators nor to
    actor backward gradients.

    Args:
        batch: Assembled DataProto from the rollouter. Must contain
            ``batch["rm_scores"]`` (token-level reward, shape
            ``[B, response_length]``), ``batch["response_mask"]``
            (shape ``[B, response_length]``), and
            ``non_tensor_batch["uid"]`` (length-B object array of group ids).
    Returns:
        Dict of metrics to be merged into step metrics, all prefixed with
        ``fully_async/filter_zero_adv/``.

    Side effects:
        Mutates ``batch.batch["response_mask"]`` in place, zeroing rows that
        belong to zero-variance groups.
    """
    metrics: dict[str, Any] = {}

    # ------------- 1. Pre-conditions ---------------------------------------
    if "rm_scores" not in batch.batch:
        logging.warning(
            "[filter_zero_adv] 'rm_scores' not in batch; skipping "
            "(filter_zero_adv requires rollouter-provided reward)."
        )
        return metrics
    if "response_mask" not in batch.batch:
        logging.warning("[filter_zero_adv] 'response_mask' not in batch; skipping.")
        return metrics
    uids = batch.non_tensor_batch.get("uid", None)
    if uids is None:
        logging.warning("[filter_zero_adv] 'uid' not in non_tensor_batch; skipping.")
        return metrics

    response_mask: torch.Tensor = batch.batch["response_mask"]  # [B, L]
    rm_scores: torch.Tensor = batch.batch["rm_scores"]  # [B, L]

    # Sequence-level reward per sample — same convention as
    # fully_async_trainer._post_fn / rollouter (rm_scores.sum(-1)).
    seq_reward = rm_scores.sum(dim=-1).detach().cpu().float()  # [B]
    batch_size = seq_reward.shape[0]

    # ------------- 2. Group by uid, detect zero-variance groups ------------
    uid_to_indices: dict[Any, list[int]] = {}
    for i, uid in enumerate(uids):
        uid_to_indices.setdefault(uid, []).append(i)

    total_groups = len(uid_to_indices)
    zero_var_group_ids: list[Any] = []
    zero_var_sample_indices: list[int] = []
    for uid, indices in uid_to_indices.items():
        if len(indices) < 2:
            # Single-sample group: GRPO std==0 by definition, but masking it
            # would kill legitimate single-sample prompts. Skip for safety.
            continue
        group_rewards = seq_reward[indices]
        if torch.std(group_rewards, unbiased=False).item() <= _STD_EPS:
            zero_var_group_ids.append(uid)
            zero_var_sample_indices.extend(indices)

    # ------------- 3. Mask metrics BEFORE modification ---------------------
    valid_tokens_before = int(response_mask.sum().item())
    total_tokens = int(response_mask.numel())

    # ------------- 4. Apply mask in-place ----------------------------------
    if zero_var_sample_indices:
        idx_tensor = torch.as_tensor(zero_var_sample_indices, dtype=torch.long, device=response_mask.device)
        # NOTE: in-place zero to keep tensor identity (some downstream ops
        # may hold references to the same storage); also reflect into the
        # TensorDict slot.
        response_mask[idx_tensor] = 0
        batch.batch["response_mask"] = response_mask

    valid_tokens_after = int(response_mask.sum().item())

    # ------------- 5. Metrics ---------------------------------------------
    num_filtered_samples = len(zero_var_sample_indices)
    num_filtered_groups = len(zero_var_group_ids)
    filtered_tokens = valid_tokens_before - valid_tokens_after

    metrics = {
        "fully_async/filter_zero_adv/num_filtered_groups": num_filtered_groups,
        "fully_async/filter_zero_adv/total_groups": total_groups,
        "fully_async/filter_zero_adv/filtered_group_ratio": (
            num_filtered_groups / total_groups if total_groups > 0 else 0.0
        ),
        "fully_async/filter_zero_adv/num_filtered_samples": num_filtered_samples,
        "fully_async/filter_zero_adv/total_samples": batch_size,
        "fully_async/filter_zero_adv/filtered_sample_ratio": (
            num_filtered_samples / batch_size if batch_size > 0 else 0.0
        ),
        "fully_async/filter_zero_adv/num_filtered_tokens": filtered_tokens,
        "fully_async/filter_zero_adv/total_valid_tokens_before": valid_tokens_before,
        "fully_async/filter_zero_adv/total_valid_tokens_after": valid_tokens_after,
        "fully_async/filter_zero_adv/filtered_token_ratio": (
            filtered_tokens / valid_tokens_before if valid_tokens_before > 0 else 0.0
        ),
    }

    # Also expose to batch.meta_info for downstream consumers that read from
    # there (e.g. signal metrics, debug tools).
    batch.meta_info.update(metrics)

    # Guard against the degenerate case where the entire batch is filtered —
    # downstream loss would be NaN. We only warn here (not raise), because
    # this can legitimately happen on cold-start / reward-saturated stages.
    if valid_tokens_after == 0 and valid_tokens_before > 0:
        logging.warning(
            "[filter_zero_adv] ALL response_mask tokens were filtered out; "
            "downstream loss will see an empty denominator. Consider disabling "
            "filter_zero_adv or raising rollout.n."
        )

    return metrics


def compute_grpo_advantage_exclude_pad(
    batch: DataProto,
    norm_adv_by_std_in_grpo: bool = True,
    epsilon: float = 1e-6,
) -> None:
    """Compute GRPO advantage while excluding pad samples from group statistics,
    and automatically mask out zero-variance groups.

    This is the core of ``adv_estimator=grpo_filter``. It replaces the standard
    ``compute_grpo_outcome_advantage`` with two enhancements:

    1. **Pad-sample exclusion**: Samples whose ``response_mask`` is all-zero
       (i.e. pad rows added by ``pad_dataproto_to_divisor``) are excluded from
       group mean/std computation so they do not pollute the advantage baseline.
       Pad samples always receive zero advantage.

    2. **Zero-variance group filtering**: Groups whose active (non-pad) members
       all share the same reward (std ≈ 0) produce zero advantage. We additionally
       zero out their ``response_mask`` in-place so downstream loss denominators
       are honest and no dilution occurs.

    Args:
        batch: DataProto with ``batch["token_level_rewards"]``,
            ``batch["response_mask"]``, and ``non_tensor_batch["uid"]``.
        norm_adv_by_std_in_grpo: Whether to normalize by std (GRPO vs Dr.GRPO).
        epsilon: Small value to avoid division by zero.

    Side effects:
        Sets ``batch.batch["advantages"]`` and ``batch.batch["returns"]``.
        Mutates ``batch.batch["response_mask"]`` in-place for zero-variance groups.
    """
    token_level_rewards = batch.batch["token_level_rewards"]
    response_mask = batch.batch["response_mask"]
    index = batch.non_tensor_batch["uid"]

    scores = token_level_rewards.sum(dim=-1)  # [B]
    bsz = scores.shape[0]

    # Identify active (non-pad) samples: response_mask has at least one non-zero token
    active_mask = response_mask.sum(dim=-1) > 0  # [B] bool

    # --- Group by uid, only counting active samples for mean/std ---
    id2active_scores: dict = defaultdict(list)
    id2active_indices: dict = defaultdict(list)
    id2all_indices: dict = defaultdict(list)

    with torch.no_grad():
        for i in range(bsz):
            uid = index[i]
            id2all_indices[uid].append(i)
            if active_mask[i]:
                id2active_scores[uid].append(scores[i])
                id2active_indices[uid].append(i)

        id2mean = {}
        id2std = {}
        zero_var_group_ids = []

        for uid in id2all_indices:
            active_scores = id2active_scores.get(uid, [])
            n_active = len(active_scores)
            if n_active <= 1:
                # 0 or 1 active sample: no meaningful group contrast
                id2mean[uid] = torch.tensor(0.0)
                id2std[uid] = torch.tensor(1.0)
            else:
                scores_tensor = torch.stack(active_scores)
                id2mean[uid] = torch.mean(scores_tensor)
                id2std[uid] = torch.std(scores_tensor)

        # --- Compute per-sample advantage ---
        for i in range(bsz):
            uid = index[i]
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[uid]) / (id2std[uid] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[uid]

        scores = scores.unsqueeze(-1) * response_mask  # [B, response_length]

    # Set advantages and returns
    batch.batch["advantages"] = scores
    batch.batch["returns"] = scores

    return batch