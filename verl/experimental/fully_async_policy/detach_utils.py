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
import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from math import gcd
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.ray_trainer import compute_response_mask


@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: Any

    # Metadata
    sample_id: str
    epoch: int

    # Processing metadata
    rollout_status: dict[str, Any]

    # Priority sampling metadata
    prompt_uid: str = ""
    sampling_prob: float = 0.0
    rollout_n: int = 1

    # Prompt buffer tracking metadata (populated after rollout completes)
    pass_rate: float = -1.0  # -1.0 means not yet observed
    sample_count: int = 0

    # Per-prompt dynamic rollout diagnostics.
    positive_count: int = -1
    stop_reason: str = ""
    raw_rollout_n: int = -1
    raw_positive_count: int = -1
    pos_neg_target_positive: int = -1
    pos_neg_target_negative: int = -1
    pos_neg_ratio_applied: bool = False
    compensation_applied: bool = False
    compensation_samples: int = 0
    compensation_saved_budget: int = 0
    compensation_base_n: int = -1
    compensation_initial_n: int = -1
    compensation_roi: float = 0.0
    compensation_success_prob: float = 0.0
    compensation_state: str = ""


@dataclass
class ValidateMetrics:
    """Metrics for validation"""

    timing_raw: dict[str, Any]
    metrics: Optional[dict[str, Any]] = None


def prepare_single_generation_data(batch_dict, config, repeat_times: Optional[int] = None) -> DataProto:
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Args:
        batch_dict: Single sample dict from dataset/collate_fn.
        config: Configuration object.
        repeat_times: Number of times to repeat the sample. If None, uses config.actor_rollout_ref.rollout.n.

    Returns:
        DataProto: Prepared generation data with repeat.
    """

    full_batch = DataProto.from_single_dict(batch_dict)

    batch_keys_to_pop = []
    non_tensor_batch_keys_to_pop = []

    existing_batch_keys = [k for k in batch_keys_to_pop if k in full_batch.batch.keys()]
    existing_non_tensor_keys = [k for k in non_tensor_batch_keys_to_pop if k in full_batch.non_tensor_batch.keys()]

    if existing_batch_keys or existing_non_tensor_keys:
        full_batch.pop(
            batch_keys=existing_batch_keys,
            non_tensor_batch_keys=existing_non_tensor_keys,
        )

    # Setting selected agent, that supports partial
    if config.actor_rollout_ref.rollout.multi_turn.enable:
        full_batch.non_tensor_batch["agent_name"] = np.array(["tool_agent"] * len(full_batch), dtype=object)
    else:
        full_batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(full_batch), dtype=object)

    # Add global step count to generated data
    n = repeat_times if repeat_times is not None else config.actor_rollout_ref.rollout.n
    full_batch = full_batch.repeat(repeat_times=n, interleave=True)
    return full_batch


def addition_process(output: DataProto):
    """collect metrics.

    ``meta_info["metrics"]`` has two possible shapes depending on the upstream path:
    1. ``List[Dict[str, Any]]`` — produced directly by ``AgentLoopWorker._postprocess``
       for a single ``generate_sequences`` call (one dict per sequence).
    2. ``Dict[str, List[Any]]`` — produced by ``DataProto.concat`` which flattens the
       above list-of-dicts into a dict-of-lists (see ``verl/protocol.py``). This is
       the shape we get in the per-prompt aggregation path inside
       ``FullyAsyncRollouter._emit_prompt_as_rollout_sample``.
    """
    metrics = output.meta_info.pop("metrics")
    if isinstance(metrics, dict):
        # Already flattened to dict-of-lists by DataProto.concat.
        processing_times_list = list(metrics.get("generate_sequences", []))
        tool_calls_times_list = list(metrics.get("tool_calls", []))
    else:
        # Original list-of-dicts (un-concatenated) shape.
        processing_times_list = [item["generate_sequences"] for item in metrics]
        tool_calls_times_list = [item["tool_calls"] for item in metrics]
    output.non_tensor_batch["processing_times"] = processing_times_list
    output.non_tensor_batch["tool_calls_times"] = tool_calls_times_list
    return output


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None, actor_world_size: int = 1
) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_samples: List of RolloutSample objects
        tokenizer: Tokenizer instance
        config: Configuration object containing trainer settings
        balance_batch: Whether to balance the batch (simplified version)
        actor_world_size: Actor world size used to compute padding divisor
            ``lcm(ppo_mini_batch_size, actor_world_size)``.

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    print(f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects")

    rollout_samples_batch = []
    rollout_status = rollout_samples[0].rollout_status
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        batch = addition_process(rs.full_batch)
        rollout_samples_batch.append(batch)
    final_batch = DataProto.concat(rollout_samples_batch)

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    # Pad assembled batch to satisfy actor update mini-batch divisibility.
    # Actor side uses `ppo_mini_batch_size * rollout.n` (sequence-level) as the
    # global mini-batch size, so we align with that and world size together.
    # This must happen before optional balance_batch; otherwise variable-size
    # fixed_samples batches may fail early on divisibility assertions.
    mbs = int(config.actor_rollout_ref.actor.ppo_mini_batch_size)
    rollout_n = int(config.actor_rollout_ref.rollout.n)
    world_size = int(actor_world_size or 1)
    mbs_seq = mbs * rollout_n if mbs > 0 and rollout_n > 0 else max(mbs, 1)
    divisor = (
        mbs_seq * world_size // gcd(mbs_seq, world_size)
        if mbs_seq > 0 and world_size > 0
        else max(mbs_seq, 1)
    )
    pre_pad_len = len(final_batch)
    if divisor > 1 and pre_pad_len % divisor != 0:
        _saved_meta = {}
        for k in list(final_batch.meta_info.keys()):
            v = final_batch.meta_info[k]
            if isinstance(v, (list, np.ndarray)):
                _saved_meta[k] = final_batch.meta_info.pop(k)
        final_batch, pad_size = pad_dataproto_to_divisor(final_batch, divisor)
        final_batch.meta_info.update(_saved_meta)
        if pad_size > 0 and "response_mask" in final_batch.batch.keys():
            response_mask = final_batch.batch["response_mask"]
            response_mask[-pad_size:] = torch.zeros_like(response_mask[-pad_size:])
            final_batch.batch["response_mask"] = response_mask
        final_batch.meta_info["fully_async/pad/pre_pad_sequences"] = pre_pad_len
        final_batch.meta_info["fully_async/pad/pad_size"] = int(pad_size)
        final_batch.meta_info["fully_async/pad/divisor"] = int(divisor)
    else:
        final_batch.meta_info["fully_async/pad/pre_pad_sequences"] = pre_pad_len
        final_batch.meta_info["fully_async/pad/pad_size"] = 0
        final_batch.meta_info["fully_async/pad/divisor"] = int(divisor)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    processing_times = final_batch.non_tensor_batch["processing_times"]
    tool_calls = final_batch.non_tensor_batch["tool_calls_times"]
    # Collect statistics
    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    tool_calls_stats = {}
    if len(tool_calls) > 0:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
        }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    param_version_start = final_batch.non_tensor_batch["min_global_steps"]
    param_version_end = final_batch.non_tensor_batch["max_global_steps"]
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # Collect per-sample prompt buffer tracking info
    prompt_uids = [rs.prompt_uid for rs in rollout_samples]
    sampling_probs = [rs.sampling_prob for rs in rollout_samples]
    pass_rates = [rs.pass_rate for rs in rollout_samples]
    sample_counts = [rs.sample_count for rs in rollout_samples]
    rollout_ns = [rs.rollout_n for rs in rollout_samples]
    positive_counts = [rs.positive_count for rs in rollout_samples]
    stop_reasons = [rs.stop_reason for rs in rollout_samples]
    raw_rollout_ns = [rs.raw_rollout_n for rs in rollout_samples if rs.raw_rollout_n >= 0]
    raw_positive_counts = [rs.raw_positive_count for rs in rollout_samples if rs.raw_positive_count >= 0]
    pos_neg_applied = [rs for rs in rollout_samples if rs.pos_neg_ratio_applied]
    compensated_samples = [rs for rs in rollout_samples if rs.compensation_applied]
    saved_budget_samples = [rs for rs in rollout_samples if rs.compensation_saved_budget > 0]
    posneg_flags = [
        bool(pc >= 0 and pc > 0 and (rn - pc) > 0)
        for pc, rn in zip(positive_counts, rollout_ns, strict=False)
    ]
    posneg_prompt_count = sum(1 for flag in posneg_flags if flag)
    posneg_sample_count = sum(
        int(rn)
        for pc, rn in zip(positive_counts, rollout_ns, strict=False)
        if pc >= 0 and pc > 0 and (rn - pc) > 0
    )

    # Compute aggregate statistics for prompt buffer tracking
    observed_pass_rates = [pr for pr in pass_rates if pr >= 0]
    prompt_buffer_stats = {}
    if observed_pass_rates:
        prompt_buffer_stats = {
            "fully_async/prompt_buffer/pass_rate_mean": float(np.mean(observed_pass_rates)),
            "fully_async/prompt_buffer/pass_rate_std": float(np.std(observed_pass_rates)),
            "fully_async/prompt_buffer/pass_rate_min": float(np.min(observed_pass_rates)),
            "fully_async/prompt_buffer/pass_rate_max": float(np.max(observed_pass_rates)),
            "fully_async/prompt_buffer/sample_count_mean": float(np.mean(sample_counts)),
            "fully_async/prompt_buffer/sample_count_max": float(np.max(sample_counts)),
            "fully_async/prompt_buffer/sampling_prob_mean": float(np.mean(sampling_probs)),
            "fully_async/prompt_buffer/sampling_prob_std": float(np.std(sampling_probs)),
            "fully_async/prompt_buffer/sampling_prob_max": float(np.max(sampling_probs)),
            "fully_async/prompt_buffer/sampling_prob_min": float(np.min(sampling_probs)),
            "fully_async/prompt_buffer/rollout_n_mean": float(np.mean(rollout_ns)),
            "fully_async/prompt_buffer/unique_prompt_uids": len(set(prompt_uids)),
        }

    has_k_stats = {}
    if rollout_ns:
        valid_positive_counts = [pc for pc in positive_counts if pc >= 0]
        total_prompts = len(rollout_ns)
        total_samples = int(sum(rollout_ns))
        reached_k = sum(1 for reason in stop_reasons if reason == "reached_k")
        hit_max = sum(1 for reason in stop_reasons if reason == "hit_max")
        reached_ratio = sum(1 for reason in stop_reasons if reason == "reached_ratio")
        zero_positive = sum(1 for pc in valid_positive_counts if pc == 0)
        has_k_stats = {
            "fully_async/has_k/batch_prompt_count": total_prompts,
            "fully_async/has_k/batch_sample_count": total_samples,
            "fully_async/has_k/actual_rollout_n_mean": float(np.mean(rollout_ns)),
            "fully_async/has_k/actual_rollout_n_p90": float(np.percentile(rollout_ns, 90)),
            "fully_async/has_k/actual_rollout_n_max": int(max(rollout_ns)),
            "fully_async/has_k/reached_k_ratio": reached_k / total_prompts if total_prompts else 0.0,
            "fully_async/has_k/reached_ratio_ratio": reached_ratio / total_prompts if total_prompts else 0.0,
            "fully_async/has_k/hit_max_ratio": hit_max / total_prompts if total_prompts else 0.0,
            "fully_async/posneg/prompts_per_step": float(posneg_prompt_count),
            "fully_async/posneg/samples_per_step": float(posneg_sample_count),
            "fully_async/posneg/prompt_ratio": (
                float(posneg_prompt_count) / float(total_prompts) if total_prompts else 0.0
            ),
            "fully_async/posneg/sample_ratio": (
                float(posneg_sample_count) / float(total_samples) if total_samples else 0.0
            ),
        }
        if valid_positive_counts:
            has_k_stats.update(
                {
                    "fully_async/has_k/positive_count_mean": float(np.mean(valid_positive_counts)),
                    "fully_async/has_k/zero_positive_ratio": (
                        zero_positive / len(valid_positive_counts) if valid_positive_counts else 0.0
                    ),
                }
            )
        if pos_neg_applied:
            selected_negative_counts = [
                rs.rollout_n - rs.positive_count for rs in pos_neg_applied if rs.positive_count >= 0
            ]
            pos_neg_stats = {
                "fully_async/pos_neg_ratio/batch_applied_prompt_count": len(pos_neg_applied),
                "fully_async/pos_neg_ratio/batch_applied_ratio": (
                    len(pos_neg_applied) / total_prompts if total_prompts else 0.0
                ),
                "fully_async/pos_neg_ratio/target_positive": float(pos_neg_applied[0].pos_neg_target_positive),
                "fully_async/pos_neg_ratio/target_negative": float(pos_neg_applied[0].pos_neg_target_negative),
                "fully_async/pos_neg_ratio/selected_positive_count_mean": float(
                    np.mean([rs.positive_count for rs in pos_neg_applied])
                ),
                "fully_async/pos_neg_ratio/selected_negative_count_mean": float(
                    np.mean(selected_negative_counts)
                ),
            }
            if raw_rollout_ns:
                pos_neg_stats.update(
                    {
                        "fully_async/pos_neg_ratio/raw_rollout_n_mean": float(np.mean(raw_rollout_ns)),
                        "fully_async/pos_neg_ratio/raw_rollout_n_max": int(max(raw_rollout_ns)),
                    }
                )
            if raw_positive_counts:
                pos_neg_stats["fully_async/pos_neg_ratio/raw_positive_count_mean"] = float(
                    np.mean(raw_positive_counts)
                )
            has_k_stats.update(pos_neg_stats)

        compensation_prompt_count = len(compensated_samples)
        compensation_sample_count = int(sum(rs.compensation_samples for rs in compensated_samples))
        compensation_success_count = sum(
            1
            for rs in compensated_samples
            if rs.positive_count > 0 and (rs.rollout_n - rs.positive_count) > 0
        )
        compensation_accs = [
            float(rs.positive_count) / float(rs.rollout_n)
            for rs in compensated_samples
            if rs.rollout_n > 0 and rs.positive_count >= 0
        ]
        compensation_rois = [float(rs.compensation_roi) for rs in compensated_samples]
        compensation_stats = {
            "fully_async/compensation/bank_deposited_prompts_per_step": float(len(saved_budget_samples)),
            "fully_async/compensation/bank_deposited_samples_per_step": float(
                sum(rs.compensation_saved_budget for rs in saved_budget_samples)
            ),
            "fully_async/compensation/prompts_per_step": float(compensation_prompt_count),
            "fully_async/compensation/samples_per_step": float(compensation_sample_count),
            "fully_async/compensation/prompt_ratio": (
                compensation_prompt_count / total_prompts if total_prompts else 0.0
            ),
            "fully_async/compensation/success_ratio_per_step": (
                compensation_success_count / compensation_prompt_count if compensation_prompt_count else 0.0
            ),
            "fully_async/compensation/final_acc_mean_per_step": (
                float(np.mean(compensation_accs)) if compensation_accs else 0.0
            ),
            "fully_async/compensation/roi_mean_per_step": (
                float(np.mean(compensation_rois)) if compensation_rois else 0.0
            ),
        }
        has_k_stats.update(compensation_stats)

    # add meta_info
    prompt_start_versions = final_batch.non_tensor_batch.get(
        "prompt_start_version",
        final_batch.non_tensor_batch["min_global_steps"],
    )
    response_end_versions = final_batch.non_tensor_batch.get(
        "response_end_version",
        final_batch.non_tensor_batch["max_global_steps"],
    )
    trajectory_param_versions = response_end_versions

    final_batch.meta_info.update(
        {
            "param_version_diversity": len(set(trajectory_param_versions)),
            "trajectory_param_versions": trajectory_param_versions,
            "prompt_start_versions": prompt_start_versions,
            "response_end_versions": response_end_versions,
            # Per-sample prompt buffer tracking (list-level, for detailed inspection)
            "prompt_uids": prompt_uids,
            "sampling_probs": sampling_probs,
            "pass_rates": pass_rates,
            "sample_counts": sample_counts,
            "rollout_ns": rollout_ns,
            "positive_counts": positive_counts,
            "stop_reasons": stop_reasons,
            "posneg_flags": posneg_flags,
            "compensation_applied": [rs.compensation_applied for rs in rollout_samples],
            "compensation_samples": [rs.compensation_samples for rs in rollout_samples],
            "compensation_saved_budget": [rs.compensation_saved_budget for rs in rollout_samples],
            "compensation_roi": [rs.compensation_roi for rs in rollout_samples],
            "compensation_success_prob": [rs.compensation_success_prob for rs in rollout_samples],
            "compensation_state": [rs.compensation_state for rs in rollout_samples],
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **tool_calls_stats,
            **prompt_buffer_stats,
            **has_k_stats,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch

def compute_staleness_metrics(
    batch: DataProto,
    current_param_version: int,
    prompt_aggregation: str = "max",
) -> dict[str, float]:
    """Compute hytuner-style staleness metrics for a training batch.

    Staleness definition:

    - Prompt staleness = ``current_param_version - prompt_start_version``.
      This matches hytuner's ``current_step - start_version``.

    - Response staleness = ``current_param_version - response_end_version``.
      This matches hytuner's ``current_step - end_version`` and measures how
      long a finished response waited before being consumed for training.

    The old policy-staleness signal is still reported separately as
    ``policy_response_staleness_*`` using ``min_global_steps`` so we can keep
    diagnosing off-policy drift without overloading ``response_staleness``.

    Reported metrics:

    - ``fully_async/staleness/response_staleness_mean``: mean over all training
      samples (responses).
    - ``fully_async/staleness/response_staleness_max``: max over all training
      samples (responses).
    - ``fully_async/staleness/prompt_staleness_mean``: mean over all prompts in
      the mini-batch (after per-prompt aggregation).
    - ``fully_async/staleness/prompt_staleness_max``: max over all prompts in
      the mini-batch (after per-prompt aggregation).

    Args:
        batch: Assembled ``DataProto``. Preferably contains per-response
            ``prompt_start_version`` and ``response_end_version``. Falls back
            to ``min_global_steps`` / ``max_global_steps`` for older samples.
        current_param_version: The policy version of the trainer at the moment
            this batch is consumed for training.
        prompt_aggregation: How to aggregate response-level staleness into
            prompt-level staleness. One of ``"max"`` (default) or ``"mean"``.

    Returns:
        Dict of staleness metrics. Empty dict if required fields are missing.
    """
    if prompt_aggregation not in ("max", "mean"):
        raise ValueError(
            f"prompt_aggregation must be 'max' or 'mean', got {prompt_aggregation!r}"
        )

    param_version_start = batch.non_tensor_batch["min_global_steps"]
    param_version_end = batch.non_tensor_batch["max_global_steps"]
    prompt_start_versions = batch.non_tensor_batch.get("prompt_start_version", param_version_start)
    response_end_versions = batch.non_tensor_batch.get("response_end_version", param_version_end)

    # Hytuner-style response staleness: how long completed responses waited
    # before being consumed by the trainer.
    per_response_staleness = [
        float(current_param_version) - float(v) for v in response_end_versions
    ]
    # Keep the old policy staleness under an explicit name.
    per_policy_response_staleness = [
        float(current_param_version) - float(v) for v in param_version_start
    ]
    # How many param updates happened while the sequence was being generated.
    per_response_rollout_span = [
        float(e) - float(s) for s, e in zip(param_version_start, param_version_end, strict=True)
    ]

    metrics: dict[str, float] = {
        "fully_async/staleness/response_staleness_mean": float(np.mean(per_response_staleness)),
        "fully_async/staleness/response_staleness_max": float(np.max(per_response_staleness)),
        "fully_async/staleness/response_staleness_min": float(np.min(per_response_staleness)),
        "fully_async/staleness/policy_response_staleness_mean": float(
            np.mean(per_policy_response_staleness)
        ),
        "fully_async/staleness/policy_response_staleness_max": float(
            np.max(per_policy_response_staleness)
        ),
        "fully_async/staleness/policy_response_staleness_min": float(
            np.min(per_policy_response_staleness)
        ),
        "fully_async/staleness/response_rollout_span_mean": float(np.mean(per_response_rollout_span)),
        "fully_async/staleness/response_rollout_span_max": float(np.max(per_response_rollout_span)),
    }

    # Compute prompt-level staleness using rollout_ns to slice per-response
    # prompt_start_version values. All rows in a prompt should share the same
    # prompt_start_version; aggregating keeps the function robust to older data.
    rollout_ns = batch.meta_info.get("rollout_ns", None) if hasattr(batch, "meta_info") else None
    if rollout_ns is not None and len(rollout_ns) > 0:
        per_prompt_staleness: list[float] = []
        idx = 0
        for n in rollout_ns:
            n = int(n)
            if n <= 0:
                continue
            group = [
                float(current_param_version) - float(v)
                for v in prompt_start_versions[idx : idx + n]
            ]
            idx += n
            if prompt_aggregation == "max":
                per_prompt_staleness.append(float(np.max(group)))
            else:  # "mean"
                per_prompt_staleness.append(float(np.mean(group)))

        if per_prompt_staleness:
            metrics["fully_async/staleness/prompt_staleness_mean"] = float(np.mean(per_prompt_staleness))
            metrics["fully_async/staleness/prompt_staleness_max"] = float(np.max(per_prompt_staleness))
            metrics["fully_async/staleness/prompt_staleness_min"] = float(np.min(per_prompt_staleness))

    return metrics

class MetricsAggregator:
    """Metrics aggregator, used to combine metrics from multiple training steps"""

    def __init__(self, total_gpus: int):
        # Store all values ​​for each metric
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        # Store the number of samples at each step for weighted averaging
        self.sample_counts: list[int] = []
        # Store the timestamp of each step for time-related calculations
        self.timestamps: list[float] = []
        # Step Count
        self.step_count = 0
        # total num gpus used
        self.total_gpus = total_gpus

        # Metric aggregation rule configuration
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, dict[str, list[str]]]:
        """Initialize metrics aggregation rules"""
        return {
            # Time-Based metrics, can add metrics here
            "time_sum": ["perf/time_per_step"],
            "sum": [
                "fully_async/posneg/prompts_per_step",
                "fully_async/posneg/samples_per_step",
            ],
            "min": ["timing_s/agent_loop/tool_calls/min"],
            "avg": ["timing_s/agent_loop/tool_calls/mean"],
            "max": ["timing_s/agent_loop/tool_calls/max"],
            "last": [
                "fully_async/count/total_generated_prompts",
                "fully_async/count/total_generated_samples",
                "fully_async/count/stale_samples",
                "fully_async/count/stale_prompts",
                "fully_async/count/current_param_version",
                "fully_async/count/dropped_prompts",
                "fully_async/count/dropped_samples",
                "fully_async/count/rejected_prompts",
                "fully_async/count/rejected_samples",
                "fully_async/posneg/prompts_cum",
                "fully_async/posneg/samples_cum",
                "fully_async/posneg/prompts_per_sec",
                "fully_async/posneg/samples_per_sec",
                "fully_async/posneg/window_seconds",
                "training/global_step",  # TODO change name to: total_step
            ],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int, timestamp: float = None):
        """Adding a single-step metrics"""
        if timestamp is None:
            timestamp = time.time()

        self.sample_counts.append(sample_count)
        self.timestamps.append(timestamp)
        self.step_count += 1

        # Store all metrics values
        for key, value in metrics.items():
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
            elif isinstance(value, torch.Tensor):
                self.metric_values[key].append(float(value.item()))

    def _get_aggregation_type(self, metric_name: str) -> str:
        """Determine the aggregation type based on the metric name"""
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if any(keyword in metric_lower for keyword in ["timing_s/"]):
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["mean", "avg", "average"]):
            return "avg"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg"]):
            return "weighted_avg"

        return "avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        """Aggregating a single metric"""
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)

        if agg_type == "last":
            return values[-1]

        elif agg_type == "weighted_avg":
            # Weighted average
            if len(values) != len(self.sample_counts):
                # If the lengths do not match, use a simple average
                return sum(values) / len(values)

            total_samples = sum(self.sample_counts)
            if total_samples == 0:
                return sum(values) / len(values)

            weighted_sum = sum(v * c for v, c in zip(values, self.sample_counts, strict=False))
            return weighted_sum / total_samples

        elif agg_type == "sum" or agg_type == "time_sum":
            return sum(values)

        elif agg_type == "avg":
            return sum(values) / len(values)

        elif agg_type == "max":
            return max(values)

        elif agg_type == "min":
            return min(values)

        else:
            # Default average
            return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """aggregated metrics"""
        t = time.time()
        if self.step_count == 0:
            return {}

        aggregated = {}

        # Aggregate all metrics
        for metric_name, values in self.metric_values.items():
            aggregated[metric_name] = self._aggregate_single_metric(metric_name, values)

        # Aggregate special metrics
        aggregated = self._special_metrics_aggergate(aggregated)

        print(f"aggregated metrics done. cost {time.time() - t:.4f} seconds.")

        return aggregated

    def _special_metrics_aggergate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """calculate special metrics"""

        # global_seqlen/minmax_diff
        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        # perf/throughput
        REQUIRED_PERF_KEYS = {"perf/throughput", "perf/total_num_tokens", "perf/time_per_step"}
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        # trainer/idle_ratio
        if "timing_s/gen" in aggregated.keys() and "timing_s/step" in aggregated.keys():
            aggregated["fully_async/trainer/idle_ratio"] = aggregated["timing_s/gen"] / aggregated["timing_s/step"]

        return aggregated

    def reset(self):
        """Reset Aggregator"""
        self.metric_values.clear()
        self.sample_counts.clear()
        self.timestamps.clear()
        self.step_count = 0

    def get_current_stats(self) -> dict[str, Any]:
        """Get statistics about the current aggregation state (for debugging)"""
        return {
            "step_count": self.step_count,
            "metric_count": len(self.metric_values),
            "total_samples": sum(self.sample_counts),
            "metric_names": list(self.metric_values.keys()),
        }


def task_exception_handler(task: asyncio.Task):
    """Handle task exceptions and log them"""
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task was cancelled, this is expected
    except Exception as e:
        print(f"Task {task.get_name()} failed with exception: {e}")
        raise e


def safe_create_task(coro, name: str, task_set: set = None):
    """Safely create a task with exception handling

    Args:
        coro: The coroutine to run
        name: Name for the task
        task_set: Optional set to add the task to

    Returns:
        The created asyncio.Task
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(task_exception_handler)
    if task_set is not None:
        task_set.add(task)
        task.add_done_callback(task_set.discard)
    return task
