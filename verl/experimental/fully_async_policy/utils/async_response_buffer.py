# Copyright 2025 Tencent Ltd. and/or its affiliates
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

"""
AsyncResponseBuffer - Tracks in-flight & finished sequences per prompt_uid for
the asynchronous adaptive rollout scheme.

Responsibilities:
    - Track per-prompt_uid state: in-flight count, finished sequences with their rewards.
    - Provide stop_generation() to decide whether a prompt's rollout should stop.
    - Aggregate all finished sequences of a stopped prompt into a single bundle,
      ready to be wrapped into one RolloutSample for the trainer.

Thread-safety:
    This buffer itself is NOT thread / asyncio safe. It relies on the following
    assumptions for correctness in the fully-async rollouter:
      1. All callers run inside the SAME asyncio event loop (single-threaded).
      2. For a given prompt_uid, mutations (on_response_finished, on_job_submitted,
         stop_generation, mark_if_complete, pop_completed_prompt) are called
         sequentially within one ``_process_one_response`` coroutine — no two
         coroutines ever mutate the same PromptState concurrently.
      3. Different prompt_uids operate on independent PromptState objects; the
         shared ``_states`` dict is only mutated via atomic CPython dict ops
         (protected by the GIL).
    If any of these assumptions are violated (e.g. multi-threaded access, or
    external callers outside the event loop), an asyncio.Lock MUST be added
    around the mutation sequences.

The ``train_trigger`` decision (``fixed_prompt`` vs ``fixed_samples``) is
    NOT owned by this buffer -- it lives on the trainer side, which inspects
    the message queue's pending-sequence count / RolloutSample count directly.
    We keep the ``train_trigger`` argument purely as a config record that the
    rollouter can forward to the trainer; it has no effect on buffer behavior.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from verl.experimental.fully_async_policy.utils.adaptive_debug import (
    CHANNEL_ADAPTIVE_N,
    trace,
)


logger = logging.getLogger(__name__)


# Stop rules for per-prompt_uid rollout
STOP_RULE_FIXED = "fixed_rollout"
STOP_RULE_PREFIXED = "prefixed_rollout"
STOP_RULE_AT_LEAST_POS = "has_at_least_positive"
STOP_RULE_AT_LEAST_POS_NEG = "has_at_least_positive_and_negative"
STOP_RULE_CONTINUE_ON_MIXED_CHUNK = "continue_on_mixed_chunk"
STOP_RULE_ADAPTIVE_CHUNK_MIXED = "adaptive_chunk_mixed"
STOP_RULE_OFFLINE_MEDIUM_FOCUS_COMPENSATE = "offline_medium_focus_compensate"
STOP_RULE_OFFLINE_MEDIUM_FOCUS_BUDGET_COMPENSATE = "offline_medium_focus_budget_compensate"
STOP_RULE_KNAPSACK_VALUE_COMPENSATE = "knapsack_value_compensate"
STOP_RULE_MAX = "max_rollout"

_VALID_STOP_RULES = {
    STOP_RULE_FIXED,
    STOP_RULE_PREFIXED,
    STOP_RULE_AT_LEAST_POS,
    STOP_RULE_AT_LEAST_POS_NEG,
    STOP_RULE_CONTINUE_ON_MIXED_CHUNK,
    STOP_RULE_ADAPTIVE_CHUNK_MIXED,
    STOP_RULE_OFFLINE_MEDIUM_FOCUS_COMPENSATE,
    STOP_RULE_OFFLINE_MEDIUM_FOCUS_BUDGET_COMPENSATE,
    STOP_RULE_KNAPSACK_VALUE_COMPENSATE,
    STOP_RULE_MAX,
}

# Train-trigger rules
TRAIN_TRIGGER_FIXED_PROMPT = "fixed_prompt"
TRAIN_TRIGGER_FIXED_SAMPLES = "fixed_samples"

# How the one-shot compensation size is chosen for the budget-compensate rule.
#   roi_max          largest affordable extra whose ROI still clears the threshold.
#   min_rescue_prob  smallest extra whose probability of producing the missing
#                    class reaches ``compensation_rescue_prob``; give up when the
#                    required extra does not fit under max_rollout_n.
COMPENSATION_RULE_ROI_MAX = "roi_max"
COMPENSATION_RULE_MIN_RESCUE_PROB = "min_rescue_prob"

_VALID_COMPENSATION_RULES = {
    COMPENSATION_RULE_ROI_MAX,
    COMPENSATION_RULE_MIN_RESCUE_PROB,
}


@dataclass
class PromptState:
    """Per-prompt_uid tracking state."""

    prompt_uid: str
    sampling_prob: float = 0.0
    # Number of sequence generation jobs currently in flight (submitted but not yet finished)
    in_flight: int = 0
    # Finished sequences and their rewards (aligned by index)
    sequences: list[Any] = field(default_factory=list)  # list[DataProto of length 1]
    rewards: list[float] = field(default_factory=list)
    # Whether this prompt is marked as stopped (no more new jobs should be issued)
    stopped: bool = False
    # Cached per-prompt target N for STOP_RULE_FIXED. Resolved lazily (once)
    # on the first stop_generation() call and reused for the remainder of this
    # prompt's lifetime, so the target never drifts mid-generation and the
    # dynamic_fixed_n_provider is never called more than once per prompt.
    target_n: Optional[int] = None
    # Policy parameter version active at the moment this prompt was first
    # registered (i.e. its first sequence job was about to be submitted).
    # Used to compute per-prompt staleness =
    #     current_param_version - start_param_version
    # which drives the prompt-level staleness pause in the rollouter.
    start_param_version: int = 0
    # Offline medium-focus compensation diagnostics.
    base_target_n: Optional[int] = None
    initial_target_n: Optional[int] = None
    saved_budget: int = 0
    compensated: bool = False
    compensation_granted: bool = False
    compensation_state: str = ""
    compensation_samples: int = 0
    compensation_roi: float = 0.0
    compensation_success_prob: float = 0.0
    budget_record_id: Optional[int] = None
    # Drawn during the prompt-buffer warmup: fixed N, no budget record, no compensation.
    warmup: bool = False

    @property
    def num_finished(self) -> int:
        return len(self.rewards)

    @property
    def num_positive(self) -> int:
        return sum(1 for r in self.rewards if r > 0)

    @property
    def num_negative(self) -> int:
        return sum(1 for r in self.rewards if r <= 0)

    @property
    def is_complete(self) -> bool:
        """Stopped and no sequence still in flight -> ready to emit."""
        return self.stopped and self.in_flight == 0


class AsyncResponseBuffer:
    """Per-prompt response buffer for async adaptive rollout.

    Args:
        stop_rule: Per-prompt stop rule, one of STOP_RULE_*.
        fixed_rollout_n: Static N used by STOP_RULE_FIXED when no dynamic provider
            is given (stop once finished >= N).
        max_rollout_n: Global cap on # sequences per prompt for all rules.
        min_positive_k: For STOP_RULE_AT_LEAST_POS, stop when finished sequences
            include at least this many positive rewards (reward > 0).
        train_trigger: "fixed_prompt" or "fixed_samples". Recorded here for
            introspection / logging only -- the actual trigger decision lives
            on the trainer side (via the message queue's pending-sequence /
            pending-RolloutSample counters).
        dynamic_fixed_n_provider: Optional callable ``prompt_uid -> int`` that
            returns a per-prompt N for STOP_RULE_FIXED (e.g. a bell-shaped
            function of pass_rate). When provided, overrides ``fixed_rollout_n``
            on a per-prompt basis. Returned values are clamped to [1, max_rollout_n].
            Called AT MOST ONCE per prompt (result cached in ``PromptState.target_n``).
    """

    def __init__(
        self,
        stop_rule: str = STOP_RULE_FIXED,
        fixed_rollout_n: int = 8,
        max_rollout_n: int = 16,
        min_rollout_n: int = 1,
        min_positive_k: int = 1,
        target_positive_count: int = 0,
        target_negative_count: int = 0,
        continue_on_mixed_chunk_size: int = 4,
        adaptive_chunk_alpha: float = 0.5,
        adaptive_mixed_target_prob: float = 0.85,
        adaptive_chunk_prior_tau: float = 8.0,
        adaptive_chunk_target_strategy: str = "medium_focus_plateau",
        medium_focus_peak_n: Optional[int] = None,
        compensation_bank_max: int = 8,
        compensation_chunk_size: int = 1,
        compensation_save_n: int = 4,
        compensation_save_low: float = 0.4,
        compensation_save_high: float = 0.6,
        compensation_roi_threshold: float = 1.5,
        compensation_rule: str = COMPENSATION_RULE_ROI_MAX,
        compensation_rescue_prob: float = 0.85,
        compensation_minority_target: int = 1,
        compensation_estimate_clip: float = 0.02,
        compensation_budget_baseline_n: Optional[int] = None,
        compensation_budget_window_size: int = 0,
        train_trigger: str = TRAIN_TRIGGER_FIXED_PROMPT,
        dynamic_fixed_n_provider: Optional[Callable[[str], int]] = None,
        prompt_pass_rate_provider: Optional[Callable[[str], float]] = None,
        min_rollout_n_provider: Optional[Callable[[], int]] = None,
        warmup_prompt_provider: Optional[Callable[[str], bool]] = None,
    ):
        if stop_rule not in _VALID_STOP_RULES:
            raise ValueError(
                f"Invalid stop_rule={stop_rule}. Must be one of {sorted(_VALID_STOP_RULES)}."
            )
        if train_trigger not in (TRAIN_TRIGGER_FIXED_PROMPT, TRAIN_TRIGGER_FIXED_SAMPLES):
            raise ValueError(
                f"Invalid train_trigger={train_trigger}. "
                f"Must be '{TRAIN_TRIGGER_FIXED_PROMPT}' or '{TRAIN_TRIGGER_FIXED_SAMPLES}'."
            )
        if fixed_rollout_n < 1 or max_rollout_n < 1:
            raise ValueError(
                f"fixed_rollout_n ({fixed_rollout_n}) and max_rollout_n ({max_rollout_n}) must be >= 1."
            )
        if fixed_rollout_n > max_rollout_n:
            raise ValueError(
                f"fixed_rollout_n ({fixed_rollout_n}) must not exceed max_rollout_n ({max_rollout_n})."
            )
        if min_positive_k < 1:
            raise ValueError(f"min_positive_k ({min_positive_k}) must be >= 1.")
        if min_rollout_n < 1 or min_rollout_n > max_rollout_n:
            raise ValueError(
                f"min_rollout_n ({min_rollout_n}) must be in [1, max_rollout_n={max_rollout_n}]."
            )
        if target_positive_count < 0 or target_negative_count < 0:
            raise ValueError(
                f"target_positive_count ({target_positive_count}) and "
                f"target_negative_count ({target_negative_count}) must be >= 0."
            )
        if target_positive_count + target_negative_count > max_rollout_n:
            raise ValueError(
                f"target positive+negative ({target_positive_count}+{target_negative_count}) "
                f"must not exceed max_rollout_n ({max_rollout_n})."
            )
        if continue_on_mixed_chunk_size < 1:
            raise ValueError(
                f"continue_on_mixed_chunk_size ({continue_on_mixed_chunk_size}) must be >= 1."
            )
        if not 0.0 <= adaptive_chunk_alpha <= 1.0:
            raise ValueError(
                f"adaptive_chunk_alpha ({adaptive_chunk_alpha}) must be in [0, 1]."
            )
        if not 0.0 <= adaptive_mixed_target_prob <= 1.0:
            raise ValueError(
                f"adaptive_mixed_target_prob ({adaptive_mixed_target_prob}) must be in [0, 1]."
            )
        if adaptive_chunk_prior_tau <= 0:
            raise ValueError(
                f"adaptive_chunk_prior_tau ({adaptive_chunk_prior_tau}) must be > 0."
            )
        if medium_focus_peak_n is not None and (
            medium_focus_peak_n < min_rollout_n or medium_focus_peak_n > max_rollout_n
        ):
            raise ValueError(
                f"medium_focus_peak_n ({medium_focus_peak_n}) must be in "
                f"[min_rollout_n={min_rollout_n}, max_rollout_n={max_rollout_n}]."
            )
        if compensation_bank_max < 0:
            raise ValueError(f"compensation_bank_max ({compensation_bank_max}) must be >= 0.")
        if compensation_chunk_size < 1:
            raise ValueError(f"compensation_chunk_size ({compensation_chunk_size}) must be >= 1.")
        if compensation_save_n < 0:
            raise ValueError(f"compensation_save_n ({compensation_save_n}) must be >= 0.")
        if not 0.0 <= compensation_save_low <= compensation_save_high <= 1.0:
            raise ValueError(
                "compensation_save_low/high must satisfy "
                f"0 <= low <= high <= 1, got {compensation_save_low}/{compensation_save_high}."
            )
        if compensation_roi_threshold < 0:
            raise ValueError(
                f"compensation_roi_threshold ({compensation_roi_threshold}) must be >= 0."
            )
        if compensation_budget_baseline_n is not None and compensation_budget_baseline_n < 1:
            raise ValueError(
                f"compensation_budget_baseline_n ({compensation_budget_baseline_n}) must be >= 1."
            )
        if compensation_budget_window_size < 0:
            raise ValueError(
                f"compensation_budget_window_size ({compensation_budget_window_size}) must be >= 0."
            )
        if not 0.0 <= compensation_estimate_clip < 0.5:
            raise ValueError(
                f"compensation_estimate_clip ({compensation_estimate_clip}) must be in [0, 0.5)."
            )

        self.stop_rule = stop_rule
        self.fixed_rollout_n = fixed_rollout_n
        self.max_rollout_n = max_rollout_n
        self.min_rollout_n = int(min_rollout_n)
        self.min_positive_k = min_positive_k
        self.target_positive_count = int(target_positive_count)
        self.target_negative_count = int(target_negative_count)
        self.continue_on_mixed_chunk_size = int(continue_on_mixed_chunk_size)
        self.adaptive_chunk_alpha = float(adaptive_chunk_alpha)
        self.adaptive_mixed_target_prob = float(adaptive_mixed_target_prob)
        self.adaptive_chunk_prior_tau = float(adaptive_chunk_prior_tau)
        self.adaptive_chunk_target_strategy = str(adaptive_chunk_target_strategy or "medium_focus_plateau").strip().lower()
        # The initial offline allocation peaks at the nominal GRPO group size,
        # not at max_rollout_n: the [fixed_rollout_n, max_rollout_n] headroom is
        # what compensation is allowed to spend later. Anchoring the peak at
        # max_rollout_n would let mid-difficulty prompts consume the whole cap
        # up front and leave nothing to compensate with.
        self.medium_focus_peak_n = int(medium_focus_peak_n or fixed_rollout_n)
        self.compensation_bank_max = int(compensation_bank_max)
        self.compensation_chunk_size = int(compensation_chunk_size)
        self.compensation_save_n = int(compensation_save_n)
        self.compensation_save_low = float(compensation_save_low)
        self.compensation_save_high = float(compensation_save_high)
        self.compensation_roi_threshold = float(compensation_roi_threshold)
        self.compensation_rule = str(compensation_rule)
        self.compensation_rescue_prob = float(compensation_rescue_prob)
        self.compensation_minority_target = int(compensation_minority_target)
        self.compensation_estimate_clip = float(compensation_estimate_clip)
        self.compensation_budget_bank = 0
        self.compensation_budget_baseline_n = int(compensation_budget_baseline_n or fixed_rollout_n)
        self.compensation_budget_prompt_count = 0
        self.compensation_budget_baseline_total = 0
        self.compensation_budget_committed_total = 0
        self.compensation_budget_window_size = int(compensation_budget_window_size)
        self.compensation_budget_window = deque()
        self.compensation_budget_records: dict[int, list[int]] = {}
        self.compensation_budget_next_record_id = 0
        self.compensation_budget_window_baseline_total = 0
        self.compensation_budget_window_committed_total = 0
        self.train_trigger = train_trigger
        self.dynamic_fixed_n_provider = dynamic_fixed_n_provider
        self.prompt_pass_rate_provider = prompt_pass_rate_provider
        self.min_rollout_n_provider = min_rollout_n_provider
        self.warmup_prompt_provider = warmup_prompt_provider
        self.warmup_prompts = 0

        # prompt_uid -> PromptState
        self._states: dict[str, PromptState] = {}

        # Aggregated statistics
        self._completed_prompts_pending: list[str] = []  # prompt_uids waiting to be popped for training
        self._pending_samples: int = 0  # sum of samples inside _completed_prompts_pending
        self.compensation_stats: dict[str, float | int] = {
            "bank_deposited_prompts": 0,
            "bank_deposited_samples": 0,
            "bank_deposit_skipped_full": 0,
            "compensated_prompts": 0,
            "compensated_samples": 0,
            "compensation_candidates": 0,
            "compensation_rejected_roi": 0,
            "compensation_rejected_unrescuable": 0,
            "compensation_rejected_budget": 0,
            "compensation_success_prompts": 0,
            "compensation_final_acc_sum": 0.0,
            "compensation_roi_sum": 0.0,
            "compensation_success_prob_sum": 0.0,
            "compensation_extra_n_sum": 0,
        }

        logger.info(
            "[AsyncResponseBuffer] init: stop_rule=%s, fixed_rollout_n=%d, max_rollout_n=%d, "
            "min_rollout_n=%d, min_positive_k=%d, target_positive_count=%d, target_negative_count=%d, "
            "continue_on_mixed_chunk_size=%d, adaptive_chunk_alpha=%.3f, "
            "adaptive_mixed_target_prob=%.3f, adaptive_chunk_prior_tau=%.3f, "
            "adaptive_chunk_target_strategy=%s, medium_focus_peak_n=%d, compensation_bank_max=%d, "
            "compensation_chunk_size=%d, compensation_save_n=%d, compensation_save_range=[%.3f, %.3f], "
            "compensation_roi_threshold=%.3f, compensation_rule=%s, "
            "compensation_rescue_prob=%.3f, compensation_minority_target=%d, "
            "compensation_estimate_clip=%.3f, "
            "compensation_budget_baseline_n=%d, "
            "compensation_budget_window_size=%d, "
            "train_trigger=%s, dynamic_fixed_n_provider=%s, prompt_pass_rate_provider=%s, "
            "min_rollout_n_provider=%s",
            stop_rule,
            fixed_rollout_n,
            max_rollout_n,
            self.min_rollout_n,
            min_positive_k,
            self.target_positive_count,
            self.target_negative_count,
            self.continue_on_mixed_chunk_size,
            self.adaptive_chunk_alpha,
            self.adaptive_mixed_target_prob,
            self.adaptive_chunk_prior_tau,
            self.adaptive_chunk_target_strategy,
            self.medium_focus_peak_n,
            self.compensation_bank_max,
            self.compensation_chunk_size,
            self.compensation_save_n,
            self.compensation_save_low,
            self.compensation_save_high,
            self.compensation_roi_threshold,
            self.compensation_rule,
            self.compensation_rescue_prob,
            self.compensation_minority_target,
            self.compensation_estimate_clip,
            self.compensation_budget_baseline_n,
            self.compensation_budget_window_size,
            train_trigger,
            "enabled" if dynamic_fixed_n_provider is not None else "disabled",
            "enabled" if prompt_pass_rate_provider is not None else "disabled",
            "enabled" if min_rollout_n_provider is not None else "disabled",
        )

    # ------------------------------------------------------------------
    # Life-cycle: register, submit, finish
    # ------------------------------------------------------------------

    def register_prompt(
        self,
        prompt_uid: str,
        sampling_prob: float = 0.0,
        start_param_version: int = 0,
    ) -> PromptState:
        """Ensure a state entry exists for ``prompt_uid`` and return it.

        This is typically called when the first sequence job of a new prompt is
        about to be submitted. Safe to call multiple times for the same uid.

        ``start_param_version`` records the rollouter's current policy parameter
        version at registration time; it is used downstream to compute
        prompt-level staleness. The value is set ONLY on first registration --
        subsequent calls with the same uid do not overwrite it, so the prompt's
        "birth version" is stable for its whole lifetime.
        """
        if prompt_uid not in self._states:
            self._states[prompt_uid] = PromptState(
                prompt_uid=prompt_uid,
                sampling_prob=sampling_prob,
                start_param_version=int(start_param_version),
            )
        return self._states[prompt_uid]

    def on_job_submitted(self, prompt_uid: str) -> None:
        """Mark that a new sequence job has been submitted for this prompt."""
        state = self._states.get(prompt_uid)
        if state is None:
            state = self.register_prompt(prompt_uid)
        state.in_flight += 1

    def on_response_finished(
        self,
        prompt_uid: str,
        sequence: Any,
        reward: float,
    ) -> None:
        """Record a finished sequence (single sample DataProto) and its reward."""
        state = self._states.get(prompt_uid)
        if state is None:
            # Shouldn't happen if register/submit is called; create a state just in case.
            state = self.register_prompt(prompt_uid)
            logger.warning(
                "[AsyncResponseBuffer] on_response_finished called for unregistered prompt_uid=%s",
                prompt_uid,
            )
        state.in_flight = max(0, state.in_flight - 1)
        state.sequences.append(sequence)
        state.rewards.append(float(reward))

    # ------------------------------------------------------------------
    # Stop rule
    # ------------------------------------------------------------------

    def _resolve_fixed_n(self, prompt_uid: str) -> int:
        """Resolve the fixed_rollout_n threshold for this prompt.

        Result is cached on ``PromptState.target_n`` so we only ever call
        ``dynamic_fixed_n_provider`` ONCE per prompt. This has two benefits:
          * performance -- avoids a dict lookup + float() + strategy dispatch
            on every on_response_finished callback;
          * semantic stability -- the per-prompt target N is locked at the
            moment we first need to check it (typically right after the first
            finished sequence), so late pass_rate updates on a *different*
            co-running prompt won't cause this prompt's target N to jitter
            mid-generation (which would break the GRPO group-size contract
            downstream when stop_rule=fixed_rollout).

        If a dynamic provider is installed (e.g. bell-shaped based on pass_rate),
        query it on first access; otherwise fall back to the static
        ``fixed_rollout_n``. Result is clamped to [1, max_rollout_n].
        """
        state = self._states.get(prompt_uid)
        if state is not None and state.target_n is not None:
            return state.target_n

        n = self.fixed_rollout_n
        if self.dynamic_fixed_n_provider is not None:
            try:
                n = int(self.dynamic_fixed_n_provider(prompt_uid))
            except Exception as e:
                logger.warning(
                    "[AsyncResponseBuffer] dynamic_fixed_n_provider raised for prompt_uid=%s: %s; "
                    "falling back to static fixed_rollout_n=%d",
                    prompt_uid,
                    e,
                    self.fixed_rollout_n,
                )
                n = self.fixed_rollout_n
        resolved = max(1, min(int(n), self.max_rollout_n))
        if state is not None:
            state.target_n = resolved
        return resolved

    def _prompt_prior_pass_rate(self, prompt_uid: str) -> float:
        prior = 0.5
        if self.prompt_pass_rate_provider is not None:
            try:
                prior = float(self.prompt_pass_rate_provider(prompt_uid))
            except Exception as e:
                logger.warning(
                    "[AsyncResponseBuffer] prompt_pass_rate_provider raised for prompt_uid=%s: %s; "
                    "falling back to prior pass_rate=0.5",
                    prompt_uid,
                    e,
                )
                prior = 0.5
        if not math.isfinite(prior):
            prior = 0.5
        return min(1.0, max(0.0, prior))

    def _current_min_rollout_n(self) -> int:
        min_n = int(self.min_rollout_n)
        if self.min_rollout_n_provider is not None:
            try:
                min_n = int(self.min_rollout_n_provider())
            except Exception as e:
                logger.warning(
                    "[AsyncResponseBuffer] min_rollout_n_provider raised: %s; "
                    "falling back to min_rollout_n=%d",
                    e,
                    self.min_rollout_n,
                )
                min_n = int(self.min_rollout_n)
        return max(1, min(int(min_n), int(self.max_rollout_n)))

    def _floor_to_chunk(self, value: float, chunk_size: Optional[int] = None) -> int:
        chunk = max(1, int(chunk_size or self.continue_on_mixed_chunk_size))
        target = int(math.floor(float(value) / float(chunk)) * chunk)
        return max(int(self._current_min_rollout_n()), min(int(target), int(self.max_rollout_n)))

    def _floor_to_int(self, value: float) -> int:
        target = int(math.floor(float(value)))
        return max(int(self._current_min_rollout_n()), min(int(target), int(self.max_rollout_n)))

    def _medium_focus_n_from_pass_rate(self, pass_rate: float) -> int:
        p = min(1.0, max(0.0, float(pass_rate)))
        min_n = self._current_min_rollout_n()
        peak_n = max(min_n, min(int(self.medium_focus_peak_n), int(self.max_rollout_n)))
        raw = min_n + (4.0 * p * (1.0 - p)) * (peak_n - min_n)
        return self._floor_to_int(raw)

    def _record_initial_budget_commitment(self, state: PromptState, committed_n: int) -> None:
        baseline = int(self.compensation_budget_baseline_n)
        committed = int(committed_n)
        record_id = int(self.compensation_budget_next_record_id)
        self.compensation_budget_next_record_id += 1
        state.budget_record_id = record_id

        self.compensation_budget_prompt_count += 1
        self.compensation_budget_baseline_total += baseline
        self.compensation_budget_committed_total += committed

        if self.compensation_budget_window_size > 0:
            self.compensation_budget_records[record_id] = [baseline, committed]
            self.compensation_budget_window.append(record_id)
            self.compensation_budget_window_baseline_total += baseline
            self.compensation_budget_window_committed_total += committed
            while len(self.compensation_budget_window) > self.compensation_budget_window_size:
                old_id = self.compensation_budget_window.popleft()
                old = self.compensation_budget_records.pop(old_id, None)
                if old is None:
                    continue
                old_baseline, old_committed = old
                self.compensation_budget_window_baseline_total -= int(old_baseline)
                self.compensation_budget_window_committed_total -= int(old_committed)

    def _budget_available(self) -> int:
        if self.compensation_budget_window_size > 0:
            available = (
                self.compensation_budget_window_baseline_total
                - self.compensation_budget_window_committed_total
            )
        else:
            available = (
                self.compensation_budget_baseline_total
                - self.compensation_budget_committed_total
            )
        return max(0, int(available))

    def _commit_extra_budget(self, state: PromptState, extra_n: int) -> None:
        extra = int(extra_n)
        self.compensation_budget_committed_total += extra
        if self.compensation_budget_window_size <= 0:
            return
        record_id = state.budget_record_id
        if record_id is None:
            return
        record = self.compensation_budget_records.get(int(record_id))
        if record is None:
            return
        record[1] += extra
        self.compensation_budget_window_committed_total += extra

    def _resolve_offline_medium_focus_compensate_target_n(
        self,
        prompt_uid: str,
        state: PromptState,
    ) -> int:
        """Resolve initial offline medium-focus N and deposit saved budget once.

        This rule is intentionally offline: the initial N is computed from the
        historical pass-rate only and cached. Prompts with p in the configured
        save region donate up to ``compensation_save_n`` samples into a bounded
        budget bank, and may later receive exactly one compensation chunk if
        their finished group is all-positive/all-negative and its ROI is high.
        """
        if state.initial_target_n is not None:
            return int(state.initial_target_n)

        p = self._prompt_prior_pass_rate(prompt_uid)
        base_n = self._medium_focus_n_from_pass_rate(p)
        init_n = base_n
        saved = 0
        if base_n >= self.max_rollout_n:
            save_n_for_prompt = self.compensation_save_n
        elif base_n == self.max_rollout_n - self.continue_on_mixed_chunk_size:
            save_n_for_prompt = self.continue_on_mixed_chunk_size
        else:
            save_n_for_prompt = 0

        can_save = (
            self.compensation_budget_bank < self.compensation_bank_max
            and save_n_for_prompt > 0
            and self.compensation_save_low <= p <= self.compensation_save_high
            and base_n > self._current_min_rollout_n()
        )
        if can_save:
            min_n = self._current_min_rollout_n()
            requested = min(save_n_for_prompt, base_n - min_n)
            room = max(0, self.compensation_bank_max - self.compensation_budget_bank)
            saved = min(requested, room)
            init_n = max(min_n, base_n - saved)
            init_n = self._floor_to_int(init_n)
            saved = max(0, base_n - init_n)
            if saved > 0:
                self.compensation_budget_bank = min(
                    self.compensation_bank_max,
                    self.compensation_budget_bank + saved,
                )
                self.compensation_stats["bank_deposited_prompts"] += 1
                self.compensation_stats["bank_deposited_samples"] += saved
        elif self.compensation_budget_bank >= self.compensation_bank_max and (
            self.compensation_save_low <= p <= self.compensation_save_high
        ):
            self.compensation_stats["bank_deposit_skipped_full"] += 1

        state.base_target_n = int(base_n)
        state.initial_target_n = int(init_n)
        state.target_n = int(init_n)
        state.saved_budget = int(saved)
        return int(init_n)

    def _resolve_offline_medium_focus_budget_target_n(
        self,
        prompt_uid: str,
        state: PromptState,
    ) -> int:
        """Resolve offline medium-focus N and charge it against a global budget.

        Baseline budget grows by ``compensation_budget_baseline_n`` per prompt
        (usually 8). The initial target N is committed immediately when the
        prompt first gets a target; later compensation can only spend the
        accumulated gap between baseline and committed budget.
        """
        if state.initial_target_n is not None:
            return int(state.initial_target_n)

        if self.warmup_prompt_provider is not None and self.warmup_prompt_provider(prompt_uid):
            init_n = int(self.fixed_rollout_n)
            state.warmup = True
            state.base_target_n = init_n
            state.initial_target_n = init_n
            state.target_n = init_n
            self.warmup_prompts += 1
            return init_n

        if self.dynamic_fixed_n_provider is not None:
            init_n = self._resolve_fixed_n(prompt_uid)
        else:
            p = self._prompt_prior_pass_rate(prompt_uid)
            init_n = self._medium_focus_n_from_pass_rate(p)

        init_n = max(
            int(self._current_min_rollout_n()),
            min(int(init_n), int(self.max_rollout_n)),
        )
        init_n = self._floor_to_int(init_n)
        state.base_target_n = int(init_n)
        state.initial_target_n = int(init_n)
        state.target_n = int(init_n)

        self._record_initial_budget_commitment(state, init_n)
        trace(
            CHANNEL_ADAPTIVE_N,
            "initial_n",
            uid=prompt_uid,
            p=self._prompt_prior_pass_rate(prompt_uid),
            n=init_n,
            peak_n=self.medium_focus_peak_n,
            min_n=self._current_min_rollout_n(),
            max_n=self.max_rollout_n,
            headroom=int(self.max_rollout_n) - int(init_n),
            baseline_n=self.compensation_budget_baseline_n,
            budget_available=self._budget_available(),
            window_prompts=len(self.compensation_budget_window),
            from_provider=self.dynamic_fixed_n_provider is not None,
        )
        return int(init_n)

    def _compensation_roi(self, prompt_uid: str, state: PromptState) -> tuple[str, float, float]:
        """Return (state_name, success_prob, roi) for one-shot compensation."""
        if state.num_finished <= 0:
            return "", 0.0, 0.0
        if state.num_positive == state.num_finished:
            state_name = "all_positive"
            p = self._prompt_prior_pass_rate(prompt_uid)
            success_prob = 1.0 - (p ** self.compensation_chunk_size)
        elif state.num_positive == 0:
            state_name = "all_negative"
            p = self._prompt_prior_pass_rate(prompt_uid)
            success_prob = 1.0 - ((1.0 - p) ** self.compensation_chunk_size)
        else:
            return "mixed", 0.0, 0.0

        c = max(1, int(self.compensation_chunk_size))
        effective_gain = (state.num_finished + c) * success_prob
        roi = effective_gain / float(c)
        return state_name, float(success_prob), float(roi)

    def _maybe_grant_compensation(self, prompt_uid: str, state: PromptState) -> bool:
        """Grant one compensation chunk if bank and ROI constraints pass."""
        if state.compensated or state.compensation_granted:
            return False
        if self.compensation_budget_bank < self.compensation_chunk_size:
            return False
        if state.num_finished + self.compensation_chunk_size > self.max_rollout_n:
            return False

        state_name, success_prob, roi = self._compensation_roi(prompt_uid, state)
        if state_name not in ("all_positive", "all_negative"):
            return False

        self.compensation_stats["compensation_candidates"] += 1
        if roi <= self.compensation_roi_threshold:
            self.compensation_stats["compensation_rejected_roi"] += 1
            return False

        self.compensation_budget_bank -= self.compensation_chunk_size
        state.compensated = True
        state.compensation_granted = True
        state.compensation_state = state_name
        state.compensation_samples = self.compensation_chunk_size
        state.compensation_roi = roi
        state.compensation_success_prob = success_prob
        state.target_n = state.num_finished + self.compensation_chunk_size
        self.compensation_stats["compensated_prompts"] += 1
        self.compensation_stats["compensated_samples"] += self.compensation_chunk_size
        self.compensation_stats["compensation_roi_sum"] += roi
        self.compensation_stats["compensation_success_prob_sum"] += success_prob
        self.compensation_stats["compensation_extra_n_sum"] += self.compensation_chunk_size
        return True

    def _rescue_probability(self, q: float, extra: int, minority_target: int) -> float:
        """P(at least ``minority_target`` samples of the missing class in ``extra`` draws)."""
        target = int(minority_target)
        if extra < target:
            return 0.0
        q = min(1.0, max(0.0, float(q)))
        below_target = sum(
            math.comb(extra, i) * (q ** i) * ((1.0 - q) ** (extra - i))
            for i in range(target)
        )
        return min(1.0, max(0.0, 1.0 - below_target))

    def _plan_roi_max_extra(
        self,
        state_name: str,
        p: float,
        num_finished: int,
        capacity: int,
        available: int,
        unit: int,
    ) -> tuple[int, float]:
        """Largest affordable extra whose ROI still clears the threshold."""
        max_extra = int(math.floor(float(min(capacity, available)) / float(unit)) * unit)
        if max_extra <= 0:
            self.compensation_stats["compensation_rejected_budget"] += 1
            trace(
                CHANNEL_ADAPTIVE_N,
                "compensation_reject",
                reason="budget",
                group=state_name,
                p=p,
                capacity=capacity,
                available=available,
            )
            return 0, 0.0

        best_extra = 0
        best_success_prob = 0.0
        for extra in range(unit, max_extra + 1, unit):
            if state_name == "all_positive":
                success_prob = 1.0 - (p ** extra)
            else:
                success_prob = 1.0 - ((1.0 - p) ** extra)
            roi = (float(num_finished) + float(extra)) * float(success_prob) / float(extra)
            if roi > self.compensation_roi_threshold:
                best_extra = int(extra)
                best_success_prob = float(success_prob)

        if best_extra <= 0:
            self.compensation_stats["compensation_rejected_roi"] += 1
            trace(
                CHANNEL_ADAPTIVE_N,
                "compensation_reject",
                reason="roi",
                group=state_name,
                p=p,
                k=num_finished,
                capacity=capacity,
                available=available,
                max_extra=max_extra,
                roi_threshold=self.compensation_roi_threshold,
            )
        return best_extra, best_success_prob

    def _minority_draw_probability(self, state_name: str, prior: float, num_finished: int) -> tuple[float, float]:
        """Estimate P(next sample belongs to the missing class) for a homogeneous group.

        The offline prior systematically *under*-estimates the pass rate once the
        policy has improved past the model the dataset was labelled with, and
        that bias is safe in one direction only:

        * all-negative groups use ``q = p``, so an under-estimated prior asks for
          more extras than needed. Measured hit rate 70% against an 85% target.
        * all-positive groups use ``q = 1 - p``, so the same bias *over*-states
          the chance of a flip. A prior of 0.125 on a 7/7 correct group claims
          0.875, one extra sample looks sufficient, and the measured hit rate
          collapses to 26% (implied true q ~0.13).

        So the group's own evidence is only mixed in for the all-positive side:
        the prior is blended toward the observed pass rate (1.0) with weight
        ``num_finished / adaptive_chunk_prior_tau``, the same shrinkage the
        online chunk estimator uses. The blend is clipped away from 0/1 so a
        group with ``num_finished >= tau`` does not assert a certainty it cannot
        have; those groups get rejected by the capacity check instead of by a
        degenerate zero probability.

        Returns ``(q, p_hat)``.
        """
        prior = float(prior)
        if state_name != "all_positive":
            return prior, prior

        weight = min(1.0, float(num_finished) / max(1e-9, float(self.adaptive_chunk_prior_tau)))
        p_hat = (1.0 - weight) * prior + weight * 1.0
        lo = float(self.compensation_estimate_clip)
        p_hat = min(1.0 - lo, max(lo, p_hat))
        return 1.0 - p_hat, p_hat

    def _plan_min_rescue_prob_extra(
        self,
        state_name: str,
        p: float,
        num_finished: int,
        capacity: int,
        available: int,
        unit: int,
    ) -> tuple[int, float]:
        """Smallest chunk-aligned extra whose rescue probability reaches the target.

        ``capacity`` is the room left under ``max_rollout_n`` and ``available`` the
        remaining window budget. A prompt that cannot reach the target probability
        within ``capacity`` is treated as unrescuable and gets nothing, so its
        budget stays available for prompts that can still be turned into a mixed
        group.
        """
        q, p_hat = self._minority_draw_probability(state_name, p, num_finished)
        for extra in range(1, int(capacity) + 1):
            if self._rescue_probability(q, extra, self.compensation_minority_target) < self.compensation_rescue_prob:
                continue
            aligned = int(math.ceil(float(extra) / float(unit)) * unit)
            if aligned > capacity:
                self.compensation_stats["compensation_rejected_unrescuable"] += 1
                trace(
                    CHANNEL_ADAPTIVE_N,
                    "compensation_reject",
                    reason="unrescuable_after_align",
                    group=state_name,
                    p=p,
                    p_hat=p_hat,
                    q=q,
                    k=num_finished,
                    required=extra,
                    aligned=aligned,
                    capacity=capacity,
                    available=available,
                )
                return 0, 0.0
            if aligned > available:
                self.compensation_stats["compensation_rejected_budget"] += 1
                trace(
                    CHANNEL_ADAPTIVE_N,
                    "compensation_reject",
                    reason="budget",
                    group=state_name,
                    p=p,
                    p_hat=p_hat,
                    q=q,
                    k=num_finished,
                    required=extra,
                    aligned=aligned,
                    capacity=capacity,
                    available=available,
                )
                return 0, 0.0
            return aligned, self._rescue_probability(q, aligned, self.compensation_minority_target)

        self.compensation_stats["compensation_rejected_unrescuable"] += 1
        trace(
            CHANNEL_ADAPTIVE_N,
            "compensation_reject",
            reason="unrescuable_capacity",
            group=state_name,
            p=p,
            p_hat=p_hat,
            q=q,
            k=num_finished,
            capacity=capacity,
            available=available,
            best_prob=self._rescue_probability(q, int(capacity), self.compensation_minority_target),
            target_prob=self.compensation_rescue_prob,
        )
        return 0, 0.0

    def _maybe_grant_budget_compensation(self, prompt_uid: str, state: PromptState) -> bool:
        """Grant a one-shot compensation to a homogeneous group under the window budget."""
        if state.warmup or state.compensated or state.compensation_granted:
            return False
        if state.num_finished <= 0 or state.num_finished >= self.max_rollout_n:
            return False

        if state.num_positive == state.num_finished:
            state_name = "all_positive"
        elif state.num_positive == 0:
            state_name = "all_negative"
        else:
            return False

        available = self._budget_available()
        unit = max(1, int(self.compensation_chunk_size))
        capacity = int(self.max_rollout_n) - int(state.num_finished)
        p = self._prompt_prior_pass_rate(prompt_uid)
        self.compensation_stats["compensation_candidates"] += 1

        if self.compensation_rule == COMPENSATION_RULE_MIN_RESCUE_PROB:
            best_extra, best_success_prob = self._plan_min_rescue_prob_extra(
                state_name=state_name,
                p=p,
                num_finished=int(state.num_finished),
                capacity=capacity,
                available=available,
                unit=unit,
            )
        else:
            best_extra, best_success_prob = self._plan_roi_max_extra(
                state_name=state_name,
                p=p,
                num_finished=int(state.num_finished),
                capacity=capacity,
                available=available,
                unit=unit,
            )

        if best_extra <= 0:
            return False

        best_roi = (
            (float(state.num_finished) + float(best_extra)) * float(best_success_prob) / float(best_extra)
        )

        state.compensated = True
        state.compensation_granted = True
        state.compensation_state = state_name
        state.compensation_samples = int(best_extra)
        state.compensation_roi = float(best_roi)
        state.compensation_success_prob = float(best_success_prob)
        state.target_n = int(state.num_finished) + int(best_extra)
        self._commit_extra_budget(state, best_extra)
        trace(
            CHANNEL_ADAPTIVE_N,
            "compensation_grant",
            uid=prompt_uid,
            rule=self.compensation_rule,
            group=state_name,
            p=p,
            p_hat=(
                self._minority_draw_probability(state_name, p, int(state.num_finished))[1]
                if self.compensation_rule == COMPENSATION_RULE_MIN_RESCUE_PROB
                else None
            ),
            k=int(state.num_finished),
            extra=int(best_extra),
            target_n=int(state.target_n),
            prob=best_success_prob,
            roi=best_roi,
            capacity=capacity,
            budget_before=available,
            budget_after=self._budget_available(),
        )
        self.compensation_stats["compensated_prompts"] += 1
        self.compensation_stats["compensated_samples"] += int(best_extra)
        self.compensation_stats["compensation_roi_sum"] += float(best_roi)
        self.compensation_stats["compensation_success_prob_sum"] += float(best_success_prob)
        self.compensation_stats["compensation_extra_n_sum"] += int(best_extra)
        return True

    def _maybe_grant_small_n_knapsack_compensation(self, prompt_uid: str, state: PromptState) -> bool:
        """Grant one compensation chunk when a small-N group is all-positive/all-negative."""
        del prompt_uid
        if state.compensated or state.compensation_granted:
            return False
        base_n = int(state.initial_target_n or state.target_n or state.num_finished)
        if base_n > 6:
            return False
        if state.num_finished < base_n:
            return False
        c = max(1, int(self.compensation_chunk_size))
        if state.num_finished + c > self.max_rollout_n:
            return False
        if state.num_positive == state.num_finished:
            state_name = "all_positive"
        elif state.num_positive == 0:
            state_name = "all_negative"
        else:
            return False

        state.compensated = True
        state.compensation_granted = True
        state.compensation_state = state_name
        state.compensation_samples = c
        state.compensation_roi = 0.0
        state.compensation_success_prob = 0.0
        state.target_n = state.num_finished + c
        self.compensation_stats["compensated_prompts"] += 1
        self.compensation_stats["compensated_samples"] += c
        self.compensation_stats["compensation_extra_n_sum"] += c
        return True

    def record_completed_prompt_stats(self, state: PromptState) -> None:
        """Record final compensation outcomes after a completed prompt is emitted."""
        trace(
            CHANNEL_ADAPTIVE_N,
            "emit",
            uid=state.prompt_uid,
            n=state.num_finished,
            pos=state.num_positive,
            neg=state.num_negative,
            initial_n=state.initial_target_n,
            target_n=state.target_n,
            compensated=state.compensated,
            extra=state.compensation_samples,
            group=state.compensation_state or "-",
            mixed=state.num_positive > 0 and state.num_negative > 0,
            # A completed group whose size disagrees with its target means the
            # rollouter over/under-issued sequences for this prompt.
            size_matches_target=state.target_n is None or state.num_finished == int(state.target_n),
            staleness=state.start_param_version,
        )
        if not state.compensation_granted or state.num_finished <= 0:
            return
        final_acc = float(state.num_positive) / float(state.num_finished)
        self.compensation_stats["compensation_final_acc_sum"] += final_acc
        if state.num_positive > 0 and state.num_negative > 0:
            self.compensation_stats["compensation_success_prompts"] += 1

    def _resolve_adaptive_chunk_mixed_target_n(self, prompt_uid: str, state: PromptState) -> int:
        """Estimate target N from historical pass-rate and current chunk evidence.

        The estimate gradually trusts the current rollout more until a cutoff:

            w = min(1, k / prior_tau)
            p_hat = (1 - w) * historical_pass_rate + w * current_rollout_pass_rate

        Once k >= prior_tau (default: 8), p_hat is exactly the current rollout
        pass rate.

        Target N follows the configured online target strategy:
          * medium_focus: min_n + 4 * p_hat * (1 - p_hat) * (max_n - min_n);
          * medium_focus_plateau: medium_focus, but 0.5 < p_hat < 0.8 uses max_n;
          * productive_curve: one-sided curve from earlier experiments.

        The chosen n is rounded down to the chunk size so the rollouter keeps
        issuing whole chunks, and finally clamped to [min_rollout_n, max_rollout_n].
        """
        prior = self._prompt_prior_pass_rate(prompt_uid)
        if state.num_finished > 0:
            current = float(state.num_positive) / float(state.num_finished)
        else:
            current = prior
        current_weight = min(1.0, float(state.num_finished) / self.adaptive_chunk_prior_tau)
        p = (1.0 - current_weight) * prior + current_weight * current
        p = min(1.0, max(0.0, p))

        min_n = self._current_min_rollout_n()
        span = float(self.max_rollout_n - min_n)
        strategy = self.adaptive_chunk_target_strategy
        if strategy == "medium_focus":
            target = min_n + (4.0 * p * (1.0 - p)) * span
        elif strategy == "medium_focus_plateau":
            if 0.5 < p < 0.8:
                target = float(self.max_rollout_n)
            else:
                target = min_n + (4.0 * p * (1.0 - p)) * span
        elif strategy == "productive_curve":
            low_threshold = 0.5
            high_threshold = 0.8
            easy_cutoff = 0.95
            if p <= 0.0:
                target = float(min_n)
            elif p >= easy_cutoff:
                target = float(min_n)
            elif p < low_threshold:
                target = min_n + (p / low_threshold) * span
            elif p <= high_threshold:
                target = float(self.max_rollout_n)
            else:
                decay = (p - high_threshold) / (easy_cutoff - high_threshold)
                target = self.max_rollout_n - decay * span
        else:
            logger.warning(
                "[AsyncResponseBuffer] Unknown adaptive_chunk_target_strategy=%s; falling back to medium_focus",
                strategy,
            )
            self.adaptive_chunk_target_strategy = "medium_focus"
            target = min_n + (4.0 * p * (1.0 - p)) * span

        chunk_size = max(1, int(self.continue_on_mixed_chunk_size))
        target = int(math.floor(float(target) / float(chunk_size)) * chunk_size)
        target = max(int(min_n), min(int(target), int(self.max_rollout_n)))
        state.target_n = target
        return target

    def stop_generation(self, prompt_uid: str) -> bool:
        """Decide whether to stop issuing new sequence jobs for ``prompt_uid``.

        Returns True once the configured rule is satisfied OR the hard max is
        reached. Idempotent - once True, it remains True (via state.stopped).
        """
        state = self._states.get(prompt_uid)
        if state is None:
            return True  # unknown uid -> definitely stop

        if state.stopped:
            return True

        # Hard cap
        if state.num_finished >= self.max_rollout_n:
            state.stopped = True
            return True

        rule = self.stop_rule
        stop = False
        if rule == STOP_RULE_FIXED:
            threshold = self._resolve_fixed_n(prompt_uid)
            stop = state.num_finished >= threshold
        elif rule == STOP_RULE_PREFIXED:
            # Stop when finished >= ceil(1 / p_i), capped at max_rollout_n
            p = max(state.sampling_prob, 1e-8)
            threshold = min(self.max_rollout_n, int(math.ceil(1.0 / p)))
            stop = state.num_finished >= max(1, threshold)
        elif rule == STOP_RULE_AT_LEAST_POS:
            if self.target_positive_count + self.target_negative_count > 0:
                stop = (
                    state.num_positive >= self.target_positive_count
                    and state.num_negative >= self.target_negative_count
                )
            else:
                stop = state.num_positive >= self.min_positive_k
        elif rule == STOP_RULE_AT_LEAST_POS_NEG:
            stop = state.num_positive >= 1 and state.num_negative >= 1
        elif rule == STOP_RULE_CONTINUE_ON_MIXED_CHUNK:
            chunk_size = max(1, int(self.continue_on_mixed_chunk_size))
            if state.num_finished < chunk_size:
                stop = False
            else:
                last_chunk = state.rewards[-chunk_size:]
                has_positive = any(r > 0 for r in last_chunk)
                has_negative = any(r <= 0 for r in last_chunk)
                stop = not (has_positive and has_negative)
        elif rule == STOP_RULE_ADAPTIVE_CHUNK_MIXED:
            if state.num_finished < self._current_min_rollout_n():
                stop = False
            else:
                threshold = self._resolve_adaptive_chunk_mixed_target_n(prompt_uid, state)
                stop = state.num_finished >= threshold
        elif rule == STOP_RULE_OFFLINE_MEDIUM_FOCUS_COMPENSATE:
            threshold = self._resolve_offline_medium_focus_compensate_target_n(prompt_uid, state)
            if state.num_finished < threshold:
                stop = False
            elif self._maybe_grant_compensation(prompt_uid, state):
                stop = False
            else:
                stop = state.num_finished >= int(state.target_n or threshold)
        elif rule == STOP_RULE_OFFLINE_MEDIUM_FOCUS_BUDGET_COMPENSATE:
            threshold = self._resolve_offline_medium_focus_budget_target_n(prompt_uid, state)
            if state.num_finished < threshold:
                stop = False
            elif self._maybe_grant_budget_compensation(prompt_uid, state):
                stop = False
            else:
                stop = state.num_finished >= int(state.target_n or threshold)
        elif rule == STOP_RULE_KNAPSACK_VALUE_COMPENSATE:
            threshold = self._resolve_fixed_n(prompt_uid)
            if state.initial_target_n is None:
                state.base_target_n = int(threshold)
                state.initial_target_n = int(threshold)
            if state.num_finished < threshold:
                stop = False
            elif self._maybe_grant_small_n_knapsack_compensation(prompt_uid, state):
                stop = False
            else:
                stop = state.num_finished >= int(state.target_n or threshold)
        elif rule == STOP_RULE_MAX:
            stop = state.num_finished >= self.max_rollout_n

        if stop:
            state.stopped = True
            trace(
                CHANNEL_ADAPTIVE_N,
                "stop",
                uid=prompt_uid,
                rule=rule,
                n=state.num_finished,
                pos=state.num_positive,
                neg=state.num_negative,
                in_flight=state.in_flight,
                initial_n=state.initial_target_n,
                target_n=state.target_n,
                compensated=state.compensated,
                extra=state.compensation_samples,
                mixed=state.num_positive > 0 and state.num_negative > 0,
            )
        return stop

    # ------------------------------------------------------------------
    # Completion: pop prompts whose generation is fully done
    # ------------------------------------------------------------------

    def mark_if_complete(self, prompt_uid: str) -> bool:
        """If the prompt is stopped AND has no in-flight jobs, push it to the
        pending-completed queue. Returns True iff it was just pushed.
        """
        state = self._states.get(prompt_uid)
        if state is None:
            return False
        if not state.is_complete:
            return False
        # Guard against double push
        if prompt_uid in self._completed_prompts_pending:
            return False
        self._completed_prompts_pending.append(prompt_uid)
        self._pending_samples += state.num_finished
        return True

    def pop_completed_prompt(self, prompt_uid: Optional[str] = None) -> Optional[PromptState]:
        """Pop a completed prompt's full state (for emitting a RolloutSample).

        If ``prompt_uid`` is given, pop that specific one (if completed);
        otherwise pop the oldest pending completed prompt.
        """
        if prompt_uid is not None:
            if prompt_uid not in self._completed_prompts_pending:
                return None
            self._completed_prompts_pending.remove(prompt_uid)
        else:
            if not self._completed_prompts_pending:
                return None
            prompt_uid = self._completed_prompts_pending.pop(0)

        state = self._states.pop(prompt_uid, None)
        if state is not None:
            self._pending_samples = max(0, self._pending_samples - state.num_finished)
        return state

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def num_active_prompts(self) -> int:
        """Prompts that are registered but not yet popped (i.e. in working pool)."""
        return len(self._states)

    def num_unstopped_prompts(self) -> int:
        """Prompts that have not been marked as stopped yet."""
        return sum(1 for s in self._states.values() if not s.stopped)

    def num_pending_completed_prompts(self) -> int:
        return len(self._completed_prompts_pending)

    def num_pending_samples(self) -> int:
        return self._pending_samples

    def num_in_flight_samples(self) -> int:
        """Responses submitted to rollout workers but not finished yet."""
        return sum(s.in_flight for s in self._states.values())

    def get_state(self, prompt_uid: str) -> Optional[PromptState]:
        return self._states.get(prompt_uid)

    def max_live_prompt_staleness(self, current_param_version: int) -> int:
        """Return max(current_param_version - start_param_version) over all
        live prompts (i.e. registered but not yet popped). Returns 0 when no
        prompts are alive, so a freshly-drained rollouter is never considered
        stale.

        "Live" here matches ``num_active_prompts()`` -- every entry in
        ``_states``, including stopped-but-not-emitted prompts. We include
        stopped prompts on purpose: as long as they have not yet been popped
        (their sequences are still held by the buffer and will be emitted to
        the trainer), their birth version still contributes to how stale the
        next emitted RolloutSample can be.
        """
        if not self._states:
            return 0
        return max(
            int(current_param_version) - int(s.start_param_version)
            for s in self._states.values()
        )

    def num_stale_prompts(self, current_param_version: int) -> int:
        """Return the number of live prompts whose ``start_param_version`` is
        strictly less than ``current_param_version`` (i.e. staleness > 0).

        This is a prompt-level staleness counter: it counts how many prompts
        were born under an older parameter version and are still alive in the
        buffer (registered but not yet popped).
        """
        if not self._states:
            return 0
        cpv = int(current_param_version)
        return sum(
            1 for s in self._states.values()
            if cpv > int(s.start_param_version)
        )

    def count_stale_samples(self, current_param_version: int) -> int:
        """Return the total number of stale samples (in_flight + finished) held
        by this buffer, i.e. samples belonging to live prompts whose
        ``start_param_version < current_param_version``.

        This is a sample-level staleness counter intended to be called by the
        rollouter from ``reset_staleness()`` to recompute the running
        ``stale_samples`` accumulator after a parameter version bump. It does
        NOT include samples that have already been popped out of the buffer
        (those have already been emitted to the message queue and are governed
        by the queue's own purge mechanism instead).

        We deliberately keep this scoped to buffer-resident samples only:
        samples sitting in the downstream message queue may have arbitrary
        staleness in ``[0, staleness_threshold-1]`` (the ones with
        ``staleness >= staleness_threshold`` have already been purged by the
        trainer), so conflating them with buffer-side stale samples would make
        the counter's semantics fuzzy.
        """
        if not self._states:
            return 0
        cpv = int(current_param_version)
        total = 0
        for s in self._states.values():
            if cpv > int(s.start_param_version):
                total += s.in_flight + s.num_finished
        return total

    def stats(self) -> dict:
        total_in_flight = self.num_in_flight_samples()
        total_finished = sum(s.num_finished for s in self._states.values())
        compensated_prompts = int(self.compensation_stats["compensated_prompts"])
        compensation_success_prompts = int(self.compensation_stats["compensation_success_prompts"])
        global_budget_available = int(self.compensation_budget_baseline_total - self.compensation_budget_committed_total)
        window_budget_available = int(
            self.compensation_budget_window_baseline_total
            - self.compensation_budget_window_committed_total
        )
        budget_available = self._budget_available()
        stats = {
            "response_buffer/active_prompts": self.num_active_prompts(),
            "response_buffer/unstopped_prompts": self.num_unstopped_prompts(),
            "response_buffer/pending_completed_prompts": self.num_pending_completed_prompts(),
            "response_buffer/pending_samples": self.num_pending_samples(),
            "response_buffer/in_flight_samples": total_in_flight,
            "response_buffer/live_finished_samples": total_finished,
            "compensation/warmup_prompts_total": int(self.warmup_prompts),
            "compensation/bank_current": int(self.compensation_budget_bank),
            "compensation/bank_max": int(self.compensation_bank_max),
            "compensation/bank_deposited_prompts_total": int(self.compensation_stats["bank_deposited_prompts"]),
            "compensation/bank_deposited_samples_total": int(self.compensation_stats["bank_deposited_samples"]),
            "compensation/bank_deposit_skipped_full_total": int(self.compensation_stats["bank_deposit_skipped_full"]),
            "compensation/candidates_total": int(self.compensation_stats["compensation_candidates"]),
            "compensation/rejected_roi_total": int(self.compensation_stats["compensation_rejected_roi"]),
            "compensation/rejected_unrescuable_total": int(
                self.compensation_stats["compensation_rejected_unrescuable"]
            ),
            "compensation/rejected_budget_total": int(
                self.compensation_stats["compensation_rejected_budget"]
            ),
            "compensation/prompts_total": compensated_prompts,
            "compensation/samples_total": int(self.compensation_stats["compensated_samples"]),
            "compensation/success_prompts_total": compensation_success_prompts,
            "compensation/success_ratio": (
                compensation_success_prompts / compensated_prompts if compensated_prompts else 0.0
            ),
            "compensation/final_acc_mean": (
                float(self.compensation_stats["compensation_final_acc_sum"]) / compensated_prompts
                if compensated_prompts else 0.0
            ),
            "compensation/roi_mean": (
                float(self.compensation_stats["compensation_roi_sum"]) / compensated_prompts
                if compensated_prompts else 0.0
            ),
            "compensation/success_prob_mean": (
                float(self.compensation_stats["compensation_success_prob_sum"]) / compensated_prompts
                if compensated_prompts else 0.0
            ),
            "compensation/extra_n_mean": (
                float(self.compensation_stats["compensation_extra_n_sum"]) / compensated_prompts
                if compensated_prompts else 0.0
            ),
            "budget/baseline_n": int(self.compensation_budget_baseline_n),
            "budget/prompt_count": int(self.compensation_budget_prompt_count),
            "budget/baseline_total": int(self.compensation_budget_baseline_total),
            "budget/committed_total": int(self.compensation_budget_committed_total),
            "budget/available": budget_available,
            "budget/global_available": global_budget_available,
            "budget/window_size": int(self.compensation_budget_window_size),
            "budget/window_count": int(len(self.compensation_budget_window)),
            "budget/window_baseline_total": int(self.compensation_budget_window_baseline_total),
            "budget/window_committed_total": int(self.compensation_budget_window_committed_total),
            "budget/window_available": max(0, window_budget_available),
            "budget/window_available_raw": window_budget_available,
        }
        return stats
