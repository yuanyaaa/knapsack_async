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
AsyncPromptBuffer - A prompt buffer designed for fully-async streaming rollout.

Responsibilities:
- Configurable prompt row sampling: sequential, uniform, pass_rate-ordered (weight_sequential),
  or pass_rate-weighted priority
- Per-prompt pass_rate / count tracking (updated after each rollout via on_sample_end)
- Stateful checkpointing via state_dict() / load_state_dict()

Unlike the batch-oriented PromptBuffer in hytuner, this buffer is designed for
single-sample streaming: each call to get_single_sample() returns one prompt,
and on_sample_end() updates the buffer immediately after each rollout completes.

Prompt **sample_strategy** (see ``resolve_prompt_sample_strategy``):

- ``None`` / ``shuffle_sequential``: create one random permutation per dataset
  epoch, then traverse it without replacement (default).
- ``sequential``: iterate prompts in dataset order (round-robin).
- ``uniform``: uniform random row sampling.
- ``uniform_extreme_ucb``: uniform-like weighted sampling that only downweights
  pass_rate endpoints (p=0 or p=1) by ``uniform_extreme_alpha`` and optionally
  adds a UCB-style exploration multiplier controlled by ``uniform_ucb_c``.
- ``medium_ucb``: useful-sample focused weighted sampling. The exploitation
  term is ``4 * p * (1-p)`` and the exploration term is a count-based UCB bonus
  controlled by ``uniform_ucb_c``.
- ``weight_sequential``: reuses ``row_weights`` with ``0.5 - abs(pass_rate - 0.5)`` per row
  (same buffer as priority; **sampling** is round-robin in descending ``row_weights`` order).
  ``pass_rate`` updates every rollout; ``row_weights`` and traversal order refresh **once per
  epoch** (after one full pass over all dataset rows in the current order).
- Any registered priority name (e.g. ``medium_sharp``): weighted by ``pass_rate``.

NOTE: "How many sequences to roll out for a given prompt" (including any
adaptive pass_rate -> N heuristic) is NOT decided here. It is owned by the
rollouter side, specifically by AsyncResponseBuffer's stop_rule together with
an optional dynamic_fixed_n_provider that reads pass_rate from
``self.meta_data[prompt_uid]``. This module only exposes the adaptive_n
strategy registry (``register_adaptive_n`` / ``get_adaptive_n_fn``) for that
provider to reuse, plus a backward-compatible ``adaptive_rollout_n`` helper
that defaults to the ``medium_focus`` strategy.
"""

import json
import math
import os
from typing import Any, Callable, Optional

from verl.experimental.fully_async_policy.utils.adaptive_debug import (
    CHANNEL_PRIORITY,
    is_enabled,
    trace,
)

import numpy as np
import torch

from verl.experimental.fully_async_policy.utils.priority_sampling import get_priority_weight_fn


def pass_rate_to_weight_sequential_row_weight(pass_rate: float) -> float:
    """Scalar written to ``row_weights`` for ``weight_sequential`` (same array as priority).

    Uses ``0.5 - abs(p - 0.5)``: largest at ``p == 0.5``, zero at 0 and 1. Traversal order
    is descending ``row_weights`` (then row index), same as ascending ``abs(p - 0.5)``.
    """
    p = float(pass_rate)
    return float(0.5 - abs(p - 0.5))


def resolve_prompt_sample_strategy(async_training_cfg) -> str:
    """Map ``async_training`` config to the buffer's logical ``sample_strategy`` value.

    Returns:
        ``"shuffle_sequential"``: epoch-shuffled traversal without replacement
        (default when the config is null or omitted).
        ``"sequential"``: explicit sequential traversal in dataset row order.
        ``\"uniform\"``: uniform random sampling over rows.
        ``\"uniform_extreme_ucb\"``: weighted-random sampling that is uniform by
        default, downweights p=0/p=1 prompts by ``uniform_extreme_alpha``, and
        can add a UCB-style exploration multiplier via ``uniform_ucb_c``.
        ``\"medium_ucb\"``: weighted-random sampling using ``4*p*(1-p)`` plus
        count-based UCB exploration.
        ``\"weight_sequential\"``: ordered traversal by distance of ``pass_rate`` from 0.5.
        Any other non-empty string: registered priority weight strategy (e.g. ``medium_sharp``).

    Explicit ``\"sequential\"`` selects unshuffled row order. ``None``, ``\"\"``,
    ``\"null\"``, and ``\"none\"`` select the default ``shuffle_sequential`` mode.
    """
    ss = async_training_cfg.get("sample_strategy", None)

    def _is_uniform(x: Any) -> bool:
        return isinstance(x, str) and x.strip().lower() == "uniform"

    if _is_uniform(ss):
        return "uniform"
    if ss is None:
        return "shuffle_sequential"
    if isinstance(ss, str):
        normalized = ss.strip().lower()
        if normalized in ("", "null", "none"):
            return "shuffle_sequential"
        if normalized == "sequential":
            return "sequential"
        return ss.strip()
    return str(ss)


# ---------------------------------------------------------------------------
# Adaptive rollout_n strategy registry
# ---------------------------------------------------------------------------
# A strategy maps (pass_rate, base_n, min_n, max_n) -> int rollout_n.
# All strategies should return a RAW integer; the top-level dispatcher
# ``get_adaptive_rollout_n`` is responsible for clamping to [min_n, max_n],
# so individual strategies don't need to repeat the clip logic.

AdaptiveNFn = Callable[[float, int, int, int], int]

ADAPTIVE_N_REGISTRY: dict[str, AdaptiveNFn] = {}


def register_adaptive_n(name: str) -> Callable[[AdaptiveNFn], AdaptiveNFn]:
    """Decorator to register an adaptive rollout_n strategy under a given name.

    Usage:
        @register_adaptive_n("my_strategy")
        def _my_strategy(pass_rate, base_n, min_n, max_n):
            ...
            return n
    """

    def decorator(fn: AdaptiveNFn) -> AdaptiveNFn:
        if name in ADAPTIVE_N_REGISTRY and ADAPTIVE_N_REGISTRY[name] != fn:
            raise ValueError(
                f"Adaptive-n strategy '{name}' has already been registered: "
                f"{ADAPTIVE_N_REGISTRY[name]} vs {fn}"
            )
        ADAPTIVE_N_REGISTRY[name] = fn
        return fn

    return decorator


def get_adaptive_n_fn(name: str) -> AdaptiveNFn:
    """Look up a registered adaptive-n strategy by name."""
    if name not in ADAPTIVE_N_REGISTRY:
        raise ValueError(
            f"Invalid adaptive_rollout_n strategy: {name}. "
            f"Supported strategies are: {list(ADAPTIVE_N_REGISTRY.keys())}"
        )
    return ADAPTIVE_N_REGISTRY[name]


def get_adaptive_rollout_n(
    pass_rate: float,
    base_n: int,
    min_n: int = 1,
    max_n: int = 16,
    strategy: str = "medium_focus",
) -> int:
    """Compute adaptive rollout_n via the named strategy, clamped to [min_n, max_n].

    This is the canonical entry point for adaptive-n dispatch. All rollouter-side
    code should go through this function rather than calling strategy fns directly,
    so the clip invariant is applied uniformly.
    """
    fn = get_adaptive_n_fn(strategy)
    raw = int(fn(float(pass_rate), int(base_n), int(min_n), int(max_n)))
    return max(min_n, min(raw, max_n))


# ---------------------------------------------------------------------------
# Built-in adaptive-n strategies
# ---------------------------------------------------------------------------


@register_adaptive_n("medium_focus")
def _adaptive_n_medium_focus(pass_rate: float, base_n: int, min_n: int, max_n: int) -> int:
    """Bell-shaped: maximal N for pass_rate near 0.5, minimal for p≈0 or p≈1.

    Rationale: variance of a Bernoulli with rate p is p(1-p), peaking at 0.5.
    Prompts with the highest outcome variance provide the strongest learning
    signal per rollout, so they deserve more samples. Too-easy (p≈1) and
    too-hard (p≈0) prompts are down-sampled to save budget.

        n = min_n + 4 * p * (1-p) * (max_n - min_n)
    """
    signal_strength = 4.0 * pass_rate * (1.0 - pass_rate)  # in [0, 1]
    return int(min_n + signal_strength * (max_n - min_n))


@register_adaptive_n("inverse_bernoulli_variance")
def _adaptive_n_inverse_bernoulli_variance(pass_rate: float, base_n: int, min_n: int, max_n: int) -> int:
    """Allocate N according to ``floor(1 / (p * (1-p)) - 1)``.

    Pass rates are clipped away from 0 and 1 before taking the reciprocal.
    The registry dispatcher applies the configured [min_n, max_n] clamp.
    """
    p = min(0.99, max(0.01, float(pass_rate)))
    return math.floor(1.0 / (p * (1.0 - p)) - 1.0)


@register_adaptive_n("productive_curve")
def _adaptive_n_productive_curve(pass_rate: float, base_n: int, min_n: int, max_n: int) -> int:
    """Productive-utility curve: spend most budget on prompts likely to be mixed.

    The curve is intentionally asymmetric on the easy side:
    - p < 0.5: increase N linearly from min_n to max_n.
    - 0.5 <= p <= 0.8: keep max_n.
    - 0.8 < p <= 0.9: decrease N linearly back to min_n.
    - p > 0.9: keep min_n.

    This matches the adaptive_chunk_mixed target-N rule, but uses historical
    prompt-buffer pass_rate only because fixed_rollout decides N before the
    current rollout starts.
    """
    p = min(1.0, max(0.0, float(pass_rate)))
    span = float(max_n - min_n)
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
        target = float(max_n)
    else:
        decay = (p - high_threshold) / (easy_cutoff - high_threshold)
        target = max_n - decay * span

    return int(target)


@register_adaptive_n("medium_focus_plateau")
def _adaptive_n_medium_focus_plateau(pass_rate: float, base_n: int, min_n: int, max_n: int) -> int:
    """Medium-focus curve with a max-N plateau on the productive-easy side.

    This keeps the original medium_focus utility ``4*p*(1-p)`` everywhere
    except ``0.5 < p < 0.8``, where we allocate max_n.
    """
    p = min(1.0, max(0.0, float(pass_rate)))
    if 0.5 < p < 0.8:
        return int(max_n)
    signal_strength = 4.0 * p * (1.0 - p)
    return int(min_n + signal_strength * (max_n - min_n))


@register_adaptive_n("knapsack_value_margin50")
def _adaptive_n_knapsack_value_margin50(pass_rate: float, base_n: int, min_n: int, max_n: int) -> int:
    """Choose N where knapsack-RL value marginal gain drops to 50%.

    The offline value curve is:

        V(n, p) = (1 - p**n - (1-p)**n) * p * (1-p)**2

    For a prompt with historical pass_rate ``p``, choose the smallest even N in
    [min_n, max_n] whose one-more-sample marginal gain is at most half of the
    marginal gain at ``min_n``. Even-N search matches the chunk-aligned rollout
    setups used in these experiments and avoids odd crossing points being
    floored back to the previous chunk.
    """
    p = min(1.0, max(0.0, float(pass_rate)))
    min_n = max(1, int(min_n))
    max_n = max(min_n, int(max_n))

    def _value(n: int) -> float:
        return (1.0 - (p**n) - ((1.0 - p) ** n)) * p * ((1.0 - p) ** 2)

    initial_gain = _value(min_n) - _value(max(0, min_n - 1))
    if initial_gain <= 0.0:
        return min_n

    threshold = 0.5 * initial_gain
    first_candidate = min_n if min_n % 2 == 0 else min_n + 1
    for n in range(first_candidate, max_n + 1, 2):
        gain = _value(n) - _value(max(0, n - 1))
        if gain <= threshold:
            return n
    return max_n


@register_adaptive_n("knapsack_value_halfmax")
def _adaptive_n_knapsack_value_halfmax(pass_rate: float, base_n: int, min_n: int, max_n: int) -> int:
    """Choose the smallest even N reaching 50% of max knapsack-RL value.

    Value:

        V(n, p) = (1 - p**n - (1-p)**n) * p * (1-p)**2

    Since the p-dependent multiplier is constant with respect to n, this rule
    mainly asks how many responses are needed for a prompt to have a reasonable
    chance of producing a mixed positive/negative group. It uses the offline
    historical pass_rate and fixes N before rollout starts.
    """
    p = min(1.0, max(0.0, float(pass_rate)))
    min_n = max(1, int(min_n))
    max_n = max(min_n, int(max_n))

    def _value(n: int) -> float:
        return (1.0 - (p**n) - ((1.0 - p) ** n)) * p * ((1.0 - p) ** 2)

    target_value = 0.5 * _value(max_n)
    if target_value <= 0.0:
        return min_n

    first_candidate = min_n if min_n % 2 == 0 else min_n + 1
    for n in range(first_candidate, max_n + 1, 2):
        if _value(n) >= target_value:
            return n
    return max_n


@register_adaptive_n("mixed_prob_target")
def _adaptive_n_mixed_prob_target(pass_rate: float, base_n: int, min_n: int, max_n: int) -> int:
    """Choose the smallest even N whose mixed-group probability reaches a target.

    For a prompt with historical pass-rate ``p``, the probability that ``n``
    rollouts contain at least one positive and at least one negative sample is:

        P_mixed(n, p) = 1 - p**n - (1-p)**n

    This offline rule picks the smallest even N in [min_n, max_n] with
    ``P_mixed >= MIXED_PROB_TARGET_TAU`` (default: 0.9). If the target is not
    reachable under ``max_n``, it returns ``max_n`` directly; extreme p values
    are intentionally not special-cased for this ablation.
    """
    del base_n
    p = min(1.0, max(0.0, float(pass_rate)))
    min_n = max(1, int(min_n))
    max_n = max(min_n, int(max_n))
    try:
        tau = float(os.environ.get("MIXED_PROB_TARGET_TAU", "0.9"))
    except Exception:
        tau = 0.9
    tau = min(1.0, max(0.0, tau))

    first_candidate = min_n if min_n % 2 == 0 else min_n + 1
    for n in range(first_candidate, max_n + 1, 2):
        mixed_prob = 1.0 - (p**n) - ((1.0 - p) ** n)
        if mixed_prob >= tau:
            return n
    return max_n


@register_adaptive_n("constant_positive")
def _adaptive_n_constant_positive(pass_rate: float, base_n: int, min_n: int, max_n: int) -> int:
    """Allocate N so the expected number of *positive* rollouts stays ~constant.

    Rationale: for any prompt with pass_rate p, E[#positives] = n * p. To keep
    the positive-sample count roughly equal to ``base_n`` across prompts of
    different difficulty, set n = base_n / p. Hard prompts (small p) thus get
    upsampled, easy prompts (p≈1) stay at base_n.

    The floor ``p_floor = 1 / max_n`` prevents divide-by-zero and saturates
    extremely hard prompts at max_n rather than letting n blow up:
        p=0    -> n = base_n * max_n  (clamped to max_n by the dispatcher)
        p=0.1  -> n = 10 * base_n     (clamped to max_n if > max_n)
        p=1.0  -> n = base_n

        n = round(base_n / max(p, 1 / max_n))
    """
    p_floor = 1.0 / max(max_n, 1)
    p_eff = max(pass_rate, p_floor)
    return int(round(base_n / p_eff))


@register_adaptive_n("uniform")
def _adaptive_n_uniform(pass_rate: float, base_n: int, min_n: int, max_n: int) -> int:
    """Trivial fallback: ignore pass_rate and return base_n.

    Useful for ablation / debugging: keeps the adaptive code path live but
    effectively disables the pass_rate -> N mapping.
    """
    return int(base_n)


# ---------------------------------------------------------------------------
# Backward-compatible shim
# ---------------------------------------------------------------------------


def adaptive_rollout_n(pass_rate: float, base_n: int, min_n: int = 1, max_n: int = 16) -> int:
    """Deprecated alias: defaults to the ``medium_focus`` bell-shaped strategy.

    Kept for backward compatibility with existing callers. New code should use
    ``get_adaptive_rollout_n(pass_rate, base_n, min_n, max_n, strategy=...)``.
    """
    return get_adaptive_rollout_n(pass_rate, base_n, min_n, max_n, strategy="medium_focus")


class AsyncPromptBuffer:
    """
    Async-friendly prompt buffer with configurable row sampling.

    Designed for fully-async streaming rollout where samples are processed one at
    a time. Maintains per-prompt metadata (pass_rate, count) that is updated
    incrementally after each rollout completes.

    ``sample_strategy`` controls how dataset rows are chosen:

    - ``None`` / ``"shuffle_sequential"``: one shuffled, without-replacement
      traversal per dataset epoch.
    - ``"sequential"``: round-robin in dataset index order.
    - ``\"uniform\"``: uniform random row (optionally use ``resolve_prompt_sample_strategy``
      from config, which may map legacy keys to this mode).
    - ``\"uniform_extreme_ucb\"``: uniform-like weighted-random row sampling; p=0/p=1 rows
      are downweighted by ``uniform_extreme_alpha`` and ``uniform_ucb_c`` adds optional
      exploration for prompts with lower observed counts.
    - ``\"weight_sequential\"``: fills ``row_weights`` from ``pass_rate`` (see
      ``pass_rate_to_weight_sequential_row_weight``), then round-robin in descending
      weight order. ``meta_data`` / ``pass_rate`` update every rollout; ``row_weights`` and
      order rebuild only at **epoch** boundary (one full cycle through all rows).
    - Any registered priority name: sample rows with weights from ``pass_rate``.

    NOTE: The number of sequences to roll out per prompt is **not** decided here.
    The rollouter-side ``AsyncResponseBuffer`` (together with an optional
    ``dynamic_fixed_n_provider`` that reads ``pass_rate`` from
    ``self.meta_data[prompt_uid]``) owns that decision. This class only exposes
    the module-level adaptive-n registry (``get_adaptive_rollout_n`` /
    ``register_adaptive_n``) and the backward-compatible ``adaptive_rollout_n``
    helper for the provider to reuse.

    Args:
        dataset: The dataset to sample from (must support __len__ and __getitem__).
        collate_fn: Function to collate a single sample into a batch dict.
        seed: Random seed for reproducibility.
        sample_strategy: ``None`` / ``"shuffle_sequential"`` = epoch-shuffled
            traversal without replacement; ``"sequential"`` = dataset row order;
            ``\"uniform\"`` = uniform random;
            ``\"uniform_extreme_ucb\"`` = endpoint-penalized uniform with optional
            UCB exploration; ``\"medium_ucb\"`` = medium-focused weight plus UCB;
            ``\"weight_sequential\"`` = sorted-by-pass_rate order;
            else a registered priority weight strategy name.
        reward_ema: EMA coefficient for pass-rate updates. 0.0 means always overwrite.
        pass_rate_clip_min: Lower bound to clip updated pass_rate. Defaults to 0.0 (no clip).
        pass_rate_clip_max: Upper bound to clip updated pass_rate. Defaults to 1.0 (no clip).
        uniform_extreme_alpha: Downweight factor for p=0/p=1 under ``uniform_extreme_ucb``.
        uniform_ucb_c: Strength of UCB-style exploration multiplier under ``uniform_extreme_ucb``.
        warmup_epochs: Number of leading sampling epochs drawn with ``shuffle_sequential``
            regardless of ``sample_strategy``; ``sample_strategy`` takes over afterwards.
        use_dataset_pass_rate: If False, ignore ``extra_info.pass_rate`` so every prompt's
            pass_rate starts unknown and is learned online.
    """

    def __init__(
        self,
        dataset,
        collate_fn: Callable,
        seed: Optional[int] = None,
        sample_strategy: Optional[str] = None,
        reward_ema: float = 0.0,
        pass_rate_clip_min: float = 0.0,
        pass_rate_clip_max: float = 1.0,
        uniform_extreme_alpha: float = 0.2,
        uniform_ucb_c: float = 0.0,
        warmup_epochs: int = 0,
        use_dataset_pass_rate: bool = True,
    ):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.seed = seed
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.use_dataset_pass_rate = bool(use_dataset_pass_rate)
        self._warmup_finished_logged = self.warmup_epochs == 0
        self._warmup_drawn: dict[str, int] = {}

        # Initialize generator for reproducible sampling
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

        # --- sample_strategy: shuffle_sequential | sequential | uniform | weight_sequential | priority ---
        raw = sample_strategy
        normalized = raw.strip().lower() if isinstance(raw, str) else None
        if raw is None or normalized in ("", "null", "none", "shuffle_sequential"):
            raw = "shuffle_sequential"
        self.sample_strategy = raw
        if isinstance(raw, str) and raw.strip().lower() == "shuffle_sequential":
            self._sample_mode = "shuffle_sequential"
            self._priority_key: Optional[str] = None
            self._priority_weight_fn = None
        elif isinstance(raw, str) and raw.strip().lower() == "sequential":
            self._sample_mode = "sequential"
            self._priority_key = None
            self._priority_weight_fn = None
        elif isinstance(raw, str) and raw.strip().lower() == "uniform":
            self._sample_mode = "uniform"
            self._priority_key = None
            self._priority_weight_fn = None
        elif isinstance(raw, str) and raw.strip().lower() == "uniform_extreme_ucb":
            self._sample_mode = "uniform_extreme_ucb"
            self._priority_key = None
            self._priority_weight_fn = None
        elif isinstance(raw, str) and raw.strip().lower() == "medium_ucb":
            self._sample_mode = "medium_ucb"
            self._priority_key = None
            self._priority_weight_fn = None
        elif isinstance(raw, str) and raw.strip().lower() == "medium_to_productive_mixed":
            self._sample_mode = "medium_to_productive_mixed"
            self._priority_key = None
            self._priority_weight_fn = None
        elif isinstance(raw, str) and raw.strip().lower() == "medium_to_target_025":
            self._sample_mode = "medium_to_target_025"
            self._priority_key = None
            self._priority_weight_fn = None
        elif isinstance(raw, str) and raw.strip().lower() == "weight_sequential":
            self._sample_mode = "weight_sequential"
            self._priority_key = None
            self._priority_weight_fn = None
        else:
            key = str(raw).strip().lower()
            self._sample_mode = "priority"
            self._priority_key = key
            self._priority_weight_fn = get_priority_weight_fn(key)

        self.use_priority = self._sample_mode == "priority"
        self.use_uniform_extreme_ucb = self._sample_mode == "uniform_extreme_ucb"
        self.use_medium_ucb = self._sample_mode == "medium_ucb"
        self.use_medium_to_productive_mixed = self._sample_mode == "medium_to_productive_mixed"
        self.use_medium_to_target_025 = self._sample_mode == "medium_to_target_025"
        self.current_training_step = 0
        self.medium_to_productive_mixed_decay_steps = int(os.environ.get("MEDIUM_TO_PRODUCTIVE_MIXED_DECAY_STEPS", "400"))
        self.medium_to_target_decay_steps = int(os.environ.get("MEDIUM_TO_TARGET_DECAY_STEPS", "300"))
        self.medium_to_target_center = float(os.environ.get("MEDIUM_TO_TARGET_CENTER", "0.25"))
        self.productive_mixed_fallback_weight = float(os.environ.get("PRODUCTIVE_MIXED_FALLBACK_WEIGHT", "0.2"))
        self.productive_mixed_threshold = float(os.environ.get("PRODUCTIVE_MIXED_THRESHOLD", "0.8"))
        self.productive_mixed_min_n = int(os.environ.get("PRODUCTIVE_MIXED_MIN_N", "2"))
        self.productive_mixed_max_n = int(os.environ.get("PRODUCTIVE_MIXED_MAX_N", "16"))
        self.productive_mixed_chunk_size = int(os.environ.get("PRODUCTIVE_MIXED_CHUNK_SIZE", "2"))

        self._seq_cursor: int = 0
        self._shuffle_sequential_order: list[int] = []
        self._weight_sequential_order: list[int] = []
        self.alpha = reward_ema  # EMA coefficient: 0.0 means always overwrite

        # Per sampling-epoch coverage observability. One sampling epoch means
        # ``len(dataset)`` calls to get_single_sample(), regardless of strategy.
        # This intentionally tracks what the sampler *selected*; it does not
        # mutate pass_rate/count, which remain cumulative training state.
        self._sampling_epoch_index: int = 0
        self._sampling_epoch_draws: int = 0
        self._sampling_epoch_seen_rows: set[int] = set()
        self._sampling_epoch_seen_prompts: set[str] = set()
        self._last_completed_sampling_epoch: int = -1
        self._last_epoch_draws: int = 0
        self._last_epoch_unique_rows: int = 0
        self._last_epoch_unique_prompts: int = 0
        self._last_epoch_unseen_rows: int = 0
        self._last_epoch_unseen_prompts: int = 0

        # NOTE: Pass-rate clipping keeps updated pass_rate within [clip_min, clip_max].
        # Useful when reward_ema is small: prevents pass_rate from collapsing to exactly
        # 0 or 1 (which would zero out priority weights under e.g. the 'medium' strategy)
        # and keeps every prompt reachable by the sampler.
        if not (0.0 <= pass_rate_clip_min <= pass_rate_clip_max <= 1.0):
            raise ValueError(
                f"Invalid pass_rate clip range: [{pass_rate_clip_min}, {pass_rate_clip_max}]. "
                f"Must satisfy 0.0 <= clip_min <= clip_max <= 1.0."
            )
        self.pass_rate_clip_min = float(pass_rate_clip_min)
        self.pass_rate_clip_max = float(pass_rate_clip_max)
        if not (0.0 <= uniform_extreme_alpha <= 1.0):
            raise ValueError(
                f"Invalid uniform_extreme_alpha: {uniform_extreme_alpha}. Must satisfy 0.0 <= alpha <= 1.0."
            )
        if uniform_ucb_c < 0.0:
            raise ValueError(f"Invalid uniform_ucb_c: {uniform_ucb_c}. Must be non-negative.")
        self.uniform_extreme_alpha = float(uniform_extreme_alpha)
        self.uniform_ucb_c = float(uniform_ucb_c)

        if self._sample_mode == "priority":
            print(
                f"[AsyncPromptBuffer] sample_strategy=priority ('{self._priority_key}'), "
                f"reward_ema={reward_ema}, "
                f"pass_rate_clip=[{self.pass_rate_clip_min}, {self.pass_rate_clip_max}]"
            )
        elif self._sample_mode == "uniform":
            print("[AsyncPromptBuffer] sample_strategy=uniform (random row sampling)")
        elif self._sample_mode == "uniform_extreme_ucb":
            print(
                "[AsyncPromptBuffer] sample_strategy=uniform_extreme_ucb "
                f"(endpoint alpha={self.uniform_extreme_alpha}, ucb_c={self.uniform_ucb_c})"
            )
        elif self._sample_mode == "medium_ucb":
            print(
                "[AsyncPromptBuffer] sample_strategy=medium_ucb "
                f"(weight=4*p*(1-p)+ucb, ucb_c={self.uniform_ucb_c})"
            )
        elif self._sample_mode == "medium_to_productive_mixed":
            print(
                "[AsyncPromptBuffer] sample_strategy=medium_to_productive_mixed "
                f"(medium -> two-weight mixed by step {self.medium_to_productive_mixed_decay_steps}, "
                f"fallback={self.productive_mixed_fallback_weight}, "
                f"threshold={self.productive_mixed_threshold})"
            )
        elif self._sample_mode == "medium_to_target_025":
            print(
                "[AsyncPromptBuffer] sample_strategy=medium_to_target_025 "
                f"(medium peak=0.5 -> target peak={self.medium_to_target_center} "
                f"by step {self.medium_to_target_decay_steps})"
            )
        elif self._sample_mode == "weight_sequential":
            print(
                "[AsyncPromptBuffer] sample_strategy=weight_sequential "
                "(row_weights = 0.5-|pass_rate-0.5|, traverse descending; "
                "refresh row_weights + order once per dataset epoch)"
            )
        elif self._sample_mode == "shuffle_sequential":
            print(
                "[AsyncPromptBuffer] sample_strategy=shuffle_sequential "
                "(one random permutation per dataset epoch, without replacement)"
            )
        else:
            print("[AsyncPromptBuffer] sample_strategy=sequential (round-robin by row index)")

        # Row-wise storage: prompt_uid -> {'pass_rate': float, 'count': int, 'row_indices': list[int]}
        self.meta_data: dict[str, dict[str, float | int | list[int]]] = {}

        # Per-row priority weights array for fast sampling
        self._default_priority_weight: float = 1.0
        self.row_weights: np.ndarray = np.ones(len(dataset), dtype=np.float32)

        # Initialize pass_rates and row_indices from dataset
        self._init_pass_rates_from_dataset(dataset)

        if self.use_priority:
            self.row_weights[:] = self._default_priority_weight
            # Initialize row_weights based on pass_rates loaded from dataset
            for prompt_uid, meta in self.meta_data.items():
                if "pass_rate" in meta:
                    weight = self._priority_weight_fn(meta["pass_rate"])
                    for row_idx in meta.get("row_indices", []):
                        self.row_weights[row_idx] = weight
        elif self._sample_mode in ("uniform_extreme_ucb", "medium_ucb", "medium_to_productive_mixed", "medium_to_target_025"):
            self._sync_row_weights()
        elif self._sample_mode == "weight_sequential":
            self._sync_row_weights()
        elif self._sample_mode == "shuffle_sequential":
            self._reshuffle_sequential_order()
        if self.warmup_epochs > 0 and self._sample_mode != "shuffle_sequential":
            self._reshuffle_sequential_order()

        print(
            f"[AsyncPromptBuffer] Initialized with {len(dataset)} samples, "
            f"{len(self.meta_data)} unique prompt_uids, warmup_epochs={self.warmup_epochs} "
            f"(shuffle_sequential), use_dataset_pass_rate={self.use_dataset_pass_rate}"
        )

    def in_warmup(self) -> bool:
        """True while the leading ``warmup_epochs`` sampling epochs are being drawn."""
        return self._sampling_epoch_index < self.warmup_epochs

    def consume_warmup_draw(self, prompt_uid: str) -> bool:
        """True (once per draw) if ``prompt_uid`` was drawn during warmup."""
        pending = self._warmup_drawn.get(prompt_uid, 0)
        if pending <= 0:
            return False
        if pending == 1:
            del self._warmup_drawn[prompt_uid]
        else:
            self._warmup_drawn[prompt_uid] = pending - 1
        return True

    def _on_warmup_finished(self) -> None:
        """Hand over to ``sample_strategy`` using the pass rates learned during warmup."""
        self._warmup_finished_logged = True
        unobserved = 0
        if self._sample_mode != "shuffle_sequential":
            self._sync_row_weights()
        if self.use_priority:
            # Prompts still in flight from warmup have no pass_rate yet; weight them as p=0.5
            # without writing a fake observation into the EMA.
            unobserved_weight = self._priority_weight_fn(0.5)
            for meta in self.meta_data.values():
                if "pass_rate" not in meta:
                    unobserved += 1
                    for row_idx in meta.get("row_indices", []):
                        self.row_weights[row_idx] = unobserved_weight
        rates = [float(m["pass_rate"]) for m in self.meta_data.values() if "pass_rate" in m]
        print(
            f"[AsyncPromptBuffer][Warmup] finished after {self.warmup_epochs} epoch(s); switching to "
            f"sample_strategy={self.sample_strategy}. prompts_with_pass_rate={len(rates)}/{len(self.meta_data)}, "
            f"unobserved_weighted_as_0.5={unobserved}, mean_pass_rate={np.mean(rates) if rates else float('nan'):.4f}, "
            f"zero_frac={np.mean([r == 0.0 for r in rates]) if rates else float('nan'):.4f}",
            flush=True,
        )

    def get_single_sample(self) -> tuple[dict, str, float]:
        """Sample a single prompt from the buffer.

        Returns:
            tuple of (batch_dict, prompt_uid, sampling_prob):
                - batch_dict: Collated single-sample dict ready for prepare_single_generation_data
                - prompt_uid: Unique identifier for this prompt
                - sampling_prob: The sampling probability of this prompt (for IS correction)
        """
        eps = 0.001
        n = len(self.dataset)

        warmup = self.in_warmup()
        if not warmup and not self._warmup_finished_logged:
            self._on_warmup_finished()
        mode = "shuffle_sequential" if warmup else self._sample_mode

        if mode == "shuffle_sequential":
            if n > 0:
                position = self._seq_cursor % n
                if len(self._shuffle_sequential_order) != n:
                    self._reshuffle_sequential_order()
                elif position == 0 and self._seq_cursor > 0:
                    self._reshuffle_sequential_order()
                idx = self._shuffle_sequential_order[position]
                self._seq_cursor += 1
                sampling_prob = 1.0 / n
            else:
                idx = 0
                sampling_prob = 0.0
        elif mode == "sequential":
            if n > 0:
                idx = self._seq_cursor % n
                self._seq_cursor += 1
                sampling_prob = 1.0 / n
            else:
                idx = 0
                sampling_prob = 0.0
        elif mode == "weight_sequential":
            if n > 0 and self._weight_sequential_order:
                order_len = len(self._weight_sequential_order)
                pos = self._seq_cursor % order_len
                idx = self._weight_sequential_order[pos]
                self._seq_cursor += 1
                sampling_prob = 1.0 / n
                # One epoch = one full pass over rows in the frozen order; then resync weights.
                if self._seq_cursor % order_len == 0:
                    self._sync_row_weights()
            else:
                idx = 0
                sampling_prob = 0.0
        elif mode == "uniform":
            idx = torch.randint(0, n, (1,), generator=self.generator).item() if n > 0 else 0
            sampling_prob = 1.0 / n if n > 0 else 0.0
        else:
            # Priority-weighted sampling
            sampling_weights = torch.from_numpy(self.row_weights).float() + eps
            sampling_probs = sampling_weights / sampling_weights.sum()
            idx_tensor = torch.multinomial(sampling_probs, 1, replacement=True, generator=self.generator)
            idx = idx_tensor.item()
            sampling_prob = sampling_probs[idx].item()

        # Get sample from dataset
        sample = self.dataset[idx]

        # Extract prompt_uid from the dataset's extra_info["prompt_uid"] field
        extra_info = sample.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except (json.JSONDecodeError, TypeError):
                extra_info = {}
        prompt_uid = str(extra_info.get("prompt_uid", idx))

        # Attach metadata to sample
        sample["sampling_prob"] = sampling_prob
        sample["prompt_uid"] = prompt_uid
        sample["pass_rate"] = self.meta_data.get(prompt_uid, {}).get("pass_rate", 0.5)

        if warmup:
            self._warmup_drawn[prompt_uid] = self._warmup_drawn.get(prompt_uid, 0) + 1
        self._record_sampling_epoch_draw(idx, prompt_uid)

        trace(
            CHANNEL_PRIORITY,
            "draw",
            uid=prompt_uid,
            mode=mode,
            row=idx,
            p=sample["pass_rate"],
            sampling_prob=sampling_prob,
            weight=float(self.row_weights[idx]) if idx < len(self.row_weights) else None,
            count=self.meta_data.get(prompt_uid, {}).get("count", 0),
            cursor=self._seq_cursor,
            step=self.current_training_step,
        )

        # Collate single sample into batch dict
        batch_dict = self.collate_fn([sample])

        return batch_dict, prompt_uid, sampling_prob

    def on_sample_end(self, prompt_uid: str, reward_scores: list[float]) -> None:
        """Update buffer after a single prompt's rollout completes.

        Called immediately after rollout finishes, before putting the sample
        into the message queue. This enables real-time priority updates.

        For ``weight_sequential``, only ``meta_data`` / ``pass_rate`` is updated here;
        ``row_weights`` and visit order refresh at epoch end in ``get_single_sample``.
        For non-priority modes otherwise, ``row_weights`` stay at defaults.

        Args:
            prompt_uid: The prompt identifier.
            reward_scores: List of reward scores from the n rollouts.
                Expected to be in [0, 1] for binary correctness rewards.
        """
        if not reward_scores:
            return

        # Compute pass_rate from reward scores
        # For binary rewards (0/1), pass_rate = fraction of correct answers
        pass_rate = sum(1.0 for r in reward_scores if r > 0) / len(reward_scores)
        trace(
            CHANNEL_PRIORITY,
            "feedback",
            uid=prompt_uid,
            # The feedback group is the FULL emitted group, so compensation
            # samples are included in the pass_rate that drives the next
            # initial-n and priority weight.
            group_size=len(reward_scores),
            positives=sum(1 for r in reward_scores if r > 0),
            observed_p=pass_rate,
            prior_p=self.meta_data.get(prompt_uid, {}).get("pass_rate"),
        )
        self._update_meta_data(prompt_uid, pass_rate, count=1)

    def set_training_step(self, step: int) -> None:
        """Update trainer-step-dependent sampler state."""
        step = max(0, int(step))
        if step == self.current_training_step:
            return
        self.current_training_step = step
        if self.use_medium_to_productive_mixed or self.use_medium_to_target_025:
            self._sync_row_weights()
        else:
            # Priority modes patch row_weights per prompt instead of resyncing,
            # so take an explicit per-step snapshot of the distribution.
            self._trace_row_weights("training_step")

    def get_dataset_info(self) -> dict[str, float]:
        """Return priority-related dataset-level metrics for logging."""
        seen_meta = [m for m in self.meta_data.values() if m.get("count", 0) > 0]
        seen_pass_rates = [m["pass_rate"] for m in seen_meta]
        seen_counts = [m["count"] for m in seen_meta]

        num_seen = len(seen_meta)
        num_total = len(self.dataset)
        num_zero_count = num_total - num_seen

        info: dict[str, float] = {
            "prompt_buffer/dataset_size": num_total,
            "prompt_buffer/unique_seen_prompts_total": num_seen,
            "prompt_buffer/unique_seen_ratio": num_seen / num_total if num_total > 0 else 0.0,
            "prompt_buffer/zero_count_ratio": num_zero_count / num_total if num_total > 0 else 0.0,
            "prompt_buffer/zero_count_number": num_zero_count,
            "prompt_buffer/warmup_active": float(self.in_warmup()),
        }
        info.update(self._sampling_epoch_info())

        if seen_pass_rates:
            info.update({
                "prompt_buffer/score_for_uniform_dist": float(np.mean(seen_pass_rates)),
                "prompt_buffer/count_max": max(seen_counts),
                "prompt_buffer/count_mean": float(np.mean(seen_counts)),
                "prompt_buffer/count_min": min(seen_counts),
                "prompt_buffer/count_std": float(np.std(seen_counts)),
            })

            # Classify pass_rates into buckets (aligned with batch_metrics.py)
            # extremely_hard: p<=0.01, hard: 0.01<p<=0.2,
            # medium: 0.2<p<0.8, easy: 0.8<=p<0.99, extremely_easy: p>=0.99.
            buckets = {
                "extremely_hard": 0,
                "hard": 0,
                "medium": 0,
                "easy": 0,
                "extremely_easy": 0,
            }
            for pr in seen_pass_rates:
                if pr <= 0.01:
                    buckets["extremely_hard"] += 1
                elif pr <= 0.2:
                    buckets["hard"] += 1
                elif pr < 0.8:
                    buckets["medium"] += 1
                elif pr < 0.99:
                    buckets["easy"] += 1
                else:
                    buckets["extremely_easy"] += 1
            for bucket_name, count in buckets.items():
                info[f"prompt_buffer/classify/{bucket_name}"] = count
                info[f"prompt_buffer/classify/{bucket_name}_ratio"] = count / num_total if num_total > 0 else 0.0

        return info

    def state_dict(self) -> dict:
        """Get the current state for checkpointing.

        Returns:
            Dictionary containing generator state and meta_data (pass_rate, count).
        """
        state = {
            "generator_state": self.generator.get_state(),
            "seed": self.seed,
            "dataset_len": len(self.dataset),
            "seq_cursor": self._seq_cursor,
            "shuffle_sequential_order": list(self._shuffle_sequential_order),
            "current_training_step": self.current_training_step,
            "sampling_epoch_index": self._sampling_epoch_index,
            "sampling_epoch_draws": self._sampling_epoch_draws,
            "sampling_epoch_seen_rows": sorted(self._sampling_epoch_seen_rows),
            "sampling_epoch_seen_prompts": sorted(self._sampling_epoch_seen_prompts),
            "last_completed_sampling_epoch": self._last_completed_sampling_epoch,
            "last_epoch_draws": self._last_epoch_draws,
            "last_epoch_unique_rows": self._last_epoch_unique_rows,
            "last_epoch_unique_prompts": self._last_epoch_unique_prompts,
            "last_epoch_unseen_rows": self._last_epoch_unseen_rows,
            "last_epoch_unseen_prompts": self._last_epoch_unseen_prompts,
            # Deep copy meta_data including row_indices for debug inspection.
            # row_indices will NOT be restored on load – see _merge_meta_data_from_state.
            # NOTE: sample_strategy / alpha are intentionally NOT saved;
            # the values from __init__ (config) always take precedence.
            "meta_data": {k: dict(v) for k, v in self.meta_data.items()},
        }
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from a checkpoint.

        Args:
            state_dict: State dictionary from state_dict().
        """
        self.generator.set_state(state_dict["generator_state"])
        if "seq_cursor" in state_dict:
            self._seq_cursor = int(state_dict["seq_cursor"])
        if self._sample_mode == "shuffle_sequential":
            saved_order = state_dict.get("shuffle_sequential_order")
            if self._is_valid_shuffle_order(saved_order):
                self._shuffle_sequential_order = [int(x) for x in saved_order]
            else:
                # Backward compatibility for checkpoints created before the
                # shuffled order was persisted, or after a dataset-size change.
                self._reshuffle_sequential_order()
        if "current_training_step" in state_dict:
            self.current_training_step = int(state_dict["current_training_step"])
        if "sampling_epoch_index" in state_dict:
            self._sampling_epoch_index = int(state_dict["sampling_epoch_index"])
        if "sampling_epoch_draws" in state_dict:
            self._sampling_epoch_draws = int(state_dict["sampling_epoch_draws"])
        if "sampling_epoch_seen_rows" in state_dict:
            self._sampling_epoch_seen_rows = set(int(x) for x in state_dict["sampling_epoch_seen_rows"])
        if "sampling_epoch_seen_prompts" in state_dict:
            self._sampling_epoch_seen_prompts = set(str(x) for x in state_dict["sampling_epoch_seen_prompts"])
        if "last_completed_sampling_epoch" in state_dict:
            self._last_completed_sampling_epoch = int(state_dict["last_completed_sampling_epoch"])
        if "last_epoch_draws" in state_dict:
            self._last_epoch_draws = int(state_dict["last_epoch_draws"])
        if "last_epoch_unique_rows" in state_dict:
            self._last_epoch_unique_rows = int(state_dict["last_epoch_unique_rows"])
        if "last_epoch_unique_prompts" in state_dict:
            self._last_epoch_unique_prompts = int(state_dict["last_epoch_unique_prompts"])
        if "last_epoch_unseen_rows" in state_dict:
            self._last_epoch_unseen_rows = int(state_dict["last_epoch_unseen_rows"])
        if "last_epoch_unseen_prompts" in state_dict:
            self._last_epoch_unseen_prompts = int(state_dict["last_epoch_unseen_prompts"])

        # Restore meta_data: merge checkpoint pass_rate/count into the current
        # dataset's meta_data rather than blindly overwriting.
        if "meta_data" in state_dict:
            self._merge_meta_data_from_state(state_dict["meta_data"])

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    # ---------------------------------------------------------------------------
    # Internal methods
    # ---------------------------------------------------------------------------

    def _is_valid_shuffle_order(self, order: Any) -> bool:
        """Return whether ``order`` is a full permutation of current row indices."""
        n = len(self.dataset)
        if not isinstance(order, (list, tuple)) or len(order) != n:
            return False
        try:
            normalized = [int(x) for x in order]
        except (TypeError, ValueError):
            return False
        return len(set(normalized)) == n and (n == 0 or (min(normalized) == 0 and max(normalized) == n - 1))

    def _reshuffle_sequential_order(self) -> None:
        """Generate the next epoch's deterministic, without-replacement order."""
        n = len(self.dataset)
        self._shuffle_sequential_order = (
            torch.randperm(n, generator=self.generator).tolist() if n > 0 else []
        )
        trace(
            CHANNEL_PRIORITY,
            "reshuffle",
            mode=self._sample_mode,
            rows=n,
            cursor=self._seq_cursor,
            step=self.current_training_step,
        )

    def _record_sampling_epoch_draw(self, row_idx: int, prompt_uid: str) -> None:
        """Track per-epoch sampler coverage and reset after one dataset-sized pass."""
        n_rows = len(self.dataset)
        if n_rows <= 0:
            return

        self._sampling_epoch_draws += 1
        self._sampling_epoch_seen_rows.add(int(row_idx))
        self._sampling_epoch_seen_prompts.add(str(prompt_uid))

        if self._sampling_epoch_draws < n_rows:
            return

        total_prompts = max(1, len(self.meta_data))
        self._last_completed_sampling_epoch = self._sampling_epoch_index
        self._last_epoch_draws = self._sampling_epoch_draws
        self._last_epoch_unique_rows = len(self._sampling_epoch_seen_rows)
        self._last_epoch_unique_prompts = len(self._sampling_epoch_seen_prompts)
        self._last_epoch_unseen_rows = max(0, n_rows - self._last_epoch_unique_rows)
        self._last_epoch_unseen_prompts = max(0, len(self.meta_data) - self._last_epoch_unique_prompts)

        print(
            "[AsyncPromptBuffer][EpochCoverage] "
            f"epoch={self._last_completed_sampling_epoch}, draws={self._last_epoch_draws}, "
            f"unique_rows={self._last_epoch_unique_rows}/{n_rows} "
            f"({self._last_epoch_unique_rows / n_rows:.4f}), "
            f"unique_prompts={self._last_epoch_unique_prompts}/{len(self.meta_data)} "
            f"({self._last_epoch_unique_prompts / total_prompts:.4f}), "
            f"unseen_rows={self._last_epoch_unseen_rows}, "
            f"unseen_prompts={self._last_epoch_unseen_prompts}"
        )

        self._sampling_epoch_index += 1
        self._sampling_epoch_draws = 0
        self._sampling_epoch_seen_rows.clear()
        self._sampling_epoch_seen_prompts.clear()

    def _sampling_epoch_info(self) -> dict[str, float]:
        """Return current and last-completed sampling-epoch coverage metrics."""
        n_rows = len(self.dataset)
        n_prompts = len(self.meta_data)
        cur_rows = len(self._sampling_epoch_seen_rows)
        cur_prompts = len(self._sampling_epoch_seen_prompts)
        last_complete = 1.0 if self._last_completed_sampling_epoch >= 0 else 0.0
        return {
            "prompt_buffer/epoch/current_index": float(self._sampling_epoch_index),
            "prompt_buffer/epoch/current_draws": float(self._sampling_epoch_draws),
            "prompt_buffer/epoch/current_unique_rows": float(cur_rows),
            "prompt_buffer/epoch/current_unique_row_ratio": cur_rows / n_rows if n_rows > 0 else 0.0,
            "prompt_buffer/epoch/current_unseen_rows": float(max(0, n_rows - cur_rows)),
            "prompt_buffer/epoch/current_unique_prompts": float(cur_prompts),
            "prompt_buffer/epoch/current_unique_prompt_ratio": cur_prompts / n_prompts if n_prompts > 0 else 0.0,
            "prompt_buffer/epoch/current_unseen_prompts": float(max(0, n_prompts - cur_prompts)),
            "prompt_buffer/epoch/last_completed": last_complete,
            "prompt_buffer/epoch/last_index": float(self._last_completed_sampling_epoch),
            "prompt_buffer/epoch/last_draws": float(self._last_epoch_draws),
            "prompt_buffer/epoch/last_unique_rows": float(self._last_epoch_unique_rows),
            "prompt_buffer/epoch/last_unique_row_ratio": (
                self._last_epoch_unique_rows / n_rows if n_rows > 0 and last_complete else 0.0
            ),
            "prompt_buffer/epoch/last_unseen_rows": float(self._last_epoch_unseen_rows),
            "prompt_buffer/epoch/last_unique_prompts": float(self._last_epoch_unique_prompts),
            "prompt_buffer/epoch/last_unique_prompt_ratio": (
                self._last_epoch_unique_prompts / n_prompts if n_prompts > 0 and last_complete else 0.0
            ),
            "prompt_buffer/epoch/last_unseen_prompts": float(self._last_epoch_unseen_prompts),
        }

    def _rebuild_weight_sequential_order_from_row_weights(self) -> None:
        """Derive traversal permutation from ``self.row_weights`` (descending weight, then row index)."""
        n = len(self.row_weights)
        if n <= 0:
            self._weight_sequential_order = []
            return
        rows = list(range(n))
        rows.sort(key=lambda i: (-float(self.row_weights[i]), i))
        self._weight_sequential_order = rows

    def _init_pass_rates_from_dataset(self, dataset) -> None:
        """Initialize pass_rates and row_indices mapping from dataset.

        Iterates over every row in the dataset's dataframe to:
        1. Build ``prompt_uid -> row_indices`` reverse index in ``meta_data``.
        2. Initialize ``meta_data`` pass_rate entries (for rows that have one).
        """
        has_extra_info = (
            hasattr(dataset, "dataframe")
            and hasattr(dataset.dataframe, "column_names")
            and "extra_info" in dataset.dataframe.column_names
        )

        if not has_extra_info:
            # Fallback: build row_indices using row index as prompt_uid
            if not hasattr(dataset, "dataframe"):
                print("[AsyncPromptBuffer] Dataset has no dataframe attribute, building row_indices with fallback")
            else:
                print("[AsyncPromptBuffer] Dataset has no extra_info column, building row_indices with fallback")
            for i in range(len(dataset)):
                prompt_uid = str(i)
                self.meta_data.setdefault(prompt_uid, {})
                self.meta_data[prompt_uid].setdefault("row_indices", []).append(i)
            return

        extra_info_col = dataset.dataframe["extra_info"]
        num_init_samples = 0
        for i, extra_info in enumerate(extra_info_col):
            prompt_uid = str(i)  # default fallback

            if extra_info is not None:
                if isinstance(extra_info, str):
                    try:
                        extra_info = json.loads(extra_info)
                    except (json.JSONDecodeError, TypeError):
                        extra_info = None

            if isinstance(extra_info, dict):
                prompt_uid = str(extra_info.get("prompt_uid", i))
                pr = extra_info.get("pass_rate") if self.use_dataset_pass_rate else None
                if pr is not None and not (isinstance(pr, float) and np.isnan(pr)):
                    if prompt_uid not in self.meta_data:
                        self.meta_data[prompt_uid] = {}
                    self.meta_data[prompt_uid]["pass_rate"] = float(
                        np.clip(float(pr), self.pass_rate_clip_min, self.pass_rate_clip_max)
                    )
                    self.meta_data[prompt_uid]["count"] = 0
                    num_init_samples += 1

            # Always register the row_idx -> prompt_uid reverse index
            self.meta_data.setdefault(prompt_uid, {})
            self.meta_data[prompt_uid].setdefault("row_indices", []).append(i)

        print(f"[AsyncPromptBuffer] Initialized pass_rates from dataset with {num_init_samples} samples")

    def _merge_meta_data_from_state(self, saved_meta: dict[str, dict]) -> None:
        """Merge checkpoint meta_data into the current dataset's meta_data.

        Only ``pass_rate`` and ``count`` are restored from the checkpoint.
        ``row_indices`` always comes from the *current* dataset, so it is never overwritten.
        After merging, ``row_weights`` is rebuilt to stay in sync.
        """
        merged_count = 0
        skipped_count = 0

        for prompt_uid, saved_entry in saved_meta.items():
            if prompt_uid not in self.meta_data:
                # prompt_uid does not exist in the current dataset – skip it.
                skipped_count += 1
                continue

            cur = self.meta_data[prompt_uid]
            if "pass_rate" in saved_entry:
                cur["pass_rate"] = saved_entry["pass_rate"]
            if "count" in saved_entry:
                cur["count"] = saved_entry["count"]
            merged_count += 1

        # Rebuild row_weights from the (potentially updated) meta_data
        self._sync_row_weights()

        print(
            f"[AsyncPromptBuffer] Merged meta_data from checkpoint: {merged_count} prompt_uids restored, "
            f"{skipped_count} prompt_uids skipped (not in current dataset)"
        )

    def _uniform_extreme_ucb_weight(self, pass_rate: float, count: int, log_t: float) -> float:
        """Uniform-like weight with endpoint penalty and optional count-based exploration."""
        p = float(pass_rate)
        endpoint_factor = self.uniform_extreme_alpha if p <= 0.0 or p >= 1.0 else 1.0
        if self.uniform_ucb_c <= 0.0:
            return endpoint_factor

        n_i = max(1, int(count))
        exploration = self.uniform_ucb_c * float(np.sqrt(log_t / n_i))
        return endpoint_factor * (1.0 + exploration)

    def _medium_ucb_weight(self, pass_rate: float, count: int, log_t: float) -> float:
        """Medium-focused utility with a count-based UCB exploration bonus."""
        p = min(0.99, max(0.01, float(pass_rate)))
        exploitation = 4.0 * p * (1.0 - p)
        n_i = max(1, int(count))
        exploration = self.uniform_ucb_c * float(np.sqrt(log_t / n_i)) if self.uniform_ucb_c > 0.0 else 0.0
        return exploitation + exploration

    def _medium_focus_target_n(self, pass_rate: float) -> int:
        p = min(0.99, max(0.01, float(pass_rate)))
        min_n = int(self.productive_mixed_min_n)
        max_n = int(self.productive_mixed_max_n)
        chunk_size = max(1, int(self.productive_mixed_chunk_size))
        target = min_n + (4.0 * p * (1.0 - p)) * (max_n - min_n)
        target = int(math.floor(float(target) / float(chunk_size)) * chunk_size)
        return max(min_n, min(target, max_n))

    def _productive_mixed_two_weight(self, pass_rate: float) -> float:
        p = min(0.99, max(0.01, float(pass_rate)))
        n = self._medium_focus_target_n(p)
        mixed_prob = 1.0 - (p ** n) - ((1.0 - p) ** n)
        return 1.0 if mixed_prob >= self.productive_mixed_threshold else self.productive_mixed_fallback_weight

    def _medium_to_productive_mixed_weight(self, pass_rate: float) -> float:
        p = min(0.99, max(0.01, float(pass_rate)))
        medium_weight = 4.0 * p * (1.0 - p)
        productive_weight = self._productive_mixed_two_weight(p)
        decay_steps = max(1, int(self.medium_to_productive_mixed_decay_steps))
        ratio = min(1.0, max(0.0, float(self.current_training_step) / float(decay_steps)))
        return (1.0 - ratio) * medium_weight + ratio * productive_weight

    def _target_center_weight(self, pass_rate: float, center: float) -> float:
        p = min(0.99, max(0.01, float(pass_rate)))
        c = min(0.99, max(0.01, float(center)))
        return (p / c) ** c * ((1.0 - p) / (1.0 - c)) ** (1.0 - c)

    def _medium_to_target_025_weight(self, pass_rate: float) -> float:
        p = min(0.99, max(0.01, float(pass_rate)))
        medium_weight = 4.0 * p * (1.0 - p)
        target_weight = self._target_center_weight(p, self.medium_to_target_center)
        decay_steps = max(1, int(self.medium_to_target_decay_steps))
        ratio = min(1.0, max(0.0, float(self.current_training_step) / float(decay_steps)))
        return (1.0 - ratio) * medium_weight + ratio * target_weight

    def _trace_row_weights(self, reason: str) -> None:
        """Emit a weight-distribution snapshot so degenerate priorities are visible."""
        if not is_enabled(CHANNEL_PRIORITY):
            return
        weights = np.asarray(self.row_weights, dtype=float)
        if weights.size == 0:
            return
        total = float(weights.sum())
        top_k = max(1, int(weights.size * 0.01))
        top_share = float(np.sort(weights)[-top_k:].sum() / total) if total > 0 else 0.0
        trace(
            CHANNEL_PRIORITY,
            "row_weights",
            reason=reason,
            mode=self._sample_mode,
            rows=int(weights.size),
            total=total,
            min=float(weights.min()),
            max=float(weights.max()),
            mean=float(weights.mean()),
            zeros=int((weights <= 0.0).sum()),
            top1pct_share=top_share,
            step=self.current_training_step,
        )

    def _sync_row_weights(self) -> None:
        """Rebuild ``self.row_weights`` from ``self.meta_data``."""
        self.row_weights[:] = self._default_priority_weight

        if self.use_priority:
            for prompt_uid, meta in self.meta_data.items():
                if "pass_rate" in meta:
                    weight = self._priority_weight_fn(meta["pass_rate"])
                    for row_idx in meta.get("row_indices", []):
                        self.row_weights[row_idx] = weight
            self._trace_row_weights("sync_priority")
            return

        if self._sample_mode == "uniform_extreme_ucb":
            total_count = sum(int(meta.get("count", 0)) for meta in self.meta_data.values())
            log_t = float(np.log(max(2, total_count)))
            for prompt_uid, meta in self.meta_data.items():
                pr = float(meta["pass_rate"]) if "pass_rate" in meta else 0.5
                count = int(meta.get("count", 0))
                weight = self._uniform_extreme_ucb_weight(pr, count, log_t)
                for row_idx in meta.get("row_indices", []):
                    self.row_weights[row_idx] = weight
            return

        if self._sample_mode == "medium_ucb":
            total_count = sum(int(meta.get("count", 0)) for meta in self.meta_data.values())
            log_t = float(np.log(max(2, total_count)))
            for prompt_uid, meta in self.meta_data.items():
                pr = float(meta["pass_rate"]) if "pass_rate" in meta else 0.5
                count = int(meta.get("count", 0))
                weight = self._medium_ucb_weight(pr, count, log_t)
                for row_idx in meta.get("row_indices", []):
                    self.row_weights[row_idx] = weight
            return

        if self._sample_mode == "medium_to_productive_mixed":
            for prompt_uid, meta in self.meta_data.items():
                pr = float(meta["pass_rate"]) if "pass_rate" in meta else 0.5
                weight = self._medium_to_productive_mixed_weight(pr)
                for row_idx in meta.get("row_indices", []):
                    self.row_weights[row_idx] = weight
            return

        if self._sample_mode == "medium_to_target_025":
            for prompt_uid, meta in self.meta_data.items():
                pr = float(meta["pass_rate"]) if "pass_rate" in meta else 0.5
                weight = self._medium_to_target_025_weight(pr)
                for row_idx in meta.get("row_indices", []):
                    self.row_weights[row_idx] = weight
            return

        if self._sample_mode == "weight_sequential":
            for prompt_uid, meta in self.meta_data.items():
                pr = float(meta["pass_rate"]) if "pass_rate" in meta else 0.5
                w = pass_rate_to_weight_sequential_row_weight(pr)
                for row_idx in meta.get("row_indices", []):
                    self.row_weights[row_idx] = w
            self._rebuild_weight_sequential_order_from_row_weights()

        self._trace_row_weights(f"sync_{self._sample_mode}")

    def _update_meta_data(self, prompt_uid: str, pass_rate: float, count: int) -> None:
        """Apply exponential moving average update for a single sample's pass_rate."""
        old_pass_rate = self.meta_data.get(prompt_uid, {}).get("pass_rate")
        if prompt_uid not in self.meta_data:
            self.meta_data[prompt_uid] = {
                "pass_rate": pass_rate,
                "count": count,
            }
        elif "pass_rate" not in self.meta_data[prompt_uid]:
            self.meta_data[prompt_uid]["pass_rate"] = pass_rate
            self.meta_data[prompt_uid]["count"] = count
        else:
            self.meta_data[prompt_uid]["pass_rate"] = (
                self.meta_data[prompt_uid]["pass_rate"] * self.alpha + pass_rate * (1 - self.alpha)
            )
            self.meta_data[prompt_uid]["count"] = self.meta_data[prompt_uid].get("count", 0) + count

        # NOTE: Clip the (possibly freshly-initialized or EMA-updated) pass_rate so that
        # priority weights never collapse to exactly zero for consistently easy/hard prompts.
        self.meta_data[prompt_uid]["pass_rate"] = float(
            np.clip(self.meta_data[prompt_uid]["pass_rate"], self.pass_rate_clip_min, self.pass_rate_clip_max)
        )

        new_pass_rate = float(self.meta_data[prompt_uid]["pass_rate"])
        new_weight = None
        if (
            self.use_uniform_extreme_ucb
            or self.use_medium_ucb
            or self.use_medium_to_productive_mixed
            or self.use_medium_to_target_025
        ):
            self._sync_row_weights()
        elif self.use_priority:
            weight = self._priority_weight_fn(new_pass_rate)
            new_weight = weight
            for row_idx in self.meta_data[prompt_uid].get("row_indices", []):
                self.row_weights[row_idx] = weight

        trace(
            CHANNEL_PRIORITY,
            "pass_rate_update",
            uid=prompt_uid,
            observed=pass_rate,
            old=old_pass_rate,
            new=new_pass_rate,
            ema_alpha=self.alpha,
            clip_lo=self.pass_rate_clip_min,
            clip_hi=self.pass_rate_clip_max,
            clipped=new_pass_rate in (self.pass_rate_clip_min, self.pass_rate_clip_max),
            count=self.meta_data[prompt_uid].get("count", 0),
            weight=new_weight,
            rows=len(self.meta_data[prompt_uid].get("row_indices", [])),
        )
