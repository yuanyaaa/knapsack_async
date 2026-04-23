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
- Priority-based prompt selection (weighted by pass_rate)
- Per-prompt pass_rate / count tracking (updated after each rollout via on_sample_end)
- Stateful checkpointing via state_dict() / load_state_dict()

Unlike the batch-oriented PromptBuffer in hytuner, this buffer is designed for
single-sample streaming: each call to get_single_sample() returns one prompt,
and on_sample_end() updates the buffer immediately after each rollout completes.

When priority_strategy=None, the buffer degrades to uniform random sampling,
which is functionally equivalent to the original StatefulDataLoader-based approach.

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
from typing import Any, Callable, Optional

import numpy as np
import torch

from verl.experimental.fully_async_policy.utils.priority_sampling import get_priority_weight_fn


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
    Async-friendly prompt buffer with priority sampling.

    Designed for fully-async streaming rollout where samples are processed one at
    a time. Maintains per-prompt metadata (pass_rate, count) that is updated
    incrementally after each rollout completes, and exposes priority-weighted
    single-prompt sampling.

    When ``priority_strategy=None``, the buffer degrades to uniform random
    sampling, functionally equivalent to the original ``StatefulDataLoader``-based
    approach and making this class a drop-in replacement.

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
        priority_strategy: Priority weighting strategy name. None disables priority sampling.
        reward_ema: EMA coefficient for pass-rate updates. 0.0 means always overwrite.
        pass_rate_clip_min: Lower bound to clip updated pass_rate. Defaults to 0.0 (no clip).
        pass_rate_clip_max: Upper bound to clip updated pass_rate. Defaults to 1.0 (no clip).
    """

    def __init__(
        self,
        dataset,
        collate_fn: Callable,
        seed: Optional[int] = None,
        priority_strategy: Optional[str] = None,
        reward_ema: float = 0.0,
        pass_rate_clip_min: float = 0.0,
        pass_rate_clip_max: float = 1.0,
    ):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.seed = seed

        # Initialize generator for reproducible sampling
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

        # Priority sampling setup
        self.use_priority = priority_strategy is not None
        self.priority_strategy = priority_strategy
        self._priority_weight_fn = get_priority_weight_fn(priority_strategy) if self.use_priority else None
        self.alpha = reward_ema  # EMA coefficient: 0.0 means always overwrite

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

        if self.use_priority:
            print(
                f"[AsyncPromptBuffer] Priority sampling enabled with strategy='{priority_strategy}', "
                f"reward_ema={reward_ema}, "
                f"pass_rate_clip=[{self.pass_rate_clip_min}, {self.pass_rate_clip_max}]"
            )
        else:
            print("[AsyncPromptBuffer] Priority sampling disabled, using uniform random sampling")

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

        print(
            f"[AsyncPromptBuffer] Initialized with {len(dataset)} samples, "
            f"{len(self.meta_data)} unique prompt_uids"
        )

    def get_single_sample(self) -> tuple[dict, str, float]:
        """Sample a single prompt from the buffer.

        When priority sampling is disabled, this performs uniform random sampling,
        which is equivalent to the original dataloader behavior.

        Returns:
            tuple of (batch_dict, prompt_uid, sampling_prob):
                - batch_dict: Collated single-sample dict ready for prepare_single_generation_data
                - prompt_uid: Unique identifier for this prompt
                - sampling_prob: The sampling probability of this prompt (for IS correction)
        """
        eps = 0.001

        if self.use_priority:
            # Priority-weighted sampling
            sampling_weights = torch.from_numpy(self.row_weights).float() + eps
            sampling_probs = sampling_weights / sampling_weights.sum()
            idx_tensor = torch.multinomial(sampling_probs, 1, replacement=True, generator=self.generator)
            idx = idx_tensor.item()
            sampling_prob = sampling_probs[idx].item()
        else:
            # Uniform random sampling (equivalent to original dataloader)
            idx = torch.randint(0, len(self.dataset), (1,), generator=self.generator).item()
            sampling_prob = 1.0 / len(self.dataset) if len(self.dataset) > 0 else 0.0

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

        # Collate single sample into batch dict
        batch_dict = self.collate_fn([sample])

        return batch_dict, prompt_uid, sampling_prob

    def on_sample_end(self, prompt_uid: str, reward_scores: list[float]) -> None:
        """Update buffer after a single prompt's rollout completes.

        Called immediately after rollout finishes, before putting the sample
        into the message queue. This enables real-time priority updates.

        When priority sampling is disabled, this still tracks pass_rate/count
        for logging purposes, but does not affect sampling weights.

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
        self._update_meta_data(prompt_uid, pass_rate, count=1)

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
            "prompt_buffer/zero_count_ratio": num_zero_count / num_total if num_total > 0 else 0.0,
            "prompt_buffer/zero_count_number": num_zero_count,
        }

        if seen_pass_rates:
            info.update({
                "prompt_buffer/score_for_uniform_dist": float(np.mean(seen_pass_rates)),
                "prompt_buffer/count_max": max(seen_counts),
                "prompt_buffer/count_mean": float(np.mean(seen_counts)),
                "prompt_buffer/count_min": min(seen_counts),
                "prompt_buffer/count_std": float(np.std(seen_counts)),
            })

            # Classify pass_rates into buckets (aligned with batch_metrics.py)
            # extremely_hard: p=0, hard: 0<p<=0.2, medium: 0.2<p<0.8, easy: 0.8<=p<1.0, extremely_easy: p=1.0
            buckets = {
                "extremely_hard": 0,
                "hard": 0,
                "medium": 0,
                "easy": 0,
                "extremely_easy": 0,
            }
            for pr in seen_pass_rates:
                if pr == 0.0:
                    buckets["extremely_hard"] += 1
                elif pr <= 0.2:
                    buckets["hard"] += 1
                elif pr < 0.8:
                    buckets["medium"] += 1
                elif pr < 1.0:
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
            # Deep copy meta_data including row_indices for debug inspection.
            # row_indices will NOT be restored on load – see _merge_meta_data_from_state.
            # NOTE: priority_strategy / alpha are intentionally NOT saved;
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
                pr = extra_info.get("pass_rate")
                if pr is not None and not (isinstance(pr, float) and np.isnan(pr)):
                    if prompt_uid not in self.meta_data:
                        self.meta_data[prompt_uid] = {}
                    self.meta_data[prompt_uid]["pass_rate"] = float(pr)
                    self.meta_data[prompt_uid]["count"] = 1
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

    def _sync_row_weights(self) -> None:
        """Rebuild ``self.row_weights`` from ``self.meta_data``."""
        self.row_weights[:] = self._default_priority_weight

        if not self.use_priority:
            return

        for prompt_uid, meta in self.meta_data.items():
            if "pass_rate" in meta:
                weight = self._priority_weight_fn(meta["pass_rate"])
                for row_idx in meta.get("row_indices", []):
                    self.row_weights[row_idx] = weight

    def _update_meta_data(self, prompt_uid: str, pass_rate: float, count: int) -> None:
        """Apply exponential moving average update for a single sample's pass_rate."""
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

        if self.use_priority:
            weight = self._priority_weight_fn(self.meta_data[prompt_uid]["pass_rate"])
            for row_idx in self.meta_data[prompt_uid].get("row_indices", []):
                self.row_weights[row_idx] = weight
