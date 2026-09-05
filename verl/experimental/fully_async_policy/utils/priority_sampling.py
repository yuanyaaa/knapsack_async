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
Priority weight strategies for priority-based sampling.

Each strategy maps a pass_rate (float in [0, 1]) to a sampling weight.
Strategies are registered via the ``register_priority_weight`` decorator
and retrieved at runtime with ``get_priority_weight_fn``.

Usage:
    from verl.experimental.fully_async_policy.utils.priority_sampling import get_priority_weight_fn

    weight_fn = get_priority_weight_fn("medium_with_exploration")
    weight = weight_fn(pass_rate=0.5)
"""

__all__ = ["register_priority_weight", "get_priority_weight_fn"]

import math
import os
from typing import Callable

PriorityWeightFn = Callable[[float], float]

PRIORITY_WEIGHT_REGISTRY: dict[str, PriorityWeightFn] = {}


def register_priority_weight(name: str) -> Callable[[PriorityWeightFn], PriorityWeightFn]:
    """Decorator to register a priority weight function with a given name.

    Args:
        name: The name of the priority weight strategy.

    Returns:
        Decorator function that registers the priority weight function.
    """

    def decorator(fn: PriorityWeightFn) -> PriorityWeightFn:
        if name in PRIORITY_WEIGHT_REGISTRY and PRIORITY_WEIGHT_REGISTRY[name] != fn:
            raise ValueError(
                f"Priority weight strategy '{name}' has already been registered: "
                f"{PRIORITY_WEIGHT_REGISTRY[name]} vs {fn}"
            )
        PRIORITY_WEIGHT_REGISTRY[name] = fn
        return fn

    return decorator


def get_priority_weight_fn(name: str) -> PriorityWeightFn:
    """Get the priority weight function with a given name.

    Args:
        name: The name of the priority weight strategy.

    Returns:
        The priority weight function.

    Raises:
        ValueError: If the strategy name is not registered.
    """
    if name not in PRIORITY_WEIGHT_REGISTRY:
        raise ValueError(
            f"Invalid priority strategy: {name}. Supported strategies are: {list(PRIORITY_WEIGHT_REGISTRY.keys())}"
        )
    base_fn = PRIORITY_WEIGHT_REGISTRY[name]

    def _wrapped(pass_rate: float) -> float:
        # Treat exact endpoints as their clipped counterparts so p=0/p=1
        # prompts remain reachable under priority sampling.
        p = min(0.99, max(0.01, float(pass_rate)))
        return base_fn(p)

    return _wrapped


# ---------------------------------------------------------------------------
# Built-in priority weight strategies
# ---------------------------------------------------------------------------


@register_priority_weight("medium")
def priority_weight_medium(pass_rate: float) -> float:
    """Peak weight at pass_rate=0.5, symmetric bell shape."""
    return pass_rate * (1 - pass_rate) * 4


@register_priority_weight("lower_mid")
def priority_weight_lower_mid(pass_rate: float) -> float:
    """Lower-mid focus: p * (1-p)^2, peak at pass_rate=1/3."""
    return pass_rate * (1 - pass_rate) ** 2


@register_priority_weight("medium_sharp")
def priority_weight_medium_sharp(pass_rate: float) -> float:
    """Peak weight at pass_rate=0.5, symmetric bell shape."""
    return pass_rate ** 2 * (1 - pass_rate) ** 2 * 16

@register_priority_weight("hard")
def priority_weight_hard(pass_rate: float) -> float:
    """Skewed towards harder samples (lower pass_rate)."""
    return pass_rate * (1 - pass_rate) ** 3 * 256 / 27


@register_priority_weight("target_03")
def priority_weight_target_03(pass_rate: float) -> float:
    """Peak weight at pass_rate=0.3, smoothly decaying to both sides."""
    target = 0.3
    p = min(0.99, max(0.01, float(pass_rate)))
    return (p / target) ** target * ((1.0 - p) / (1.0 - target)) ** (1.0 - target)


@register_priority_weight("medium_with_exploration")
def priority_weight_medium_with_exploration(pass_rate: float) -> float:
    """Medium strategy with an exploration floor to avoid starving any sample."""
    epsilon = 0.05
    return epsilon + (1 - epsilon) * pass_rate * (1 - pass_rate) * 4


def _productive_curve_n(
    pass_rate: float,
    min_n: int = 2,
    max_n: int = 16,
    chunk_size: int = 2,
) -> int:
    """Mirror the productive_curve budget rule used by adaptive rollout."""
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

    chunk_size = max(1, int(chunk_size))
    target = int(math.floor(float(target) / float(chunk_size)) * chunk_size)
    return max(int(min_n), min(int(target), int(max_n)))


def _medium_focus_plateau_n(
    pass_rate: float,
    min_n: int = 2,
    max_n: int = 16,
    chunk_size: int = 2,
) -> int:
    """Mirror the medium_focus_plateau budget rule used by adaptive rollout."""
    p = min(1.0, max(0.0, float(pass_rate)))
    if 0.5 < p < 0.8:
        target = float(max_n)
    else:
        target = min_n + (4.0 * p * (1.0 - p)) * (max_n - min_n)

    chunk_size = max(1, int(chunk_size))
    target = int(math.floor(float(target) / float(chunk_size)) * chunk_size)
    return max(int(min_n), min(int(target), int(max_n)))


@register_priority_weight("productive_mixed")
def priority_weight_productive_mixed(pass_rate: float) -> float:
    """Prefer prompts likely to produce both positive and negative responses.

    A prompt receives weight 1.0 if the probability of seeing at least one
    positive and one negative response under the productive_curve-assigned N is
    at least ``PRODUCTIVE_MIXED_THRESHOLD`` (default 0.8). Otherwise it receives
    the exploration weight ``PRODUCTIVE_MIXED_FALLBACK_WEIGHT`` (default 1/3).
    """
    p = min(1.0, max(0.0, float(pass_rate)))
    min_n = int(os.environ.get("PRODUCTIVE_MIXED_MIN_N", "2"))
    max_n = int(os.environ.get("PRODUCTIVE_MIXED_MAX_N", "16"))
    chunk_size = int(os.environ.get("PRODUCTIVE_MIXED_CHUNK_SIZE", "2"))
    threshold = float(os.environ.get("PRODUCTIVE_MIXED_THRESHOLD", "0.8"))
    fallback = float(os.environ.get("PRODUCTIVE_MIXED_FALLBACK_WEIGHT", str(1.0 / 3.0)))

    n = _productive_curve_n(p, min_n=min_n, max_n=max_n, chunk_size=chunk_size)
    mixed_prob = 1.0 - (p ** n) - ((1.0 - p) ** n)
    return 1.0 if mixed_prob >= threshold else fallback


@register_priority_weight("productive_mixed_plateau")
def priority_weight_productive_mixed_plateau(pass_rate: float) -> float:
    """Two-weight sampling using the medium_focus_plateau-assigned N.

    Weight is 1.0 if the prompt is likely to produce both positive and negative
    responses under its assigned N; otherwise the fallback weight is used.
    """
    p = min(1.0, max(0.0, float(pass_rate)))
    min_n = int(os.environ.get("PRODUCTIVE_MIXED_MIN_N", "2"))
    max_n = int(os.environ.get("PRODUCTIVE_MIXED_MAX_N", "16"))
    chunk_size = int(os.environ.get("PRODUCTIVE_MIXED_CHUNK_SIZE", "2"))
    threshold = float(os.environ.get("PRODUCTIVE_MIXED_THRESHOLD", "0.8"))
    fallback = float(os.environ.get("PRODUCTIVE_MIXED_FALLBACK_WEIGHT", str(1.0 / 3.0)))

    n = _medium_focus_plateau_n(p, min_n=min_n, max_n=max_n, chunk_size=chunk_size)
    mixed_prob = 1.0 - (p ** n) - ((1.0 - p) ** n)
    return 1.0 if mixed_prob >= threshold else fallback
