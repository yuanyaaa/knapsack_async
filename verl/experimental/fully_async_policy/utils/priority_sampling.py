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
    return PRIORITY_WEIGHT_REGISTRY[name]


# ---------------------------------------------------------------------------
# Built-in priority weight strategies
# ---------------------------------------------------------------------------


@register_priority_weight("medium")
def priority_weight_medium(pass_rate: float) -> float:
    """Peak weight at pass_rate=0.5, symmetric bell shape."""
    return pass_rate * (1 - pass_rate) * 4

@register_priority_weight("medium_sharp")
def priority_weight_medium_sharp(pass_rate: float) -> float:
    """Peak weight at pass_rate=0.5, symmetric bell shape."""
    return pass_rate ** 2 * (1 - pass_rate) ** 2 * 16

@register_priority_weight("hard")
def priority_weight_hard(pass_rate: float) -> float:
    """Skewed towards harder samples (lower pass_rate)."""
    return pass_rate * (1 - pass_rate) ** 3 * 256 / 27


@register_priority_weight("medium_with_exploration")
def priority_weight_medium_with_exploration(pass_rate: float) -> float:
    """Medium strategy with an exploration floor to avoid starving any sample."""
    epsilon = 0.05
    return epsilon + (1 - epsilon) * pass_rate * (1 - pass_rate) * 4
