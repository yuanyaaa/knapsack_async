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
"""VIP baseline: Variance-Informed Predictive rollout allocation.

Nguyen et al., "Adaptive Rollout Allocation for Online Reinforcement Learning
with Verifiable Rewards" (arXiv:2602.01601).

* Success probabilities come from a GP over prompt embeddings with a sigmoid
  link and the recursive posterior-mean update of Algorithm 1.
* A block of prompts shares a hard budget ``C = budget_per_prompt * B``; the
  per-prompt counts minimize the summed gradient variance (Theorems 5.1/5.2)
  and are rounded with the greedy incentive heuristic of Algorithm 2.
"""

from __future__ import annotations

import heapq
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

VIP_VARIANCE_DR_GRPO = "dr_grpo"
VIP_VARIANCE_RLOO = "rloo"


def _variance_factor(n: np.ndarray, variance: str) -> np.ndarray:
    """Per-prompt objective divided by ``a_q``: (n-1)/n^2 or 1/(n-1)."""
    if variance == VIP_VARIANCE_DR_GRPO:
        return (n - 1.0) / (n * n)
    return 1.0 / (n - 1.0)


def _marginal_decrease(n: np.ndarray, variance: str) -> np.ndarray:
    """``-d/dn`` of the variance factor: (n-2)/n^3 or 1/(n-1)^2."""
    if variance == VIP_VARIANCE_DR_GRPO:
        return (n - 2.0) / (n * n * n)
    return 1.0 / ((n - 1.0) ** 2)


def continuous_allocation(
    a: np.ndarray, budget: float, lower: int, upper: int, variance: str = VIP_VARIANCE_DR_GRPO
) -> np.ndarray:
    """Exact minimizer of ``sum_q a_q * factor(n_q)`` s.t. ``sum n = C, L <= n <= U``.

    For a fixed multiplier lambda the KKT condition ``a_q * h(n_q) = lambda``
    (h = marginal decrease, strictly decreasing on [L, U] when L >= 3) gives
    ``n_q(lambda)``; lambda is then found by bisection on ``sum n_q = C``.
    """
    a = np.maximum(np.asarray(a, dtype=np.float64), 0.0)
    B = a.shape[0]
    if B == 0:
        return np.zeros(0)
    if budget <= B * lower:
        return np.full(B, float(lower))
    if budget >= B * upper:
        return np.full(B, float(upper))
    if not np.any(a > 0):
        return np.full(B, float(budget) / B)

    h_lo = float(_marginal_decrease(np.array([float(upper)]), variance)[0])
    h_hi = float(_marginal_decrease(np.array([float(lower)]), variance)[0])
    pos = a > 0

    def n_of_lambda(lam: float) -> np.ndarray:
        n = np.full(B, float(lower))
        ratio = np.zeros(B)
        ratio[pos] = lam / a[pos]
        at_upper = pos & (ratio <= h_lo)
        interior = pos & (ratio > h_lo) & (ratio < h_hi)
        n[at_upper] = float(upper)
        if np.any(interior):
            if variance == VIP_VARIANCE_RLOO:
                n[interior] = 1.0 + np.sqrt(1.0 / ratio[interior])
            else:
                lo = np.full(int(interior.sum()), float(lower))
                hi = np.full(int(interior.sum()), float(upper))
                target = ratio[interior]
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    too_small = _marginal_decrease(mid, variance) > target
                    lo = np.where(too_small, mid, lo)
                    hi = np.where(too_small, hi, mid)
                n[interior] = 0.5 * (lo + hi)
        return np.clip(n, lower, upper)

    log_lo = math.log(float(a[pos].min()) * h_lo) - 2.0
    log_hi = math.log(float(a[pos].max()) * h_hi) + 2.0
    for _ in range(200):
        mid = 0.5 * (log_lo + log_hi)
        if n_of_lambda(math.exp(mid)).sum() > budget:
            log_lo = mid
        else:
            log_hi = mid
    return n_of_lambda(math.exp(0.5 * (log_lo + log_hi)))


def round_allocation(
    n_cont: np.ndarray, a: np.ndarray, budget: int, lower: int, upper: int, variance: str = VIP_VARIANCE_DR_GRPO
) -> np.ndarray:
    """Algorithm 2: floor, then hand out the remainder by largest objective decrease."""
    a = np.maximum(np.asarray(a, dtype=np.float64), 0.0)
    n = np.clip(np.floor(np.asarray(n_cont) + 1e-9).astype(np.int64), lower, upper)
    remaining = int(budget) - int(n.sum())
    heap = []
    for q in range(n.shape[0]):
        if n[q] < upper:
            gain = a[q] * (_variance_factor(np.array([n[q]], dtype=np.float64), variance)[0]
                           - _variance_factor(np.array([n[q] + 1], dtype=np.float64), variance)[0])
            heapq.heappush(heap, (-gain, q))
    while remaining > 0 and heap:
        _, q = heapq.heappop(heap)
        n[q] += 1
        remaining -= 1
        if n[q] < upper:
            gain = a[q] * (_variance_factor(np.array([n[q]], dtype=np.float64), variance)[0]
                           - _variance_factor(np.array([n[q] + 1], dtype=np.float64), variance)[0])
            heapq.heappush(heap, (-gain, q))
    while remaining < 0:
        reducible = np.where(n > lower)[0]
        if reducible.size == 0:
            break
        q = reducible[np.argmin(a[reducible])]
        n[q] -= 1
        remaining += 1
    return n


class VIPRolloutAllocator:
    """GP success-rate predictor + per-block variance-minimizing allocation."""

    def __init__(
        self,
        embedding_path: str,
        prior_pass_rates: Mapping[str, float],
        budget_per_prompt: int,
        min_n: int,
        max_n: int,
        length_scale: float = 0.0,
        eps: float = 1e-6,
        jitter: float = 1e-4,
        variance: str = VIP_VARIANCE_DR_GRPO,
    ):
        if variance not in (VIP_VARIANCE_DR_GRPO, VIP_VARIANCE_RLOO):
            raise ValueError(f"vip_variance must be '{VIP_VARIANCE_DR_GRPO}' or '{VIP_VARIANCE_RLOO}', got {variance!r}")
        if min_n < 3:
            raise ValueError(f"VIP requires min_rollout_n >= 3 (paper Sec. 5.2), got {min_n}")
        if not (min_n <= budget_per_prompt <= max_n):
            raise ValueError(f"VIP needs min_n <= budget_per_prompt <= max_n, got {min_n}, {budget_per_prompt}, {max_n}")

        data = np.load(embedding_path, allow_pickle=False)
        self.uids = [str(u) for u in data["uids"].tolist()]
        self.uid_to_idx = {u: i for i, u in enumerate(self.uids)}
        self.X = np.ascontiguousarray(data["embeddings"], dtype=np.float32)
        self.sq_norm = np.einsum("ij,ij->i", self.X, self.X)
        median = float(data["median_dist"]) if "median_dist" in data else 1.0
        self.length_scale = float(length_scale) if length_scale and length_scale > 0 else median

        self.budget_per_prompt = int(budget_per_prompt)
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        self.eps = float(eps)
        self.jitter = float(jitter)
        self.variance = variance

        self.mean = np.zeros(len(self.uids), dtype=np.float64)
        num_prior = 0
        for uid, p in prior_pass_rates.items():
            idx = self.uid_to_idx.get(str(uid))
            if idx is None or p is None or not math.isfinite(float(p)):
                continue
            self.mean[idx] = self._logit(float(p))
            num_prior += 1

        self._pending: dict[str, float] = {}
        self._targets: dict[str, int] = {}
        self._last_stats: dict[str, float] = {}
        self.num_blocks = 0
        self.num_fits = 0
        self.num_missing_uid = 0
        print(
            f"[VIP] loaded {len(self.uids)} embeddings dim={self.X.shape[1]} from {embedding_path}; "
            f"length_scale={self.length_scale:.4f} (median={median:.4f}), prior from offline pass_rate "
            f"for {num_prior} prompts, budget_per_prompt={self.budget_per_prompt}, "
            f"L={self.min_n}, U={self.max_n}, eps={self.eps:g}, jitter={self.jitter:g}, variance={self.variance}"
        )

    def _logit(self, p: float | np.ndarray):
        p = np.clip(p, self.eps, 1.0 - self.eps)
        return np.log(p / (1.0 - p))

    def _kernel_to(self, idx: np.ndarray) -> np.ndarray:
        """RBF kernel between every prompt and the prompts in ``idx``: (Q, |idx|)."""
        cross = self.X @ self.X[idx].T
        sq = np.maximum(self.sq_norm[:, None] + self.sq_norm[idx][None, :] - 2.0 * cross, 0.0)
        return np.exp(-sq.astype(np.float64) / (2.0 * self.length_scale ** 2))

    def observe(self, uid: str, rewards: Iterable[float]) -> None:
        rewards = list(rewards)
        if not rewards:
            return
        self._pending[str(uid)] = sum(1.0 for r in rewards if r > 0) / len(rewards)

    def _fit_pending(self) -> dict[str, float]:
        if not self._pending:
            return {}
        uids = [u for u in self._pending if u in self.uid_to_idx]
        obs = np.array([self._pending[u] for u in uids], dtype=np.float64)
        self._pending.clear()
        if not uids:
            return {}
        idx = np.array([self.uid_to_idx[u] for u in uids], dtype=np.int64)
        prior_p = 1.0 / (1.0 + np.exp(-self.mean[idx]))
        g_obs = self._logit(obs)
        K_all = self._kernel_to(idx)
        K_bb = K_all[idx] + self.jitter * np.eye(len(idx))
        alpha = np.linalg.solve(K_bb, g_obs - self.mean[idx])
        self.mean += K_all @ alpha
        self.mean[idx] = g_obs
        self.num_fits += 1
        return {
            "vip/gp_fit_size": float(len(idx)),
            "vip/gp_prior_mae": float(np.mean(np.abs(prior_p - obs))),
            "vip/gp_observed_p_mean": float(obs.mean()),
        }

    def plan(self, uids: Sequence[str]) -> dict[str, int]:
        """Refit on finished prompts, then allocate ``budget_per_prompt * len(uids)``."""
        fit_stats = self._fit_pending()
        uids = [str(u) for u in uids]
        known = [u for u in uids if u in self.uid_to_idx]
        self.num_missing_uid += len(uids) - len(known)

        targets = {u: self.budget_per_prompt for u in uids}
        if known:
            idx = np.array([self.uid_to_idx[u] for u in known], dtype=np.int64)
            p_hat = 1.0 / (1.0 + np.exp(-self.mean[idx]))
            a = p_hat * (1.0 - p_hat)
            budget = self.budget_per_prompt * len(known)
            n_cont = continuous_allocation(a, budget, self.min_n, self.max_n, self.variance)
            n_int = round_allocation(n_cont, a, budget, self.min_n, self.max_n, self.variance)
            targets.update({u: int(n) for u, n in zip(known, n_int.tolist())})
            self._last_stats = {
                **fit_stats,
                "vip/block_size": float(len(uids)),
                "vip/block_budget": float(budget),
                "vip/block_allocated": float(n_int.sum()),
                "vip/n_mean": float(n_int.mean()),
                "vip/n_max": float(n_int.max()),
                "vip/frac_at_lower": float(np.mean(n_int == self.min_n)),
                "vip/frac_at_upper": float(np.mean(n_int == self.max_n)),
                "vip/pred_p_mean": float(p_hat.mean()),
                "vip/pred_p_mid_frac": float(np.mean((p_hat > 0.1) & (p_hat < 0.9))),
            }
        self._targets.update(targets)
        self.num_blocks += 1
        return targets

    def target_n(self, uid: str) -> Optional[int]:
        return self._targets.pop(str(uid), None)

    def stats(self) -> dict[str, float]:
        return {
            **self._last_stats,
            "vip/num_blocks": float(self.num_blocks),
            "vip/num_fits": float(self.num_fits),
            "vip/num_missing_uid": float(self.num_missing_uid),
        }
