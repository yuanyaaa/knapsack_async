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

"""Structured debug tracing for adaptive rollout N and priority sampling.

Every record is a single line so a whole run can be filtered out of the Ray job
log with::

    ray job logs <job-id> | rg '\\[adaptive_n\\]'
    ray job logs <job-id> | rg '\\[priority\\]'

Records are emitted as ``[<channel>] event=<name> k=v k=v ...`` with stable key
names, which keeps them greppable and parseable::

    [adaptive_n] event=initial_n uid=abc p=0.312 peak_n=8 min_n=4 max_n=16 n=7

Tracing is off by default. It is enabled per channel through the rollout config
(``debug_adaptive_n`` / ``debug_priority_sampling``) and rate-limited so a
misconfigured run cannot flood the driver log: each event name gets its own
budget of ``max_records_per_event`` lines, after which only every
``sample_every`` -th record is kept. Counters keep incrementing regardless, and
the suppressed total is reported by :func:`summary`.
"""

__all__ = [
    "CHANNEL_ADAPTIVE_N",
    "CHANNEL_PRIORITY",
    "configure",
    "is_enabled",
    "trace",
    "summary",
]

import threading
from typing import Any


CHANNEL_ADAPTIVE_N = "adaptive_n"
CHANNEL_PRIORITY = "priority"

_lock = threading.Lock()
_enabled: dict[str, bool] = {CHANNEL_ADAPTIVE_N: False, CHANNEL_PRIORITY: False}
_max_records_per_event: int = 2000
_sample_every: int = 200
_emitted: dict[str, int] = {}
_seen: dict[str, int] = {}
_suppressed: dict[str, int] = {}


def configure(
    debug_adaptive_n: bool = False,
    debug_priority_sampling: bool = False,
    max_records_per_event: int = 2000,
    sample_every: int = 200,
) -> None:
    """Enable/disable tracing channels and reset the rate-limiter budgets."""
    global _max_records_per_event, _sample_every
    with _lock:
        _enabled[CHANNEL_ADAPTIVE_N] = bool(debug_adaptive_n)
        _enabled[CHANNEL_PRIORITY] = bool(debug_priority_sampling)
        _max_records_per_event = max(0, int(max_records_per_event))
        _sample_every = max(1, int(sample_every))
        _emitted.clear()
        _seen.clear()
        _suppressed.clear()
    if debug_adaptive_n or debug_priority_sampling:
        print(
            f"[adaptive_debug] tracing enabled: adaptive_n={debug_adaptive_n}, "
            f"priority={debug_priority_sampling}, "
            f"max_records_per_event={max_records_per_event}, sample_every={sample_every}",
            flush=True,
        )


def is_enabled(channel: str) -> bool:
    return _enabled.get(channel, False)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    text = str(value)
    return text.replace(" ", "_") if " " in text else text


def _should_emit(channel: str, event: str) -> bool:
    key = f"{channel}/{event}"
    with _lock:
        seen = _seen.get(key, 0) + 1
        _seen[key] = seen
        if seen <= _max_records_per_event:
            _emitted[key] = _emitted.get(key, 0) + 1
            return True
        if seen % _sample_every == 0:
            _emitted[key] = _emitted.get(key, 0) + 1
            return True
        _suppressed[key] = _suppressed.get(key, 0) + 1
        return False


def trace(channel: str, event: str, **fields: Any) -> None:
    """Emit one structured debug record if ``channel`` is enabled.

    Records go to stdout rather than ``logging`` because the rollouter runs as a
    Ray actor whose root logger level is not guaranteed to pass INFO, while
    stdout is always captured into the Ray job log.
    """
    if not _enabled.get(channel, False):
        return
    if not _should_emit(channel, event):
        return
    payload = " ".join(f"{k}={_format_value(v)}" for k, v in fields.items())
    print(f"[{channel}] event={event} {payload}", flush=True)


def summary() -> dict[str, int]:
    """Per-event emitted / suppressed counters, for end-of-run reporting."""
    with _lock:
        out: dict[str, int] = {}
        for key, seen in _seen.items():
            out[f"{key}/seen"] = int(seen)
            out[f"{key}/emitted"] = int(_emitted.get(key, 0))
            out[f"{key}/suppressed"] = int(_suppressed.get(key, 0))
        return out
