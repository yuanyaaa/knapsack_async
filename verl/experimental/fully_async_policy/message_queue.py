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
import logging
from collections import deque
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)
        self.queue = deque(maxlen=self.max_queue_size)
        # Sidecar deque holding the per-item sample count so we can report
        # an EXACT pending-samples total (used by rollouter reset_staleness).
        # Kept strictly in lock-step with ``self.queue``: every append/popleft
        # on ``queue`` has a matching append/popleft here.
        self._sample_counts = deque(maxlen=self.max_queue_size)
        self._start_versions = deque(maxlen=self.max_queue_size)
        self._pending_samples = 0

        self.val_queue = deque()

        # Asyncio for message handling
        self.running = True

        # async safe
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        print(f"[MessageQueue] initialized with max_queue_size={max_queue_size}")

    async def put_sample(self, sample: Any, n_samples: int = 1, start_version: int = -1) -> bool:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data
            n_samples: Number of underlying samples carried by this item.
                Used to maintain an exact ``pending_samples`` counter that
                the rollouter consumes via ``get_pending_sample_count()``
                (and thus via ``reset_staleness``). For the ``None``
                end-of-stream sentinel or ``val`` samples, pass 1 (default).
            start_version: The minimum param version at which the sequences
                in this sample started rolling out. Used by
                ``purge_stale_samples`` to evict stale items without
                deserializing. Pass -1 (default) for sentinels.

        Returns:
            bool: Whether the sample was successfully put into the queue
        """
        async with self._lock:
            # If queue is full, remove the oldest sample (rarely happens)
            is_drop = False
            if len(self.queue) >= self.max_queue_size:
                self.queue.popleft()
                dropped_n = self._sample_counts.popleft() if self._sample_counts else 0
                self._start_versions.popleft() if self._start_versions else None
                self._pending_samples = max(0, self._pending_samples - int(dropped_n))
                self.dropped_samples += 1
                is_drop = True
                logger.warning("Queue full, dropped sample")
            self.queue.append(sample)
            _n = max(0, int(n_samples))
            self._sample_counts.append(_n)
            self._start_versions.append(int(start_version))
            self._pending_samples += _n
            self.total_produced += 1

            # Notify waiting consumers
            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(
                    f"[MessageQueue] stats: "
                    f"produced={self.total_produced} (prompts), "
                    f"consumed={self.total_consumed} (prompts), "
                    f"dropped={self.dropped_samples} (prompts) | "
                    f"queue_size={len(self.queue)}/{self.max_queue_size} (prompts), "
                f"pending_samples={self._pending_samples}"
                )
            if is_drop:
                return False
            return True

    async def get_batch(
        self,
        target_samples: int | None = None,
        target_prompts: int | None = None,
    ) -> dict | None:
        """Wait until enough samples are available, then take a subset < target.

        Two-phase logic:
          Phase 1 (wait): block until pending_samples >= target_samples
                          (or queue_size >= target_prompts).
          Phase 2 (take): pop items from the queue such that the total taken
                          is as close to (but strictly less than) the target as
                          possible. Remaining items stay for the next step.

        Exactly one of ``target_samples`` or ``target_prompts`` must be set.

        Returns:
            dict with keys:
              - "items": list of (data, n_samples) tuples
              - "total_samples": total n_samples across all taken items
              - "total_prompts": number of items taken
              - "remaining_queue_size": queue length after taking
              - "remaining_pending_samples": pending samples after taking
              - "terminated": bool, True if queue was shut down during wait
            Or ``None`` if queue is shut down and empty (no items at all).
        """
        if (target_samples is None) == (target_prompts is None):
            raise ValueError("Exactly one of target_samples or target_prompts must be set")

        use_sample_mode = target_samples is not None
        target = target_samples if use_sample_mode else target_prompts

        async with self._lock:
            # Phase 1: Wait until enough samples/prompts are available
            terminated = False
            while True:
                current = self._pending_samples if use_sample_mode else len(self.queue)
                if current >= target:
                    break
                if not self.running:
                    # Queue shut down but may still have partial items
                    terminated = True
                    break
                await self._consumer_condition.wait()

            # If queue is empty after shutdown, return None
            if len(self.queue) == 0:
                return None

            # Phase 2: Take items such that total <= target
            # Strategy: greedily pop items while adding the next item would
            # NOT cause total to strictly exceed target. This gives us the
            # largest subset not exceeding target (target itself is allowed).
            items = []
            total_taken_samples = 0
            total_taken_prompts = 0

            while len(self.queue) > 0:
                # Peek at the next item's sample count
                next_n = int(self._sample_counts[0]) if self._sample_counts else 1

                if use_sample_mode:
                    # Would adding this item strictly exceed target?
                    if total_taken_samples + next_n > target:
                        break
                else:
                    # In prompt mode, each item is 1 prompt
                    if total_taken_prompts + 1 > target:
                        break

                # Pop the item
                data = self.queue.popleft()
                popped_n = int(self._sample_counts.popleft()) if self._sample_counts else 0
                self._start_versions.popleft() if self._start_versions else None
                self._pending_samples = max(0, self._pending_samples - popped_n)
                self.total_consumed += 1

                items.append((data, popped_n))
                total_taken_samples += popped_n
                total_taken_prompts += 1

            # Edge case: if we couldn't take anything (first item alone >= target),
            # take exactly one item to avoid deadlock.
            if not items and len(self.queue) > 0:
                data = self.queue.popleft()
                popped_n = int(self._sample_counts.popleft()) if self._sample_counts else 0
                self._start_versions.popleft() if self._start_versions else None
                self._pending_samples = max(0, self._pending_samples - popped_n)
                self.total_consumed += 1
                items.append((data, popped_n))
                total_taken_samples += popped_n
                total_taken_prompts += 1

            return {
                "items": items,
                "total_samples": total_taken_samples,
                "total_prompts": total_taken_prompts,
                "remaining_queue_size": len(self.queue),
                "remaining_pending_samples": int(self._pending_samples),
                "terminated": terminated,
            }

    async def get_queue_size(self) -> int:
        """Get current queue length (in RolloutSample units, NOT sequences)."""
        async with self._lock:
            return len(self.queue)

    async def get_pending_sample_count(self) -> int:
        """Get the exact total number of samples sitting in the queue.

        This is the sum of ``n_samples`` values supplied to ``put_sample``
        for all items currently queued. Used by the rollouter's
        ``reset_staleness`` to avoid the ``queue_size * rollout.n`` estimate,
        which is wrong for variable-N stop rules (e.g. has_at_least_positive
        or adaptive_rollout_n_strategy).
        """
        async with self._lock:
            return int(self._pending_samples)

    async def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                "queue_size": len(self.queue),
                "pending_samples": self._pending_samples,
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "max_queue_size": self.max_queue_size,
            }

    async def purge_stale_samples(
        self,
        current_param_version: int,
        staleness_threshold: float,
    ) -> dict[str, int]:
        """Remove items whose staleness >= staleness_threshold.

        Staleness of an item = ``current_param_version - start_version``.
        Items with ``start_version == -1`` (sentinels) are never purged.

        To guarantee that the NEXT training step sees only samples with
        staleness ≤ S, the caller should invoke this with the freshly
        updated ``current_param_version`` right after ``reset_staleness``.
        Because the next training step will consume samples at this same
        ``current_param_version`` (it only increments AFTER training),
        any item with ``current_param_version - start_version < S`` will
        have staleness < S at consumption time, i.e. ≤ S - 1 < S. ✓

        Returns:
            dict with ``purged_prompts`` and ``purged_samples`` counts.
        """
        async with self._lock:
            if not self.queue:
                return {"purged_prompts": 0, "purged_samples": 0}

            keep_queue = deque(maxlen=self.max_queue_size)
            keep_counts = deque(maxlen=self.max_queue_size)
            keep_versions = deque(maxlen=self.max_queue_size)
            purged_prompts = 0
            purged_samples = 0

            while self.queue:
                item = self.queue.popleft()
                n = int(self._sample_counts.popleft()) if self._sample_counts else 0
                sv = int(self._start_versions.popleft()) if self._start_versions else -1

                # Never purge sentinels (start_version == -1)
                if sv >= 0 and (current_param_version - sv) >= staleness_threshold:
                    purged_prompts += 1
                    purged_samples += n
                    self._pending_samples = max(0, self._pending_samples - n)
                else:
                    keep_queue.append(item)
                    keep_counts.append(n)
                    keep_versions.append(sv)

            self.queue = keep_queue
            self._sample_counts = keep_counts
            self._start_versions = keep_versions

            if purged_prompts > 0:
                print(
                    f"[MessageQueue] purge_stale_samples: removed {purged_prompts} prompts "
                    f"({purged_samples} samples) with staleness >= {staleness_threshold} "
                    f"(current_param_version={current_param_version}). "
                    f"Remaining: {len(self.queue)} prompts, {self._pending_samples} samples."
                )

            return {"purged_prompts": purged_prompts, "purged_samples": purged_samples}

    async def clear_queue(self):
        """Clear the queue"""
        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            self._sample_counts.clear()
            self._start_versions.clear()
            self._pending_samples = 0
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def shutdown(self):
        """Shutdown the message queue"""
        async with self._lock:
            self.running = False
            # Notify all waiting coroutines so they can exit
            self._consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        async with self._lock:
            # Estimate memory usage of samples in queue
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # Estimate size of a single sample (simplified estimation)
                sample = list(self.queue)[0]
                try:
                    sample_size = sys.getsizeof(sample)
                    # Since we now store RolloutSample directly, estimate based on its components
                    if hasattr(sample, "original_batch_dict") and sample.original_batch_dict:
                        # Estimate batch data size
                        batch_data = sample.original_batch_dict.get("batch", {})
                        sample_size += len(batch_data) * 1000  # Roughly estimate 1KB per batch entry
                    if hasattr(sample, "agent_loop_output"):
                        # Estimate AgentLoopOutput size
                        sample_size += 5000  # Roughly estimate 5KB for AgentLoopOutput
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 15000  # Roughly estimate 15KB per RolloutSample

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }

    async def put_validate(self, data):
        async with self._lock:
            self.val_queue.append(data)

    async def get_validate(self):
        async with self._lock:
            if self.val_queue:
                return self.val_queue.popleft()
            else:
                return None


class MessageQueueClient:
    """Asyncio-compatible MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any, n_samples: int = 1, start_version: int = -1) -> bool:
        """Put batch into queue (async).

        ``n_samples`` is forwarded to the actor so it can maintain an exact
        pending-samples counter (see ``MessageQueue.get_pending_sample_count``).
        Defaults to 1 for backward compatibility (e.g. validation samples and
        the end-of-stream ``None`` sentinel).

        ``start_version`` is the minimum param version at which the sequences
        started rolling out. Used by ``purge_stale_samples`` to evict stale
        items without deserializing. Pass -1 (default) for sentinels.
        """
        future = self.queue_actor.put_sample.remote(sample, n_samples, start_version)
        return await asyncio.wrap_future(future.future())

    async def put_validate(self, data: Any) -> bool:
        future = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(future.future())

    def get_validate_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_validate.remote())

    async def get_batch(
        self,
        target_samples: int | None = None,
        target_prompts: int | None = None,
    ) -> dict | None:
        """Wait until enough samples are available, then take a subset < target.

        See ``MessageQueue.get_batch`` for full semantics.
        """
        future = self.queue_actor.get_batch.remote(
            target_samples=target_samples, target_prompts=target_prompts
        )
        return await asyncio.wrap_future(future.future())

    async def get_queue_size(self) -> int:
        """Get queue size (async, in RolloutSample units)."""
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_pending_sample_count(self) -> int:
        """Get the exact number of pending samples in the queue (async)."""
        future = self.queue_actor.get_pending_sample_count.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics (async)"""
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        """Clear queue (async)"""
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def shutdown(self):
        """Shutdown queue (async)"""
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    async def purge_stale_samples(
        self,
        current_param_version: int,
        staleness_threshold: float,
    ) -> dict[str, int]:
        """Remove items whose staleness >= staleness_threshold (async).

        See ``MessageQueue.purge_stale_samples`` for full semantics.
        """
        future = self.queue_actor.purge_stale_samples.remote(
            current_param_version, staleness_threshold
        )
        return await asyncio.wrap_future(future.future())

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics (async)"""
        future = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(future.future())

    def get_statistics_sync(self) -> dict[str, Any]:
        """Get statistics (sync - deprecated, use get_statistics instead)"""
        return ray.get(self.queue_actor.get_statistics.remote())
