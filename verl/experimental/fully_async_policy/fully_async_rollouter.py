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
import multiprocessing
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat

import numpy as np
import ray
import torch

from verl.experimental.fully_async_policy.utils.async_prompt_buffer import AsyncPromptBuffer, get_adaptive_rollout_n
from verl.experimental.fully_async_policy.utils.async_response_buffer import (
    TRAIN_TRIGGER_FIXED_PROMPT,
    TRAIN_TRIGGER_FIXED_SAMPLES,
    AsyncResponseBuffer,
)
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    ValidateMetrics,
    prepare_single_generation_data,
    safe_create_task,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger


@ray.remote(num_cpus=10, max_concurrency=100)
class FullyAsyncRollouter(SeparateRayPPOTrainer):
    """
    Asynchronous sample generator, responsible for continuously generating training samples
    and putting them into MessageQueue
    Based on the mature implementation improvements of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger or equal than 1"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = False

        self.use_rm = False

        self.use_critic = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # ==================== fully async config ====================

        print("[FullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        if self.config.async_training.use_trainer_do_validate:
            rollout_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
            train_gpus = config.trainer.nnodes * config.trainer.n_gpus_per_node
            total_gpus = rollout_gpus + train_gpus
            print(f"[FullyAsyncRollouter] split before val_dataset total len: {len(val_dataset)}")
            split_dataset = val_dataset.split(total_gpus)
            rollout_val_dataset0 = split_dataset[:rollout_gpus]
            from torch.utils.data import ConcatDataset

            val_dataset = ConcatDataset(rollout_val_dataset0)
            print(f"[FullyAsyncRollouter] split after val_dataset total len: {len(val_dataset)}")
        print(f"[FullyAsyncRollouter] Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== AsyncPromptBuffer ====================
        # Always create AsyncPromptBuffer. When priority_strategy=None it
        # degrades to uniform random sampling, equivalent to the original
        # dataloader behavior.
        #
        # NOTE: "rollout n per prompt" is decided by the rollouter side
        # (AsyncResponseBuffer + optional dynamic_fixed_n_provider reading
        # pass_rate from prompt_buffer.meta_data). Those knobs live in
        # ``async_training.rollout_config.*`` and are NOT forwarded here.
        async_training_cfg = config.async_training
        priority_strategy = async_training_cfg.get("priority_strategy", None)
        self.prompt_buffer = AsyncPromptBuffer(
            dataset=train_dataset,
            collate_fn=collate_fn,
            seed=async_training_cfg.get("prompt_buffer_seed", 42),
            priority_strategy=priority_strategy,
            reward_ema=async_training_cfg.get("reward_ema", 0.0),
            pass_rate_clip_min=async_training_cfg.get("pass_rate_clip_min", 0.01),
            pass_rate_clip_max=async_training_cfg.get("pass_rate_clip_max", 0.99),
        )
        print("[FullyAsyncRollouter] AsyncPromptBuffer created successfully")

        # `total_rollout_steps` is SEQUENCE-level: the total number of sequences
        # to generate across the whole run (equivalent to
        # ``required_samples * total_train_steps``). The default budget is
        # derived from the dataset size, NOT from ``len(self.train_dataloader)``:
        # in fully-async mode ``gen_batch_size`` is typically 1 (prompts are
        # drawn one-by-one from AsyncPromptBuffer), and even when it is larger
        # the dataloader length would be "# of gen-batches", which does not
        # align with the sequence-level unit. Using ``len(train_dataset)``
        # keeps the default stable across ``gen_batch_size`` values and
        # consistent with user overrides.
        #
        # Users override via ``rollout.total_rollout_steps`` in the same unit,
        # i.e. ``train_batch_size * rollout.n * max_training_steps``.
        _rollout_n_for_budget = max(1, int(config.actor_rollout_ref.rollout.n))
        _dataset_epoch_sequences = max(1, len(train_dataset) * _rollout_n_for_budget)
        _default_total_rollout_steps = _dataset_epoch_sequences * self.config.trainer.total_epochs
        self.total_rollout_steps = _default_total_rollout_steps
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        # Report how many epochs the effective budget covers. When the user
        # overrides ``rollout.total_rollout_steps`` to a value smaller than the
        # dataset default, this will be fewer than ``total_epochs``.
        _effective_epochs = self.total_rollout_steps / _dataset_epoch_sequences
        print(
            f"[FullyAsyncRollouter] Total rollout steps (sequences): {self.total_rollout_steps} "
            f"(~{_effective_epochs:.3f} epochs over {len(train_dataset)} prompts x rollout.n={_rollout_n_for_budget}; "
            f"configured total_epochs={self.config.trainer.total_epochs})"
        )
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Config
        _raw_staleness = config.async_training.get("staleness_threshold", 1)
        assert int(_raw_staleness) == float(_raw_staleness), (
            f"staleness_threshold must be an integer value, got {_raw_staleness}"
        )
        self.staleness_threshold: int = int(_raw_staleness)
        # required_samples is SEQUENCE-LEVEL: the total number of generated sequences
        # required for one training step. Equals ppo_mini_batch_size (prompt units,
        # as consumed by verl trainer) * rollout.n * require_batches.
        # Example: ppo_mini_batch_size=32, rollout.n=8, require_batches=1 -> 256.
        self.require_batches = config.async_training.require_batches
        self._rollout_n = max(1, int(config.actor_rollout_ref.rollout.n))
        self.required_samples = (
            config.actor_rollout_ref.actor.ppo_mini_batch_size
            * self._rollout_n
            * self.require_batches
        )
        # Prompt-level equivalent (still used by prompt-unit derivations such
        # as the default ``working_pool_size`` and informational logging).
        # Keep it derived so changing ``rollout.n`` can't drift the two apart.
        self.required_prompts = max(1, self.required_samples // self._rollout_n)
        self.max_concurrent_samples = None
        self.max_queue_size = None

        # Statistics — dual-level (prompt + sample) counters
        # Prompt-level
        self.total_generated_prompts = 0
        self.dropped_prompts = 0
        self.rejected_prompts = 0
        # Sample-level (one prompt may produce N samples via rollout_n)
        self.total_generated_samples = 0
        self.dropped_samples = 0
        self.rejected_samples = 0
        # Staleness counter: counts samples generated under an older
        # parameter version (start_param_version < current_param_version).
        self.stale_samples = 0
        self.processed_sample_count = 0
        # Rollouter-side mirror of the trainer's current policy parameter
        # version. Updated by ``reset_staleness(current_param_version=...)``
        # after every parameter sync. Used to:
        #   (a) stamp ``PromptState.start_param_version`` at register time, so
        #       we can compute per-prompt staleness later;
        #   (b) drive the prompt-level staleness pause in
        #       ``_should_pause_generation``.
        # Starts at 0 (no param sync has happened yet at startup).
        self.current_param_version: int = 0
        # `global_steps` is SEQUENCE-level: it counts the cumulative number of
        # generated sequences so far and is compared against the sequence-level
        # `total_rollout_steps`. It is incremented in `_process_one_response`
        # whenever a sequence job finishes (fresh or continue-rollout), NOT in
        # `_feed_responses` (which only draws fresh prompts).
        # We start from step 1.
        self.global_steps = 1
        self.idle_start_time = -1  # Sentinel: -1 means no idle has occurred yet
        self.step_start_time = time.time()

        # Concurrency control
        # Modified by self.pause() or self._should_pause_generation()
        self.paused = False
        self.running = True

        # Add dataloader lock
        self.dataloader_lock = asyncio.Lock()

        # Active task set for concurrent sequence generation
        self.active_tasks = set()

        cpu_cores = multiprocessing.cpu_count()
        # cpu case use cpu_cores; io case use cpu_cores*2
        self.validate_executor = ThreadPoolExecutor(max_workers=cpu_cores)
        self.validate_task = None

        # ==================== Rollout Config (always on) ====================
        # Async rollout is always streaming per-sequence: per-prompt stop rules
        # decide when to stop generating more sequences. The classic
        # "every prompt rollout N times" mode == stop_rule=fixed_rollout with
        # N = ``actor_rollout_ref.rollout.n`` (the GRPO group size).
        #
        # NOTE: we deliberately do NOT expose a separate ``fixed_rollout_n``
        # knob. In fixed_rollout mode the per-prompt target N must equal
        # ``rollout.n`` so that each emitted RolloutSample has exactly one
        # complete GRPO group. Decoupling the two would silently break the
        # group-size contract consumed downstream.
        rollout_cfg = config.async_training.get("rollout_config", None) or {}

        # NOTE: ``working_pool_size`` has been REMOVED (pure streaming mode).
        #
        # The previous design capped the number of prompt_uids concurrently
        # alive in the response buffer. That upper bound has been replaced by:
        #   * vLLM-side concurrency (``max_num_seqs`` / per-server request
        #     capacity) which natively back-pressures the generation layer;
        #   * processor-side ``max_concurrent_samples`` which caps concurrent
        #     asyncio ``_process_one_response`` tasks in this actor;
        #   * a new prompt-level staleness gate (see ``_should_pause_feed``):
        #     as soon as ANY live prompt's staleness
        #       (current_param_version - start_param_version)
        #     reaches ``staleness_threshold``, the feed loop stops drawing
        #     fresh prompts and waits for every currently-live prompt to
        #     finish and be emitted before resuming.
        # Continue-rollout (additional sequences for an ALREADY-registered
        # prompt) is NOT blocked by this gate -- in-flight prompts always run
        # to their per-prompt stop criterion.
        pool_size = rollout_cfg.get("working_pool_size", None)
        if pool_size is not None:
            print(
                "[FullyAsyncRollouter] rollout_config.working_pool_size is DEPRECATED "
                "and will be ignored (pure streaming mode is now always on). "
                "Concurrency is controlled by vLLM (max_num_seqs) and "
                "max_concurrent_samples; prompt drain is controlled by the "
                "prompt-level staleness threshold."
            )

        # Adaptive rollout_n: the per-prompt target N is dynamically computed
        # from pass_rate via the strategy registry in
        # ``utils.async_prompt_buffer``. A single knob controls everything:
        #   adaptive_rollout_n_strategy: null / "" -> disabled (use static rollout.n as N)
        #                                "<name>"  -> enabled, dispatched via get_adaptive_rollout_n
        # Only effective when stop_rule == "fixed_rollout".
        _strategy_raw = rollout_cfg.get("adaptive_rollout_n_strategy", None)
        _strategy = str(_strategy_raw).strip() if _strategy_raw is not None else ""
        stop_rule = rollout_cfg.get("stop_rule", "fixed_rollout")
        # Static fixed_rollout threshold is locked to ``actor_rollout_ref.rollout.n``
        # (= GRPO group size). No separate config knob.
        _base_n = self._rollout_n
        # Fail-fast: adaptive_rollout_n_strategy is only meaningful under
        # stop_rule="fixed_rollout" (it overrides the static per-prompt N
        # threshold). Under any other stop_rule the per-prompt sequence count
        # is controlled by that rule itself (e.g. has_at_least_positive,
        # prefixed_rollout, max_rollout), and silently ignoring the strategy
        # would mask a real config error -- user sets a strategy, expects
        # adaptive behavior, but gets static rollout.n with no warning.
        if _strategy and stop_rule != "fixed_rollout":
            raise ValueError(
                f"[FullyAsyncRollouter] Config conflict: "
                f"adaptive_rollout_n_strategy='{_strategy}' is only supported when "
                f"stop_rule='fixed_rollout', but got stop_rule='{stop_rule}'. "
                f"Either set adaptive_rollout_n_strategy=null to use the static "
                f"per-prompt sequence count defined by stop_rule='{stop_rule}', "
                f"or switch stop_rule to 'fixed_rollout' to enable the adaptive strategy."
            )
        # Global hard cap on sequences per prompt. Used (a) as the buffer-layer
        # hard cap for *every* stop_rule, and (b) as the adaptive-strategy
        # clamp upper bound, so the two layers share a single source of truth
        # and there is no way for the strategy to return a value above the
        # buffer's hard cap.
        _hard_max_n = int(rollout_cfg.get("max_rollout_n", 32))
        dynamic_provider = None
        if _strategy and stop_rule == "fixed_rollout":
            _hard_min_n = int(rollout_cfg.get("min_rollout_n", 2))

            def _dynamic_fixed_n(
                prompt_uid: str,
                _base=_base_n,
                _min=_hard_min_n,
                _max=_hard_max_n,
                _strat=_strategy,
            ) -> int:
                meta = self.prompt_buffer.meta_data.get(prompt_uid, {})
                pr = float(meta.get("pass_rate", 0.5))
                return get_adaptive_rollout_n(pr, _base, min_n=_min, max_n=_max, strategy=_strat)

            dynamic_provider = _dynamic_fixed_n
            print(
                f"[FullyAsyncRollouter] adaptive_rollout_n ENABLED: "
                f"strategy='{_strategy}', base_n={_base_n}, min_n={_hard_min_n}, max_n={_hard_max_n} "
                f"(shared with buffer-level max_rollout_n). "
                f"per-prompt N will be computed dynamically from pass_rate."
            )

        self.response_buffer: AsyncResponseBuffer = AsyncResponseBuffer(
            stop_rule=stop_rule,
            fixed_rollout_n=_base_n,
            max_rollout_n=_hard_max_n,
            train_trigger=rollout_cfg.get("train_trigger", TRAIN_TRIGGER_FIXED_PROMPT),
            dynamic_fixed_n_provider=dynamic_provider,
        )
        # Event used to signal the feed task that: (a) a prompt has just been
        # emitted (live-prompt count decreased, drain may be finished), or
        # (b) a parameter sync has just happened (resume feeding). It is NOT
        # used as a "pool slot" signal anymore -- there is no pool.
        self.pool_slot_event: asyncio.Event | None = None
        print(
            f"[FullyAsyncRollouter] rollout_config: stop_rule={stop_rule}, "
            f"pure_streaming=True (working_pool_size removed), "
            f"fixed_rollout_n=rollout.n={_base_n}"
        )

        # ==================== Rejection Sampling (rollouter-side) ====================
        # When enabled, samples whose all sequences share the same reward sign
        # (solve_all or solve_none) are dropped here and NOT put onto the trainer
        # queue. We still accumulate per-prompt reward info (correct, total,
        # sampling_prob) so the trainer can compute unbiased raw_acc later via
        # consume_rejection_stats().
        self.rejection_sampling_enabled: bool = bool(
            rollout_cfg.get("rejection_sampling", False)
        )
        self._rejection_stats: dict = {
            # prompt-level counts (each _emit_prompt_as_rollout_sample call = 1 prompt)
            "total_prompts_seen": 0,
            "rejected_solve_all_prompts": 0,
            "rejected_solve_none_prompts": 0,
            "accepted_prompts": 0,
            # sample-level counts (each prompt contributes num_finished samples)
            "total_samples_seen": 0,
            "rejected_solve_all_samples": 0,
            "rejected_solve_none_samples": 0,
            "accepted_samples": 0,
        }
        # List[(correct_count, total_count, sampling_prob)] for ALL samples
        # (accepted + rejected). Drained each time the trainer consumes stats.
        self._rejection_reward_info: list[tuple[int, int, float]] = []
        # Lock to protect the above mutable state across coroutines.
        self._rejection_lock: asyncio.Lock | None = None
        if self.rejection_sampling_enabled:
            print(
                "[FullyAsyncRollouter] rejection_sampling: ENABLED "
                "(solve_all / solve_none samples are dropped before queueing)."
            )

    def _init_async_objects(self):
        # Initialize asyncio synchronization primitives.
        # We let asyncio.Condition create the Lock internally to ensure they share the same Event Loop.
        # This avoids 'ValueError: loop argument must agree with lock' which can occur in Ray environments
        # where the lock's captured loop (get_running_loop) differs from Condition's default loop check.
        # Explicitly passing the loop is deprecated/removed in Python 3.10+, so this reverse-initialization
        # is the most robust workaround.
        self.condition = asyncio.Condition()
        self.lock = self.condition._lock

        # Used by the adaptive-rollout feed task to block until a working-pool slot frees up.
        # Created here (not __init__) so that it binds to the Ray actor's running loop.
        self.pool_slot_event = asyncio.Event()

        # Lock guarding rejection-sampling counters / reward_info list
        # (mutated by _emit_prompt_as_rollout_sample and drained by consume_rejection_stats).
        self._rejection_lock = asyncio.Lock()

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_max_queue_size(self):
        async with self.lock:
            self.max_queue_size = int(
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            # `total_rollout_steps` and `required_samples` are both in SEQUENCE
            # units, so the ratio directly yields the number of training steps.
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )

            self.max_concurrent_samples = len(self.async_rollout_manager.server_handles) * 32
            self.max_concurrent_samples = min(self.max_concurrent_samples, self.max_queue_size)

            print(
                f"[FullyAsyncRollouter] required_samples: {self.required_samples} "
                f"required_prompts: {self.required_prompts} "
                f"max_queue_size: {self.max_queue_size} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps (sequences): {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
            )

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def get_replicas(self):
        """Get rollout worker group"""
        return self.async_rollout_manager.rollout_replicas

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        return self.total_train_steps

    async def reset_staleness(self, current_param_version: int | None = None):
        """
        Reset staleness counters after parameter update.
        Returns timing_raw dictionary for metrics.

        Args:
            current_param_version: The trainer's ``current_param_version``
                AFTER the just-finished parameter sync. When provided, the
                rollouter uses this value to (a) stamp freshly registered
                prompts' ``start_param_version`` and (b) drive the
                prompt-level staleness gate. When ``None`` (backwards
                compatibility), falls back to incrementing the previous
                value by 1 (assuming one sync per call).
        """
        async with self.lock:
            if current_param_version is not None:
                self.current_param_version = int(current_param_version)
            else:
                self.current_param_version += 1
            self.paused = False
            # Wake up feed task waiting for drain + resume.
            if self.pool_slot_event is not None:
                self.pool_slot_event.set()
            self.condition.notify_all()
            # Reset stale_samples using the CONDITIONAL accumulation rule:
            # only count samples whose start_param_version < current_param_version.
            # Samples born under the NEW current_param_version have staleness 0
            # and must NOT be counted.
            #
            # Scope = buffer-resident samples ONLY (in-flight + not-yet-emitted
            # finished). We intentionally do NOT include message-queue pending
            # samples here, because:
            #   * The trainer calls ``reset_staleness`` BEFORE
            #     ``purge_stale_samples``, so any queue sample whose staleness
            #     has reached ``staleness_threshold`` is about to be removed.
            #     Including them would transiently over-count.
            #   * The queue may legitimately hold samples with staleness in
            #     [0, staleness_threshold - 1] -- these are NOT stale in the
            #     "expired" sense and should not inflate ``stale_samples``.
            #   * The queue's own purge counters (``purged_prompts`` /
            #     ``purged_samples``) already track queue-side drops, so there
            #     is no statistic loss.
            # This matches the scope of the running ``stale_samples``
            # accumulator updated in ``_process_one_response`` (buffer-side
            # only, at finish time) and keeps the two paths symmetric.
            self.stale_samples = self.response_buffer.count_stale_samples(
                self.current_param_version
            )
            timing_raw = {}
            rollout_version_time = time.time() - self.step_start_time
            # idle_start_time == -1 means no idle occurred during this version interval,
            # so active_time equals the full version_time and idle_ratio is 0.
            if self.idle_start_time < 0:
                rollout_active_time = rollout_version_time
                idle_ratio = 0.0
            else:
                rollout_active_time = self.idle_start_time - self.step_start_time
                idle_ratio = 1 - rollout_active_time / rollout_version_time if rollout_version_time > 0 else 0.0
            timing_raw["fully_async/rollouter/active_time"] = rollout_active_time
            timing_raw["fully_async/rollouter/version_time"] = rollout_version_time
            timing_raw["fully_async/rollouter/idle_ratio"] = idle_ratio

            stale_prompts = self.response_buffer.num_stale_prompts(
                self.current_param_version
            )
            print(
                f"[FullyAsyncRollouter][Public][reset_staleness] "
                f"reset stale_samples to: {self.stale_samples}, "
                f"stale_prompts: {stale_prompts}, "
                f"idle_ratio: {timing_raw['fully_async/rollouter/idle_ratio']:.4f}"
            )
            self.step_start_time = time.time()
            self.idle_start_time = -1  # Reset sentinel: no idle until next pause
        return timing_raw

    async def resume_generation(self):
        """Resume generation without parameter sync.

        Called by the Trainer when rejection sampling has exhausted the queue
        but not enough valid samples have been collected.  This resets the
        staleness counter so the Rollouter can produce more samples using the
        *current* model weights (no checkpoint transfer happens).
        """
        async with self.lock:
            old_staleness = self.stale_samples
            self.stale_samples = 0
            self.paused = False
            self.condition.notify_all()
            print(
                f"[FullyAsyncRollouter][Public][resume_generation] "
                f"Rollouter resumed (no param sync). "
                f"stale_samples: {old_staleness} -> 0"
            )

    def do_validate(self) -> ValidateMetrics:
        """Run validation and return metrics"""
        timing_raw = {}
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = self._validate()
        return ValidateMetrics(timing_raw=timing_raw, metrics=val_metrics)

    def _validate(self, merged: bool = False):
        """Override parent _validate to collect response_length for enhanced metrics.

        This is a copy of RayPPOTrainer._validate with the following additions:
        - #NOTE: collect response_length and max_response_length per sample
        """
        from verl import DataProto
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from verl.trainer.ppo.reward import extract_reward

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # NOTE: collect response_length and max_response_length for enhanced val metrics
            tensor_response_length = test_batch.batch["responses"].shape[-1]
            response_mask = test_batch.batch["attention_mask"][:, -tensor_response_length:]
            batch_response_lengths = response_mask.sum(-1).cpu().tolist()  # list of int
            reward_extra_infos_dict["response_length"].extend(batch_response_lengths)
            # Use tensor_response_length (responses.shape[-1]) for clip_ratio calculation.
            # This matches the training-stage logic in metric_utils.py where
            # max_response_length = batch.batch["responses"].shape[-1].
            # Using val_kwargs.max_response_length would be incorrect because
            # response_mask.sum(-1) is bounded by tensor_response_length, which
            # may be smaller than val_max_response_length (e.g. when max_model_len
            # limits the total sequence length).
            reward_extra_infos_dict["max_response_length"].extend(
                [tensor_response_length] * len(batch_response_lengths)
            )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), (
                f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            )

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        """Override parent to use enhanced validation metrics with overall averaging
        and per-data-source response length stats.
        """
        from verl.experimental.fully_async_policy.utils.parallel_validation_metrics import enhanced_val_metrics_update

        return enhanced_val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    async def save_checkpoint(self, local_global_step_folder: str):
        # WARNING!: Due to the asynchronous nature, there are some in-flight samples
        # (pending/cancel/result queue and message queue).
        # Therefore, directly saving the state of the dataloader will result in losing these
        # samples when resuming training.
        # TODO: Implement dataloader recovery without losing in-flight samples.
        from verl.utils.fs import local_mkdir_safe

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        async with self.dataloader_lock:
            dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        print(f"[FullyAsyncRollouter] Saved dataloader checkpoint to {dataloader_local_path}")

        # Save prompt buffer state (pass_rate, count, generator state)
        buffer_local_path = os.path.join(local_global_step_folder, "prompt_buffer.pt")
        async with self.dataloader_lock:
            buffer_state_dict = self.prompt_buffer.state_dict()
        torch.save(buffer_state_dict, buffer_local_path)
        print(f"[FullyAsyncRollouter] Saved prompt buffer checkpoint to {buffer_local_path}")

    def load_checkpoint(self):
        """Load checkpoint including dataloader state based on resume mode"""

        if self.config.trainer.resume_mode == "disable":
            print("[FullyAsyncRollouter] Resume mode is disabled, starting from scratch")
            return 0

        # Determine checkpoint folder path
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("[FullyAsyncRollouter] Load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        # Find and validate global_step_folder based on resume mode
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[FullyAsyncRollouter] Training from scratch (no checkpoint found)")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), (
                "[FullyAsyncRollouter] resume_from_path must be str type"
            )
            assert "global_step_" in self.config.trainer.resume_from_path, (
                "[FullyAsyncRollouter] resume_from_path must specify the global_steps"
            )
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            raise ValueError(f"[FullyAsyncRollouter] Unknown resume_mode: {self.config.trainer.resume_mode}")

        print(f"[FullyAsyncRollouter] Loading checkpoint from: {global_step_folder}")

        # Extract and set global step
        trainer_global_steps = int(global_step_folder.split("global_step_")[-1])
        # `self.global_steps` is in SEQUENCE units (see _process_one_response),
        # matching the sequence-level `required_samples`, so the restore point
        # is simply: trainer_steps * required_samples * trigger_parameter_sync_step.
        self.global_steps = (
            trainer_global_steps * self.required_samples * self.config.async_training.trigger_parameter_sync_step + 1
        )
        print(f"[FullyAsyncRollouter] Setting global_steps to {self.global_steps}")

        # Load dataloader state
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            print(f"[FullyAsyncRollouter] Loaded dataloader state from {dataloader_local_path}")
        else:
            print(
                f"[FullyAsyncRollouter] Warning: No dataloader state found at {dataloader_local_path}, "
                f"will start from scratch"
            )

        # Load prompt buffer state (pass_rate, count, generator state)
        buffer_local_path = os.path.join(global_step_folder, "prompt_buffer.pt")
        if os.path.exists(buffer_local_path):
            buffer_state_dict = torch.load(buffer_local_path, weights_only=False)
            self.prompt_buffer.load_state_dict(buffer_state_dict)
            print(f"[FullyAsyncRollouter] Loaded prompt buffer state from {buffer_local_path}")
        else:
            print(
                f"[FullyAsyncRollouter] Warning: No prompt buffer state found at {buffer_local_path}, "
                f"will start from scratch"
            )

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_async_objects()
        self._create_worker_classes()
        self._init_reward_loop()
        await self._init_async_rollout_manager()

    def _create_actor_rollout_classes(self):
        # Skip rollout creation and let agentloop handle it
        pass

    def _init_models(self):
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _init_async_rollout_manager(self):
        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from verl.experimental.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config, worker_group=self.rollout_wg, reward_loop_worker_handles=reward_loop_worker_handles
        )

    async def _feed_responses(self):
        """Main generation loop: draws prompts, applies back-pressure, and
        directly dispatches ``_process_one_response`` tasks.

        This method replaces the previous ``_feed_responses`` + ``_processor_worker``
        + ``pending_queue`` architecture. All scheduling logic (pause-on-staleness,
        max-concurrent-sequences cap) is handled inline, and tasks are created
        directly via ``safe_create_task`` -- no intermediate queue is needed.

        Semantics
        ---------
        * Continuously draws fresh prompts from ``AsyncPromptBuffer``.
        * Applies three back-pressure gates before dispatching:
          1. Pause-on-staleness (``_should_pause_generation``)
          2. Prompt-level staleness drain (``_feed_gate_closed``)
          3. Concurrency cap (``max_concurrent_samples``)
        * For every new prompt, dispatches ONE sequence task; additional
          sequence tasks for the same prompt are dispatched by
          ``_process_one_response`` (continue-rollout path) via
          ``_submit_one_response_job``.
        """

        async def _wait_for_drain() -> None:
            """Wait until all live prompts have drained
            (``num_active_prompts() == 0``). As soon as the live set is empty,
            ``_feed_gate_closed()`` trivially evaluates to False (max staleness
            over an empty set is 0), and feed may resume -- regardless of
            whether a parameter sync happened in the meantime.
            """
            while True:
                if self.global_steps >= self.total_rollout_steps:
                    return
                if self.response_buffer.num_active_prompts() == 0:
                    return
                if self.pool_slot_event is not None:
                    self.pool_slot_event.clear()
                    try:
                        await asyncio.wait_for(self.pool_slot_event.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(0.5)

        _resample_consecutive = 0

        while self.global_steps < self.total_rollout_steps:
            # ---- Back-pressure gate 1: pause-on-staleness ----
            if self.paused or await self._should_pause_generation():
                async with self.lock:
                    self.paused = True
                # Drain all active tasks before sleeping
                while self.active_tasks:
                    done_tasks, _ = await asyncio.wait(
                        self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done_tasks:
                        await task
                async with self.lock:
                    while self.paused:
                        self.idle_start_time = time.time()
                        await self.condition.wait()
                continue

            # Prompt-level staleness gate: when closed, stop drawing fresh
            # prompts and wait for the currently-live set to drain to zero.
            # We do NOT wait for a parameter sync -- see docstring above.
            if self._feed_gate_closed():
                max_live_staleness = self.response_buffer.max_live_prompt_staleness(
                    self.current_param_version
                )
                print(
                    "[FullyAsyncRollouter][Feed] prompt-level staleness gate CLOSED "
                    f"(max_live_prompt_staleness={max_live_staleness} "
                    f">= staleness_threshold={self.staleness_threshold}). "
                    "Draining in-flight prompts before resuming feed...",
                    flush=True,
                )
                await _wait_for_drain()
                if self.global_steps >= self.total_rollout_steps:
                    break
                print(
                    "[FullyAsyncRollouter][Feed] live-prompt set drained; "
                    f"resuming feed with current_param_version={self.current_param_version}.",
                    flush=True,
                )
                continue  # Re-evaluate all back-pressure gates from the top

            # Draw a fresh prompt from the AsyncPromptBuffer (prompt-side sampling).
            async with self.dataloader_lock:
                batch_dict, prompt_uid, sampling_prob = self.prompt_buffer.get_single_sample()

            # Avoid duplicate active prompt_uid (same uid rolled out concurrently would
            # confuse the response buffer). If a collision happens, just skip and retry.
            if self.response_buffer.get_state(prompt_uid) is not None:
                _resample_consecutive += 1
                if _resample_consecutive >= 20:
                    await asyncio.sleep(0.01)
                else:
                    await asyncio.sleep(0)
                continue

            _resample_consecutive = 0

            # Register in response buffer and submit ONE sequence job.
            # Stamp the prompt's birth param version so we can compute its
            # prompt-level staleness later.
            self.response_buffer.register_prompt(
                prompt_uid,
                sampling_prob=sampling_prob,
                start_param_version=self.current_param_version,
            )
            self.response_buffer.on_job_submitted(prompt_uid)

            # ---- Back-pressure gate 3: concurrency cap ----
            # Wait until there is room before dispatching.
            while len(self.active_tasks) >= self.max_concurrent_samples:
                if not self.active_tasks:
                    break
                done_tasks, _ = await asyncio.wait(
                    self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done_tasks:
                    await task

            # Dispatch the first sequence task directly (no queue).
            self._submit_one_response_job(prompt_uid, batch_dict, sampling_prob)

        print(
            f"[FullyAsyncRollouter][Feed] reached total_rollout_steps: "
            f"{self.global_steps} >= {self.total_rollout_steps}. Waiting for residual tasks..."
        )
        # Drain all remaining active tasks.
        while self.active_tasks:
            done_tasks, _ = await asyncio.wait(
                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done_tasks:
                await task
        print(
            f"[FullyAsyncRollouter][Feed] All rollout tasks drained. "
            f"Total feed steps: {self.global_steps}"
        )

    def _feed_gate_closed(self) -> bool:
        """Return True iff the prompt-level staleness gate is CLOSED and the
        feed loop should stop drawing fresh prompts.

        The gate is closed when any live prompt's staleness
        (``current_param_version - start_param_version``) reaches
        ``staleness_threshold``. When there are no live prompts, the gate is
        trivially open (max_live_prompt_staleness returns 0).

        When ``staleness_threshold <= 0`` the gate is always OPEN (disabled).
        Without this short-circuit, staleness >= 0 would be trivially True
        for every live prompt, permanently blocking the feed loop.
        """
        if self.staleness_threshold <= 0:
            return False
        max_live_staleness = self.response_buffer.max_live_prompt_staleness(
            self.current_param_version
        )
        return max_live_staleness >= self.staleness_threshold

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        print(f"[FullyAsyncRollouter] Start streaming mode, maximum concurrent sequences: {self.max_concurrent_samples}")

        # Single coroutine handles everything: prompt drawing, back-pressure,
        # task dispatch, and draining. No separate processor or queue needed.
        self.feed_task = safe_create_task(self._feed_responses(), name="feed_task")

        try:
            await self.feed_task
            print("[FullyAsyncRollouter] Feed task completed")

        except Exception as e:
            print(f"[FullyAsyncRollouter] Streaming process exception: {e}")
            raise e

        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)

            self.feed_task = None

            # Send a finish signal. The end-of-stream sentinel carries no real
            # samples, so pass ``n_samples=0`` to keep the queue's
            # pending_samples counter exact.
            await self.message_queue_client.put_sample(sample=None, n_samples=0)

        async with self.lock:
            self.running = False

    async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines
        """

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.running = True

        # Create the main asynchronous task
        generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            # Run build and monitoring tasks concurrently
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[FullyAsyncRollouter] Rollouter fit completed")

    async def _async_monitor_loop(self):
        """
        Async coroutine for monitoring:
        Function 1: Log information output
        Function 2: Trigger rollout recovery
        """
        last_stats_time = time.time()
        stats_interval = 60.0
        check_interval = 10.0

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)
            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[FullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            # Trigger rollout recovery
            if self.paused and not await self._should_pause_generation():
                async with self.lock:
                    self.paused = False
                    print("[FullyAsyncRollouter][ShouldPause] notify all wait tasks.")
                    self.condition.notify_all()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused.

        There are TWO back-pressure sources:

        1. ``pending_samples >= max_queue_size``: MessageQueue is full
           (trainer is slower than rollouter). Hard cap. Both quantities are
           in sample units; comparing against ``queue_size`` (prompts)
           would make the branch effectively dead.
        2. *Prompt-level staleness gate*: any live prompt whose
           ``current_param_version - start_param_version`` reaches
           ``staleness_threshold``. This is the primary gate in pure-streaming
           mode; it is evaluated against the same condition ``_feed_responses``
           uses so feed-drain and processor-pause agree on when to stop.
        """
        queue_stats = self.message_queue_client.get_statistics_sync()
        queue_size = queue_stats["queue_size"]
        pending_samples = queue_stats["pending_samples"]

        if pending_samples >= self.max_queue_size:
            if not self.paused:
                print(
                    f"[FullyAsyncRollouter][ShouldPause]  "
                    f"due to full queue: pending_samples={pending_samples}, "
                    f"max={self.max_queue_size} (queue_size={queue_size} prompts)"
                )
            return True

        # Prompt-level staleness gate (pure-streaming primary gate).
        if self._feed_gate_closed():
            if not self.paused:
                max_live_staleness = self.response_buffer.max_live_prompt_staleness(
                    self.current_param_version
                )
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to prompt-level staleness gate closed: "
                    f"max_live_prompt_staleness={max_live_staleness} "
                    f">= staleness_threshold={self.staleness_threshold}"
                )
            return True

        return False

    async def get_statistics(self) -> dict:
        queue_stats = self.message_queue_client.get_statistics_sync()

        stats = {
            # monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            "monitor/queue/mq_pending_samples": queue_stats["pending_samples"],
            # counting stats — prompt level
            "count/total_generated_prompts": self.total_generated_prompts,
            "count/stale_prompts": self.response_buffer.num_stale_prompts(
                self.current_param_version
            ),
            "count/dropped_prompts": self.dropped_prompts,
            "count/rejected_prompts": self.rejected_prompts,
            # counting stats — sample level
            "count/total_generated_samples": self.total_generated_samples,
            "count/stale_samples": self.stale_samples,
            "count/dropped_samples": self.dropped_samples,
            "count/rejected_samples": self.rejected_samples,
            # other counting stats
            "count/current_param_version": self.current_param_version,
            "count/max_live_prompt_staleness": self.response_buffer.max_live_prompt_staleness(
                self.current_param_version
            ),
            # static stats
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
        }

        stats.update(self.response_buffer.stats())

        # Rejection sampling live counters (cumulative since last consume).
        if self.rejection_sampling_enabled:
            stats["rejection/total_prompts_seen"] = self._rejection_stats["total_prompts_seen"]
            stats["rejection/accepted_prompts"] = self._rejection_stats["accepted_prompts"]
            stats["rejection/rejected_solve_all_prompts"] = self._rejection_stats["rejected_solve_all_prompts"]
            stats["rejection/rejected_solve_none_prompts"] = self._rejection_stats["rejected_solve_none_prompts"]
            stats["rejection/total_samples_seen"] = self._rejection_stats["total_samples_seen"]
            stats["rejection/accepted_samples"] = self._rejection_stats["accepted_samples"]
            stats["rejection/rejected_solve_all_samples"] = self._rejection_stats["rejected_solve_all_samples"]
            stats["rejection/rejected_solve_none_samples"] = self._rejection_stats["rejected_solve_none_samples"]

        return stats

    # =====================================================================
    # Streaming per-sequence rollout with per-prompt stop rule
    # =====================================================================

    def _submit_one_response_job(self, prompt_uid: str, batch_dict: dict, sampling_prob: float):
        """Prepare a single-sequence (repeat_times=1) job and dispatch it
        directly as an asyncio task.

        Caller is responsible for calling response_buffer.on_job_submitted()
        BEFORE calling this method, so that the in-flight counter is accurate.

        NOTE: this helper does NOT advance self.global_steps. Since
        `total_rollout_steps` is sequence-level, `global_steps` is advanced by
        `_process_one_response()` every time a sequence finishes -- regardless
        of whether the sequence belongs to a freshly drawn prompt or to a
        continue-rollout of an existing prompt.
        """
        full_batch = prepare_single_generation_data(batch_dict, self.config, repeat_times=1)
        # `global_steps` is now sequence-level; convert to (approximate) epoch
        # by dividing by ``#prompts * rollout.n`` so the value stays consistent
        # with the dataloader's pass over the dataset.
        epoch = self.global_steps // max(1, len(self.train_dataset) * self._rollout_n)
        sample_id = f"sample_{epoch}_{self.global_steps}_{prompt_uid}"
        rollout_sample = RolloutSample(
            full_batch=full_batch,
            sample_id=sample_id,
            epoch=epoch,
            rollout_status={},
            prompt_uid=prompt_uid,
            sampling_prob=sampling_prob,
            rollout_n=1,
        )
        safe_create_task(
            self._process_one_response(rollout_sample),
            name=rollout_sample.sample_id,
            task_set=self.active_tasks,
        )

    async def _process_one_response(self, rollout_sample: "RolloutSample"):
        """Processor body: generates ONE sequence, records it in the response
        buffer, and either:
          * submits another sequence job for the same prompt_uid (continue), or
          * aggregates all finished sequences and emits a single RolloutSample (stop).

        Thread-safety note (asyncio single-threaded event loop):
            Multiple ``_process_one_response`` coroutines run concurrently for
            *different* prompt_uids. They share ``self.response_buffer`` and
            ``self.processed_sample_count`` without an explicit lock.
            This is safe under the following assumptions:
              1. CPython's GIL guarantees that pure-Python dict/int operations
                 are atomic at the bytecode level.
              2. For the *same* prompt_uid, the next sequence job is only
                 created inside this coroutine (via ``_submit_one_response_job``),
                 so two coroutines never operate on the same PromptState
                 simultaneously.
              3. The ``await`` points (``generate_sequences_single``,
                 ``self.lock``) only yield control between logically
                 independent prompt_uid operations.
            If this code is ever moved to a multi-threaded executor or the
            response_buffer is accessed from outside the event loop, an
            asyncio.Lock MUST be added around the response_buffer mutation
            sequence (on_response_finished → stop_generation →
            mark_if_complete → pop_completed_prompt).
        """
        prompt_uid = rollout_sample.prompt_uid
        # Preserve fields from the original batch that are lost after agent loop processing.
        original_ntb = rollout_sample.full_batch.non_tensor_batch
        preserved_fields = {}
        for key in ("reward_model", "data_source"):
            if key in original_ntb:
                preserved_fields[key] = original_ntb[key]

        ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
        rollout_sample.full_batch = ret

        for key, val in preserved_fields.items():
            if key not in rollout_sample.full_batch.non_tensor_batch:
                rollout_sample.full_batch.non_tensor_batch[key] = val

        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{prompt_uid}"] * len(rollout_sample.full_batch), dtype=object
        )

        # Extract per-sequence reward. rm_scores shape: [1, response_length].
        reward = 0.0
        try:
            if "rm_scores" in rollout_sample.full_batch.batch.keys():
                reward = float(rollout_sample.full_batch.batch["rm_scores"].sum(dim=-1).item())
        except Exception as e:
            print(f"[FullyAsyncRollouter] Failed to extract reward for {prompt_uid}: {e}")

        # Record in response buffer.
        self.response_buffer.on_response_finished(
            prompt_uid=prompt_uid,
            sequence=rollout_sample.full_batch,
            reward=reward,
        )

        # Only count sequences whose prompt was born under an older parameter
        # version as "stale". A sequence whose start_param_version ==
        # current_param_version has staleness 0 and should NOT inflate the
        # staleness counter -- it was generated with the latest weights.
        state_for_staleness = self.response_buffer.get_state(prompt_uid)
        async with self.lock:
            if state_for_staleness is not None and (
                self.current_param_version > state_for_staleness.start_param_version
            ):
                self.stale_samples += 1
            self.global_steps += 1

        # Decide: stop & emit, or continue with one more sequence.
        state = self.response_buffer.get_state(prompt_uid)
        should_stop = self.response_buffer.stop_generation(prompt_uid)

        if not should_stop:
            # Submit another sequence job for the same prompt_uid.
            try:
                batch_dict = self._materialize_batch_dict_for_uid(prompt_uid)
                self.response_buffer.on_job_submitted(prompt_uid)
                self._submit_one_response_job(
                    prompt_uid=prompt_uid,
                    batch_dict=batch_dict,
                    sampling_prob=state.sampling_prob if state else 0.0,
                )
            except Exception as e:
                print(
                    f"[FullyAsyncRollouter] Failed to re-submit for prompt_uid={prompt_uid}: {e}. "
                    f"Forcing stop."
                )
                # Roll back the in_flight counter that was incremented by
                # on_job_submitted *before* the failed _submit_one_response_job.
                # Without this, in_flight stays > 0 and mark_if_complete will
                # never succeed, leaking the prompt from the working pool.
                if state is not None:
                    state.in_flight = max(0, state.in_flight - 1)
                    state.stopped = True
                should_stop = True

        if should_stop:
            # Mark as complete if no in-flight jobs remain, and emit.
            if self.response_buffer.mark_if_complete(prompt_uid):
                # Update prompt buffer pass-rate tracking using *all* rewards
                # collected for this prompt. Called exactly ONCE per prompt
                # (at emit time) so that count is incremented once and EMA
                # is applied to the final, complete set of rewards.
                try:
                    final_state = self.response_buffer.get_state(prompt_uid)
                    if final_state is not None:
                        self.prompt_buffer.on_sample_end(prompt_uid, list(final_state.rewards))
                except Exception as e:
                    print(f"[FullyAsyncRollouter] prompt_buffer.on_sample_end failed: {e}")
                await self._emit_prompt_as_rollout_sample(prompt_uid)
                # Notify feed task: a working-pool slot has freed up.
                if self.pool_slot_event is not None:
                    self.pool_slot_event.set()

        # NOTE: Safe without a lock in asyncio single-threaded event loop
        # (CPython GIL makes int += atomic). See docstring thread-safety note.
        self.processed_sample_count += 1

    def _materialize_batch_dict_for_uid(self, prompt_uid: str) -> dict:
        """Re-fetch the dataset row for ``prompt_uid`` and collate it into a batch dict.

        Used when we need to submit another sequence job for the same prompt_uid
        without advancing the prompt_buffer's sampler state.
        """
        pb = self.prompt_buffer
        meta = pb.meta_data.get(prompt_uid, {})
        row_indices = meta.get("row_indices", [])
        if not row_indices:
            raise RuntimeError(f"prompt_uid={prompt_uid} has no row_indices in prompt_buffer")
        row_idx = row_indices[0]
        sample = pb.dataset[row_idx]
        # Mirror AsyncPromptBuffer.get_single_sample() attachments.
        # Use the real sampling_prob from the response buffer state (set at
        # register_prompt time), NOT pass_rate which is a different quantity.
        state = self.response_buffer.get_state(prompt_uid)
        sample["sampling_prob"] = state.sampling_prob if state is not None else 0.0
        sample["prompt_uid"] = prompt_uid
        sample["pass_rate"] = meta.get("pass_rate", 0.5)
        return pb.collate_fn([sample])

    async def _emit_prompt_as_rollout_sample(self, prompt_uid: str):
        """Aggregate all finished sequences of ``prompt_uid`` into a single
        RolloutSample (length = actual_n) and push it onto the message queue.

        When ``rejection_sampling_enabled`` is True, samples whose sequence-level
        rewards are all identical (all > 0 = solve_all, or all <= 0 = solve_none)
        are DROPPED here and not put on the queue. Per-prompt reward info is
        always appended to ``_rejection_reward_info`` so the trainer can compute
        an unbiased raw_acc / unweighted_acc later via consume_rejection_stats().
        """
        from verl import DataProto

        state = self.response_buffer.pop_completed_prompt(prompt_uid)
        if state is None or state.num_finished == 0:
            return

        try:
            aggregated = DataProto.concat(state.sequences)
        except Exception as e:
            print(f"[FullyAsyncRollouter] DataProto.concat failed for {prompt_uid}: {e}")
            return

        # Rebuild uid column: all sequences share the same prompt uid.
        aggregated.non_tensor_batch["uid"] = np.array(
            [f"uid_{prompt_uid}"] * len(aggregated), dtype=object
        )

        # ---- Rejection sampling + reward-info accounting ----
        # Compute (correct_count, total_count) from the aggregated rm_scores.
        correct_count = 0
        total_count = state.num_finished
        try:
            if "rm_scores" in aggregated.batch.keys():
                _rs = aggregated.batch["rm_scores"].sum(dim=-1)  # [num_finished]
                correct_count = int((_rs > 0).sum().item())
        except Exception as e:
            print(f"[FullyAsyncRollouter] Failed to extract rm_scores for {prompt_uid}: {e}")

        reject_reason = ""
        if self.rejection_sampling_enabled and total_count >= 2:
            if correct_count == total_count:
                reject_reason = "solve_all"
            elif correct_count == 0:
                reject_reason = "solve_none"

        # Always record reward info for raw_acc / unweighted_acc (incl. rejected).
        async with self._rejection_lock:
            # prompt-level
            self._rejection_stats["total_prompts_seen"] += 1
            # sample-level
            self._rejection_stats["total_samples_seen"] += total_count
            self._rejection_reward_info.append(
                (correct_count, total_count, float(state.sampling_prob))
            )
            if reject_reason == "solve_all":
                self._rejection_stats["rejected_solve_all_prompts"] += 1
                self._rejection_stats["rejected_solve_all_samples"] += total_count
            elif reject_reason == "solve_none":
                self._rejection_stats["rejected_solve_none_prompts"] += 1
                self._rejection_stats["rejected_solve_none_samples"] += total_count
            else:
                self._rejection_stats["accepted_prompts"] += 1
                self._rejection_stats["accepted_samples"] += total_count

        if reject_reason:
            # Dual-level rejection counting
            self.rejected_prompts += 1
            self.rejected_samples += total_count
            # Drop without queueing. Log every 32 rejects.
            if self._rejection_stats["total_samples_seen"] % 32 == 0:
                print(
                    f"[FullyAsyncRollouter][RejectionSampling] "
                    f"Rejected {prompt_uid} ({reject_reason}). "
                    f"Stats: {self._rejection_stats}"
                )
            return

        rollout_sample = RolloutSample(
            full_batch=aggregated,
            sample_id=f"adaptive_{prompt_uid}_{len(state.sequences)}",
            epoch=0,
            rollout_status=await self.get_statistics(),
            prompt_uid=prompt_uid,
            sampling_prob=state.sampling_prob,
            rollout_n=state.num_finished,
        )

        # Attach prompt_buffer metadata (pass_rate / count) after on_sample_end updates.
        try:
            meta = self.prompt_buffer.meta_data.get(prompt_uid, {})
            rollout_sample.pass_rate = float(meta.get("pass_rate", -1.0))
            rollout_sample.sample_count = int(meta.get("count", 0))
        except Exception:
            pass

        success = await self.message_queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
            n_samples=int(state.num_finished),
            start_version=int(state.start_param_version),
        )
        if success:
            self.total_generated_prompts += 1
            self.total_generated_samples += int(state.num_finished)
        else:
            self.dropped_prompts += 1
            self.dropped_samples += int(state.num_finished)

    async def consume_rejection_stats(self) -> dict:
        """Drain and return the accumulated rejection-sampling stats + reward info.

        Called by the trainer once per training step to attach unbiased raw_acc
        metadata to the assembled batch. The internal counters / list are reset
        so the next call returns only the delta since the previous consumption.
        """
        async with self._rejection_lock:
            stats = dict(self._rejection_stats)
            reward_info = list(self._rejection_reward_info)
            # Reset for the next window.
            for k in self._rejection_stats:
                self._rejection_stats[k] = 0
            self._rejection_reward_info.clear()
        return {
            "enabled": self.rejection_sampling_enabled,
            "stats": stats,
            "reward_info": reward_info,
        }
