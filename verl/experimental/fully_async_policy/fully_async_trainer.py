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
import json
import logging
import os
import time
from datetime import datetime
from pprint import pprint
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    ValidateMetrics,
    assemble_batch_from_rollout_samples,
    compute_staleness_metrics,
)
from verl.experimental.fully_async_policy.utils.batch_metrics import compute_batch_training_signal_metrics
from verl.experimental.fully_async_policy.utils.filter_zero_adv import (
    mask_out_zero_variance_groups,
    compute_grpo_advantage_exclude_pad,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, apply_kl_penalty, compute_advantage
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking

logger = logging.getLogger(__name__)


class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(SeparateRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
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
        # ==================== RayPPOTrainer config ====================

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # ==================== SeparateRayPPOTrainer config ====================
        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # ==================== fully async config ====================

        self.message_queue_client = None

        # Statistics
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        _raw_staleness = config.async_training.get("staleness_threshold", 1)
        assert int(_raw_staleness) == float(_raw_staleness), (
            f"staleness_threshold must be an integer value, got {_raw_staleness}"
        )
        self.staleness_threshold: int = int(_raw_staleness)
        self.last_ckpt_version = 0
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        # required_samples is SEQUENCE-LEVEL: total generated sequences required for
        # one training step. Equals ppo_mini_batch_size (prompt units, as consumed by
        # verl's downstream trainer) * rollout.n * require_batches.
        # Example: ppo_mini_batch_size=32, rollout.n=8, require_batches=1 -> 256.
        self.require_batches = config.async_training.require_batches
        self._rollout_n = max(1, int(config.actor_rollout_ref.rollout.n))
        self.required_prompts = (
            config.actor_rollout_ref.actor.ppo_mini_batch_size
            * self.require_batches
        )
        self.required_samples = self.required_prompts * self._rollout_n

        # ==================== Rollout training trigger ====================
        # When train_trigger=fixed_samples, we accept a variable number of
        # RolloutSamples per training step as long as the TOTAL number of
        # samples (summed over RolloutSamples) reaches a configured target.
        # Default (fixed_prompt): collect required_samples samples (spread
        # across one or more RolloutSamples, each bundling 1..max_rollout_n
        # samples depending on the stop rule).
        rollout_cfg = config.async_training.get("rollout_config", None) or {}
        self.train_trigger = rollout_cfg.get("train_trigger", "fixed_prompt")
        if self.train_trigger == "fixed_samples":
            _ts = rollout_cfg.get("target_samples", None)
            if _ts is None:
                # required_samples is already sample-level, no extra * rollout.n needed.
                _ts = self.required_samples
            self.target_samples = int(_ts)
            logger.info(
                "[FullyAsyncTrainer] train_trigger=fixed_samples, target_samples=%d",
                self.target_samples,
            )
        else:
            self.target_samples = None
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        # use trainer to do validation
        if self.config.async_training.use_trainer_do_validate:
            from verl.trainer.main_ppo import create_rl_dataset
            from verl.utils.dataset.rl_dataset import collate_fn

            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
            print(f"[FullyAsyncTrainer] val_dataset total len: {len(val_dataset)}")

            # In val_only mode, use the full dataset without splitting
            if not self.config.trainer.get("val_only", False):
                rollout_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
                split_dataset = val_dataset.split(total_gpus)
                rollout_val_dataset0 = split_dataset[rollout_gpus:]
                from torch.utils.data import ConcatDataset

                val_dataset = ConcatDataset(rollout_val_dataset0)
                print(f"[FullyAsyncTrainer] split after val_dataset total len: {len(val_dataset)}")
            self.val_dataset = val_dataset
            # update val_dataloader
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(val_dataset)
            from torchdata.stateful_dataloader import StatefulDataLoader

            print(f"[FullyAsyncTrainer] create val_dataloader with batch_size: {val_batch_size}")
            self.val_dataloader = StatefulDataLoader(
                dataset=val_dataset,
                batch_size=val_batch_size,
                num_workers=self.config.data["dataloader_num_workers"],
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
        # ==================== Rejection Sampling (trainer-side flag only) ====================
        # The actual filtering is done by the rollouter; here we only remember whether
        # it is enabled so we know to query consume_rejection_stats() after each batch.
        # Config lives inside rollout_config (rollout-behavior family).
        _rollout_cfg_for_rej = config.async_training.get("rollout_config", None) or {}
        self.rejection_sampling_enabled = _rollout_cfg_for_rej.get("rejection_sampling", False)

        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.checkpoint_manager = None

        # when use_trainer_do_validate == Ture, use colocate_checkpoint_manager to sync params
        self.colocate_checkpoint_manager = None

    def _setup_checkpoint_manager(self, rollouter):
        """Setup checkpoint manager after rollouter is initialized"""
        replicas = ray.get(rollouter.get_replicas.remote())
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, trainer=self.actor_wg, replicas=replicas
        )
        print("[FullyAsyncTrainer] Checkpoint manager initialized")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    def set_rollouter(self, rollouter):
        """Set rollouter reference for parameter synchronization"""
        self.rollouter = rollouter
        # Setup checkpoint manager after rollouter is set
        self._setup_checkpoint_manager(rollouter)

    def set_total_train_steps(self, total_training_steps):
        self.total_train_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    async def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """Get samples from message queue and compose gen_batch_output.

        Rejection sampling (if enabled) is now done on the rollouter side: it
        drops solve_all / solve_none samples before they ever reach the queue
        and maintains raw reward info for unbiased raw_acc. The trainer
        therefore only has to:
          1. pull RolloutSamples until ``_enough()`` is satisfied
          2. assemble them into a single DataProto
          3. pad the final sequence count up to a multiple of ppo_mini_batch_size
             (response_mask of pad rows is zeroed so they contribute no gradient)
          4. query ``rollouter.consume_rejection_stats()`` and attach the info
             to batch.meta_info so downstream raw_acc metrics still work.

        Returns:
            tuple: (epoch, batch) where ``batch`` is the assembled (and padded)
            DataProto, or (None, None) on termination.
        """
        # ------------------------------------------------------------------
        # Two-phase sample collection via MessageQueue.get_batch():
        #   Phase 1 (wait): block until pending samples >= target (or prompts >= target)
        #   Phase 2 (take): pop a subset that is close to but STRICTLY < target
        #                   (remaining items stay in queue for the next step)
        # After taking, we pad up to ppo_mini_batch_size divisor downstream.
        # ------------------------------------------------------------------
        _fixed_samples_mode = self.target_samples is not None
        if _fixed_samples_mode:
            _target = self.target_samples
            _target_label = "target_samples"
        else:
            # fixed_prompt mode: target is PROMPT-level (not sample-level)
            _target = self.required_prompts
            _target_label = "target_prompts"

        print(
            f"[FullyAsyncTrainer] Requesting samples from queue "
            f"(mode={'fixed_samples' if _fixed_samples_mode else 'fixed_prompt'}, "
            f"{_target_label}={_target})"
            f"{' (rejection_sampling=ON, filtering on rollouter)' if self.rejection_sampling_enabled else ''}",
            flush=True,
        )

        consumer_start = time.time()

        # Call get_batch: waits until enough, then takes subset < target
        if _fixed_samples_mode:
            result = await self.message_queue_client.get_batch(target_samples=_target)
        else:
            result = await self.message_queue_client.get_batch(target_prompts=_target)

        consumer_end = time.time()
        total_wait_time = consumer_end - consumer_start

        if result is None:
            print("[FullyAsyncTrainer] no samples collected (queue shut down), terminating")
            return None, None

        items = result["items"]
        collected_sample_count = result["total_samples"]
        collected_prompt_count = result["total_prompts"]
        remaining_queue = result["remaining_queue_size"]
        remaining_pending = result["remaining_pending_samples"]
        terminated = result["terminated"]

        if not items:
            print("[FullyAsyncTrainer] no samples collected, terminating")
            return None, None

        # Extract raw sample data from items (each item is (data, n_samples))
        queue_samples = [item[0] for item in items]

        if terminated:
            print(
                f"[FullyAsyncTrainer] Rollouter terminated with partial batch: "
                f"{collected_prompt_count} prompts / {collected_sample_count} samples "
                f"(target={_target}). Training on available samples then stopping."
            )

        print(
            f"[FullyAsyncTrainer] Batch collection completed: "
            f"{collected_prompt_count} prompts / {collected_sample_count} samples "
            f"({_target_label}={_target}, mode={'fixed_samples' if _fixed_samples_mode else 'fixed_prompt'}), "
            f"wait_time: {total_wait_time:.2f}s, "
            f"remaining: queue={remaining_queue} prompts, pending={remaining_pending} samples"
        )

        # Deserialize samples (rollouter always pushes pickled RolloutSample).
        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]

        # Assemble batch - now working directly with RolloutSample objects
        _actor_world_size = int(getattr(self.actor_wg, "world_size", 1) or 1)
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(
                queue_samples, self.tokenizer, self.config, self._balance_batch, actor_world_size=_actor_world_size
            )
        else:
            batch = assemble_batch_from_rollout_samples(
                queue_samples, self.tokenizer, self.config, None, actor_world_size=_actor_world_size
            )

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time

        # ------------------------------------------------------------------
        # Fetch rejection-sampling stats + reward_info accumulated on the rollouter
        # since the last training step. This keeps raw_acc / unweighted_acc honest
        # even though the trainer never sees the rejected samples.
        # ------------------------------------------------------------------

        if self.rejection_sampling_enabled:
            try:
                rej_info = await asyncio.wrap_future(
                    self.rollouter.consume_rejection_stats.remote().future()
                )
            except Exception as e:
                print(f"[FullyAsyncTrainer] consume_rejection_stats failed: {e}")
                rej_info = None

            if rej_info:
                stats = rej_info.get("stats") or {}
                reward_info = rej_info.get("reward_info") or []

                # --- Prompt-level rejection stats ---
                total_prompts_seen = int(stats.get("total_prompts_seen", 0))
                if total_prompts_seen > 0:
                    accepted_prompts = int(stats.get("accepted_prompts", 0))
                    rej_all_prompts = int(stats.get("rejected_solve_all_prompts", 0))
                    rej_none_prompts = int(stats.get("rejected_solve_none_prompts", 0))
                    batch.meta_info["fully_async/rejection_sampling/prompt/total_seen"] = total_prompts_seen
                    batch.meta_info["fully_async/rejection_sampling/prompt/accepted"] = accepted_prompts
                    batch.meta_info["fully_async/rejection_sampling/prompt/rejected_solve_all"] = rej_all_prompts
                    batch.meta_info["fully_async/rejection_sampling/prompt/rejected_solve_none"] = rej_none_prompts
                    batch.meta_info["fully_async/rejection_sampling/prompt/accept_ratio"] = accepted_prompts / total_prompts_seen
                    rej_total_prompts = rej_all_prompts + rej_none_prompts
                    denom_prompts = rej_total_prompts + accepted_prompts
                    if denom_prompts > 0:
                        batch.meta_info["fully_async/rejection_sampling/prompt/solve_all_ratio"] = rej_all_prompts / denom_prompts
                        batch.meta_info["fully_async/rejection_sampling/prompt/solve_none_ratio"] = rej_none_prompts / denom_prompts

                # --- Sample-level rejection stats ---
                total_samples_seen = int(stats.get("total_samples_seen", 0))
                if total_samples_seen > 0:
                    accepted_samples = int(stats.get("accepted_samples", 0))
                    rej_all_samples = int(stats.get("rejected_solve_all_samples", 0))
                    rej_none_samples = int(stats.get("rejected_solve_none_samples", 0))
                    batch.meta_info["fully_async/rejection_sampling/sample/total_seen"] = total_samples_seen
                    batch.meta_info["fully_async/rejection_sampling/sample/accepted"] = accepted_samples
                    batch.meta_info["fully_async/rejection_sampling/sample/rejected_solve_all"] = rej_all_samples
                    batch.meta_info["fully_async/rejection_sampling/sample/rejected_solve_none"] = rej_none_samples
                    batch.meta_info["fully_async/rejection_sampling/sample/accept_ratio"] = accepted_samples / total_samples_seen
                    rej_total_samples = rej_all_samples + rej_none_samples
                    denom_samples = rej_total_samples + accepted_samples
                    if denom_samples > 0:
                        batch.meta_info["fully_async/rejection_sampling/sample/solve_all_ratio"] = rej_all_samples / denom_samples
                        batch.meta_info["fully_async/rejection_sampling/sample/solve_none_ratio"] = rej_none_samples / denom_samples

                if reward_info:
                    batch.meta_info["rejection_sampling_reward_info"] = reward_info

        return 0, batch

    def _create_actor_rollout_classes(self):
        # create actor
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        self._init_reward_loop()
        await self._init_async_rollout_manager()

    def _init_reward_loop(self):
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] Init reward loop")
            super()._init_reward_loop()

    async def _init_async_rollout_manager(self):
        # use async rollout do validate
        print(f"[FullyAsyncTrainer] use_trainer_do_validate: {self.config.async_training.use_trainer_do_validate}")
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] Init async rollout manager")

            # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
            # agent_reward_loop: streaming reward computation with actor rollout
            # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
            enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

            # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
            # to stream reward computation with actor rollout
            reward_loop_worker_handles = (
                self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
            )

            # create async rollout manager and request scheduler
            assert self.config.actor_rollout_ref.rollout.mode == "async"

            self.async_rollout_mode = True
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_manager = await AgentLoopManager.create(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                reward_loop_worker_handles=reward_loop_worker_handles,
            )
            print("[FullyAsyncTrainer] async_rollout_manager initialized")

            # Modify checkpoint_engine config to use naive backend
            checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
            original_backend = checkpoint_engine_cfg.backend
            with open_dict(checkpoint_engine_cfg):
                checkpoint_engine_cfg.backend = "naive"
            checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

            print(f"[FullyAsyncTrainer] checkpoint_engine_config: {checkpoint_engine_config}")

            self.colocate_checkpoint_manager = CheckpointEngineManager(
                config=checkpoint_engine_config,
                trainer=self.actor_rollout_wg,
                replicas=self.async_rollout_manager.rollout_replicas,
            )

            # sleep all replicas to load checkpoint
            # In val_only mode, skip sleep_replicas() because:
            # - wake_up_replicas() is not supported in HYBRID rollout mode
            # - update_weights() fails because NCCL checkpoint engine is not initialized
            # So we keep replicas awake and skip the sleep/wake_up cycle entirely.
            if not self.config.trainer.get("val_only", False):
                await self.colocate_checkpoint_manager.sleep_replicas()
            else:
                print("[FullyAsyncTrainer] val_only mode: skip sleep_replicas (replicas stay awake)")
                # In val_only mode, verify that load_format is NOT 'dummy'.
                # When rollout_mode=HYBRID, vLLM keeps load_format=dummy (expecting
                # weights to be synced via update_weights from FSDP). But in val_only
                # mode, update_weights is skipped, so dummy weights would produce
                # garbage output. The fix is to set load_format=auto in the shell
                # script or config BEFORE launching. Here we just warn loudly.
                load_fmt = self.config.actor_rollout_ref.rollout.get("load_format", "dummy")
                if load_fmt == "dummy":
                    print(
                        "[FullyAsyncTrainer] ⚠️  WARNING: val_only mode with load_format='dummy' detected! "
                        "vLLM will use random weights and produce garbage output. "
                        "Set actor_rollout_ref.rollout.load_format=auto in your config/shell script."
                    )

            # Restore original backend value
            with open_dict(checkpoint_engine_cfg):
                checkpoint_engine_cfg.backend = original_backend

            print("[FullyAsyncTrainer] colocate_checkpoint_manager initialized")

        else:
            print("[FullyAsyncTrainer] Skip async rollout manager (use_trainer_do_validate=False)")

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        self.max_steps_duration = 0

        self.global_steps += 1

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[FullyAsyncTrainer] Training stopped by queue termination signal")
                break

        self.progress_bar.close()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)

    async def fit_step(self, batch_dict: dict = None):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_dict: Raw data dictionary
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate(None)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()
            await self._fit_update_weights()
            self._fit_dump_data(batch)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_torch_memory()
        self._fit_postprocess_step()

    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
            # NOTE: filter_zero_adv — zero out response_mask for GRPO groups with
            # zero intra-group reward variance (⇒ adv==0 for all their tokens),
            # BEFORE log_prob / actor forward. This makes loss mask denominators
            # honest and removes dilution of the learning signal in backward,
            # at the cost of still running forward for the masked rows (no drop).
            if self.config.algorithm.adv_estimator == "grpo_filter":
                with marked_timer("filter_zero_adv", timing_raw, color="grey"):
                    filter_metrics = mask_out_zero_variance_groups(batch)
                    metrics.update(filter_metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    def _fit_compute_advantage(self, batch) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw
        reward_tensor = self.reward_tensor
        reward_extra_infos_dict = self.reward_extra_infos_dict

        with marked_timer("adv", timing_raw, color="brown"):
            # we combine with rule-based rm
            reward_extra_infos_dict: dict[str, list]
            batch.batch["token_level_scores"] = reward_tensor

            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            # compute rewards. apply_kl_penalty if available
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # Compute rollout correction: IS weights, rejection sampling, and metrics
            # Only runs in decoupled mode (computes once per batch using stable π_old)
            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
            if (
                rollout_corr_config is not None
                and "rollout_log_probs" in batch.batch
                and not bypass_recomputing_logprobs  # Only in decoupled mode
            ):
                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                # Compute IS weights, apply rejection sampling, compute metrics
                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                # IS and off-policy metrics already have rollout_corr/ prefix
                metrics.update(is_metrics)

            # compute advantages, executed on the driver process
            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            )  # GRPO adv normalization factor

            if self.config.algorithm.adv_estimator == "grpo_filter":
                batch = compute_grpo_advantage_exclude_pad(batch, norm_adv_by_std_in_grpo)
            else:
                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )
        return batch

    def _compute_old_log_prob(self, batch: DataProto):
        """
        If algorithm.rollout_correction.bypass_mode is False,
        use model engine and first version model params to re-calculate old_log_prob.

        If local_trigger_step == 1, load the training engine's parameters to the CPU
          and save a copy for subsequent MIS use.

        If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
        then restore the parameters of the current version.
        """
        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _fit_collect_metrics(self, batch):
        """Override parent to add batch-level training signal metrics.

        In addition to the standard metrics (data, timing, throughput, variance),
        this computes:
        - Difficulty distribution (from prompt buffer pass_rates)
        - Non-zero-advantage prompt & sample statistics
        - Response length statistics split by correct / incorrect
        - Overlong ratio per correct / incorrect
        """
        super()._fit_collect_metrics(batch)
        self.metrics.update(compute_batch_training_signal_metrics(batch))

    def _fit_dump_data(self, batch: DataProto):
        """Override parent to inject prompt buffer tracking info into rollout dump.

        Adds prompt_uid, pass_rate, sample_count, and sampling_prob as extra
        columns in the JSONL output so that priority sampling behavior can be
        tracked offline.

        The per-prompt lists (length = num_prompts) are expanded to per-response
        lists (length = num_responses) using ``rollout_ns`` from ``meta_info``
        which records how many responses each prompt generated.
        """
        meta = batch.meta_info
        n_responses = len(batch)  # total number of responses in the batch

        # Expand per-prompt tracking info to per-response level using rollout_ns.
        # rollout_ns[i] tells how many responses prompt i generated.
        per_prompt_keys = ("prompt_uids", "pass_rates", "sample_counts", "sampling_probs")
        rollout_ns = meta.get("rollout_ns", None)

        if rollout_ns is not None and isinstance(rollout_ns, list):
            for key in per_prompt_keys:
                if key in meta and isinstance(meta[key], list):
                    per_prompt_list = meta[key]
                    expanded = []
                    for val, rn in zip(per_prompt_list, rollout_ns):
                        expanded.extend([val] * rn)
                    if len(expanded) == n_responses:
                        self.reward_extra_infos_dict[key] = expanded

        # Inject ground_truth (gts) into reward_extra_infos_dict.
        # In fully async flow, ground_truth is preserved from the original dataset
        # batch into non_tensor_batch["reward_model"]["ground_truth"] by the rollouter.
        gts = []
        for item in batch:
            rm = item.non_tensor_batch.get("reward_model", None)
            gt = None
            if rm is not None and isinstance(rm, dict):
                gt = rm.get("ground_truth", None)
            gts.append(gt)
        if any(gt is not None for gt in gts):
            self.reward_extra_infos_dict["gts"] = gts

        # Delegate to parent which calls _log_rollout_data -> _dump_generations
        super()._fit_dump_data(batch)

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Override parent to handle numpy types that are not JSON serializable."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        def _json_default(obj):
            """Handle numpy types that are not JSON serializable."""
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False, default=_json_default))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _fit_update_local_step(self):
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        if self.local_trigger_step < self.trigger_parameter_sync_step:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return

        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

        # Reset staleness in rollouter
        timing_raw = ray.get(self.rollouter.reset_staleness.remote(
            current_param_version=self.current_param_version
        ))
        self.logger.log(
            data=timing_raw,
            step=self.current_param_version,
        )

        # Purge stale samples from the queue to guarantee that the NEXT
        # training step only sees samples with staleness ≤ S.
        # An item with start_version V has staleness = current_param_version - V.
        # We evict items where staleness >= staleness_threshold so that any
        # surviving item satisfies staleness < S, i.e. staleness ≤ S - 1.
        # Since current_param_version does NOT change until the next
        # _fit_update_local_step, the next consumption sees the same version,
        # hence consumed staleness ≤ S - 1 < S. ✓
        if self.staleness_threshold > 0:
            purge_result = ray.get(
                self.message_queue_client.queue_actor.purge_stale_samples.remote(
                    current_param_version=self.current_param_version,
                    staleness_threshold=self.staleness_threshold,
                )
            )
            if purge_result["purged_prompts"] > 0:
                self.logger.log(
                    data={
                        "fully_async/purge/purged_prompts": purge_result["purged_prompts"],
                        "fully_async/purge/purged_samples": purge_result["purged_samples"],
                    },
                    step=self.current_param_version,
                )

        # Log aggregated training metrics
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.metrics_aggregator.reset()

    def _validate(self, merged: bool = False):
        """Override parent to inject prompt_uid from extra_info into reward_extra_infos_dict.

        Reads prompt_uid from each sample's extra_info["prompt_uid"] field.
        Falls back to the sample index within the batch if not available.
        The uid is stored in reward_extra_infos_dict["prompt_uid"] so that
        _dump_generations writes it into the JSONL output for downstream
        grouping and analysis.
        """
        from collections import defaultdict

        from verl.trainer.ppo.reward import extract_reward

        # Record validation time as timing_s/test
        with marked_timer("test", self.timing_raw):
            data_source_lst = []
            reward_extra_infos_dict: dict[str, list] = defaultdict(list)

            sample_inputs = []
            sample_outputs = []
            sample_gts = []
            sample_scores = []
            sample_turns = []
            sample_uids = []
            prompt_uid_list = []

            global_idx = 0  # running index across all val batches

            # Progress tracking for val_only time estimation
            total_val_batches = len(self.val_dataloader)
            val_start_time = time.time()
            total_prompts = len(self.val_dataset) if hasattr(self, 'val_dataset') else 0
            repeat_times_for_log = self.config.actor_rollout_ref.rollout.val_kwargs.n
            print(
                f"[Validate] Starting validation: {total_val_batches} batches, "
                f"~{total_prompts} prompts x {repeat_times_for_log} samples = ~{total_prompts * repeat_times_for_log} generations"
            )

        for batch_idx, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                import uuid
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # Extract prompt_uid from extra_info before repeat
            batch_size = len(test_batch.batch)
            batch_prompt_uids = []
            for i in range(batch_size):
                extra_info = test_batch.non_tensor_batch.get("extra_info", None)
                prompt_uid = None
                if extra_info is not None and i < len(extra_info):
                    ei = extra_info[i]
                    if isinstance(ei, dict):
                        prompt_uid = ei.get("prompt_uid", None)
                    elif isinstance(ei, str):
                        try:
                            import json as _json
                            ei_dict = _json.loads(ei)
                            prompt_uid = ei_dict.get("prompt_uid", None)
                        except (ValueError, TypeError):
                            pass
                # Fallback: generate content-based MD5 uid from the prompt text.
                # Uses canonical JSON serialization (no extra spaces, sorted keys)
                # to match generate_prompt_uid() in analyze_val_jsonl.py and
                # evaluate_and_filter.py, ensuring consistent hashing regardless
                # of whether orjson is installed.
                if prompt_uid is None:
                    import hashlib
                    prompt_data = test_batch.non_tensor_batch.get("raw_prompt", None)
                    if prompt_data is not None and i < len(prompt_data):
                        raw = prompt_data[i]
                        if isinstance(raw, (list, np.ndarray)):
                            prompt_str = json.dumps(
                                list(raw) if isinstance(raw, np.ndarray) else raw,
                                ensure_ascii=False, separators=(',', ':'), sort_keys=True,
                            )
                        elif isinstance(raw, str):
                            prompt_str = raw
                        else:
                            prompt_str = str(raw)
                        prompt_uid = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()[:16]
                    else:
                        prompt_uid = str(global_idx + i)
                batch_prompt_uids.append(str(prompt_uid))
            global_idx += batch_size

            # repeat test batch
            repeat_times = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)

            # Expand prompt_uids to match repeated batch
            expanded_prompt_uids = []
            for uid in batch_prompt_uids:
                expanded_prompt_uids.extend([uid] * repeat_times)
            prompt_uid_list.extend(expanded_prompt_uids)

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
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            # Progress tracking with ETA
            elapsed = time.time() - val_start_time
            completed = batch_idx + 1
            avg_per_batch = elapsed / completed
            remaining_batches = total_val_batches - completed
            eta_seconds = avg_per_batch * remaining_batches
            eta_min = eta_seconds / 60
            elapsed_min = elapsed / 60
            print(
                f"[Validate] Batch {completed}/{total_val_batches} done | "
                f"Elapsed: {elapsed_min:.1f}min | ETA: {eta_min:.1f}min | "
                f"Avg: {avg_per_batch:.1f}s/batch | Prompts so far: {global_idx}/{total_prompts}"
            )

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

            # collect response_length for each sample
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
            reward_extra_infos_dict["max_response_length"].extend([tensor_response_length] * len(batch_response_lengths))

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        # Final validation summary
        val_total_time = time.time() - val_start_time
        val_total_min = val_total_time / 60
        total_generations = len(sample_scores)
        gen_per_sec = total_generations / val_total_time if val_total_time > 0 else 0
        print(
            f"[Validate] ✅ Validation complete! "
            f"Total time: {val_total_min:.1f}min ({val_total_time:.0f}s) | "
            f"Batches: {total_val_batches} | Prompts: {total_prompts} | "
            f"Generations: {total_generations} | Throughput: {gen_per_sec:.1f} gen/s"
        )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # Inject prompt_uid into reward_extra_infos_dict for JSONL dump
        if prompt_uid_list:
            reward_extra_infos_dict["prompt_uid"] = prompt_uid_list

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
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

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

    async def _validate_process(self):
        """Run trainer-side validation using async rollout manager"""
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] _validate_process")
            from verl.utils.profiler import marked_timer

            is_val_only = self.config.trainer.get("val_only", False)

            if not is_val_only:
                # Wake up rollouter replicas before validation.
                # Always use update_weights() instead of wake_up_replicas() because
                # in HYBRID rollout mode, wake_up() is not supported on vLLM servers.
                # The naive backend's update_weights() correctly handles HYBRID mode
                # by calling trainer.update_weights() which triggers the proper
                # wake_up flow (via update_weights_from_ipc).
                print("[FullyAsyncTrainer] wake up replicas via update_weights")
                await self.colocate_checkpoint_manager.update_weights(global_steps=self.current_param_version)
            else:
                # In val_only mode, replicas were never put to sleep during init,
                # so no need to wake them up. Both wake_up_replicas() and
                # update_weights() would fail:
                # - wake_up_replicas(): HYBRID mode doesn't support wake_up()
                # - update_weights(): NCCL checkpoint engine not initialized (no rank)
                print("[FullyAsyncTrainer] val_only mode: replicas already awake, skip wake up")

            with marked_timer("trainer/validate_time", self.timing_raw):
                train_val_metrics = self._validate(True)

            if not is_val_only:
                # Sleep rollouter replicas to free GPU memory
                print("[FullyAsyncTrainer] sleep replicas after validation")
                await self.colocate_checkpoint_manager.sleep_replicas()
            else:
                print("[FullyAsyncTrainer] val_only mode: skip sleep_replicas")

            print(f"[FullyAsyncTrainer] validate timing: {self.timing_raw['trainer/validate_time']}")

            return train_val_metrics
        else:
            print("[FullyAsyncTrainer] _validate_process without async_rollout_manager")
            return None

    async def _fit_validate(self, val_before_train=False):
        if self.local_trigger_step != 1:
            return

        # Check if validation is needed
        need_validate = (
            self.config.trainer.test_freq > 0
            and self.current_param_version % self.config.trainer.test_freq == 0
            and self.current_param_version > 0
        )
        # Skip validation if not needed and not validation before training
        if not need_validate and not val_before_train:
            return

        # Trigger rollouter validation and get future
        val_future = self.rollouter.do_validate.remote()

        # Run trainer-side validation
        train_val_metrics = await self._validate_process()

        # Wait for rollouter validation result and log
        val_metrics: ValidateMetrics = ray.get(val_future)
        if train_val_metrics:
            # Merge trainer and rollouter validation results
            with marked_timer("timing_s/merge_val", self.timing_raw):
                new_metrics = self._merge_validation_results(train_val_metrics, val_metrics.metrics)
            if new_metrics:
                self.logger.log(data=new_metrics, step=self.current_param_version)
                pprint(
                    f"[FullyAsyncTrainer] parameter version: {self.current_param_version} "
                    f"Validation metrics: {new_metrics}, timing: {self.timing_raw['timing_s/merge_val']}"
                )
        else:
            if val_metrics.metrics:
                self.logger.log(data=val_metrics.metrics, step=self.current_param_version)
                pprint(
                    f"[FullyAsyncTrainer] parameter version: {self.current_param_version} "
                    f"Validation metrics: {val_metrics.metrics}"
                )
        self.logger.log(data=val_metrics.timing_raw, step=self.current_param_version)

    def _fit_save_checkpoint(self, force=False):
        if self.current_param_version == self.last_ckpt_version:
            return

        timing_raw = self.timing_raw
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. It's the last training step (force=True).
        # 3. The current step number is a multiple of the save frequency.
        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (
            self.current_param_version % self.config.trainer.save_freq == 0
            or force
            or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _fit_postprocess_step(self):
        self.global_steps += 1

        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )

        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    async def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        if self.colocate_checkpoint_manager:
            await self.colocate_checkpoint_manager.update_weights(self.current_param_version)
            await self.colocate_checkpoint_manager.sleep_replicas()

        return self.current_param_version

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value

            # --- Policy staleness metrics ---
            # Definition (per user): sequence staleness = current_param_version
            # (the policy version being trained right now) minus the policy
            # version at which the sequence started rolling out (min_global_steps).
            # Prompt staleness aggregates per-response staleness within a prompt
            # via ``max`` (default) or ``mean``; we then report mean/max across
            # all prompts in the mini-batch and across all responses.
            prompt_agg = getattr(self.config.actor_rollout_ref.rollout, "staleness_prompt_aggregation", "max")
            staleness_metrics = compute_staleness_metrics(
                batch,
                current_param_version=self.current_param_version,
                prompt_aggregation=prompt_agg,
            )
            if staleness_metrics:
                metrics.update(staleness_metrics)


