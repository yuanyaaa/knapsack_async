#!/usr/bin/env bash
set -xeuo pipefail

# ==================== Experiment ====================
project_name="LZN_DEBUG"
exp_name='DAPO-Qwen2.5-Math-1.5B-MATH-v0'

# ==================== Paths ====================
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"/apdcephfs_gy4/share_302774114/ziniuli/models/Qwen/Qwen2.5-Math-1.5B"}
CKPTS_DIR=${CKPTS_DIR:-"/apdcephfs_gy4/share_302774114/hunyuan/ziniuli/dev/verl-main/exp_hub"}
TRAIN_FILE=${TRAIN_FILE:-"/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/lighteval-MATH/train_evaluated_qwen2.5_math_1.5b_n8_trunc2048-v0.parquet"}
TEST_FILE=${TEST_FILE:-"/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/lighteval-MATH/test_500.parquet"}
REWARD_FN_PATH=${REWARD_FN_PATH:-"/apdcephfs_gy4/share_302774114/hunyuan/ziniuli/dev/verl-main/verl/experimental/fully_async_policy/utils/reward_fn.py"}

# ==================== Rollout Engine ====================
rollout_mode="async"
rollout_name="vllm"  # "vllm" or "sglang"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# ==================== Algorithm (GRPO / KL / Clipping) ====================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# ==================== Sequence Length ====================
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 2))

# ==================== Training ====================
loss_agg_mode="token-mean"

# ==================== Sampling ====================
temperature=1.0
top_p=1.0
top_k=-1  # 0 for HF rollout, -1 for vLLM rollout
val_temperature=0.6
val_n=4

# ==================== Performance ====================
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
ref_offload=True
actor_offload=False
gen_tp=2
sp_size=2
fsdp_size=-1

# ==================== Fully Async ====================
NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-4}
N_GPUS_TRAINING=${N_GPUS_TRAINING:-4}

max_training_steps=1000
train_prompt_bsz=0       # 0 = auto (n_resp_per_prompt * require_batches * train_prompt_mini_bsz)
gen_prompt_bsz=1
n_resp_per_prompt=8
ppo_train_bsz=64
train_prompt_mini_bsz=64
require_batches=$((ppo_train_bsz / train_prompt_mini_bsz))
trigger_parameter_sync_step=1
staleness_threshold=3
# total_rollout_steps is SEQUENCE-level: total number of generated sequences.
# Equivalent to ppo_train_bsz (prompts) * n_resp_per_prompt * max_training_steps.
total_rollout_steps=$(((ppo_train_bsz * n_resp_per_prompt * max_training_steps)))
test_freq=10
partial_rollout=True

# ==================== Priority Sampling ====================
# Strategy: "medium", "hard", "medium_with_exploration", or null (uniform)
priority_strategy="medium_with_exploration"
# EMA coefficient for pass-rate (0.0 = overwrite, higher = more smoothing)
reward_ema=0.5
# Adaptive rollout_n: set a non-empty strategy name to enable the pass_rate -> N
# mapping (only effective when stop_rule=fixed_rollout). Set empty string "" to
# disable and fall back to rollout.n. Strategy options:
#   medium_focus       bell-shaped, peak at pass_rate≈0.5
#   constant_positive  n ≈ base_n / p, keeps E[#positives] constant across difficulty
#   uniform            always return base_n (debug / ablation)
adaptive_rollout_n_strategy=""
# Lower clamp on adaptive rollout_n; upper clamp is ``max_rollout_n`` (yaml default 32).
min_rollout_n=2
# Rejection sampling: discard all-correct / all-incorrect groups
rejection_sampling=True
rejection_queue_timeout=60  # seconds

# ==================== Rollout Correction ====================
# Bypass Mode: old_log_probs = rollout_log_probs, PPO ratio = π_θ / π_rollout
# PPO clip (clip_ratio_low / clip_ratio_high) serves as the safety mechanism
rollout_correction_bypass_mode=True
# Token-level rejection mask: filter tokens where π_θ / π_rollout ∉ [0.5, 2.0]
rollout_correction_rs="token_k1"
rollout_correction_rs_threshold="0.5_2.0"
# rollout_is / rollout_is_threshold: not set → YAML defaults (null)

# ==================== Launch ====================
CKPTS_DIR=${CKPTS_DIR}/$project_name/$exp_name
mkdir -p $CKPTS_DIR

python -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=${rollout_name}  \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    critic.strategy=fsdp2 \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive \
    trainer.logger=['console','swanlab']  \
    trainer.project_name=${project_name} \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${CKPTS_DIR}/rollout_data" \
    trainer.validation_data_dir="${CKPTS_DIR}/validation_data" \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAINING}" \
    trainer.total_epochs=100 \
    trainer.test_freq="${test_freq}" \
    rollout.nnodes="${NNODES_ROLLOUT}"  \
    rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.reward_ema="${reward_ema}" \
    async_training.rollout_config.adaptive_rollout_n_strategy=\"${adaptive_rollout_n_strategy}\" \\
    async_training.rollout_config.min_rollout_n=\"${min_rollout_n}\" \\    async_training.rollout_config.rejection_sampling="${rejection_sampling}" \
    algorithm.rollout_correction.bypass_mode="${rollout_correction_bypass_mode}" \
    algorithm.rollout_correction.rollout_rs="${rollout_correction_rs}" \
    algorithm.rollout_correction.rollout_rs_threshold="${rollout_correction_rs_threshold}" \
    2>&1 | tee "${CKPTS_DIR}/training.log"
    # async_training.priority_strategy="${priority_strategy}" \
