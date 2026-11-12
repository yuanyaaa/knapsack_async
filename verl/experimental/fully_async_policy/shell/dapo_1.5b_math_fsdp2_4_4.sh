#!/usr/bin/env bash
set -euo pipefail
# Prompt sample strategy: SAMPLE_STRATEGY overrides Hydra default.
# null|shuffle_sequential = per-epoch shuffled traversal (default); sequential = fixed dataset order;
# uniform = random row; uniform_extreme_ucb = endpoint-penalized uniform + UCB;
# medium_ucb = useful-sample focused 4*p*(1-p) + UCB exploration;
# lower_mid = priority sampling with p*(1-p)^2 weights (peak around p=1/3);
# productive_mixed = prefer prompts likely to produce mixed positive/negative groups;
# medium_sharp|hard|... = pass_rate weights.
sample_strategy="${SAMPLE_STRATEGY:-shuffle_sequential}"
trainer_logger="${TRAINER_LOGGER:-['console','swanlab']}"
reward_ema="${REWARD_EMA:-0.1}"
warmup_epochs="${WARMUP_EPOCHS:-0}"
use_dataset_pass_rate="${USE_DATASET_PASS_RATE:-True}"
uniform_extreme_alpha="${UNIFORM_EXTREME_ALPHA:-0.2}"
uniform_ucb_c="${UNIFORM_UCB_C:-0.0}"
adaptive_rollout_n_strategy="${ADAPTIVE_ROLLOUT_N_STRATEGY:-}"
adaptive_chunk_alpha="${ADAPTIVE_CHUNK_ALPHA:-0.5}"
adaptive_mixed_target_prob="${ADAPTIVE_MIXED_TARGET_PROB:-0.85}"
adaptive_chunk_prior_tau="${ADAPTIVE_CHUNK_PRIOR_TAU:-8.0}"
adaptive_chunk_target_strategy="${ADAPTIVE_CHUNK_TARGET_STRATEGY:-medium_focus_plateau}"
# VIP baseline (ADAPTIVE_ROLLOUT_N_STRATEGY=vip, STOP_RULE=fixed_rollout).
vip_embedding_path="${VIP_EMBEDDING_PATH:-null}"
vip_block_size="${VIP_BLOCK_SIZE:-0}"
vip_budget_per_prompt="${VIP_BUDGET_PER_PROMPT:-0}"
vip_length_scale="${VIP_LENGTH_SCALE:-0.0}"
vip_eps="${VIP_EPS:-1.0e-6}"
vip_variance="${VIP_VARIANCE:-dr_grpo}"
compensation_bank_max="${COMPENSATION_BANK_MAX:-8}"
compensation_chunk_size="${COMPENSATION_CHUNK_SIZE:-1}"
compensation_save_n="${COMPENSATION_SAVE_N:-4}"
compensation_save_low="${COMPENSATION_SAVE_LOW:-0.4}"
compensation_save_high="${COMPENSATION_SAVE_HIGH:-0.6}"
compensation_roi_threshold="${COMPENSATION_ROI_THRESHOLD:-1.5}"
compensation_rule="${COMPENSATION_RULE:-roi_max}"
compensation_rescue_prob="${COMPENSATION_RESCUE_PROB:-0.85}"
compensation_minority_target="${COMPENSATION_MINORITY_TARGET:-1}"
compensation_estimate_clip="${COMPENSATION_ESTIMATE_CLIP:-0.02}"
compensation_budget_baseline_n="${COMPENSATION_BUDGET_BASELINE_N:-${N_RESP_PER_PROMPT:-16}}"
compensation_budget_window_size="${COMPENSATION_BUDGET_WINDOW_SIZE:-0}"
debug_adaptive_n="${DEBUG_ADAPTIVE_N:-False}"
debug_priority_sampling="${DEBUG_PRIORITY_SAMPLING:-False}"
debug_max_records_per_event="${DEBUG_MAX_RECORDS_PER_EVENT:-2000}"
debug_sample_every="${DEBUG_SAMPLE_EVERY:-200}"
stop_rule="${STOP_RULE:-fixed_rollout}"
stop_rule_min_n="${STOP_RULE_MIN_N:-2}"
stop_rule_max_n="${STOP_RULE_MAX_N:-32}"
min_rollout_n_schedule_interval="${MIN_ROLLOUT_N_SCHEDULE_INTERVAL:-0}"
min_rollout_n_schedule_increment="${MIN_ROLLOUT_N_SCHEDULE_INCREMENT:-0}"
rejection_sampling="${REJECTION_SAMPLING:-False}"
rejection_queue_timeout="${REJECTION_QUEUE_TIMEOUT:-60}"
train_trigger="${TRAIN_TRIGGER:-fixed_prompt}"
batch_size="${BATCH_SIZE:-128}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-16}"
pos_neg_ratio="${POS_NEG_RATIO:-}"

# Under the compensate stop rules the initial offline allocation must peak at the
# nominal group size so that [n, max_n] stays available for compensation; the
# plain adaptive rules have nothing to reserve headroom for and peak at max_n.
case "${stop_rule}" in
  offline_medium_focus_compensate|offline_medium_focus_budget_compensate|knapsack_value_compensate)
    medium_focus_peak_n="${MEDIUM_FOCUS_PEAK_N:-${n_resp_per_prompt}}"
    ;;
  *)
    medium_focus_peak_n="${MEDIUM_FOCUS_PEAK_N:-${stop_rule_max_n}}"
    ;;
esac

usage() {
  cat <<'EOF'
Usage: bash dapo_1.5b_math_fsdp2_4_4.sh [-h|--help]

  Fully-async priority sampling experiment launcher. Configure via environment variables.

Environment (optional overrides):
  SAMPLE_STRATEGY          shuffle_sequential (default) | sequential | uniform | uniform_extreme_ucb | medium_ucb | medium_to_productive_mixed | productive_mixed | medium | hard | ...
  REWARD_EMA               EMA coefficient for pass-rate updates (default: 0.1)
  UNIFORM_EXTREME_ALPHA    endpoint downweight for uniform_extreme_ucb, p=0/p=1 (default: 0.2)
  UNIFORM_UCB_C            UCB exploration strength for uniform_extreme_ucb / medium_ucb (default: 0.0)
  REJECTION_SAMPLING      true | false (default: False)
  REJECTION_QUEUE_TIMEOUT queue starvation timeout in seconds (default: 60)
  ADAPTIVE_ROLLOUT_N_STRATEGY medium_focus | inverse_bernoulli_variance | productive_curve | knapsack_value_margin50 | knapsack_value_halfmax | mixed_prob_target | constant_positive | uniform | "" (default: "")
  STOP_RULE              fixed_rollout | prefixed_rollout | has_at_least_positive | has_at_least_positive_and_negative | continue_on_mixed_chunk | adaptive_chunk_mixed | offline_medium_focus_compensate | offline_medium_focus_budget_compensate | knapsack_value_compensate | max_rollout (default: fixed_rollout)
  STOP_RULE_MIN_N        minimum n for stop rule / k for has_at_least_positive (default: 2)
  STOP_RULE_MAX_N        maximum n for stop rule (default: 32)
  MIN_ROLLOUT_N_SCHEDULE_INTERVAL
                           if >0, increase min rollout n every this many trainer steps
  MIN_ROLLOUT_N_SCHEDULE_INCREMENT
                           increment added to min rollout n per schedule stage
  ADAPTIVE_CHUNK_ALPHA   current rollout pass-rate weight for adaptive_chunk_mixed (default: 0.5)
  ADAPTIVE_MIXED_TARGET_PROB target probability of >=1 positive and >=1 negative (default: 0.85)
  MIXED_PROB_TARGET_TAU target P(>=1 positive and >=1 negative) for mixed_prob_target (default: 0.9)
  ADAPTIVE_CHUNK_PRIOR_TAU historical pass-rate virtual sample count for adaptive_chunk_mixed (default: 8.0)
  ADAPTIVE_CHUNK_TARGET_STRATEGY medium_focus | medium_focus_plateau | productive_curve (default: medium_focus_plateau)
  MEDIUM_FOCUS_PEAK_N peak n used by offline medium_focus initial allocation
                         (default: N_RESP_PER_PROMPT under the compensate stop rules so that
                          [n, STOP_RULE_MAX_N] is left for compensation, else STOP_RULE_MAX_N)
  COMPENSATION_BANK_MAX max saved response bank for offline_medium_focus_compensate (default: 8)
  COMPENSATION_CHUNK_SIZE one-shot compensation size (default: 1)
  COMPENSATION_SAVE_N response budget saved from p in [low, high] (default: 4)
  COMPENSATION_SAVE_LOW lower p bound for saving budget (default: 0.4)
  COMPENSATION_SAVE_HIGH upper p bound for saving budget (default: 0.6)
  COMPENSATION_ROI_THRESHOLD compensate only when ROI is larger than this (default: 1.5, roi_max rule only)
  COMPENSATION_RULE      roi_max | min_rescue_prob (default: roi_max)
                         roi_max: largest affordable extra whose ROI clears the threshold.
                         min_rescue_prob: smallest extra reaching COMPENSATION_RESCUE_PROB;
                         prompts that cannot reach it under STOP_RULE_MAX_N get nothing.
  COMPENSATION_RESCUE_PROB target probability of getting the missing class (default: 0.85, min_rescue_prob only)
  COMPENSATION_MINORITY_TARGET required samples of the missing class (default: 1, min_rescue_prob only)
  COMPENSATION_ESTIMATE_CLIP clip for the blended pass-rate estimate (default: 0.02, min_rescue_prob only).
                         min_rescue_prob blends the prior toward the observed all-positive/all-negative
                         rate with weight num_finished/adaptive_chunk_prior_tau; this clip keeps the
                         blend off 0/1 so large homogeneous groups are rejected on capacity grounds
                         instead of on an asserted certainty.
  DEBUG_ADAPTIVE_N       trace initial-n / budget / compensation / stop decisions (default: False)
                         filter with: ray job logs <id> | rg '\[adaptive_n\]'
  DEBUG_PRIORITY_SAMPLING trace prompt draws, pass_rate updates and weight distribution (default: False)
                         filter with: ray job logs <id> | rg '\[priority\]'
  DEBUG_MAX_RECORDS_PER_EVENT full-rate records per event before down-sampling (default: 2000)
  DEBUG_SAMPLE_EVERY     keep 1 of every N records after the full-rate budget (default: 200)
  COMPENSATION_BUDGET_BASELINE_N baseline samples per prompt for budget compensation (default: N_RESP_PER_PROMPT)
  COMPENSATION_BUDGET_WINDOW_SIZE if >0, use recent-window budget for compensation (default: 0)
  POS_NEG_RATIO          optional positive:negative ratio for emitted has-k groups, e.g. 1:7
  TRAIN_TRIGGER          fixed_prompt | fixed_samples (default: fixed_prompt)
  BATCH_SIZE             ppo train batch size (default: 128)
  N_RESP_PER_PROMPT      rollout responses per prompt n (default: 16)
  ADV_ESTIMATOR          grpo | grpo_filter | ... (default: grpo_filter for variable-N rollout, else grpo)
  ACTOR_LR               actor optimizer learning rate (default: 2e-6)
  TRAIN_PROMPT_MINI_BSZ  prompt mini batch size consumed by trainer (default: BATCH_SIZE, on-policy)
  STALENESS_THRESHOLD    max rollout staleness allowed before actor update (default: 3)
  GEN_TP                 rollout vLLM tensor parallel size (default: 1)
  SP_SIZE                actor/ref Ulysses sequence parallel size (default: 1)
  ACTOR_PPO_MAX_TOKEN_MULT max token len per GPU multiplier for actor update (default: 8)
  INFER_PPO_MAX_TOKEN_MULT max token len per GPU multiplier for log-prob/ref (default: 8)
  ACTOR_PPO_MAX_TOKEN_LEN direct override for actor ppo max token len per GPU
  INFER_PPO_MAX_TOKEN_LEN direct override for log-prob/ref max token len per GPU
  ROLLOUT_GPU_MEMORY_UTILIZATION vLLM GPU memory utilization (default: 0.85)
  VLLM_MAX_NUM_BATCHED_TOKENS direct override for vLLM max_num_batched_tokens
  TEST_FREQ              validation frequency in trainer steps (default: 20)
  SAVE_FREQ              checkpoint save frequency in trainer steps (default: 20)
  VAL_BEFORE_TRAIN       run validation before step 1 (default: True)
  VAL_N                  responses per validation prompt (default: 16)
  VAL_BATCH_SIZE         validation prompt batch size (default: 32)
  PERF_TAG               optional suffix added to experiment name
  ROLLOUT_CHUNK_SIZE     sequences per generation request for all rollout stop rules (default: 2)

Examples:
  SAMPLE_STRATEGY=hard REWARD_EMA=0.3 bash dapo_1.5b_math_fsdp2_4_4.sh
  ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus bash dapo_1.5b_math_fsdp2_4_4.sh
  ADAPTIVE_ROLLOUT_N_STRATEGY=productive_curve bash dapo_1.5b_math_fsdp2_4_4.sh
  SAMPLE_STRATEGY=uniform bash dapo_1.5b_math_fsdp2_4_4.sh
  SAMPLE_STRATEGY=uniform_extreme_ucb UNIFORM_EXTREME_ALPHA=0.2 UNIFORM_UCB_C=0.5 bash dapo_1.5b_math_fsdp2_4_4.sh
  SAMPLE_STRATEGY=medium_ucb UNIFORM_UCB_C=0.5 bash dapo_1.5b_math_fsdp2_4_4.sh
  REJECTION_SAMPLING=true REJECTION_QUEUE_TIMEOUT=120 bash dapo_1.5b_math_fsdp2_4_4.sh

trainer.experiment_name includes sample_strategy and uniform_extreme_ucb hyperparameters.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo "error: unexpected arguments (use environment variables): $*" >&2
  usage >&2
  exit 1
fi

# Hydra null must be passed as bare `null` (not quoted string "null").
if [[ -z "${sample_strategy}" || "${sample_strategy,,}" == "null" || "${sample_strategy,,}" == "none" ]]; then
  sample_strategy_hydra=null
else
  sample_strategy_hydra="${sample_strategy}"
fi

# ==================== Experiment ====================
if [[ "${trainer_logger}" == *"swanlab"* && -z "${USER_SWANLAB_API_KEY:-}" && -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "error: USER_SWANLAB_API_KEY or SWANLAB_API_KEY is required" >&2
  exit 1
fi
if [[ -n "${USER_SWANLAB_API_KEY:-}" ]]; then
  export USER_SWANLAB_API_KEY
fi
if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  export SWANLAB_API_KEY
fi

project_name="${PROJECT_NAME:-wangpengyuan_verl}"
exp_model_name="${EXP_MODEL_NAME:-DeepSeek-R1-Distill-Qwen-1.5B}"
stop_rule_display="${stop_rule}"
if [[ "${stop_rule}" == "has_at_least_positive" ]]; then
  if [[ -n "${pos_neg_ratio}" ]]; then
    stop_rule_display="has_posneg${pos_neg_ratio//:/_}_finaln${n_resp_per_prompt}"
  else
    stop_rule_display="has_${stop_rule_min_n}_positive"
  fi
fi
exp_name_base="GRPO-${exp_model_name}-DAPO-raw-4-4-${sample_strategy}-${adaptive_rollout_n_strategy}-${train_trigger}-${stop_rule_display}"
if [[ "${sample_strategy,,}" == "uniform_extreme_ucb" ]]; then
  exp_name_base="${exp_name_base}-alpha${uniform_extreme_alpha}-ucbc${uniform_ucb_c}"
elif [[ "${sample_strategy,,}" == "medium_ucb" ]]; then
  exp_name_base="${exp_name_base}-ucbc${uniform_ucb_c}"
fi
# Log / checkpoint / SwanLab run name: include key rollout policy knobs.
exp_version=${EXP_VERSION:-v2}
exp_name="${exp_name_base}-stop_maxn${stop_rule_max_n}-bs${batch_size}-n${n_resp_per_prompt}-rejection_sampling${rejection_sampling}-${exp_version}"
if [[ -n "${PERF_TAG:-}" ]]; then
  exp_name="${exp_name}-${PERF_TAG}"
fi

# ==================== Paths ====================
RAY_DATA_HOME=${RAY_DATA_HOME:-"/data/chencongliang/wangpengyuan/project/data_parquet"}
MODEL_PATH=${MODEL_PATH:-"/data/chencongliang/wangpengyuan/models/Qwen/Qwen2.5-Math-1.5B"}
CKPTS_DIR=${CKPTS_DIR:-"./ckpts"}
if [[ -z "${TRAIN_FILE:-}" ]]; then
  if [[ "${exp_model_name}" == "Qwen2.5-Math-7B" || "${MODEL_PATH}" == *"Qwen2.5-Math-7B"* ]]; then
    TRAIN_FILE="local_data/train_deepscaler_format_evaluated_qwen2.5_math_7b_n8_trunc2048-v0.pq"
  else
    # TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/dapo/train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw.parquet"}
    TRAIN_FILE="/data/chencongliang/wangpengyuan/project/data_parquet/dapo/train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw_unclipped.parquet"
  fi
fi
TEST_FILE=${TEST_FILE:-"[${RAY_DATA_HOME}/aime.parquet,${RAY_DATA_HOME}/AIME2025.parquet,${RAY_DATA_HOME}/deepscaler/amc_fixed.parquet,${RAY_DATA_HOME}/deepscaler/math_100.parquet,${RAY_DATA_HOME}/deepscaler/minerva_fixed_100.parquet,${RAY_DATA_HOME}/deepscaler/olympiad_bench_fixed_100.parquet,${RAY_DATA_HOME}/deepscaler/gpqa_diamond_100.parquet]"}
REWARD_FN_PATH=${REWARD_FN_PATH:-"verl/experimental/fully_async_policy/utils/reward_fn.py"}

# ==================== Rollout Engine ====================
rollout_mode="async"
rollout_name="vllm"  # "vllm" or "sglang"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# Official Qwen3 hybrid models default to thinking mode. Keep this optional so older
# model families and pure Instruct checkpoints retain their native chat-template behavior.
thinking_override=()
if [[ -n "${ENABLE_THINKING:-}" ]]; then
    thinking_override+=("+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}")
fi

# ==================== Algorithm (GRPO / KL / Clipping) ====================
if [[ -n "${ADV_ESTIMATOR:-}" ]]; then
  adv_estimator="${ADV_ESTIMATOR}"
elif [[ "${stop_rule}" != "fixed_rollout" || "${train_trigger}" == "fixed_samples" ]]; then
  # Variable-N rollout batches are often padded before actor update. grpo_filter
  # excludes pad rows from GRPO group statistics; plain grpo would let padded
  # duplicate rows skew the per-uid mean/std.
  adv_estimator=grpo_filter
else
  adv_estimator=grpo
fi

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# ==================== Sequence Length ====================
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 2))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 2))}
val_max_response_length=${VAL_MAX_RESPONSE_LENGTH:-$((1024 * 2))}
rollout_max_model_len=$((max_prompt_length + max_response_length))
val_max_model_len=$((max_prompt_length + val_max_response_length))
if (( val_max_model_len > rollout_max_model_len )); then
  rollout_max_model_len=${val_max_model_len}
fi

# ==================== Training ====================
# token-mean: denominator is the surviving token count, so the update size is
# invariant to how many samples clear the zero-advantage filter.
# seq-mean-token-mean: denominator is ppo_mini_batch_size * rollout.n (a constant),
# so more surviving sequences means a proportionally larger update, and every
# sequence carries equal weight regardless of its length.
loss_agg_mode=${LOSS_AGG_MODE:-token-mean}
case "${loss_agg_mode}" in
  token-mean|seq-mean-token-mean|seq-mean-token-sum|seq-mean-token-sum-norm) ;;
  *) echo "[ERROR] unsupported LOSS_AGG_MODE='${loss_agg_mode}'" >&2; exit 1 ;;
esac

# ==================== Sampling ====================
temperature=1.0
top_p=1.0
top_k=-1  # 0 for HF rollout, -1 for vLLM rollout
val_temperature=0.6
val_n=${VAL_N:-16}
val_batch_size=${VAL_BATCH_SIZE:-32}
val_before_train=${VAL_BEFORE_TRAIN:-True}

# ==================== Performance ====================
use_dynamic_bsz=True
ref_offload=False
actor_offload=False
gen_tp=${GEN_TP:-1}
sp_size=${SP_SIZE:-1}
fsdp_size=-1

actor_ppo_max_token_mult=${ACTOR_PPO_MAX_TOKEN_MULT:-8}
infer_ppo_max_token_mult=${INFER_PPO_MAX_TOKEN_MULT:-8}
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) * actor_ppo_max_token_mult))}
infer_ppo_max_token_len=${INFER_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) * infer_ppo_max_token_mult))}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
vllm_max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS:-${rollout_max_model_len}}

# ==================== Fully Async ====================
NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-4}
N_GPUS_TRAINING=${N_GPUS_TRAINING:-4}

# Confine the rollout replicas to NNODES_ROLLOUT nodes instead of letting Ray scatter one
# single-GPU placement group per replica across the whole cluster. Leaving them scattered
# fragments every node and blocks a co-running job's single-node trainer group.
rollout_pack_replicas=${ROLLOUT_PACK_REPLICAS:-False}
case "${rollout_pack_replicas}" in
  True|False) ;;
  *) echo "[ERROR] ROLLOUT_PACK_REPLICAS must be True or False, got '${rollout_pack_replicas}'" >&2; exit 1 ;;
esac

max_training_steps=${MAX_TRAINING_STEPS:-1000}
train_prompt_bsz=0       # 0 = auto (n_resp_per_prompt * require_batches * train_prompt_mini_bsz)
gen_prompt_bsz=1
# n_resp_per_prompt=16
ppo_train_bsz=${batch_size}
# On-policy default: one full prompt batch per PPO update.
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-${ppo_train_bsz}}
if (( ppo_train_bsz % train_prompt_mini_bsz != 0 )); then
  echo "error: BATCH_SIZE=${ppo_train_bsz} must be divisible by TRAIN_PROMPT_MINI_BSZ=${train_prompt_mini_bsz}" >&2
  exit 1
fi
require_batches=$((ppo_train_bsz / train_prompt_mini_bsz))
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-1}
staleness_threshold=${STALENESS_THRESHOLD:-3}
rollout_budget_mult=${ROLLOUT_BUDGET_MULT:-1}
total_rollout_steps=$(((ppo_train_bsz * n_resp_per_prompt * max_training_steps * rollout_budget_mult)))
test_freq=${TEST_FREQ:-20}
save_freq=${SAVE_FREQ:-20}
partial_rollout=${PARTIAL_ROLLOUT:-True}

# ==================== Priority Sampling ====================
# Strategy: "uniform_extreme_ucb", "medium_ucb", "medium", "hard", "medium_with_exploration", or null (sequential)
# sample_strategy="medium_with_exploration"
# EMA coefficient for pass-rate (0.0 = overwrite, higher = more smoothing)
# reward_ema is configured from environment variable at the top.
# uniform_extreme_alpha / uniform_ucb_c are configured from environment variables at the top.
# Stop rule for dynamic early stop
# stop_rule / stop_rule_min_n / stop_rule_max_n are configured from env vars at the top.
# Rejection sampling: discard all-correct / all-incorrect groups
rejection_queue_timeout="${REJECTION_QUEUE_TIMEOUT:-60}"  # seconds

# ==================== Rollout Correction ====================
# Bypass Mode: old_log_probs = rollout_log_probs, PPO ratio = π_θ / π_rollout
# PPO clip (clip_ratio_low / clip_ratio_high) serves as the safety mechanism
rollout_correction_bypass_mode=False  # ensure log prob and entropy calculation
# Token-level rejection mask: filter tokens where π_θ / π_rollout ∉ [0.5, 2.0]
rollout_correction_rs="token_k1"
rollout_correction_rs_threshold="0.5_2.0"
# Importance sampling ratio: π_old / π_rollout (in addition to PPO's IS)
rollout_correction_is="token"

# ==================== Launch ====================
CKPTS_DIR=${CKPTS_DIR}/$project_name/$exp_name
mkdir -p $CKPTS_DIR
rollout_data_dir="${ROLLOUT_DATA_DIR:-${CKPTS_DIR}/rollout_data}"
validation_data_dir="${VALIDATION_DATA_DIR:-${CKPTS_DIR}/validation_data}"

python -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.val_batch_size=${val_batch_size} \
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
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-2e-6}" \
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
    actor_rollout_ref.rollout.pack_standalone_replicas=${rollout_pack_replicas} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${vllm_max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.val_kwargs.max_response_length=${val_max_response_length} \
    critic.strategy=fsdp2 \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive \
    trainer.logger="${trainer_logger}"  \
    trainer.project_name=${project_name} \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq="${save_freq}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${rollout_data_dir}" \
    trainer.validation_data_dir="${validation_data_dir}" \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAINING}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-100}" \
    trainer.test_freq="${test_freq}" \
    rollout.nnodes="${NNODES_ROLLOUT}"  \
    rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    +ray_kwargs.ray_init.dashboard_port=8268 \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.sample_strategy=${sample_strategy_hydra} \
    async_training.reward_ema="${reward_ema}" \
    async_training.warmup_epochs="${warmup_epochs}" \
    async_training.use_dataset_pass_rate="${use_dataset_pass_rate}" \
    async_training.uniform_extreme_alpha="${uniform_extreme_alpha}" \
    async_training.uniform_ucb_c="${uniform_ucb_c}" \
    async_training.rollout_config.stop_rule="${stop_rule}" \
    async_training.rollout_config.min_rollout_n="${stop_rule_min_n}" \
    async_training.rollout_config.max_rollout_n="${stop_rule_max_n}" \
    +async_training.rollout_config.min_rollout_n_schedule_interval="${min_rollout_n_schedule_interval}" \
    +async_training.rollout_config.min_rollout_n_schedule_increment="${min_rollout_n_schedule_increment}" \
    async_training.rollout_config.rollout_chunk_size="${ROLLOUT_CHUNK_SIZE:-2}" \
    async_training.rollout_config.positive_negative_ratio="${pos_neg_ratio}" \
    async_training.rollout_config.adaptive_rollout_n_strategy="${adaptive_rollout_n_strategy}" \
    async_training.rollout_config.vip_embedding_path="${vip_embedding_path}" \
    async_training.rollout_config.vip_block_size="${vip_block_size}" \
    async_training.rollout_config.vip_budget_per_prompt="${vip_budget_per_prompt}" \
    async_training.rollout_config.vip_length_scale="${vip_length_scale}" \
    async_training.rollout_config.vip_eps="${vip_eps}" \
    async_training.rollout_config.vip_variance="${vip_variance}" \
    async_training.rollout_config.adaptive_chunk_alpha="${adaptive_chunk_alpha}" \
    async_training.rollout_config.adaptive_mixed_target_prob="${adaptive_mixed_target_prob}" \
    async_training.rollout_config.adaptive_chunk_prior_tau="${adaptive_chunk_prior_tau}" \
    +async_training.rollout_config.adaptive_chunk_target_strategy="${adaptive_chunk_target_strategy}" \
    +async_training.rollout_config.medium_focus_peak_n="${medium_focus_peak_n}" \
    +async_training.rollout_config.compensation_bank_max="${compensation_bank_max}" \
    +async_training.rollout_config.compensation_chunk_size="${compensation_chunk_size}" \
    +async_training.rollout_config.compensation_save_n="${compensation_save_n}" \
    +async_training.rollout_config.compensation_save_low="${compensation_save_low}" \
    +async_training.rollout_config.compensation_save_high="${compensation_save_high}" \
    +async_training.rollout_config.compensation_roi_threshold="${compensation_roi_threshold}" \
    +async_training.rollout_config.compensation_rule="${compensation_rule}" \
    +async_training.rollout_config.compensation_rescue_prob="${compensation_rescue_prob}" \
    +async_training.rollout_config.compensation_minority_target="${compensation_minority_target}" \
    +async_training.rollout_config.compensation_estimate_clip="${compensation_estimate_clip}" \
    +async_training.rollout_config.debug_adaptive_n="${debug_adaptive_n}" \
    +async_training.rollout_config.debug_priority_sampling="${debug_priority_sampling}" \
    +async_training.rollout_config.debug_max_records_per_event="${debug_max_records_per_event}" \
    +async_training.rollout_config.debug_sample_every="${debug_sample_every}" \
    +async_training.rollout_config.compensation_budget_baseline_n="${compensation_budget_baseline_n}" \
    +async_training.rollout_config.compensation_budget_window_size="${compensation_budget_window_size}" \
    async_training.rollout_config.train_trigger="${train_trigger}" \
    async_training.rollout_config.rejection_sampling="${rejection_sampling}" \
    algorithm.rollout_correction.bypass_mode="${rollout_correction_bypass_mode}" \
    algorithm.rollout_correction.rollout_rs="${rollout_correction_rs}" \
    algorithm.rollout_correction.rollout_rs_threshold="${rollout_correction_rs_threshold}" \
    algorithm.rollout_correction.rollout_is="${rollout_correction_is}" \
    "${thinking_override[@]}" \
    2>&1 | tee "${CKPTS_DIR}/training.log"
