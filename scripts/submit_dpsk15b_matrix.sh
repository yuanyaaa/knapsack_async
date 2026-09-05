#!/usr/bin/env bash
# DPSK 1.5B rollout-budget experiment matrix, six arms over one shared configuration.
#
#   usage: bash scripts/submit_dpsk15b_matrix.sh <arm> [<arm> ...]
#          bash scripts/submit_dpsk15b_matrix.sh all
#
# Arms:
#   baseline               shuffle_sequential, fixed n=8
#   haspos1                shuffle_sequential, stop once the group has 1 positive
#   rej_fixedn8            baseline + rejection sampling
#   adaptn14               shuffle_sequential, adaptive n 4..14 + compensation
#   lowermid_adaptn14      lower_mid, adaptive n 4..14 + compensation
#   rej_lowermid_adaptn14  lower_mid, adaptive n 4..14 + compensation + rejection sampling
#
# Everything not listed per-arm below is identical across arms, including token-mean loss
# aggregation, so every arm starts from the same base model with the same budget.
set -euo pipefail

RAY_DASHBOARD="${RAY_DASHBOARD:-http://28.85.104.254:8081}"
ROOT_DIR="${ROOT_DIR:-/apdcephfs_hldy/share_303922427/pencywang/project/knapsack_async}"

DATA_ROOT=/taijifs_zw35/r2/ziniuli/processed_dataset
VAL_ROOT="${DATA_ROOT}/deepscaler"

ALL_ARMS=(baseline haspos1 rej_fixedn8 adaptn14 lowermid_adaptn14 rej_lowermid_adaptn14)

# medium_focus peaks at max_n, so the initial allocation may consume the whole cap and leave no
# room for compensation. Under the dataset's starting pass-rate distribution the mean initial n is
# 9.11 against a baseline_n of 8, so the sliding-window budget starts exhausted and compensation is
# inactive. It opens up on its own as training polarises the online pass rates toward 0/1 and
# 4p(1-p) shrinks the allocation back below 8.
read -r -d '' COMPENSATION_EXPORTS <<'EOF' || true
export MEDIUM_FOCUS_PEAK_N=14
export COMPENSATION_RULE=min_rescue_prob
export COMPENSATION_RESCUE_PROB=0.85
export COMPENSATION_ESTIMATE_CLIP=0.02
export COMPENSATION_BUDGET_BASELINE_N=8
export COMPENSATION_BUDGET_WINDOW_SIZE=512
export COMPENSATION_ROI_THRESHOLD=1.0
export ADAPTIVE_CHUNK_PRIOR_TAU=8.0
export DEBUG_ADAPTIVE_N=True
export DEBUG_PRIORITY_SAMPLING=True
export DEBUG_MAX_RECORDS_PER_EVENT=5000
export DEBUG_SAMPLE_EVERY=50
EOF

# total_rollout_steps counts every sequence the rollouter *generates*; groups that rejection
# sampling drops still consume it. At the measured 0.60-0.85 accept ratio a nominal 1500 steps
# would really deliver ~1000, so the nominal budget is oversized to guarantee 1500 real training
# steps. total_epochs has to rise with it, since the effective budget is capped by
# len(train_dataset) * rollout.n * total_epochs.
read -r -d '' REJECTION_BUDGET_EXPORTS <<'EOF' || true
export MAX_TRAINING_STEPS=2600
export TOTAL_EPOCHS=35
EOF

arm_config() {
  # Sets TAG and ARM_EXPORTS for the requested arm.
  case "$1" in
    baseline)
      # Fresh tag rather than resuming dpsk15b_ab_fixedn8_tokenmean_0827_2335: that checkpoint
      # was written under torch 2.10 and its pickled state dict references
      # torch.distributed._mesh_layout, which does not exist in this cluster's torch 2.9.1.
      # Starting from scratch also puts the baseline on the same footing as the other five arms.
      TAG=dpsk15b_base_fixedn8_0830
      ARM_EXPORTS="export SAMPLE_STRATEGY=shuffle_sequential
export ADAPTIVE_ROLLOUT_N_STRATEGY=
export STOP_RULE=fixed_rollout
export STOP_RULE_MIN_N=1
export STOP_RULE_MAX_N=8
export ROLLOUT_CHUNK_SIZE=8
export REJECTION_SAMPLING=False"
      ;;
    haspos1)
      # STOP_RULE_MIN_N is k for has_at_least_positive. The rule has no lower bound on n, so
      # the smallest possible group equals ROLLOUT_CHUNK_SIZE.
      TAG=dpsk15b_haspos1_n14_0830
      ARM_EXPORTS="export SAMPLE_STRATEGY=shuffle_sequential
export ADAPTIVE_ROLLOUT_N_STRATEGY=
export STOP_RULE=has_at_least_positive
export STOP_RULE_MIN_N=1
export STOP_RULE_MAX_N=14
export ROLLOUT_CHUNK_SIZE=2
export REJECTION_SAMPLING=False"
      ;;
    rej_fixedn8)
      TAG=dpsk15b_rej_fixedn8_s1500_0830
      ARM_EXPORTS="export SAMPLE_STRATEGY=shuffle_sequential
export ADAPTIVE_ROLLOUT_N_STRATEGY=
export STOP_RULE=fixed_rollout
export STOP_RULE_MIN_N=1
export STOP_RULE_MAX_N=8
export ROLLOUT_CHUNK_SIZE=8
export REJECTION_SAMPLING=True
${REJECTION_BUDGET_EXPORTS}"
      ;;
    adaptn14)
      TAG=dpsk15b_adaptn14_comp_p14_0830
      ARM_EXPORTS="export SAMPLE_STRATEGY=shuffle_sequential
export ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus
export STOP_RULE=offline_medium_focus_budget_compensate
export STOP_RULE_MIN_N=4
export STOP_RULE_MAX_N=14
export ROLLOUT_CHUNK_SIZE=2
export REJECTION_SAMPLING=False
${COMPENSATION_EXPORTS}"
      ;;
    lowermid_adaptn14)
      TAG=dpsk15b_lowermid_adaptn14_comp_p14_0830
      ARM_EXPORTS="export SAMPLE_STRATEGY=lower_mid
export ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus
export STOP_RULE=offline_medium_focus_budget_compensate
export STOP_RULE_MIN_N=4
export STOP_RULE_MAX_N=14
export ROLLOUT_CHUNK_SIZE=2
export REJECTION_SAMPLING=False
${COMPENSATION_EXPORTS}"
      ;;
    rej_lowermid_adaptn14)
      TAG=dpsk15b_rej_lowermid_adaptn14_p14_s1500_0830
      ARM_EXPORTS="export SAMPLE_STRATEGY=lower_mid
export ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus
export STOP_RULE=offline_medium_focus_budget_compensate
export STOP_RULE_MIN_N=4
export STOP_RULE_MAX_N=14
export ROLLOUT_CHUNK_SIZE=2
export REJECTION_SAMPLING=True
${COMPENSATION_EXPORTS}
${REJECTION_BUDGET_EXPORTS}"
      ;;
    *)
      echo "[ERROR] unknown arm '$1'; valid arms: ${ALL_ARMS[*]} all" >&2
      return 1
      ;;
  esac
}

submit_arm() {
  local arm="$1"
  arm_config "${arm}"
  local submission_id="${SUBMISSION_ID:-codex_${TAG}}"

  cd "${ROOT_DIR}"
  # The bundled ray CLI lets a pre-existing RAY_ADDRESS win over --address, which silently
  # sends the job to whatever cluster the caller's shell last pointed at.
  export RAY_ADDRESS="${RAY_DASHBOARD}"
  ray job submit --address "${RAY_DASHBOARD}" --working-dir . --no-wait \
    --submission-id "${submission_id}" -- bash -lc "set -euo pipefail
unset PYTORCH_CUDA_ALLOC_CONF
export RAY_ADDRESS=auto
export USER_SWANLAB_API_KEY=${USER_SWANLAB_API_KEY:?set_USER_SWANLAB_API_KEY}
export SWANLAB_WORKSPACE=X4
export PROJECT_NAME=wangpengyuan_verl
export TRAINER_LOGGER=\"['console','swanlab']\"
export EXP_MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B
export EXP_VERSION=${TAG}
export TRITON_CACHE_DIR=/tmp/triton_cache_${TAG}

export MODEL_PATH=/taijifs_zw35/r2/ziniuli/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export TRAIN_FILE=${DATA_ROOT}/dapo/train_deepscaler_format_evaluated_dpsk1.5b_n8_trunc32768-v0_clip001_099.parquet
export TEST_FILE='[${VAL_ROOT}/aime24.parquet,${VAL_ROOT}/aime25.parquet,${VAL_ROOT}/amc_fixed.parquet,${VAL_ROOT}/math_100.parquet,${VAL_ROOT}/minerva_fixed_100.parquet,${VAL_ROOT}/olympiad_bench_fixed_100.parquet,${VAL_ROOT}/gpqa_diamond_100.parquet]'
export CKPTS_DIR=/taijifs_zw35/r2/ziniuli/knapsack_async_ckpts_dpsk_1p5b
export RAY_DATA_HOME=${DATA_ROOT}
export REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py

export N_RESP_PER_PROMPT=8
export REWARD_EMA=0.1
export LOSS_AGG_MODE=token-mean

export TRAIN_TRIGGER=fixed_samples
export BATCH_SIZE=128
export TRAIN_PROMPT_MINI_BSZ=128
export STALENESS_THRESHOLD=0

export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=16384
export VAL_MAX_RESPONSE_LENGTH=30720
export ACTOR_PPO_MAX_TOKEN_MULT=2
export INFER_PPO_MAX_TOKEN_MULT=2

export NNODES_ROLLOUT=1
export NNODES_TRAIN=1
export N_GPUS_ROLLOUT=8
export N_GPUS_TRAINING=8
export ROLLOUT_PACK_REPLICAS=True
export NGPUS_PER_NODE=8

export TOTAL_EPOCHS=20
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=50
export SAVE_FREQ=50

# Arm-specific settings come last so they win over the shared defaults above.
# ---- arm: ${arm} ----
${ARM_EXPORTS}

bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh"
}

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <arm> [<arm> ...] | all" >&2
  echo "arms: ${ALL_ARMS[*]}" >&2
  exit 1
fi

if [[ "$1" == "all" ]]; then
  requested=("${ALL_ARMS[@]}")
else
  requested=("$@")
fi

# Validate every arm before submitting any, so a typo cannot leave a half-launched matrix.
for arm in "${requested[@]}"; do
  ( arm_config "${arm}" ) >/dev/null
done

for arm in "${requested[@]}"; do
  echo "=== submitting ${arm} ==="
  submit_arm "${arm}"
done
