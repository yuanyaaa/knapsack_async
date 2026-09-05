#!/usr/bin/env bash
# Launch one official Qwen3-4B non-thinking arm on an already-started two-node cluster.
# Matches reval_async: run python -m on the Ray head. Do not use ray job submit.
#
# Usage:
#   RAY_ADDRESS=HEAD_IP:PORT bash scripts/submit_qwen3_4b_three_arms_zwfy8.sh \
#     baseline|adaptn14|haspos1|eval_n8
set -euo pipefail

arm="${1:?usage: $0 baseline|adaptn14|haspos1|eval_n8}"
ROOT=/apdcephfs_zwfy8/share_303922427/pencywang/project/knapsack_async
ZW_ROOT=/apdcephfs_zwfy8/share_303922427/pencywang

MODEL_PATH="${ZW_ROOT}/models/Qwen3-4B"
TRAIN_FILE="${ZW_ROOT}/data/qwen3_4b_nothinking/dapo-math-17k-deepscaler_format_evaluated_qwen3_4b_nothinking_n8_trunc32768-v0_clip001_099.parquet"
VAL_ROOT="${ZW_ROOT}/project/data_parquet/deepscaler"
CKPTS_ROOT="${ZW_ROOT}/ckpts/knapsack_async_qwen3_4b"
DATE_TAG="${DATE_TAG:-$(date +%m%d_%H%M)}"

if [[ "${arm}" == "eval_n8" ]]; then
  TAG="q3_4b_thinking_official_dapo17k_n8_32k_${DATE_TAG}"
  TRAIN_FILE="${ZW_ROOT}/project/data_parquet/dapo/train.parquet"
  EVAL_ROOT="${ZW_ROOT}/data/qwen3_4b_thinking"

  for path in "${MODEL_PATH}/config.json" "${TRAIN_FILE}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
  done

  mkdir -p "${EVAL_ROOT}/logs"
  cd "${ROOT}"
  export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
  export TRITON_CACHE_DIR="/tmp/triton_cache_${TAG}"
  export MODEL_PATH MODEL_NAME=qwen3_4b_thinking TRAIN_FILE
  export REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py
  export REWARD_FN_NAME=compute_score
  export NNODES=2 NGPUS_PER_NODE=8 TP_SIZE=2 GPU_MEM_UTIL=0.85
  export N_SAMPLES=8 MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=32768
  export TRUNCATION_LENGTH=32768 TEMPERATURE=1.0 TOP_P=1.0 TOP_K=-1
  export ENABLE_THINKING=True
  export SAFE_LOW_BOUND=0.01 SAFE_UPPER_BOUND=0.99
  export FILTER_PASS_RATE_MIN="" FILTER_PASS_RATE_MAX="" FILTER_OVERLONG_MAX=""
  export VAL_DATA_DIR="${EVAL_ROOT}/${TAG}_validation_data"
  export LIGHTWEIGHT_JSON="${EVAL_ROOT}/${TAG}_samples.jsonl"
  export OUTPUT_PARQUET="${EVAL_ROOT}/${TAG}_full.parquet"
  export LOG_DIR="${EVAL_ROOT}/logs"
  export EXTRACT_LIGHTWEIGHT=true

  echo "launching ${TAG} on 2x8 GPUs with RAY_ADDRESS=${RAY_ADDRESS}"
  echo "output parquet: ${OUTPUT_PARQUET}"
  exec bash verl/experimental/fully_async_policy/shell/run_evaluate_and_filter.sh
fi

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

case "${arm}" in
  baseline)
    TAG="q3_4b_nothink_base_fixedn8_${DATE_TAG}"
    ARM_EXPORTS="export SAMPLE_STRATEGY=shuffle_sequential
export ADAPTIVE_ROLLOUT_N_STRATEGY=
export STOP_RULE=fixed_rollout
export STOP_RULE_MIN_N=1
export STOP_RULE_MAX_N=8
export ROLLOUT_CHUNK_SIZE=8
export REJECTION_SAMPLING=False"
    ;;
  adaptn14)
    TAG="q3_4b_nothink_adaptn14_comp_${DATE_TAG}"
    ARM_EXPORTS="export SAMPLE_STRATEGY=shuffle_sequential
export ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus
export STOP_RULE=offline_medium_focus_budget_compensate
export STOP_RULE_MIN_N=4
export STOP_RULE_MAX_N=14
export ROLLOUT_CHUNK_SIZE=2
export REJECTION_SAMPLING=False
${COMPENSATION_EXPORTS}"
    ;;
  haspos1)
    TAG="q3_4b_nothink_haspos1_n14_${DATE_TAG}"
    ARM_EXPORTS="export SAMPLE_STRATEGY=shuffle_sequential
export ADAPTIVE_ROLLOUT_N_STRATEGY=
export STOP_RULE=has_at_least_positive
export STOP_RULE_MIN_N=1
export STOP_RULE_MAX_N=14
export ROLLOUT_CHUNK_SIZE=2
export REJECTION_SAMPLING=False"
    ;;
  *)
    echo "unknown arm '${arm}'; expected baseline, adaptn14, haspos1, or eval_n8" >&2
    exit 2
    ;;
esac

for path in \
  "${MODEL_PATH}/config.json" \
  "${TRAIN_FILE}" \
  "${VAL_ROOT}/aime24.parquet" \
  "${VAL_ROOT}/aime25.parquet"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

cd "${ROOT}"
eval "${ARM_EXPORTS}"
: "${USER_SWANLAB_API_KEY:?set USER_SWANLAB_API_KEY before launching}"
export USER_SWANLAB_API_KEY

export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-X4}"
export SWANLAB_HOST="${SWANLAB_HOST:-https://train-exp.taiji.woa.com}"
export SWANLAB_API_HOST="${SWANLAB_API_HOST:-https://train-exp.taiji.woa.com}"
export SWANLAB_WEB_HOST="${SWANLAB_WEB_HOST:-https://train-exp.taiji.woa.com}"
export PROJECT_NAME="${PROJECT_NAME:-wangpengyuan_verl}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-['console','swanlab']}"
export EXP_MODEL_NAME=Qwen3-4B
export EXP_VERSION="${TAG}"
export TRITON_CACHE_DIR="/tmp/triton_cache_${TAG}"

export MODEL_PATH
export TRAIN_FILE
export TEST_FILE="[${VAL_ROOT}/aime24.parquet,${VAL_ROOT}/aime25.parquet,${VAL_ROOT}/amc_fixed.parquet,${VAL_ROOT}/math_100.parquet,${VAL_ROOT}/minerva_fixed_100.parquet,${VAL_ROOT}/olympiad_bench_fixed_100.parquet,${VAL_ROOT}/gpqa_diamond_100.parquet]"
export CKPTS_DIR="${CKPTS_ROOT}"
export RAY_DATA_HOME="${ZW_ROOT}/project/data_parquet"
export REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py
export ENABLE_THINKING=False

export N_RESP_PER_PROMPT=8
export REWARD_EMA=0.1
export LOSS_AGG_MODE=token-mean
export TRAIN_TRIGGER=fixed_samples
export BATCH_SIZE=128
export TRAIN_PROMPT_MINI_BSZ=128
export STALENESS_THRESHOLD=0

export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=16384
export VAL_MAX_RESPONSE_LENGTH=16384
export ACTOR_PPO_MAX_TOKEN_MULT=2
export INFER_PPO_MAX_TOKEN_MULT=2
export SP_SIZE=2

export NNODES_ROLLOUT=1
export NNODES_TRAIN=1
export N_GPUS_ROLLOUT=8
export N_GPUS_TRAINING=8
export ROLLOUT_PACK_REPLICAS=True
export NGPUS_PER_NODE=8

export MAX_TRAINING_STEPS=1500
export TOTAL_EPOCHS=14
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=50
export SAVE_FREQ=50

echo "launching ${TAG} with RAY_ADDRESS=${RAY_ADDRESS}"
exec bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh
