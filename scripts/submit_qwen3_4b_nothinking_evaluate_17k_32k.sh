#!/usr/bin/env bash
# Evaluate the full DAPO-Math-17k set with official Qwen/Qwen3-4B in non-thinking mode.
# The output is the matched offline pass-rate prior for 32K-response RL experiments.
#
# Keep raw-pass-rate-zero prompts with a 0.01 prior, drop all-correct prompts, and match the
# legacy v0 overlong policy by allowing at most one of eight responses to hit the 32K limit.
set -euo pipefail

RAY_DASHBOARD="${RAY_DASHBOARD:-http://28.85.104.254:8081}"
ROOT_DIR="${ROOT_DIR:-/apdcephfs_hldy/share_303922427/pencywang/project/knapsack_async}"

DATA_ROOT=/taijifs_zw35/r2/ziniuli/processed_dataset
MODEL_ROOT=/taijifs_zw35/r2/ziniuli/models

TAG="${TAG:-q3_4b_nothink_eval17k_32k_$(date +%m%d_%H%M)}"
SUBMISSION_ID="${SUBMISSION_ID:-codex_${TAG}}"

# val_only creates only the trainer resource pool. Twelve 8-GPU nodes use the 96 GPUs released
# by the stopped 2507 matrix; TP=2 gives 48 independent generation replicas.
NNODES="${NNODES:-12}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-2}"

PREFIX="${DATA_ROOT}/dapo/dapo-math-17k-deepscaler_format_evaluated_qwen3_4b_nothinking_n8"

cd "${ROOT_DIR}"
export RAY_ADDRESS="${RAY_DASHBOARD}"

ray job submit --address "${RAY_DASHBOARD}" --working-dir . --no-wait \
  --submission-id "${SUBMISSION_ID}" -- bash -lc "set -euo pipefail
unset PYTORCH_CUDA_ALLOC_CONF
export RAY_ADDRESS=auto
export TRITON_CACHE_DIR=/tmp/triton_cache_${TAG}

export MODEL_PATH=${MODEL_ROOT}/Qwen/Qwen3-4B
export MODEL_NAME=qwen3_4b_nothinking
export TRAIN_FILE=${DATA_ROOT}/dapo/dapo-math-17k-deepscaler_format.parquet
export REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py
export REWARD_FN_NAME=compute_score
export LOG_DIR=/taijifs_zw35/r2/ziniuli/logs_evaluate_pencywang

export NNODES=${NNODES}
export NGPUS_PER_NODE=${NGPUS_PER_NODE}
export TP_SIZE=${TP_SIZE}
export GPU_MEM_UTIL=0.85

export N_SAMPLES=8
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=32768
export TRUNCATION_LENGTH=32768
export TEMPERATURE=1.0
export TOP_P=1.0
export TOP_K=-1
export ENABLE_THINKING=False

export SAFE_LOW_BOUND=0.01
export SAFE_UPPER_BOUND=0.99
export FILTER_PASS_RATE_MIN=
export FILTER_PASS_RATE_MAX=1
export FILTER_OVERLONG_MAX=0.125

export VAL_DATA_DIR=${PREFIX}_maxlen32768_validation_data
export LIGHTWEIGHT_JSON=${PREFIX}_maxlen32768_lightweight.jsonl
export OUTPUT_PARQUET=${PREFIX}_trunc32768-v0_clip001_099.parquet
export EXTRACT_LIGHTWEIGHT=true

bash verl/experimental/fully_async_policy/shell/run_evaluate_and_filter.sh"

echo "submitted: ${SUBMISSION_ID}"
echo "expected output: ${PREFIX}_trunc32768-v0_clip001_099.parquet"
