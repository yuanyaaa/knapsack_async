#!/usr/bin/env bash
# Reproduce the official DAPO-17K Qwen3-4B thinking n=8, 32K evaluation.
set -euo pipefail

ROOT=/apdcephfs_zwfy8/share_303922427/pencywang/project/knapsack_async
ZW_ROOT=/apdcephfs_zwfy8/share_303922427/pencywang
DATE_TAG="${DATE_TAG:-$(date +%m%d_%H%M)}"
TAG="q3_4b_thinking_official_dapo17k_n8_32k_${DATE_TAG}"

export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export MODEL_PATH="${ZW_ROOT}/models/Qwen3-4B"
export MODEL_NAME=qwen3_4b_thinking
export TRAIN_FILE="${ZW_ROOT}/project/data_parquet/dapo/train.parquet"
export REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py
export REWARD_FN_NAME=compute_score

export NNODES=2
export NGPUS_PER_NODE=8
export TP_SIZE=2
export GPU_MEM_UTIL=0.85
export N_SAMPLES=8
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=32768
export TRUNCATION_LENGTH=32768
export TEMPERATURE=1.0
export TOP_P=1.0
export TOP_K=-1
export ENABLE_THINKING=True

export SAFE_LOW_BOUND=0.01
export SAFE_UPPER_BOUND=0.99
export FILTER_PASS_RATE_MIN=
export FILTER_PASS_RATE_MAX=
export FILTER_OVERLONG_MAX=

EVAL_ROOT="${ZW_ROOT}/data/qwen3_4b_thinking"
export VAL_DATA_DIR="${EVAL_ROOT}/${TAG}_validation_data"
export LIGHTWEIGHT_JSON="${EVAL_ROOT}/${TAG}_samples.jsonl"
export OUTPUT_PARQUET="${EVAL_ROOT}/${TAG}_full.parquet"
export LOG_DIR="${EVAL_ROOT}/logs"
export EXTRACT_LIGHTWEIGHT=true
export TRITON_CACHE_DIR="/tmp/triton_cache_${TAG}"

for path in "${MODEL_PATH}/config.json" "${TRAIN_FILE}"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${LOG_DIR}"
cd "${ROOT}"
echo "launching ${TAG} on 2x8 GPUs with RAY_ADDRESS=${RAY_ADDRESS}"
echo "output parquet: ${OUTPUT_PARQUET}"
exec bash verl/experimental/fully_async_policy/shell/run_evaluate_and_filter.sh
