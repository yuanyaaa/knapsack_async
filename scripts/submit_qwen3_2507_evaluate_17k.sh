#!/usr/bin/env bash
# Re-evaluate the full DAPO-Math-17k set with Qwen3-4B-Instruct-2507 to produce the offline
# pass_rate prior the adaptive-n arms consume.
#
# Why regenerate instead of reusing train_deepscaler_format_evaluated_qwen3_4b_instruct_2507_*:
#   - those files were already filtered down to 6298 of the original 17398 rows
#   - their prior is clipped to [0.125, 0.875], so never-solved problems get 4p(1-p)=0.4375
#     and medium_focus keeps handing them near-peak n forever
# This run keeps every row except the all-correct ones and clips to [0.01, 0.99].
#
#   usage: bash scripts/submit_qwen3_2507_evaluate_17k.sh
set -euo pipefail

RAY_DASHBOARD="${RAY_DASHBOARD:-http://28.85.104.254:8081}"
ROOT_DIR="${ROOT_DIR:-/apdcephfs_hldy/share_303922427/pencywang/project/knapsack_async}"

DATA_ROOT=/taijifs_zw35/r2/ziniuli/processed_dataset
MODEL_ROOT=/taijifs_zw35/r2/ziniuli/models

TAG="${TAG:-q3_2507_eval17k_$(date +%m%d_%H%M)}"
SUBMISSION_ID="${SUBMISSION_ID:-codex_${TAG}}"

# Sampling has to mirror the training rollout (temperature=1.0, top_p=1.0, top_k=-1 in
# dapo_1.5b_math_fsdp2_4_4.sh), otherwise the offline prior describes a different
# distribution than the one the rollouter will actually draw from.
# val_only only builds the trainer, so the job occupies NNODES * NGPUS_PER_NODE GPUs.
NNODES="${NNODES:-8}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-2}"

cd "${ROOT_DIR}"
# The bundled ray CLI lets a pre-existing RAY_ADDRESS win over --address.
export RAY_ADDRESS="${RAY_DASHBOARD}"

ray job submit --address "${RAY_DASHBOARD}" --working-dir . --no-wait \
  --submission-id "${SUBMISSION_ID}" -- bash -lc "set -euo pipefail
unset PYTORCH_CUDA_ALLOC_CONF
export RAY_ADDRESS=auto
export TRITON_CACHE_DIR=/tmp/triton_cache_${TAG}

export MODEL_PATH=${MODEL_ROOT}/Qwen/Qwen3-4B-Instruct-2507
export MODEL_NAME=qwen3_4b_instruct_2507
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
export MAX_RESPONSE_LENGTH=16384
export TRUNCATION_LENGTH=16384
export TEMPERATURE=1.0
export TOP_P=1.0
export TOP_K=-1

export SAFE_LOW_BOUND=0.01
export SAFE_UPPER_BOUND=0.99

# Keep never-solved problems: at a 0.01 prior they cost the minimum n and still carry signal
# once the policy improves. Drop only the all-correct ones, which can never produce advantage.
export FILTER_PASS_RATE_MIN=
export FILTER_PASS_RATE_MAX=1
export FILTER_OVERLONG_MAX=

export EXTRACT_LIGHTWEIGHT=true

bash verl/experimental/fully_async_policy/shell/run_evaluate_and_filter.sh"

echo "submitted: ${SUBMISSION_ID}"
echo "  输出 parquet 预计: ${DATA_ROOT}/dapo/dapo-math-17k-deepscaler_format_evaluated_qwen3_4b_instruct_2507_n8_trunc16384.parquet"
