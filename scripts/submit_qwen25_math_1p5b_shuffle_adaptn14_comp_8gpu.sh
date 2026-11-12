#!/usr/bin/env bash
# Qwen2.5-Math-1.5B aligned to the DPSK adaptn14 compensation arm:
# shuffle_sequential + medium_focus + sliding-window min_rescue_prob,
# N in [4, 14], peak_n=14. Uses the raw-unclipped Qwen2.5-Math N=8 data.
set -euo pipefail

RAY_DASHBOARD="${RAY_DASHBOARD:-http://28.83.35.117:8081}"
ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_VERSION="${EXP_VERSION:-qwen25_math_1p5b_shuffle_adaptn14_comp_p14_raw_unclipped_8gpu_${RUN_TAG}}"
SUBMISSION_ID="${SUBMISSION_ID:-codex_${EXP_VERSION}}"
STAGE="${STAGE:-/tmp/knapsack_async_qwen25_adaptn14_${RUN_TAG}}"

SWANLAB_KEY="${USER_SWANLAB_API_KEY:-${SWANLAB_API_KEY:-}}"
if [[ -z "${SWANLAB_KEY}" ]]; then
  echo "error: set USER_SWANLAB_API_KEY or SWANLAB_API_KEY" >&2
  exit 1
fi

MODEL_PATH=/apdcephfs_gy4/share_302774114/ziniuli/models/Qwen/Qwen2.5-Math-1.5B
DATA_ROOT=/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/knapsack_async_data_parquet
TRAIN_FILE="${DATA_ROOT}/dapo/train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw_unclipped.parquet"
VAL_ROOT=/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler
TEST_FILE="[${VAL_ROOT}/aime24.parquet,${VAL_ROOT}/aime25.parquet,${VAL_ROOT}/amc_fixed.parquet,${VAL_ROOT}/math_100.parquet,${VAL_ROOT}/minerva_fixed_100.parquet,${VAL_ROOT}/olympiad_bench_fixed_100.parquet,${VAL_ROOT}/gpqa_diamond_100.parquet]"
CKPTS_DIR=/apdcephfs_gy4/share_302774114/ziniuli/exp_hub/knapsack_async_ckpts

cd "${ROOT_DIR}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

mkdir -p "${STAGE}"
rsync -a ./ "${STAGE}/" \
  --exclude '.git' \
  --exclude 'ckpts' \
  --exclude 'logs' \
  --exclude 'monitor_logs' \
  --exclude 'outputs' \
  --exclude 'swanlog' \
  --exclude 'tmp' \
  --exclude '__pycache__'

echo "submitting ${SUBMISSION_ID}"
echo "model=${MODEL_PATH}"
echo "train_file=${TRAIN_FILE}"

ray job submit \
  --address "${RAY_DASHBOARD}" \
  --working-dir "${STAGE}" \
  --submission-id "${SUBMISSION_ID}" \
  --no-wait \
  -- \
  env \
    RAY_ADDRESS=auto \
    USER_SWANLAB_API_KEY="${SWANLAB_KEY}" \
    SWANLAB_API_KEY="${SWANLAB_KEY}" \
    SWANLAB_HOST=https://train-exp.taiji.woa.com \
    SWANLAB_API_HOST=https://train-exp.taiji.woa.com \
    SWANLAB_WEB_HOST=https://train-exp.taiji.woa.com \
    SWANLAB_WORKSPACE=X4 \
    PROJECT_NAME=wangpengyuan_verl \
    "TRAINER_LOGGER=['console','swanlab']" \
    EXP_MODEL_NAME=Qwen2.5-Math-1.5B \
    EXP_VERSION="${EXP_VERSION}" \
    TRITON_CACHE_DIR="/tmp/triton_cache_${EXP_VERSION}" \
    MODEL_PATH="${MODEL_PATH}" \
    TRAIN_FILE="${TRAIN_FILE}" \
    TEST_FILE="${TEST_FILE}" \
    CKPTS_DIR="${CKPTS_DIR}" \
    RAY_DATA_HOME="${DATA_ROOT}" \
    REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py \
    SAMPLE_STRATEGY=shuffle_sequential \
    ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus \
    STOP_RULE=offline_medium_focus_budget_compensate \
    STOP_RULE_MIN_N=4 \
    STOP_RULE_MAX_N=14 \
    MEDIUM_FOCUS_PEAK_N=14 \
    ROLLOUT_CHUNK_SIZE=2 \
    N_RESP_PER_PROMPT=8 \
    REWARD_EMA=0.1 \
    ADAPTIVE_CHUNK_PRIOR_TAU=8.0 \
    COMPENSATION_RULE=min_rescue_prob \
    COMPENSATION_RESCUE_PROB=0.85 \
    COMPENSATION_ESTIMATE_CLIP=0.02 \
    COMPENSATION_BUDGET_BASELINE_N=8 \
    COMPENSATION_BUDGET_WINDOW_SIZE=512 \
    COMPENSATION_ROI_THRESHOLD=1.0 \
    TRAIN_TRIGGER=fixed_samples \
    BATCH_SIZE=128 \
    TRAIN_PROMPT_MINI_BSZ=128 \
    STALENESS_THRESHOLD=0 \
    LOSS_AGG_MODE=token-mean \
    REJECTION_SAMPLING=False \
    ROLLOUT_BUDGET_MULT=1 \
    MAX_PROMPT_LENGTH=2048 \
    MAX_RESPONSE_LENGTH=2048 \
    VAL_MAX_RESPONSE_LENGTH=2048 \
    ACTOR_PPO_MAX_TOKEN_MULT=8 \
    INFER_PPO_MAX_TOKEN_MULT=8 \
    NNODES_ROLLOUT=1 \
    NNODES_TRAIN=1 \
    N_GPUS_ROLLOUT=4 \
    N_GPUS_TRAINING=4 \
    NGPUS_PER_NODE=8 \
    ROLLOUT_PACK_REPLICAS=True \
    GEN_TP=1 \
    SP_SIZE=1 \
    MAX_TRAINING_STEPS=600 \
    TOTAL_EPOCHS=10 \
    VAL_BEFORE_TRAIN=False \
    TEST_FREQ=20 \
    SAVE_FREQ=20 \
    DEBUG_ADAPTIVE_N=True \
    DEBUG_PRIORITY_SAMPLING=True \
    DEBUG_MAX_RECORDS_PER_EVENT=5000 \
    DEBUG_SAMPLE_EVERY=50 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh

echo "submitted ${SUBMISSION_ID}"
