#!/usr/bin/env bash
# Qwen3-4B-Base RL on a dedicated 2-node/16-GPU pair:
#   shuffle_sequential + adaptive n=4..16 (peak/baseline n=8) + compensation.
# Run this entrypoint on 28.83.1.158. It starts Ray on both nodes and launches training.
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${SCRIPT_PATH}")" && pwd -P)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
INTERNAL_ARM="filtered_shuf_adaptn_comp_bs256"

if [[ "${1:-}" != "${INTERNAL_ARM}" ]]; then
  export HEAD_IP=28.83.1.158
  export WORKER_IP=28.85.98.167
  export ARM="${INTERNAL_ARM}"
  export SUBMIT_SCRIPT="${SCRIPT_PATH}"
  export CONDA_ENV="${CONDA_ENV:-knapsack_async}"

  exec bash "${SCRIPT_DIR}/run_qwen3_4b_pair_zwfy8.sh"
fi

ZW_ROOT=/apdcephfs_zwfy8/share_303922427/pencywang
MODEL_PATH="${MODEL_PATH:-${ZW_ROOT}/models/Qwen3-4B-Base}"
VAL_ROOT="${VAL_ROOT:-${ZW_ROOT}/project/data_parquet/deepscaler}"
CKPTS_DIR="${CKPTS_DIR:-${ZW_ROOT}/ckpts/knapsack_async_qwen3_4b_base}"

DATA_BASENAME=train_knapsack_prompt_evaluated_qwen3_4b_base_n8_32k_drop_acc_ge075.parquet
if [[ -z "${TRAIN_FILE:-}" ]]; then
  for candidate in \
    "${ZW_ROOT}/data/qwen3_4b_base/qwen3_4b_base_knapsack_n8_32k_drop_acc1_0910_0021/${DATA_BASENAME}" \
    "${ZW_ROOT}/data/qwen3_4b_base/${DATA_BASENAME}" \
    "${ZW_ROOT}/data/qwen3_4b_base"/*/"${DATA_BASENAME}" \
    "${ZW_ROOT}/project/data_parquet/dapo/${DATA_BASENAME}" \
    "${ROOT_DIR}/data/${DATA_BASENAME}"; do
    if [[ -f "${candidate}" ]]; then
      TRAIN_FILE="${candidate}"
      break
    fi
  done
fi
: "${TRAIN_FILE:=${ZW_ROOT}/data/qwen3_4b_base/${DATA_BASENAME}}"

for path in \
  "${MODEL_PATH}/config.json" \
  "${TRAIN_FILE}" \
  "${VAL_ROOT}/aime24.parquet" \
  "${VAL_ROOT}/aime25.parquet" \
  "${VAL_ROOT}/amc_fixed.parquet" \
  "${VAL_ROOT}/math_100.parquet" \
  "${VAL_ROOT}/minerva_fixed_100.parquet" \
  "${VAL_ROOT}/olympiad_bench_fixed_100.parquet" \
  "${VAL_ROOT}/gpqa_diamond_100.parquet"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

: "${USER_SWANLAB_API_KEY:?set USER_SWANLAB_API_KEY before launching}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_VERSION="${EXP_VERSION:-q3_4b_base_dropacc075_shuf_mf8_16_comp_bs256_n8_${RUN_TAG}}"

cd "${ROOT_DIR}"

export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export USER_SWANLAB_API_KEY
export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-X4}"
export SWANLAB_HOST="${SWANLAB_HOST:-https://train-exp.taiji.woa.com}"
export SWANLAB_API_HOST="${SWANLAB_API_HOST:-${SWANLAB_HOST}}"
export SWANLAB_WEB_HOST="${SWANLAB_WEB_HOST:-${SWANLAB_HOST}}"
export PROJECT_NAME="${PROJECT_NAME:-wangpengyuan_verl}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-['console','swanlab']}"
export EXP_MODEL_NAME=Qwen3-4B-Base
export EXP_VERSION
export TRITON_CACHE_DIR="/tmp/triton_cache_${EXP_VERSION}"

export MODEL_PATH TRAIN_FILE CKPTS_DIR
export TEST_FILE="[${VAL_ROOT}/aime24.parquet,${VAL_ROOT}/aime25.parquet,${VAL_ROOT}/amc_fixed.parquet,${VAL_ROOT}/math_100.parquet,${VAL_ROOT}/minerva_fixed_100.parquet,${VAL_ROOT}/olympiad_bench_fixed_100.parquet,${VAL_ROOT}/gpqa_diamond_100.parquet]"
export RAY_DATA_HOME="$(dirname "${TRAIN_FILE}")"
export REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py
export ENABLE_THINKING=False

export SAMPLE_STRATEGY=shuffle_sequential
export ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus
export STOP_RULE=offline_medium_focus_budget_compensate
export STOP_RULE_MIN_N=4
export STOP_RULE_MAX_N=16
export MEDIUM_FOCUS_PEAK_N=8
export N_RESP_PER_PROMPT=8
export ROLLOUT_CHUNK_SIZE=2
export REWARD_EMA=0.1
export ADAPTIVE_CHUNK_PRIOR_TAU=8.0

export COMPENSATION_RULE=min_rescue_prob
export COMPENSATION_RESCUE_PROB=0.85
export COMPENSATION_ESTIMATE_CLIP=0.02
export COMPENSATION_BUDGET_BASELINE_N=8
export COMPENSATION_BUDGET_WINDOW_SIZE=512
export COMPENSATION_ROI_THRESHOLD=1.0

export TRAIN_TRIGGER=fixed_samples
export BATCH_SIZE=256
export TRAIN_PROMPT_MINI_BSZ=256
export STALENESS_THRESHOLD=0
export REJECTION_SAMPLING=False
export ROLLOUT_BUDGET_MULT=1
export LOSS_AGG_MODE=token-mean

export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=30720
export VAL_MAX_RESPONSE_LENGTH=30720
export ACTOR_PPO_MAX_TOKEN_MULT=1
export INFER_PPO_MAX_TOKEN_MULT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GEN_TP=1
export SP_SIZE=2

export NNODES_ROLLOUT=1
export NNODES_TRAIN=1
export N_GPUS_ROLLOUT=8
export N_GPUS_TRAINING=8
export NGPUS_PER_NODE=8
export ROLLOUT_PACK_REPLICAS=True

export MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-600}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
export VAL_BEFORE_TRAIN=False
export TEST_FREQ="${TEST_FREQ:-20}"
export SAVE_FREQ="${SAVE_FREQ:-20}"

export DEBUG_ADAPTIVE_N=True
export DEBUG_PRIORITY_SAMPLING=True
export DEBUG_MAX_RECORDS_PER_EVENT=5000
export DEBUG_SAMPLE_EVERY=50

echo "launching ${EXP_VERSION} with RAY_ADDRESS=${RAY_ADDRESS}"
echo "model=${MODEL_PATH}"
echo "train_file=${TRAIN_FILE}"
echo "batch_size=${BATCH_SIZE}, nominal_n=${N_RESP_PER_PROMPT}, adaptive_n=${STOP_RULE_MIN_N}..${STOP_RULE_MAX_N}"
exec bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh
