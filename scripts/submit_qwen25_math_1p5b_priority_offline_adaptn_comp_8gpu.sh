#!/usr/bin/env bash
# Qwen2.5-Math-1.5B on one 8-GPU Ray node:
# 4 rollout GPUs + 4 training GPUs, lower-mid priority sampling,
# offline medium-focus adaptive N, and sliding-window budget compensation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${ROOT_DIR:-$(dirname "${SCRIPT_DIR}")}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-${ROOT_DIR}/data/eval/deepscaler}"

CONDA_ENV="${CONDA_ENV:-knapsack_async}"
CONDA_ENV_DIR="/root/miniconda3/envs/${CONDA_ENV}"
if [[ -x "${CONDA_ENV_DIR}/bin/python" ]]; then
  export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
fi

command -v python >/dev/null || { echo "error: python is not available" >&2; exit 1; }
python - <<'PY'
import sys
import torch
import ray
import vllm
import verl

if torch.cuda.device_count() != 8:
    raise RuntimeError(f"expected 8 local GPUs, found {torch.cuda.device_count()}")
print("python", sys.executable)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("ray", ray.__version__, "vllm", vllm.__version__)
print("verl", verl.__file__)
PY

pick_existing() {
  local fallback="$1"
  shift
  local path
  for path in "$@"; do
    if [[ -e "${path}" ]]; then
      printf '%s\n' "${path}"
      return
    fi
  done
  printf '%s\n' "${fallback}"
}

if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_PATH="$(pick_existing \
    /apdcephfs_zwfy8/share_303922427/pencywang/models/Qwen2.5-Math-1.5B \
    /taijifs_zw35/r2/ziniuli/models/Qwen/Qwen2.5-Math-1.5B \
    /data/chencongliang/wangpengyuan/models/Qwen/Qwen2.5-Math-1.5B \
    /apdcephfs_gy4/share_302774114/ziniuli/models/Qwen/Qwen2.5-Math-1.5B \
    /apdcephfs_zwfy8/share_303922427/pencywang/models/Qwen2.5-Math-1.5B)"
fi

if [[ -z "${TRAIN_FILE:-}" ]]; then
  TRAIN_FILE="$(pick_existing \
    /apdcephfs_zwfy8/share_303922427/pencywang/project/data_parquet/dapo/train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw_unclipped.parquet \
    /apdcephfs_zwfy8/share_303922427/pencywang/project/data_parquet/dapo/train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw_unclipped.parquet \
    /taijifs_zw35/r2/ziniuli/processed_dataset/dapo/train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_n8_trunc2048-v0.parquet \
    /data/chencongliang/wangpengyuan/project/data_parquet/dapo/train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw_unclipped.parquet \
    /apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/knapsack_async_data_parquet/dapo/train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw_unclipped.parquet)"
fi

CKPTS_DIR="${CKPTS_DIR:-${ROOT_DIR}/ckpts/qwen25_math_1p5b}"

REJECTION_SAMPLING="${REJECTION_SAMPLING:-False}"
ROLLOUT_BUDGET_MULT="${ROLLOUT_BUDGET_MULT:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
ARM_SUFFIX="${ARM_SUFFIX:-norej}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_VERSION="${EXP_VERSION:-qwen25_math_1p5b_priority_lower_mid_offline_adaptn_comp_${ARM_SUFFIX}_8gpu_${RUN_TAG}}"

SWANLAB_KEY="${USER_SWANLAB_API_KEY:-${SWANLAB_API_KEY:-}}"
if [[ -z "${TRAINER_LOGGER:-}" ]]; then
  TRAINER_LOGGER="['console','swanlab']"
fi
if [[ "${TRAINER_LOGGER}" == *swanlab* && -z "${SWANLAB_KEY}" ]]; then
  echo "error: set USER_SWANLAB_API_KEY or SWANLAB_API_KEY" >&2
  exit 1
fi

for path in \
  "${MODEL_PATH}/config.json" \
  "${TRAIN_FILE}" \
  "${EVAL_DATA_ROOT}/aime24.parquet" \
  "${EVAL_DATA_ROOT}/aime25.parquet" \
  "${EVAL_DATA_ROOT}/amc_fixed.parquet" \
  "${EVAL_DATA_ROOT}/math_100.parquet" \
  "${EVAL_DATA_ROOT}/minerva_fixed_100.parquet" \
  "${EVAL_DATA_ROOT}/olympiad_bench_fixed_100.parquet" \
  "${EVAL_DATA_ROOT}/gpqa_diamond_100.parquet"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 1; }
done

TEST_FILE="[${EVAL_DATA_ROOT}/aime24.parquet,${EVAL_DATA_ROOT}/aime25.parquet,${EVAL_DATA_ROOT}/amc_fixed.parquet,${EVAL_DATA_ROOT}/math_100.parquet,${EVAL_DATA_ROOT}/minerva_fixed_100.parquet,${EVAL_DATA_ROOT}/olympiad_bench_fixed_100.parquet,${EVAL_DATA_ROOT}/gpqa_diamond_100.parquet]"

cd "${ROOT_DIR}"
unset RAY_ADDRESS
export USER_SWANLAB_API_KEY="${SWANLAB_KEY}"
export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-X4}"
export PROJECT_NAME="${PROJECT_NAME:-wangpengyuan_verl}"
export TRAINER_LOGGER
export EXP_MODEL_NAME=Qwen2.5-Math-1.5B
export EXP_VERSION
export MODEL_PATH TRAIN_FILE TEST_FILE CKPTS_DIR
export RAY_DATA_HOME="$(dirname "${TRAIN_FILE}")"
export REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py

export SAMPLE_STRATEGY=lower_mid
export REWARD_EMA=0.1
export ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus
export STOP_RULE=offline_medium_focus_budget_compensate
export STOP_RULE_MIN_N=4
export STOP_RULE_MAX_N=16
export MEDIUM_FOCUS_PEAK_N=8
export ROLLOUT_CHUNK_SIZE=2
export ADAPTIVE_CHUNK_PRIOR_TAU=8.0
export COMPENSATION_RULE=roi_max
export COMPENSATION_BUDGET_BASELINE_N=8
export COMPENSATION_BUDGET_WINDOW_SIZE=512
export COMPENSATION_ROI_THRESHOLD=1.0

export REJECTION_SAMPLING REJECTION_QUEUE_TIMEOUT=60
export ROLLOUT_BUDGET_MULT
export N_RESP_PER_PROMPT=8
export TRAIN_TRIGGER=fixed_samples
export BATCH_SIZE=128
export TRAIN_PROMPT_MINI_BSZ=128
export STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-0}"
export LOSS_AGG_MODE=token-mean

export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=2048
export VAL_MAX_RESPONSE_LENGTH=2048
export ACTOR_PPO_MAX_TOKEN_MULT=8
export INFER_PPO_MAX_TOKEN_MULT=8
export NNODES_ROLLOUT=1
export NNODES_TRAIN=1
export N_GPUS_ROLLOUT=4
export N_GPUS_TRAINING=4
export NGPUS_PER_NODE=8
export ROLLOUT_PACK_REPLICAS=True
export SP_SIZE=2

export MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-600}"
export TOTAL_EPOCHS
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=20
export SAVE_FREQ=20
export DEBUG_ADAPTIVE_N=True
export DEBUG_PRIORITY_SAMPLING=True
export DEBUG_MAX_RECORDS_PER_EVENT=5000
export DEBUG_SAMPLE_EVERY=50

echo "starting local run ${EXP_VERSION}"
echo "python=${CONDA_ENV_DIR}/bin/python model=${MODEL_PATH}"
echo "train_file=${TRAIN_FILE}"
exec bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh
