#!/usr/bin/env bash
# Offline pass-rate evaluation (n=8 per prompt) for the 1.5B models on ONE local zwfy8 node
# with 8 GPUs. Starts a private single-node Ray head, runs the requested models one after
# another, then stops Ray. No cluster submission.
#
#   usage: bash scripts/run_eval_1p5b_local_zwfy8.sh qwen25 | dpsk | all
#
# Settings mirror the parquets the training runs consume:
#   qwen25  train_knapsack_prompt.parquet, gen/trunc 2048, prior unclipped [0, 1], no filtering
#           -> reproduces ..._evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw_unclipped.parquet
#   dpsk    train_deepscaler_format.parquet, gen/trunc 32768, prior clipped [0.01, 0.99],
#           drop all-correct and overlong_ratio > 0.125
#           -> reproduces ..._evaluated_dpsk1.5b_n8_trunc32768-v0_clip001_099.parquet
set -euo pipefail

[[ $# -ge 1 ]] || { echo "usage: $0 qwen25 | dpsk | all" >&2; exit 1; }
if [[ "$1" == "all" ]]; then
  MODELS=(qwen25 dpsk)
else
  MODELS=("$@")
fi

ZW_ROOT=/apdcephfs_zwfy8/share_303922427/pencywang
ROOT="${ROOT:-${ZW_ROOT}/project/knapsack_async}"
CONDA_ENV="${CONDA_ENV:-knapsack_async}"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-/root/miniconda3/envs/${CONDA_ENV}}"
DATE_TAG="${DATE_TAG:-$(date +%m%d_%H%M)}"
HEAD_IP="${HEAD_IP:-$(hostname -I | awk '{print $1}')}"
RAY_PORT="${RAY_PORT:-6499}"
NGPUS="${NGPUS:-8}"

export PATH="${CONDA_PREFIX_DIR}/bin:${PATH}"
PYTHON="${CONDA_PREFIX_DIR}/bin/python"
RAY="${CONDA_PREFIX_DIR}/bin/ray"

unset RAY_ADDRESS grpc_proxy GRPC_PROXY
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,${HEAD_IP}"
export NO_PROXY="${no_proxy}"
export HYDRA_FULL_ERROR=1
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

echo "[preflight] env=${CONDA_PREFIX_DIR} root=${ROOT} head=${HEAD_IP}:${RAY_PORT}"
[[ -x "${PYTHON}" && -x "${RAY}" ]] || {
  echo "missing conda env ${CONDA_PREFIX_DIR}; build it with: bash ${ROOT}/scripts/install_knapsack_async_env_zwfy8.sh" >&2
  exit 20
}
"${PYTHON}" -c 'import ray, torch, vllm, verl; print("versions ray", ray.__version__, "torch", torch.__version__, "vllm", vllm.__version__, "gpus", torch.cuda.device_count())'
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge "${NGPUS}" ]] ||
  { echo "need ${NGPUS} GPUs on this node" >&2; exit 21; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024 {bad=1} END {exit bad}' ||
  { echo "a GPU already uses over 1 GiB; free the node first (nvidia-smi)" >&2; exit 21; }

configure_model() {
  case "$1" in
    qwen25)
      TAG="qwen25_math_1p5b_dapo_n8_trunc2048_raw_unclipped_${DATE_TAG}"
      EVAL_ROOT="${ZW_ROOT}/data/qwen25_math_1p5b/${TAG}"
      export MODEL_PATH="${ZW_ROOT}/models/Qwen2.5-Math-1.5B"
      export MODEL_NAME=qwen2.5_math_1.5b
      export TRAIN_FILE="${ZW_ROOT}/project/data_parquet/dapo/train_knapsack_prompt.parquet"
      export OUTPUT_PARQUET="${EVAL_ROOT}/train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw_unclipped.parquet"
      # Qwen2.5-Math-1.5B has a 4096-token context: 2048 prompt + 2048 response.
      export MAX_RESPONSE_LENGTH=2048
      export TRUNCATION_LENGTH=2048
      export SAFE_LOW_BOUND=0.0
      export SAFE_UPPER_BOUND=1.0
      export FILTER_PASS_RATE_MIN=
      export FILTER_PASS_RATE_MAX=
      export FILTER_OVERLONG_MAX=
      ;;
    dpsk)
      TAG="dpsk1p5b_dapo_n8_trunc32768_clip001_099_${DATE_TAG}"
      EVAL_ROOT="${ZW_ROOT}/data/dpsk1.5b/${TAG}"
      export MODEL_PATH="${ZW_ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B"
      export MODEL_NAME=dpsk1.5b
      export TRAIN_FILE="${ZW_ROOT}/data/dpsk1.5b/train_deepscaler_format.parquet"
      export OUTPUT_PARQUET="${EVAL_ROOT}/train_deepscaler_format_evaluated_dpsk1.5b_n8_trunc32768-v0_clip001_099.parquet"
      export MAX_RESPONSE_LENGTH=32768
      export TRUNCATION_LENGTH=32768
      export SAFE_LOW_BOUND=0.01
      export SAFE_UPPER_BOUND=0.99
      export FILTER_PASS_RATE_MIN=
      export FILTER_PASS_RATE_MAX=1
      export FILTER_OVERLONG_MAX=0.125
      ;;
    *)
      echo "unknown model '$1' (expected qwen25 | dpsk | all)" >&2
      return 1
      ;;
  esac
  export REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py
  export REWARD_FN_NAME=compute_score
  export NNODES=1
  export NGPUS_PER_NODE="${NGPUS}"
  export TP_SIZE=1
  export GPU_MEM_UTIL=0.85
  export N_SAMPLES=8
  export MAX_PROMPT_LENGTH=2048
  export TEMPERATURE=1.0
  export TOP_P=1.0
  export TOP_K=-1
  export ENABLE_THINKING=
  export VAL_DATA_DIR="${EVAL_ROOT}/validation_data"
  export LIGHTWEIGHT_JSON="${EVAL_ROOT}/samples.jsonl"
  export LOG_DIR="${EVAL_ROOT}/logs"
  export EXTRACT_LIGHTWEIGHT=true
  export TRITON_CACHE_DIR="/tmp/triton_cache_${TAG}"
  for path in "${MODEL_PATH}/config.json" "${TRAIN_FILE}"; do
    [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; return 1; }
  done
}

for model in "${MODELS[@]}"; do
  ( configure_model "${model}" ) >/dev/null
done

stop_ray() { "${RAY}" stop --force >/dev/null 2>&1 || true; }
trap stop_ray EXIT

echo "[ray] starting single-node head on ${HEAD_IP}:${RAY_PORT}"
stop_ray
rm -rf /tmp/ray || true
"${RAY}" start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
  --num-gpus="${NGPUS}" --num-cpus="$(nproc)" --include-dashboard=false
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
for _ in $(seq 1 30); do
  "${RAY}" status --address="${RAY_ADDRESS}" >/dev/null 2>&1 && break
  sleep 2
done
"${RAY}" status --address="${RAY_ADDRESS}" | grep -E "GPU|CPU" || true

cd "${ROOT}"
for model in "${MODELS[@]}"; do
  (
    configure_model "${model}"
    mkdir -p "${LOG_DIR}"
    echo "============================================================"
    echo "[eval] ${model}: ${TAG}"
    echo "  output parquet: ${OUTPUT_PARQUET}"
    echo "============================================================"
    bash verl/experimental/fully_async_policy/shell/run_evaluate_and_filter.sh
  )
done
echo "[done] ${MODELS[*]}"
