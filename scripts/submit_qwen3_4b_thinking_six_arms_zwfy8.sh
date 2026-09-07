#!/usr/bin/env bash
# Launch one Qwen3-4B thinking experiment arm on an existing 2-node/16-GPU Ray cluster.
set -euo pipefail

arm="${1:?usage: $0 baseline|haspos1|rej_fixedn8|adaptn14|lowermid_adaptn14|rej_lowermid_adaptn14}"

ROOT=/apdcephfs_zwfy8/share_303922427/pencywang/project/knapsack_async
ZW_ROOT=/apdcephfs_zwfy8/share_303922427/pencywang
MODEL_PATH="${ZW_ROOT}/models/Qwen3-4B"
TRAIN_FILE="${ZW_ROOT}/data/qwen3_4b_thinking/dapo17k_qwen3_4b_thinking_n8_gen32k_trunc16k_train_clip001_099_drop_allcorrect_overlong0125.parquet"
VAL_ROOT="${EVAL_DATA_ROOT:-${ROOT}/data/eval/deepscaler}"
CKPTS_ROOT="${ZW_ROOT}/ckpts/knapsack_async_qwen3_4b_thinking"
DATE_TAG="${DATE_TAG:-$(date +%m%d_%H%M)}"

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

read -r -d '' REJECTION_BUDGET_EXPORTS <<'EOF' || true
export MAX_TRAINING_STEPS=2600
export TOTAL_EPOCHS=35
EOF

case "${arm}" in
  baseline)
    TAG="q3_4b_thinking_base_fixedn8_${DATE_TAG}"
    ARM_EXPORTS="export SAMPLE_STRATEGY=shuffle_sequential
export ADAPTIVE_ROLLOUT_N_STRATEGY=
export STOP_RULE=fixed_rollout
export STOP_RULE_MIN_N=1
export STOP_RULE_MAX_N=8
export ROLLOUT_CHUNK_SIZE=8
export REJECTION_SAMPLING=False"
    ;;
  haspos1)
    TAG="q3_4b_thinking_haspos1_n14_${DATE_TAG}"
    ARM_EXPORTS="export SAMPLE_STRATEGY=shuffle_sequential
export ADAPTIVE_ROLLOUT_N_STRATEGY=
export STOP_RULE=has_at_least_positive
export STOP_RULE_MIN_N=1
export STOP_RULE_MAX_N=14
export ROLLOUT_CHUNK_SIZE=2
export REJECTION_SAMPLING=False"
    ;;
  rej_fixedn8)
    TAG="q3_4b_thinking_rej_fixedn8_${DATE_TAG}"
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
    TAG="q3_4b_thinking_adaptn14_comp_${DATE_TAG}"
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
    TAG="q3_4b_thinking_lowermid_adaptn14_comp_${DATE_TAG}"
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
    TAG="q3_4b_thinking_rej_lowermid_adaptn14_${DATE_TAG}"
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
    echo "unknown arm '${arm}'" >&2
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

export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-X4}"
export SWANLAB_HOST="${SWANLAB_HOST:-https://train-exp.taiji.woa.com}"
export SWANLAB_API_HOST="${SWANLAB_API_HOST:-${SWANLAB_HOST}}"
export SWANLAB_WEB_HOST="${SWANLAB_WEB_HOST:-${SWANLAB_HOST}}"
export PROJECT_NAME="${PROJECT_NAME:-wangpengyuan_verl}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-['console','swanlab']}"
export EXP_MODEL_NAME=Qwen3-4B-Thinking
export EXP_VERSION="${TAG}"
export TRITON_CACHE_DIR="/tmp/triton_cache_${TAG}"

export MODEL_PATH TRAIN_FILE
export TEST_FILE="[${VAL_ROOT}/aime24.parquet,${VAL_ROOT}/aime25.parquet,${VAL_ROOT}/amc_fixed.parquet,${VAL_ROOT}/math_100.parquet,${VAL_ROOT}/minerva_fixed_100.parquet,${VAL_ROOT}/olympiad_bench_fixed_100.parquet,${VAL_ROOT}/gpqa_diamond_100.parquet]"
export CKPTS_DIR="${CKPTS_ROOT}"
export RAY_DATA_HOME="${ZW_ROOT}/project/data_parquet"
export REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py
export ENABLE_THINKING=True

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

export MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-1500}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=50
export SAVE_FREQ=50

echo "launching ${TAG} with RAY_ADDRESS=${RAY_ADDRESS}"
echo "training data: ${TRAIN_FILE}"
exec bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh
