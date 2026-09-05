#!/usr/bin/env bash
# DPSK 1.5B + lower_mid priority sampling + adaptive n (medium_focus 4/8/16) + min_rescue_prob compensation.
# Identical to submit_dpsk15b_rescuefix.sh except SAMPLE_STRATEGY, so the only variable under
# test is the prompt sampling composition. lower_mid weight = p*(1-p)^2, peak at p=1/3.
set -euo pipefail

RAY_DASHBOARD="${RAY_DASHBOARD:-http://28.83.35.117:8081}"
ROOT_DIR="${ROOT_DIR:-/apdcephfs_hldy/share_303922427/pencywang/project/knapsack_async}"
TAG="${TAG:-dpsk15b_lowermid_mf8_16_rescuefix_0825_2040}"
SUBMISSION_ID="${SUBMISSION_ID:-codex_${TAG}}"

DATA_ROOT=/taijifs_zw35/r2/ziniuli/processed_dataset
VAL_ROOT="${DATA_ROOT}/deepscaler"

cd "${ROOT_DIR}"
ray job submit --address "${RAY_DASHBOARD}" --working-dir . --no-wait \
  --submission-id "${SUBMISSION_ID}" -- bash -lc "set -euo pipefail
unset PYTORCH_CUDA_ALLOC_CONF
export RAY_ADDRESS=auto
: "${USER_SWANLAB_API_KEY:?set USER_SWANLAB_API_KEY before launching}"
export USER_SWANLAB_API_KEY
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

export SAMPLE_STRATEGY=lower_mid
export ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus
export STOP_RULE=offline_medium_focus_budget_compensate
export STOP_RULE_MIN_N=4
export STOP_RULE_MAX_N=16
export MEDIUM_FOCUS_PEAK_N=8
export N_RESP_PER_PROMPT=8
export REWARD_EMA=0.1
export ADAPTIVE_CHUNK_PRIOR_TAU=8.0

export COMPENSATION_RULE=min_rescue_prob
export COMPENSATION_RESCUE_PROB=0.85
export COMPENSATION_ESTIMATE_CLIP=0.02
export COMPENSATION_BUDGET_BASELINE_N=8
export COMPENSATION_BUDGET_WINDOW_SIZE=512
export COMPENSATION_ROI_THRESHOLD=1.0

export TRAIN_TRIGGER=fixed_samples
export BATCH_SIZE=128
export TRAIN_PROMPT_MINI_BSZ=128
export STALENESS_THRESHOLD=0
export REJECTION_SAMPLING=False

export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=16384
export VAL_MAX_RESPONSE_LENGTH=30720
export ACTOR_PPO_MAX_TOKEN_MULT=2
export INFER_PPO_MAX_TOKEN_MULT=2

export NNODES_ROLLOUT=1
export NNODES_TRAIN=1
export N_GPUS_ROLLOUT=8
export N_GPUS_TRAINING=8
export NGPUS_PER_NODE=8

export TOTAL_EPOCHS=20
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=50
export SAVE_FREQ=50

export DEBUG_ADAPTIVE_N=True
export DEBUG_PRIORITY_SAMPLING=True
export DEBUG_MAX_RECORDS_PER_EVENT=5000
export DEBUG_SAMPLE_EVERY=50

bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh"
