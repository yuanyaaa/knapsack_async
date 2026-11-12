#!/usr/bin/env bash
# Qwen2.5-Math-1.5B experiment matrix on the zw35 cluster, one shared configuration.
#
#   usage: USER_SWANLAB_API_KEY=... bash scripts/submit_qwen25_math_1p5b_matrix_zw35.sh <arm> [<arm> ...] | all
#
# Arms:
#   lowermid_mf32_comp  lower_mid priority (p(1-p)^2, peak p=1/3) + medium_focus N in [4, 32],
#                       peak_n=32 + min_rescue_prob compensation, 600 steps
#   lowermid_mf16_comp  same as lowermid_mf32_comp but initial K peaks at 16 and the compensation
#                       window budget is 16 per prompt (at 8 the lower_mid initial K alone exhausts it);
#                       compensation can raise a prompt to max N=32, 600 steps
#   edca_warmup1ep      ablation without offline pass rates: epoch 1 is shuffle_sequential GRPO with N=8
#                       (pass rates learned online), then lowermid_mf16_comp from epoch 2, 600 steps
#   grpo_fixedn8        shuffle_sequential, fixed N=8 GRPO baseline, 800 steps, 32 GPUs (rollout 2x8 + train 2x8)
#   shuf_invvar32       shuffle_sequential, N = floor(1/[p(1-p)] - 1) in [3, 32], no compensation, 600 steps
#   vip_b8              VIP baseline (arXiv:2602.01601): GP over MiniLM prompt embeddings with the offline
#                       pass rate as prior mean, Dr.GRPO-variance allocation of 8 x 128 rollouts per
#                       128-prompt block, N in [4, 32], RBF length scale 0.27, 600 steps
#
# All arms train on the raw-unclipped offline pass rates (17,398 prompts).
set -euo pipefail

RAY_DASHBOARD="${RAY_DASHBOARD:-http://28.81.130.218:8081}"
ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
RUN_DATE="${RUN_DATE:-$(date +%m%d)}"

SWANLAB_KEY="${USER_SWANLAB_API_KEY:-${SWANLAB_API_KEY:-}}"
if [[ -z "${SWANLAB_KEY}" ]]; then
  echo "error: set USER_SWANLAB_API_KEY or SWANLAB_API_KEY" >&2
  exit 1
fi

ZW_ROOT=/taijifs_zw35/r2/ziniuli
MODEL_PATH="${ZW_ROOT}/models/Qwen/Qwen2.5-Math-1.5B"
DATA_ROOT="${ZW_ROOT}/processed_dataset"
TRAIN_FILE="${DATA_ROOT}/knapsack_async_data_parquet/dapo/train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw_unclipped.parquet"
VAL_ROOT="${DATA_ROOT}/deepscaler"
TEST_FILE="[${VAL_ROOT}/aime24.parquet,${VAL_ROOT}/aime25.parquet,${VAL_ROOT}/amc_fixed.parquet,${VAL_ROOT}/math_100.parquet,${VAL_ROOT}/minerva_fixed_100.parquet,${VAL_ROOT}/olympiad_bench_fixed_100.parquet,${VAL_ROOT}/gpqa_diamond_100.parquet]"
CKPTS_DIR="${ZW_ROOT}/knapsack_async_ckpts_qwen25_math_1p5b"
VIP_EMBEDDING_PATH="${DATA_ROOT}/knapsack_async_data_parquet/dapo/vip/minilm_l6_v2_train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw_unclipped.npz"

ALL_ARMS=(lowermid_mf32_comp lowermid_mf16_comp edca_warmup1ep grpo_fixedn8 shuf_invvar32 vip_b8)

arm_config() {
  # Sets TAG and ARM_ENV (an array of KEY=VALUE pairs) for the requested arm.
  case "$1" in
    lowermid_mf32_comp)
      TAG="qwen25_lowermid_mf32_comp_p32_raw_s600_${RUN_DATE}"
      ARM_ENV=(
        SAMPLE_STRATEGY=lower_mid
        ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus
        STOP_RULE=offline_medium_focus_budget_compensate
        STOP_RULE_MIN_N=4
        STOP_RULE_MAX_N=32
        MEDIUM_FOCUS_PEAK_N=32
        ROLLOUT_CHUNK_SIZE=2
        ADAPTIVE_CHUNK_PRIOR_TAU=8.0
        COMPENSATION_RULE=min_rescue_prob
        COMPENSATION_RESCUE_PROB=0.85
        COMPENSATION_ESTIMATE_CLIP=0.02
        COMPENSATION_BUDGET_BASELINE_N=8
        COMPENSATION_BUDGET_WINDOW_SIZE=512
        COMPENSATION_ROI_THRESHOLD=1.0
        MAX_TRAINING_STEPS=600
      )
      ;;
    lowermid_mf16_comp)
      TAG="qwen25_lowermid_mf16_comp_p16_max32_b16_raw_s600_${RUN_DATE}"
      ARM_ENV=(
        SAMPLE_STRATEGY=lower_mid
        ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus
        STOP_RULE=offline_medium_focus_budget_compensate
        STOP_RULE_MIN_N=4
        STOP_RULE_MAX_N=32
        MEDIUM_FOCUS_PEAK_N=16
        ROLLOUT_CHUNK_SIZE=2
        ADAPTIVE_CHUNK_PRIOR_TAU=8.0
        COMPENSATION_RULE=min_rescue_prob
        COMPENSATION_RESCUE_PROB=0.85
        COMPENSATION_ESTIMATE_CLIP=0.02
        COMPENSATION_BUDGET_BASELINE_N=16
        COMPENSATION_BUDGET_WINDOW_SIZE=512
        COMPENSATION_ROI_THRESHOLD=1.0
        MAX_TRAINING_STEPS=600
      )
      ;;
    edca_warmup1ep)
      TAG="qwen25_edca_warmup1ep_grpo8_then_lowermid_mf16_max32_b16_nooffline_s600_${RUN_DATE}"
      ARM_ENV=(
        USE_DATASET_PASS_RATE=False
        WARMUP_EPOCHS=1
        SAMPLE_STRATEGY=lower_mid
        ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus
        STOP_RULE=offline_medium_focus_budget_compensate
        STOP_RULE_MIN_N=4
        STOP_RULE_MAX_N=32
        MEDIUM_FOCUS_PEAK_N=16
        ROLLOUT_CHUNK_SIZE=2
        ADAPTIVE_CHUNK_PRIOR_TAU=8.0
        COMPENSATION_RULE=min_rescue_prob
        COMPENSATION_RESCUE_PROB=0.85
        COMPENSATION_ESTIMATE_CLIP=0.02
        COMPENSATION_BUDGET_BASELINE_N=16
        COMPENSATION_BUDGET_WINDOW_SIZE=512
        COMPENSATION_ROI_THRESHOLD=1.0
        MAX_TRAINING_STEPS=600
      )
      ;;
    grpo_fixedn8)
      TAG="qwen25_grpo_fixedn8_raw_s800_32g_${RUN_DATE}"
      ARM_ENV=(
        SAMPLE_STRATEGY=shuffle_sequential
        ADAPTIVE_ROLLOUT_N_STRATEGY=
        STOP_RULE=fixed_rollout
        STOP_RULE_MIN_N=1
        STOP_RULE_MAX_N=8
        ROLLOUT_CHUNK_SIZE=8
        MAX_TRAINING_STEPS=800
        NNODES_ROLLOUT=2
        NNODES_TRAIN=2
        N_GPUS_ROLLOUT=8
        N_GPUS_TRAINING=8
      )
      ;;
    shuf_invvar32)
      TAG="qwen25_shuf_invvar_n3_32_nocomp_raw_s600_${RUN_DATE}"
      ARM_ENV=(
        SAMPLE_STRATEGY=shuffle_sequential
        ADAPTIVE_ROLLOUT_N_STRATEGY=inverse_bernoulli_variance
        STOP_RULE=fixed_rollout
        STOP_RULE_MIN_N=3
        STOP_RULE_MAX_N=32
        ROLLOUT_CHUNK_SIZE=2
        MAX_TRAINING_STEPS=600
      )
      ;;
    vip_b8)
      TAG="qwen25_vip_b8_n4_32_ls027_raw_s600_${RUN_DATE}"
      ARM_ENV=(
        SAMPLE_STRATEGY=shuffle_sequential
        ADAPTIVE_ROLLOUT_N_STRATEGY=vip
        STOP_RULE=fixed_rollout
        STOP_RULE_MIN_N=4
        STOP_RULE_MAX_N=32
        ROLLOUT_CHUNK_SIZE=2
        VIP_EMBEDDING_PATH="${VIP_EMBEDDING_PATH}"
        VIP_BLOCK_SIZE=128
        VIP_BUDGET_PER_PROMPT=8
        VIP_LENGTH_SCALE=0.27
        VIP_EPS=1.0e-6
        VIP_VARIANCE=dr_grpo
        MAX_TRAINING_STEPS=600
      )
      ;;
    *)
      echo "[ERROR] unknown arm '$1'; valid arms: ${ALL_ARMS[*]} all" >&2
      return 1
      ;;
  esac
}

STAGE="${STAGE:-/tmp/knapsack_async_qwen25_matrix_$(date +%Y%m%d_%H%M%S)}"

stage_code() {
  mkdir -p "${STAGE}"
  rsync -a "${ROOT_DIR}/" "${STAGE}/" \
    --exclude '.git' \
    --exclude 'ckpts' \
    --exclude 'logs' \
    --exclude 'monitor_logs' \
    --exclude 'outputs' \
    --exclude 'swanlog' \
    --exclude 'tmp' \
    --exclude '__pycache__'
}

submit_arm() {
  local arm="$1"
  arm_config "${arm}"
  local submission_id="${SUBMISSION_ID:-codex_${TAG}}"

  echo "=== submitting ${arm} as ${submission_id} ==="
  ray job submit \
    --address "${RAY_DASHBOARD}" \
    --working-dir "${STAGE}" \
    --submission-id "${submission_id}" \
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
      EXP_VERSION="${TAG}" \
      TRITON_CACHE_DIR="/tmp/triton_cache_${TAG}" \
      MODEL_PATH="${MODEL_PATH}" \
      TRAIN_FILE="${TRAIN_FILE}" \
      TEST_FILE="${TEST_FILE}" \
      CKPTS_DIR="${CKPTS_DIR}" \
      RAY_DATA_HOME="${DATA_ROOT}" \
      REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py \
      N_RESP_PER_PROMPT=8 \
      REWARD_EMA=0.1 \
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
      TOTAL_EPOCHS=10 \
      VAL_BEFORE_TRAIN=False \
      TEST_FREQ=20 \
      SAVE_FREQ=50 \
      DEBUG_ADAPTIVE_N=True \
      DEBUG_PRIORITY_SAMPLING=True \
      DEBUG_MAX_RECORDS_PER_EVENT=5000 \
      DEBUG_SAMPLE_EVERY=50 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${ARM_ENV[@]}" \
      bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh
}

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <arm> [<arm> ...] | all" >&2
  echo "arms: ${ALL_ARMS[*]}" >&2
  exit 1
fi

if [[ "$1" == "all" ]]; then
  requested=("${ALL_ARMS[@]}")
else
  requested=("$@")
fi

# Validate every arm before submitting any, so a typo cannot leave a half-launched matrix.
for arm in "${requested[@]}"; do
  ( arm_config "${arm}" ) >/dev/null
done

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY RAY_ADDRESS
stage_code
for arm in "${requested[@]}"; do
  submit_arm "${arm}"
done
