#!/usr/bin/env bash
set -euo pipefail

export RAY_ADDRESS="${RAY_ADDRESS:-http://29.81.226.183:8081}"

WORKING_DIR="${WORKING_DIR:-/tmp/knapsack_async_submit_min_20260621_1335}"
SWANLAB_KEY="${SWANLAB_API_KEY:-${USER_SWANLAB_API_KEY:-}}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${SWANLAB_KEY}" ]]; then
  echo "error: set SWANLAB_API_KEY or USER_SWANLAB_API_KEY first" >&2
  exit 1
fi

echo "Using RUN_SUFFIX=${RUN_SUFFIX}"

COMMON_ENV=(
  USER_SWANLAB_API_KEY="${SWANLAB_KEY}"
  SWANLAB_API_KEY="${SWANLAB_KEY}"
  SWANLAB_WORKSPACE=X4
  PROJECT_NAME=wangpengyuan_verl
  TRAINER_LOGGER="['console','swanlab']"
  EXP_MODEL_NAME=Qwen2.5-Math-1.5B
  MODEL_PATH=/apdcephfs_gy4/share_302774114/ziniuli/models/Qwen/Qwen2.5-Math-1.5B
  TRAIN_FILE=/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/knapsack_async_data_parquet/dapo/train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw_unclipped.parquet
  TEST_FILE=[/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/aime24.parquet,/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/aime25.parquet,/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/amc_fixed.parquet,/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/math_100.parquet,/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/minerva_fixed_100.parquet,/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/olympiad_bench_fixed_100.parquet,/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/gpqa_diamond_100.parquet]
  CKPTS_DIR=/apdcephfs_gy4/share_302774114/ziniuli/exp_hub/knapsack_async_ckpts
  STALENESS_THRESHOLD=10
  BATCH_SIZE=128
  N_RESP_PER_PROMPT=8
  TRAIN_PROMPT_MINI_BSZ=128
  TRAIN_TRIGGER=fixed_samples
  REJECTION_SAMPLING=false
  ADV_ESTIMATOR=grpo
  VAL_BEFORE_TRAIN=False
  TEST_FREQ=20
  SAVE_FREQ=20
  TOTAL_EPOCHS=10
  ROLLOUT_BUDGET_MULT=2
)

ray job submit \
  --address="${RAY_ADDRESS}" \
  --no-wait \
  --submission-id "codex_knapsack_qwen25_priority_adaptiven_mediumfocus_chunk4_fs1024_stale10_grpo_norej_ep10_eval20_fresh_${RUN_SUFFIX}" \
  --working-dir "${WORKING_DIR}" \
  -- \
  env \
  "${COMMON_ENV[@]}" \
  SAMPLE_STRATEGY=medium \
  ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus \
  STOP_RULE=fixed_rollout \
  STOP_RULE_MIN_N=2 \
  STOP_RULE_MAX_N=16 \
  ROLLOUT_CHUNK_SIZE=4 \
  EXP_VERSION="priority_adaptiven_mediumfocus_chunk4_fs1024_stale10_grpo_norej_ep10_eval20_fresh_${RUN_SUFFIX}" \
  bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh

ray job submit \
  --address="${RAY_ADDRESS}" \
  --no-wait \
  --submission-id "codex_knapsack_qwen25_seq_adaptive_chunk_mixed_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_fresh_${RUN_SUFFIX}" \
  --working-dir "${WORKING_DIR}" \
  -- \
  env \
  "${COMMON_ENV[@]}" \
  SAMPLE_STRATEGY=sequential \
  STOP_RULE=adaptive_chunk_mixed \
  STOP_RULE_MIN_N=2 \
  STOP_RULE_MAX_N=16 \
  ROLLOUT_CHUNK_SIZE=2 \
  ADAPTIVE_CHUNK_ALPHA=0.5 \
  ADAPTIVE_MIXED_TARGET_PROB=0.85 \
  EXP_VERSION="seq_adaptive_chunk_mixed_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_fresh_${RUN_SUFFIX}" \
  bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh
