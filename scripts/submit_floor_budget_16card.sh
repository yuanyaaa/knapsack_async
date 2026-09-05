#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_hldy/share_303922427/pencywang/project/knapsack_async

export RAY_ADDRESS="${RAY_ADDRESS:-http://29.81.226.183:8081}"

STAMP="${STAMP:-20260624_0028}"
STAGE="${STAGE:-/tmp/knapsack_async_submit_floor_budget_${STAMP}}"

mkdir -p "${STAGE}"
rsync -a ./ "${STAGE}/" \
  --exclude '.git' \
  --exclude 'logs' \
  --exclude 'outputs' \
  --exclude 'monitor_logs' \
  --exclude 'tmp' \
  --exclude 'swanlog' \
  --exclude '__pycache__'

OLD_JOBS=(
  codex_knapsack_qwen25_seq_adaptive_chunk_mixed_productive_curve_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_fresh_20260622_111751
  codex_knapsack_qwen25_priority_medium_adaptive_chunk_mixed_productive_curve_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260622_2007
  codex_knapsack_qwen25_seq_adaptive_chunk_mixed_floor_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260623_0100
  codex_knapsack_qwen25_seq_mediumfocus_floor_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260623_0100
  codex_knapsack_qwen25_seq_productive_curve_floor_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260623_0100
  codex_knapsack_qwen25_seq_adaptive_chunk_mixed_p95_floor_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260624_1430
  codex_knapsack_qwen25_productive_mixed_adaptive_chunk_mixed_p95_floor_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260624_1430
  codex_knapsack_qwen25_seq_adaptive_chunk_mixed_medium_focus_plateau_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260624_0028
  codex_knapsack_qwen25_productive_mixed_plateau_adaptive_chunk_mixed_medium_focus_plateau_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260624_0028
)

for job in "${OLD_JOBS[@]}"; do
  echo "[stop-if-running] ${job}"
  ray job stop --address="${RAY_ADDRESS}" "${job}" || true
done

TEST_FILE='[/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/aime24.parquet,/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/aime25.parquet,/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/amc_fixed.parquet,/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/math_100.parquet,/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/minerva_fixed_100.parquet,/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/olympiad_bench_fixed_100.parquet,/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/deepscaler/gpqa_diamond_100.parquet]'

COMMON_ENV=(
  "USER_SWANLAB_API_KEY=${USER_SWANLAB_API_KEY:?set USER_SWANLAB_API_KEY}"
  "SWANLAB_API_KEY=${USER_SWANLAB_API_KEY}"
  SWANLAB_WORKSPACE=X4
  PROJECT_NAME=wangpengyuan_verl
  "TRAINER_LOGGER=['console','swanlab']"
  EXP_MODEL_NAME=Qwen2.5-Math-1.5B
  MODEL_PATH=/apdcephfs_gy4/share_302774114/ziniuli/models/Qwen/Qwen2.5-Math-1.5B
  TRAIN_FILE=/apdcephfs_gy4/share_302774114/ziniuli/processed_dataset/knapsack_async_data_parquet/dapo/train_knapsack_prompt_evaluated_qwen2.5_math_1.5b_dapo_n8_trunc2048-raw_unclipped.parquet
  "TEST_FILE=${TEST_FILE}"
  STOP_RULE_MIN_N=2
  STOP_RULE_MAX_N=16
  ROLLOUT_CHUNK_SIZE=2
  PRODUCTIVE_MIXED_MIN_N=2
  PRODUCTIVE_MIXED_MAX_N=16
  PRODUCTIVE_MIXED_CHUNK_SIZE=2
  PRODUCTIVE_MIXED_THRESHOLD=0.8
  PRODUCTIVE_MIXED_FALLBACK_WEIGHT=0.3333333333333333
  STALENESS_THRESHOLD=10
  BATCH_SIZE=128
  N_RESP_PER_PROMPT=8
  TRAIN_PROMPT_MINI_BSZ=128
  TRAIN_TRIGGER=fixed_samples
  ROLLOUT_BUDGET_MULT=2
  REJECTION_SAMPLING=false
  ADV_ESTIMATOR=grpo
  VAL_BEFORE_TRAIN=False
  TEST_FREQ=20
  SAVE_FREQ=20
  TOTAL_EPOCHS=10
  MAX_RESPONSE_LENGTH=2048
  VAL_MAX_RESPONSE_LENGTH=2048
)

ray job submit \
  --address="${RAY_ADDRESS}" \
  --working-dir "${STAGE}" \
  --submission-id codex_knapsack_qwen25_seq_adaptive_chunk_mixed_medium_focus_plateau_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260624_0028 \
  --no-wait \
  -- \
  env \
  "${COMMON_ENV[@]}" \
  SAMPLE_STRATEGY=sequential \
  STOP_RULE=adaptive_chunk_mixed \
  ADAPTIVE_CHUNK_PRIOR_TAU=8.0 \
  EXP_VERSION=seq_adaptive_chunk_mixed_medium_focus_plateau_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260624_0028 \
  bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh

ray job submit \
  --address="${RAY_ADDRESS}" \
  --working-dir "${STAGE}" \
  --submission-id codex_knapsack_qwen25_productive_mixed_plateau_adaptive_chunk_mixed_medium_focus_plateau_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260624_0028 \
  --no-wait \
  -- \
  env \
  "${COMMON_ENV[@]}" \
  SAMPLE_STRATEGY=productive_mixed_plateau \
  STOP_RULE=adaptive_chunk_mixed \
  ADAPTIVE_CHUNK_PRIOR_TAU=8.0 \
  EXP_VERSION=productive_mixed_plateau_adaptive_chunk_mixed_medium_focus_plateau_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260624_0028 \
  bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh

ray job status --address="${RAY_ADDRESS}" codex_knapsack_qwen25_seq_adaptive_chunk_mixed_medium_focus_plateau_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260624_0028 || true
ray job status --address="${RAY_ADDRESS}" codex_knapsack_qwen25_productive_mixed_plateau_adaptive_chunk_mixed_medium_focus_plateau_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260624_0028 || true
