#!/usr/bin/env bash
set -uo pipefail

cd /apdcephfs_hldy/share_303922427/pencywang/project/knapsack_async || exit 1

RAY_ADDRESS="${RAY_ADDRESS:-http://29.232.242.135:8081}"
BS32_JOB_ID="${BS32_JOB_ID:-raysubmit_2AWYKCTkc96Zpz9t}"
WAIT_SECONDS="${WAIT_SECONDS:-1200}"
LOG_DIR="${LOG_DIR:-/apdcephfs_hldy/share_303922427/pencywang/project/knapsack_async/monitor_logs/bs32_to_bs64_watchdog}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/watchdog.log}"
SUBMIT_LOG="${SUBMIT_LOG:-${LOG_DIR}/bs64_submit.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/watchdog.pid}"

mkdir -p "${LOG_DIR}"
echo "$$" > "${PID_FILE}"

ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log() {
  printf '[%s] %s\n' "$(ts)" "$*" | tee -a "${LOG_FILE}"
}

submit_bs64() {
  export RAY_ADDRESS
  ray job submit --address "${RAY_ADDRESS}" --no-wait --working-dir . -- \
    bash -lc 'export SWANLAB_API_KEY='"${SWANLAB_API_KEY:?set SWANLAB_API_KEY}"'; \
    export PROJECT_NAME=lzn_verl; \
    export EXP_VERSION=dpsk_distill_seq_fixedn8_fixprompt_16krollout_32keval; \
    export PERF_TAG=bs64_32gpu_tokenmult4_minibsz32_noallocconf; \
    export MODEL_PATH=/apdcephfs/share_302503006/ziniuli/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; \
    export TRAIN_FILE=/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/train.parquet; \
    export TEST_FILE="[/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/aime24.parquet,/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/aime25.parquet,/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/amc_fixed.parquet,/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/math_100.parquet,/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/minerva_fixed_100.parquet,/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/olympiad_bench_fixed_100.parquet,/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/gpqa_diamond_100.parquet]"; \
    export CKPTS_DIR=/apdcephfs/share_302503006/ziniuli/checkpoint/knapsack_async_dpsk_distill; \
    export RAY_DATA_HOME=/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler; \
    export REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py; \
    export SAMPLE_STRATEGY=null; \
    export STOP_RULE=fixed_rollout; \
    export STOP_RULE_MIN_N=1; \
    export STOP_RULE_MAX_N=8; \
    export TRAIN_TRIGGER=fixed_prompt; \
    export N_RESP_PER_PROMPT=8; \
    export BATCH_SIZE=64; \
    export TRAIN_PROMPT_MINI_BSZ=32; \
    export ACTOR_PPO_MAX_TOKEN_MULT=4; \
    export INFER_PPO_MAX_TOKEN_MULT=4; \
    export MAX_PROMPT_LENGTH=2048; \
    export MAX_RESPONSE_LENGTH=16384; \
    export VAL_MAX_RESPONSE_LENGTH=32768; \
    export VAL_BEFORE_TRAIN=False; \
    export NNODES_ROLLOUT=2; \
    export NNODES_TRAIN=2; \
    export N_GPUS_ROLLOUT=8; \
    export N_GPUS_TRAINING=8; \
    export NGPUS_PER_NODE=8; \
    export TEST_FREQ=50; \
    export SAVE_FREQ=50; \
    bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh'
}

log "watchdog_started pid=$$ bs32_job=${BS32_JOB_ID} wait_seconds=${WAIT_SECONDS}"
sleep "${WAIT_SECONDS}"

status_out="$(ray job status --address "${RAY_ADDRESS}" "${BS32_JOB_ID}" 2>&1)"
log "bs32_status_begin"
printf '%s\n' "${status_out}" >> "${LOG_FILE}"
log "bs32_status_end"

if ! printf '%s\n' "${status_out}" | grep -q "RUNNING"; then
  log "skip_bs64 reason=bs32_not_running"
  exit 0
fi

log_tmp="$(mktemp)"
if ! timeout 240 ray job logs --address "${RAY_ADDRESS}" "${BS32_JOB_ID}" > "${log_tmp}" 2>&1; then
  log "skip_bs64 reason=bs32_logs_unavailable"
  tail -120 "${log_tmp}" >> "${LOG_FILE}" 2>/dev/null || true
  rm -f "${log_tmp}"
  exit 0
fi

if grep -qiE "OutOfMemoryError|CUDA out of memory|Traceback|RayTaskError|TaskCancelledError|Error executing job|failed with exit code|ActorDiedError|WorkerCrashed" "${log_tmp}"; then
  log "skip_bs64 reason=bs32_error_keyword_found"
  grep -niE "OutOfMemoryError|CUDA out of memory|Traceback|RayTaskError|TaskCancelledError|Error executing job|failed with exit code|ActorDiedError|WorkerCrashed" "${log_tmp}" | tail -80 >> "${LOG_FILE}" || true
  rm -f "${log_tmp}"
  exit 0
fi

if ! grep -qE "training/global_step:[1-9]|global_steps: [1-9]|Training Progress:" "${log_tmp}"; then
  log "skip_bs64 reason=bs32_no_training_progress_yet"
  rm -f "${log_tmp}"
  exit 0
fi

rm -f "${log_tmp}"
log "submit_bs64_begin"
submit_bs64 > "${SUBMIT_LOG}" 2>&1
submit_rc=$?
cat "${SUBMIT_LOG}" >> "${LOG_FILE}"
log "submit_bs64_end rc=${submit_rc}"
exit "${submit_rc}"
