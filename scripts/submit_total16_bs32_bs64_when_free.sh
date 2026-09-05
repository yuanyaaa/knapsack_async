#!/usr/bin/env bash
set -euo pipefail

RAY_DASHBOARD="${RAY_DASHBOARD:-http://29.232.242.135:8081}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
MIN_AVAILABLE_GPUS="${MIN_AVAILABLE_GPUS:-32}"
ROOT_DIR="${ROOT_DIR:-/apdcephfs_hldy/share_303922427/pencywang/project/knapsack_async}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/monitor_logs/total16_bs32_bs64_submitter}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/submitter.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/submitter.pid}"

mkdir -p "${LOG_DIR}"
echo "$$" > "${PID_FILE}"

ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log() {
  printf '[%s] %s\n' "$(ts)" "$*" | tee -a "${LOG_FILE}"
}

available_gpus() {
  ray job submit --address "${RAY_DASHBOARD}" --no-wait -- bash -lc 'python3 - <<'"'"'PY'"'"'
import json
import ray

ray.init(address="auto")
print(json.dumps(ray.available_resources(), sort_keys=True))
PY' >/tmp/total16_probe_submit.out 2>&1

  local probe_job
  probe_job="$(grep -o "raysubmit_[A-Za-z0-9]*" /tmp/total16_probe_submit.out | tail -1 || true)"
  if [[ -z "${probe_job}" ]]; then
    log "probe_submit_failed output=$(tr '\n' ' ' </tmp/total16_probe_submit.out)"
    echo 0
    return
  fi

  sleep 5
  ray job logs --address "${RAY_DASHBOARD}" "${probe_job}" >/tmp/total16_probe_logs.out 2>&1 || true
  python3 - <<'PY'
import json
import re

text = open("/tmp/total16_probe_logs.out", encoding="utf-8", errors="replace").read()
matches = re.findall(r"\{[^\n]*\}", text)
for raw in reversed(matches):
    try:
        data = json.loads(raw)
    except Exception:
        continue
    print(float(data.get("GPU", 0.0)))
    break
else:
    print(0.0)
PY
}

submit_one() {
  local batch_size="$1"
  local tag="$2"
  local out

  out="$(cd "${ROOT_DIR}" && ray job submit --address "${RAY_DASHBOARD}" --working-dir . --no-wait -- bash -lc "set -xeuo pipefail
unset PYTORCH_CUDA_ALLOC_CONF
export RAY_ADDRESS=auto
export SWANLAB_API_KEY=${SWANLAB_API_KEY:?set_SWANLAB_API_KEY}
export PROJECT_NAME=lzn_verl
export EXP_MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B
export EXP_VERSION=dpsk_distill_seq_fixedn8_fixprompt_16krollout_32keval_total16
export PERF_TAG=${tag}
export MODEL_PATH=/apdcephfs/share_302503006/ziniuli/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export TRAIN_FILE=/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/train.parquet
export TEST_FILE='[/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/aime24.parquet,/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/aime25.parquet,/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/amc_fixed.parquet,/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/math_100.parquet,/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/minerva_fixed_100.parquet,/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/olympiad_bench_fixed_100.parquet,/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler/gpqa_diamond_100.parquet]'
export CKPTS_DIR=/apdcephfs/share_302503006/ziniuli/checkpoint/knapsack_async_dpsk_distill
export RAY_DATA_HOME=/apdcephfs/share_302503006/ziniuli/processed_dataset/deepscaler
export REWARD_FN_PATH=verl/experimental/fully_async_policy/utils/reward_fn.py
export SAMPLE_STRATEGY=null
export STOP_RULE=fixed_rollout
export STOP_RULE_MIN_N=1
export STOP_RULE_MAX_N=8
export TRAIN_TRIGGER=fixed_prompt
export N_RESP_PER_PROMPT=8
export BATCH_SIZE=${batch_size}
export TRAIN_PROMPT_MINI_BSZ=32
export ACTOR_PPO_MAX_TOKEN_MULT=2
export INFER_PPO_MAX_TOKEN_MULT=2
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=16384
export VAL_MAX_RESPONSE_LENGTH=32768
export VAL_BEFORE_TRAIN=False
export NNODES_ROLLOUT=1
export NNODES_TRAIN=1
export N_GPUS_ROLLOUT=8
export N_GPUS_TRAINING=8
export NGPUS_PER_NODE=8
export TEST_FREQ=50
export SAVE_FREQ=50
bash verl/experimental/fully_async_policy/shell/dapo_1.5b_math_fsdp2_4_4.sh" 2>&1)"

  printf '%s\n' "${out}" >> "${LOG_FILE}"
  grep -o "raysubmit_[A-Za-z0-9]*" <<<"${out}" | tail -1
}

log "submitter_started min_available_gpus=${MIN_AVAILABLE_GPUS} interval=${CHECK_INTERVAL_SECONDS}s"

while true; do
  avail="$(available_gpus)"
  log "available_gpus=${avail}"
  if python3 - "${avail}" "${MIN_AVAILABLE_GPUS}" <<'PY'
import sys
avail = float(sys.argv[1])
need = float(sys.argv[2])
raise SystemExit(0 if avail >= need else 1)
PY
  then
    bs32_job="$(submit_one 32 bs32_total16_tokenmult2_minibsz32_noallocconf)"
    log "submitted bs32_total16 job=${bs32_job}"
    sleep 20
    bs64_job="$(submit_one 64 bs64_total16_tokenmult2_minibsz32_noallocconf)"
    log "submitted bs64_total16 job=${bs64_job}"
    log "submitter_done"
    exit 0
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done
