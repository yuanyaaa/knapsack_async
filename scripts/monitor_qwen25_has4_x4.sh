#!/usr/bin/env bash
set -uo pipefail

RAY_DASHBOARD="${RAY_DASHBOARD:-http://29.232.242.135:8081}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1200}"
LOG_DIR="${LOG_DIR:-/apdcephfs_hldy/share_303922427/pencywang/project/knapsack_async/monitor_logs/qwen25_has4_x4}"
HEARTBEAT_LOG="${HEARTBEAT_LOG:-${LOG_DIR}/heartbeat.log}"
ALERT_LOG="${ALERT_LOG:-${LOG_DIR}/alerts.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/monitor.pid}"

JOB_LABEL="${JOB_LABEL:-qwen25_math_has4_positive_chunk2_x4}"
JOB_ID="${JOB_ID:-raysubmit_LrJHRGMBr8tYDHrw}"

mkdir -p "${LOG_DIR}"

ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log() {
  printf '[%s] %s\n' "$(ts)" "$*" >> "${HEARTBEAT_LOG}"
}

alert() {
  local msg="$*"
  printf '[%s] ALERT %s\n' "$(ts)" "${msg}" >> "${ALERT_LOG}"
  printf '[%s] ALERT %s\n' "$(ts)" "${msg}" >> "${HEARTBEAT_LOG}"
}

summarize_cluster() {
  local tmp
  tmp="$(mktemp)"
  if ! curl -fsS "${RAY_DASHBOARD}/api/cluster_status" > "${tmp}" 2>>"${HEARTBEAT_LOG}"; then
    alert "cluster_status_api_failed dashboard=${RAY_DASHBOARD}"
    rm -f "${tmp}"
    return
  fi

  python - "${tmp}" >> "${HEARTBEAT_LOG}" 2>>"${ALERT_LOG}" <<'PY'
import json
import re
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)

status = (payload.get("data") or {}).get("autoscalingStatus", "")
resource_line = ""
for line in status.splitlines():
    if "ResourceUsage:" in line:
        resource_line = line.strip()
        break

print("cluster_status:", resource_line or "ResourceUsage line not found")
m = re.search(r"([0-9.]+)/([0-9.]+)\s+GPU", resource_line)
if m:
    print(f"cluster_gpu_usage: used={float(m.group(1)):g} total={float(m.group(2)):g}")
else:
    print("ALERT cluster_gpu_usage_not_found", file=sys.stderr)
PY
  rm -f "${tmp}"
}

summarize_job() {
  local status_out status_line log_file progress_lines error_lines

  status_out="$(ray job status --address "${RAY_DASHBOARD}" "${JOB_ID}" 2>&1)"
  printf '[%s] job_status label=%s job=%s\n%s\n' "$(ts)" "${JOB_LABEL}" "${JOB_ID}" "${status_out}" >> "${HEARTBEAT_LOG}"

  status_line="$(printf '%s\n' "${status_out}" | grep -E "Status for job|Job .* was|Job .* failed" | tail -1 || true)"
  if printf '%s\n' "${status_out}" | grep -q "FAILED"; then
    alert "job_failed label=${JOB_LABEL} job=${JOB_ID} status=${status_line:-unknown}"
  elif printf '%s\n' "${status_out}" | grep -q "STOPPED"; then
    alert "job_stopped label=${JOB_LABEL} job=${JOB_ID} status=${status_line:-unknown}"
  elif printf '%s\n' "${status_out}" | grep -q "SUCCEEDED"; then
    alert "job_succeeded label=${JOB_LABEL} job=${JOB_ID} status=${status_line:-succeeded}"
  elif ! printf '%s\n' "${status_out}" | grep -q "RUNNING"; then
    alert "job_status_unknown label=${JOB_LABEL} job=${JOB_ID} status=${status_line:-unknown}"
  fi

  log_file="${LOG_DIR}/${JOB_LABEL}_${JOB_ID}.current.log"
  if timeout 300 ray job logs --address "${RAY_DASHBOARD}" "${JOB_ID}" > "${log_file}" 2>&1; then
    progress_lines="$(
      grep -aoE "Hi X4|View project at|@X4/wangpengyuan_verl|rollout_chunk_size': 2|rollout_chunk_size=2|min_rollout_n': 4|stop_rule': 'has_at_least_positive'|FullyAsyncTrainer created and initialized successfully|Rollouter created and initialized successfully|All components initialized successfully|Starting Rollouter and Trainer|Starting FullyAsyncTrainer|Starting FullyAsyncRollouter|Training Progress: *[0-9]+%|global_steps[: ]+[0-9]+|step:[0-9]+|training/global_step:[0-9.]+|total_generated_samples'?: ?[0-9.]+|required_samples: [0-9]+|required_prompts: [0-9]+|_fit_update_weights|Param sync before fit|fully_async/has_k/[A-Za-z0-9_./:-]+" "${log_file}" \
        | tail -80 || true
    )"
    {
      printf '[%s] job_progress label=%s job=%s log=%s\n' "$(ts)" "${JOB_LABEL}" "${JOB_ID}" "${log_file}"
      if [[ -n "${progress_lines}" ]]; then
        printf '%s\n' "${progress_lines}"
      else
        printf 'progress_keywords: none\n'
      fi
    } >> "${HEARTBEAT_LOG}"

    error_lines="$(
      grep -Ei "Traceback|RayTaskError|TaskCancelledError|CUDA out of memory|OutOfMemoryError|out of memory|FileNotFoundError|No such file|HFValidationError|RuntimeError|AssertionError|ValueError|Error executing job|failed with exit code|worker died|WorkerCrashed|ActorDiedError|Exception|Forbidden|ApiError" "${log_file}" \
        | grep -Ev "FutureWarning|NCCL_DEBUG|nccl_timeout|checkpoint_engine|backend.: .nccl|RequestsDependencyWarning" \
        | tail -100 || true
    )"
    if [[ -n "${error_lines}" ]]; then
      alert "job_log_error_keyword label=${JOB_LABEL} job=${JOB_ID} log=${log_file}"
      printf '%s\n' "${error_lines}" >> "${ALERT_LOG}"
    fi
  else
    alert "job_logs_timeout_or_failed label=${JOB_LABEL} job=${JOB_ID} log=${log_file}"
    tail -120 "${log_file}" >> "${ALERT_LOG}" 2>/dev/null || true
  fi
}

echo "$$" > "${PID_FILE}"
log "qwen25_has4_x4_monitor_started pid=$$ interval=${INTERVAL_SECONDS}s dashboard=${RAY_DASHBOARD} job=${JOB_ID}"

while true; do
  log "heartbeat_check_begin"
  summarize_cluster
  summarize_job
  log "heartbeat_check_end"
  sleep "${INTERVAL_SECONDS}"
done
