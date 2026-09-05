#!/usr/bin/env bash
set -uo pipefail

RAY_DASHBOARD="${RAY_DASHBOARD:-http://29.232.242.135:8081}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
JOB_LABEL="${JOB_LABEL:-bs32_32gpu_rollout16_train16}"
JOB_ID="${JOB_ID:-raysubmit_zL7FNGLEKWRZhiJa}"
LOG_DIR="${LOG_DIR:-/apdcephfs_hldy/share_303922427/pencywang/project/knapsack_async/monitor_logs/current_32gpu}"
HEARTBEAT_LOG="${HEARTBEAT_LOG:-${LOG_DIR}/heartbeat.log}"
ALERT_LOG="${ALERT_LOG:-${LOG_DIR}/alerts.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/monitor.pid}"

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
  local tmp used total
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

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

data = payload.get("data") or {}
status = data.get("autoscalingStatus", "")
resource_line = ""
for line in status.splitlines():
    if "ResourceUsage:" in line:
        resource_line = line.strip()
        break

print("cluster_status:", resource_line or "ResourceUsage line not found")
m = re.search(r"([0-9.]+)/([0-9.]+)\s+GPU", resource_line)
if not m:
    print("ALERT cluster_gpu_usage_not_found", file=sys.stderr)
    raise SystemExit(0)

used = float(m.group(1))
total = float(m.group(2))
print(f"cluster_gpu_usage: used={used:g} total={total:g}")
if total < 32:
    print(f"ALERT cluster_gpu_total_below_expected total={total:g}", file=sys.stderr)
elif used < 24:
    print(f"ALERT cluster_gpu_used_low used={used:g} total={total:g}", file=sys.stderr)
PY
  if grep -q "ALERT" "${ALERT_LOG}" 2>/dev/null; then
    :
  fi
  rm -f "${tmp}"
}

summarize_job() {
  local status_out status_line log_tmp progress_lines error_lines

  status_out="$(ray job status --address "${RAY_DASHBOARD}" "${JOB_ID}" 2>&1)"
  printf '[%s] job_status label=%s job=%s\n%s\n' "$(ts)" "${JOB_LABEL}" "${JOB_ID}" "${status_out}" >> "${HEARTBEAT_LOG}"

  status_line="$(printf '%s\n' "${status_out}" | grep -E "Status for job" | tail -1 || true)"
  if printf '%s\n' "${status_out}" | grep -q "FAILED"; then
    alert "job_failed label=${JOB_LABEL} job=${JOB_ID} status=${status_line:-unknown}"
  elif printf '%s\n' "${status_out}" | grep -q "STOPPED"; then
    alert "job_stopped label=${JOB_LABEL} job=${JOB_ID} status=${status_line:-unknown}"
  elif printf '%s\n' "${status_out}" | grep -q "SUCCEEDED"; then
    alert "job_succeeded label=${JOB_LABEL} job=${JOB_ID} status=${status_line:-succeeded}"
  elif ! printf '%s\n' "${status_out}" | grep -q "RUNNING"; then
    alert "job_status_unknown label=${JOB_LABEL} job=${JOB_ID} status=${status_line:-unknown}"
  fi

  log_tmp="$(mktemp)"
  if timeout 240 ray job logs --address "${RAY_DASHBOARD}" "${JOB_ID}" > "${log_tmp}" 2>&1; then
    progress_lines="$(
      grep -aoE "FullyAsyncTrainer created and initialized successfully|Rollouter created and initialized successfully|All components initialized successfully|Starting Rollouter and Trainer|Training Progress: *[0-9]+%|step:[0-9]+|global_steps[: ]+[0-9]+|training/global_step:[0-9.]+|total_generated_samples'?: ?[0-9.]+|required_samples: [0-9]+|required_prompts: [0-9]+|validation generation end|_fit_update_weights|Param sync before fit" "${log_tmp}" \
        | tail -30 || true
    )"
    {
      printf '[%s] job_progress label=%s job=%s\n' "$(ts)" "${JOB_LABEL}" "${JOB_ID}"
      if [[ -n "${progress_lines}" ]]; then
        printf '%s\n' "${progress_lines}"
      else
        printf 'progress_keywords: none\n'
      fi
    } >> "${HEARTBEAT_LOG}"

    error_lines="$(
      grep -Ei "Traceback|RayTaskError|TaskCancelledError|CUDA out of memory|out of memory|FileNotFoundError|No such file|HFValidationError|RuntimeError|AssertionError|Error executing job|failed with exit code|worker died|WorkerCrashed|ActorDiedError|Exception" "${log_tmp}" \
        | grep -Ev "FutureWarning|NCCL_DEBUG|nccl_timeout|checkpoint_engine|backend.: .nccl" \
        | tail -40 || true
    )"
    if [[ -n "${error_lines}" ]]; then
      alert "job_log_error_keyword label=${JOB_LABEL} job=${JOB_ID}"
      printf '%s\n' "${error_lines}" >> "${ALERT_LOG}"
    fi
  else
    alert "job_logs_timeout_or_failed label=${JOB_LABEL} job=${JOB_ID}"
    tail -120 "${log_tmp}" >> "${ALERT_LOG}" 2>/dev/null || true
  fi
  rm -f "${log_tmp}"
}

echo "$$" > "${PID_FILE}"
log "heartbeat_monitor_started pid=$$ interval=${INTERVAL_SECONDS}s dashboard=${RAY_DASHBOARD} job=${JOB_ID}"

while true; do
  log "heartbeat_check_begin"
  summarize_cluster
  summarize_job
  log "heartbeat_check_end"
  sleep "${INTERVAL_SECONDS}"
done
