#!/usr/bin/env bash
set -uo pipefail

RAY_DASHBOARD="${RAY_DASHBOARD:-http://29.232.242.135:8081}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
LOG_DIR="${LOG_DIR:-/apdcephfs_hldy/share_303922427/pencywang/project/knapsack_async/monitor_logs/total16_probe}"
HEARTBEAT_LOG="${HEARTBEAT_LOG:-${LOG_DIR}/heartbeat.log}"
ALERT_LOG="${ALERT_LOG:-${LOG_DIR}/alerts.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/monitor.pid}"

JOBS=(
  "qwen25_has4_positive_chunk2_4plus4:raysubmit_YRFK4KikJgYYTD21"
)

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

  python - "${tmp}" >> "${HEARTBEAT_LOG}" 2>> "${ALERT_LOG}" <<'PY'
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
else:
    used = float(m.group(1))
    total = float(m.group(2))
    print(f"cluster_gpu_usage: used={used:g} total={total:g}")
    if total < 32:
        print(f"ALERT cluster_gpu_total_below_expected total={total:g}", file=sys.stderr)
PY
  rm -f "${tmp}"
}

summarize_job() {
  local label="$1"
  local job_id="$2"
  local status_out status_line log_file progress_lines error_lines

  status_out="$(ray job status --address "${RAY_DASHBOARD}" "${job_id}" 2>&1)"
  printf '[%s] job_status label=%s job=%s\n%s\n' "$(ts)" "${label}" "${job_id}" "${status_out}" >> "${HEARTBEAT_LOG}"

  status_line="$(printf '%s\n' "${status_out}" | grep -E "Status for job|Job .* was" | tail -1 || true)"
  if printf '%s\n' "${status_out}" | grep -q "FAILED"; then
    alert "job_failed label=${label} job=${job_id} status=${status_line:-unknown}"
  elif printf '%s\n' "${status_out}" | grep -q "STOPPED"; then
    alert "job_stopped label=${label} job=${job_id} status=${status_line:-unknown}"
  elif printf '%s\n' "${status_out}" | grep -q "SUCCEEDED"; then
    alert "job_succeeded label=${label} job=${job_id} status=${status_line:-succeeded}"
  elif ! printf '%s\n' "${status_out}" | grep -q "RUNNING"; then
    alert "job_status_unknown label=${label} job=${job_id} status=${status_line:-unknown}"
  fi

  log_file="${LOG_DIR}/${label}_${job_id}.current.log"
  if timeout 240 ray job logs --address "${RAY_DASHBOARD}" "${job_id}" > "${log_file}" 2>&1; then
    progress_lines="$(
      grep -aoE "FullyAsyncTrainer created and initialized successfully|Rollouter created and initialized successfully|All components initialized successfully|Starting Rollouter and Trainer|Starting FullyAsyncTrainer|Starting FullyAsyncRollouter|Training Progress: *[0-9]+%|step:[0-9]+|global_steps[: ]+[0-9]+|training/global_step:[0-9.]+|total_generated_samples'?: ?[0-9.]+|required_samples: [0-9]+|required_prompts: [0-9]+|validation generation end|_fit_update_weights|Param sync before fit|perf/mfu/actor[^ ]*:[0-9.]+" "${log_file}" \
        | tail -40 || true
    )"
    {
      printf '[%s] job_progress label=%s job=%s log=%s\n' "$(ts)" "${label}" "${job_id}" "${log_file}"
      if [[ -n "${progress_lines}" ]]; then
        printf '%s\n' "${progress_lines}"
      else
        printf 'progress_keywords: none\n'
      fi
    } >> "${HEARTBEAT_LOG}"

    error_lines="$(
      grep -Ei "Traceback|RayTaskError|TaskCancelledError|CUDA out of memory|OutOfMemoryError|out of memory|FileNotFoundError|No such file|HFValidationError|RuntimeError|AssertionError|ValueError|Error executing job|failed with exit code|worker died|WorkerCrashed|ActorDiedError|Exception" "${log_file}" \
        | grep -Ev "FutureWarning|NCCL_DEBUG|nccl_timeout|checkpoint_engine|backend.: .nccl|RequestsDependencyWarning" \
        | tail -80 || true
    )"
    if [[ -n "${error_lines}" ]]; then
      alert "job_log_error_keyword label=${label} job=${job_id} log=${log_file}"
      printf '%s\n' "${error_lines}" >> "${ALERT_LOG}"
    fi
  else
    alert "job_logs_timeout_or_failed label=${label} job=${job_id} log=${log_file}"
    tail -120 "${log_file}" >> "${ALERT_LOG}" 2>/dev/null || true
  fi
}

echo "$$" > "${PID_FILE}"
log "total16_probe_started pid=$$ interval=${INTERVAL_SECONDS}s dashboard=${RAY_DASHBOARD}"

while true; do
  log "heartbeat_check_begin"
  summarize_cluster
  for item in "${JOBS[@]}"; do
    summarize_job "${item%%:*}" "${item#*:}"
  done
  log "heartbeat_check_end"
  sleep "${INTERVAL_SECONDS}"
done
