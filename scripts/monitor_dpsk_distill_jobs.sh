#!/usr/bin/env bash
set -uo pipefail

RAY_DASHBOARD="${RAY_DASHBOARD:-http://29.232.242.135:8081}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
LOG_DIR="${LOG_DIR:-/apdcephfs_hldy/share_303922427/pencywang/project/knapsack_async/monitor_logs}"
HEARTBEAT_LOG="${HEARTBEAT_LOG:-${LOG_DIR}/heartbeat.log}"
ALERT_LOG="${ALERT_LOG:-${LOG_DIR}/heartbeat_alerts.log}"

JOBS=(
  "bs32:raysubmit_5RH4bGtN2p9x9fN5"
  "bs64:raysubmit_uFvXCjhBGVfuSNcL"
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

check_cluster() {
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
if m:
    used = float(m.group(1))
    total = float(m.group(2))
    print(f"cluster_gpu_usage: used={used:g} total={total:g}")
    if total < 32:
        print(f"ALERT cluster_gpu_total_below_expected total={total:g}", file=sys.stderr)
else:
    print("ALERT cluster_gpu_usage_not_found", file=sys.stderr)
PY
  rm -f "${tmp}"
}

check_job() {
  local label="$1"
  local job_id="$2"
  local status_out status_line log_tmp latest_step latest_samples error_lines

  status_out="$(ray job status --address "${RAY_DASHBOARD}" "${job_id}" 2>&1)"
  printf '[%s] job_status label=%s job=%s\n%s\n' "$(ts)" "${label}" "${job_id}" "${status_out}" >> "${HEARTBEAT_LOG}"

  status_line="$(printf '%s\n' "${status_out}" | grep -E "Status for job" | tail -1 || true)"
  if printf '%s\n' "${status_out}" | grep -qE "failed|STOPPED|FAILED"; then
    alert "job_not_running label=${label} job=${job_id} status=${status_line:-unknown}"
  elif printf '%s\n' "${status_out}" | grep -q "SUCCEEDED"; then
    alert "job_finished label=${label} job=${job_id} status=${status_line:-succeeded}"
  elif ! printf '%s\n' "${status_out}" | grep -q "RUNNING"; then
    alert "job_status_unknown label=${label} job=${job_id} status=${status_line:-unknown}"
  fi

  log_tmp="$(mktemp)"
  if timeout 180 ray job logs --address "${RAY_DASHBOARD}" "${job_id}" > "${log_tmp}" 2>&1; then
    latest_step="$(grep -aoE 'training/global_step:[0-9.]+' "${log_tmp}" | tail -1 || true)"
    latest_samples="$(grep -aoE "total_generated_samples'?: ?[0-9.]+" "${log_tmp}" | tail -1 || true)"
    printf '[%s] job_progress label=%s job=%s %s %s\n' \
      "$(ts)" "${label}" "${job_id}" "${latest_step:-training/global_step:NA}" "${latest_samples:-total_generated_samples:NA}" \
      >> "${HEARTBEAT_LOG}"

    error_lines="$(grep -Ei 'Traceback|(^|[^A-Za-z])ERROR([^A-Za-z]|$)|Error executing job|RayTaskError|CUDA out of memory|out of memory|HFValidationError|FileNotFound|No such file|RuntimeError|AssertionError|Exception|failed with exit code' "${log_tmp}" | grep -Ev 'FutureWarning|NCCL_DEBUG|nccl_timeout|checkpoint_engine|backend.: .nccl' | tail -20 || true)"
    if [[ -n "${error_lines}" ]]; then
      alert "job_log_error_keyword label=${label} job=${job_id}"
      printf '%s\n' "${error_lines}" >> "${ALERT_LOG}"
    fi
  else
    alert "job_logs_timeout_or_failed label=${label} job=${job_id}"
    tail -80 "${log_tmp}" >> "${ALERT_LOG}" 2>/dev/null || true
  fi
  rm -f "${log_tmp}"
}

log "heartbeat_monitor_started pid=$$ interval=${INTERVAL_SECONDS}s dashboard=${RAY_DASHBOARD}"

while true; do
  log "heartbeat_check_begin"
  check_cluster
  for item in "${JOBS[@]}"; do
    label="${item%%:*}"
    job_id="${item#*:}"
    check_job "${label}" "${job_id}"
  done
  log "heartbeat_check_end"
  sleep "${INTERVAL_SECONDS}"
done
