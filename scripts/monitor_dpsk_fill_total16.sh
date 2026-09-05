#!/usr/bin/env bash
set -euo pipefail

export RAY_ADDRESS="${RAY_ADDRESS:-http://29.232.242.135:8081}"

JOB_ID="${JOB_ID:-raysubmit_MmXCVayXf7rX2y8d}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
LOG_DIR="${LOG_DIR:-monitor_logs/dpsk_fill_total16}"
HEARTBEAT_LOG="${LOG_DIR}/heartbeat.log"
ALERT_LOG="${LOG_DIR}/alerts.log"

mkdir -p "${LOG_DIR}"

while true; do
  ts="$(date '+%F %T %Z')"
  {
    echo "===== ${ts} ====="
    echo "[job_status]"
    ray job status "${JOB_ID}" || true
    echo
    echo "[cluster_gpu]"
    curl -sf "${RAY_ADDRESS}/api/cluster_status" | python3 -c 'import json,sys,re
d=json.load(sys.stdin)
s=d["data"]["autoscalingStatus"]
m=re.search(r"ResourceUsage: ([^\n]+)", s)
u=m.group(1) if m else s
g=re.search(r"([0-9.]+/[0-9.]+ GPU)", u)
print(g.group(1) if g else u[:500])' || true
    echo
    echo "[recent_signals]"
    ray job logs "${JOB_ID}" \
      | rg "Hi X4|View run|FullyAsyncTrainer created|Rollouter created|All components initialized|global_step|CUDA out of memory|OutOfMemory|Traceback|ActorDiedError|TaskCancelled|Forbidden|failed|ERROR" \
      | tail -80 || true
    echo
  } >> "${HEARTBEAT_LOG}" 2>&1

  if tail -300 "${HEARTBEAT_LOG}" | rg -q "CUDA out of memory|OutOfMemory|Traceback|ActorDiedError|TaskCancelled|Forbidden|Job .* failed|Status for job .*: FAILED"; then
    {
      echo "===== ${ts} ====="
      tail -300 "${HEARTBEAT_LOG}"
      echo
    } >> "${ALERT_LOG}"
  fi

  sleep "${INTERVAL_SECONDS}"
done
