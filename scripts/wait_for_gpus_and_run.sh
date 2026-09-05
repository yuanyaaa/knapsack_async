#!/usr/bin/env bash
set -euo pipefail

required_gpus="${REQUIRED_GPUS:-8}"
check_interval_seconds="${CHECK_INTERVAL_SECONDS:-300}"
launch_stagger_seconds="${LAUNCH_STAGGER_SECONDS:-600}"
lock_wait_seconds="${LOCK_WAIT_SECONDS:-30}"
lock_file="${LOCK_FILE:-/tmp/knapsack_batchsweep_launch.lock}"

ts() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

log() {
  printf "[%s] %s\n" "$(ts)" "$*"
}

available_gpus() {
  python3 - <<'PY'
import ray

ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
print(float(ray.available_resources().get("GPU", 0.0)))
PY
}

if [[ $# -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

log "wait_wrapper_started required_gpus=${required_gpus} interval=${check_interval_seconds}s stagger=${launch_stagger_seconds}s command=$*"

while true; do
  exec 9>"${lock_file}"
  if flock -w "${lock_wait_seconds}" 9; then
    avail="$(available_gpus | tail -1)"
    log "available_gpus=${avail}"
    if python3 - "${avail}" "${required_gpus}" <<'PY'
import sys

avail = float(sys.argv[1])
need = float(sys.argv[2])
raise SystemExit(0 if avail >= need else 1)
PY
    then
      log "resources_ready start_command"
      "$@" &
      child=$!
      sleep "${launch_stagger_seconds}"
      flock -u 9
      log "launch_lock_released child_pid=${child}"
      wait "${child}"
      rc=$?
      log "command_finished rc=${rc}"
      exit "${rc}"
    fi
    flock -u 9
  else
    log "launch_lock_busy"
  fi
  sleep "${check_interval_seconds}"
done
