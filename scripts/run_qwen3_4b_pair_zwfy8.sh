#!/usr/bin/env bash
# Start a two-node Ray cluster the same way reval_async does, then run one Qwen3-4B arm
# on the head with python -m (not ray job submit).
set -euo pipefail

: "${HEAD_IP:?HEAD_IP is required}"
: "${WORKER_IP:?WORKER_IP is required}"
: "${ARM:?ARM is required}"
: "${SSH_PASSWORD:?SSH_PASSWORD is required for the worker}"

SSH_USER="${SSH_USER:-root}"
RAY_PORT="${RAY_PORT:-6487}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
CONDA_ENV="${CONDA_ENV:-knapsack_async}"
ROOT=/apdcephfs_zwfy8/share_303922427/pencywang/project/knapsack_async
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-scripts/submit_qwen3_4b_three_arms_zwfy8.sh}"
PYTHON=/root/miniconda3/envs/${CONDA_ENV}/bin/python
RAY=/root/miniconda3/envs/${CONDA_ENV}/bin/ray

# Never let an inherited Ray address or HTTP/gRPC proxy redirect cluster-internal
# GCS traffic. RAY_ADDRESS is exported again only after the new head is ready.
unset RAY_ADDRESS grpc_proxy GRPC_PROXY
cluster_no_proxy="127.0.0.1,localhost,${HEAD_IP},${WORKER_IP}"
export no_proxy="${no_proxy:+${no_proxy},}${cluster_no_proxy}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${cluster_no_proxy}"

ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-120}"

# Same IB/NIC settings as reval_async's zwfy8 TBRM launcher.
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_SL="${NCCL_IB_SL:-3}"
export NCCL_CHECK_DISABLE="${NCCL_CHECK_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_LL_THRESHOLD="${NCCL_LL_THRESHOLD:-16384}"
export NCCL_IB_CUDA_SUPPORT="${NCCL_IB_CUDA_SUPPORT:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-bond1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6}"
export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
export NCCL_IB_TC="${NCCL_IB_TC:-160}"
export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"
export HYDRA_FULL_ERROR=1

proxy_exports=""
for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; do
  if [[ -v "${proxy_var}" ]]; then
    printf -v proxy_value '%q' "${!proxy_var}"
    proxy_exports+="export ${proxy_var}=${proxy_value}; "
  fi
done

remote_bash() {
  local host="$1"
  local command="$2"
  SSH_TARGET_HOST="${host}" \
  SSH_TARGET_USER="${SSH_USER}" \
  SSH_TARGET_PASSWORD="${SSH_PASSWORD}" \
  SSH_TARGET_COMMAND="${command}" \
  python3 - <<'PY'
import os
import pty
import select
import shlex
import sys

host = os.environ["SSH_TARGET_HOST"]
user = os.environ["SSH_TARGET_USER"]
password = os.environ["SSH_TARGET_PASSWORD"]
command = os.environ["SSH_TARGET_COMMAND"]
argv = [
    "ssh",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=20",
    f"{user}@{host}",
    "bash -lc " + shlex.quote(command),
]

pid, fd = pty.fork()
if pid == 0:
    os.execvp("ssh", argv)

buffer = ""
sent_password = False
while True:
    ready, _, _ = select.select([fd], [], [], 180)
    if not ready:
        print(f"\nTimed out on {host}", file=sys.stderr)
        os.kill(pid, 9)
        sys.exit(124)
    try:
        chunk = os.read(fd, 4096)
    except OSError:
        chunk = b""
    if not chunk:
        break
    text = chunk.decode("utf-8", errors="replace")
    sys.stdout.write(text)
    sys.stdout.flush()
    buffer = (buffer + text)[-2000:].lower()
    if "yes/no" in buffer:
        os.write(fd, b"yes\n")
        buffer = ""
    elif "password:" in buffer and not sent_password:
        os.write(fd, (password + "\n").encode())
        sent_password = True
        buffer = ""

_, status = os.waitpid(pid, 0)
if os.WIFEXITED(status):
    sys.exit(os.WEXITSTATUS(status))
sys.exit(128 + os.WTERMSIG(status))
PY
}

local_preflight() {
  [[ -x "${PYTHON}" ]] || { echo "missing env ${CONDA_ENV} on head" >&2; exit 20; }
  [[ -x "${RAY}" ]] || { echo "missing ray in ${CONDA_ENV} on head" >&2; exit 20; }
  "${PYTHON}" -c 'from opentelemetry.exporter.prometheus import PrometheusMetricReader; import flash_attn, ray, torch, vllm, swanlab, verl; print("head versions", ray.__version__, torch.__version__, vllm.__version__, "flash_attn", flash_attn.__version__)'
  [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 8 ]]
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    awk '$1 > 1024 {bad=1} END {exit bad}' ||
    { echo "head has a GPU using over 1 GiB; refusing to start" >&2; exit 21; }
}

remote_prefix="set -euo pipefail; PYTHON=${PYTHON}; RAY=${RAY}; ROOT=${ROOT}; \
ulimit -n 1048576 2>/dev/null || true; \
unset RAY_ADDRESS grpc_proxy GRPC_PROXY; \
${proxy_exports} \
export no_proxy=${no_proxy}; export NO_PROXY=${NO_PROXY}; \
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s}; \
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}; \
export UCX_NET_DEVICES=${UCX_NET_DEVICES}; \
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}; \
test -x \${PYTHON}; test -x \${RAY}; cd \${ROOT}"

echo "[1/6] preflight head ${HEAD_IP}"
local_preflight

echo "[2/6] preflight worker ${WORKER_IP}"
remote_bash "${WORKER_IP}" "${remote_prefix}; \
\${PYTHON} -c 'from opentelemetry.exporter.prometheus import PrometheusMetricReader; import flash_attn, ray, torch, vllm, swanlab, verl; print(\"worker versions\", ray.__version__, torch.__version__, vllm.__version__, \"flash_attn\", flash_attn.__version__)'; \
test \"\$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)\" -eq 8; \
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '\$1 > 1024 {bad=1} END {exit bad}'"

echo "[3/6] start Ray head ${HEAD_IP}"
"${RAY}" stop --force || true
rm -rf /tmp/ray || true
"${RAY}" start \
  --head \
  --node-ip-address="${HEAD_IP}" \
  --port="${RAY_PORT}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --num-cpus="$(nproc)"

ready=0
for _ in $(seq 1 30); do
  if "${RAY}" status --address="${HEAD_IP}:${RAY_PORT}" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" -ne 1 ]]; then
  echo "Ray head failed to start" >&2
  "${RAY}" status --address="${HEAD_IP}:${RAY_PORT}" || true
  pgrep -af "gcs_server|raylet" || true
  for log in /tmp/ray/session_latest/logs/gcs_server.err /tmp/ray/session_latest/logs/gcs_server.out; do
    [[ -f "${log}" ]] && { echo "===== ${log} ====="; cat "${log}"; }
  done
  exit 1
fi
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

echo "[4/6] start Ray worker ${WORKER_IP}"
remote_bash "${WORKER_IP}" "${remote_prefix}; \
\${RAY} stop --force || true; \
rm -rf /tmp/ray || true; \
\${RAY} start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${WORKER_IP} --num-cpus=\$(nproc)"

echo "[5/6] wait for 16 GPUs"
cluster_ready=0
for _ in $(seq 1 60); do
  status="$("${RAY}" status --address="${RAY_ADDRESS}" 2>&1 || true)"
  total_gpus="$(printf '%s\n' "${status}" | "${PYTHON}" -c 'import re,sys; m=re.findall(r"/([0-9.]+) GPU",sys.stdin.read()); print(int(float(m[-1])) if m else 0)')"
  echo "Ray GPUs: ${total_gpus:-0}/16"
  if [[ "${total_gpus:-0}" -ge 16 ]]; then
    cluster_ready=1
    break
  fi
  sleep 5
done
if [[ "${cluster_ready}" -ne 1 ]]; then
  echo "${status:-}"
  echo "Ray cluster did not reach 16 GPUs" >&2
  exit 22
fi
echo "Ray ready: ${total_gpus}/16 GPUs"

echo "[6/6] launch ${ARM} on head with RAY_ADDRESS=${RAY_ADDRESS}"
cd "${ROOT}"
exec bash "${SUBMIT_SCRIPT}" "${ARM}"
