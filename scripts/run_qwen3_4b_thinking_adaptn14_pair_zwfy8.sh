#!/usr/bin/env bash
set -euo pipefail

: "${HEAD_IP:?set HEAD_IP to this task's 8-GPU head node}"
: "${WORKER_IP:?set WORKER_IP to this task's 8-GPU worker node}"
export ARM=adaptn14
export SUBMIT_SCRIPT=scripts/submit_qwen3_4b_thinking_six_arms_zwfy8.sh
export CONDA_ENV="${CONDA_ENV:-knapsack_async}"

exec bash /apdcephfs_zwfy8/share_303922427/pencywang/project/knapsack_async/scripts/run_qwen3_4b_pair_zwfy8.sh
