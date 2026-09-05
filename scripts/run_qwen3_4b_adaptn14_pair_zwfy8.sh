#!/usr/bin/env bash
set -euo pipefail

export HEAD_IP=28.85.44.237
export WORKER_IP=28.82.240.93
export ARM=adaptn14
export CONDA_ENV="${CONDA_ENV:-knapsack_async}"

exec bash /apdcephfs_zwfy8/share_303922427/pencywang/project/knapsack_async/scripts/run_qwen3_4b_pair_zwfy8.sh
