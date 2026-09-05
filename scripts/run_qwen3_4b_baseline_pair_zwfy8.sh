#!/usr/bin/env bash
set -euo pipefail

export HEAD_IP=28.83.1.158
export WORKER_IP=28.85.98.167
export ARM=baseline
export CONDA_ENV="${CONDA_ENV:-knapsack_async}"

exec bash /apdcephfs_zwfy8/share_303922427/pencywang/project/knapsack_async/scripts/run_qwen3_4b_pair_zwfy8.sh
