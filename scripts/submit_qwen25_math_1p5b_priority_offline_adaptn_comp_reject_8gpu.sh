#!/usr/bin/env bash
# Same 8-GPU offline Adaptive-N arm as the paired script, with rejection sampling enabled.
set -euo pipefail

export REJECTION_SAMPLING=True
export ROLLOUT_BUDGET_MULT="${ROLLOUT_BUDGET_MULT:-2}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
export ARM_SUFFIX=reject

exec bash "$(dirname "${BASH_SOURCE[0]}")/submit_qwen25_math_1p5b_priority_offline_adaptn_comp_8gpu.sh"
