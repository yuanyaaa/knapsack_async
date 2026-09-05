#!/usr/bin/env bash
# Repair the Ray dashboard OpenTelemetry mismatch in an existing knapsack_async env.
# Run on every GPU node that will join the cluster.
set -euo pipefail

ENV_NAME="${CONDA_ENV:-knapsack_async}"
PYTHON="/root/miniconda3/envs/${ENV_NAME}/bin/python"
RAY="/root/miniconda3/envs/${ENV_NAME}/bin/ray"

[[ -x "${PYTHON}" ]] || { echo "missing env ${ENV_NAME}: ${PYTHON}" >&2; exit 20; }

# click 8.3.0 breaks Ray CLI deepcopy on Python 3.10 ("is not a valid Sentinel").
# Install click before invoking `ray`, otherwise `ray stop` itself crashes.
"${PYTHON}" -m pip install \
  "click==8.2.1" \
  "ray[default]==2.45.0" \
  "opentelemetry-api==1.26.0" \
  "opentelemetry-sdk==1.26.0" \
  "opentelemetry-semantic-conventions==0.47b0" \
  "opentelemetry-exporter-prometheus==0.47b0"

"${PYTHON}" -m pip install "numpy==1.26.4" "fastrlock>=0.5"
"${PYTHON}" -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x || true
"${PYTHON}" -m pip install --no-deps --force-reinstall "cupy-cuda12x==13.6.0"

"${RAY}" stop --force || true

"${PYTHON}" - <<'PY'
from opentelemetry.exporter.prometheus import PrometheusMetricReader
import click
import cupy
import ray
import vllm
from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine

assert ray.__version__.startswith("2.45."), ray.__version__
assert click.__version__ == "8.2.1", click.__version__
assert "nccl" in CheckpointEngineRegistry._registry, sorted(CheckpointEngineRegistry._registry)
print("fixed", "ray", ray.__version__, "click", click.__version__, "cupy", cupy.__version__, "vllm", vllm.__version__, "engines", sorted(CheckpointEngineRegistry._registry))
PY
"${RAY}" --version
