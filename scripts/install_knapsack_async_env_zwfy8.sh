#!/usr/bin/env bash
# Build a clean environment matching the first four zwfy8 nodes.
set -euo pipefail

ENV_NAME="${CONDA_ENV:-knapsack_async}"
ROOT=/apdcephfs_zwfy8/share_303922427/pencywang/project/knapsack_async

CONDA=/root/miniconda3/bin/conda
[[ -x "${CONDA}" ]] || { echo "missing ${CONDA}" >&2; exit 1; }
source /root/miniconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "environment already exists: ${ENV_NAME}"
else
  conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel packaging ninja

# These are the working versions on the first four H20 nodes.
python -m pip install \
  "vllm==0.8.5.post1" \
  "ray[default]==2.45.0"

cd "${ROOT}"
python -m pip install -e ".[math]"
python -m pip install "flash-attn==2.7.4.post1" --no-build-isolation

# Keep NumPy inside the project's declared range and install a protobuf runtime new enough
# for SwanLab's generated gRPC bindings.
python -m pip install \
  "numpy==1.26.4" \
  "protobuf==5.29.6" \
  "swanlab==0.7.20"

# setup.py allows ray[default]>=2.41, so the editable install can upgrade Ray past 2.45.
# Ray >=2.51 dashboard extras then pull a Prometheus exporter that imports
# OtelComponentTypeValues, which vLLM 0.8.5's OpenTelemetry 1.26 stack does not have.
python -m pip install \
  "click==8.2.1" \
  "ray[default]==2.45.0" \
  "opentelemetry-api==1.26.0" \
  "opentelemetry-sdk==1.26.0" \
  "opentelemetry-semantic-conventions==0.47b0" \
  "opentelemetry-exporter-prometheus==0.47b0"

# Fully-async weight sync registers backend "nccl" only if CuPy imports.
python -m pip install "fastrlock>=0.5"
python -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x || true
python -m pip install --no-deps --force-reinstall "cupy-cuda12x==13.6.0"

python - <<'PY'
import click
import cupy
import flash_attn
import numpy
import ray
import swanlab
import torch
import transformers
import vllm
import verl

from opentelemetry.exporter.prometheus import PrometheusMetricReader
from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine

print("environment ready")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("ray", ray.__version__)
print("click", click.__version__)
print("cupy", cupy.__version__)
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("flash_attn", flash_attn.__version__)
print("numpy", numpy.__version__)
print("prometheus_exporter", PrometheusMetricReader.__name__)
print("checkpoint_engines", sorted(CheckpointEngineRegistry._registry))
PY
