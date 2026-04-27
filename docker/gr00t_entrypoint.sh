#!/bin/bash
set -euo pipefail

# Validate required env var.
if [ -z "${GR00T_MODEL_PATH:-}" ]; then
    echo "[gr00t] ERROR: GR00T_MODEL_PATH is not set." >&2
    exit 1
fi

if [ ! -d "${GR00T_MODEL_PATH}" ]; then
    echo "[gr00t] ERROR: Model path '${GR00T_MODEL_PATH}' does not exist inside the container." >&2
    echo "[gr00t]        Mount your checkpoint with: -v /host/path/to/checkpoint:/model:ro" >&2
    exit 1
fi

echo "[gr00t] Starting GR00T inference server"
echo "[gr00t]   model_path    : ${GR00T_MODEL_PATH}"
echo "[gr00t]   embodiment_tag: ${GR00T_EMBODIMENT_TAG}"
echo "[gr00t]   device        : ${GR00T_DEVICE}"
echo "[gr00t]   port          : ${GR00T_PORT}"

exec python /workspace/gr00t/eval/run_gr00t_server.py \
    --model-path      "${GR00T_MODEL_PATH}" \
    --embodiment-tag  "${GR00T_EMBODIMENT_TAG}" \
    --device          "${GR00T_DEVICE}" \
    --port            "${GR00T_PORT}" \
    "$@"
