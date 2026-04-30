#!/bin/bash
set -euo pipefail

if [ -z "${GR00T_MODEL_PATH:-}" ]; then
    echo "[bridge] ERROR: GR00T_MODEL_PATH is not set." >&2
    exit 1
fi
if [ ! -d "${GR00T_MODEL_PATH}" ]; then
    echo "[bridge] ERROR: Model path '${GR00T_MODEL_PATH}' does not exist inside the container." >&2
    exit 1
fi

echo "[bridge] Starting GR00T HTTP policy server (in-process)"
echo "[bridge]   model_path        : ${GR00T_MODEL_PATH}"
echo "[bridge]   embodiment_tag    : ${GR00T_EMBODIMENT_TAG}"
echo "[bridge]   device            : ${GR00T_DEVICE}"
echo "[bridge]   http_port         : ${BRIDGE_HTTP_PORT}"
echo "[bridge]   task_description  : ${BRIDGE_TASK_DESCRIPTION}"

exec python /workspace/gr00t_http_bridge.py \
    --mode inproc \
    --model_path "${GR00T_MODEL_PATH}" \
    --embodiment "${GR00T_EMBODIMENT_TAG}" \
    --device "${GR00T_DEVICE}" \
    --http_host 0.0.0.0 \
    --http_port "${BRIDGE_HTTP_PORT}" \
    --task_description "${BRIDGE_TASK_DESCRIPTION}" \
    "$@"
