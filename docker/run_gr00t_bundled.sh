#!/usr/bin/env bash
# Run a bundled GR00T inference image (model baked in) for the bi-SO101 setup.
# Intended for recipients — no model path or compose file needed.
#
# Usage:
#   ./run_gr00t_bundled.sh [options]
#
# Options:
#   --image NAME[:TAG]     Image to run (default: gr00t-inference-bundled:bi-so101)
#   --port PORT            Host port to expose ZMQ on (default: 5555)
#   --embodiment TAG       Override embodiment tag baked into the image.
#   --device DEV           Override torch device (default: cuda).
#   --name NAME            Container name (default: gr00t_server).
#   --detach               Run in the background.
#   --shell                Drop into a bash shell instead of starting the server.
#   -h | --help            Show this help.
#
# Prerequisites on the host:
#   - Docker 20.10+
#   - NVIDIA GPU with up-to-date driver
#   - NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
#
# After the server starts, IsaacSim on this host connects to localhost:<port>.

set -euo pipefail

IMAGE="gr00t-inference-bundled:bi-so101"
HOST_PORT="5555"
EMBODIMENT=""
DEVICE=""
CONTAINER_NAME="gr00t_server"
DETACH=0
SHELL_MODE=0

usage() { sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)       IMAGE="$2"; shift 2 ;;
        --port)        HOST_PORT="$2"; shift 2 ;;
        --embodiment)  EMBODIMENT="$2"; shift 2 ;;
        --device)      DEVICE="$2"; shift 2 ;;
        --name)        CONTAINER_NAME="$2"; shift 2 ;;
        --detach|-d)   DETACH=1; shift ;;
        --shell)       SHELL_MODE=1; shift ;;
        -h|--help)     usage 0 ;;
        *)             echo "Unknown arg: $1" >&2; usage 1 ;;
    esac
done

# ── Pre-flight checks ──────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on PATH." >&2; exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: cannot talk to Docker daemon (is it running? do you need sudo?)." >&2; exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' not found locally." >&2
    echo "       Pull it (docker pull $IMAGE) or load it (docker load -i <file>.tar) first." >&2
    exit 1
fi
if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    echo "WARNING: nvidia container runtime not detected. GPU access may fail." >&2
    echo "         Install nvidia-container-toolkit and restart Docker." >&2
fi

# Stop a stale container with the same name.
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "[run] removing existing container '$CONTAINER_NAME'..."
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# ── Assemble docker args ───────────────────────────────────────────────────────
RUN_ARGS=(
    --rm
    --name "$CONTAINER_NAME"
    --gpus all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -p "${HOST_PORT}:5555"
)

[[ -n "$EMBODIMENT" ]] && RUN_ARGS+=(-e "GR00T_EMBODIMENT_TAG=$EMBODIMENT")
[[ -n "$DEVICE" ]]     && RUN_ARGS+=(-e "GR00T_DEVICE=$DEVICE")

if [[ $DETACH -eq 1 ]]; then
    RUN_ARGS+=(-d)
else
    RUN_ARGS+=(-it)
fi

if [[ $SHELL_MODE -eq 1 ]]; then
    echo "[run] launching shell in $IMAGE"
    exec docker run "${RUN_ARGS[@]}" --entrypoint /bin/bash "$IMAGE"
fi

cat <<EOF
[run] starting GR00T inference server
        image     : $IMAGE
        container : $CONTAINER_NAME
        host port : $HOST_PORT  (clients connect to localhost:$HOST_PORT)
EOF

exec docker run "${RUN_ARGS[@]}" "$IMAGE"
