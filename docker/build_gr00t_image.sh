#!/usr/bin/env bash
# Build (and optionally export/push) a GR00T inference Docker image with the
# trained checkpoint baked in. Produces an image that the recipient runs with
# nothing but the image itself + an NVIDIA GPU.
#
# Usage:
#   ./build_gr00t_image.sh --model PATH [options]
#
# Required:
#   --model PATH           Host path to the trained checkpoint directory.
#
# Optional:
#   --tag NAME[:TAG]       Image tag (default: gr00t-inference-bundled:bi-so101)
#   --embodiment TAG       Embodiment tag baked as default (default: new_embodiment)
#   --groot-ref REF        Isaac-GR00T git ref to clone (default: main)
#   --save FILE.tar.gz     After build, export image as a gzipped tarball.
#   --push                 After build, docker push the tag (must be a registry-qualified tag).
#   --no-cache             Pass --no-cache to docker build.
#   -h | --help            Show this help.
#
# Examples:
#   # Build only.
#   ./build_gr00t_image.sh --model /data/checkpoints/bi_so101_v3
#
#   # Build + save tarball for offline delivery.
#   ./build_gr00t_image.sh \
#       --model /data/checkpoints/bi_so101_v3 \
#       --tag   bi-so101-gr00t:v3 \
#       --save  ./bi-so101-gr00t-v3.tar.gz
#
#   # Build + push to a registry.
#   ./build_gr00t_image.sh \
#       --model /data/checkpoints/bi_so101_v3 \
#       --tag   ghcr.io/myorg/bi-so101-gr00t:v3 \
#       --push

set -euo pipefail

# ── Defaults ───────────────────────────────────────────────────────────────────
MODEL_PATH=""
IMAGE_TAG="gr00t-inference-bundled:bi-so101"
EMBODIMENT_TAG="new_embodiment"
GROOT_REF="main"
SAVE_PATH=""
DO_PUSH=0
NO_CACHE=""

# ── Resolve repo paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd )"
DOCKERFILE="$SCRIPT_DIR/Dockerfile.gr00t-bundled"

usage() {
    sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# ── Parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL_PATH="$2"; shift 2 ;;
        --tag)         IMAGE_TAG="$2"; shift 2 ;;
        --embodiment)  EMBODIMENT_TAG="$2"; shift 2 ;;
        --groot-ref)   GROOT_REF="$2"; shift 2 ;;
        --save)        SAVE_PATH="$2"; shift 2 ;;
        --push)        DO_PUSH=1; shift ;;
        --no-cache)    NO_CACHE="--no-cache"; shift ;;
        -h|--help)     usage 0 ;;
        *)             echo "Unknown arg: $1" >&2; usage 1 ;;
    esac
done

# ── Validate ───────────────────────────────────────────────────────────────────
if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: --model is required." >&2
    usage 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: --model path '$MODEL_PATH' is not a directory." >&2
    exit 1
fi
MODEL_PATH="$( cd "$MODEL_PATH" >/dev/null 2>&1 && pwd )"  # absolute

if [[ ! -f "$DOCKERFILE" ]]; then
    echo "ERROR: Dockerfile not found at $DOCKERFILE" >&2
    exit 1
fi

# Sanity-check the checkpoint dir is plausibly a HF/GR00T checkpoint.
if ! ls "$MODEL_PATH"/*.safetensors "$MODEL_PATH"/*.bin "$MODEL_PATH"/config.json 2>/dev/null | head -n1 >/dev/null; then
    echo "WARNING: '$MODEL_PATH' has no .safetensors / .bin / config.json — make sure this is the right directory." >&2
fi

MODEL_SIZE_HUMAN="$( du -sh "$MODEL_PATH" 2>/dev/null | awk '{print $1}' )"

cat <<EOF

GR00T inference image build
  Image tag      : $IMAGE_TAG
  Model path     : $MODEL_PATH ($MODEL_SIZE_HUMAN)
  Embodiment tag : $EMBODIMENT_TAG
  Isaac-GR00T ref: $GROOT_REF
  Save tarball   : ${SAVE_PATH:-<none>}
  Push to registry: $([[ $DO_PUSH -eq 1 ]] && echo yes || echo no)

EOF

# ── Build ──────────────────────────────────────────────────────────────────────
export DOCKER_BUILDKIT=1

docker build $NO_CACHE \
    --network host \
    --build-context "model=$MODEL_PATH" \
    --build-arg "ISAAC_GROOT_REF=$GROOT_REF" \
    --build-arg "GR00T_EMBODIMENT_TAG=$EMBODIMENT_TAG" \
    -f "$DOCKERFILE" \
    -t "$IMAGE_TAG" \
    "$REPO_ROOT"

echo
echo "Built image: $IMAGE_TAG"
docker images --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}' | head -n1
docker images "$IMAGE_TAG" --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}' | tail -n+2

# ── Optional: save as tarball ──────────────────────────────────────────────────
if [[ -n "$SAVE_PATH" ]]; then
    echo
    echo "Saving image to $SAVE_PATH ..."
    mkdir -p "$( dirname "$SAVE_PATH" )"
    case "$SAVE_PATH" in
        *.tar.gz|*.tgz) docker save "$IMAGE_TAG" | gzip > "$SAVE_PATH" ;;
        *.tar)          docker save "$IMAGE_TAG" -o "$SAVE_PATH" ;;
        *)              echo "ERROR: --save must end in .tar, .tar.gz, or .tgz" >&2; exit 1 ;;
    esac
    SAVE_SIZE="$( du -h "$SAVE_PATH" | awk '{print $1}' )"
    echo "Saved $SAVE_SIZE → $SAVE_PATH"
    cat <<EOF

Recipient runs:
  gunzip -c $( basename "$SAVE_PATH" ) | docker load     # if .tar.gz
  docker load -i $( basename "$SAVE_PATH" )              # if .tar
  docker run --rm --gpus all --ipc=host \\
      --ulimit memlock=-1 --ulimit stack=67108864 \\
      -p 5555:5555 \\
      $IMAGE_TAG
EOF
fi

# ── Optional: push ─────────────────────────────────────────────────────────────
if [[ $DO_PUSH -eq 1 ]]; then
    echo
    echo "Pushing $IMAGE_TAG ..."
    docker push "$IMAGE_TAG"
    cat <<EOF

Recipient runs:
  docker pull $IMAGE_TAG
  docker run --rm --gpus all --ipc=host \\
      --ulimit memlock=-1 --ulimit stack=67108864 \\
      -p 5555:5555 \\
      $IMAGE_TAG
EOF
fi

echo
echo "Done."
