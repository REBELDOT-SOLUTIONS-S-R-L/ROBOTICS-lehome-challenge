# Building the GR00T Inference Image (bi-SO101)

This guide is for the **publisher** — the person who has a trained GR00T 1.7
checkpoint for the bi-SO101 setup and wants to package it into a Docker image
that anyone with an NVIDIA GPU can run.

The image bundles:
- The GR00T 1.7 codebase (cloned from `NVIDIA/Isaac-GR00T`)
- All Python / CUDA dependencies (PyTorch 2.7 + CUDA 12.8, flash-attn, TensorRT…)
- Your trained checkpoint, baked in at `/model`

The recipient runs the image with no extra setup — see [GR00T_RUN.md](./GR00T_RUN.md).

---

## Prerequisites

On the build machine:

- Linux x86_64 (Apple Silicon / arm64 will work via emulation but is slow and
  produces a non-portable image)
- Docker 20.10+ with **BuildKit** (enabled by default on recent versions)
- ~30 GB free disk space for build cache + final image
- Network access to `github.com`, `pypi.org`, `download.pytorch.org`,
  `huggingface.co`
- An NVIDIA GPU is **not** required to build, only to run.

You do **not** need the Isaac-GR00T repo cloned locally — the Dockerfile
clones it itself.

---

## Files involved

| File | Purpose |
|---|---|
| `docker/Dockerfile.gr00t-bundled` | Image recipe with the model baked in |
| `docker/build_gr00t_image.sh` | Build wrapper — handles tagging, save, push |
| `docker/gr00t_entrypoint.sh` | Container entrypoint (already wired in) |

---

## Quick start

From the repo root:

```bash
./docker/build_gr00t_image.sh \
    --model /absolute/path/to/checkpoint \
    --tag bi-so101-gr00t:v1 \
    --embodiment new_embodiment
```

The `--model` path should be the **directory** containing your trained
checkpoint files (typically `config.json`, `*.safetensors` or `*.bin`,
`tokenizer.json`, the modality config, etc.). The script warns if it can't
find anything that looks like a checkpoint.

The `--embodiment` value must match the embodiment tag used during training.
For a custom bi-SO101 fine-tune this is most often `new_embodiment` (GR00T's
default for non-built-in embodiments).

---

## Distribution options

### A. Push to a container registry (easiest for the recipient)

Tag the image with a registry-qualified name and pass `--push`:

```bash
docker login ghcr.io     # or docker.io, your-ecr-url, etc.

./docker/build_gr00t_image.sh \
    --model /data/checkpoints/bi_so101_v1 \
    --tag ghcr.io/yourorg/bi-so101-gr00t:v1 \
    --push
```

Recipient: `docker pull ghcr.io/yourorg/bi-so101-gr00t:v1`.

### B. Export as a tarball (offline / air-gapped delivery)

```bash
./docker/build_gr00t_image.sh \
    --model /data/checkpoints/bi_so101_v1 \
    --tag bi-so101-gr00t:v1 \
    --save ./bi-so101-gr00t-v1.tar.gz
```

Send the resulting `.tar.gz` over SFTP / S3 / external drive. Recipient
loads with `gunzip -c file.tar.gz | docker load`.

Use `.tar` instead of `.tar.gz` if you'd rather keep the archive uncompressed
(model weights compress poorly anyway — gzip typically saves only 5-10%).

### C. Local only

Omit both `--save` and `--push`. The image stays in your local Docker daemon.
You can list it later with `docker images bi-so101-gr00t`.

---

## Build script reference

```
./docker/build_gr00t_image.sh --model PATH [options]

Required:
  --model PATH           Host path to the trained checkpoint directory.

Optional:
  --tag NAME[:TAG]       Image tag (default: gr00t-inference-bundled:bi-so101)
  --embodiment TAG       Default embodiment tag baked into the image
                         (default: new_embodiment). Still overridable at run time.
  --groot-ref REF        Isaac-GR00T git ref to clone (default: main).
                         Pin to a commit SHA for reproducible builds.
  --save FILE.tar[.gz]   Export image as a tarball after build.
  --push                 docker push the tag after build.
  --no-cache             Pass --no-cache to docker build.
```

---

## Pinning the GR00T version

`--groot-ref` defaults to `main`, which is fine for prototyping but **not
reproducible**. For a release build, pin to a specific commit:

```bash
./docker/build_gr00t_image.sh \
    --model /data/checkpoints/bi_so101_v1 \
    --tag bi-so101-gr00t:v1 \
    --groot-ref a1b2c3d4
```

Find the commit SHA your training run used (`git rev-parse HEAD` inside the
Isaac-GR00T checkout you trained from).

---

## Image size expectations

| Layer | Approx size |
|---|---|
| CUDA 12.8 base + system deps | ~6 GB |
| Python venv (torch, flash-attn, TRT, …) | ~9 GB |
| Isaac-GR00T source | <100 MB |
| Your checkpoint | typically 3–7 GB for a fine-tune |
| **Total uncompressed** | **~18–22 GB** |
| **Tarball (gzipped)** | **~16–20 GB** |

The bulk is CUDA + PyTorch + TensorRT, not the model. There's no easy way to
shrink that without a non-trivial slim-image rewrite.

---

## Things to double-check before publishing

- The checkpoint dir doesn't contain training-only artifacts you don't want
  to ship: `optimizer.pt`, `wandb/`, intermediate epoch checkpoints, dataset
  shards, etc. Prune before building.
- The embodiment tag matches training. Recipients can override at run time,
  but the default should be correct.
- License terms for your fine-tune are clear to recipients (the GR00T base
  is Apache 2.0; your checkpoint inherits whatever license your training
  data + base model allow).

---

## Troubleshooting

**`flash-attn` build fails or hangs**
The lockfile pulls a prebuilt wheel for `cu12 + torch 2.7 + cp310`. If the
download fails, check connectivity to `github.com/Dao-AILab/flash-attention`
and retry with `--no-cache`.

**`uv sync --frozen` complains about a missing aarch64 wheel**
You're building on arm64. The Dockerfile is intended for x86_64 deployment.
Build on a Linux x86_64 host (or use `docker buildx --platform linux/amd64`).

**Image is enormous**
Verify your `--model` path doesn't include training artifacts. `du -sh
$MODEL_PATH/*` will show the offenders.

**`docker push` fails with `unauthorized`**
You need to `docker login <registry>` first. For GHCR the token needs
`write:packages` scope.
