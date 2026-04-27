# Running the GR00T Inference Server (bi-SO101)

This guide is for the **recipient** — you have a bundled GR00T inference
image (either pulled from a registry or loaded from a tarball) and you want
to start the inference server so an IsaacSim client on this same host can
connect to it.

The image bundles the model and all dependencies. You don't need the
Isaac-GR00T source, the checkpoint, or any Python setup.

---

## Prerequisites

- Linux x86_64 (Ubuntu 22.04 or similar recommended)
- An NVIDIA GPU. Recommended ≥24 GB VRAM for GR00T 1.7 inference; less may
  work depending on batch size and precision.
- Up-to-date NVIDIA driver. Since the image ships CUDA 12.8, the driver
  must be **≥ 525** (CUDA forward-compatibility requires a recent driver).
  Check with `nvidia-smi`.
- Docker 20.10+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  installed and Docker restarted. Verify with:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
  ```

  If that prints a GPU table, you're good.

---

## 1. Get the image onto your machine

### From a registry

```bash
docker pull ghcr.io/yourorg/bi-so101-gr00t:v1
```

(Use the actual tag the publisher gave you.)

### From a tarball

```bash
# .tar.gz form
gunzip -c bi-so101-gr00t-v1.tar.gz | docker load

# .tar form
docker load -i bi-so101-gr00t-v1.tar
```

`docker load` prints the loaded image tag — note it for the next step.

Verify it's there:

```bash
docker images | grep gr00t
```

---

## 2. Start the server

### Option A — using the helper script (recommended)

If you have `docker/run_gr00t_bundled.sh` from the publisher:

```bash
./run_gr00t_bundled.sh --image bi-so101-gr00t:v1
```

Common flags:

```
--image NAME[:TAG]   Which image to run (default: gr00t-inference-bundled:bi-so101)
--port PORT          Host port to expose ZMQ on (default: 5555)
--embodiment TAG     Override the embodiment tag baked into the image
--device DEV         Override torch device (default: cuda)
--name NAME          Container name (default: gr00t_server)
--detach / -d        Run in background
--shell              Drop into a bash shell instead of starting the server
```

### Option B — plain `docker run`

```bash
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 5555:5555 \
    --name gr00t_server \
    bi-so101-gr00t:v1
```

You should see, after a few seconds of model loading:

```
[gr00t] Starting GR00T inference server
[gr00t]   model_path    : /model
[gr00t]   embodiment_tag: new_embodiment
[gr00t]   device        : cuda
[gr00t]   port          : 5555
✓ Server ready — listening on 0.0.0.0:5555
```

Leave this terminal open. The server is now reachable at `localhost:5555`.

### Run in the background

```bash
./run_gr00t_bundled.sh --image bi-so101-gr00t:v1 --detach
docker logs -f gr00t_server      # follow logs
docker stop gr00t_server         # shut it down
```

---

## 3. Connect from IsaacSim

IsaacSim runs natively on this host. Configure your GR00T client to connect
to:

```
host = localhost   (or 127.0.0.1)
port = 5555
```

The communication uses ZMQ, set up by `gr00t.policy.server_client.PolicyServer`.

If IsaacSim is running on a **different machine** on the LAN, use the server
machine's IP address. The container already binds `0.0.0.0` internally;
just make sure port 5555 is open in your firewall.

---

## 4. Configuration overrides

The image ships with sensible defaults baked in, but you can override at run
time via env vars:

| Env var | Default | Notes |
|---|---|---|
| `GR00T_EMBODIMENT_TAG` | what publisher set | Must match training |
| `GR00T_DEVICE` | `cuda` | Use `cuda:1` to pin a specific GPU |
| `GR00T_PORT` | `5555` | Internal port — also adjust `-p` mapping |
| `GR00T_STRICT` | `true` | Strict input/output validation |

Example — pin to GPU 1 and change the host port:

```bash
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e GR00T_DEVICE=cuda:1 \
    -p 6000:5555 \
    bi-so101-gr00t:v1
```

(Client now connects to `localhost:6000`.)

---

## Smoke test

A quick one-liner to confirm the port is reachable from the host:

```bash
nc -zv localhost 5555
# Connection to localhost 5555 port [tcp/*] succeeded!
```

For a full client round-trip, use the GR00T client utilities from the
Isaac-GR00T repo (`gr00t.policy.server_client.PolicyClient`).

---

## Troubleshooting

**`could not select device driver "" with capabilities: [[gpu]]`**
NVIDIA Container Toolkit isn't installed (or Docker wasn't restarted after
install). See the prerequisites section.

**`CUDA error: no kernel image is available for execution on the device`**
Driver too old for CUDA 12.8. Update the host NVIDIA driver to ≥525.

**`OSError: [Errno 28] No space left on device` during model load**
Container's tmpfs is full — add `--shm-size=8g` to the `docker run` args
(or use the helper script which sets `--ipc=host`, removing the limit).

**Out-of-memory on the GPU**
GR00T 1.7 inference wants real VRAM. Check with `nvidia-smi`. If the GPU
is shared, stop other GPU users first. There's no batch-size flag in the
default server — reducing memory generally requires re-exporting the model
in lower precision.

**`Address already in use` on port 5555**
Another process is using that port. Pick a different host port:
`./run_gr00t_bundled.sh --image ... --port 6000`.

**Server starts but IsaacSim client times out**
Check that nothing on the host is firewalling localhost connections. Try
`nc -zv localhost 5555` to confirm reachability before debugging the client.
