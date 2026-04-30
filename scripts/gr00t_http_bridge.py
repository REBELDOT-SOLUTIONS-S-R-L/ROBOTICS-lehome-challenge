"""
HTTP server that wraps a GR00T policy in the upstream lehome-challenge
DockerPolicy interface (POST /reset, POST /infer with the observation dict).

Two modes:

* zmq      — forwards each request to a separate GR00T inference container on
             ZMQ :5555. Useful in development; lets you reuse a long-running
             model server.
* inproc   — loads the GR00T model directly in this process. Single container,
             single port. Use this for the submission image.

Examples
--------
ZMQ (dev):

    /workspace/gr00t/Isaac-GR00T/.venv/bin/python scripts/gr00t_http_bridge.py \
        --mode zmq --gr00t_host localhost --gr00t_port 5555 \
        --http_port 8080 --task_description "fold the garment"

In-process (production / submission):

    python scripts/gr00t_http_bridge.py \
        --mode inproc --model_path /model --embodiment new_embodiment \
        --device cuda --http_port 8080 --task_description "fold the garment"
"""

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List

import numpy as np


# Mapping from upstream image keys → suffix portion of the GR00T video key.
# At startup the bridge queries the server's modality config and prefixes these
# with "video." only if the server's modality_keys are stored that way.
CAMERA_SUFFIX_MAP = {
    "observation.images.top_rgb": "observation.images.top",
    "observation.images.left_rgb": "observation.images.left_wrist",
    "observation.images.right_rgb": "observation.images.right_wrist",
}

# Slices into the upstream 12-d state vector, using the suffix portion only.
STATE_SLICE_SUFFIXES = [
    ("left_arm", (0, 5)),
    ("left_gripper", (5, 6)),
    ("right_arm", (6, 11)),
    ("right_gripper", (11, 12)),
]

# Order used to flatten the GR00T action chunk — same suffixes as state.
ACTION_SUFFIX_ORDER = ["left_arm", "left_gripper", "right_arm", "right_gripper"]

LANGUAGE_SUFFIX = "human.task_description"


def _deserialize(raw: dict) -> Dict[str, np.ndarray]:
    """Match docker_policy.py's serialization: base64+shape+dtype for big arrays,
    plain lists for small ones."""
    out: Dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "base64" in value:
            buf = base64.b64decode(value["base64"])
            out[key] = np.frombuffer(buf, dtype=value["dtype"]).reshape(value["shape"])
        elif isinstance(value, list):
            out[key] = np.asarray(value, dtype=np.float32)
    return out


def _resolve_keys(modality_configs: dict) -> dict:
    """Use the server's modality_config to discover the exact keys we must send/receive.

    The server stores modality_keys as either bare ('left_arm') or prefixed
    ('state.left_arm'); we just take the keys it reports and match by suffix.
    """

    def pick(keys: List[str], suffixes: List[str]) -> Dict[str, str]:
        out = {}
        for suffix in suffixes:
            matches = [k for k in keys if k == suffix or k.endswith("." + suffix)]
            if not matches:
                raise KeyError(
                    f"server modality keys {keys} are missing expected suffix '{suffix}'"
                )
            out[suffix] = matches[0]
        return out

    video_keys = list(modality_configs["video"].modality_keys)
    state_keys = list(modality_configs["state"].modality_keys)
    action_keys = list(modality_configs["action"].modality_keys)
    language_keys = list(modality_configs["language"].modality_keys)

    return {
        "video": pick(video_keys, list(CAMERA_SUFFIX_MAP.values())),
        "state": pick(state_keys, [s for s, _ in STATE_SLICE_SUFFIXES]),
        "action": pick(action_keys, ACTION_SUFFIX_ORDER),
        "language": pick(language_keys, [LANGUAGE_SUFFIX])[LANGUAGE_SUFFIX],
    }


def _build_gr00t_obs(
    observation: Dict[str, np.ndarray],
    task_description: str,
    keymap: dict,
) -> dict:
    state = np.asarray(observation["observation.state"], dtype=np.float32)

    state_dict: Dict[str, np.ndarray] = {}
    for suffix, (start, end) in STATE_SLICE_SUFFIXES:
        state_dict[keymap["state"][suffix]] = (
            state[start:end].astype(np.float32)[None, None, :]
        )

    video_dict: Dict[str, np.ndarray] = {}
    for src, suffix in CAMERA_SUFFIX_MAP.items():
        if src not in observation:
            raise KeyError(f"missing required image key: {src}")
        img = observation[src]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim == 3:
            img = img[None, None, ...]
        elif img.ndim == 4:
            img = img[None, ...]
        video_dict[keymap["video"][suffix]] = img

    return {
        "video": video_dict,
        "state": state_dict,
        "language": {keymap["language"]: [[task_description]]},
    }


def _flatten_action_chunk(action: Dict[str, np.ndarray], keymap: dict) -> List[np.ndarray]:
    first_key = keymap["action"][ACTION_SUFFIX_ORDER[0]]
    horizon = int(action[first_key].shape[1])
    chunk: List[np.ndarray] = []
    for t in range(horizon):
        parts = [
            np.asarray(action[keymap["action"][s]][0, t, :], dtype=np.float32).reshape(-1)
            for s in ACTION_SUFFIX_ORDER
        ]
        chunk.append(np.concatenate(parts))
    return chunk


class Bridge:
    """Translates the lehome DockerPolicy contract to a BasePolicy backend."""

    def __init__(self, policy: Any, task_description: str):
        self.policy = policy
        self.task_description = task_description

        modality_configs = policy.get_modality_config()
        self.keymap = _resolve_keys(modality_configs)
        print(f"[bridge] resolved modality keys: {self.keymap}", flush=True)

    def reset(self) -> None:
        try:
            self.policy.reset(options={})
        except Exception as exc:
            print(f"[bridge] reset call failed (continuing): {exc}", flush=True)

    def infer(self, observation: Dict[str, np.ndarray]) -> List[np.ndarray]:
        obs = _build_gr00t_obs(observation, self.task_description, self.keymap)
        action, _info = self.policy.get_action(obs)
        return _flatten_action_chunk(action, self.keymap)


def _build_zmq_policy(host: str, port: int):
    from gr00t.policy.server_client import PolicyClient
    return PolicyClient(host=host, port=port, strict=False)


def _build_inproc_policy(model_path: str, embodiment_tag: str, device: str):
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    print(
        f"[bridge] loading GR00T in-process: model={model_path} "
        f"embodiment={embodiment_tag} device={device}",
        flush=True,
    )
    return Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=model_path,
        device=device,
        strict=False,
    )


def make_handler(bridge: Bridge):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b"{}"
            request = json.loads(body)

            try:
                if self.path == "/reset":
                    bridge.reset()
                    response = {"status": "ok"}
                elif self.path == "/infer":
                    obs = _deserialize(request)
                    actions = bridge.infer(obs)
                    response = {"actions": [a.tolist() for a in actions]}
                else:
                    self.send_error(404, f"unknown endpoint: {self.path}")
                    return
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.send_error(500, f"bridge error: {exc}")
                return

            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            print(f"[bridge] {self.command} {self.path}", flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zmq", "inproc"], default="zmq")
    parser.add_argument("--http_host", default="0.0.0.0")
    parser.add_argument("--http_port", type=int, default=8080)
    parser.add_argument("--task_description", default="fold the garment")

    # zmq backend
    parser.add_argument("--gr00t_host", default="localhost")
    parser.add_argument("--gr00t_port", type=int, default=5555)

    # inproc backend
    parser.add_argument("--model_path", default="/model")
    parser.add_argument("--embodiment", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.mode == "zmq":
        policy = _build_zmq_policy(args.gr00t_host, args.gr00t_port)
        backend_desc = f"gr00t {args.gr00t_host}:{args.gr00t_port}"
    else:
        policy = _build_inproc_policy(args.model_path, args.embodiment, args.device)
        backend_desc = f"in-process {args.model_path}"

    bridge = Bridge(policy=policy, task_description=args.task_description)

    server = HTTPServer((args.http_host, args.http_port), make_handler(bridge))
    print(
        f"[bridge] listening on {args.http_host}:{args.http_port} → {backend_desc}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
