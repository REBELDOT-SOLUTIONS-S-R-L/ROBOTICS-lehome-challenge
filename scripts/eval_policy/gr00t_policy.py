"""
GR00T Policy Client for LeHome Challenge.

Connects to a running GR00T inference server over ZeroMQ and adapts its
chunked action output to the per-step `select_action` interface expected
by the evaluation loop.

The GR00T server is launched separately from its own environment:

    cd /workspace/gr00t/Isaac-GR00T && \
    .venv/bin/python -m gr00t.eval.run_gr00t_server \
        --model_path /workspace/gr00t/models/checkpoint-25000 \
        --embodiment_tag new_embodiment --port 5555
"""

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgpack
import numpy as np
import zmq

from lehome.utils.logger import get_logger

from .base_policy import BasePolicy
from .registry import PolicyRegistry

logger = get_logger(__name__)


DEFAULT_MODALITY_JSON = "configs/gr00t/modality.json"


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__ndarray_class__"):
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class _ZmqPolicyClient:
    """Minimal ZMQ REQ client compatible with gr00t.policy.server_client.PolicyServer."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_ms: int = 60000,
        api_token: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.context = zmq.Context.instance()
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: Optional[dict] = None, requires_input: bool = True) -> Any:
        request: Dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self.socket.send(msgpack.packb(request, default=_encode))
            reply = self.socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise
        response = msgpack.unpackb(reply, object_hook=_decode)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def close(self):
        try:
            self.socket.close(linger=0)
        except Exception:
            pass


@PolicyRegistry.register("gr00t")
class GR00TPolicy(BasePolicy):
    """
    Adapter that turns a remote GR00T PolicyServer into a per-step action stream.

    The server returns an action chunk (shape ``(1, T, D)`` per modality key).
    This adapter caches the chunk and plays back ``action_horizon`` steps before
    re-querying. Joint order is rebuilt from ``modality.json``:

        state/action = [left_arm(5), left_gripper(1), right_arm(5), right_gripper(1)]
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        action_horizon: int = 8,
        action_repeat: int = 3,
        task_description: str = "fold the garment on the table",
        modality_json_path: str = DEFAULT_MODALITY_JSON,
        api_token: Optional[str] = None,
        timeout_ms: int = 60000,
        device: str = "cuda",
        **_: Any,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        self.action_repeat = int(action_repeat)
        if self.action_repeat <= 0:
            raise ValueError(f"action_repeat must be positive, got {action_repeat}")
        self.task_description = task_description

        modality_path = Path(modality_json_path)
        if not modality_path.is_absolute():
            modality_path = Path.cwd() / modality_path
        if not modality_path.exists():
            raise FileNotFoundError(f"modality.json not found at {modality_path}")
        with open(modality_path, "r") as f:
            self.modality = json.load(f)

        self._state_slices: List[tuple] = self._build_slices(self.modality["state"])
        self._action_slices: List[tuple] = self._build_slices(self.modality["action"])

        video_cfg = self.modality["video"]
        self._video_keys: Dict[str, str] = {
            gr00t_key: spec["original_key"] for gr00t_key, spec in video_cfg.items()
        }

        ann_cfg = self.modality.get("annotation", {})
        task_spec = ann_cfg.get("human.task_description", {})
        self._language_original_key = task_spec.get("original_key", "task")

        logger.info(
            f"Connecting GR00T client to tcp://{host}:{port} "
            f"(action_horizon={self.action_horizon}, modality={modality_path})"
        )
        self._client = _ZmqPolicyClient(host, port, timeout_ms=timeout_ms, api_token=api_token)

        self._action_buffer: Optional[List[np.ndarray]] = None
        self._chunk_idx: int = 0
        self._repeat_idx: int = 0
        self._frames_dumped: bool = False
        self._query_count: int = 0
        self._last_top_hash: Optional[int] = None
        self._stale_streak: int = 0

    @staticmethod
    def _build_slices(group: Dict[str, Dict[str, int]]) -> List[tuple]:
        # Preserve dict insertion order (Python 3.7+); this matches the logical
        # left_arm / left_gripper / right_arm / right_gripper ordering.
        return [(key, int(v["start"]), int(v["end"])) for key, v in group.items()]

    def reset(self):
        self._action_buffer = None
        self._chunk_idx = 0
        self._repeat_idx = 0
        try:
            self._client.call("reset", {"options": None})
        except Exception as e:
            logger.warning(f"GR00T server reset failed (continuing): {e}")

    def _build_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if "observation.state" not in obs:
            raise KeyError("observation.state missing from env observation")
        state = np.asarray(obs["observation.state"], dtype=np.float32)

        state_dict: Dict[str, np.ndarray] = {}
        for key, start, end in self._state_slices:
            # GR00T expects (T=1, D) per modality key, batched below to (B=1, T, D).
            state_dict[key] = state[start:end][None, :].astype(np.float32)

        video_dict: Dict[str, np.ndarray] = {}
        for gr00t_key, env_key in self._video_keys.items():
            if env_key not in obs:
                raise KeyError(
                    f"Camera '{env_key}' required by modality.json not present in observation"
                )
            img = np.asarray(obs[env_key])
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            # (H,W,C) -> (T=1, H, W, C)
            video_dict[gr00t_key] = img[None, ...]

        language = obs.get(self._language_original_key, self.task_description)
        lang_payload = [[language]] if isinstance(language, str) else language

        def _batch(d):
            return {k: v[None, ...] if isinstance(v, np.ndarray) else v for k, v in d.items()}

        return {
            "video": _batch(video_dict),
            "state": _batch(state_dict),
            "language": {"annotation.human.task_description": lang_payload},
        }

    def _decode_chunk(self, action_chunk: Dict[str, np.ndarray]) -> List[np.ndarray]:
        any_key = next(iter(action_chunk))
        arr = np.asarray(action_chunk[any_key])
        # Expected shape: (B=1, T, D). Fall back defensively if the server drops B.
        if arr.ndim == 3:
            horizon = arr.shape[1]

            def pick(key, t):
                return np.asarray(action_chunk[key])[0, t]
        elif arr.ndim == 2:
            horizon = arr.shape[0]

            def pick(key, t):
                return np.asarray(action_chunk[key])[t]
        else:
            raise ValueError(f"Unexpected action shape for '{any_key}': {arr.shape}")

        total_dim = max(end for _, _, end in self._action_slices)
        steps: List[np.ndarray] = []
        for t in range(horizon):
            out = np.zeros(total_dim, dtype=np.float32)
            for key, start, end in self._action_slices:
                piece = np.atleast_1d(pick(key, t)).astype(np.float32)
                if piece.shape[0] != (end - start):
                    raise ValueError(
                        f"Action slice '{key}' expects dim {end - start}, got {piece.shape[0]}"
                    )
                out[start:end] = piece
            steps.append(out)
        return steps

    def _dump_frames_once(self, observation: Dict[str, np.ndarray]) -> None:
        # Dump once at start, then every 25 queries, so a frozen camera is
        # obvious by inspecting outputs/gr00t_debug/.
        dump_every = 25
        if self._frames_dumped and (self._query_count % dump_every) != 0:
            return
        out_dir = Path("outputs/gr00t_debug")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            for gr00t_key, env_key in self._video_keys.items():
                img = np.asarray(observation[env_key])
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                safe = gr00t_key.replace("/", "_").replace(".", "_")
                suffix = "first" if not self._frames_dumped else f"q{self._query_count:04d}"
                path = out_dir / f"{safe}_{suffix}.png"
                Image.fromarray(img).save(path)
                logger.info("Dumped debug image: %s (shape=%s)", path, img.shape)
        except Exception as e:
            logger.warning("Frame dump failed: %s", e)
        self._frames_dumped = True

    def _query_server(self, observation: Dict[str, np.ndarray]) -> None:
        self._dump_frames_once(observation)

        top_env_key = self._video_keys.get("observation.images.top")
        if top_env_key is None:
            top_env_key = next(iter(self._video_keys.values()))
        top_img = np.asarray(observation[top_env_key])
        top_hash = int(hash(top_img.tobytes()))
        changed = self._last_top_hash is None or top_hash != self._last_top_hash
        if changed:
            self._stale_streak = 0
        else:
            self._stale_streak += 1
        self._last_top_hash = top_hash
        self._query_count += 1

        model_obs = self._build_observation(observation)
        t0 = time.perf_counter()
        response = self._client.call(
            "get_action", {"observation": model_obs, "options": None}
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # Server returns (action_dict, info_dict) — msgpack delivers it as a list/tuple.
        if isinstance(response, (list, tuple)):
            action_dict = response[0]
        elif isinstance(response, dict):
            action_dict = response
        else:
            raise RuntimeError(f"Unexpected GR00T server response type: {type(response)}")

        steps = self._decode_chunk(action_dict)
        if not steps:
            raise RuntimeError("GR00T server returned empty action chunk")

        self._action_buffer = steps[: self.action_horizon]
        self._chunk_idx = 0
        self._repeat_idx = 0

        first = self._action_buffer[0]
        state = np.asarray(observation.get("observation.state", np.zeros(first.shape)))
        delta = first - state[: first.shape[0]] if state.shape[0] >= first.shape[0] else first
        logger.info(
            "GR00T chunk #%d: latency=%.0fms horizon=%d "
            "action[min=%.3f max=%.3f mean=%.3f] "
            "state[min=%.3f max=%.3f] "
            "delta[max_abs=%.3f] "
            "top_cam[%s, stale_streak=%d]",
            self._query_count,
            dt_ms,
            len(self._action_buffer),
            float(first.min()),
            float(first.max()),
            float(first.mean()),
            float(state.min()),
            float(state.max()),
            float(np.abs(delta).max()),
            "CHANGED" if self._stale_streak == 0 else "STALE",
            self._stale_streak,
        )

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        if self._action_buffer is None or self._chunk_idx >= len(self._action_buffer):
            self._query_server(observation)
        action = self._action_buffer[self._chunk_idx]
        self._repeat_idx += 1
        if self._repeat_idx >= self.action_repeat:
            self._repeat_idx = 0
            self._chunk_idx += 1
        return action.astype(np.float32)

    def __del__(self):
        try:
            if getattr(self, "_client", None) is not None:
                self._client.close()
        except Exception:
            pass
