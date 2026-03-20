"""Direct HDF5 recording primitives for MimicGen teleoperation datasets."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectRLEnv

from lehome.utils.logger import get_logger

logger = get_logger(__name__)

FLUSH_INTERVAL = 100
NUMERIC_CHUNK_ROWS = 256

try:
    import h5py
except ImportError:
    h5py = None


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError(
            "h5py is required for direct HDF5 recording. "
            "Install it in your environment (e.g. `pip install h5py`)."
        )


def _as_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert tensor or array-like values to numpy arrays."""
    if isinstance(value, np.ndarray):
        arr = value
    elif torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, Mapping):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_json_compatible(v) for v in value]
    return value


def _quat_wxyz_to_xyzw(quat: Any) -> np.ndarray:
    quat_arr = _as_numpy(quat, dtype=np.float32).reshape(-1)
    if quat_arr.shape[0] != 4:
        return quat_arr
    return np.array([quat_arr[1], quat_arr[2], quat_arr[3], quat_arr[0]], dtype=np.float32)


def _resolve_garment_pose_value(
    object_initial_pose: dict[str, Any] | None,
    garment_name: str | None,
) -> Any:
    if object_initial_pose is None:
        return None
    if not isinstance(object_initial_pose, dict):
        return object_initial_pose
    if "Garment" in object_initial_pose:
        return object_initial_pose["Garment"]
    if garment_name and garment_name in object_initial_pose:
        return object_initial_pose[garment_name]
    return next(iter(object_initial_pose.values()), None)


def _get_scene_articulation(env: DirectRLEnv, name: str) -> Any | None:
    scene = getattr(env, "scene", None)
    if scene is None:
        return None

    try:
        return scene[name]
    except Exception:
        pass

    articulations = getattr(scene, "articulations", None)
    if articulations is None:
        return None

    try:
        return articulations.get(name)
    except Exception:
        pass

    try:
        if name in articulations:
            return articulations[name]
    except Exception:
        pass

    return None


def _get_single_arm_candidates(env: DirectRLEnv) -> list[str]:
    scene = getattr(env, "scene", None)
    articulations = getattr(scene, "articulations", None) if scene is not None else None

    names: list[str] = []
    if articulations is not None:
        try:
            names.extend(str(name) for name in articulations.keys())
        except Exception:
            pass

    ordered: list[str] = []
    seen: set[str] = set()
    for name in ("robot", "arm", "left_arm", "right_arm"):
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _resolve_eef_body_idx(
    env: DirectRLEnv,
    arm_name: str,
    arm: Any,
    eef_body_idx_cache: dict[str, int],
) -> int | None:
    del env
    if arm_name in eef_body_idx_cache:
        return eef_body_idx_cache[arm_name]

    candidate_patterns = (
        "^gripper_frame_link$",
        "^gripper_link$",
        ".*gripper_frame.*",
        ".*gripper.*",
        ".*wrist.*",
    )
    for pattern in candidate_patterns:
        try:
            body_ids, _ = arm.find_bodies(pattern, preserve_order=True)
            body_ids = _as_numpy(body_ids, dtype=np.int64).reshape(-1)
            if body_ids.size > 0:
                body_idx = int(body_ids[0])
                eef_body_idx_cache[arm_name] = body_idx
                return body_idx
        except Exception:
            continue

    body_positions = getattr(arm.data, "body_link_pos_w", None)
    if body_positions is None:
        body_positions = getattr(arm.data, "body_pos_w", None)
    if body_positions is None:
        return None

    body_idx = int(body_positions.shape[1] - 1)
    eef_body_idx_cache[arm_name] = body_idx
    return body_idx


class DirectHDF5Recorder:
    """Episode-buffered recorder that flushes teleop data into HDF5 on finalize."""

    _ACTIVE_EPISODE_NAME = "_active_demo"

    def __init__(
        self,
        env: DirectRLEnv,
        file_path: Path,
        env_args: dict[str, Any],
        fps: int,
        is_bi_arm: bool,
        garment_info_json_path: Path | None = None,
    ) -> None:
        _require_h5py()
        self._env = env
        self._file_path = Path(file_path)
        self._json_path = Path(garment_info_json_path) if garment_info_json_path else None
        self._is_bi_arm = bool(is_bi_arm)
        self._compression = "lzf"
        self._flush_steps = FLUSH_INTERVAL
        self._eef_body_idx_cache: dict[str, int] = {}
        self._root_velocity_source_cache: dict[str, str] = {}
        self._state_arm_names = self._resolve_state_arm_names()
        self._metadata_store: dict[str, Any] = {}

        self._active_group: Any | None = None
        self._active_groups: dict[str, Any] = {}
        self._active_datasets: dict[str, Any] = {}
        self._active_episode_meta: dict[str, Any] | None = None
        self._active_num_samples = 0
        self._episode_frames: list[dict[str, Any]] = []
        self._episode_articulation_states: list[dict[str, dict[str, np.ndarray]]] = []

        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._file_path, "w")
        self._data_group = self._file.create_group("data")
        self._data_group.attrs["env_args"] = json.dumps(env_args)
        self._data_group.attrs["total"] = np.int64(0)
        self._data_group.attrs["fps"] = np.int64(int(fps))
        self._data_group.attrs["actions_mode"] = "joint"
        self._data_group.attrs["num_episodes"] = np.int64(0)

        self._demo_count = 0
        self._total_samples = 0

    @property
    def file_path(self) -> Path:
        return self._file_path

    @staticmethod
    def _slice_env0(value: Any, dtype: np.dtype = np.float32) -> np.ndarray:
        arr = _as_numpy(value, dtype=dtype)
        if arr.ndim >= 2:
            arr = arr[0]
        return arr.reshape(-1)

    def _resolve_state_arm_names(self) -> list[str]:
        if self._is_bi_arm:
            return ["left_arm", "right_arm"]
        for arm_name in _get_single_arm_candidates(self._env):
            if _get_scene_articulation(self._env, arm_name) is not None:
                return [arm_name]
        return ["robot"]

    def _reset_active_episode(self) -> None:
        self._active_group = None
        self._active_groups = {}
        self._active_datasets = {}
        self._active_episode_meta = None
        self._active_num_samples = 0
        self._episode_frames = []
        self._episode_articulation_states = []

    def _ensure_group(self, path: str) -> Any:
        if self._active_group is None:
            raise RuntimeError("No active episode group.")
        if not path:
            return self._active_group
        group = self._active_groups.get(path)
        if group is not None:
            return group
        group = self._active_group
        for part in path.split("/"):
            group = group.require_group(part)
        self._active_groups[path] = group
        return group

    def _chunk_shape_for_sample(self, sample: np.ndarray) -> tuple[int, ...]:
        if sample.ndim >= 3 and sample.shape[-1] in (1, 3, 4):
            return (1, *sample.shape)
        return (NUMERIC_CHUNK_ROWS, *sample.shape)

    def _ensure_resizable_dataset(
        self,
        path: str,
        sample: Any,
        dtype: np.dtype | None = None,
    ) -> Any:
        dataset = self._active_datasets.get(path)
        if dataset is not None:
            return dataset

        sample_arr = _as_numpy(sample, dtype=dtype)
        if sample_arr.ndim == 0:
            sample_arr = sample_arr.reshape(1)
        parent_path, name = path.rsplit("/", 1) if "/" in path else ("", path)
        parent_group = self._ensure_group(parent_path)
        dataset = parent_group.create_dataset(
            name,
            shape=(0, *sample_arr.shape),
            maxshape=(None, *sample_arr.shape),
            dtype=sample_arr.dtype,
            chunks=self._chunk_shape_for_sample(sample_arr),
            compression=self._compression,
        )
        self._active_datasets[path] = dataset
        return dataset

    def _append_dataset(
        self,
        path: str,
        sample: Any,
        dtype: np.dtype | None = None,
    ) -> None:
        dataset = self._ensure_resizable_dataset(path, sample, dtype=dtype)
        sample_arr = _as_numpy(sample, dtype=dataset.dtype)
        if sample_arr.ndim == 0:
            sample_arr = sample_arr.reshape(1)
        dataset.resize(self._active_num_samples + 1, axis=0)
        dataset[self._active_num_samples] = sample_arr

    def _write_fixed_dataset(self, group: Any, name: str, value: Any) -> None:
        arr = np.atleast_1d(_as_numpy(value, dtype=np.float32))
        group.create_dataset(name, data=arr[None, ...], compression=self._compression)

    def _get_scene_articulation(self, name: str) -> Any | None:
        return _get_scene_articulation(self._env, name)

    def _resolve_eef_body_idx(self, arm_name: str, arm: Any) -> int | None:
        return _resolve_eef_body_idx(self._env, arm_name, arm, self._eef_body_idx_cache)

    def _get_root_velocity_source(self, arm_name: str, arm: Any) -> str:
        cached = self._root_velocity_source_cache.get(arm_name)
        if cached is not None:
            return cached

        if getattr(arm.data, "root_vel_w", None) is not None:
            source = "root_vel_w"
        elif getattr(arm.data, "root_lin_vel_w", None) is not None and getattr(arm.data, "root_ang_vel_w", None) is not None:
            source = "root_lin_ang_vel_w"
        else:
            source = "scene"
        self._root_velocity_source_cache[arm_name] = source
        return source

    def _get_scene_articulation_state_fallback(self) -> dict[str, Any]:
        try:
            state = self._env.scene.get_state(is_relative=False)
        except Exception:
            return {}
        articulation_state = state.get("articulation", {})
        return articulation_state if isinstance(articulation_state, dict) else {}

    def _capture_arm_state(
        self,
        arm_name: str,
        arm: Any,
        scene_entry: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        joint_pos = getattr(arm.data, "joint_pos", None)
        joint_vel = getattr(arm.data, "joint_vel", None)
        if joint_vel is None:
            joint_vel = getattr(arm.data, "joint_velocity", None)

        root_pos_w = getattr(arm.data, "root_pos_w", None)
        root_quat_w = getattr(arm.data, "root_quat_w", None)

        state: dict[str, np.ndarray] = {}

        if joint_pos is not None:
            state["joint_position"] = self._slice_env0(joint_pos)
        elif scene_entry and "joint_position" in scene_entry:
            state["joint_position"] = self._slice_env0(scene_entry["joint_position"])
        else:
            state["joint_position"] = np.zeros(6, dtype=np.float32)

        if joint_vel is not None:
            state["joint_velocity"] = self._slice_env0(joint_vel)
        elif scene_entry and "joint_velocity" in scene_entry:
            state["joint_velocity"] = self._slice_env0(scene_entry["joint_velocity"])
        else:
            state["joint_velocity"] = np.zeros_like(state["joint_position"], dtype=np.float32)

        if root_pos_w is not None and root_quat_w is not None:
            root_pos = self._slice_env0(root_pos_w)
            root_quat = _quat_wxyz_to_xyzw(self._slice_env0(root_quat_w))
            state["root_pose"] = np.concatenate([root_pos[:3], root_quat[:4]], axis=0)
        elif scene_entry and "root_pose" in scene_entry:
            state["root_pose"] = self._slice_env0(scene_entry["root_pose"])
        else:
            state["root_pose"] = np.zeros(7, dtype=np.float32)

        root_velocity_source = self._get_root_velocity_source(arm_name, arm)
        if root_velocity_source == "root_vel_w":
            state["root_velocity"] = self._slice_env0(arm.data.root_vel_w)
        elif root_velocity_source == "root_lin_ang_vel_w":
            root_lin = self._slice_env0(arm.data.root_lin_vel_w)
            root_ang = self._slice_env0(arm.data.root_ang_vel_w)
            state["root_velocity"] = np.concatenate([root_lin[:3], root_ang[:3]], axis=0)
        elif scene_entry and "root_velocity" in scene_entry:
            state["root_velocity"] = self._slice_env0(scene_entry["root_velocity"])
        else:
            state["root_velocity"] = np.zeros(6, dtype=np.float32)

        return state

    def _compute_arm_ee_frame_state_from_arm(
        self,
        arm_name: str,
        arm: Any,
    ) -> np.ndarray | None:
        root_pos_w = getattr(arm.data, "root_pos_w", None)
        root_quat_w = getattr(arm.data, "root_quat_w", None)
        if root_pos_w is None or root_quat_w is None:
            return None

        eef_body_idx = self._resolve_eef_body_idx(arm_name, arm)
        if eef_body_idx is None:
            return None

        body_pos_w = getattr(arm.data, "body_link_pos_w", None)
        body_quat_w = getattr(arm.data, "body_link_quat_w", None)
        if body_pos_w is None or body_quat_w is None:
            body_pos_w = getattr(arm.data, "body_pos_w", None)
            body_quat_w = getattr(arm.data, "body_quat_w", None)
        if body_pos_w is None or body_quat_w is None:
            return None

        device = getattr(self._env, "device", "cpu")
        root_pos = torch.as_tensor(root_pos_w[0], device=device, dtype=torch.float32).unsqueeze(0)
        root_quat = torch.as_tensor(root_quat_w[0], device=device, dtype=torch.float32).unsqueeze(0)
        eef_pos = torch.as_tensor(body_pos_w[0, eef_body_idx], device=device, dtype=torch.float32).unsqueeze(0)
        eef_quat = torch.as_tensor(body_quat_w[0, eef_body_idx], device=device, dtype=torch.float32).unsqueeze(0)

        ee_pos_robot, ee_quat_robot = math_utils.subtract_frame_transforms(
            root_pos,
            root_quat,
            eef_pos,
            eef_quat,
        )
        return torch.cat([ee_pos_robot, ee_quat_robot], dim=-1)[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def _capture_articulation_snapshot(self) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray | None]:
        scene_fallback: dict[str, Any] | None = None
        articulation_state: dict[str, dict[str, np.ndarray]] = {}
        ee_frame_state_by_arm: dict[str, np.ndarray] = {}

        for arm_name in self._state_arm_names:
            arm = self._get_scene_articulation(arm_name)
            if arm is None:
                continue
            scene_entry = None
            if self._get_root_velocity_source(arm_name, arm) == "scene":
                if scene_fallback is None:
                    scene_fallback = self._get_scene_articulation_state_fallback()
                candidate = scene_fallback.get(arm_name)
                if isinstance(candidate, dict):
                    scene_entry = candidate
            articulation_state[arm_name] = self._capture_arm_state(
                arm_name,
                arm,
                scene_entry=scene_entry,
            )
            ee_frame_state = self._compute_arm_ee_frame_state_from_arm(arm_name, arm)
            if ee_frame_state is not None:
                ee_frame_state_by_arm[arm_name] = ee_frame_state

        combined_ee_frame_state = None
        if self._is_bi_arm:
            left_ee = ee_frame_state_by_arm.get("left_arm")
            right_ee = ee_frame_state_by_arm.get("right_arm")
            if left_ee is not None and right_ee is not None:
                combined_ee_frame_state = np.concatenate([left_ee, right_ee], axis=0).astype(np.float32, copy=False)
        else:
            for arm_name in self._state_arm_names:
                combined_ee_frame_state = ee_frame_state_by_arm.get(arm_name)
                if combined_ee_frame_state is not None:
                    break

        return articulation_state, combined_ee_frame_state

    def _write_initial_state_snapshot(self) -> None:
        initial_state_group = self._ensure_group("initial_state")
        articulation_group = self._ensure_group("initial_state/articulation")
        articulation_state, _ = self._capture_articulation_snapshot()

        for arm_name, state in articulation_state.items():
            arm_group = articulation_group.create_group(arm_name)
            for key, value in state.items():
                self._write_fixed_dataset(arm_group, key, value)

        self._write_initial_garment_state_if_available(initial_state_group)

    def _write_initial_garment_state_if_available(self, initial_state_group: Any) -> None:
        if self._active_episode_meta is None:
            return

        garment_name = str(self._active_episode_meta.get("garment_name") or "Garment")
        pose_value = _resolve_garment_pose_value(
            self._active_episode_meta.get("object_initial_pose"),
            garment_name,
        )
        if pose_value is None:
            return

        garment_group = initial_state_group.require_group("garment")
        garment_entry_group = garment_group.require_group(garment_name)
        self._write_fixed_dataset(garment_entry_group, "initial_pose", pose_value)

        scale_value = self._active_episode_meta.get("scale")
        if scale_value is not None:
            self._write_fixed_dataset(garment_entry_group, "scale", scale_value)

    def _write_metadata_json(self) -> None:
        if self._json_path is None:
            return
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._json_path, "w", encoding="utf-8") as fout:
            json.dump(self._metadata_store, fout, indent=2, ensure_ascii=False)

    def _append_sidecar_metadata(self) -> None:
        if self._active_episode_meta is None:
            return
        garment_name = str(self._active_episode_meta.get("garment_name") or "unknown")
        episode_key = str(int(self._active_episode_meta["episode_index"]))
        pose_value = _resolve_garment_pose_value(
            self._active_episode_meta.get("object_initial_pose"),
            garment_name,
        )
        episode_rec: dict[str, Any] = {
            "object_initial_pose": _to_json_compatible(pose_value),
        }
        scale_value = self._active_episode_meta.get("scale")
        if scale_value is not None:
            episode_rec["scale"] = _to_json_compatible(scale_value)
        self._metadata_store.setdefault(garment_name, {})[episode_key] = episode_rec

    def begin_episode(
        self,
        episode_index: int,
        object_initial_pose: dict[str, Any] | None,
        garment_name: str | None,
        scale: Any | None,
    ) -> None:
        if self._active_group is not None:
            raise RuntimeError("Cannot begin a new episode while another is active.")
        if self._ACTIVE_EPISODE_NAME in self._data_group:
            del self._data_group[self._ACTIVE_EPISODE_NAME]

        self._active_episode_meta = {
            "episode_index": int(episode_index),
            "object_initial_pose": _to_json_compatible(object_initial_pose),
            "garment_name": str(garment_name) if garment_name else "unknown",
            "scale": _to_json_compatible(scale),
        }
        self._active_group = self._data_group.create_group(self._ACTIVE_EPISODE_NAME)
        self._active_group.attrs["num_samples"] = np.int64(0)
        self._active_group.attrs["seed"] = np.int64(random.randint(0, 2**31 - 1))
        self._active_group.attrs["success"] = np.bool_(True)
        self._active_groups = {"": self._active_group}
        self._active_datasets = {}
        self._active_num_samples = 0
        self._ensure_group("obs")
        self._ensure_group("initial_state")
        self._ensure_group("states")
        self._ensure_group("states/articulation")
        self._write_initial_state_snapshot()

    @staticmethod
    def _normalize_rgb_frame(frame_value: Any) -> np.ndarray | None:
        if frame_value is None:
            return None
        rgb = _as_numpy(frame_value, dtype=np.uint8)
        if rgb.ndim == 4:
            rgb = rgb.squeeze(0)
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
            return None
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        return rgb

    @staticmethod
    def _normalize_buffer_sample(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
        arr = _as_numpy(value, dtype=dtype)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr

    def _stack_buffer_key(self, key: str, dtype: np.dtype) -> np.ndarray:
        stacked: list[np.ndarray] = []
        for frame in self._episode_frames:
            if key not in frame:
                raise KeyError(f"Missing buffered frame key '{key}' while finalizing episode.")
            stacked.append(self._normalize_buffer_sample(frame[key], dtype=dtype))
        return np.stack(stacked, axis=0)

    def _stack_optional_buffer_key(self, key: str, dtype: np.dtype) -> np.ndarray | None:
        stacked: list[np.ndarray] = []
        for frame in self._episode_frames:
            value = frame.get(key)
            if value is None:
                return None
            stacked.append(self._normalize_buffer_sample(value, dtype=dtype))
        if not stacked:
            return None
        return np.stack(stacked, axis=0)

    def _extract_joint_velocities_from_buffer(
        self,
        num_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left = np.zeros((num_frames, 6), dtype=np.float32)
        right = np.zeros((num_frames, 6), dtype=np.float32)
        single = np.zeros((num_frames, 6), dtype=np.float32)

        for i, articulation in enumerate(self._episode_articulation_states):
            if not isinstance(articulation, dict):
                continue

            if "left_arm" in articulation and "joint_velocity" in articulation["left_arm"]:
                arr = _as_numpy(articulation["left_arm"]["joint_velocity"], dtype=np.float32)
                if arr.ndim == 1 and arr.shape[0] >= 6:
                    left[i] = arr[:6]
            if "right_arm" in articulation and "joint_velocity" in articulation["right_arm"]:
                arr = _as_numpy(articulation["right_arm"]["joint_velocity"], dtype=np.float32)
                if arr.ndim == 1 and arr.shape[0] >= 6:
                    right[i] = arr[:6]

            robot_state = articulation.get("robot")
            if isinstance(robot_state, dict) and "joint_velocity" in robot_state:
                arr = _as_numpy(robot_state["joint_velocity"], dtype=np.float32)
                if arr.ndim == 1 and arr.shape[0] >= 6:
                    single[i] = arr[:6]
                    continue

            for _, entity in sorted(articulation.items()):
                if not isinstance(entity, dict):
                    continue
                if "joint_velocity" not in entity:
                    continue
                arr = _as_numpy(entity["joint_velocity"], dtype=np.float32)
                if arr.ndim == 1 and arr.shape[0] >= 6:
                    single[i] = arr[:6]
                    break

        return left, right, single

    def _write_buffered_obs_group(self, joint_actions: np.ndarray) -> None:
        obs_group = self._ensure_group("obs")
        obs_group.create_dataset("actions", data=joint_actions, compression=self._compression)

        joint_pos = self._stack_buffer_key("joint_pos", dtype=np.float32)
        joint_vel_left, joint_vel_right, joint_vel_single = self._extract_joint_velocities_from_buffer(
            num_frames=joint_pos.shape[0]
        )

        if self._is_bi_arm:
            left_joint_pos = joint_pos[:, :6]
            right_joint_pos = joint_pos[:, 6:12]
            left_target = joint_actions[:, :6]
            right_target = joint_actions[:, 6:12]

            obs_group.create_dataset("left_joint_pos", data=left_joint_pos, compression=self._compression)
            obs_group.create_dataset("right_joint_pos", data=right_joint_pos, compression=self._compression)
            obs_group.create_dataset("left_joint_pos_target", data=left_target, compression=self._compression)
            obs_group.create_dataset("right_joint_pos_target", data=right_target, compression=self._compression)
            obs_group.create_dataset("left_joint_pos_rel", data=(left_target - left_joint_pos), compression=self._compression)
            obs_group.create_dataset("right_joint_pos_rel", data=(right_target - right_joint_pos), compression=self._compression)
            obs_group.create_dataset("left_joint_vel", data=joint_vel_left, compression=self._compression)
            obs_group.create_dataset("right_joint_vel", data=joint_vel_right, compression=self._compression)
            obs_group.create_dataset("left_joint_vel_rel", data=joint_vel_left, compression=self._compression)
            obs_group.create_dataset("right_joint_vel_rel", data=joint_vel_right, compression=self._compression)

            ee_frame_state = self._stack_optional_buffer_key("ee_frame_state", dtype=np.float32)
            if ee_frame_state is not None and ee_frame_state.ndim == 2 and ee_frame_state.shape[1] >= 14:
                obs_group.create_dataset("left_ee_frame_state", data=ee_frame_state[:, :7], compression=self._compression)
                obs_group.create_dataset("right_ee_frame_state", data=ee_frame_state[:, 7:14], compression=self._compression)

            top_rgb = self._stack_optional_buffer_key("top_rgb", dtype=np.uint8)
            if top_rgb is not None:
                obs_group.create_dataset("top", data=top_rgb, compression=self._compression)
            left_rgb = self._stack_optional_buffer_key("left_rgb", dtype=np.uint8)
            if left_rgb is not None:
                obs_group.create_dataset("left_wrist", data=left_rgb, compression=self._compression)
            right_rgb = self._stack_optional_buffer_key("right_rgb", dtype=np.uint8)
            if right_rgb is not None:
                obs_group.create_dataset("right_wrist", data=right_rgb, compression=self._compression)
        else:
            single_joint_pos = joint_pos[:, :6]
            single_target = joint_actions[:, :6]
            obs_group.create_dataset("joint_pos", data=single_joint_pos, compression=self._compression)
            obs_group.create_dataset("joint_pos_target", data=single_target, compression=self._compression)
            obs_group.create_dataset("joint_pos_rel", data=(single_target - single_joint_pos), compression=self._compression)
            obs_group.create_dataset("joint_vel", data=joint_vel_single, compression=self._compression)
            obs_group.create_dataset("joint_vel_rel", data=joint_vel_single, compression=self._compression)

            ee_frame_state = self._stack_optional_buffer_key("ee_frame_state", dtype=np.float32)
            if ee_frame_state is not None and ee_frame_state.ndim == 2 and ee_frame_state.shape[1] >= 7:
                obs_group.create_dataset("ee_frame_state", data=ee_frame_state[:, :7], compression=self._compression)

            top_rgb = self._stack_optional_buffer_key("top_rgb", dtype=np.uint8)
            if top_rgb is not None:
                obs_group.create_dataset("top", data=top_rgb, compression=self._compression)
            wrist_rgb = self._stack_optional_buffer_key("wrist_rgb", dtype=np.uint8)
            if wrist_rgb is not None:
                obs_group.create_dataset("wrist", data=wrist_rgb, compression=self._compression)

    def _write_buffered_state_groups(self) -> None:
        articulation_group = self._ensure_group("states/articulation")
        if not self._episode_articulation_states:
            return

        series_by_arm: dict[str, dict[str, list[np.ndarray]]] = {}
        for articulation in self._episode_articulation_states:
            if not isinstance(articulation, dict):
                continue
            for arm_name, state in articulation.items():
                if not isinstance(state, dict):
                    continue
                arm_series = series_by_arm.setdefault(arm_name, {})
                for key, value in state.items():
                    arm_series.setdefault(key, []).append(_as_numpy(value, dtype=np.float32))

        for arm_name, state_series in sorted(series_by_arm.items()):
            arm_group = articulation_group.create_group(arm_name)
            for key, values in sorted(state_series.items()):
                if not values:
                    continue
                arm_group.create_dataset(
                    key,
                    data=np.stack(values, axis=0).astype(np.float32, copy=False),
                    compression=self._compression,
                )

    def _flush_buffered_episode_to_hdf5(self) -> None:
        if self._active_group is None:
            raise RuntimeError("Cannot flush buffered episode without an active group.")
        if not self._episode_frames:
            return

        joint_actions = self._stack_buffer_key("action", dtype=np.float32)
        self._active_group.create_dataset("actions", data=joint_actions, compression=self._compression)
        self._active_group.create_dataset("processed_actions", data=joint_actions, compression=self._compression)
        self._write_buffered_obs_group(joint_actions)
        self._write_buffered_state_groups()

    def append_frame(self, frame: dict[str, Any]) -> None:
        if self._active_group is None:
            raise RuntimeError("Cannot append a frame without an active episode.")

        joint_action = _as_numpy(frame["action"], dtype=np.float32).reshape(-1)
        joint_pos = _as_numpy(frame["observation.state"], dtype=np.float32).reshape(-1)
        articulation_state, ee_frame_state = self._capture_articulation_snapshot()

        frame_to_store: dict[str, Any] = {
            "action": joint_action,
            "joint_pos": joint_pos,
        }
        if ee_frame_state is not None:
            frame_to_store["ee_frame_state"] = ee_frame_state

        top_rgb = self._normalize_rgb_frame(frame.get("observation.images.top_rgb"))
        if top_rgb is not None:
            frame_to_store["top_rgb"] = top_rgb

        if self._is_bi_arm:
            left_rgb = self._normalize_rgb_frame(frame.get("observation.images.left_rgb"))
            if left_rgb is not None:
                frame_to_store["left_rgb"] = left_rgb
            right_rgb = self._normalize_rgb_frame(frame.get("observation.images.right_rgb"))
            if right_rgb is not None:
                frame_to_store["right_rgb"] = right_rgb
        else:
            wrist_rgb = self._normalize_rgb_frame(frame.get("observation.images.wrist_rgb"))
            if wrist_rgb is not None:
                frame_to_store["wrist_rgb"] = wrist_rgb

        self._episode_frames.append(frame_to_store)
        self._episode_articulation_states.append(articulation_state)
        self._active_num_samples += 1
        self._active_group.attrs["num_samples"] = np.int64(self._active_num_samples)

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.append_frame(frame)

    def discard_episode(self) -> None:
        if self._active_group is not None and self._ACTIVE_EPISODE_NAME in self._data_group:
            del self._data_group[self._ACTIVE_EPISODE_NAME]
            self._file.flush()
        self._reset_active_episode()

    def clear_episode_buffer(self) -> None:
        self.discard_episode()

    def finalize_episode(self) -> None:
        if self._active_group is None:
            return
        if self._active_num_samples <= 0:
            self.discard_episode()
            return

        self._flush_buffered_episode_to_hdf5()

        demo_name = f"demo_{self._demo_count}"
        if demo_name in self._data_group:
            raise ValueError(f"Episode group '{demo_name}' already exists in HDF5.")

        self._data_group.move(self._ACTIVE_EPISODE_NAME, demo_name)
        self._demo_count += 1
        self._total_samples += self._active_num_samples
        self._data_group.attrs["total"] = np.int64(self._total_samples)
        self._data_group.attrs["num_episodes"] = np.int64(self._demo_count)
        self._append_sidecar_metadata()
        self._file.flush()
        self._reset_active_episode()

    def save_episode(self) -> None:
        self.finalize_episode()

    def finalize(self) -> None:
        if self._file is not None:
            if self._active_group is not None:
                self.discard_episode()
            if self._metadata_store:
                self._write_metadata_json()
            self._file.flush()
            self._file.close()
            self._file = None


__all__ = [
    "DirectHDF5Recorder",
    "_as_numpy",
    "_get_scene_articulation",
    "_get_single_arm_candidates",
    "_require_h5py",
    "_resolve_eef_body_idx",
]
