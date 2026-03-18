"""Dataset recording utility functions for teleoperation data collection."""

import argparse
import json
import random
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaacsim.simulation_app import SimulationApp
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from lehome.devices import (
    Se3Keyboard,
    SO101Leader,
    BiSO101Leader,
    BiKeyboard,
)
from lehome.utils.record import get_next_experiment_path_with_gap
from lehome.utils.logger import get_logger
from lehome.tasks.fold_cloth.checkpoint_mappings import (
    ARM_KEYPOINT_GROUPS,
    CHECKPOINT_LABELS,
)

from ..utils.common import stabilize_garment_after_reset

logger = get_logger(__name__)
SUCCESS_LOG_INTERVAL = 50
DEBUG_POSE_LOG_INTERVAL = 50
FLUSH_INTERVAL = 100
NUMERIC_CHUNK_ROWS = 256
GARMENT_CHECKPOINT_LABELS = CHECKPOINT_LABELS
ARM_KEYPOINT_DISTANCE_LABELS = ARM_KEYPOINT_GROUPS

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


def _as_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert tensor/array-like values to numpy arrays."""
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
    # Isaac Lab articulation tensors use wxyz, while dataset root_pose follows xyzw.
    return np.array([quat_arr[1], quat_arr[2], quat_arr[3], quat_arr[0]], dtype=np.float32)


def _resolve_garment_pose_value(
    object_initial_pose: Optional[Dict[str, Any]],
    garment_name: Optional[str],
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


def _get_episode_garment_meta(env: DirectRLEnv) -> Tuple[Optional[str], Optional[Any]]:
    garment_name = None
    if hasattr(env, "cfg") and hasattr(env.cfg, "garment_name"):
        garment_name = env.cfg.garment_name

    scale = None
    if hasattr(env, "object") and hasattr(env.object, "init_scale"):
        try:
            scale = env.object.init_scale
        except Exception:
            logger.warning("Failed to get scale from garment object")
    return garment_name, scale


def _safe_get_all_pose(env: DirectRLEnv) -> Optional[Dict[str, Any]]:
    try:
        return env.get_all_pose()
    except Exception as e:
        logger.error(f"[Recording] Failed to get initial pose: {e}")
        traceback.print_exc()
        return None


def _get_or_build_maintain_action(
    env: DirectRLEnv,
    args: argparse.Namespace,
    control_state: Dict[str, Any],
    current_obs: Dict[str, Any],
) -> torch.Tensor:
    current_state = current_obs.get("observation.state")
    if current_state is not None:
        state_arr = _as_numpy(current_state, dtype=np.float32).reshape(-1)
        action_dim = int(state_arr.shape[0])
    else:
        state_arr = None
        action_dim = 12 if "Bi" in (args.task or "") else 6

    maintain_action = control_state.get("maintain_action")
    if (
        maintain_action is None
        or not torch.is_tensor(maintain_action)
        or tuple(maintain_action.shape) != (1, action_dim)
        or str(maintain_action.device) != str(env.device)
    ):
        maintain_action = torch.zeros(1, action_dim, dtype=torch.float32, device=env.device)
        control_state["maintain_action"] = maintain_action

    if state_arr is None:
        maintain_action.zero_()
    else:
        maintain_action[0].copy_(
            torch.as_tensor(state_arr, dtype=torch.float32, device=env.device)
        )
    return maintain_action


def _evaluate_success_result(env: DirectRLEnv) -> Optional[Dict[str, Any]]:
    """Evaluate garment success without relying on the environment's internal log throttle."""
    if (
        hasattr(env, "object")
        and env.object is not None
        and hasattr(env.object, "_cloth_prim_view")
        and hasattr(env, "garment_loader")
        and hasattr(env, "cfg")
        and hasattr(env.cfg, "garment_name")
    ):
        from lehome.utils.success_checker_chanllege import evaluate_garment_fold_success

        garment_type = env.garment_loader.get_garment_type(env.cfg.garment_name)
        return evaluate_garment_fold_success(env.object, garment_type)

    if not hasattr(env, "_get_success"):
        return None

    success_value = env._get_success()
    if torch.is_tensor(success_value):
        success = bool(success_value.reshape(-1)[0].item()) if success_value.numel() > 0 else False
    else:
        success = bool(success_value)
    return {"success": success, "garment_type": "unknown", "thresholds": [], "details": {}}


def _log_success_result(
    env: DirectRLEnv,
    episode_index: int,
    step_in_episode: Optional[int] = None,
    context: str = "progress",
) -> Optional[bool]:
    """Log a deterministic success breakdown from the recorder."""
    try:
        result = _evaluate_success_result(env)
    except Exception as e:
        logger.warning(
            f"[Recording][Episode {episode_index}] Failed to evaluate success during {context}: {e}"
        )
        return None

    if result is None:
        logger.warning(
            f"[Recording][Episode {episode_index}] Success evaluation unavailable during {context}."
        )
        return None

    prefix = f"[Recording][Episode {episode_index}]"
    if step_in_episode is not None:
        prefix += f"[step {step_in_episode}]"

    logger.info(
        f"{prefix} [Success Check] Garment type: {result.get('garment_type', 'unknown')}, "
        f"Thresholds: {result.get('thresholds', [])}"
    )
    details = result.get("details", {})
    for condition_info in details.values():
        status = "✓" if condition_info.get("passed", False) else "✗"
        logger.info(f"{prefix}   {condition_info.get('description', '')} -> {status}")

    success = bool(result.get("success", False))
    logger.info(f"{prefix} [Success Check] Final result: {'Success ✓' if success else 'Failed ✗'}")
    return success


def _get_scene_articulation(env: DirectRLEnv, name: str) -> Optional[Any]:
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
    eef_body_idx_cache: Dict[str, int],
) -> Optional[int]:
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


def _get_arm_eef_world_position_cm(
    env: DirectRLEnv,
    arm_name: str,
    eef_body_idx_cache: Dict[str, int],
) -> Optional[np.ndarray]:
    arm = _get_scene_articulation(env, arm_name)
    if arm is None:
        return None

    eef_body_idx = _resolve_eef_body_idx(env, arm_name, arm, eef_body_idx_cache)
    if eef_body_idx is None:
        return None

    body_pos_w = getattr(arm.data, "body_link_pos_w", None)
    if body_pos_w is None:
        body_pos_w = getattr(arm.data, "body_pos_w", None)
    if body_pos_w is None:
        return None

    return (
        _as_numpy(body_pos_w[0, eef_body_idx], dtype=np.float32).reshape(-1) * 100.0
    )


def _get_debug_arm_names(env: DirectRLEnv) -> list[str]:
    left_arm = _get_scene_articulation(env, "left_arm")
    right_arm = _get_scene_articulation(env, "right_arm")
    if left_arm is not None or right_arm is not None:
        names = []
        if left_arm is not None:
            names.append("left_arm")
        if right_arm is not None:
            names.append("right_arm")
        return names

    for arm_name in _get_single_arm_candidates(env):
        if _get_scene_articulation(env, arm_name) is not None:
            return [arm_name]
    return []


def _get_garment_checkpoint_positions_world_cm(
    particle_object: Any,
    check_points: Sequence[int],
) -> Optional[list[list[float]]]:
    try:
        world_points, _, _, _ = particle_object.get_current_mesh_points()
        world_points = _as_numpy(world_points, dtype=np.float32)
        return (world_points[check_points] * 100.0).tolist()
    except Exception:
        pass

    try:
        world_points = (
            particle_object._cloth_prim_view.get_world_positions()
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
        world_points = _as_numpy(world_points, dtype=np.float32)
        return (world_points[check_points] * 100.0).tolist()
    except Exception:
        return None


def _log_debug_pose_snapshot(
    env: DirectRLEnv,
    step_count: int,
    eef_body_idx_cache: Dict[str, int],
) -> None:
    prefix = f"[Debug Pose][step {step_count}]"

    arm_names = _get_debug_arm_names(env)
    eef_positions_by_arm: Dict[str, np.ndarray] = {}
    if arm_names:
        logger.info(f"{prefix} EEF world positions (cm):")
        for arm_name in arm_names:
            eef_pos_cm = _get_arm_eef_world_position_cm(env, arm_name, eef_body_idx_cache)
            if eef_pos_cm is None or eef_pos_cm.size < 3:
                logger.info(f"{prefix}   {arm_name}: unavailable")
                continue
            eef_positions_by_arm[arm_name] = eef_pos_cm
            logger.info(
                f"{prefix}   {arm_name}: "
                f"[{eef_pos_cm[0]:.2f}, {eef_pos_cm[1]:.2f}, {eef_pos_cm[2]:.2f}]"
            )
    else:
        logger.warning(f"{prefix} Could not resolve any robot arm articulation for EEF logging.")

    particle_object = getattr(env, "object", None)
    check_points = getattr(particle_object, "check_points", None)
    if particle_object is None or check_points is None:
        logger.warning(f"{prefix} Garment checkpoints unavailable.")
        return

    garment_type = None
    if hasattr(env, "garment_loader") and hasattr(env, "cfg") and hasattr(env.cfg, "garment_name"):
        try:
            garment_type = env.garment_loader.get_garment_type(env.cfg.garment_name)
        except Exception:
            garment_type = None

    from lehome.utils.success_checker_chanllege import get_object_particle_position

    garment_positions_cm = get_object_particle_position(particle_object, check_points)
    if garment_positions_cm is None:
        logger.warning(f"{prefix} Failed to fetch garment checkpoint positions.")
        return
    garment_world_positions_cm = _get_garment_checkpoint_positions_world_cm(
        particle_object, check_points
    )

    if garment_type is not None:
        logger.info(
            f"{prefix} Garment checkpoints used by success checker (cm) "
            f"for garment_type={garment_type}:"
        )
    else:
        logger.info(f"{prefix} Garment checkpoints used by success checker (cm):")
    garment_positions_by_label: Dict[str, np.ndarray] = {}
    for point_idx, (mesh_idx, point_pos_cm) in enumerate(zip(check_points, garment_positions_cm)):
        point_arr = _as_numpy(point_pos_cm, dtype=np.float32).reshape(-1)
        checkpoint_name = (
            GARMENT_CHECKPOINT_LABELS[point_idx]
            if point_idx < len(GARMENT_CHECKPOINT_LABELS)
            else f"checkpoint_{point_idx}"
        )
        if point_arr.size < 3:
            logger.info(
                f"{prefix}   p[{point_idx}] {checkpoint_name} mesh_idx={mesh_idx}: unavailable"
            )
            continue
        garment_positions_by_label[checkpoint_name] = point_arr
        logger.info(
            f"{prefix}   p[{point_idx}] {checkpoint_name} mesh_idx={mesh_idx}: "
            f"[{point_arr[0]:.2f}, {point_arr[1]:.2f}, {point_arr[2]:.2f}]"
        )

    if eef_positions_by_arm:
        garment_world_positions_by_label: Dict[str, np.ndarray] = {}
        if garment_world_positions_cm is not None:
            for point_idx, point_pos_cm in enumerate(garment_world_positions_cm):
                checkpoint_name = (
                    GARMENT_CHECKPOINT_LABELS[point_idx]
                    if point_idx < len(GARMENT_CHECKPOINT_LABELS)
                    else f"checkpoint_{point_idx}"
                )
                point_arr = _as_numpy(point_pos_cm, dtype=np.float32).reshape(-1)
                if point_arr.size >= 3:
                    garment_world_positions_by_label[checkpoint_name] = point_arr

        logger.info(
            f"{prefix} Same-side EEF to garment checkpoint distances "
            "(world frame, cm):"
        )
        for arm_name, keypoint_names in ARM_KEYPOINT_DISTANCE_LABELS.items():
            eef_pos_cm = eef_positions_by_arm.get(arm_name)
            if eef_pos_cm is None:
                continue
            for keypoint_name in keypoint_names:
                keypoint_pos_cm = garment_world_positions_by_label.get(keypoint_name)
                if keypoint_pos_cm is None or keypoint_pos_cm.size < 3:
                    logger.info(f"{prefix}   {arm_name} -> {keypoint_name}: unavailable")
                    continue
                distance_cm = float(np.linalg.norm(eef_pos_cm[:3] - keypoint_pos_cm[:3]))
                logger.info(
                    f"{prefix}   {arm_name} -> {keypoint_name}: {distance_cm:.2f} cm"
                )


def _log_debug_pose_snapshot_if_enabled(
    env: DirectRLEnv,
    args: argparse.Namespace,
    debug_pose_state: Dict[str, Any],
) -> None:
    if not getattr(args, "debugging_log_pose", False):
        return

    step_count = int(debug_pose_state.get("step_count", 0))
    if step_count == 0 or step_count % DEBUG_POSE_LOG_INTERVAL == 0:
        eef_body_idx_cache = debug_pose_state.setdefault("eef_body_idx_cache", {})
        _log_debug_pose_snapshot(env, step_count, eef_body_idx_cache)

    debug_pose_state["step_count"] = step_count + 1


class DirectHDF5Recorder:
    """Streaming recorder that writes teleop episodes directly into HDF5."""

    _ACTIVE_EPISODE_NAME = "_active_demo"

    def __init__(
        self,
        env: DirectRLEnv,
        file_path: Path,
        env_args: Dict[str, Any],
        fps: int,
        is_bi_arm: bool,
        garment_info_json_path: Optional[Path] = None,
    ) -> None:
        _require_h5py()
        self._env = env
        self._file_path = Path(file_path)
        self._json_path = Path(garment_info_json_path) if garment_info_json_path else None
        self._is_bi_arm = bool(is_bi_arm)
        self._compression = "lzf"
        self._flush_steps = FLUSH_INTERVAL
        self._eef_body_idx_cache: Dict[str, int] = {}
        self._root_velocity_source_cache: Dict[str, str] = {}
        self._state_arm_names = self._resolve_state_arm_names()
        self._metadata_store: Dict[str, Any] = {}

        self._active_group: Optional[Any] = None
        self._active_groups: Dict[str, Any] = {}
        self._active_datasets: Dict[str, Any] = {}
        self._active_episode_meta: Optional[Dict[str, Any]] = None
        self._active_num_samples = 0

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
        dtype: Optional[np.dtype] = None,
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
        dtype: Optional[np.dtype] = None,
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

    def _get_scene_articulation(self, name: str) -> Optional[Any]:
        return _get_scene_articulation(self._env, name)

    def _resolve_eef_body_idx(self, arm_name: str, arm: Any) -> Optional[int]:
        return _resolve_eef_body_idx(self._env, arm_name, arm, self._eef_body_idx_cache)

    def _get_root_velocity_source(self, arm_name: str, arm: Any) -> str:
        cached = self._root_velocity_source_cache.get(arm_name)
        if cached is not None:
            return cached

        if getattr(arm.data, "root_vel_w", None) is not None:
            source = "root_vel_w"
        elif (
            getattr(arm.data, "root_lin_vel_w", None) is not None
            and getattr(arm.data, "root_ang_vel_w", None) is not None
        ):
            source = "root_lin_ang_vel_w"
        else:
            source = "scene"
        self._root_velocity_source_cache[arm_name] = source
        return source

    def _get_scene_articulation_state_fallback(self) -> Dict[str, Any]:
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
        scene_entry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        joint_pos = getattr(arm.data, "joint_pos", None)
        joint_vel = getattr(arm.data, "joint_vel", None)
        if joint_vel is None:
            joint_vel = getattr(arm.data, "joint_velocity", None)

        root_pos_w = getattr(arm.data, "root_pos_w", None)
        root_quat_w = getattr(arm.data, "root_quat_w", None)

        state: Dict[str, np.ndarray] = {}

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
    ) -> Optional[np.ndarray]:
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
        eef_pos = torch.as_tensor(
            body_pos_w[0, eef_body_idx], device=device, dtype=torch.float32
        ).unsqueeze(0)
        eef_quat = torch.as_tensor(
            body_quat_w[0, eef_body_idx], device=device, dtype=torch.float32
        ).unsqueeze(0)

        ee_pos_robot, ee_quat_robot = math_utils.subtract_frame_transforms(
            root_pos,
            root_quat,
            eef_pos,
            eef_quat,
        )
        return (
            torch.cat([ee_pos_robot, ee_quat_robot], dim=-1)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def _capture_articulation_snapshot(
        self,
    ) -> tuple[Dict[str, Dict[str, np.ndarray]], Optional[np.ndarray]]:
        scene_fallback: Optional[Dict[str, Any]] = None
        articulation_state: Dict[str, Dict[str, np.ndarray]] = {}
        ee_frame_state_by_arm: Dict[str, np.ndarray] = {}

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
                combined_ee_frame_state = np.concatenate(
                    [left_ee, right_ee], axis=0
                ).astype(np.float32, copy=False)
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
        episode_rec: Dict[str, Any] = {
            "object_initial_pose": _to_json_compatible(pose_value),
        }
        scale_value = self._active_episode_meta.get("scale")
        if scale_value is not None:
            episode_rec["scale"] = _to_json_compatible(scale_value)
        self._metadata_store.setdefault(garment_name, {})[episode_key] = episode_rec

    def begin_episode(
        self,
        episode_index: int,
        object_initial_pose: Optional[Dict[str, Any]],
        garment_name: Optional[str],
        scale: Optional[Any],
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

    def _append_rgb_if_available(self, target_path: str, frame_value: Any) -> None:
        if frame_value is None:
            return
        rgb = _as_numpy(frame_value, dtype=np.uint8)
        if rgb.ndim == 4:
            rgb = rgb.squeeze(0)
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
            return
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        self._append_dataset(target_path, rgb, dtype=np.uint8)

    def append_frame(self, frame: Dict[str, Any]) -> None:
        if self._active_group is None:
            raise RuntimeError("Cannot append a frame without an active episode.")

        joint_action = _as_numpy(frame["action"], dtype=np.float32).reshape(-1)
        joint_pos = _as_numpy(frame["observation.state"], dtype=np.float32).reshape(-1)
        articulation_state, ee_frame_state = self._capture_articulation_snapshot()

        self._append_dataset("actions", joint_action, dtype=np.float32)
        self._append_dataset("processed_actions", joint_action, dtype=np.float32)
        self._append_dataset("obs/actions", joint_action, dtype=np.float32)

        if self._is_bi_arm:
            left_joint_pos = joint_pos[:6]
            right_joint_pos = joint_pos[6:12]
            left_target = joint_action[:6]
            right_target = joint_action[6:12]
            left_state = articulation_state.get("left_arm", {})
            right_state = articulation_state.get("right_arm", {})
            left_joint_vel = _as_numpy(left_state.get("joint_velocity", np.zeros(6)), dtype=np.float32).reshape(-1)
            right_joint_vel = _as_numpy(right_state.get("joint_velocity", np.zeros(6)), dtype=np.float32).reshape(-1)

            self._append_dataset("obs/left_joint_pos", left_joint_pos, dtype=np.float32)
            self._append_dataset("obs/right_joint_pos", right_joint_pos, dtype=np.float32)
            self._append_dataset("obs/left_joint_pos_target", left_target, dtype=np.float32)
            self._append_dataset("obs/right_joint_pos_target", right_target, dtype=np.float32)
            self._append_dataset("obs/left_joint_pos_rel", left_target - left_joint_pos, dtype=np.float32)
            self._append_dataset("obs/right_joint_pos_rel", right_target - right_joint_pos, dtype=np.float32)
            self._append_dataset("obs/left_joint_vel", left_joint_vel, dtype=np.float32)
            self._append_dataset("obs/right_joint_vel", right_joint_vel, dtype=np.float32)
            self._append_dataset("obs/left_joint_vel_rel", left_joint_vel, dtype=np.float32)
            self._append_dataset("obs/right_joint_vel_rel", right_joint_vel, dtype=np.float32)

            if ee_frame_state is not None and ee_frame_state.shape[0] >= 14:
                self._append_dataset("obs/left_ee_frame_state", ee_frame_state[:7], dtype=np.float32)
                self._append_dataset("obs/right_ee_frame_state", ee_frame_state[7:14], dtype=np.float32)

            self._append_rgb_if_available("obs/top", frame.get("observation.images.top_rgb"))
            self._append_rgb_if_available("obs/left_wrist", frame.get("observation.images.left_rgb"))
            self._append_rgb_if_available("obs/right_wrist", frame.get("observation.images.right_rgb"))
        else:
            joint_target = joint_action[:6]
            single_joint_pos = joint_pos[:6]
            arm_name = self._state_arm_names[0]
            arm_state = articulation_state.get(arm_name, {})
            joint_vel = _as_numpy(arm_state.get("joint_velocity", np.zeros(6)), dtype=np.float32).reshape(-1)

            self._append_dataset("obs/joint_pos", single_joint_pos, dtype=np.float32)
            self._append_dataset("obs/joint_pos_target", joint_target, dtype=np.float32)
            self._append_dataset("obs/joint_pos_rel", joint_target - single_joint_pos, dtype=np.float32)
            self._append_dataset("obs/joint_vel", joint_vel, dtype=np.float32)
            self._append_dataset("obs/joint_vel_rel", joint_vel, dtype=np.float32)

            if ee_frame_state is not None and ee_frame_state.shape[0] >= 7:
                self._append_dataset("obs/ee_frame_state", ee_frame_state[:7], dtype=np.float32)

            self._append_rgb_if_available("obs/top", frame.get("observation.images.top_rgb"))
            self._append_rgb_if_available("obs/wrist", frame.get("observation.images.wrist_rgb"))

        for arm_name, state in articulation_state.items():
            for key, value in state.items():
                self._append_dataset(
                    f"states/articulation/{arm_name}/{key}",
                    value,
                    dtype=np.float32,
                )

        self._active_num_samples += 1
        self._active_group.attrs["num_samples"] = np.int64(self._active_num_samples)
        if self._active_num_samples % self._flush_steps == 0:
            self._file.flush()

    def add_frame(self, frame: Dict[str, Any]) -> None:
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


def validate_task_and_device(args: argparse.Namespace) -> None:
    """Validate that task name matches the teleop device configuration.

    Args:
        args: Command-line arguments containing task and teleop_device.

    Raises:
        ValueError: If task is not specified.
        AssertionError: If task and device configuration mismatch.
    """
    if args.task is None:
        raise ValueError("Please specify --task.")
    if "Bi" in args.task:
        assert (
            args.teleop_device == "bi-so101leader"
            or args.teleop_device == "bi-keyboard"
        ), "Only support bi-so101leader or bi-keyboard for bi-arm task"
    else:
        assert (
            args.teleop_device == "so101leader" or args.teleop_device == "keyboard"
        ), "Only support so101leader or keyboard for single-arm task"


def create_teleop_interface(
    env: DirectRLEnv, args: argparse.Namespace
) -> Union[Se3Keyboard, SO101Leader, BiSO101Leader, BiKeyboard]:
    """Create teleoperation interface based on device type.

    Args:
        env: Environment instance.
        args: Command-line arguments containing teleop_device and related config.

    Returns:
        Teleoperation interface instance.

    Raises:
        ValueError: If teleop_device is invalid.
    """
    if args.teleop_device == "keyboard":
        return Se3Keyboard(env, sensitivity=0.25 * args.sensitivity)
    if args.teleop_device == "so101leader":
        return SO101Leader(env, port=args.port, recalibrate=args.recalibrate)
    if args.teleop_device == "bi-so101leader":
        return BiSO101Leader(
            env,
            left_port=args.left_arm_port,
            right_port=args.right_arm_port,
            recalibrate=args.recalibrate,
        )
    if args.teleop_device == "bi-keyboard":
        return BiKeyboard(env, sensitivity=0.25 * args.sensitivity)
    raise ValueError(
        f"Invalid device interface '{args.teleop_device}'. "
        f"Supported: 'keyboard', 'so101leader', 'bi-so101leader', 'bi-keyboard'."
    )


def register_teleop_callbacks(
    teleop_interface: Any, recording_enabled: bool = False
) -> Dict[str, bool]:
    """Register callback functions for teleoperation control keys.

    Key bindings:
        S: Start recording
        N: Mark current episode as successful (only active during recording)
        D: Discard current episode and re-record (only active during recording)
        ESC: Abort entire recording process and clear buffer

    Args:
        teleop_interface: Teleoperation interface instance.
        recording_enabled: Whether recording is enabled. If False, N/D keys are
            disabled in idle phase.

    Returns:
        Dictionary of status flags for recording control.
    """
    flags = {
        "start": False,  # S: Start recording
        "success": False,  # N: Success/early termination of current episode
        "remove": False,  # D: Discard current episode
        "abort": False,  # ESC: Abort entire recording process, clear buffer
    }

    def on_start():
        flags["start"] = True
        logger.info("[S] Recording started!")

    def on_success():
        if not recording_enabled or not flags["start"]:
            # Ignore N key in idle phase (before recording starts)
            logger.debug("[N] Ignored (recording not started yet)")
            return
        flags["success"] = True
        logger.info("[N] Mark the current episode as successful.")

    def on_remove():
        if not recording_enabled or not flags["start"]:
            # Ignore D key in idle phase (before recording starts)
            logger.debug("[D] Ignored (recording not started yet)")
            return
        flags["remove"] = True
        logger.info("[D] Discard the current episode and re-record.")

    def on_abort():
        flags["abort"] = True
        logger.warning("[ESC] Abort recording, clearing the current episode buffer...")

    teleop_interface.add_callback("S", on_start)
    teleop_interface.add_callback("N", on_success)
    teleop_interface.add_callback("D", on_remove)
    teleop_interface.add_callback("ESCAPE", on_abort)

    return flags


def _resolve_recording_output_paths(dataset_root: str) -> Tuple[Path, Path]:
    """Resolve HDF5 output path and external garment-info json path."""
    root_path = Path(dataset_root)
    if root_path.suffix.lower() in {".hdf5", ".h5"}:
        hdf5_path = root_path
    else:
        run_dir = get_next_experiment_path_with_gap(root_path)
        hdf5_path = run_dir / "teleop_dataset.hdf5"
    json_path = hdf5_path.parent / "meta" / "garment_info.json"
    return hdf5_path, json_path


def create_dataset_if_needed(
    env: DirectRLEnv,
    args: argparse.Namespace,
) -> Optional[DirectHDF5Recorder]:
    """Create direct HDF5 dataset writer if recording is enabled.

    Args:
        env: Environment instance.
        args: Command-line arguments containing recording configuration.

    Returns:
        DirectHDF5Recorder instance or None if recording is disabled.
    """
    if not args.enable_record:
        return None

    is_bi_arm = ("Bi" in (args.task or "")) or (
        getattr(args, "teleop_device", "") or ""
    ).startswith("bi-")
    action_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    if is_bi_arm:
        joint_names = [f"left_{n}" for n in action_names] + [f"right_{n}" for n in action_names]
    else:
        joint_names = action_names

    hdf5_path, json_path = _resolve_recording_output_paths(
        getattr(args, "dataset_root", "Datasets/record")
    )

    env_args: Dict[str, Any] = {"env_name": args.task or "", "type": 2}
    if getattr(args, "garment_name", None):
        env_args["garment_name"] = args.garment_name
    if getattr(args, "garment_version", None):
        env_args["garment_version"] = args.garment_version
    if getattr(args, "task_description", None):
        env_args["task_description"] = args.task_description
    env_args["joint_names"] = joint_names

    if getattr(args, "record_ee_pose", False):
        logger.warning(
            "--record_ee_pose is ignored by scripts/mimicgen/dataset_record_hdf5.py; "
            "the MimicGen pipeline consumes obs/*_ee_frame_state directly."
        )

    dataset = DirectHDF5Recorder(
        env=env,
        file_path=hdf5_path,
        env_args=env_args,
        fps=int(getattr(args, "step_hz", 30)),
        is_bi_arm=is_bi_arm,
        garment_info_json_path=json_path,
    )
    logger.info(f"Recording direct HDF5 file: {hdf5_path}")

    return dataset


def run_idle_phase(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    count_render: int,
    debug_pose_state: Dict[str, Any],
    control_state: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Run idle phase before recording starts.

    Handles environment preparation, stabilization, and waits for user to press
    S key to start recording.

    Args:
        env: Environment instance.
        teleop_interface: Teleoperation interface.
        args: Command-line arguments.
        count_render: Current render count.

    Returns:
        Tuple of (object_initial_pose, updated_count_render).
    """
    actions = teleop_interface.advance()
    object_initial_pose = None

    if count_render == 0:
        logger.info("[Idle Phase] Initializing observations...")
        env.initialize_obs()
        count_render += 1

        logger.info("[Idle Phase] Stabilizing garment after initialization...")
        stabilize_garment_after_reset(env, args)
        object_initial_pose = _safe_get_all_pose(env)
        if object_initial_pose is not None:
            control_state["cached_object_initial_pose"] = object_initial_pose
        logger.info("[Idle Phase] Ready for recording")

    if actions is None:
        current_obs = env._get_observations()
        maintain_action = _get_or_build_maintain_action(
            env,
            args,
            control_state,
            current_obs,
        )
        env.step(maintain_action)
        env.render()
        if object_initial_pose is None:
            object_initial_pose = control_state.get("cached_object_initial_pose")
            if object_initial_pose is None:
                object_initial_pose = _safe_get_all_pose(env)
    else:
        env.step(actions)
        object_initial_pose = _safe_get_all_pose(env)

    if object_initial_pose is not None:
        control_state["cached_object_initial_pose"] = object_initial_pose

    _log_debug_pose_snapshot_if_enabled(env, args, debug_pose_state)

    return object_initial_pose, count_render


def run_recording_phase(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    flags: Dict[str, bool],
    dataset: DirectHDF5Recorder,
    initial_object_pose: Optional[Dict[str, Any]],
    control_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run recording phase after S key is pressed and recording is enabled.

    Records episodes until num_episode is reached. Each episode can be marked as
    successful (N key), discarded (D key), or aborted (ESC key).

    Args:
        env: Environment instance.
        teleop_interface: Teleoperation interface.
        args: Command-line arguments.
        flags: Status flags dictionary.
        dataset: Direct HDF5 recorder instance.
        initial_object_pose: Initial object pose dictionary.

    Returns:
        Final object initial pose dictionary.
    """
    if control_state is None:
        control_state = {}

    episode_index = 0
    object_initial_pose = initial_object_pose

    # Ensure we have a valid initial pose for the first episode
    if object_initial_pose is None:
        object_initial_pose = control_state.get("cached_object_initial_pose")
    if object_initial_pose is None:
        object_initial_pose = _safe_get_all_pose(env)
    if object_initial_pose is not None:
        control_state["cached_object_initial_pose"] = object_initial_pose

    while episode_index < args.num_episode:
        # Check if recording should be aborted
        if flags["abort"]:
            dataset.discard_episode()
            dataset.finalize()
            logger.warning(f"Recording aborted, completed {episode_index} episodes")
            return object_initial_pose

        flags["success"] = False
        flags["remove"] = False
        episode_step_count = 0
        episode_discarded = False
        garment_name, scale = _get_episode_garment_meta(env)
        if object_initial_pose is None:
            object_initial_pose = control_state.get("cached_object_initial_pose")
        if object_initial_pose is None:
            object_initial_pose = _safe_get_all_pose(env)
        if object_initial_pose is not None:
            control_state["cached_object_initial_pose"] = object_initial_pose

        dataset.begin_episode(
            episode_index=episode_index,
            object_initial_pose=object_initial_pose,
            garment_name=garment_name,
            scale=scale,
        )

        if args.log_success:
            _log_success_result(
                env,
                episode_index=episode_index,
                step_in_episode=episode_step_count,
                context="episode_start",
            )

        # Loop within a single episode
        while not flags["success"]:
            # Check if recording should be aborted
            if flags["abort"]:
                dataset.discard_episode()
                dataset.finalize()
                logger.warning(f"Recording aborted, completed {episode_index} episodes")
                return object_initial_pose

            try:
                actions = teleop_interface.advance()
            except Exception as e:
                logger.error(f"[Recording] Error in teleop interface: {e}")
                traceback.print_exc()
                actions = None

            if actions is None:
                env.render()
            else:
                env.step(actions)

            episode_step_count += 1
            if args.log_success and episode_step_count % SUCCESS_LOG_INTERVAL == 0:
                _log_success_result(
                    env,
                    episode_index=episode_index,
                    step_in_episode=episode_step_count,
                    context="periodic_check",
                )

            observations = env._get_observations()
            _, truncated = env._get_dones()
            frame: Dict[str, Any] = {
                "action": observations["action"],
                "observation.state": observations["observation.state"],
            }
            for key in (
                "observation.images.top_rgb",
                "observation.images.left_rgb",
                "observation.images.right_rgb",
                "observation.images.wrist_rgb",
            ):
                if key in observations:
                    frame[key] = observations[key]

            dataset.append_frame(frame)

            if truncated or flags["remove"]:
                dataset.discard_episode()
                logger.info(f"Re-recording episode {episode_index}")
                try:
                    env.reset()
                    stabilize_garment_after_reset(env, args)
                    object_initial_pose = _safe_get_all_pose(env)
                    if object_initial_pose is not None:
                        control_state["cached_object_initial_pose"] = object_initial_pose
                except Exception as e:
                    logger.error(
                        f"[Recording] Failed to reset environment during re-recording: {e}"
                    )
                    traceback.print_exc()
                    object_initial_pose = _safe_get_all_pose(env)
                flags["remove"] = False
                episode_discarded = True
                break

        if episode_discarded:
            continue

        _log_episode_success_snapshot(env, episode_index, episode_step_count)

        save_start_time = time.time()
        logger.info(f"[Recording] Saving episode {episode_index}...")
        try:
            dataset.finalize_episode()
            save_duration = time.time() - save_start_time
            logger.info(
                f"[Recording] Episode {episode_index} saved (took {save_duration:.1f}s)"
            )
        except Exception as e:
            logger.error(f"[Recording] Failed to save episode {episode_index}: {e}")
            traceback.print_exc()
            dataset.discard_episode()
            raise

        episode_index += 1
        logger.info(
            f"Episode {episode_index - 1} completed, progress: {episode_index}/{args.num_episode}"
        )

        try:
            env.reset()
            stabilize_garment_after_reset(env, args)
        except Exception as e:
            logger.error(f"[Recording] Failed to reset environment: {e}")
            traceback.print_exc()

        object_initial_pose = _safe_get_all_pose(env)
        if object_initial_pose is not None:
            control_state["cached_object_initial_pose"] = object_initial_pose
    dataset.finalize()
    logger.info(f"All {args.num_episode} episodes recording completed!")
    return object_initial_pose


def _log_episode_success_snapshot(
    env: DirectRLEnv,
    episode_index: int,
    episode_step_count: Optional[int] = None,
) -> None:
    """Emit one explicit success summary for the current episode state."""
    success = _log_success_result(
        env,
        episode_index=episode_index,
        step_in_episode=episode_step_count,
        context="episode_end",
    )
    if success is None:
        return
    logger.info(
        f"[Recording] Episode {episode_index} success snapshot: "
        f"{'Success ✓' if success else 'Failed ✗'}"
    )


def run_live_control_without_record(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    debug_pose_state: Dict[str, Any],
    control_state: Dict[str, Any],
) -> None:
    """Run live teleoperation control without recording.

    Handles the case when S key is pressed but recording is not enabled.
    Performs simple teleoperation control without writing to dataset.

    Args:
        env: Environment instance.
        teleop_interface: Teleoperation interface.
        args: Command-line arguments.
    """
    actions = teleop_interface.advance()

    if actions is None:
        current_obs = env._get_observations()
        maintain_action = _get_or_build_maintain_action(
            env,
            args,
            control_state,
            current_obs,
        )
        env.step(maintain_action)
        env.render()
    else:
        env.step(actions)

    _log_debug_pose_snapshot_if_enabled(env, args, debug_pose_state)

    if args.log_success:
        _ = env._get_success()


def record_dataset(args: argparse.Namespace, simulation_app: SimulationApp) -> None:
    """Record dataset."""
    # Get device configuration (default to "cpu" for compatibility)
    device = getattr(args, "device", "cpu")

    env_cfg = parse_env_cfg(
        args.task,
        device=device,
    )
    task_name = args.task

    env_cfg.garment_name = args.garment_name
    env_cfg.garment_version = args.garment_version
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    if args.use_random_seed:
        env_cfg.use_random_seed = True
        logger.info("Using random seed (no fixed seed)")
    else:
        env_cfg.use_random_seed = False
        env_cfg.random_seed = args.seed
        logger.info(f"Using fixed random seed: {args.seed}")

    env: DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    dataset: Optional[DirectHDF5Recorder] = None
    teleop_interface = create_teleop_interface(env, args)
    flags = register_teleop_callbacks(
        teleop_interface, recording_enabled=args.enable_record
    )
    teleop_interface.reset()
    dataset = create_dataset_if_needed(env, args)
    count_render = 0
    printed_instructions = False
    idle_frame_counter = 0
    object_initial_pose: Optional[Dict[str, Any]] = None
    debug_pose_state: Dict[str, Any] = {"step_count": 0, "eef_body_idx_cache": {}}
    control_state: Dict[str, Any] = {"maintain_action": None}

    if getattr(args, "debugging_log_pose", False):
        logger.info(
            "[Debug Pose] Enabled. Logging EEF and garment checkpoint positions in cm "
            f"at step 0 and every {DEBUG_POSE_LOG_INTERVAL} sim steps."
        )

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if not flags["start"]:
                    pose, count_render = run_idle_phase(
                        env,
                        teleop_interface,
                        args,
                        count_render,
                        debug_pose_state,
                        control_state,
                    )
                    if pose is not None:
                        object_initial_pose = pose

                    if count_render > 0:
                        idle_frame_counter += 1
                        if idle_frame_counter == 100 and not printed_instructions:
                            logger.info("=" * 60)
                            logger.info("🎮 CONTROL INSTRUCTIONS 🎮")
                            logger.info("=" * 60)
                            logger.info(str(teleop_interface))
                            logger.info("=" * 60 + "\n\n")
                            printed_instructions = True
                elif args.enable_record and dataset is not None:
                    object_initial_pose = run_recording_phase(
                        env,
                        teleop_interface,
                        args,
                        flags,
                        dataset,
                        object_initial_pose,
                        control_state,
                    )
                    break
                else:
                    run_live_control_without_record(
                        env,
                        teleop_interface,
                        args,
                        debug_pose_state,
                        control_state,
                    )
    except KeyboardInterrupt:
        logger.warning("\n[Ctrl+C] Interrupt signal detected")
        # If Ctrl+C is pressed during recording, clear the current buffer
        if args.enable_record and dataset is not None and flags["start"]:
            logger.info("Clearing current episode buffer...")
            dataset.discard_episode()
            logger.info("Buffer cleared, dataset remains intact")
            dataset.finalize()
            logger.info("Dataset saved")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

    finally:
        if dataset is not None:
            dataset.finalize()
        env.close()
