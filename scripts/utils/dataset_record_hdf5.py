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
from lehome.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from lehome.utils.record import (
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from lehome.utils.logger import get_logger
from lehome.tasks.fold_cloth.checkpoint_mappings import (
    ARM_KEYPOINT_GROUPS,
    CHECKPOINT_LABELS,
)

from .common import stabilize_garment_after_reset

logger = get_logger(__name__)
SUCCESS_LOG_INTERVAL = 50
DEBUG_POSE_LOG_INTERVAL = 50
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


def _maybe_log_debug_pose_snapshot(
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
    """Direct recorder that writes teleop episodes into one HDF5 file."""

    def __init__(
        self,
        env: DirectRLEnv,
        file_path: Path,
        env_args: Dict[str, Any],
        fps: int,
        is_bi_arm: bool,
    ) -> None:
        _require_h5py()
        self._env = env
        self._file_path = Path(file_path)
        self._is_bi_arm = bool(is_bi_arm)
        self._episode_frames: list[Dict[str, Any]] = []
        self._episode_scene_states: list[Dict[str, Any]] = []
        self._pending_episode_meta: Optional[Dict[str, Any]] = None
        self._eef_body_idx_cache: Dict[str, int] = {}

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

    def set_pending_episode_meta(
        self,
        episode_index: int,
        object_initial_pose: Optional[Dict[str, Any]],
        garment_name: Optional[str],
        scale: Optional[Any],
    ) -> None:
        self._pending_episode_meta = {
            "episode_index": int(episode_index),
            "object_initial_pose": _to_json_compatible(object_initial_pose),
            "garment_name": str(garment_name) if garment_name else "unknown",
            "scale": _to_json_compatible(scale),
        }

    def add_frame(self, frame: Dict[str, Any]) -> None:
        frame_to_store = dict(frame)
        ee_frame_state = self._compute_live_ee_frame_state()
        if ee_frame_state is not None:
            frame_to_store["observation.ee_frame_state"] = ee_frame_state
        self._episode_frames.append(frame_to_store)
        self._episode_scene_states.append(self._capture_scene_state())

    def clear_episode_buffer(self) -> None:
        self._episode_frames.clear()
        self._episode_scene_states.clear()
        self._pending_episode_meta = None

    def save_episode(self) -> None:
        if len(self._episode_frames) == 0:
            return

        num_samples = len(self._episode_frames)
        demo_name = f"demo_{self._demo_count}"
        if demo_name in self._data_group:
            raise ValueError(f"Episode group '{demo_name}' already exists in HDF5.")
        demo_group = self._data_group.create_group(demo_name)

        demo_group.attrs["num_samples"] = np.int64(num_samples)
        demo_group.attrs["seed"] = np.int64(random.randint(0, 2**31 - 1))
        demo_group.attrs["success"] = True

        # Top-level actions (joint-space) and processed actions.
        joint_actions = self._stack_frame_key("action", dtype=np.float32)
        demo_group.create_dataset("actions", data=joint_actions, compression="gzip")
        demo_group.create_dataset("processed_actions", data=joint_actions, compression="gzip")

        self._write_obs_group(demo_group, joint_actions)
        self._write_state_groups(demo_group)

        self._demo_count += 1
        self._total_samples += num_samples
        self._data_group.attrs["total"] = np.int64(self._total_samples)
        self._data_group.attrs["num_episodes"] = np.int64(self._demo_count)
        self._file.flush()
        self.clear_episode_buffer()

    def finalize(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def _capture_scene_state(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"articulation": {}, "rigid_object": {}}
        try:
            state = self._env.scene.get_state(is_relative=False)
        except Exception:
            return out

        for family in ("articulation", "rigid_object"):
            family_state = state.get(family, {})
            if not isinstance(family_state, dict):
                continue
            for name, entry in family_state.items():
                if not isinstance(entry, dict):
                    continue
                entity_out: Dict[str, np.ndarray] = {}
                if family == "articulation":
                    keys = ("root_pose", "root_velocity", "joint_position", "joint_velocity")
                else:
                    keys = ("root_pose", "root_velocity")
                for key in keys:
                    if key not in entry:
                        continue
                    arr = _as_numpy(entry[key], dtype=np.float32)
                    # Keep only env 0 because this recorder supports num_envs=1.
                    if arr.ndim >= 2:
                        arr = arr[0]
                    entity_out[key] = arr
                if entity_out:
                    out[family][str(name)] = entity_out
        return out

    def _get_scene_articulation(self, name: str) -> Optional[Any]:
        scene = getattr(self._env, "scene", None)
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

    def _get_single_arm_candidates(self) -> list[str]:
        scene = getattr(self._env, "scene", None)
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

    def _resolve_eef_body_idx(self, arm_name: str, arm: Any) -> Optional[int]:
        if arm_name in self._eef_body_idx_cache:
            return self._eef_body_idx_cache[arm_name]

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
                    self._eef_body_idx_cache[arm_name] = body_idx
                    return body_idx
            except Exception:
                continue

        body_positions = getattr(arm.data, "body_link_pos_w", None)
        if body_positions is None:
            body_positions = getattr(arm.data, "body_pos_w", None)
        if body_positions is None:
            return None

        body_idx = int(body_positions.shape[1] - 1)
        self._eef_body_idx_cache[arm_name] = body_idx
        return body_idx

    def _compute_arm_ee_frame_state(self, arm_name: str) -> Optional[np.ndarray]:
        arm = self._get_scene_articulation(arm_name)
        if arm is None:
            return None

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

    def _compute_live_ee_frame_state(self) -> Optional[np.ndarray]:
        if self._is_bi_arm:
            left_ee = self._compute_arm_ee_frame_state("left_arm")
            right_ee = self._compute_arm_ee_frame_state("right_arm")
            if left_ee is None or right_ee is None:
                return None
            return np.concatenate([left_ee, right_ee], axis=0).astype(np.float32, copy=False)

        for arm_name in self._get_single_arm_candidates():
            ee_frame_state = self._compute_arm_ee_frame_state(arm_name)
            if ee_frame_state is not None:
                return ee_frame_state
        return None

    def _stack_frame_key(self, key: str, dtype: np.dtype) -> np.ndarray:
        stacked: list[np.ndarray] = []
        for frame in self._episode_frames:
            if key not in frame:
                raise KeyError(f"Missing frame key '{key}' while saving episode.")
            stacked.append(_as_numpy(frame[key], dtype=dtype))
        return np.stack(stacked, axis=0)

    def _stack_optional_frame_key(self, key: str, dtype: np.dtype) -> Optional[np.ndarray]:
        stacked: list[np.ndarray] = []
        for frame in self._episode_frames:
            if key not in frame:
                return None
            stacked.append(_as_numpy(frame[key], dtype=dtype))
        return np.stack(stacked, axis=0)

    def _convert_ee_pose_to_ee_frame_state(
        self, ee_pose: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if ee_pose is None or ee_pose.ndim != 2:
            return None

        if (not self._is_bi_arm) and ee_pose.shape[1] == 8:
            return np.concatenate(
                [ee_pose[:, :3], ee_pose[:, 6:7], ee_pose[:, 3:6]],
                axis=1,
            ).astype(np.float32, copy=False)
        if (not self._is_bi_arm) and ee_pose.shape[1] == 7:
            return ee_pose.astype(np.float32, copy=False)

        if self._is_bi_arm and ee_pose.shape[1] == 16:
            left = np.concatenate([ee_pose[:, :3], ee_pose[:, 6:7], ee_pose[:, 3:6]], axis=1)
            right = np.concatenate(
                [ee_pose[:, 8:11], ee_pose[:, 14:15], ee_pose[:, 11:14]],
                axis=1,
            )
            return np.concatenate([left, right], axis=1).astype(np.float32, copy=False)
        if self._is_bi_arm and ee_pose.shape[1] == 14:
            return ee_pose.astype(np.float32, copy=False)

        return None

    def _write_obs_group(self, demo_group: Any, joint_actions: np.ndarray) -> None:
        obs_group = demo_group.create_group("obs")
        obs_group.create_dataset("actions", data=joint_actions, compression="gzip")

        joint_pos = self._stack_frame_key("observation.state", dtype=np.float32)
        joint_vel_left, joint_vel_right, joint_vel_single = self._extract_joint_velocities(
            num_frames=joint_pos.shape[0]
        )

        if self._is_bi_arm:
            left_joint_pos = joint_pos[:, :6]
            right_joint_pos = joint_pos[:, 6:12]
            left_target = joint_actions[:, :6]
            right_target = joint_actions[:, 6:12]

            obs_group.create_dataset("left_joint_pos", data=left_joint_pos, compression="gzip")
            obs_group.create_dataset("right_joint_pos", data=right_joint_pos, compression="gzip")
            obs_group.create_dataset("left_joint_pos_target", data=left_target, compression="gzip")
            obs_group.create_dataset("right_joint_pos_target", data=right_target, compression="gzip")
            obs_group.create_dataset(
                "left_joint_pos_rel", data=(left_target - left_joint_pos), compression="gzip"
            )
            obs_group.create_dataset(
                "right_joint_pos_rel", data=(right_target - right_joint_pos), compression="gzip"
            )
            obs_group.create_dataset("left_joint_vel", data=joint_vel_left, compression="gzip")
            obs_group.create_dataset("right_joint_vel", data=joint_vel_right, compression="gzip")
            obs_group.create_dataset("left_joint_vel_rel", data=joint_vel_left, compression="gzip")
            obs_group.create_dataset("right_joint_vel_rel", data=joint_vel_right, compression="gzip")
        else:
            single_joint_pos = joint_pos[:, :6]
            single_target = joint_actions[:, :6]
            obs_group.create_dataset("joint_pos", data=single_joint_pos, compression="gzip")
            obs_group.create_dataset("joint_pos_target", data=single_target, compression="gzip")
            obs_group.create_dataset(
                "joint_pos_rel", data=(single_target - single_joint_pos), compression="gzip"
            )
            obs_group.create_dataset("joint_vel", data=joint_vel_single, compression="gzip")
            obs_group.create_dataset("joint_vel_rel", data=joint_vel_single, compression="gzip")

        self._write_rgb_if_available(
            obs_group,
            target_key="top",
            frame_key="observation.images.top_rgb",
        )

        if self._is_bi_arm:
            self._write_rgb_if_available(
                obs_group,
                target_key="left_wrist",
                frame_key="observation.images.left_rgb",
            )
            self._write_rgb_if_available(
                obs_group,
                target_key="right_wrist",
                frame_key="observation.images.right_rgb",
            )
        else:
            self._write_rgb_if_available(
                obs_group,
                target_key="wrist",
                frame_key="observation.images.wrist_rgb",
            )

        ee_frame_state = self._stack_optional_frame_key(
            "observation.ee_frame_state", dtype=np.float32
        )
        if ee_frame_state is None:
            ee_pose = self._stack_optional_frame_key("observation.ee_pose", dtype=np.float32)
            ee_frame_state = self._convert_ee_pose_to_ee_frame_state(ee_pose)

        if ee_frame_state is not None and ee_frame_state.ndim == 2:
            if (not self._is_bi_arm) and ee_frame_state.shape[1] >= 7:
                obs_group.create_dataset(
                    "ee_frame_state", data=ee_frame_state[:, :7], compression="gzip"
                )
            elif self._is_bi_arm and ee_frame_state.shape[1] >= 14:
                obs_group.create_dataset(
                    "left_ee_frame_state", data=ee_frame_state[:, :7], compression="gzip"
                )
                obs_group.create_dataset(
                    "right_ee_frame_state", data=ee_frame_state[:, 7:14], compression="gzip"
                )

    def _write_rgb_if_available(self, obs_group: Any, target_key: str, frame_key: str) -> None:
        rgb = self._stack_optional_frame_key(frame_key, dtype=np.uint8)
        if rgb is None:
            return
        if rgb.ndim == 4 and rgb.shape[-1] in (3, 4):
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            obs_group.create_dataset(target_key, data=rgb, compression="gzip")

    def _extract_joint_velocities(
        self, num_frames: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        left = np.zeros((num_frames, 6), dtype=np.float32)
        right = np.zeros((num_frames, 6), dtype=np.float32)
        single = np.zeros((num_frames, 6), dtype=np.float32)

        for i, state in enumerate(self._episode_scene_states):
            articulation = state.get("articulation", {})
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

    def _write_state_groups(self, demo_group: Any) -> None:
        initial_state_group = demo_group.create_group("initial_state")
        states_group = demo_group.create_group("states")

        self._write_state_family(
            initial_state_group,
            family_name="articulation",
            is_initial=True,
        )
        self._write_initial_garment_state_if_available(initial_state_group)
        self._write_state_family(
            initial_state_group,
            family_name="rigid_object",
            is_initial=True,
        )
        self._write_state_family(
            states_group,
            family_name="articulation",
            is_initial=False,
        )
        self._write_state_family(
            states_group,
            family_name="rigid_object",
            is_initial=False,
        )

    def _write_state_family(self, parent_group: Any, family_name: str, is_initial: bool) -> None:
        collected = self._collect_family_state_series(family_name)
        if not collected:
            return

        family_group = parent_group.create_group(family_name)

        for entity_name, key_series in sorted(collected.items()):
            entity_group = family_group.create_group(entity_name)
            for key, values in sorted(key_series.items()):
                if len(values) == 0:
                    continue
                if is_initial:
                    data = values[0][None, ...]
                else:
                    data = np.stack(values, axis=0)
                entity_group.create_dataset(key, data=data.astype(np.float32), compression="gzip")

    def _collect_family_state_series(self, family_name: str) -> Dict[str, Dict[str, list[np.ndarray]]]:
        out: Dict[str, Dict[str, list[np.ndarray]]] = {}
        for scene_state in self._episode_scene_states:
            family_state = scene_state.get(family_name, {})
            if not isinstance(family_state, dict):
                continue
            for entity_name, entity_data in family_state.items():
                if not isinstance(entity_data, dict):
                    continue
                if entity_name not in out:
                    out[entity_name] = {}
                for key, value in entity_data.items():
                    out[entity_name].setdefault(key, []).append(
                        _as_numpy(value, dtype=np.float32)
                    )
        return out

    def _write_initial_garment_state_if_available(self, initial_state_group: Any) -> None:
        if self._pending_episode_meta is None:
            return

        garment_name = str(self._pending_episode_meta.get("garment_name") or "Garment")
        object_initial_pose = self._pending_episode_meta.get("object_initial_pose")
        pose_value: Any = object_initial_pose
        if isinstance(object_initial_pose, dict):
            if "Garment" in object_initial_pose:
                pose_value = object_initial_pose["Garment"]
            elif garment_name in object_initial_pose:
                pose_value = object_initial_pose[garment_name]
            else:
                pose_value = next(iter(object_initial_pose.values()), None)

        if pose_value is None:
            return

        garment_group = initial_state_group.create_group("garment")
        garment_entry_group = garment_group.create_group(garment_name)
        initial_pose = np.atleast_1d(_as_numpy(pose_value, dtype=np.float32))
        garment_entry_group.create_dataset(
            "initial_pose",
            data=initial_pose[None, ...],
            compression="gzip",
        )

        scale_value = self._pending_episode_meta.get("scale")
        if scale_value is not None:
            scale = np.atleast_1d(_as_numpy(scale_value, dtype=np.float32))
            garment_entry_group.create_dataset(
                "scale",
                data=scale[None, ...],
                compression="gzip",
            )

    def _write_episode_meta_group_if_available(self, demo_group: Any) -> None:
        if self._pending_episode_meta is None:
            return
        meta_group = demo_group.create_group("meta")
        episode_index = str(self._pending_episode_meta["episode_index"])
        garment_name = str(self._pending_episode_meta["garment_name"])
        payload: Dict[str, Any] = {
            garment_name: {
                episode_index: {
                    "object_initial_pose": _to_json_compatible(
                        self._pending_episode_meta.get("object_initial_pose")
                    )
                }
            }
        }
        scale = self._pending_episode_meta.get("scale")
        if scale is not None:
            payload[garment_name][episode_index]["scale"] = _to_json_compatible(scale)
        content = json.dumps(payload, ensure_ascii=False)
        meta_group.create_dataset(
            "garment_info.json",
            data=content,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


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
) -> Tuple[Optional[DirectHDF5Recorder], Optional[Path], Optional[Any], bool]:
    """Create direct HDF5 dataset writer if recording is enabled.

    Args:
        args: Command-line arguments containing recording configuration.

    Returns:
        Tuple of (dataset, json_path, solver, is_bi_arm):
            - dataset: DirectHDF5Recorder instance or None if not recording
            - json_path: Path to object initial pose JSON file or None
            - solver: RobotKinematics solver instance or None
            - is_bi_arm: Boolean indicating if dual-arm configuration

    Raises:
        ValueError: If record_ee_pose is enabled but ee_urdf_path is not provided.
        FileNotFoundError: If URDF file is not found.
    """
    if not args.enable_record:
        return None, None, None, False

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
    json_path.parent.mkdir(parents=True, exist_ok=True)

    env_args: Dict[str, Any] = {"env_name": args.task or "", "type": 2}
    if getattr(args, "garment_name", None):
        env_args["garment_name"] = args.garment_name
    if getattr(args, "garment_version", None):
        env_args["garment_version"] = args.garment_version
    if getattr(args, "task_description", None):
        env_args["task_description"] = args.task_description
    env_args["joint_names"] = joint_names

    solver = None
    if getattr(args, "record_ee_pose", False):
        if not args.ee_urdf_path:
            raise ValueError("--record_ee_pose requires --ee_urdf_path")

        urdf_path = Path(args.ee_urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        from lehome.utils import RobotKinematics

        if is_bi_arm:
            solver_joint_names = [n.replace("left_", "") for n in joint_names[:5]]
        else:
            solver_joint_names = joint_names[:5]

        solver = RobotKinematics(
            str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=solver_joint_names,
        )
        arm_type = "dual-arm" if is_bi_arm else "single-arm"
        logger.info(f"End-effector pose solver loaded ({arm_type})")

    dataset = DirectHDF5Recorder(
        env=env,
        file_path=hdf5_path,
        env_args=env_args,
        fps=int(getattr(args, "step_hz", 30)),
        is_bi_arm=is_bi_arm,
    )
    logger.info(f"Recording direct HDF5 file: {hdf5_path}")

    return dataset, json_path, solver, is_bi_arm


def run_idle_phase(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    count_render: int,
    debug_pose_state: Dict[str, Any],
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
    dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)

    actions = teleop_interface.advance()
    object_initial_pose = None

    if count_render == 0:
        logger.info("[Idle Phase] Initializing observations...")
        env.initialize_obs()
        count_render += 1

        logger.info("[Idle Phase] Stabilizing garment after initialization...")
        stabilize_garment_after_reset(env, args)
        logger.info("[Idle Phase] Ready for recording")

    if actions is None:
        current_obs = env._get_observations()
        if "observation.state" in current_obs:
            current_state = current_obs["observation.state"]
            if isinstance(current_state, np.ndarray):
                maintain_action = (
                    torch.from_numpy(current_state).float().unsqueeze(0).to(env.device)
                )
            else:
                maintain_action = torch.zeros(
                    1, len(current_state), dtype=torch.float32, device=env.device
                )
        else:
            action_dim = 12 if "Bi" in args.task else 6
            maintain_action = torch.zeros(
                1, action_dim, dtype=torch.float32, device=env.device
            )
        env.step(maintain_action)
        env.render()
    else:
        env.step(actions)
        object_initial_pose = env.get_all_pose()

    if object_initial_pose is None:
        object_initial_pose = env.get_all_pose()

    _maybe_log_debug_pose_snapshot(env, args, debug_pose_state)

    return object_initial_pose, count_render


def run_recording_phase(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    flags: Dict[str, bool],
    dataset: DirectHDF5Recorder,
    json_path: Path,
    initial_object_pose: Optional[Dict[str, Any]],
    ee_solver: Optional[Any] = None,
    is_bi_arm: bool = False,
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
        json_path: Path to object initial pose JSON file.
        initial_object_pose: Initial object pose dictionary.
        ee_solver: Optional kinematic solver for end-effector pose computation.
        is_bi_arm: Whether using dual-arm configuration.

    Returns:
        Final object initial pose dictionary.
    """
    episode_index = 0
    object_initial_pose = initial_object_pose

    # Ensure we have a valid initial pose for the first episode
    if object_initial_pose is None:
        object_initial_pose = env.get_all_pose()

    while episode_index < args.num_episode:
        # Check if recording should be aborted
        if flags["abort"]:
            dataset.clear_episode_buffer()
            dataset.finalize()
            logger.warning(f"Recording aborted, completed {episode_index} episodes")
            return object_initial_pose

        flags["success"] = False
        flags["remove"] = False
        episode_step_count = 0

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
                dataset.clear_episode_buffer()
                dataset.finalize()
                logger.warning(f"Recording aborted, completed {episode_index} episodes")
                return object_initial_pose

            try:
                dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
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
            if (
                getattr(args, "disable_depth", False)
                and "observation.top_depth" in observations
            ):
                observations.pop("observation.top_depth")

            if getattr(args, "enable_pointcloud", False):
                # Converting pointcloud online is time-consuming, please convert offline
                # pointcloud = env._get_workspace_pointcloud(
                #     num_points=4096, use_fps=True
                # )
                print("Converting pointcloud online is time-consuming, please convert offline")
            _, truncated = env._get_dones()
            frame = {**observations, "task": args.task_description}

            if (
                ee_solver is not None
                and "observation.state" in observations
                and "action" in observations
            ):
                from lehome.utils import compute_ee_pose_single_arm

                obs_state = np.array(
                    observations["observation.state"], dtype=np.float32
                )
                action_state = np.array(observations["action"], dtype=np.float32)

                if is_bi_arm:
                    obs_left = compute_ee_pose_single_arm(
                        ee_solver, obs_state[:6], args.ee_state_unit
                    )
                    obs_right = compute_ee_pose_single_arm(
                        ee_solver, obs_state[6:12], args.ee_state_unit
                    )
                    frame["observation.ee_pose"] = np.concatenate(
                        [obs_left, obs_right], axis=0
                    )

                    act_left = compute_ee_pose_single_arm(
                        ee_solver, action_state[:6], args.ee_state_unit
                    )
                    act_right = compute_ee_pose_single_arm(
                        ee_solver, action_state[6:12], args.ee_state_unit
                    )
                    frame["action.ee_pose"] = np.concatenate(
                        [act_left, act_right], axis=0
                    )
                else:
                    frame["observation.ee_pose"] = compute_ee_pose_single_arm(
                        ee_solver, obs_state, args.ee_state_unit
                    )
                    frame["action.ee_pose"] = compute_ee_pose_single_arm(
                        ee_solver, action_state, args.ee_state_unit
                    )

            dataset.add_frame(frame)

            if truncated or flags["remove"]:
                dataset.clear_episode_buffer()
                logger.info(f"Re-recording episode {episode_index}")
                try:
                    env.reset()
                    stabilize_garment_after_reset(env, args)
                    object_initial_pose = env.get_all_pose()
                except Exception as e:
                    logger.error(
                        f"[Recording] Failed to reset environment during re-recording: {e}"
                    )
                    traceback.print_exc()
                    try:
                        object_initial_pose = env.get_all_pose()
                    except Exception:
                        object_initial_pose = None
                flags["remove"] = False
                continue

        _log_episode_success_snapshot(env, episode_index, episode_step_count)

        save_start_time = time.time()
        logger.info(f"[Recording] Saving episode {episode_index}...")
        try:
            garment_name = None
            if hasattr(env, "cfg") and hasattr(env.cfg, "garment_name"):
                garment_name = env.cfg.garment_name

            scale = None
            if hasattr(env, "object") and hasattr(env.object, "init_scale"):
                try:
                    scale = env.object.init_scale
                except Exception:
                    logger.warning("Failed to get scale from garment object")

            dataset.set_pending_episode_meta(
                episode_index=episode_index,
                object_initial_pose=object_initial_pose,
                garment_name=garment_name,
                scale=scale,
            )
            dataset.save_episode()
            save_duration = time.time() - save_start_time
            logger.info(
                f"[Recording] Episode {episode_index} saved (took {save_duration:.1f}s)"
            )
        except Exception as e:
            logger.error(f"[Recording] Failed to save episode {episode_index}: {e}")
            traceback.print_exc()

        try:
            append_episode_initial_pose(
                json_path,
                episode_index,
                object_initial_pose,
                garment_name=garment_name,
                scale=scale,
            )
        except Exception as e:
            logger.error(
                f"[Recording] Failed to save episode metadata for episode {episode_index}: {e}"
            )
            traceback.print_exc()

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

        try:
            object_initial_pose = env.get_all_pose()
        except Exception as e:
            logger.error(f"[Recording] Failed to get initial pose: {e}")
            traceback.print_exc()
            object_initial_pose = None
    dataset.clear_episode_buffer()
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
) -> None:
    """Run live teleoperation control without recording.

    Handles the case when S key is pressed but recording is not enabled.
    Performs simple teleoperation control without writing to dataset.

    Args:
        env: Environment instance.
        teleop_interface: Teleoperation interface.
        args: Command-line arguments.
    """
    dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
    actions = teleop_interface.advance()

    if actions is None:
        current_obs = env._get_observations()
        if "observation.state" in current_obs:
            current_state = current_obs["observation.state"]
            if isinstance(current_state, np.ndarray):
                maintain_action = (
                    torch.from_numpy(current_state).float().unsqueeze(0).to(env.device)
                )
            else:
                maintain_action = torch.zeros(
                    1, len(current_state), dtype=torch.float32, device=env.device
                )
        else:
            action_dim = 12 if "Bi" in args.task else 6
            maintain_action = torch.zeros(
                1, action_dim, dtype=torch.float32, device=env.device
            )
        env.step(maintain_action)
        env.render()
    else:
        env.step(actions)

    _maybe_log_debug_pose_snapshot(env, args, debug_pose_state)

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
    teleop_interface = create_teleop_interface(env, args)
    flags = register_teleop_callbacks(
        teleop_interface, recording_enabled=args.enable_record
    )
    teleop_interface.reset()
    dataset, json_path, ee_solver, is_bi_arm = create_dataset_if_needed(env, args)
    count_render = 0
    printed_instructions = False
    idle_frame_counter = 0
    object_initial_pose: Optional[Dict[str, Any]] = None
    debug_pose_state: Dict[str, Any] = {"step_count": 0, "eef_body_idx_cache": {}}

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
                        json_path,
                        object_initial_pose,
                        ee_solver,
                        is_bi_arm,
                    )
                    break
                else:
                    run_live_control_without_record(
                        env,
                        teleop_interface,
                        args,
                        debug_pose_state,
                    )
    except KeyboardInterrupt:
        logger.warning("\n[Ctrl+C] Interrupt signal detected")
        # If Ctrl+C is pressed during recording, clear the current buffer
        if args.enable_record and dataset is not None and flags["start"]:
            logger.info("Clearing current episode buffer...")
            dataset.clear_episode_buffer()
            logger.info("Buffer cleared, dataset remains intact")
            dataset.finalize()
            logger.info("Dataset saved")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

    finally:
        env.close()
