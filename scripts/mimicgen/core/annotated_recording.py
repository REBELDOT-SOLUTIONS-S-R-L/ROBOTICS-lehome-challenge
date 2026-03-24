"""Lean HDF5 recording primitives for annotated Mimic teleoperation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import isaaclab.utils.math as PoseUtils
import torch

from lehome.tasks.fold_cloth.checkpoint_mappings import validate_semantic_object_pose_dict

from .data_utils import as_numpy, to_json_compatible
from .dataset_io import require_h5py

try:
    import h5py
except ImportError:
    h5py = None


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


def _normalize_pose_matrix(value: Any) -> np.ndarray:
    arr = as_numpy(value, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (4, 4):
        raise ValueError(f"Expected pose matrix shape (4, 4), got {tuple(arr.shape)}.")
    return arr


def _normalize_signal_column(value: Any) -> np.ndarray:
    arr = as_numpy(value).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected scalar/column signal value, got shape {tuple(arr.shape)}.")
    return arr.astype(np.bool_, copy=False).reshape(1)


def _normalize_joint_row(value: Any, *, expected_dim: int | None = None) -> np.ndarray:
    arr = as_numpy(value, dtype=np.float32).reshape(-1)
    if expected_dim is not None and arr.size != expected_dim:
        raise ValueError(
            f"Expected joint row with {expected_dim} values, got shape {tuple(arr.shape)}."
        )
    return arr.astype(np.float32, copy=False)


class AnnotatedMimicHDF5Recorder:
    """Buffered writer for generation-ready Mimic teleoperation episodes."""

    def __init__(self, file_path: Path, env_args: dict[str, Any], fps: int) -> None:
        require_h5py("annotated Mimic HDF5 recording")
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._file_path, "w")
        self._data_group = self._file.create_group("data")
        self._data_group.attrs["env_args"] = json.dumps(to_json_compatible(env_args))
        self._data_group.attrs["total"] = np.int64(0)
        self._data_group.attrs["fps"] = np.int64(int(fps))
        self._data_group.attrs["num_episodes"] = np.int64(0)
        self._data_group.attrs["actions_mode"] = "ee_pose"
        self._data_group.attrs["actions_frame"] = "base"
        self._data_group.attrs["ik_quat_order"] = "wxyz"

        self._episode_index: int | None = None
        self._episode_meta: dict[str, Any] | None = None
        self._buffers: dict[str, Any] = {}
        self._num_episodes = 0
        self._total_samples = 0
        self._validated_object_pose_once = False

    @property
    def file_path(self) -> Path:
        return self._file_path

    def begin_episode(
        self,
        episode_index: int,
        object_initial_pose: dict[str, Any] | None,
        garment_name: str | None,
        scale: Any | None,
    ) -> None:
        if self._episode_index is not None:
            raise RuntimeError("Cannot begin a new episode while another episode is active.")

        self._episode_index = int(episode_index)
        self._episode_meta = {
            "object_initial_pose": to_json_compatible(object_initial_pose),
            "garment_name": str(garment_name) if garment_name else "unknown",
            "scale": to_json_compatible(scale),
        }
        self._buffers = {
            "object_pose": {},
            "eef_pose": {},
            "target_eef_pose": {},
            "subtask_term_signals": {},
            "gripper_actions": {},
            "joint_actions": [],
            "joint_pos": {},
            "joint_vel": {},
        }
        self._validated_object_pose_once = False

    def discard_episode(self) -> None:
        self._episode_index = None
        self._episode_meta = None
        self._buffers = {}
        self._validated_object_pose_once = False

    def append_step(
        self,
        *,
        object_pose: dict[str, Any],
        eef_pose: dict[str, Any],
        target_eef_pose: dict[str, Any],
        subtask_term_signals: dict[str, Any],
        gripper_actions: dict[str, Any],
        joint_actions: Any | None,
        joint_pos: dict[str, Any],
        joint_vel: dict[str, Any],
    ) -> None:
        if self._episode_index is None or self._episode_meta is None:
            raise RuntimeError("No active episode. Call begin_episode() first.")

        if not self._validated_object_pose_once:
            validate_semantic_object_pose_dict(
                object_pose,
                context="AnnotatedMimicHDF5Recorder.append_step",
            )
            self._validated_object_pose_once = True

        self._append_pose_section("object_pose", object_pose)
        self._append_pose_section("eef_pose", eef_pose)
        self._append_pose_section("target_eef_pose", target_eef_pose)
        self._append_signal_section(subtask_term_signals)
        self._append_gripper_action_section(gripper_actions)
        self._append_joint_observation_step(
            joint_actions=joint_actions,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

    def _append_pose_section(self, section_name: str, values: dict[str, Any]) -> None:
        section = self._buffers[section_name]
        for key, value in values.items():
            section.setdefault(str(key), []).append(_normalize_pose_matrix(value))

    def _append_signal_section(self, values: dict[str, Any]) -> None:
        section = self._buffers["subtask_term_signals"]
        for key, value in values.items():
            section.setdefault(str(key), []).append(_normalize_signal_column(value))

    def _append_gripper_action_section(self, values: dict[str, Any]) -> None:
        section = self._buffers["gripper_actions"]
        for key, value in values.items():
            section.setdefault(str(key), []).append(_normalize_signal_column(value))

    def _append_joint_observation_step(
        self,
        *,
        joint_actions: Any | None,
        joint_pos: dict[str, Any],
        joint_vel: dict[str, Any],
    ) -> None:
        if joint_actions is None:
            raise RuntimeError("Annotated recorder requires joint_actions for IK->joint conversion support.")
        self._buffers["joint_actions"].append(_normalize_joint_row(joint_actions, expected_dim=12))

        pos_section = self._buffers["joint_pos"]
        vel_section = self._buffers["joint_vel"]
        for arm_name in ("left_arm", "right_arm"):
            if arm_name not in joint_pos:
                raise RuntimeError(f"Missing joint_pos for {arm_name}.")
            if arm_name not in joint_vel:
                raise RuntimeError(f"Missing joint_vel for {arm_name}.")
            pos_section.setdefault(arm_name, []).append(
                _normalize_joint_row(joint_pos[arm_name], expected_dim=6)
            )
            vel_section.setdefault(arm_name, []).append(
                _normalize_joint_row(joint_vel[arm_name], expected_dim=6)
            )

    def _stack_pose_section(self, section_name: str) -> dict[str, np.ndarray]:
        return {
            key: np.stack(frames, axis=0).astype(np.float32, copy=False)
            for key, frames in self._buffers[section_name].items()
        }

    def _stack_signal_section(self) -> dict[str, np.ndarray]:
        return {
            key: np.stack(frames, axis=0).astype(np.bool_, copy=False)
            for key, frames in self._buffers["subtask_term_signals"].items()
        }

    def _stack_joint_actions(self) -> np.ndarray:
        return np.stack(self._buffers["joint_actions"], axis=0).astype(np.float32, copy=False)

    def _stack_joint_section(self, section_name: str) -> dict[str, np.ndarray]:
        return {
            key: np.stack(frames, axis=0).astype(np.float32, copy=False)
            for key, frames in self._buffers[section_name].items()
        }

    def _synthesize_native_actions(self, env: Any) -> np.ndarray:
        target_eef_pose = self._buffers["target_eef_pose"]
        gripper_actions = self._buffers["gripper_actions"]
        if "left_arm" not in target_eef_pose or "right_arm" not in target_eef_pose:
            raise RuntimeError("Missing target_eef_pose buffers for one or both arms.")
        if "left_arm" not in gripper_actions or "right_arm" not in gripper_actions:
            raise RuntimeError("Missing gripper action buffers for one or both arms.")

        num_samples = len(target_eef_pose["left_arm"])
        actions = np.zeros((num_samples, 16), dtype=np.float32)
        arm_offsets = {"left_arm": 0, "right_arm": 8}
        for step_idx in range(num_samples):
            for arm_name, offset in arm_offsets.items():
                world_pose = np.asarray(target_eef_pose[arm_name][step_idx], dtype=np.float32)
                base_pose = env._world_pose_to_base_pose_np(arm_name, 0, world_pose)
                base_pose = np.asarray(base_pose, dtype=np.float32)
                actions[step_idx, offset : offset + 3] = base_pose[:3, 3]
                quat_wxyz = (
                    PoseUtils.quat_from_matrix(
                        torch.as_tensor(base_pose[:3, :3], dtype=torch.float32).unsqueeze(0)
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                actions[step_idx, offset + 3 : offset + 7] = quat_wxyz
                grip = np.asarray(gripper_actions[arm_name][step_idx], dtype=np.float32).reshape(-1)
                if grip.size == 0:
                    raise RuntimeError(f"Missing gripper action for {arm_name} at step {step_idx}.")
                actions[step_idx, offset + 7] = grip[0]
        return actions

    def finalize_episode(self, env: Any) -> None:
        if self._episode_index is None or self._episode_meta is None:
            raise RuntimeError("No active episode to finalize.")
        if not self._buffers.get("eef_pose"):
            raise RuntimeError("Cannot finalize an empty episode.")

        episode_name = f"demo_{self._episode_index}"
        if episode_name in self._data_group:
            del self._data_group[episode_name]

        object_pose_section = self._stack_pose_section("object_pose")
        validate_semantic_object_pose_dict(
            object_pose_section,
            context="AnnotatedMimicHDF5Recorder.finalize_episode",
        )
        eef_pose_section = self._stack_pose_section("eef_pose")
        target_eef_pose_section = self._stack_pose_section("target_eef_pose")
        signal_section = self._stack_signal_section()
        joint_actions = self._stack_joint_actions()
        joint_pos_section = self._stack_joint_section("joint_pos")
        joint_vel_section = self._stack_joint_section("joint_vel")

        demo_group = self._data_group.create_group(episode_name)
        actions = self._synthesize_native_actions(env)
        demo_group.create_dataset("actions", data=actions, compression="lzf")
        demo_group.attrs["num_samples"] = np.int64(actions.shape[0])
        demo_group.attrs["success"] = np.bool_(True)

        obs_group = demo_group.create_group("obs")
        self._write_joint_obs_section(
            obs_group,
            joint_actions=joint_actions,
            joint_pos=joint_pos_section,
            joint_vel=joint_vel_section,
        )
        datagen_group = obs_group.create_group("datagen_info")
        self._write_pose_section(datagen_group, "object_pose", object_pose_section)
        self._write_pose_section(datagen_group, "eef_pose", eef_pose_section)
        self._write_pose_section(datagen_group, "target_eef_pose", target_eef_pose_section)
        self._write_signal_section(datagen_group, signal_section)
        self._write_initial_garment_state(demo_group)

        self._num_episodes += 1
        self._total_samples += int(actions.shape[0])
        self._data_group.attrs["num_episodes"] = np.int64(self._num_episodes)
        self._data_group.attrs["total"] = np.int64(self._total_samples)
        self._file.flush()
        self.discard_episode()

    def _write_pose_section(
        self,
        datagen_group: Any,
        section_name: str,
        section_values: dict[str, np.ndarray],
    ) -> None:
        section_group = datagen_group.create_group(section_name)
        for key, stacked in section_values.items():
            section_group.create_dataset(key, data=stacked, compression="lzf")

    def _write_signal_section(
        self,
        datagen_group: Any,
        signal_values: dict[str, np.ndarray],
    ) -> None:
        signal_group = datagen_group.create_group("subtask_term_signals")
        for key, stacked in signal_values.items():
            signal_group.create_dataset(key, data=stacked, compression="lzf")

    def _write_joint_obs_section(
        self,
        obs_group: Any,
        *,
        joint_actions: np.ndarray,
        joint_pos: dict[str, np.ndarray],
        joint_vel: dict[str, np.ndarray],
    ) -> None:
        obs_group.create_dataset("actions", data=joint_actions, compression="lzf")

        left_joint_pos = joint_pos["left_arm"]
        right_joint_pos = joint_pos["right_arm"]
        left_joint_vel = joint_vel["left_arm"]
        right_joint_vel = joint_vel["right_arm"]
        left_target = joint_actions[:, :6]
        right_target = joint_actions[:, 6:12]

        obs_group.create_dataset("left_joint_pos", data=left_joint_pos, compression="lzf")
        obs_group.create_dataset("right_joint_pos", data=right_joint_pos, compression="lzf")
        obs_group.create_dataset("left_joint_pos_target", data=left_target, compression="lzf")
        obs_group.create_dataset("right_joint_pos_target", data=right_target, compression="lzf")
        obs_group.create_dataset("left_joint_pos_rel", data=(left_target - left_joint_pos), compression="lzf")
        obs_group.create_dataset("right_joint_pos_rel", data=(right_target - right_joint_pos), compression="lzf")
        obs_group.create_dataset("left_joint_vel", data=left_joint_vel, compression="lzf")
        obs_group.create_dataset("right_joint_vel", data=right_joint_vel, compression="lzf")
        obs_group.create_dataset("left_joint_vel_rel", data=left_joint_vel, compression="lzf")
        obs_group.create_dataset("right_joint_vel_rel", data=right_joint_vel, compression="lzf")

    def _write_initial_garment_state(self, demo_group: Any) -> None:
        if self._episode_meta is None:
            return

        garment_name = str(self._episode_meta.get("garment_name") or "Garment")
        pose_value = _resolve_garment_pose_value(
            self._episode_meta.get("object_initial_pose"),
            garment_name,
        )
        if pose_value is None:
            raise RuntimeError(
                f"Missing initial garment pose for recorded episode ({garment_name})."
            )

        initial_state_group = demo_group.create_group("initial_state")
        garment_group = initial_state_group.create_group("garment")
        garment_entry = garment_group.create_group(garment_name)
        garment_entry.create_dataset(
            "initial_pose",
            data=as_numpy(pose_value, dtype=np.float32),
        )
        scale_value = self._episode_meta.get("scale")
        if scale_value is not None:
            garment_entry.create_dataset(
                "scale",
                data=as_numpy(scale_value, dtype=np.float32),
            )

    def finalize(self) -> None:
        if getattr(self, "_file", None) is not None:
            self._data_group.attrs["num_episodes"] = np.int64(self._num_episodes)
            self._data_group.attrs["total"] = np.int64(self._total_samples)
            self._file.flush()
            self._file.close()
            self._file = None


__all__ = ["AnnotatedMimicHDF5Recorder"]
