"""HDF5 replay dataset readers for MimicGen workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset_io import (
    decode_attr as _decode_attr,
    demo_sort_key as _demo_sort_key,
    read_hdf5_node as _read_hdf5_node,
    require_h5py as _require_h5py,
    to_python_scalar_or_list as _to_python_scalar_or_list,
)

try:
    import h5py
except ImportError:
    h5py = None


class HDF5ReplaySource:
    """Reader for IsaacLab-style HDF5 replay datasets."""

    def __init__(self, hdf5_path: str):
        _require_h5py("HDF5 replay")
        self.path = Path(hdf5_path)
        self.file = h5py.File(self.path, "r")

        if "data" not in self.file:
            self.file.close()
            raise ValueError(f"HDF5 missing required '/data' group: {self.path}")

        self.data_group = self.file["data"]
        self.demo_names = sorted(
            [name for name in self.data_group.keys() if name.startswith("demo_")],
            key=_demo_sort_key,
        )

        if len(self.demo_names) == 0:
            self.file.close()
            raise ValueError(f"No demo groups found under '/data' in {self.path}")

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "HDF5ReplaySource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def num_episodes(self) -> int:
        return len(self.demo_names)

    @property
    def fps(self) -> int:
        raw = self.data_group.attrs.get("fps", 30)
        try:
            return int(_decode_attr(raw))
        except (TypeError, ValueError):
            return 30

    def get_demo_group(self, episode_index: int) -> Any:
        return self.data_group[self.demo_names[episode_index]]

    def get_source_episode_index(self, episode_index: int) -> int:
        demo = self.get_demo_group(episode_index)
        if "source_episode_index" in demo.attrs:
            try:
                return int(_decode_attr(demo.attrs["source_episode_index"]))
            except (TypeError, ValueError):
                return episode_index
        return episode_index

    def get_env_args(self) -> dict[str, Any]:
        raw_env_args = self.data_group.attrs.get("env_args")
        if raw_env_args is None:
            return {}
        try:
            value = _decode_attr(raw_env_args)
            if isinstance(value, str):
                return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return {}

    def get_episode_meta(self, episode_index: int) -> dict[str, Any]:
        """Read demo-level metadata from /data/demo_*/meta or synthesize it from initial_state."""
        demo = self.get_demo_group(episode_index)
        parsed: dict[str, Any] = {}
        if "meta" in demo:
            meta_value = _read_hdf5_node(demo["meta"])
            if isinstance(meta_value, dict):
                parsed.update(meta_value)

        initial_state = demo.get("initial_state")
        garment_group = None if initial_state is None else initial_state.get("garment")
        if garment_group is None:
            return parsed

        for garment_name in garment_group.keys():
            garment_entry = garment_group[garment_name]
            garment_meta: dict[str, Any] = {}

            if "initial_pose" in garment_entry:
                pose_value = _to_python_scalar_or_list(garment_entry["initial_pose"][()])
                if isinstance(pose_value, list) and len(pose_value) == 1 and isinstance(pose_value[0], list):
                    pose_value = pose_value[0]
                garment_meta["initial_pose"] = pose_value

            if "scale" in garment_entry:
                scale_value = _to_python_scalar_or_list(garment_entry["scale"][()])
                if isinstance(scale_value, list) and len(scale_value) == 1 and isinstance(scale_value[0], list):
                    scale_value = scale_value[0]
                garment_meta["scale"] = scale_value

            if garment_meta:
                parsed.setdefault(str(garment_name), {}).update(garment_meta)

        return parsed

    def get_garment_name_from_env_args(self) -> str | None:
        env_args = self.get_env_args()
        garment_name = env_args.get("garment_name")
        if isinstance(garment_name, str) and garment_name:
            return garment_name
        return None

    def has_ee_pose(self) -> bool:
        demo = self.get_demo_group(0)
        if "obs" in demo and "ee_pose" in demo["obs"]:
            return True

        if "actions" in demo:
            actions = np.asarray(demo["actions"][:])
            if actions.ndim == 2 and actions.shape[1] in (8, 16):
                has_joint_actions = "obs" in demo and "actions" in demo["obs"]
                actions_mode = _decode_attr(self.data_group.attrs.get("actions_mode", ""))
                actions_mode = str(actions_mode).lower()
                return has_joint_actions or actions_mode == "ee_pose"

        return False

    def get_state_dim(self) -> int:
        demo = self.get_demo_group(0)
        joint_actions = self._extract_joint_actions(demo)
        obs_state = self._extract_observation_state(demo, joint_actions)
        return int(obs_state.shape[1])

    def get_episode_frames(
        self, episode_index: int, require_ee_pose: bool = False
    ) -> list[dict[str, torch.Tensor]]:
        """Load one episode into replay frame format used by replay runtime."""
        demo = self.get_demo_group(episode_index)

        joint_actions = self._extract_joint_actions(demo)
        obs_state = self._extract_observation_state(demo, joint_actions)
        ee_pose = self._extract_action_ee_pose(demo)

        if require_ee_pose and ee_pose is None:
            raise ValueError(
                f"{self.demo_names[episode_index]} does not contain action ee_pose data."
            )

        total_frames = min(joint_actions.shape[0], obs_state.shape[0])
        if ee_pose is not None:
            total_frames = min(total_frames, ee_pose.shape[0])

        if total_frames <= 0:
            return []

        frames: list[dict[str, torch.Tensor]] = []
        for idx in range(total_frames):
            frame = {
                "action": torch.from_numpy(joint_actions[idx]).float(),
                "observation.state": torch.from_numpy(obs_state[idx]).float(),
            }
            if ee_pose is not None:
                frame["action.ee_pose"] = torch.from_numpy(ee_pose[idx]).float()
            frames.append(frame)

        return frames

    def _extract_joint_actions(self, demo: Any) -> np.ndarray:
        """Prefer joint-space actions from obs/actions, fallback to top-level actions."""
        if "obs" in demo and "actions" in demo["obs"]:
            actions = np.asarray(demo["obs"]["actions"][:], dtype=np.float32)
        elif "actions" in demo:
            actions = np.asarray(demo["actions"][:], dtype=np.float32)
        else:
            raise ValueError("Demo missing both 'obs/actions' and top-level 'actions'.")

        if actions.ndim != 2:
            raise ValueError(f"Expected 2D actions array, got shape={actions.shape}")

        return actions

    def _extract_action_ee_pose(self, demo: Any) -> np.ndarray | None:
        """Load ee_pose actions when available."""
        if "obs" in demo and "ee_pose" in demo["obs"]:
            ee_pose = np.asarray(demo["obs"]["ee_pose"][:], dtype=np.float32)
            if ee_pose.ndim == 2 and ee_pose.shape[1] in (8, 16):
                return ee_pose
            return None

        if "actions" not in demo:
            return None

        top_level_actions = np.asarray(demo["actions"][:], dtype=np.float32)
        if top_level_actions.ndim != 2 or top_level_actions.shape[1] not in (8, 16):
            return None

        has_joint_actions = "obs" in demo and "actions" in demo["obs"]
        actions_mode = _decode_attr(self.data_group.attrs.get("actions_mode", ""))
        actions_mode = str(actions_mode).lower()

        if has_joint_actions or actions_mode == "ee_pose":
            return top_level_actions

        return None

    def _extract_observation_state(
        self,
        demo: Any,
        joint_actions: np.ndarray,
    ) -> np.ndarray:
        """Build observation.state from available HDF5 observation/state fields."""
        if "obs" in demo:
            obs = demo["obs"]

            if "left_joint_pos" in obs and "right_joint_pos" in obs:
                left = np.asarray(obs["left_joint_pos"][:], dtype=np.float32)
                right = np.asarray(obs["right_joint_pos"][:], dtype=np.float32)
                num_frames = min(left.shape[0], right.shape[0])
                if num_frames > 0:
                    return np.concatenate([left[:num_frames], right[:num_frames]], axis=1)

            if "left_joint_pos" in obs:
                left = np.asarray(obs["left_joint_pos"][:], dtype=np.float32)
                if left.ndim == 2 and left.shape[0] > 0:
                    return left

        initial_state = self._collect_articulation_joint_positions(demo.get("initial_state"))
        states = self._collect_articulation_joint_positions(demo.get("states"))
        if initial_state is not None and states is not None:
            return np.concatenate([initial_state[:1], states], axis=0)

        return joint_actions.copy()

    def _collect_articulation_joint_positions(self, group: Any) -> np.ndarray | None:
        """Collect and concatenate articulation joint positions from a group."""
        if group is None or "articulation" not in group:
            return None

        articulation_group = group["articulation"]
        part_arrays: list[np.ndarray] = []
        num_frames: int | None = None

        for name in sorted(articulation_group.keys()):
            entity_group = articulation_group[name]
            if "joint_position" not in entity_group:
                continue

            arr = np.asarray(entity_group["joint_position"][:], dtype=np.float32)
            if arr.ndim != 2:
                continue

            part_arrays.append(arr)
            num_frames = arr.shape[0] if num_frames is None else min(num_frames, arr.shape[0])

        if len(part_arrays) == 0 or num_frames is None or num_frames <= 0:
            return None

        trimmed = [arr[:num_frames] for arr in part_arrays]
        return np.concatenate(trimmed, axis=1)


__all__ = ["HDF5ReplaySource"]
