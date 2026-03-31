"""Source-dataset adaptation helpers for MimicGen generation."""

from __future__ import annotations

from typing import Any

import torch

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool

from lehome.tasks.fold_cloth.checkpoint_mappings import (
    ClothObjectPoseValidationError,
    validate_semantic_object_pose_dict,
)

from .dataset_io import demo_sort_key as _demo_sort_key
from .dataset_io import load_episode_compat as _load_episode_compat

# Camera observation keys to skip when loading source episodes for generation.
# These are large image tensors that are not needed by MimicGen.
_SOURCE_OBS_SKIP_KEYS = frozenset({"top", "left_wrist", "right_wrist"})


def _pose_dict_first_center_cpu(pose_dict: Any) -> torch.Tensor | None:
    """Compute mean xyz over first timestep for a pose dict on CPU."""
    if not isinstance(pose_dict, dict) or len(pose_dict) == 0:
        return None
    points = []
    for pose_value in pose_dict.values():
        try:
            pose = torch.as_tensor(pose_value, dtype=torch.float32)
        except Exception:
            continue
        if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
            points.append(pose[0, :3, 3])
        elif pose.ndim == 2 and pose.shape == (4, 4):
            points.append(pose[:3, 3])
    if not points:
        return None
    return torch.stack(points, dim=0).mean(dim=0)


def get_runtime_object_center(env: Any) -> torch.Tensor | None:
    """Get runtime object center from env.get_object_poses()."""
    try:
        object_poses = env.get_object_poses(env_ids=[0])
    except Exception:
        return None
    try:
        validate_semantic_object_pose_dict(
            object_poses,
            context="generation runtime object poses",
        )
    except ClothObjectPoseValidationError:
        return None
    return _pose_dict_first_center_cpu(object_poses)


class RobustDataGenInfoPool(DataGenInfoPool):
    """Datagen info pool with fallback episode loading for mixed HDF5 schemas."""

    def __init__(
        self,
        *args,
        prefer_eef_pose_as_target: bool = False,
        source_target_z_offset: float = 0.0,
        align_object_pose_to_runtime: bool = False,
        align_object_pose_mode: str = "object_only",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._prefer_eef_pose_as_target = bool(prefer_eef_pose_as_target)
        self._source_target_z_offset = float(source_target_z_offset)
        self._align_object_pose_to_runtime = bool(align_object_pose_to_runtime)
        self._align_object_pose_mode = str(align_object_pose_mode).strip().lower()
        if self._align_object_pose_mode not in {"object_only", "all_poses"}:
            raise ValueError(
                f"Invalid align_object_pose_mode '{align_object_pose_mode}'. "
                "Expected one of: object_only, all_poses."
            )
        self._runtime_object_center = self._get_runtime_object_center()
        self.invalid_episode_names: list[str] = []
        self.episode_names: list[str] = []

    def _get_runtime_object_center(self) -> torch.Tensor | None:
        """Get current env object center from runtime world-frame object poses."""
        try:
            object_poses = self.env.get_object_poses(env_ids=[0])
        except Exception:
            return None
        try:
            validate_semantic_object_pose_dict(
                object_poses,
                context="generation runtime object poses",
            )
        except ClothObjectPoseValidationError:
            return None
        return self._pose_dict_first_center(object_poses)

    def _pose_dict_first_center(self, pose_dict: Any) -> torch.Tensor | None:
        """Compute mean xyz over first timestep for a pose dict."""
        if not isinstance(pose_dict, dict) or len(pose_dict) == 0:
            return None
        points = []
        for pose_value in pose_dict.values():
            try:
                pose = torch.as_tensor(pose_value, device=self.device, dtype=torch.float32)
            except Exception:
                continue
            if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
                points.append(pose[0, :3, 3])
            elif pose.ndim == 2 and pose.shape == (4, 4):
                points.append(pose[:3, 3])
        if not points:
            return None
        return torch.stack(points, dim=0).mean(dim=0)

    def _apply_translation_to_pose_dict(self, pose_dict: Any, delta_xyz: torch.Tensor) -> Any:
        """Shift all pose translations by delta_xyz."""
        if not isinstance(pose_dict, dict):
            return pose_dict
        out = {}
        for key, value in pose_dict.items():
            try:
                pose = torch.as_tensor(value, device=self.device, dtype=torch.float32).clone()
            except Exception:
                out[key] = value
                continue
            if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
                pose[:, :3, 3] += delta_xyz.view(1, 3)
            elif pose.ndim == 2 and pose.shape == (4, 4):
                pose[:3, 3] += delta_xyz
            out[key] = pose
        return out

    def _apply_z_offset_to_pose_dict(self, pose_dict: Any) -> Any:
        """Apply z offset to a dict of 4x4 pose tensors."""
        if abs(self._source_target_z_offset) < 1e-9:
            return pose_dict
        if not isinstance(pose_dict, dict):
            return pose_dict

        out = {}
        for key, value in pose_dict.items():
            try:
                pose = torch.as_tensor(value, device=self.device, dtype=torch.float32).clone()
            except Exception:
                out[key] = value
                continue

            if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
                pose[:, 2, 3] += self._source_target_z_offset
            elif pose.ndim == 2 and pose.shape == (4, 4):
                pose[2, 3] += self._source_target_z_offset
            out[key] = pose
        return out

    def _prepare_source_target_eef_pose(self, episode: EpisodeData) -> None:
        """Prepare the source target_eef_pose stream before loading an episode into Mimic."""
        try:
            obs = episode.data.get("obs", {})
            datagen = obs.get("datagen_info", {})
            eef_pose = datagen.get("eef_pose")
            target_eef_pose = datagen.get("target_eef_pose")

            if self._prefer_eef_pose_as_target and eef_pose is not None:
                target_eef_pose = eef_pose
            if target_eef_pose is None:
                return
            datagen["target_eef_pose"] = self._apply_z_offset_to_pose_dict(target_eef_pose)
        except Exception:
            return

    def _align_legacy_source_datagen_poses_to_runtime_if_needed(
        self,
        episode: EpisodeData,
        episode_name: str,
    ) -> None:
        """Align source datagen pose frames to the runtime world frame when legacy offsets are detected."""
        if not self._align_object_pose_to_runtime:
            return
        if self._runtime_object_center is None:
            return
        try:
            obs = episode.data.get("obs", {})
            datagen = obs.get("datagen_info", {})
            object_pose = datagen.get("object_pose")
            if object_pose is None:
                return

            src_center = self._pose_dict_first_center(object_pose)
            if src_center is None:
                return
            delta = self._runtime_object_center - src_center
            if float(torch.linalg.norm(delta).item()) < 0.15:
                return
            datagen["object_pose"] = self._apply_translation_to_pose_dict(object_pose, delta)
            if self._align_object_pose_mode == "all_poses":
                if "eef_pose" in datagen:
                    datagen["eef_pose"] = self._apply_translation_to_pose_dict(datagen.get("eef_pose"), delta)
                if "target_eef_pose" in datagen:
                    datagen["target_eef_pose"] = self._apply_translation_to_pose_dict(
                        datagen.get("target_eef_pose"),
                        delta,
                    )
            print(
                "Info: aligned source datagen poses to runtime frame for "
                f"{episode_name} with delta xyz=({float(delta[0]):+.4f}, "
                f"{float(delta[1]):+.4f}, {float(delta[2]):+.4f}), "
                f"mode={self._align_object_pose_mode}"
            )
        except Exception:
            return

    def _validate_episode_object_pose(self, episode: EpisodeData, episode_name: str) -> None:
        """Validate recorded semantic garment object poses before loading an episode into Mimic."""
        obs = episode.data.get("obs", {})
        datagen = obs.get("datagen_info", {})
        object_pose = datagen.get("object_pose")
        validate_semantic_object_pose_dict(
            object_pose,
            context=f"source episode {episode_name} datagen_info.object_pose",
        )

    def load_from_dataset_file(self, file_path: str, select_demo_keys: str | None = None):
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(file_path)
        try:
            episode_names = sorted(list(dataset_file_handler.get_episode_names()), key=_demo_sort_key)
            for episode_name in episode_names:
                if select_demo_keys is not None and episode_name not in select_demo_keys:
                    continue
                try:
                    episode = _load_episode_compat(
                        dataset_file_handler,
                        episode_name,
                        self.device,
                        input_file=file_path,
                        info_prefix="Info",
                        obs_skip_keys=_SOURCE_OBS_SKIP_KEYS,
                    )
                    self._prepare_source_target_eef_pose(episode)
                    self._align_legacy_source_datagen_poses_to_runtime_if_needed(episode, episode_name)
                    self._validate_episode_object_pose(episode, episode_name)
                    self._add_episode(episode)
                    self.episode_names.append(episode_name)
                except ClothObjectPoseValidationError as exc:
                    self.invalid_episode_names.append(episode_name)
                    print(
                        "Warning: skipping source episode "
                        f"{episode_name} due to invalid object_pose: {exc}"
                    )
        finally:
            dataset_file_handler.close()
        if self.num_datagen_infos == 0:
            raise ValueError(
                "No valid source episodes remain after cloth object-pose validation."
            )


__all__ = ["RobustDataGenInfoPool", "get_runtime_object_center"]
