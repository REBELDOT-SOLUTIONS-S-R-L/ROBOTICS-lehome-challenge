from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv

from lehome.tasks.fold_cloth.checkpoint_mappings import (
    ClothObjectPoseUnavailableError,
    ClothObjectPoseValidationError,
    semantic_keypoints_from_positions as map_semantic_keypoints_from_positions,
    validate_semantic_object_pose_dict,
)


class DatagenObjectPoseCaptureError(RuntimeError):
    """Raised when annotation cannot capture valid garment object poses for a replay attempt."""


@dataclass
class DatasetMetadataIndex:
    """Cached dataset-level metadata and open demo groups from the input HDF5 file."""

    garment_info: dict | None
    actions_frame: str | None
    ik_quat_order: str | None
    source_episode_indices: dict[str, int]
    episode_groups: dict[str, Any]


@dataclass
class ReplayRuntimeContext:
    """Replay configuration derived from the live environment once per process."""

    expected_action_dim: int | None
    eef_names: list[str]
    native_ik_action_contract: bool
    ik_solver_checked: bool = False
    ik_solver_ready: bool = False


@dataclass
class ReplayPlan:
    """Per-episode replay data reused across auto/manual replay attempts."""

    initial_state: dict
    replay_actions: torch.Tensor
    replay_mode: str
    ik_frame: str | None
    seed: int | None
    episode_index: int | None
    garment_initial_pose: dict | None
    ik_pose_by_eef: dict[str, torch.Tensor] | None = None
    ik_gripper_by_eef: dict[str, torch.Tensor] | None = None


def normalize_hdf5_scalar(value):
    """Normalize HDF5 scalar payloads into plain Python values."""
    if hasattr(value, "shape") and getattr(value, "shape", ()) == ():
        with contextlib.suppress(Exception):
            value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def pos_to_4x4(pos: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3) position tensors into (..., 4, 4) transforms."""
    batch_shape = pos.shape[:-1]
    pose = torch.eye(4, device=pos.device, dtype=pos.dtype).expand(*batch_shape, 4, 4).clone()
    pose[..., :3, 3] = pos
    return pose


def orthonormalize_rotations(pose: torch.Tensor) -> torch.Tensor:
    """Project rotation blocks of 4x4 poses to SO(3) for robustness."""
    if pose.ndim < 2 or pose.shape[-2:] != (4, 4):
        return pose
    squeeze_batch = False
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
        squeeze_batch = True
    rot = pose[..., :3, :3]
    try:
        rot_flat = rot.reshape(-1, 3, 3)
        u, _, vh = torch.linalg.svd(rot_flat)
        rot_ortho_flat = u @ vh
        det = torch.det(rot_ortho_flat)
        neg = det < 0
        if torch.any(neg):
            u = u.clone()
            u[neg, :, -1] *= -1.0
            rot_ortho_flat = u @ vh
        rot_ortho = rot_ortho_flat.reshape_as(rot)
        pose = pose.clone()
        pose[..., :3, :3] = rot_ortho
        pose[..., 3, :3] = 0.0
        pose[..., 3, 3] = 1.0
    except Exception:
        return pose[0] if squeeze_batch else pose
    return pose[0] if squeeze_batch else pose


def sanitize_pose_dict(pose_dict: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
    """Ensure pose dict entries are proper homogeneous transforms."""
    if not isinstance(pose_dict, dict):
        return pose_dict
    out = {}
    for key, value in pose_dict.items():
        try:
            pose = torch.as_tensor(value)
        except Exception:
            out[key] = value
            continue
        if pose.ndim >= 2 and pose.shape[-2:] == (4, 4):
            out[key] = orthonormalize_rotations(pose)
        else:
            out[key] = value
    return out


def get_cloth_keypoint_object_poses_world(
    env: ManagerBasedRLMimicEnv,
) -> dict[str, torch.Tensor] | None:
    """Build cloth virtual object poses directly from world keypoints."""
    garment_obj = getattr(env, "object", None)
    if garment_obj is None or not hasattr(garment_obj, "check_points"):
        return None

    check_points = garment_obj.check_points
    if not check_points or len(check_points) < 6:
        return None

    try:
        mesh_points_world, _, _, _ = garment_obj.get_current_mesh_points()
        mesh_points = np.asarray(mesh_points_world)
    except Exception:
        try:
            mesh_points = (
                garment_obj._cloth_prim_view.get_world_positions().squeeze(0).detach().cpu().numpy()
            )
        except Exception:
            return None

    kp_positions = mesh_points[check_points]
    semantic_points = map_semantic_keypoints_from_positions(kp_positions)
    object_poses = {}
    num_envs = int(getattr(env, "num_envs", 1))
    for name, point in semantic_points.items():
        pos = torch.tensor(point, dtype=torch.float32, device=env.device).unsqueeze(0).expand(num_envs, -1)
        object_poses[name] = pos_to_4x4(pos)
    return object_poses


def resolve_valid_annotation_object_pose(env: ManagerBasedRLMimicEnv) -> dict[str, torch.Tensor]:
    """Resolve garment object poses for annotation, with mesh fallback and strict validation."""
    errors: list[str] = []

    try:
        object_pose = env.get_object_poses()
        validate_semantic_object_pose_dict(
            object_pose,
            context="annotation env.get_object_poses()",
        )
        return object_pose
    except (ClothObjectPoseUnavailableError, ClothObjectPoseValidationError) as exc:
        errors.append(f"env.get_object_poses failed: {exc}")
    except Exception as exc:
        errors.append(f"env.get_object_poses raised unexpected error: {exc}")

    object_pose = get_cloth_keypoint_object_poses_world(env)
    if object_pose is not None:
        try:
            validate_semantic_object_pose_dict(
                object_pose,
                context="annotation cloth mesh fallback",
            )
            return object_pose
        except ClothObjectPoseValidationError as exc:
            errors.append(f"cloth mesh fallback returned invalid poses: {exc}")
    else:
        errors.append("cloth mesh fallback returned no object poses")

    raise DatagenObjectPoseCaptureError("; ".join(errors))
