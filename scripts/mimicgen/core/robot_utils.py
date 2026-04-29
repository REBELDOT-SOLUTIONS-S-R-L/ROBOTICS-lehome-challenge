"""Shared robot pose and IK helpers for MimicGen runtime modules."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
import isaaclab.utils.math as PoseUtils

from lehome.utils.robot_utils import is_so101_at_rest_pose

try:
    from scripts.utils.annotate_utils import orthonormalize_rotations as _orthonormalize_rotations
except ImportError:
    scripts_dir = Path(__file__).resolve().parents[2]
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    from utils.annotate_utils import orthonormalize_rotations as _orthonormalize_rotations


def get_robot_eef_pose_world(env: ManagerBasedRLMimicEnv, eef_name: str) -> torch.Tensor | None:
    """Fetch EEF pose from articulation link world pose when available."""
    try:
        arm = env.scene[eef_name]
    except Exception:
        return None

    try:
        if hasattr(env, "_get_eef_body_idx"):
            eef_body_idx = int(env._get_eef_body_idx(eef_name))
        else:
            eef_body_idx = int(arm.data.body_link_pos_w.shape[1] - 1)
        eef_pos_w = arm.data.body_link_pos_w[:, eef_body_idx]
        eef_quat_w = arm.data.body_link_quat_w[:, eef_body_idx]
        quat_norm = torch.linalg.norm(eef_quat_w, dim=-1, keepdim=True).clamp_min(1e-12)
        eef_quat_w = eef_quat_w / quat_norm
        pose = PoseUtils.make_pose(eef_pos_w, PoseUtils.matrix_from_quat(eef_quat_w))
        return _orthonormalize_rotations(pose)
    except Exception:
        return None


def quat_xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternion tensor from (x, y, z, w) to (w, x, y, z)."""
    return torch.stack(
        [quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]],
        dim=-1,
    )


def arm_root_pose_world(env: ManagerBasedRLMimicEnv, arm_name: str) -> torch.Tensor:
    """Get arm base pose in world as 4x4 transform for env 0."""
    arm = env.scene[arm_name]
    pos_w = arm.data.root_pos_w[0:1]
    quat_w = arm.data.root_quat_w[0:1]
    return PoseUtils.make_pose(pos_w, PoseUtils.matrix_from_quat(quat_w))[0]


def get_arm_eef_pose_robot_frame(
    arm: Any,
    eef_body_idx: int,
    device: torch.device | str,
) -> np.ndarray | None:
    """Return the end-effector pose expressed in the arm root frame for env 0."""
    root_pos_w = getattr(arm.data, "root_pos_w", None)
    root_quat_w = getattr(arm.data, "root_quat_w", None)
    if root_pos_w is None or root_quat_w is None:
        return None

    body_pos_w = getattr(arm.data, "body_link_pos_w", None)
    body_quat_w = getattr(arm.data, "body_link_quat_w", None)
    if body_pos_w is None or body_quat_w is None:
        body_pos_w = getattr(arm.data, "body_pos_w", None)
        body_quat_w = getattr(arm.data, "body_quat_w", None)
    if body_pos_w is None or body_quat_w is None:
        return None

    root_pos = torch.as_tensor(root_pos_w[0], device=device, dtype=torch.float32).unsqueeze(0)
    root_quat = torch.as_tensor(root_quat_w[0], device=device, dtype=torch.float32).unsqueeze(0)
    eef_pos = torch.as_tensor(body_pos_w[0, eef_body_idx], device=device, dtype=torch.float32).unsqueeze(0)
    eef_quat = torch.as_tensor(body_quat_w[0, eef_body_idx], device=device, dtype=torch.float32).unsqueeze(0)

    ee_pos_robot, ee_quat_robot = PoseUtils.subtract_frame_transforms(
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


def decode_ik_action_trajectory(
    ik_actions: torch.Tensor,
    eef_names: list[str],
    quat_order: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Decode an entire IK action trajectory into per-arm pose and gripper tensors."""
    if not eef_names:
        raise ValueError("No end-effectors configured for IK replay.")
    if len(eef_names) > 2:
        raise ValueError(f"IK replay supports at most 2 end-effectors, got {len(eef_names)}.")

    ik_actions = ik_actions.to(device=device, dtype=torch.float32)
    action_dim = int(ik_actions.shape[1])
    expected_dim = 8 * len(eef_names)
    if action_dim != expected_dim:
        raise ValueError(
            f"IK action dimension mismatch: expected {expected_dim} for eefs={eef_names}, got {action_dim}."
        )

    pose_by_eef: dict[str, torch.Tensor] = {}
    gripper_by_eef: dict[str, torch.Tensor] = {}
    step_count = int(ik_actions.shape[0])
    eye = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(step_count, 1, 1)
    for i, eef_name in enumerate(eef_names):
        chunk = ik_actions[:, i * 8 : (i + 1) * 8]
        pos = chunk[:, :3]
        quat = chunk[:, 3:7]
        if quat_order == "xyzw":
            quat = quat_xyzw_to_wxyz(quat)
        elif quat_order != "wxyz":
            raise ValueError(f"Unsupported quaternion order: {quat_order}")

        pose = eye.clone()
        pose[:, :3, :3] = PoseUtils.matrix_from_quat(quat)
        pose[:, :3, 3] = pos
        pose_by_eef[eef_name] = pose
        gripper_by_eef[eef_name] = chunk[:, 7:8]
    return pose_by_eef, gripper_by_eef


def are_arms_at_rest(env: ManagerBasedRLMimicEnv) -> bool:
    """Check whether available arm articulations are within rest pose ranges."""
    try:
        at_rest_flags = []
        for arm_name in ("left_arm", "right_arm"):
            try:
                arm = env.scene[arm_name]
            except Exception:
                continue
            arm_rest = is_so101_at_rest_pose(
                arm.data.joint_pos,
                arm.data.joint_names,
                arm_name=arm_name,
            )
            at_rest_flags.append(bool(arm_rest[0].item()))
        if not at_rest_flags:
            return False
        return all(at_rest_flags)
    except Exception:
        return False


__all__ = [
    "are_arms_at_rest",
    "arm_root_pose_world",
    "decode_ik_action_trajectory",
    "get_arm_eef_pose_robot_frame",
    "get_robot_eef_pose_world",
    "quat_xyzw_to_wxyz",
]
