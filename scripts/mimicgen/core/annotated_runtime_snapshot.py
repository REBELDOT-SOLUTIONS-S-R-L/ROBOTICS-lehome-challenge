"""Shared per-step runtime snapshot for integrated annotated teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lehome.tasks.fold_cloth.checkpoint_mappings import CHECKPOINT_LABELS, semantic_keypoints_from_positions
from lehome.tasks.fold_cloth.mdp.observations import (
    FoldClothSubtaskObservationContext,
    build_subtask_observation_context,
    fold_success as fold_success_observation,
)
from lehome.utils.robot_utils import is_so101_at_rest_pose


@dataclass
class AnnotatedRuntimeSnapshot:
    """Single-step state shared by the annotated teleop hot path."""

    object_pose: dict[str, torch.Tensor]
    eef_pose: dict[str, torch.Tensor]
    target_eef_pose: dict[str, torch.Tensor]
    gripper_actions: dict[str, torch.Tensor]
    joint_actions: torch.Tensor | None
    joint_pos: dict[str, torch.Tensor]
    joint_vel: dict[str, torch.Tensor]
    observation_context: FoldClothSubtaskObservationContext


def _pose_from_position(pos: torch.Tensor) -> torch.Tensor:
    pose = torch.eye(4, device=pos.device, dtype=pos.dtype).unsqueeze(0)
    pose[:, :3, 3] = pos.reshape(1, 3)
    return pose


def _get_gripper_joint_index(arm: Any) -> int:
    joint_names = list(getattr(arm.data, "joint_names", []) or [])
    if "gripper" in joint_names:
        return int(joint_names.index("gripper"))
    return 5


def _normalize_pose_tensor(value: torch.Tensor) -> torch.Tensor:
    pose = torch.as_tensor(value, dtype=torch.float32)
    if pose.ndim == 2 and pose.shape == (4, 4):
        pose = pose.unsqueeze(0)
    if pose.ndim != 3 or pose.shape[-2:] != (4, 4):
        raise ValueError(f"Expected pose tensor shaped (N, 4, 4), got {tuple(pose.shape)}.")
    return pose


def _slice_joint_tensor(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor[:1].reshape(1, -1)


def capture_annotated_runtime_snapshot(
    env: Any,
    reference_action: torch.Tensor,
    *,
    include_fold_success: bool = False,
) -> AnnotatedRuntimeSnapshot:
    """Capture one shared snapshot after a teleop step."""
    garment_obj = getattr(env, "object", None)
    if garment_obj is None or not hasattr(garment_obj, "check_points"):
        raise RuntimeError("Annotated runtime snapshot requires env.object.check_points.")

    check_points = tuple(getattr(garment_obj, "check_points", ())[: len(CHECKPOINT_LABELS)])
    if len(check_points) < len(CHECKPOINT_LABELS):
        raise RuntimeError(
            f"Annotated runtime snapshot expected at least {len(CHECKPOINT_LABELS)} checkpoints, "
            f"got {len(check_points)}."
        )

    checkpoint_positions = garment_obj.get_checkpoint_world_positions(
        check_points,
        as_numpy=True,
    )
    semantic_points_np = semantic_keypoints_from_positions(checkpoint_positions)
    semantic_keypoints_world = {
        name: torch.as_tensor(point, device=env.device, dtype=torch.float32).reshape(1, 3)
        for name, point in semantic_points_np.items()
    }
    object_pose = {
        name: _pose_from_position(pos)
        for name, pos in semantic_keypoints_world.items()
    }

    eef_pose = {
        "left_arm": _normalize_pose_tensor(env.get_robot_eef_pose("left_arm", env_ids=[0])),
        "right_arm": _normalize_pose_tensor(env.get_robot_eef_pose("right_arm", env_ids=[0])),
    }
    eef_world_positions = {
        arm_name: pose[:, :3, 3]
        for arm_name, pose in eef_pose.items()
    }

    gripper_actions = {
        arm_name: torch.as_tensor(value, device=env.device, dtype=torch.float32).reshape(1, 1)
        for arm_name, value in env.actions_to_gripper_actions(reference_action).items()
    }

    joint_actions: torch.Tensor | None = None
    if reference_action is not None:
        action_tensor = torch.as_tensor(reference_action, device=env.device, dtype=torch.float32)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.ndim == 2 and action_tensor.shape[-1] >= 12:
            joint_actions = action_tensor[:1, :12].clone()

    joint_pos: dict[str, torch.Tensor] = {}
    joint_vel: dict[str, torch.Tensor] = {}

    close_threshold = float(getattr(getattr(env, "cfg", None), "subtask_gripper_close_threshold", 0.5))
    gripper_closed_by_arm: dict[str, torch.Tensor] = {}
    arm_at_rest_by_arm: dict[str, torch.Tensor] = {}
    for arm_name in ("left_arm", "right_arm"):
        arm = env.scene[arm_name]
        joint_pos[arm_name] = _slice_joint_tensor(arm.data.joint_pos)
        joint_vel_tensor = getattr(arm.data, "joint_vel", None)
        if joint_vel_tensor is None:
            joint_vel_tensor = getattr(arm.data, "joint_velocity", None)
        if joint_vel_tensor is None:
            joint_vel_tensor = torch.zeros_like(joint_pos[arm_name])
        joint_vel[arm_name] = _slice_joint_tensor(joint_vel_tensor)
        gripper_joint_idx = _get_gripper_joint_index(arm)
        gripper_closed_by_arm[arm_name] = (
            arm.data.joint_pos[:1, gripper_joint_idx : gripper_joint_idx + 1] > close_threshold
        )
        arm_at_rest_by_arm[arm_name] = is_so101_at_rest_pose(
            arm.data.joint_pos[:1],
            arm.data.joint_names,
        ).reshape(1, 1)

    fold_success_value = None
    if include_fold_success:
        fold_success_value = fold_success_observation(env, env_ids=[0])

    observation_context = build_subtask_observation_context(
        env,
        env_ids=[0],
        semantic_keypoints_world=semantic_keypoints_world,
        eef_world_positions=eef_world_positions,
        gripper_closed_by_arm=gripper_closed_by_arm,
        arm_at_rest_by_arm=arm_at_rest_by_arm,
        fold_success_value=fold_success_value,
        include_fold_success=include_fold_success,
    )

    return AnnotatedRuntimeSnapshot(
        object_pose=object_pose,
        eef_pose=eef_pose,
        target_eef_pose=eef_pose,
        gripper_actions=gripper_actions,
        joint_actions=joint_actions,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        observation_context=observation_context,
    )


__all__ = ["AnnotatedRuntimeSnapshot", "capture_annotated_runtime_snapshot"]
