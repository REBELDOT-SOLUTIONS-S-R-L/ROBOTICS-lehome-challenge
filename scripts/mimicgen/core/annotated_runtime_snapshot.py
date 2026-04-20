"""Shared per-step runtime snapshot for integrated annotated teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lehome.assets.robots.lerobot import get_so101_rest_pose_range
from lehome.tasks.fold_cloth.checkpoint_mappings import CHECKPOINT_LABELS
from lehome.tasks.fold_cloth.mdp.observations import (
    _DEFAULT_RELEASE_ZONE_LOWER_FRACTION,
    _DEFAULT_RELEASE_ZONE_WIDTH_FRACTION,
    FoldClothSubtaskObservationContext,
    _compute_eef_in_release_zone,
    arm_at_waiting_pos as _arm_at_waiting_pos,
    fold_success as fold_success_observation,
)

_RETURN_HOME_SIGNALS = {"left_return_home", "right_return_home"}
_REST_POSE_SPEC_CACHE: dict[tuple[str, tuple[str, ...]], tuple[tuple[int, float, float], ...]] = {}


@dataclass
class AnnotatedRuntimeSnapshot:
    """Single-step state shared by the annotated teleop hot path."""

    checkpoint_positions: torch.Tensor
    eef_pose_components: dict[str, torch.Tensor]
    gripper_actions: dict[str, torch.Tensor]
    joint_actions: torch.Tensor | None
    joint_pos: dict[str, torch.Tensor]
    observation_context: FoldClothSubtaskObservationContext


def _get_gripper_joint_index(arm: Any) -> int:
    joint_names = list(getattr(arm.data, "joint_names", []) or [])
    if "gripper" in joint_names:
        return int(joint_names.index("gripper"))
    return 5


def _slice_joint_tensor(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor[:1].reshape(1, -1)


def _cfg_float(env: Any, attr_name: str, default: float) -> float:
    cfg = getattr(env, "cfg", None)
    value = getattr(cfg, attr_name, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_optional_bool_column(
    device: torch.device | str,
    value: Any | None,
) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 0:
        return torch.tensor([[bool(tensor.item())]], dtype=torch.bool, device=device)
    tensor = tensor.to(dtype=torch.bool).reshape(-1)
    if tensor.numel() == 0:
        return None
    return tensor[:1].reshape(1, 1)


def _get_rest_pose_specs(
    arm_name: str | None,
    joint_names: list[str],
) -> tuple[tuple[int, float, float], ...]:
    normalized_arm_name = str(arm_name or "")
    cache_key = (normalized_arm_name, tuple(joint_names))
    cached = _REST_POSE_SPEC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rest_pose_range = get_so101_rest_pose_range(arm_name)
    specs = tuple(
        (int(joint_names.index(joint_name)), float(min_pos), float(max_pos))
        for joint_name, (min_pos, max_pos) in rest_pose_range.items()
    )
    _REST_POSE_SPEC_CACHE[cache_key] = specs
    return specs


def _is_so101_at_rest_pose_fast(
    joint_pos: torch.Tensor,
    joint_names: list[str],
    arm_name: str | None = None,
) -> torch.Tensor:
    joint_pos_deg = joint_pos[:, :].to(dtype=torch.float32) / torch.pi * 180.0
    is_reset = torch.ones((joint_pos_deg.shape[0], 1), dtype=torch.bool, device=joint_pos_deg.device)
    for joint_idx, min_pos, max_pos in _get_rest_pose_specs(arm_name, joint_names):
        joint_ok = torch.logical_and(joint_pos_deg[:, joint_idx : joint_idx + 1] > min_pos, joint_pos_deg[:, joint_idx : joint_idx + 1] < max_pos)
        is_reset = torch.logical_and(is_reset, joint_ok)
    return is_reset


def _get_eef_pose_components(env: Any, arm_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    arm = env.scene[arm_name]
    eef_body_idx = int(env._get_eef_body_idx(arm_name))
    pos_w = torch.as_tensor(
        arm.data.body_link_pos_w[:1, eef_body_idx],
        device=env.device,
        dtype=torch.float32,
    ).reshape(1, 3)
    quat_w = torch.as_tensor(
        arm.data.body_link_quat_w[:1, eef_body_idx],
        device=env.device,
        dtype=torch.float32,
    ).reshape(1, 4)
    return pos_w, quat_w


def _extract_gripper_actions(reference_action: Any, device: torch.device | str) -> dict[str, torch.Tensor]:
    action_tensor = torch.as_tensor(reference_action, device=device, dtype=torch.float32)
    if action_tensor.ndim == 1:
        action_tensor = action_tensor.unsqueeze(0)
    action_dim = int(action_tensor.shape[-1])
    if action_dim >= 12:
        return {
            "left_arm": action_tensor[:1, 5:6].clone(),
            "right_arm": action_tensor[:1, 11:12].clone(),
        }
    return {
        "left_arm": torch.zeros((1, 1), device=device, dtype=torch.float32),
        "right_arm": torch.zeros((1, 1), device=device, dtype=torch.float32),
    }


def _effective_gripper_closed(
    actual_closed: torch.Tensor,
    commanded_closed: torch.Tensor,
) -> torch.Tensor:
    # For annotation, an explicit open command should count as released even if
    # the simulated gripper joint lags behind for a few frames.
    return actual_closed & commanded_closed


def capture_annotated_runtime_snapshot(
    env: Any,
    reference_action: torch.Tensor,
    *,
    include_fold_success: bool = False,
    rest_pose_arms: tuple[str, ...] = (),
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
        as_numpy=False,
    )
    semantic_keypoints_world = {
        name: torch.as_tensor(checkpoint_positions[idx], device=env.device, dtype=torch.float32).reshape(1, 3)
        for idx, name in enumerate(CHECKPOINT_LABELS)
    }
    
    checkpoint_positions = torch.stack(
        [semantic_keypoints_world[name].reshape(3) for name in CHECKPOINT_LABELS],
        dim=0,
    )

    eef_pose_components: dict[str, torch.Tensor] = {}
    eef_world_positions = {
        arm_name: None
        for arm_name in ("left_arm", "right_arm")
    }
    for arm_name in ("left_arm", "right_arm"):
        eef_pos_w, eef_quat_w = _get_eef_pose_components(env, arm_name)
        eef_pose_components[arm_name] = torch.cat([eef_pos_w, eef_quat_w], dim=-1)
        eef_world_positions[arm_name] = eef_pos_w

    gripper_actions = _extract_gripper_actions(reference_action, env.device)

    joint_actions: torch.Tensor | None = None
    if reference_action is not None:
        action_tensor = torch.as_tensor(reference_action, device=env.device, dtype=torch.float32)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.ndim == 2 and action_tensor.shape[-1] >= 12:
            joint_actions = action_tensor[:1, :12].clone()

    joint_pos: dict[str, torch.Tensor] = {}

    close_threshold = float(getattr(getattr(env, "cfg", None), "subtask_gripper_close_threshold", 0.20))
    gripper_closed_by_arm: dict[str, torch.Tensor] = {}
    arm_at_rest_by_arm: dict[str, torch.Tensor] = {}
    rest_pose_arm_names = set(rest_pose_arms)
    for arm_name in ("left_arm", "right_arm"):
        arm = env.scene[arm_name]
        joint_pos[arm_name] = _slice_joint_tensor(arm.data.joint_pos)
        gripper_joint_idx = _get_gripper_joint_index(arm)
        actual_closed = (
            arm.data.joint_pos[:1, gripper_joint_idx : gripper_joint_idx + 1] < close_threshold
        )
        commanded_closed = (
            gripper_actions.get(
                arm_name,
                torch.zeros((1, 1), device=env.device, dtype=torch.float32),
            )
            < close_threshold
        )
        gripper_closed_by_arm[arm_name] = _effective_gripper_closed(
            actual_closed,
            commanded_closed,
        )
        if arm_name in rest_pose_arm_names:
            arm_at_rest_by_arm[arm_name] = _is_so101_at_rest_pose_fast(
                joint_pos[arm_name],
                arm.data.joint_names,
                arm_name=arm_name,
            )

    fold_success_value = None
    if include_fold_success:
        fold_success_value = fold_success_observation(env, env_ids=[0])

    # Per-arm "EEF in narrow release zone" bool, computed from the 4 corner
    # garment keypoints.  Needed by the ``*_middle_to_lower`` and
    # ``release_*_middle`` signals; without this field they evaluate to False
    # unconditionally because the hot-path snapshot bypasses
    # ``build_subtask_observation_context``.
    eef_in_release_zone_by_arm = _compute_eef_in_release_zone(
        env,
        num_envs=1,
        semantic_points=semantic_keypoints_world,
        eef_positions=eef_world_positions,
        width_fraction=_cfg_float(
            env,
            "subtask_release_zone_width_fraction",
            _DEFAULT_RELEASE_ZONE_WIDTH_FRACTION,
        ),
        lower_fraction=_cfg_float(
            env,
            "subtask_release_zone_lower_fraction",
            _DEFAULT_RELEASE_ZONE_LOWER_FRACTION,
        ),
    )

    observation_context = FoldClothSubtaskObservationContext(
        device=env.device,
        num_envs=1,
        semantic_keypoints_world=semantic_keypoints_world,
        eef_world_positions=eef_world_positions,
        gripper_closed_by_arm=gripper_closed_by_arm,
        arm_at_rest_by_arm=arm_at_rest_by_arm,
        arm_at_waiting_pos_by_arm={
            arm_name: _arm_at_waiting_pos(env, arm_name, env_ids=[0])
            for arm_name in ("left_arm", "right_arm")
        },
        grasp_eef_to_keypoint_threshold_m=_cfg_float(
            env,
            "subtask_grasp_eef_to_keypoint_threshold_m",
            0.15,
        ),
        middle_to_lower_threshold_m=_cfg_float(
            env,
            "subtask_middle_to_lower_threshold_m",
            0.10,
        ),
        middle_to_lower_middle_keypoint_max_z_m=_cfg_float(
            env,
            "subtask_middle_to_lower_middle_keypoint_max_z_m",
            0.53,
        ),
        lower_to_upper_threshold_m=_cfg_float(
            env,
            "subtask_lower_to_upper_threshold_m",
            0.12,
        ),
        eef_in_release_zone_by_arm=eef_in_release_zone_by_arm,
        fold_success=_normalize_optional_bool_column(env.device, fold_success_value),
    )

    return AnnotatedRuntimeSnapshot(
        checkpoint_positions=checkpoint_positions,
        eef_pose_components=eef_pose_components,
        gripper_actions=gripper_actions,
        joint_actions=joint_actions,
        joint_pos=joint_pos,
        observation_context=observation_context,
    )


def ensure_return_home_snapshot_fields(
    env: Any,
    snapshot: AnnotatedRuntimeSnapshot,
    arm_names: tuple[str, ...],
    *,
    include_fold_success: bool = True,
) -> AnnotatedRuntimeSnapshot:
    """Populate only the return-home fields that may be missing from an existing snapshot."""
    for arm_name in arm_names:
        if arm_name in snapshot.observation_context.arm_at_rest_by_arm:
            continue
        arm = env.scene[arm_name]
        snapshot.observation_context.arm_at_rest_by_arm[arm_name] = _is_so101_at_rest_pose_fast(
            snapshot.joint_pos[arm_name],
            arm.data.joint_names,
            arm_name=arm_name,
        )
    if include_fold_success and snapshot.observation_context.fold_success is None:
        snapshot.observation_context.fold_success = _normalize_optional_bool_column(
            env.device,
            fold_success_observation(env, env_ids=[0]),
        )
    return snapshot


__all__ = [
    "AnnotatedRuntimeSnapshot",
    "capture_annotated_runtime_snapshot",
    "ensure_return_home_snapshot_fields",
]
