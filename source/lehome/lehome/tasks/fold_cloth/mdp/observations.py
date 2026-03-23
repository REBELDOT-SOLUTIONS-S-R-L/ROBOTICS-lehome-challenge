"""Observation functions for the garment folding environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

from isaaclab.managers import SceneEntityCfg

from ..checkpoint_mappings import (
    CHECKPOINT_LABELS,
    semantic_keypoints_from_positions as map_semantic_keypoints_from_positions,
)
from lehome.utils.robot_utils import is_so101_at_rest_pose
from lehome.utils.success_checker_chanllege import success_checker_garment_fold

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_DEFAULT_GRIPPER_JOINT_IDX = 5
_DEFAULT_GRIPPER_CLOSE_THRESHOLD = 0.5
_DEFAULT_GRASP_EEF_TO_KEYPOINT_THRESHOLD_M = 0.05
_DEFAULT_MIDDLE_TO_LOWER_THRESHOLD_M = 0.10
_DEFAULT_LOWER_TO_UPPER_THRESHOLD_M = 0.12


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: Sequence[int] | None) -> tuple[Sequence[int] | slice, int]:
    if hasattr(env, "_resolve_env_ids"):
        return env._resolve_env_ids(env_ids)
    if env_ids is None:
        return slice(None), int(env.num_envs)
    if isinstance(env_ids, slice):
        return env_ids, int(env.num_envs)
    return env_ids, len(env_ids)


def _resolve_arm_name(arm: str | SceneEntityCfg) -> str:
    if isinstance(arm, SceneEntityCfg):
        return arm.name
    return str(arm)


def _resolve_checkpoint_name(checkpoint_name: str) -> str:
    checkpoint_name = str(checkpoint_name)
    if checkpoint_name not in CHECKPOINT_LABELS:
        raise KeyError(
            f"Unsupported garment checkpoint name {checkpoint_name!r}. "
            f"Expected one of {CHECKPOINT_LABELS}."
        )
    return checkpoint_name


def _cfg_float(env: ManagerBasedEnv, attr_name: str, default: float) -> float:
    cfg = getattr(env, "cfg", None)
    value = getattr(cfg, attr_name, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _false_bool_column(env: ManagerBasedEnv, num_envs: int) -> torch.Tensor:
    return torch.zeros((num_envs, 1), dtype=torch.bool, device=env.device)


def _true_bool_column(env: ManagerBasedEnv, num_envs: int) -> torch.Tensor:
    return torch.ones((num_envs, 1), dtype=torch.bool, device=env.device)


def _full_float_column(env: ManagerBasedEnv, num_envs: int, value: float) -> torch.Tensor:
    return torch.full((num_envs, 1), float(value), dtype=torch.float32, device=env.device)


def _get_scene_arm(env: ManagerBasedEnv, arm: str | SceneEntityCfg):
    arm_name = _resolve_arm_name(arm)
    return arm_name, env.scene[arm_name]


def _get_gripper_joint_index(arm: Any) -> int:
    joint_names = list(getattr(arm.data, "joint_names", []) or [])
    if "gripper" in joint_names:
        return int(joint_names.index("gripper"))
    return _DEFAULT_GRIPPER_JOINT_IDX


def _get_eef_world_position(
    env: ManagerBasedEnv,
    arm: str | SceneEntityCfg,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    env_ids, num_envs = _resolve_env_ids(env, env_ids)
    arm_name, arm_entity = _get_scene_arm(env, arm)
    if hasattr(env, "get_robot_eef_pose"):
        pose = env.get_robot_eef_pose(arm_name, env_ids=env_ids)
        pose = torch.as_tensor(pose, device=env.device, dtype=torch.float32)
        if pose.ndim == 2 and pose.shape == (4, 4):
            pose = pose.unsqueeze(0)
        return pose[:, :3, 3].reshape(num_envs, 3)

    if not hasattr(env, "_get_eef_body_idx"):
        raise AttributeError(f"{type(env).__name__} does not expose get_robot_eef_pose() or _get_eef_body_idx().")
    eef_body_idx = int(env._get_eef_body_idx(arm_name))
    return arm_entity.data.body_link_pos_w[env_ids, eef_body_idx].reshape(num_envs, 3)


def _get_semantic_keypoint_positions_world(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None = None,
) -> dict[str, torch.Tensor] | None:
    _, num_envs = _resolve_env_ids(env, env_ids)
    garment_obj = getattr(env, "object", None)
    if garment_obj is None or not hasattr(garment_obj, "check_points"):
        return None

    check_points = getattr(garment_obj, "check_points", None)
    if not check_points or len(check_points) < len(CHECKPOINT_LABELS):
        return None

    try:
        kp_positions = garment_obj.get_checkpoint_world_positions(
            tuple(check_points[: len(CHECKPOINT_LABELS)]),
            as_numpy=True,
        )
    except Exception:
        return None

    kp_positions = np.asarray(kp_positions, dtype=np.float32)
    try:
        semantic_points = map_semantic_keypoints_from_positions(kp_positions)
    except Exception:
        return None

    points_world: dict[str, torch.Tensor] = {}
    for name, point in semantic_points.items():
        points_world[name] = (
            torch.as_tensor(point, dtype=torch.float32, device=env.device)
            .reshape(1, 3)
            .expand(num_envs, 3)
            .clone()
        )
    return points_world


def _fold_success_scalar(env: ManagerBasedEnv) -> bool:
    if hasattr(env, "_check_success"):
        try:
            return bool(env._check_success())
        except Exception:
            pass

    garment_object = getattr(env, "object", None)
    garment_loader = getattr(env, "garment_loader", None)
    cfg = getattr(env, "cfg", None)
    garment_name = getattr(cfg, "garment_name", None)
    if garment_object is None or garment_loader is None or garment_name is None:
        return False

    try:
        garment_type = garment_loader.get_garment_type(garment_name)
        result = success_checker_garment_fold(garment_object, garment_type)
    except Exception:
        return False

    if isinstance(result, dict):
        return bool(result.get("success", False))
    return bool(result)


def robot_rest_pose(
    env: ManagerBasedEnv,
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
) -> torch.Tensor:
    """Return per-arm rest-pose indicators for the bimanual SO101 setup.

    The observation is a 2D vector per environment:
      [left_arm_at_rest, right_arm_at_rest]
    represented as float values in {0.0, 1.0}.
    """

    left_arm = env.scene[left_arm_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]

    left_at_rest = is_so101_at_rest_pose(left_arm.data.joint_pos, left_arm.data.joint_names)
    right_at_rest = is_so101_at_rest_pose(right_arm.data.joint_pos, right_arm.data.joint_names)

    return torch.stack(
        (
            left_at_rest.to(dtype=torch.float32),
            right_at_rest.to(dtype=torch.float32),
        ),
        dim=-1,
    )


def gripper_closed(
    env: ManagerBasedEnv,
    arm: str | SceneEntityCfg,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return whether the given arm gripper is closed."""
    env_ids, num_envs = _resolve_env_ids(env, env_ids)
    _, arm_entity = _get_scene_arm(env, arm)
    gripper_joint_idx = _get_gripper_joint_index(arm_entity)
    threshold = _cfg_float(env, "subtask_gripper_close_threshold", _DEFAULT_GRIPPER_CLOSE_THRESHOLD)
    joint_pos = arm_entity.data.joint_pos[env_ids, gripper_joint_idx].reshape(num_envs, 1)
    return joint_pos > threshold


def arm_at_rest(
    env: ManagerBasedEnv,
    arm: str | SceneEntityCfg,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return whether the given arm is at the rest pose."""
    env_ids, num_envs = _resolve_env_ids(env, env_ids)
    _, arm_entity = _get_scene_arm(env, arm)
    at_rest = is_so101_at_rest_pose(
        arm_entity.data.joint_pos[env_ids],
        arm_entity.data.joint_names,
    )
    return at_rest.reshape(num_envs, 1)


def fold_success(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return whether the garment is folded successfully, without requiring rest-pose."""
    _, num_envs = _resolve_env_ids(env, env_ids)
    if _fold_success_scalar(env):
        return _true_bool_column(env, num_envs)
    return _false_bool_column(env, num_envs)


def eef_to_keypoint_distance(
    env: ManagerBasedEnv,
    arm: str | SceneEntityCfg,
    checkpoint_name: str,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return the world-frame distance between an arm EEF and a garment checkpoint."""
    _, num_envs = _resolve_env_ids(env, env_ids)
    checkpoint_name = _resolve_checkpoint_name(checkpoint_name)
    semantic_points = _get_semantic_keypoint_positions_world(env, env_ids=env_ids)
    if semantic_points is None or checkpoint_name not in semantic_points:
        return _full_float_column(env, num_envs, float("inf"))

    eef_pos = _get_eef_world_position(env, arm, env_ids=env_ids)
    kp_pos = semantic_points[checkpoint_name]
    return torch.linalg.norm(eef_pos - kp_pos, dim=-1, keepdim=True)


def keypoint_pair_distance(
    env: ManagerBasedEnv,
    checkpoint_a: str,
    checkpoint_b: str,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return the world-frame distance between two garment checkpoints."""
    _, num_envs = _resolve_env_ids(env, env_ids)
    checkpoint_a = _resolve_checkpoint_name(checkpoint_a)
    checkpoint_b = _resolve_checkpoint_name(checkpoint_b)
    semantic_points = _get_semantic_keypoint_positions_world(env, env_ids=env_ids)
    if semantic_points is None or checkpoint_a not in semantic_points or checkpoint_b not in semantic_points:
        return _full_float_column(env, num_envs, float("inf"))

    pos_a = semantic_points[checkpoint_a]
    pos_b = semantic_points[checkpoint_b]
    return torch.linalg.norm(pos_a - pos_b, dim=-1, keepdim=True)


def grasp_left_middle(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_grasp_eef_to_keypoint_threshold_m",
        _DEFAULT_GRASP_EEF_TO_KEYPOINT_THRESHOLD_M,
    )
    return gripper_closed(env, "left_arm", env_ids=env_ids) & (
        eef_to_keypoint_distance(env, "left_arm", "garment_left_middle", env_ids=env_ids) <= threshold
    )


def grasp_right_middle(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_grasp_eef_to_keypoint_threshold_m",
        _DEFAULT_GRASP_EEF_TO_KEYPOINT_THRESHOLD_M,
    )
    return gripper_closed(env, "right_arm", env_ids=env_ids) & (
        eef_to_keypoint_distance(env, "right_arm", "garment_right_middle", env_ids=env_ids) <= threshold
    )


def left_middle_to_lower(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_middle_to_lower_threshold_m",
        _DEFAULT_MIDDLE_TO_LOWER_THRESHOLD_M,
    )
    return keypoint_pair_distance(
        env,
        "garment_left_middle",
        "garment_left_lower",
        env_ids=env_ids,
    ) <= threshold


def right_middle_to_lower(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_middle_to_lower_threshold_m",
        _DEFAULT_MIDDLE_TO_LOWER_THRESHOLD_M,
    )
    return keypoint_pair_distance(
        env,
        "garment_right_middle",
        "garment_right_lower",
        env_ids=env_ids,
    ) <= threshold


def grasp_left_lower(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_grasp_eef_to_keypoint_threshold_m",
        _DEFAULT_GRASP_EEF_TO_KEYPOINT_THRESHOLD_M,
    )
    return gripper_closed(env, "left_arm", env_ids=env_ids) & (
        eef_to_keypoint_distance(env, "left_arm", "garment_left_lower", env_ids=env_ids) <= threshold
    )


def grasp_right_lower(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_grasp_eef_to_keypoint_threshold_m",
        _DEFAULT_GRASP_EEF_TO_KEYPOINT_THRESHOLD_M,
    )
    return gripper_closed(env, "right_arm", env_ids=env_ids) & (
        eef_to_keypoint_distance(env, "right_arm", "garment_right_lower", env_ids=env_ids) <= threshold
    )


def left_lower_to_upper(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_lower_to_upper_threshold_m",
        _DEFAULT_LOWER_TO_UPPER_THRESHOLD_M,
    )
    return keypoint_pair_distance(
        env,
        "garment_left_lower",
        "garment_left_upper",
        env_ids=env_ids,
    ) <= threshold


def right_lower_to_upper(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_lower_to_upper_threshold_m",
        _DEFAULT_LOWER_TO_UPPER_THRESHOLD_M,
    )
    return keypoint_pair_distance(
        env,
        "garment_right_lower",
        "garment_right_upper",
        env_ids=env_ids,
    ) <= threshold


def left_return_home(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    return fold_success(env, env_ids=env_ids) & arm_at_rest(env, "left_arm", env_ids=env_ids)


def right_return_home(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    return fold_success(env, env_ids=env_ids) & arm_at_rest(env, "right_arm", env_ids=env_ids)


SUBTASK_SIGNAL_OBSERVATION_FNS = {
    "grasp_left_middle": grasp_left_middle,
    "grasp_right_middle": grasp_right_middle,
    "left_middle_to_lower": left_middle_to_lower,
    "right_middle_to_lower": right_middle_to_lower,
    "grasp_left_lower": grasp_left_lower,
    "grasp_right_lower": grasp_right_lower,
    "left_lower_to_upper": left_lower_to_upper,
    "right_lower_to_upper": right_lower_to_upper,
    "left_return_home": left_return_home,
    "right_return_home": right_return_home,
}


def get_subtask_signal_observation(
    env: ManagerBasedEnv,
    signal_name: str,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Evaluate a single instantaneous subtask predicate by exact signal name."""
    try:
        fn = SUBTASK_SIGNAL_OBSERVATION_FNS[str(signal_name)]
    except KeyError as exc:
        raise KeyError(
            f"Unsupported subtask signal {signal_name!r}. "
            f"Expected one of {tuple(SUBTASK_SIGNAL_OBSERVATION_FNS)}."
        ) from exc
    return fn(env, env_ids=env_ids)


def get_subtask_signal_observations(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate all instantaneous subtask predicates."""
    return {
        signal_name: fn(env, env_ids=env_ids)
        for signal_name, fn in SUBTASK_SIGNAL_OBSERVATION_FNS.items()
    }


__all__ = [
    "SUBTASK_SIGNAL_OBSERVATION_FNS",
    "arm_at_rest",
    "eef_to_keypoint_distance",
    "fold_success",
    "get_subtask_signal_observation",
    "get_subtask_signal_observations",
    "grasp_left_lower",
    "grasp_left_middle",
    "grasp_right_lower",
    "grasp_right_middle",
    "gripper_closed",
    "keypoint_pair_distance",
    "left_lower_to_upper",
    "left_middle_to_lower",
    "left_return_home",
    "right_lower_to_upper",
    "right_middle_to_lower",
    "right_return_home",
    "robot_rest_pose",
]
