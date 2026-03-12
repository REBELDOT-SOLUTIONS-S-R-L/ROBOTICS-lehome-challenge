"""Termination functions for the garment folding environment.

These functions use lehome's GarmentObject and success checker
instead of LeIsaac's ClothObject. They access the garment object
from the custom env attribute (env.object) rather than the scene.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch
import numpy as np

from isaaclab.managers import SceneEntityCfg

from lehome.assets.robots.lerobot import SO101_FOLLOWER_REST_POSE_RANGE
from lehome.utils.success_checker_chanllege import success_checker_garment_fold

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def is_so101_at_rest_pose(
    joint_pos: torch.Tensor,
    joint_names: list[str],
    tolerance_deg: float = 20.0,
) -> torch.Tensor:
    """Check if SO101 arm joints are within the rest pose range.

    Uses SO101_FOLLOWER_REST_POSE_RANGE from lehome to define the acceptable
    range for each joint.

    Args:
        joint_pos: Joint positions tensor. Shape is (num_envs, num_joints).
        joint_names: List of joint names corresponding to joint_pos columns.
        tolerance_deg: Not used (ranges already include tolerance).

    Returns:
        Boolean tensor of shape (num_envs,) indicating if all joints are at rest.
    """
    at_rest = torch.ones(joint_pos.shape[0], dtype=torch.bool, device=joint_pos.device)

    for i, name in enumerate(joint_names):
        if name in SO101_FOLLOWER_REST_POSE_RANGE:
            low_deg, high_deg = SO101_FOLLOWER_REST_POSE_RANGE[name]
            low_rad = math.radians(low_deg)
            high_rad = math.radians(high_deg)
            joint_in_range = (joint_pos[:, i] >= low_rad) & (joint_pos[:, i] <= high_rad)
            at_rest = at_rest & joint_in_range

    return at_rest


def garment_folded(
    env: ManagerBasedEnv,
    distance_threshold: float = 0.10,
) -> torch.Tensor:
    """Determine if the garment folding task is completed successfully.

    This function evaluates:
    1. Garment is properly folded (via success_checker_garment_fold)
    2. Both robot arms are at rest pose

    Accesses the garment object from env.object (set by GarmentFoldEnv).

    Args:
        env: The RL environment instance (must be GarmentFoldEnv).
        distance_threshold: Threshold for keypoint distance (unused, kept for API compat).

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # Check if both arms are at rest pose
    left_arm = env.scene["left_arm"]
    right_arm = env.scene["right_arm"]

    is_rest = torch.logical_and(
        is_so101_at_rest_pose(left_arm.data.joint_pos, left_arm.data.joint_names),
        is_so101_at_rest_pose(right_arm.data.joint_pos, right_arm.data.joint_names),
    )
    done = torch.logical_and(done, is_rest)

    # Check garment fold success
    garment_object = getattr(env, "object", None)
    if garment_object is None or not hasattr(garment_object, "_cloth_prim_view"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    garment_loader = getattr(env, "garment_loader", None)
    garment_name = getattr(env.cfg, "garment_name", None)
    if garment_loader is None or garment_name is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    garment_type = garment_loader.get_garment_type(garment_name)
    result = success_checker_garment_fold(garment_object, garment_type)

    if isinstance(result, dict):
        fold_success = result.get("success", False)
    elif isinstance(result, bool):
        fold_success = result
    else:
        fold_success = False

    fold_tensor = torch.tensor(
        [fold_success] * env.num_envs, dtype=torch.bool, device=env.device
    )
    done = torch.logical_and(done, fold_tensor)

    return done
