"""Observation functions for the garment folding environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from lehome.utils.robot_utils import is_so101_at_rest_pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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
