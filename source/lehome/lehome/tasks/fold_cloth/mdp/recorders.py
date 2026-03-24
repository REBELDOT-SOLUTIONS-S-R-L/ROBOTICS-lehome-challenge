"""Custom RecorderTerm for garment folding datagen_info.

Provides MimicGen with garment keypoint positions as virtual object poses,
subtask termination signals, and end-effector information for both arms.

The garment keypoints (from GarmentObject.check_points) are mapped to
semantic virtual objects used by mimic subtasks:
  - garment_left_middle / garment_right_middle
  - garment_left_lower / garment_right_lower
  - garment_left_upper / garment_right_upper
  - garment_upper_center / garment_lower_center
  - garment_kp_left / garment_kp_right / garment_center

These are stored as 4×4 homogeneous transforms (identity rotation, position from keypoints)
in datagen_info.object_pose, compatible with MimicGen's DatagenInfo format.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers.recorder_manager import RecorderTerm
from .observations import get_subtask_signal_observations
from ..checkpoint_mappings import (
    ClothObjectPoseUnavailableError,
    semantic_keypoints_from_positions as map_semantic_keypoints_from_positions,
    validate_semantic_object_pose_dict,
)

if TYPE_CHECKING:
    from isaaclab.managers.manager_term_cfg import RecorderTermCfg
    from isaaclab.envs import ManagerBasedEnv

def _pos_to_4x4(pos: torch.Tensor) -> torch.Tensor:
    """Convert a 3D position to a 4×4 homogeneous transform (identity rotation).

    Args:
        pos: Position tensor of shape (..., 3).

    Returns:
        4×4 transform of shape (..., 4, 4).
    """
    batch_shape = pos.shape[:-1]
    T = torch.eye(4, device=pos.device, dtype=pos.dtype).expand(*batch_shape, 4, 4).clone()
    T[..., :3, 3] = pos
    return T


def _semantic_keypoints_from_positions(kp_positions: np.ndarray) -> dict[str, np.ndarray]:
    """Map six garment checkpoints using the shared semantic checkpoint order."""
    return map_semantic_keypoints_from_positions(kp_positions)


class GarmentDatagenRecorder(RecorderTerm):
    """Records datagen_info with garment keypoint virtual poses and subtask signals.

    This recorder provides the information MimicGen needs to:
    1. Segment demonstrations into subtasks (via subtask_term_signals)
    2. Select source demos for transformation (via object_poses as keypoint positions)
    3. Transform trajectories relative to object frames (via eef_pose + object_pose)

    Subtask termination signals:
    - grasp_left_middle / grasp_right_middle
    - left_middle_to_lower / right_middle_to_lower
    - grasp_left_lower / grasp_right_lower
    - left_lower_to_upper / right_lower_to_upper
    - left_return_home / right_return_home
    """

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def record_post_step(self) -> tuple[str, dict]:
        """Record datagen_info after each environment step."""
        env = self._env
        device = env.device
        num_envs = env.num_envs

        # ---------------------------------------------------------------
        # 1. End-effector poses (world-frame 4x4 transforms)
        # ---------------------------------------------------------------
        left_arm = env.scene["left_arm"]
        right_arm = env.scene["right_arm"]
        left_joint_pos = left_arm.data.joint_pos  # (num_envs, 6)
        right_joint_pos = right_arm.data.joint_pos
        eef_pose = {
            "left_arm": env.get_robot_eef_pose("left_arm"),
            "right_arm": env.get_robot_eef_pose("right_arm"),
        }

        # ---------------------------------------------------------------
        # 2. Target EEF poses (FK from action targets)
        # ---------------------------------------------------------------
        try:
            action = env.action_manager.action  # (num_envs, action_dim)
            target_eef_pose = env.action_to_target_eef_pose(action)
        except Exception:
            target_eef_pose = eef_pose  # Fallback

        # ---------------------------------------------------------------
        # 3. Gripper actions
        # ---------------------------------------------------------------
        try:
            action = env.action_manager.action
            # Delegate extraction to env-level API so this recorder stays
            # compatible with both 12D joint and 16D native IK contracts.
            gripper_action = env.actions_to_gripper_actions(action)
            left_gripper = gripper_action["left_arm"]
            right_gripper = gripper_action["right_arm"]
        except Exception:
            left_gripper = left_joint_pos[:, _GRIPPER_JOINT_IDX:_GRIPPER_JOINT_IDX + 1]
            right_gripper = right_joint_pos[:, _GRIPPER_JOINT_IDX:_GRIPPER_JOINT_IDX + 1]
        gripper_action = {
            "left_arm": left_gripper,
            "right_arm": right_gripper,
        }

        # ---------------------------------------------------------------
        # 4. Object poses — garment keypoints as virtual objects
        # ---------------------------------------------------------------
        object_poses = self._compute_keypoint_object_poses(env, device, num_envs)

        # ---------------------------------------------------------------
        # 5. Subtask termination signals
        # ---------------------------------------------------------------
        subtask_term_signals = self._compute_subtask_signals(env, device, num_envs)

        datagen_info = {
            "eef_pose": eef_pose,
            "object_pose": object_poses,
            "target_eef_pose": target_eef_pose,
            "gripper_action": gripper_action,
            "subtask_term_signals": subtask_term_signals,
        }

        return "obs/datagen_info", datagen_info

    def record_pre_reset(self, env_ids: Sequence[int] | None):
        """Recorder has no persistent per-episode state to reset."""
        return None, None

    def _compute_keypoint_object_poses(
        self, env, device: torch.device, num_envs: int
    ) -> dict[str, torch.Tensor]:
        """Compute virtual 4×4 object poses from garment keypoints.

        Groups the 6 check_points into:
          - garment_kp_left:  centroid of left upper, left middle, left lower
          - garment_kp_right: centroid of right upper, right middle, right lower
          - garment_center:   centroid of all 6

        Returns:
            Dict mapping object names to (num_envs, 4, 4) tensors.
        """
        def _raise_unavailable(reason: str) -> None:
            raise ClothObjectPoseUnavailableError(
                f"GarmentDatagenRecorder could not capture cloth object poses: {reason}"
            )

        garment_obj = getattr(env, "object", None)
        if garment_obj is None or not hasattr(garment_obj, "check_points"):
            _raise_unavailable("garment object is missing or has no check_points.")

        check_points = garment_obj.check_points
        if not check_points or len(check_points) < 6:
            _raise_unavailable("garment check_points are missing or incomplete.")

        try:
            kp_positions = garment_obj.get_checkpoint_world_positions(
                check_points,
                as_numpy=True,
            )
        except Exception as exc:
            _raise_unavailable(
                f"unable to read garment checkpoint positions from GarmentObject: {exc}"
            )

        kp_positions = np.asarray(kp_positions, dtype=np.float32)
        semantic_points = _semantic_keypoints_from_positions(kp_positions)
        object_poses: dict[str, torch.Tensor] = {}
        for name, point in semantic_points.items():
            pos = torch.tensor(point, dtype=torch.float32, device=device).unsqueeze(0).expand(num_envs, -1)
            object_poses[name] = _pos_to_4x4(pos)
        validate_semantic_object_pose_dict(
            object_poses,
            context="GarmentDatagenRecorder._compute_keypoint_object_poses",
        )
        return object_poses

    def _compute_subtask_signals(
        self, env, device: torch.device, num_envs: int
    ) -> dict[str, torch.Tensor]:
        """Compute instantaneous subtask termination signals from shared observation helpers."""
        signal_map = get_subtask_signal_observations(env)
        return {
            signal_name: signal.to(device=device, dtype=torch.float32).reshape(num_envs, 1)
            for signal_name, signal in signal_map.items()
        }
