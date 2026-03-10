# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to add mimic annotations to demos to be used as source demos for mimic dataset generation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# Launching Isaac Sim Simulator first.


# add argparse arguments
parser = argparse.ArgumentParser(description="Annotate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--garment_name",
    type=str,
    default=None,
    help="Garment name (for garment tasks), e.g. Top_Long_Unseen_0.",
)
parser.add_argument(
    "--garment_version",
    type=str,
    default=None,
    help="Garment version (for garment tasks), e.g. Release or Holdout.",
)
parser.add_argument(
    "--input_file", type=str, default="./datasets/dataset.hdf5", help="File name of the dataset to be annotated."
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/dataset_annotated.hdf5",
    help="File name of the annotated output dataset file.",
)
parser.add_argument(
    "--task_type",
    type=str,
    default=None,
    help=(
        "Specify task type. If your dataset is recorded with keyboard, you should set it to 'keyboard', otherwise not"
        " to set it and keep default value None."
    ),
)
parser.add_argument("--auto", action="store_true", default=False, help="Automatically annotate subtasks.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--garment_info_json",
    type=str,
    default=None,
    help="Path to garment_info.json for per-episode initial cloth pose replay.",
)
parser.add_argument(
    "--step_hz",
    type=int,
    default=30,
    help="Replay speed in Hz for annotation (set <=0 to disable throttling).",
)
parser.add_argument(
    "--ignore_replay_success",
    action="store_true",
    default=False,
    help=(
        "Allow annotation/export even when replayed episode does not satisfy success term. "
        "Useful for stochastic cloth scenes where deterministic replay can fail."
    ),
)
parser.add_argument(
    "--sanitize_datagen_poses",
    action="store_true",
    default=False,
    help=(
        "Sanitize datagen pose rotations to valid SO(3) before export. "
        "Disabled by default to expose upstream pose issues."
    ),
)
parser.add_argument(
    "--garment_cfg_base_path",
    type=str,
    default="Assets/objects/Challenge_Garment",
    help="Base path of garment assets (for garment tasks).",
)
parser.add_argument(
    "--particle_cfg_path",
    type=str,
    default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
    help="Path to particle garment config yaml (for garment tasks).",
)
parser.add_argument(
    "--require_ik_actions",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Require top-level demo/actions to be IK actions before annotation (LeIsaac-style strict pipeline). "
        "Disable with --no-require-ik-actions to allow joint-action replay."
    ),
)
parser.add_argument(
    "--ik_action_dim",
    type=int,
    default=16,
    help="Expected IK action dimension (16 for bimanual cloth, 8 for single-arm).",
)
parser.add_argument(
    "--ik_quat_order",
    type=str,
    choices=["xyzw", "wxyz"],
    default="xyzw",
    help="Quaternion order used inside IK actions.",
)
parser.add_argument(
    "--legacy_joint_replay",
    action="store_true",
    default=False,
    help=(
        "Allow legacy fallback replay from joint action streams when strict IK action replay cannot be used. "
        "Disabled by default."
    ),
)
parser.add_argument(
    "--strict_pose_z_gap_threshold",
    type=float,
    default=0.55,
    help=(
        "Maximum allowed mean z-gap between target_eef_pose and object_pose in recorded datagen_info. "
        "Used only when --require-ik-actions is enabled."
    ),
)
parser.add_argument(
    "--ik_action_frame",
    type=str,
    choices=["auto", "base", "world"],
    default="auto",
    help=(
        "Frame used by IK top-level actions. "
        "'base' means action poses are in each arm base frame and will be transformed to world for replay. "
        "'world' means action poses are already in world frame. "
        "'auto' tries dataset metadata and then a numeric heuristic."
    ),
)
parser.add_argument(
    "--ik_auto_base_z_threshold",
    type=float,
    default=0.35,
    help=(
        "Auto frame heuristic threshold for IK actions: if mean z in IK action poses is below this value, "
        "actions are treated as base-frame."
    ),
)
parser.add_argument(
    "--mimic_ik_orientation_weight",
    type=float,
    default=0.01,
    help=(
        "Orientation weight forwarded to env IK conversion (target_eef_pose_to_action). "
        "Higher values enforce source wrist orientation more strongly; 0 disables orientation tracking."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import math
from copy import deepcopy

import gymnasium as gym
import torch
import numpy as np
try:
    import h5py
except ImportError:
    h5py = None

with contextlib.suppress(Exception):
    import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    with contextlib.suppress(Exception):
        import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

# Only enables inputs if this script is NOT headless mode
if not args_cli.headless and not os.environ.get("HEADLESS", 0):
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as PoseUtils
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import lehome.tasks  # noqa: F401

from lehome.utils.env_utils import (
    get_task_type,
    dynamic_reset_gripper_effort_limit_sim,
)
from lehome.assets.robots.lerobot import SO101_FOLLOWER_REST_POSE_RANGE
from lehome.tasks.fold_cloth.mdp.terminations import is_so101_at_rest_pose

is_paused = False
current_action_index = 0
marked_subtask_action_indices = []
skip_episode = False
task_type = None
expected_subtask_mark_count = None
last_marked_action_index = -10**9
last_mark_wall_time = 0.0
MARK_DEBOUNCE_SEC = 0.2
MARK_MIN_ACTION_GAP = 3
active_mark_eef_name = None
active_mark_signal_names = []
source_actions_frame_hint = None


def _pos_to_4x4(pos: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3) position tensors into (..., 4, 4) transforms."""
    batch_shape = pos.shape[:-1]
    pose = torch.eye(4, device=pos.device, dtype=pos.dtype).expand(*batch_shape, 4, 4).clone()
    pose[..., :3, 3] = pos
    return pose


def _semantic_keypoints_from_positions(kp_positions: np.ndarray) -> dict[str, np.ndarray]:
    """Map garment check points to semantic virtual object positions."""
    left_top = kp_positions[0]
    left_bottom = kp_positions[1]
    left_sleeve = kp_positions[2]
    right_sleeve = kp_positions[3]
    right_top = kp_positions[4]
    right_bottom = kp_positions[5]
    return {
        "garment_left_sleeve": left_sleeve,
        "garment_right_sleeve": right_sleeve,
        "garment_left_bottom": left_bottom,
        "garment_right_bottom": right_bottom,
        "garment_left_top": left_top,
        "garment_right_top": right_top,
        "garment_top_center": np.mean(np.stack([left_top, right_top], axis=0), axis=0),
        "garment_bottom_center": np.mean(np.stack([left_bottom, right_bottom], axis=0), axis=0),
        "garment_kp_left": np.mean(kp_positions[:3], axis=0),
        "garment_kp_right": np.mean(kp_positions[3:], axis=0),
        "garment_center": np.mean(kp_positions, axis=0),
    }


def _orthonormalize_rotations(pose: torch.Tensor) -> torch.Tensor:
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


def _sanitize_pose_dict(pose_dict: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
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
            out[key] = _orthonormalize_rotations(pose)
        else:
            out[key] = value
    return out


def _get_cloth_keypoint_object_poses_world(env: ManagerBasedRLMimicEnv) -> dict[str, torch.Tensor] | None:
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

    kp_positions = mesh_points[check_points]  # (6, 3), world frame
    semantic_points = _semantic_keypoints_from_positions(kp_positions)
    object_poses = {}
    num_envs = int(getattr(env, "num_envs", 1))
    for name, point in semantic_points.items():
        pos = torch.tensor(point, dtype=torch.float32, device=env.device).unsqueeze(0).expand(num_envs, -1)
        object_poses[name] = _pos_to_4x4(pos)
    return object_poses


def _get_robot_eef_pose_world(env: ManagerBasedRLMimicEnv, eef_name: str) -> torch.Tensor | None:
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


def _normalize_manual_subtask_signal_name(signal_name: str | None, eef_name: str, subtask_index: int) -> str:
    """Return a display/store-safe signal name for manual annotation."""
    if isinstance(signal_name, str) and signal_name.strip():
        return signal_name
    return f"{eef_name}_subtask_{subtask_index}_complete"


def stabilize_garment_after_reset_for_annotation(env: ManagerBasedRLMimicEnv, num_steps: int = 20):
    """Run a short physics-only settle phase without adding recorder entries."""
    if num_steps <= 0:
        return

    try:
        left_joint_pos = env.scene["left_arm"].data.joint_pos
        right_joint_pos = env.scene["right_arm"].data.joint_pos
        hold_action = torch.cat([left_joint_pos, right_joint_pos], dim=-1).to(env.device)
    except Exception:
        return

    try:
        env.action_manager.process_action(hold_action)
    except Exception:
        # If action manager cannot process, continue with pure sim stepping.
        pass

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    render_interval = max(1, int(getattr(env.cfg.sim, "render_interval", 1)))
    for step_index in range(num_steps):
        try:
            env.action_manager.apply_action()
        except Exception:
            pass

        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if is_rendering and (step_index + 1) % render_interval == 0:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)


def _rebuild_initial_state_from_episode(episode: EpisodeData) -> dict:
    """Build a robust initial_state, filling arm joints from obs if needed."""
    initial_state = deepcopy(episode.data.get("initial_state", {}))

    if "articulation" not in initial_state:
        initial_state["articulation"] = {}

    obs = episode.data.get("obs", {})
    actions = episode.data.get("actions", None)
    left_obs = obs.get("left_joint_pos", None)
    right_obs = obs.get("right_joint_pos", None)
    left_action_fallback = None
    right_action_fallback = None
    if actions is not None and getattr(actions, "ndim", 0) >= 2 and actions.shape[0] > 0:
        if actions.shape[-1] >= 6:
            left_action_fallback = actions[0:1, :6].clone()
        if actions.shape[-1] >= 12:
            right_action_fallback = actions[0:1, 6:12].clone()

    def _select_initial_joint_row(obs_tensor, fallback_tensor):
        if obs_tensor is None or getattr(obs_tensor, "ndim", 0) < 2 or obs_tensor.shape[0] == 0:
            return fallback_tensor
        norms = torch.sum(torch.abs(obs_tensor), dim=-1)
        nz = torch.nonzero(norms > 1e-6, as_tuple=False)
        if nz.numel() > 0:
            idx = int(nz[0, 0].item())
            return obs_tensor[idx:idx + 1].clone()
        return obs_tensor[0:1].clone() if fallback_tensor is None else fallback_tensor

    def _ensure_arm_state(arm_name: str, obs_tensor, fallback_tensor):
        obs_first = _select_initial_joint_row(obs_tensor, fallback_tensor)
        if obs_first is None:
            return

        arm_state = initial_state["articulation"].get(arm_name, {})
        if not isinstance(arm_state, dict):
            arm_state = {}

        jp = arm_state.get("joint_position", None)
        needs_fill = (
            jp is None
            or (hasattr(jp, "numel") and jp.numel() == 0)
            or (hasattr(jp, "shape") and jp.shape[-1] != obs_first.shape[-1])
            or (hasattr(jp, "abs") and float(jp.abs().sum().item()) < 1e-6)
        )
        if needs_fill:
            arm_state["joint_position"] = obs_first

        if "joint_velocity" not in arm_state or arm_state["joint_velocity"] is None:
            arm_state["joint_velocity"] = torch.zeros_like(obs_first)
        if "root_pose" not in arm_state or arm_state["root_pose"] is None:
            root_pose = torch.zeros((1, 7), dtype=obs_first.dtype, device=obs_first.device)
            root_pose[:, 3] = 1.0
            arm_state["root_pose"] = root_pose
        if "root_velocity" not in arm_state or arm_state["root_velocity"] is None:
            arm_state["root_velocity"] = torch.zeros((1, 6), dtype=obs_first.dtype, device=obs_first.device)

        initial_state["articulation"][arm_name] = arm_state

    _ensure_arm_state("left_arm", left_obs, left_action_fallback)
    _ensure_arm_state("right_arm", right_obs, right_action_fallback)
    return initial_state


def _set_arm_joint_state_from_initial_state(env: ManagerBasedRLMimicEnv, initial_state: dict):
    """Apply arm joint states from initial_state directly to sim."""
    articulation_state = initial_state.get("articulation", {})
    for arm_name in ("left_arm", "right_arm"):
        try:
            arm = env.scene[arm_name]
        except Exception:
            continue
        arm_state = articulation_state.get(arm_name, {})
        if not isinstance(arm_state, dict):
            continue

        joint_pos = arm_state.get("joint_position", None)
        if joint_pos is not None and getattr(joint_pos, "ndim", 0) == 2 and joint_pos.shape[0] > 0:
            arm.write_joint_position_to_sim(joint_pos[:1], env_ids=None)

        joint_vel = arm_state.get("joint_velocity", None)
        if joint_vel is not None and getattr(joint_vel, "ndim", 0) == 2 and joint_vel.shape[0] > 0:
            arm.write_joint_velocity_to_sim(joint_vel[:1], env_ids=None)


def _as_2d_tensor(data) -> torch.Tensor | None:
    """Convert array-like input to a 2D tensor, or return None if unavailable."""
    if data is None:
        return None
    tensor = data if torch.is_tensor(data) else torch.tensor(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        return None
    return tensor


def _get_replay_actions_for_episode(episode: EpisodeData, expected_action_dim: int | None = None) -> torch.Tensor:
    """Choose the best action trajectory for replay, matching env action dimension when possible."""
    obs = episode.data.get("obs", {})
    actions = _as_2d_tensor(episode.data.get("actions", None))
    processed_actions = _as_2d_tensor(episode.data.get("processed_actions", None))
    obs_actions = _as_2d_tensor(obs.get("actions", None))

    left_obs = _as_2d_tensor(obs.get("left_joint_pos", None))
    right_obs = _as_2d_tensor(obs.get("right_joint_pos", None))
    concat_joint_actions = None
    if left_obs is not None and right_obs is not None:
        if (
            left_obs.shape[0] == right_obs.shape[0]
            and left_obs.shape[1] >= 6
            and right_obs.shape[1] >= 6
        ):
            concat_joint_actions = torch.cat([left_obs[:, :6], right_obs[:, :6]], dim=-1)

    if actions is None:
        raise ValueError("Episode does not contain 'actions' for replay.")

    # Prefer source actions when dimensions already match the environment.
    if expected_action_dim is None or actions.shape[1] == expected_action_dim:
        replay_actions = actions
        source_name = "actions"
    else:
        # Priority for mismatched source actions:
        # 1) obs/actions (same horizon, joint-space in converted datasets)
        # 2) processed_actions
        # 3) left/right_joint_pos concat
        replay_actions = None
        source_name = None
        for name, candidate in (
            ("obs/actions", obs_actions),
            ("processed_actions", processed_actions),
            ("obs/left+right_joint_pos", concat_joint_actions),
        ):
            if candidate is not None and candidate.shape[1] == expected_action_dim:
                replay_actions = candidate
                source_name = name
                break

        if replay_actions is None:
            available_shapes = []
            for name, candidate in (
                ("actions", actions),
                ("obs/actions", obs_actions),
                ("processed_actions", processed_actions),
                ("obs/left+right_joint_pos", concat_joint_actions),
            ):
                if candidate is not None:
                    available_shapes.append(f"{name}={tuple(candidate.shape)}")
            raise ValueError(
                f"Could not find replay actions with expected dimension {expected_action_dim}. "
                f"Available: {', '.join(available_shapes)}"
            )

        print(
            f"\tReplay fallback: using {source_name} with shape {tuple(replay_actions.shape)} "
            f"instead of actions with shape {tuple(actions.shape)}."
        )

    actions_std = float(actions.float().std().item()) if actions.numel() > 0 else 0.0
    replay_std = float(replay_actions.float().std().item()) if replay_actions.numel() > 0 else 0.0
    if actions_std < 1e-3 and replay_std > 1e-2 and source_name != "actions":
        print(
            f"\tReplay note: source actions are near-constant (std={actions_std:.6f}), "
            f"selected replay stream std={replay_std:.6f}."
        )

    return replay_actions


def _quat_xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternion tensor from (x, y, z, w) to (w, x, y, z)."""
    return torch.stack(
        [quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]],
        dim=-1,
    )


def _ik_pose_vector_to_4x4(
    pose_vec: torch.Tensor,
    quat_order: str,
    device: torch.device,
) -> torch.Tensor:
    """Convert [x,y,z,q*,q*,q*,q*] vector to 4x4 homogeneous transform."""
    if pose_vec.numel() != 7:
        raise ValueError(f"Expected 7D pose vector, got {pose_vec.numel()} values.")

    pos = pose_vec[:3].to(device=device, dtype=torch.float32)
    quat = pose_vec[3:7].to(device=device, dtype=torch.float32)
    if quat_order == "xyzw":
        quat = _quat_xyzw_to_wxyz(quat)
    elif quat_order != "wxyz":
        raise ValueError(f"Unsupported quaternion order: {quat_order}")

    rot = PoseUtils.matrix_from_quat(quat.unsqueeze(0))[0]
    pose = torch.eye(4, device=device, dtype=torch.float32)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose


def _ik_action_row_to_target_and_gripper(
    ik_action_row: torch.Tensor,
    eef_names: list[str],
    quat_order: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Decode one IK action row into target_eef_pose_dict and gripper_action_dict."""
    if not eef_names:
        raise ValueError("No end-effectors configured for IK replay.")
    if len(eef_names) > 2:
        raise ValueError(f"IK replay supports at most 2 end-effectors, got {len(eef_names)}.")

    ik_action_row = ik_action_row.to(device=device, dtype=torch.float32).reshape(-1)
    action_dim = int(ik_action_row.numel())
    expected_dim = 8 * len(eef_names)
    if action_dim != expected_dim:
        raise ValueError(
            f"IK action dimension mismatch: expected {expected_dim} for eefs={eef_names}, got {action_dim}."
        )

    target_eef_pose_dict: dict[str, torch.Tensor] = {}
    gripper_action_dict: dict[str, torch.Tensor] = {}
    for i, eef_name in enumerate(eef_names):
        start = i * 8
        pose = _ik_pose_vector_to_4x4(ik_action_row[start : start + 7], quat_order=quat_order, device=device)
        grip = ik_action_row[start + 7].view(1)
        target_eef_pose_dict[eef_name] = pose.unsqueeze(0)
        gripper_action_dict[eef_name] = grip

    return target_eef_pose_dict, gripper_action_dict


def _first_pose_translation_mean(pose_dict: dict[str, torch.Tensor], max_steps: int = 128) -> torch.Tensor | None:
    """Compute mean xyz over first timesteps from a dict of [T,4,4] tensors."""
    if not isinstance(pose_dict, dict) or len(pose_dict) == 0:
        return None
    xyz_list = []
    for pose in pose_dict.values():
        pose_t = torch.as_tensor(pose)
        if pose_t.ndim != 3 or tuple(pose_t.shape[-2:]) != (4, 4):
            continue
        n = min(int(max_steps), int(pose_t.shape[0]))
        xyz_list.append(pose_t[:n, :3, 3].reshape(-1, 3))
    if not xyz_list:
        return None
    return torch.cat(xyz_list, dim=0).mean(dim=0)


def _validate_recorded_datagen_pose_contract(
    annotated_episode: EpisodeData,
    z_gap_threshold: float,
) -> None:
    """Fail fast if recorder-exported datagen_info violates strict frame sanity checks."""
    obs = annotated_episode.data.get("obs", {})
    datagen = obs.get("datagen_info", {})
    if not isinstance(datagen, dict):
        raise ValueError("Annotated episode missing obs/datagen_info.")

    required = ("object_pose", "eef_pose", "target_eef_pose")
    for key in required:
        if key not in datagen:
            raise ValueError(f"Annotated episode missing obs/datagen_info/{key}.")

    for section_name in required:
        pose_dict = datagen[section_name]
        if not isinstance(pose_dict, dict) or len(pose_dict) == 0:
            raise ValueError(f"obs/datagen_info/{section_name} is empty or invalid.")
        for pose_name, pose_value in pose_dict.items():
            pose = torch.as_tensor(pose_value)
            if pose.ndim != 3 or tuple(pose.shape[-2:]) != (4, 4):
                raise ValueError(
                    f"Invalid pose shape for {section_name}/{pose_name}: expected [T,4,4], got {tuple(pose.shape)}."
                )
            # Homogeneous transform sanity check.
            last_row = pose[..., 3, :]
            target_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=last_row.device, dtype=last_row.dtype)
            max_err = float((last_row - target_row).abs().max().item())
            if max_err > 1e-3:
                raise ValueError(
                    f"Invalid homogeneous row in {section_name}/{pose_name}: max error={max_err:.6f}."
                )

    obj_center = _first_pose_translation_mean(datagen["object_pose"])
    tgt_center = _first_pose_translation_mean(datagen["target_eef_pose"])
    if obj_center is None or tgt_center is None:
        raise ValueError("Failed to compute strict datagen pose centers.")
    z_gap = float(abs(tgt_center[2] - obj_center[2]).item())
    if z_gap > float(z_gap_threshold):
        raise ValueError(
            "Strict datagen pose frame check failed: "
            f"mean |target_z - object_z|={z_gap:.4f} exceeds threshold {float(z_gap_threshold):.4f}."
        )


def _load_garment_info(garment_info_path: str | None) -> dict | None:
    """Load garment_info.json for initial pose replay."""
    if garment_info_path is None:
        return None
    path = Path(garment_info_path)
    if not path.exists():
        print(f"Warning: garment_info_json not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"Warning: failed to parse garment_info_json ({path}): {e}")
    return None


def _load_garment_info_from_hdf5(input_file: str) -> dict | None:
    """Load merged garment_info from /data/demo_*/meta or initial_state/garment."""
    if h5py is None:
        return None

    def _normalize_scalar(value):
        if hasattr(value, "shape") and getattr(value, "shape", ()) == ():
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value

    def _merge(dst: dict, src: dict):
        for garment_name, episodes in src.items():
            if not isinstance(episodes, dict):
                continue
            dst.setdefault(garment_name, {})
            for episode_idx, payload in episodes.items():
                dst[garment_name][str(episode_idx)] = payload

    def _demo_index_from_name(demo_name: str) -> str | None:
        if not demo_name.startswith("demo_"):
            return None
        suffix = demo_name.split("_", maxsplit=1)[1]
        return suffix if suffix.isdigit() else None

    merged: dict = {}
    try:
        with h5py.File(input_file, "r") as file:
            data_group = file.get("data", None)
            if data_group is None:
                return None

            for demo_name in sorted(data_group.keys()):
                if not demo_name.startswith("demo_"):
                    continue
                demo_group = data_group[demo_name]
                if "meta" in demo_group:
                    meta_group = demo_group["meta"]

                    for key in ("garment_info", "garment_info.json"):
                        if key not in meta_group:
                            continue
                        raw = _normalize_scalar(meta_group[key][()])
                        if isinstance(raw, str):
                            try:
                                parsed = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                        elif isinstance(raw, dict):
                            parsed = raw
                        else:
                            continue

                        if isinstance(parsed, dict):
                            _merge(merged, parsed)

                initial_state_group = demo_group.get("initial_state")
                garment_group = None if initial_state_group is None else initial_state_group.get("garment")
                demo_index = _demo_index_from_name(demo_name)
                if garment_group is None or demo_index is None:
                    continue

                for garment_name in garment_group.keys():
                    garment_entry = garment_group[garment_name]
                    if "initial_pose" not in garment_entry:
                        continue

                    pose = _normalize_scalar(garment_entry["initial_pose"][()])
                    if hasattr(pose, "tolist"):
                        pose = pose.tolist()
                    if isinstance(pose, list) and len(pose) == 1 and isinstance(pose[0], list):
                        pose = pose[0]
                    if not isinstance(pose, list):
                        continue

                    payload = {"object_initial_pose": pose}
                    if "scale" in garment_entry:
                        scale = _normalize_scalar(garment_entry["scale"][()])
                        if hasattr(scale, "tolist"):
                            scale = scale.tolist()
                        if isinstance(scale, list) and len(scale) == 1 and isinstance(scale[0], list):
                            scale = scale[0]
                        payload["scale"] = scale

                    merged.setdefault(str(garment_name), {})[demo_index] = payload
    except Exception as e:
        print(f"Warning: failed to read garment info from HDF5 metadata: {e}")
        return None

    return merged if merged else None


def _get_episode_initial_pose(garment_info: dict | None, episode_index: int) -> dict | None:
    """Extract {"Garment": [...]} pose for the given episode index."""
    if not garment_info:
        return None
    episode_key = str(episode_index)
    for _, episodes in garment_info.items():
        if isinstance(episodes, dict) and episode_key in episodes:
            pose = episodes[episode_key].get("object_initial_pose")
            if pose is not None:
                return {"Garment": pose}
    return None


def _load_actions_frame_from_hdf5(input_file: str) -> str | None:
    """Read optional /data attrs['actions_frame'] hint."""
    if h5py is None:
        return None
    try:
        with h5py.File(input_file, "r") as file:
            data_group = file.get("data", None)
            if data_group is None:
                return None
            raw = data_group.attrs.get("actions_frame", None)
            if raw is None:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            value = str(raw).strip().lower()
            if value in {"base", "world"}:
                return value
    except Exception:
        return None
    return None


def _load_ik_quat_order_from_hdf5(input_file: str) -> str | None:
    """Read optional /data attrs['ik_quat_order'] hint."""
    if h5py is None:
        return None
    try:
        with h5py.File(input_file, "r") as file:
            data_group = file.get("data", None)
            if data_group is None:
                return None
            raw = data_group.attrs.get("ik_quat_order", None)
            if raw is None:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            value = str(raw).strip().lower()
            if value in {"xyzw", "wxyz"}:
                return value
    except Exception:
        return None
    return None


def _infer_ik_action_frame(
    actions: torch.Tensor,
    eef_names: list[str],
    frame_hint: str | None,
    base_z_threshold: float,
) -> str:
    """Infer IK action frame from metadata hint or pose statistics."""
    if frame_hint in {"base", "world"}:
        return frame_hint

    if actions.ndim != 2 or actions.shape[0] == 0:
        return "world"

    sample = actions[: min(32, int(actions.shape[0]))]
    sample_dim = int(sample.shape[1])
    if len(eef_names) >= 2 and sample_dim >= 16:
        z_values = torch.cat([sample[:, 2], sample[:, 10]], dim=0)
    elif sample_dim >= 8:
        z_values = sample[:, 2]
    else:
        return "world"

    mean_z = float(z_values.float().mean().item())
    return "base" if mean_z < float(base_z_threshold) else "world"


def _arm_root_pose_world(env: ManagerBasedRLMimicEnv, arm_name: str) -> torch.Tensor:
    """Get arm base pose in world as 4x4 transform for env 0."""
    arm = env.scene[arm_name]
    pos_w = arm.data.root_pos_w[0:1]
    quat_w = arm.data.root_quat_w[0:1]
    return PoseUtils.make_pose(pos_w, PoseUtils.matrix_from_quat(quat_w))[0]


def _pose_base_to_world(
    env: ManagerBasedRLMimicEnv,
    arm_name: str,
    pose_in_base: torch.Tensor,
) -> torch.Tensor:
    """Transform a 4x4 pose from arm base frame to world frame."""
    base_in_world = _arm_root_pose_world(env, arm_name)
    return base_in_world @ pose_in_base


def _get_first_garment_name(garment_info: dict | None) -> str | None:
    """Return first garment key from garment_info dict."""
    if not garment_info or not isinstance(garment_info, dict):
        return None
    for garment_name, episodes in garment_info.items():
        if isinstance(garment_name, str) and garment_name and isinstance(episodes, dict):
            return garment_name
    return None


def _get_source_episode_index_from_demo_name(input_file: str, episode_name: str, fallback: int) -> int:
    """Read source_episode_index from /data/demo_*/ attrs, fallback to sequential index."""
    if h5py is None:
        return fallback
    try:
        with h5py.File(input_file, "r") as file:
            data_group = file.get("data", None)
            if data_group is None or episode_name not in data_group:
                return fallback
            demo_group = data_group[episode_name]
            if "source_episode_index" in demo_group.attrs:
                return int(demo_group.attrs["source_episode_index"])
    except Exception:
        return fallback
    return fallback


def _load_episode_with_numeric_datasets_only(input_file: str, episode_name: str, device: str) -> EpisodeData:
    """Load an episode while skipping non-numeric datasets (e.g., demo/meta string blobs)."""
    if h5py is None:
        raise RuntimeError("h5py is required for fallback episode loading.")

    def _is_supported_numeric_array(arr: np.ndarray) -> bool:
        # torch.tensor cannot ingest object/string arrays directly.
        return arr.dtype.kind in {"b", "i", "u", "f", "c"}

    def _load_group(group) -> dict:
        data = {}
        for key in group.keys():
            # Explicitly skip metadata payloads that are strings/json.
            if key == "meta":
                continue

            node = group[key]
            if isinstance(node, h5py.Group):
                nested = _load_group(node)
                if nested:
                    data[key] = nested
                continue

            array = np.array(node)
            if not _is_supported_numeric_array(array):
                continue
            data[key] = torch.tensor(array, device=device)
        return data

    with h5py.File(input_file, "r") as file:
        data_group = file.get("data", None)
        if data_group is None or episode_name not in data_group:
            raise KeyError(f"Episode '{episode_name}' not found in {input_file}")

        h5_episode_group = data_group[episode_name]
        episode = EpisodeData()
        episode.data = _load_group(h5_episode_group)

        if "seed" in h5_episode_group.attrs:
            try:
                episode.seed = int(h5_episode_group.attrs["seed"])
            except Exception:
                episode.seed = h5_episode_group.attrs["seed"]
        if "success" in h5_episode_group.attrs:
            episode.success = bool(h5_episode_group.attrs["success"])

    return episode


def _load_episode_compat(
    dataset_file_handler: HDF5DatasetFileHandler,
    input_file: str,
    episode_name: str,
    device: str,
) -> EpisodeData:
    """Load episode robustly across mixed numeric/string HDF5 demo schemas."""
    try:
        episode = dataset_file_handler.load_episode(episode_name, device)
        if episode is None:
            raise ValueError(f"Episode '{episode_name}' not found.")
        return episode
    except TypeError as e:
        if "numpy.object_" not in str(e):
            raise
        print(
            f"\tInfo: default episode loader failed on non-numeric datasets ({e}). "
            "Using numeric-only fallback loader."
        )
        return _load_episode_with_numeric_datasets_only(input_file, episode_name, device)


def _episode_has_path(episode: EpisodeData, key_path: str) -> bool:
    """Check whether a slash-delimited key path exists in episode.data."""
    node = episode.data
    for token in key_path.split("/"):
        if not isinstance(node, dict) or token not in node:
            return False
        node = node[token]
    return True


def _merge_source_obs_into_annotated_episode(source_episode: EpisodeData, annotated_episode: EpisodeData):
    """Preserve non-datagen observation keys from source episode in annotated export."""
    source_obs = source_episode.data.get("obs", None)
    if not isinstance(source_obs, dict):
        return

    def _merge_recursive(node: dict | torch.Tensor, rel_path: str):
        if isinstance(node, dict):
            for key, value in node.items():
                # Keep recorder-generated datagen_info from the replayed episode.
                if rel_path == "" and key == "datagen_info":
                    continue
                next_rel_path = f"{rel_path}/{key}" if rel_path else key
                _merge_recursive(value, next_rel_path)
            return

        full_path = f"obs/{rel_path}"
        if _episode_has_path(annotated_episode, full_path):
            return

        if isinstance(node, torch.Tensor):
            annotated_episode.add(full_path, node.clone())
            return
        if isinstance(node, np.ndarray):
            annotated_episode.add(full_path, torch.from_numpy(node.copy()))
            return
        # Fallback for scalar numeric entries.
        if isinstance(node, (int, float, bool)):
            annotated_episode.add(full_path, torch.tensor(node))

    _merge_recursive(source_obs, "")


def _overwrite_annotated_actions_with_source_actions(
    source_episode: EpisodeData,
    annotated_episode: EpisodeData,
) -> None:
    """Keep top-level actions from source episode (strict IK pipeline contract)."""
    source_actions = _as_2d_tensor(source_episode.data.get("actions", None))
    if source_actions is None:
        raise ValueError("Source episode does not contain top-level actions.")

    recorded_actions = _as_2d_tensor(annotated_episode.data.get("actions", None))
    if recorded_actions is not None and recorded_actions.shape[0] != source_actions.shape[0]:
        raise ValueError(
            "Cannot preserve source actions: horizon mismatch between source and recorder output "
            f"({source_actions.shape[0]} vs {recorded_actions.shape[0]})."
        )

    annotated_episode.data["actions"] = source_actions.clone()


def _resolve_task_type(task_id: str, explicit_task_type: str | None) -> str:
    """Resolve task_type robustly for LeHome task naming variants."""
    if explicit_task_type is not None:
        return explicit_task_type

    lowered = task_id.lower()
    if "biso101" in lowered or "biarm" in lowered or "bimanual" in lowered or lowered.startswith("lehome-bi"):
        return "bi-so101leader"
    return get_task_type(task_id, None)


def _print_success_debug_breakdown(env: ManagerBasedRLMimicEnv) -> None:
    """Print a breakdown of success components (fold + both arms at rest)."""
    try:
        fold_success = None
        if hasattr(env, "_get_success"):
            fold_success_tensor = env._get_success()
            fold_success = bool(fold_success_tensor[0].item())

        left_arm = env.scene["left_arm"]
        right_arm = env.scene["right_arm"]
        left_at_rest = bool(
            is_so101_at_rest_pose(left_arm.data.joint_pos, left_arm.data.joint_names)[0].item()
        )
        right_at_rest = bool(
            is_so101_at_rest_pose(right_arm.data.joint_pos, right_arm.data.joint_names)[0].item()
        )

        print(
            "\tSuccess breakdown:"
            f" fold_success={fold_success}, left_at_rest={left_at_rest}, right_at_rest={right_at_rest}"
        )

        def _report_joint_out_of_range(arm_name: str, arm):
            out_of_range = []
            joint_pos = arm.data.joint_pos[0].detach().cpu()
            for i, joint_name in enumerate(arm.data.joint_names):
                if joint_name not in SO101_FOLLOWER_REST_POSE_RANGE:
                    continue
                low_deg, high_deg = SO101_FOLLOWER_REST_POSE_RANGE[joint_name]
                low_rad = math.radians(low_deg)
                high_rad = math.radians(high_deg)
                value = float(joint_pos[i].item())
                if value < low_rad or value > high_rad:
                    out_of_range.append(
                        f"{joint_name}={math.degrees(value):.1f}deg (range {low_deg:.1f}..{high_deg:.1f})"
                    )
            if out_of_range:
                print(f"\t{arm_name} joints out of rest range: {', '.join(out_of_range)}")

        _report_joint_out_of_range("left_arm", left_arm)
        _report_joint_out_of_range("right_arm", right_arm)
    except Exception as e:
        print(f"\tWarning: failed to compute success breakdown: {e}")


def _are_arms_at_rest(env: ManagerBasedRLMimicEnv) -> bool:
    """Check whether available arm articulations are within rest pose ranges."""
    try:
        at_rest_flags = []
        for arm_name in ("left_arm", "right_arm"):
            try:
                arm = env.scene[arm_name]
            except Exception:
                continue
            arm_rest = is_so101_at_rest_pose(arm.data.joint_pos, arm.data.joint_names)
            at_rest_flags.append(bool(arm_rest[0].item()))
        if not at_rest_flags:
            return False
        return all(at_rest_flags)
    except Exception:
        return False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def skip_episode_cb():
    global skip_episode
    skip_episode = True


def mark_subtask_cb():
    global current_action_index, marked_subtask_action_indices
    global expected_subtask_mark_count, last_marked_action_index, last_mark_wall_time
    global active_mark_eef_name, active_mark_signal_names
    now = time.perf_counter()

    if expected_subtask_mark_count is not None and len(marked_subtask_action_indices) >= expected_subtask_mark_count:
        return
    if (now - last_mark_wall_time) < MARK_DEBOUNCE_SEC:
        return
    if (current_action_index - last_marked_action_index) < MARK_MIN_ACTION_GAP:
        return

    marked_signal_index = len(marked_subtask_action_indices)
    marked_subtask_action_indices.append(current_action_index)
    last_mark_wall_time = now
    last_marked_action_index = current_action_index
    signal_name = None
    if marked_signal_index < len(active_mark_signal_names):
        signal_name = active_mark_signal_names[marked_signal_index]
    if signal_name is not None:
        print(
            f'Marked subtask signal "{signal_name}"'
            f' for eef "{active_mark_eef_name}" at action index: {current_action_index}'
        )
    else:
        print(f"Marked a subtask signal at action index: {current_action_index}")


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records the datagen info data in each step."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            try:
                # Keep recorder behavior aligned with generation/runtime env API.
                eef_pose = self._env.get_robot_eef_pose(eef_name=eef_name)
            except Exception:
                eef_pose = _get_robot_eef_pose_world(self._env, eef_name)
            eef_pose_dict[eef_name] = eef_pose

        try:
            # Prefer env API output so annotation and generation consume identical object frames.
            object_pose = self._env.get_object_poses()
        except Exception:
            object_pose = None
        if not isinstance(object_pose, dict) or len(object_pose) == 0:
            object_pose = _get_cloth_keypoint_object_poses_world(self._env)

        sanitize_poses = bool(getattr(args_cli, "sanitize_datagen_poses", False))
        if sanitize_poses:
            object_pose = _sanitize_pose_dict(object_pose)
            eef_pose_dict = _sanitize_pose_dict(eef_pose_dict)
            target_eef_pose = _sanitize_pose_dict(
                self._env.action_to_target_eef_pose(self._env.action_manager.action)
            )
        else:
            target_eef_pose = self._env.action_to_target_eef_pose(self._env.action_manager.action)

        datagen_info = {
            "object_pose": object_pose,
            "eef_pose": eef_pose_dict,
            "target_eef_pose": target_eef_pose,
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for the datagen info recorder term."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Mimic specific recorder terms."""

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


def main():
    """Add Isaac Lab Mimic annotations to the given demo dataset file."""
    global is_paused, current_action_index, marked_subtask_action_indices, task_type, source_actions_frame_hint

    # Load input dataset to be annotated
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    dataset_env_args = {}
    try:
        raw_env_args = dataset_file_handler._hdf5_data_group.attrs.get("env_args", "{}")
        if isinstance(raw_env_args, bytes):
            raw_env_args = raw_env_args.decode("utf-8")
        dataset_env_args = json.loads(raw_env_args) if isinstance(raw_env_args, str) else {}
    except Exception as e:
        print(f"Warning: failed to parse dataset env_args: {e}")

    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    garment_info = _load_garment_info_from_hdf5(args_cli.input_file)
    if garment_info is not None:
        print("Using garment initial poses from HDF5 demo data (/data/demo_*/meta or initial_state/garment).")
    else:
        garment_info_path = args_cli.garment_info_json
        if garment_info_path is None:
            candidate = Path(args_cli.input_file).parent / "meta" / "garment_info.json"
            if candidate.exists():
                garment_info_path = str(candidate)
        garment_info = _load_garment_info(garment_info_path)
        if garment_info is not None:
            print(f"Using garment initial poses from: {garment_info_path}")

    source_actions_frame_hint = _load_actions_frame_from_hdf5(args_cli.input_file)
    if source_actions_frame_hint is not None:
        print(f"Source actions frame hint from dataset: {source_actions_frame_hint}")

    source_ik_quat_order_hint = _load_ik_quat_order_from_hdf5(args_cli.input_file)
    explicit_ik_quat_order = any(
        arg == "--ik_quat_order" or arg.startswith("--ik_quat_order=") for arg in sys.argv
    )
    if source_ik_quat_order_hint is not None and not explicit_ik_quat_order:
        args_cli.ik_quat_order = source_ik_quat_order_hint
        print(f"Source IK quaternion order from dataset: {source_ik_quat_order_hint}")
    elif source_ik_quat_order_hint is not None:
        print(
            "Ignoring dataset IK quaternion order "
            f"({source_ik_quat_order_hint}) because --ik_quat_order was set explicitly."
        )

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    # get output directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task_id = args_cli.task or env_name
    if task_id is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=1)
    task_type = _resolve_task_type(task_id, args_cli.task_type)
    setattr(env_cfg, "task_type", task_type)
    setattr(env_cfg, "mimic_ik_orientation_weight", float(args_cli.mimic_ik_orientation_weight))

    env_cfg.env_name = task_id
    print(f"Using mimic IK orientation_weight={float(args_cli.mimic_ik_orientation_weight):.4f}")

    # Keep garment task resource paths configurable from CLI.
    if hasattr(env_cfg, "garment_cfg_base_path"):
        env_cfg.garment_cfg_base_path = args_cli.garment_cfg_base_path
    if hasattr(env_cfg, "particle_cfg_path"):
        env_cfg.particle_cfg_path = args_cli.particle_cfg_path

    # Configure garment metadata for tasks that require explicit garment loading.
    if hasattr(env_cfg, "garment_name"):
        resolved_garment_name = (
            args_cli.garment_name
            or dataset_env_args.get("garment_name")
            or _get_first_garment_name(garment_info)
            or getattr(env_cfg, "garment_name", None)
        )
        resolved_garment_version = (
            args_cli.garment_version
            or dataset_env_args.get("garment_version")
            or getattr(env_cfg, "garment_version", None)
        )

        if resolved_garment_name is None or (
            isinstance(resolved_garment_name, str) and not resolved_garment_name.strip()
        ):
            raise ValueError(
                "This task requires a garment_name, but none was provided. "
                "Pass --garment_name (e.g., Top_Long_Unseen_0) or include garment_name in data/env_args."
            )

        env_cfg.garment_name = (
            resolved_garment_name.strip()
            if isinstance(resolved_garment_name, str)
            else resolved_garment_name
        )
        if hasattr(env_cfg, "garment_version") and resolved_garment_version is not None:
            env_cfg.garment_version = resolved_garment_version

        print(
            f"Using garment config: name={env_cfg.garment_name}, "
            f"version={getattr(env_cfg, 'garment_version', 'N/A')}"
        )

    # extract success checking function to invoke manually
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    # Disable all termination terms
    env_cfg.terminations = None

    # Set up recorder terms for mimic annotations
    env_cfg.recorders: MimicRecorderManagerCfg = MimicRecorderManagerCfg()
    # ActionStateRecorder's flat policy observation term writes to key "obs" as a tensor.
    # Mimic annotations write nested keys under "obs/datagen_info/...".
    # Keeping both causes type conflicts in EpisodeData (list vs dict under "obs").
    if hasattr(env_cfg.recorders, "record_pre_step_flat_policy_observations"):
        env_cfg.recorders.record_pre_step_flat_policy_observations = None
    if not args_cli.auto:
        # disable subtask term signals recorder term if in manual mode
        env_cfg.recorders.record_pre_step_subtask_term_signals = None

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # create environment from loaded config
    env: ManagerBasedRLMimicEnv = gym.make(task_id, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if args_cli.auto:
        # check if the mimic API env.get_subtask_term_signals() is implemented
        if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
            raise NotImplementedError(
                "The environment does not implement the get_subtask_term_signals method required "
                "to run automatic annotations."
            )
    else:
        # get subtask termination signal names for each eef from the environment configs
        subtask_term_signal_names = {}
        for eef_name, eef_subtask_configs in env.cfg.subtask_configs.items():
            subtask_term_signal_names[eef_name] = [
                _normalize_manual_subtask_signal_name(
                    subtask_config.subtask_term_signal, eef_name, subtask_index
                )
                for subtask_index, subtask_config in enumerate(eef_subtask_configs)
            ]

    # reset environment
    env.reset()
    if hasattr(env, "initialize_obs"):
        try:
            env.initialize_obs()
            stabilize_garment_after_reset_for_annotation(env)
        except Exception as e:
            print(f"Warning: initial garment initialization/stabilization failed: {e}")

    # Only enables inputs if this script is NOT headless mode
    if not args_cli.headless and not os.environ.get("HEADLESS", 0):
        try:
            keyboard_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        except TypeError:
            # Backward compatibility for older keyboard API variants.
            keyboard_interface = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1)
        keyboard_interface.add_callback("N", play_cb)
        keyboard_interface.add_callback("B", pause_cb)
        keyboard_interface.add_callback("Q", skip_episode_cb)
        if not args_cli.auto:
            keyboard_interface.add_callback("S", mark_subtask_cb)
        keyboard_interface.reset()

    # simulate environment -- run everything in inference mode
    exported_episode_count = 0
    processed_episode_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            # Iterate over the episodes in the loaded dataset file
            for episode_index, episode_name in enumerate(dataset_file_handler.get_episode_names()):
                processed_episode_count += 1
                print(f"\nAnnotating episode #{episode_index} ({episode_name})")
                episode = _load_episode_compat(
                    dataset_file_handler, args_cli.input_file, episode_name, env.device
                )
                source_episode_index = _get_source_episode_index_from_demo_name(
                    args_cli.input_file, episode_name, episode_index
                )

                is_episode_annotated_successfully = False
                if args_cli.auto:
                    is_episode_annotated_successfully = annotate_episode_in_auto_mode(
                        env, episode, success_term, source_episode_index, garment_info
                    )
                else:
                    is_episode_annotated_successfully = annotate_episode_in_manual_mode(
                        env, episode, success_term, subtask_term_signal_names, source_episode_index, garment_info
                    )

                if is_episode_annotated_successfully and not skip_episode:
                    # set success to the recorded episode data and export to file
                    env.recorder_manager.set_success_to_episodes(
                        None, torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes()
                    exported_episode_count += 1
                    print("\tExported the annotated episode.")
                else:
                    print("\tSkipped exporting the episode due to incomplete subtask annotations.")
            break

    print(
        f"\nExported {exported_episode_count} (out of {processed_episode_count}) annotated"
        f" episode{'s' if exported_episode_count > 1 else ''}."
    )

    if h5py is not None and os.path.exists(args_cli.output_file):
        try:
            with h5py.File(args_cli.output_file, "r+") as out_file:
                data_group = out_file.get("data")
                if data_group is not None:
                    data_group.attrs["actions_mode"] = "ee_pose" if bool(args_cli.require_ik_actions) else "joint"
        except Exception as e:
            print(f"Warning: failed to write actions_mode attribute to annotated dataset: {e}")

    print("Exiting the app.")

    # Close environment after annotation is complete
    env.close()


def replay_episode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    episode_index: int | None = None,
    garment_info: dict | None = None,
) -> bool:
    """Replays an episode in the environment.

    This function replays the given recorded episode in the environment. It can optionally check if the task
    was successfully completed using a success termination condition input.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.

    Returns:
        True if the episode was successfully replayed and the success condition was met (if provided),
        False otherwise.
    """
    global current_action_index, skip_episode, is_paused, task_type, source_actions_frame_hint
    # read initial state and actions from the loaded episode
    initial_state = _rebuild_initial_state_from_episode(episode)
    source_actions = _as_2d_tensor(episode.data.get("actions", None))
    if source_actions is None:
        raise ValueError("Episode does not contain top-level actions for replay.")

    expected_action_dim = None
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "total_action_dim"):
        expected_action_dim = int(env.action_manager.total_action_dim)
    elif hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
        shape = getattr(env.action_space, "shape", ())
        if len(shape) > 0:
            expected_action_dim = int(shape[0])

    eef_names = list(getattr(env.cfg, "subtask_configs", {}).keys())
    if not eef_names:
        eef_names = ["left_arm", "right_arm"]
    if set(eef_names) == {"left_arm", "right_arm"}:
        eef_names = ["left_arm", "right_arm"]

    replay_mode = "joint"
    actions = source_actions
    if bool(args_cli.require_ik_actions):
        if int(source_actions.shape[1]) != int(args_cli.ik_action_dim):
            message = (
                "Strict IK replay is enabled, but source actions dimension does not match "
                f"--ik_action_dim ({source_actions.shape[1]} vs {args_cli.ik_action_dim})."
            )
            if not bool(args_cli.legacy_joint_replay):
                raise ValueError(message)
            print(f"\tWarning: {message} Falling back to legacy joint replay.")
            actions = _get_replay_actions_for_episode(episode, expected_action_dim=expected_action_dim)
            replay_mode = "joint"
        else:
            replay_mode = "ik"
    else:
        if int(source_actions.shape[1]) == int(args_cli.ik_action_dim):
            replay_mode = "ik"
        else:
            actions = _get_replay_actions_for_episode(episode, expected_action_dim=expected_action_dim)
            replay_mode = "joint"

    if replay_mode == "ik":
        ik_frame = str(args_cli.ik_action_frame).strip().lower()
        if ik_frame == "auto":
            ik_frame = _infer_ik_action_frame(
                actions,
                eef_names=eef_names,
                frame_hint=source_actions_frame_hint,
                base_z_threshold=float(args_cli.ik_auto_base_z_threshold),
            )
        if ik_frame not in {"base", "world"}:
            raise ValueError(f"Invalid IK action frame resolved: {ik_frame}")
        print(
            f"\tReplay mode: IK actions (dim={actions.shape[1]}, quat_order={args_cli.ik_quat_order}, frame={ik_frame})."
        )
        native_ik_action_contract = False
        if hasattr(env, "_is_native_mimic_ik_action_contract"):
            try:
                native_ik_action_contract = bool(env._is_native_mimic_ik_action_contract())
            except Exception:
                native_ik_action_contract = False
        if (not native_ik_action_contract) and expected_action_dim is not None:
            native_ik_action_contract = int(expected_action_dim) == int(args_cli.ik_action_dim)

        if native_ik_action_contract:
            print("\tIK replay will use native env IK action contract (no Pinocchio pose->joint conversion).")
        if (not native_ik_action_contract) and hasattr(env, "_init_ik_solver_if_needed"):
            try:
                if not bool(env._init_ik_solver_if_needed()):
                    raise RuntimeError("environment IK solver initialization returned False")
            except Exception as e:
                raise RuntimeError(
                    "Strict IK replay requires a working IK solver in the environment, "
                    f"but initialization failed: {e}"
                ) from e
    else:
        print(f"\tReplay mode: joint actions (dim={actions.shape[1]}).")

    env.seed(int(episode.seed))
    # Hard simulator resets can destabilize particle-cloth scenes.
    # For garment environments, prefer environment-level reset only.
    is_cloth_env = hasattr(env, "object")
    if not is_cloth_env:
        env.sim.reset()
        try:
            env.sim.play()
        except Exception:
            pass
    env.recorder_manager.reset()
    env.reset()
    try:
        env.sim.play()
    except Exception:
        pass
    if hasattr(env, "initialize_obs"):
        try:
            env.initialize_obs()
        except Exception as e:
            print(f"Warning: initialize_obs failed during replay reset: {e}")
    _set_arm_joint_state_from_initial_state(env, initial_state)
    if episode_index is not None and hasattr(env, "set_all_pose"):
        initial_pose = _get_episode_initial_pose(garment_info, episode_index)
        if initial_pose is not None:
            try:
                env.set_all_pose(initial_pose)
            except Exception as e:
                print(f"Warning: failed to set initial garment pose for episode {episode_index}: {e}")
    env.scene.write_data_to_sim()
    env.sim.forward()
    try:
        env.sim.play()
    except Exception:
        pass
    stabilize_garment_after_reset_for_annotation(env)
    first_action = True
    fold_success_seen = False
    step_period = (1.0 / args_cli.step_hz) if args_cli.step_hz and args_cli.step_hz > 0 else 0.0
    for action_index, action in enumerate(actions):
        step_start_time = time.perf_counter()
        current_action_index = action_index
        if first_action:
            first_action = False
        else:
            while is_paused or skip_episode:
                env.sim.render()
                if skip_episode:
                    return False
                continue
        try:
            if env.sim.is_stopped():
                env.sim.play()
        except Exception:
            pass
        if replay_mode == "ik":
            ik_action_row = torch.as_tensor(action, dtype=torch.float32, device=env.device).reshape(-1)
            target_eef_pose_dict, gripper_action_dict = _ik_action_row_to_target_and_gripper(
                ik_action_row,
                eef_names=eef_names,
                quat_order=str(args_cli.ik_quat_order),
                device=env.device,
            )
            if ik_frame == "base":
                for eef_name in list(target_eef_pose_dict.keys()):
                    pose_base = target_eef_pose_dict[eef_name][0]
                    pose_world = _pose_base_to_world(env, eef_name, pose_base)
                    target_eef_pose_dict[eef_name] = pose_world.unsqueeze(0)
            action_tensor = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_eef_pose_dict,
                gripper_action_dict=gripper_action_dict,
                action_noise_dict=None,
                env_id=0,
            )
            action_tensor = torch.as_tensor(action_tensor, dtype=torch.float32, device=env.device).reshape(1, -1)
            if expected_action_dim is not None and int(action_tensor.shape[1]) != int(expected_action_dim):
                raise ValueError(
                    "Environment action conversion produced wrong dimension: "
                    f"expected {expected_action_dim}, got {action_tensor.shape[1]}."
                )
        else:
            action_tensor = torch.as_tensor(action, dtype=torch.float32, device=env.device).reshape(1, -1)

        if getattr(env.cfg, "dynamic_reset_gripper_effort_limit", False):
            dynamic_reset_gripper_effort_limit_sim(env, task_type)
        env.step(action_tensor)
        # Match replay scripts: track fold success continuously over rollout.
        if hasattr(env, "_get_success"):
            try:
                fold_success_seen = fold_success_seen or bool(env._get_success()[0].item())
            except Exception:
                pass
        # Force viewport updates in GUI mode so replay is visually observable.
        if env.sim.has_gui() or env.sim.has_rtx_sensors():
            env.sim.render()
        if step_period > 0.0:
            elapsed = time.perf_counter() - step_start_time
            if elapsed < step_period:
                time.sleep(step_period - elapsed)
    if success_term is not None:
        # For fold-cloth garment success, align with replay behavior:
        # - fold success can occur at any rollout step
        # - arms must be at rest at the end
        if getattr(success_term.func, "__name__", "") == "garment_folded":
            arms_rest_final = _are_arms_at_rest(env)
            if not (fold_success_seen and arms_rest_final):
                _print_success_debug_breakdown(env)
                print(
                    "\tReplay success check:"
                    f" fold_success_seen={fold_success_seen}, arms_rest_final={arms_rest_final}"
                )
                return False
        else:
            if not bool(success_term.func(env, **success_term.params)[0]):
                _print_success_debug_breakdown(env)
                return False
    return True


def annotate_episode_in_auto_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    episode_index: int | None = None,
    garment_info: dict | None = None,
) -> bool:
    """Annotates an episode in automatic mode.

    This function replays the given episode in the environment and checks if the task was successfully completed.
    If the task was not completed, it will print a message and return False. Otherwise, it will check if all the
    subtask term signals are annotated and return True if they are, False otherwise.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.

    Returns:
        True if the episode was successfully annotated, False otherwise.
    """
    global skip_episode
    skip_episode = False
    is_episode_annotated_successfully = replay_episode(
        env, episode, success_term, episode_index=episode_index, garment_info=garment_info
    )
    if skip_episode:
        print("\tSkipping the episode.")
        return False
    if not is_episode_annotated_successfully:
        print("\tThe final task was not completed.")
        if not args_cli.ignore_replay_success:
            return False
        print("\tContinuing because --ignore_replay_success is enabled.")
        is_episode_annotated_successfully = True

    if is_episode_annotated_successfully:
        # check if all the subtask term signals are annotated
        annotated_episode = env.recorder_manager.get_episode(0)
        _merge_source_obs_into_annotated_episode(episode, annotated_episode)
        if bool(args_cli.require_ik_actions):
            _overwrite_annotated_actions_with_source_actions(episode, annotated_episode)
            _validate_recorded_datagen_pose_contract(
                annotated_episode, z_gap_threshold=float(args_cli.strict_pose_z_gap_threshold)
            )
        subtask_term_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_term_signals"]
        for signal_name, signal_flags in subtask_term_signal_dict.items():
            if not torch.any(signal_flags):
                is_episode_annotated_successfully = False
                print(f'\tDid not detect completion for the subtask "{signal_name}".')
    return is_episode_annotated_successfully


def annotate_episode_in_manual_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    subtask_term_signal_names: dict[str, list[str]] = {},
    episode_index: int | None = None,
    garment_info: dict | None = None,
) -> bool:
    """Annotates an episode in manual mode.

    This function replays the given episode in the environment and allows for manual marking of subtask term signals.
    It iterates over each eef and prompts the user to mark the subtask term signals for that eef.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.
        subtask_term_signal_names: Dictionary mapping eef names to lists of subtask term signal names.

    Returns:
        True if the episode was successfully annotated, False otherwise.
    """
    global is_paused, marked_subtask_action_indices, skip_episode
    global expected_subtask_mark_count, last_marked_action_index, last_mark_wall_time
    global active_mark_eef_name, active_mark_signal_names
    # iterate over the eefs for marking subtask term signals
    subtask_term_signal_action_indices = {}
    episode_replay_success = True
    for eef_name, eef_subtask_term_signal_names in subtask_term_signal_names.items():
        # skip if no subtask annotation is needed for this eef
        if len(eef_subtask_term_signal_names) == 0:
            continue
        expected_subtask_signal_count = len(eef_subtask_term_signal_names)

        while True:
            is_paused = False
            skip_episode = False
            print(f'\tPlaying the episode for subtask annotations for eef "{eef_name}".')
            print("\tSubtask signals to annotate:")
            print(f"\t\t- Termination:\t{eef_subtask_term_signal_names}")

            print('\n\tReplay starts immediately.')
            print('\tPress "B" to pause.')
            print('\tPress "N" to resume.')
            print('\tPress "S" to annotate subtask signals.')
            print('\tPress "Q" to skip the episode.\n')
            marked_subtask_action_indices = []
            expected_subtask_mark_count = expected_subtask_signal_count
            active_mark_eef_name = eef_name
            active_mark_signal_names = list(eef_subtask_term_signal_names)
            last_marked_action_index = -10**9
            last_mark_wall_time = 0.0
            task_success_result = replay_episode(
                env, episode, success_term, episode_index=episode_index, garment_info=garment_info
            )
            if skip_episode:
                print("\tSkipping the episode.")
                return False

            mark_summaries = []
            for marked_signal_index, action_index in enumerate(marked_subtask_action_indices):
                if marked_signal_index < len(eef_subtask_term_signal_names):
                    signal_name = eef_subtask_term_signal_names[marked_signal_index]
                else:
                    signal_name = f"signal_{marked_signal_index}"
                mark_summaries.append(f"{signal_name}@{action_index}")
            print(f"\tSubtasks marked: {mark_summaries}")
            marks_complete = expected_subtask_signal_count == len(marked_subtask_action_indices)

            if marks_complete:
                if task_success_result:
                    print(f'\tAll {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were annotated.')
                else:
                    episode_replay_success = False
                    print(
                        f'\tReplay did not reach final task success for eef "{eef_name}",'
                        " but all requested subtask marks were captured."
                    )
                for marked_signal_index in range(expected_subtask_signal_count):
                    # collect subtask term signal action indices
                    subtask_term_signal_action_indices[eef_subtask_term_signal_names[marked_signal_index]] = (
                        marked_subtask_action_indices[marked_signal_index]
                    )
                break

            if not task_success_result:
                episode_replay_success = False
                print(
                    "\tThe final task was not completed in this replay attempt."
                    " Continue marking until all subtask signals are captured."
                )

            if expected_subtask_signal_count != len(marked_subtask_action_indices):
                print(
                    f"\tOnly {len(marked_subtask_action_indices)} out of"
                    f' {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were'
                    " annotated."
                )

            print(f'\tThe episode will be replayed again for re-marking subtask signals for the eef "{eef_name}".\n')

    active_mark_eef_name = None
    active_mark_signal_names = []

    if not episode_replay_success and not args_cli.ignore_replay_success:
        print(
            "\tThe final task was not completed in replay."
            " Add --ignore_replay_success to export annotations anyway."
        )
        return False
    if not episode_replay_success:
        print("\tContinuing because --ignore_replay_success is enabled.")

    annotated_episode = env.recorder_manager.get_episode(0)
    _merge_source_obs_into_annotated_episode(episode, annotated_episode)
    if bool(args_cli.require_ik_actions):
        _overwrite_annotated_actions_with_source_actions(episode, annotated_episode)
        _validate_recorded_datagen_pose_contract(
            annotated_episode, z_gap_threshold=float(args_cli.strict_pose_z_gap_threshold)
        )
    for (
        subtask_term_signal_name,
        subtask_term_signal_action_index,
    ) in subtask_term_signal_action_indices.items():
        # subtask termination signal is false until subtask is complete, and true afterwards
        subtask_signals = torch.ones(len(episode.data["actions"]), dtype=torch.bool)
        subtask_signals[:subtask_term_signal_action_index] = False
        annotated_episode.add(f"obs/datagen_info/subtask_term_signals/{subtask_term_signal_name}", subtask_signals)
    return True


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
