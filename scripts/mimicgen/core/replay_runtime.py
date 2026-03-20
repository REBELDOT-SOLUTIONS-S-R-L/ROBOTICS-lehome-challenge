"""Replay runtime helpers for HDF5 MimicGen datasets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.utils.logger import get_logger
from lehome.utils.record import RateLimiter

from ...utils.common import stabilize_garment_after_reset
from ...utils.garment_debug_markers import GarmentKeypointDebugMarkers

logger = get_logger(__name__)


def create_replay_dataset(
    args: argparse.Namespace,
    source_hdf5_path: Path,
    fps: int,
) -> tuple[LeRobotDataset | None, Path | None]:
    """Create a dataset for saving replayed episodes."""
    if args.output_root is None:
        return None, None

    output_path = Path(args.output_root)
    root = output_path / source_hdf5_path.stem
    if root.exists():
        logger.warning(
            f"Target path {root} already exists. Deleting it to create a fresh replay dataset."
        )
        shutil.rmtree(root)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating replay dataset at: {root}")
    replay_dataset = LeRobotDataset.create(
        repo_id="replay_output",
        fps=fps,
        root=root,
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=None,
    )

    json_path = replay_dataset.root / "meta" / "garment_info.json"
    return replay_dataset, json_path


def compute_action_from_ee_pose(
    env: DirectRLEnv,
    frame_data: dict[str, torch.Tensor],
    ik_solver: Any,
    is_bimanual: bool,
    args: argparse.Namespace,
    ik_stats: dict[str, Any],
    device: str = "cpu",
) -> torch.Tensor | None:
    """Compute joint actions from action.ee_pose using inverse kinematics."""
    from lehome.utils import compute_joints_from_ee_pose

    try:
        if "action.ee_pose" not in frame_data:
            logger.warning(
                "action.ee_pose not found in frame data, falling back to original action"
            )
            ik_stats["total"] += 1
            ik_stats["fallback"] += 1
            return None

        action_ee_pose = frame_data["action.ee_pose"].cpu().numpy()
        current_state = frame_data["observation.state"].cpu().numpy().flatten()

        if is_bimanual:
            left_ee = action_ee_pose[:8]
            right_ee = action_ee_pose[8:16]
            current_left = current_state[:6]
            current_right = current_state[6:12]

            left_joints = compute_joints_from_ee_pose(
                ik_solver,
                current_left,
                left_ee,
                args.ee_state_unit,
                orientation_weight=1.0,
            )
            right_joints = compute_joints_from_ee_pose(
                ik_solver,
                current_right,
                right_ee,
                args.ee_state_unit,
                orientation_weight=1.0,
            )

            if left_joints is None or right_joints is None:
                ik_stats["fallback"] += 1
                return None

            action_joints = np.concatenate([left_joints, right_joints], axis=0)
        else:
            action_joints = compute_joints_from_ee_pose(
                ik_solver,
                current_state,
                action_ee_pose,
                args.ee_state_unit,
                orientation_weight=1.0,
            )
            if action_joints is None:
                ik_stats["fallback"] += 1
                return None

        action_tensor = torch.from_numpy(action_joints).float().to(device).unsqueeze(0)

        ik_stats["total"] += 1
        ik_stats["success"] += 1

        original_action = frame_data["action"].cpu().numpy()
        error = np.max(np.abs(action_joints - original_action))
        ik_stats["errors"].append(error)

        return action_tensor
    except Exception as exc:
        logger.warning(f"IK computation failed: {exc}", exc_info=True)
        ik_stats["total"] += 1
        ik_stats["fallback"] += 1
        return None


def replay_episode(
    env: DirectRLEnv,
    episode_data: list[dict[str, torch.Tensor]],
    rate_limiter: RateLimiter | None,
    initial_pose: dict[str, Any] | None,
    args: argparse.Namespace,
    replay_dataset: LeRobotDataset | None = None,
    disable_depth: bool = False,
    ik_solver: Any | None = None,
    is_bimanual: bool = False,
    ik_stats: dict[str, Any] | None = None,
    device: str = "cpu",
    task_description: str = "fold the garment on the table",
    debug_markers: GarmentKeypointDebugMarkers | None = None,
) -> bool:
    """Replay a single episode from recorded data."""
    try:
        env.reset()
        if debug_markers is not None:
            debug_markers.update_from_env(env)

        if initial_pose is not None:
            env.set_all_pose(initial_pose)
            logger.debug(f"Set initial pose from recorded data: {initial_pose}")
            if debug_markers is not None:
                debug_markers.update_from_env(env)
        else:
            logger.warning("No initial pose found in recorded data, using default pose")

        stabilize_garment_after_reset(env, args)
        if debug_markers is not None:
            debug_markers.update_from_env(env)

        success_achieved = False

        for idx in range(len(episode_data)):
            if rate_limiter:
                rate_limiter.sleep(env)

            if args.use_ee_pose and ik_solver is not None:
                action = compute_action_from_ee_pose(
                    env,
                    episode_data[idx],
                    ik_solver,
                    is_bimanual,
                    args,
                    ik_stats,
                    device,
                )
                if action is None:
                    action = episode_data[idx]["action"].to(device).unsqueeze(0)
            else:
                action = episode_data[idx]["action"].to(device).unsqueeze(0)

            env.step(action)
            if debug_markers is not None:
                debug_markers.update_from_env(env)

            if replay_dataset is not None:
                observations = env._get_observations()
                if disable_depth and "observation.top_depth" in observations:
                    observations = {
                        key: value
                        for key, value in observations.items()
                        if key != "observation.top_depth"
                    }
                frame = {**observations, "task": task_description}
                replay_dataset.add_frame(frame)

            success = env._get_success().item()
            if success:
                success_achieved = True

        return success_achieved
    except Exception as exc:
        logger.error(f"Error during episode replay: {exc}", exc_info=True)
        return False


def append_episode_initial_pose(
    json_path: Path,
    episode_idx: int,
    object_initial_pose: dict[str, Any],
    garment_name: str | None = None,
    scale: Any | None = None,
) -> None:
    """Append initial pose information to output garment_info.json."""
    from lehome.utils.record import append_episode_initial_pose as append_pose

    append_pose(
        json_path,
        episode_idx,
        object_initial_pose,
        garment_name=garment_name,
        scale=scale,
    )


__all__ = [
    "append_episode_initial_pose",
    "compute_action_from_ee_pose",
    "create_replay_dataset",
    "replay_episode",
]
