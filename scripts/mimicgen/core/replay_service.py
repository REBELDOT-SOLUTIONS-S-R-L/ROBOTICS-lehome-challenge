"""Dataset replay orchestration for recorded HDF5 episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from lehome.utils.logger import get_logger
from lehome.utils.record import RateLimiter

from ...utils.garment_debug_markers import GarmentKeypointDebugMarkers
from .dataset_meta import (
    extract_garment_name_from_episode_meta as _extract_garment_name_from_episode_meta,
    extract_initial_pose_from_episode_meta as _extract_initial_pose_from_episode_meta,
    find_garment_info_json,
    load_initial_pose_from_json,
    try_read_garment_name_from_json as _try_read_garment_name_from_json,
)
from .replay_runtime import (
    append_episode_initial_pose,
    create_replay_dataset,
    replay_episode,
)
from .replay_source import HDF5ReplaySource

logger = get_logger(__name__)


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments for HDF5 dataset replay."""
    dataset_path = Path(args.dataset_root)
    if not dataset_path.exists():
        raise ValueError(f"HDF5 dataset file does not exist: {args.dataset_root}")
    if not dataset_path.is_file():
        raise ValueError(f"dataset_root must be an HDF5 file path, got: {args.dataset_root}")
    if dataset_path.suffix.lower() not in {".hdf5", ".h5"}:
        raise ValueError(
            f"dataset_root must point to a .hdf5/.h5 file, got: {args.dataset_root}"
        )

    with HDF5ReplaySource(args.dataset_root) as dataset:
        if dataset.num_episodes == 0:
            raise ValueError(f"HDF5 dataset has no episodes: {args.dataset_root}")

        if args.start_episode >= dataset.num_episodes:
            raise ValueError(
                f"start_episode ({args.start_episode}) is out of range. "
                f"Dataset has {dataset.num_episodes} episodes."
            )

    if args.num_replays < 1:
        raise ValueError(f"num_replays must be >= 1, got {args.num_replays}")

    if args.start_episode < 0:
        raise ValueError(f"start_episode must be >= 0, got {args.start_episode}")

    if args.end_episode is not None:
        if args.end_episode < 0:
            raise ValueError(f"end_episode must be >= 0, got {args.end_episode}")
        if args.end_episode <= args.start_episode:
            raise ValueError(
                f"end_episode ({args.end_episode}) must be > start_episode ({args.start_episode})"
            )

    if getattr(args, "debugging_markers", False) and args.output_root is not None:
        raise ValueError(
            "--debugging_markers is viewport-only and cannot be used together with --output_root."
        )

    if getattr(args, "debugging_markers", False) and getattr(args, "headless", False):
        logger.warning(
            "--debugging_markers was enabled in headless mode. Replay will continue, "
            "but the markers will not be visible."
        )


def replay(args: argparse.Namespace) -> None:
    """Replay HDF5 recorded datasets for visualization and verification."""
    validate_args(args)

    source_hdf5_path = Path(args.dataset_root)
    garment_info_path = find_garment_info_json(source_hdf5_path)
    if garment_info_path is not None:
        logger.info(f"Using garment info from: {garment_info_path}")
    else:
        logger.warning(
            "No external garment_info.json found for this HDF5 file. "
            "Replay will rely on /data/demo_*/meta when available."
        )

    device = getattr(args, "device", "cpu")
    task_description = getattr(args, "task_description", "fold the garment on the table")

    ik_solver: Any | None = None
    is_bimanual = False
    ik_stats: dict[str, Any] = {"total": 0, "success": 0, "fallback": 0, "errors": []}
    debug_markers: GarmentKeypointDebugMarkers | None = None

    with HDF5ReplaySource(args.dataset_root) as dataset:
        if args.use_ee_pose and not dataset.has_ee_pose():
            raise ValueError(
                "HDF5 dataset does not contain ee_pose actions. "
                "Expected one of: 'data/demo_*/obs/ee_pose' or top-level "
                "'data/demo_*/actions' in ee_pose mode."
            )

        if args.use_ee_pose:
            urdf_path = Path(args.ee_urdf_path)
            if not urdf_path.exists():
                raise FileNotFoundError(f"URDF file not found: {urdf_path}")

            from lehome.utils import RobotKinematics

            state_dim = dataset.get_state_dim()
            is_bimanual = state_dim >= 12

            solver_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ]
            ik_solver = RobotKinematics(
                str(urdf_path),
                target_frame_name="gripper_frame_link",
                joint_names=solver_names,
            )
            arm_mode = "dual-arm" if is_bimanual else "single-arm"
            logger.info(f"IK solver loaded ({arm_mode} mode)")
            logger.warning(
                "Using action.ee_pose + IK control, which may differ from original action"
            )

        logger.info(f"Creating environment: {args.task}")
        env_cfg = parse_env_cfg(args.task, device=device)

        garment_name = getattr(args, "garment_name", None)
        if isinstance(garment_name, str):
            garment_name = garment_name.strip() or None
        if garment_name is not None:
            logger.info(f"Using garment name from CLI: {garment_name}")

        first_episode_meta = dataset.get_episode_meta(0)
        if garment_name is None and first_episode_meta:
            garment_name = _extract_garment_name_from_episode_meta(first_episode_meta)
            if garment_name is not None:
                logger.info("Using garment name from /data/demo_0/meta or initial_state/garment")

        if garment_name is None:
            garment_name = dataset.get_garment_name_from_env_args()
        if garment_name is None and garment_info_path is not None:
            garment_name = _try_read_garment_name_from_json(garment_info_path)

        default_garment_name = getattr(env_cfg, "garment_name", None)
        if isinstance(default_garment_name, str):
            default_garment_name = default_garment_name.strip() or None
        if garment_name is None:
            garment_name = default_garment_name

        if garment_name is not None:
            env_cfg.garment_name = garment_name
            logger.info(f"Using garment name: {garment_name}")
        else:
            raise ValueError(
                "Could not determine garment name for replay. "
                "This HDF5 file does not include garment metadata in /data/demo_*/meta, "
                "initial_state/garment, data/env_args, or garment_info.json. "
                "Pass --garment_name (for example, Top_Long_Unseen_0)."
            )

        env_cfg.garment_version = args.garment_version
        env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
        env_cfg.particle_cfg_path = args.particle_cfg_path

        env: DirectRLEnv = gym.make(args.task, cfg=env_cfg).unwrapped

        try:
            logger.info("Initializing observations...")
            env.initialize_obs()
            logger.info("Observations initialized successfully")

            if getattr(args, "debugging_markers", False):
                try:
                    debug_markers = GarmentKeypointDebugMarkers()
                    logger.info("Enabled garment semantic keypoint debugging markers.")
                except Exception as exc:
                    logger.warning(
                        f"Failed to initialize debugging markers. Continuing without them: {exc}",
                        exc_info=True,
                    )
                    debug_markers = None

            rate_limiter = RateLimiter(args.step_hz) if args.step_hz > 0 else None
            replay_dataset, json_path = create_replay_dataset(args, source_hdf5_path, dataset.fps)

            start_idx = args.start_episode
            end_idx = args.end_episode if args.end_episode is not None else dataset.num_episodes
            end_idx = min(end_idx, dataset.num_episodes)
            total_episodes = end_idx - start_idx

            logger.info(
                f"Replaying episodes {start_idx} to {end_idx - 1} "
                f"(displayed as 1 to {total_episodes})"
            )

            total_attempts = 0
            total_successes = 0
            saved_episodes = 0

            try:
                for episode_idx in range(start_idx, end_idx):
                    display_episode_num = episode_idx - start_idx + 1
                    source_episode_index = dataset.get_source_episode_index(episode_idx)

                    logger.info("")
                    logger.info(f"{'=' * 60}")
                    logger.info(f"Episode {display_episode_num}/{total_episodes}")
                    logger.info(
                        f"Demo: {dataset.demo_names[episode_idx]} "
                        f"(source episode {source_episode_index})"
                    )
                    logger.info(f"{'=' * 60}")

                    episode_meta = dataset.get_episode_meta(episode_idx)
                    initial_pose = _extract_initial_pose_from_episode_meta(
                        episode_meta,
                        source_episode_index=source_episode_index,
                    )
                    if initial_pose is None:
                        initial_pose = load_initial_pose_from_json(
                            garment_info_path,
                            source_episode_index,
                        )

                    try:
                        episode_data = dataset.get_episode_frames(
                            episode_idx,
                            require_ee_pose=args.use_ee_pose,
                        )
                    except Exception as exc:
                        logger.error(
                            f"Failed to load {dataset.demo_names[episode_idx]}: {exc}",
                            exc_info=True,
                        )
                        continue

                    if len(episode_data) == 0:
                        logger.warning(
                            f"Episode {display_episode_num} has no frame data, skipping..."
                        )
                        continue

                    logger.info(f"Episode length: {len(episode_data)} frames")

                    for replay_idx in range(args.num_replays):
                        total_attempts += 1

                        if replay_dataset is not None:
                            replay_dataset.clear_episode_buffer()

                        success = replay_episode(
                            env=env,
                            episode_data=episode_data,
                            rate_limiter=rate_limiter,
                            initial_pose=initial_pose,
                            args=args,
                            replay_dataset=replay_dataset,
                            disable_depth=args.disable_depth,
                            ik_solver=ik_solver,
                            is_bimanual=is_bimanual,
                            ik_stats=ik_stats,
                            device=device,
                            task_description=task_description,
                            debug_markers=debug_markers,
                        )

                        if success:
                            total_successes += 1
                            logger.info(f"  [Replay {replay_idx + 1}/{args.num_replays}] Success")
                        else:
                            logger.info(f"  [Replay {replay_idx + 1}/{args.num_replays}] Failed")

                        should_save = replay_dataset is not None and (
                            not args.save_successful_only or success
                        )
                        if should_save:
                            try:
                                replay_dataset.save_episode()
                                if json_path is not None and initial_pose is not None:
                                    append_episode_initial_pose(
                                        json_path,
                                        saved_episodes,
                                        initial_pose,
                                    )
                                saved_episodes += 1
                                logger.info(f"  Saved as episode {saved_episodes - 1}")
                            except Exception as exc:
                                logger.error(f"Failed to save episode: {exc}", exc_info=True)
                        elif replay_dataset is not None:
                            replay_dataset.clear_episode_buffer()
            finally:
                if replay_dataset is not None:
                    try:
                        replay_dataset.clear_episode_buffer()
                        replay_dataset.finalize()
                    except Exception as exc:
                        logger.error(f"Error finalizing dataset: {exc}", exc_info=True)

            logger.info("")
            logger.info(f"{'=' * 60}")
            logger.info("Replay Statistics")
            logger.info(f"{'=' * 60}")
            logger.info(f"  Total attempts: {total_attempts}")
            logger.info(f"  Total successes: {total_successes}")
            if total_attempts > 0:
                logger.info(f"  Success rate: {100.0 * total_successes / total_attempts:.1f}%")
            if replay_dataset is not None:
                logger.info(f"  Saved episodes: {saved_episodes}")

            if args.use_ee_pose and ik_stats["total"] > 0:
                logger.info("")
                logger.info("IK Statistics")
                logger.info(f"  Total IK attempts: {ik_stats['total']}")
                logger.info(f"  IK successes: {ik_stats['success']}")
                logger.info(f"  IK fallbacks: {ik_stats['fallback']}")
                logger.info(
                    f"  IK success rate: "
                    f"{100.0 * ik_stats['success'] / ik_stats['total']:.1f}%"
                )
                if ik_stats["errors"]:
                    errors = np.array(ik_stats["errors"])
                    unit = "rad" if args.ee_state_unit == "rad" else "deg"
                    logger.info(f"  Joint angle error vs original action ({unit}):")
                    logger.info(f"    mean = {np.mean(errors):.6f}")
                    logger.info(f"    max  = {np.max(errors):.6f}")

            logger.info(f"{'=' * 60}")
        finally:
            env.close()


__all__ = ["replay", "validate_args"]
