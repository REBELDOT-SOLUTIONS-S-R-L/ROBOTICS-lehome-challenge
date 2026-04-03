"""Dataset recording orchestration for MimicGen teleoperation."""

from __future__ import annotations

import argparse

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaacsim.simulation_app import SimulationApp

from lehome.utils.logger import get_logger

from .cuda_visual_sync import apply_cuda_fabric_render_settings
from .teleop_runtime import (
    DEBUG_POSE_LOG_INTERVAL,
    create_dataset_if_needed,
    create_debug_markers_if_needed,
    create_rate_limiter,
    create_teleop_interface,
    register_teleop_callbacks,
    run_idle_phase,
    run_live_control_without_record,
    run_recording_phase,
    validate_task_and_device,
)

logger = get_logger(__name__)


def record_dataset(args: argparse.Namespace, simulation_app: SimulationApp) -> None:
    """Record dataset."""
    device = getattr(args, "device", "cpu")

    env_cfg = parse_env_cfg(args.task, device=device)
    task_name = args.task
    apply_cuda_fabric_render_settings(env_cfg, device, context="teleop recording")

    env_cfg.garment_name = args.garment_name
    env_cfg.garment_version = args.garment_version
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    if args.use_random_seed:
        env_cfg.use_random_seed = True
        logger.info("Using random seed (no fixed seed)")
    else:
        env_cfg.use_random_seed = False
        env_cfg.random_seed = args.seed
        logger.info(f"Using fixed random seed: {args.seed}")

    env: DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    dataset = None
    teleop_interface = create_teleop_interface(env, args)
    flags = register_teleop_callbacks(teleop_interface, recording_enabled=args.enable_record)
    teleop_interface.reset()
    dataset = create_dataset_if_needed(env, args)
    debug_markers = create_debug_markers_if_needed(args)
    count_render = 0
    printed_instructions = False
    idle_frame_counter = 0
    object_initial_pose = None
    debug_pose_state: dict[str, object] = {"step_count": 0, "eef_body_idx_cache": {}}
    debug_marker_state: dict[str, object] = {"step_count": 0}
    control_state: dict[str, object] = {"maintain_action": None}
    rate_limiter = create_rate_limiter(args)

    if getattr(args, "debugging_log_pose", False):
        logger.info(
            "[Debug Pose] Enabled. Logging EEF and garment checkpoint positions in cm "
            f"at step 0 and every {DEBUG_POSE_LOG_INTERVAL} sim steps."
        )

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if not flags["start"]:
                    pose, count_render = run_idle_phase(
                        env,
                        teleop_interface,
                        args,
                        count_render,
                        debug_pose_state,
                        control_state,
                        rate_limiter,
                        debug_markers,
                        debug_marker_state,
                    )
                    if pose is not None:
                        object_initial_pose = pose

                    if count_render > 0:
                        idle_frame_counter += 1
                        if idle_frame_counter == 100 and not printed_instructions:
                            logger.info("=" * 60)
                            logger.info("🎮 CONTROL INSTRUCTIONS 🎮")
                            logger.info("=" * 60)
                            logger.info(str(teleop_interface))
                            logger.info("=" * 60 + "\n\n")
                            printed_instructions = True
                elif args.enable_record and dataset is not None:
                    object_initial_pose = run_recording_phase(
                        env,
                        teleop_interface,
                        args,
                        flags,
                        dataset,
                        object_initial_pose,
                        rate_limiter,
                        control_state,
                        debug_markers,
                        debug_marker_state,
                    )
                    break
                else:
                    run_live_control_without_record(
                        env,
                        teleop_interface,
                        args,
                        debug_pose_state,
                        control_state,
                        rate_limiter,
                        debug_markers,
                        debug_marker_state,
                    )
    except KeyboardInterrupt:
        logger.warning("\n[Ctrl+C] Interrupt signal detected")
        if args.enable_record and dataset is not None and flags["start"]:
            logger.info("Clearing current episode buffer...")
            dataset.discard_episode()
            logger.info("Buffer cleared, dataset remains intact")
            dataset.finalize()
            logger.info("Dataset saved")
    except Exception as exc:
        logger.error(f"An unexpected error occurred: {exc}")
    finally:
        if debug_markers is not None:
            debug_markers.close()
        if dataset is not None:
            dataset.finalize()
        env.close()


__all__ = ["record_dataset", "validate_task_and_device"]
