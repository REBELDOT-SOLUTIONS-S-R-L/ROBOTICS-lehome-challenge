import argparse
from typing import TYPE_CHECKING
import numpy as np
import torch

from isaaclab.app import AppLauncher
from isaacsim.simulation_app import SimulationApp

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv

_HOME_POSITION_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def _get_cached_home_positions() -> tuple[np.ndarray, np.ndarray]:
    """Load home joint targets lazily to avoid importing Isaac asset modules before SimulationApp starts."""
    global _HOME_POSITION_CACHE
    if _HOME_POSITION_CACHE is not None:
        return _HOME_POSITION_CACHE

    from lehome.assets.robots.lerobot import (
        ACTION_NAMES,
        SO101_FOLLOWER_HOME_JOINT_POS,
        SO101_LEFT_ARM_HOME_JOINT_POS,
        SO101_RIGHT_ARM_HOME_JOINT_POS,
    )

    single_arm_home_position = np.array(
        [SO101_FOLLOWER_HOME_JOINT_POS[action_name] for action_name in ACTION_NAMES],
        dtype=np.float32,
    )
    left_arm_home_position = np.array(
        [SO101_LEFT_ARM_HOME_JOINT_POS[action_name] for action_name in ACTION_NAMES],
        dtype=np.float32,
    )
    right_arm_home_position = np.array(
        [SO101_RIGHT_ARM_HOME_JOINT_POS[action_name] for action_name in ACTION_NAMES],
        dtype=np.float32,
    )
    dual_arm_home_position = np.concatenate([left_arm_home_position, right_arm_home_position])

    _HOME_POSITION_CACHE = (single_arm_home_position, dual_arm_home_position)
    return _HOME_POSITION_CACHE


def launch_app(parser: argparse.ArgumentParser) -> SimulationApp:
    """Launch Isaac Sim app from parser (parses args internally).

    Use this when you haven't parsed arguments yet.
    """
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    return launch_app_from_args(args)


def launch_app_from_args(args: argparse.Namespace) -> SimulationApp:
    """Launch Isaac Sim app from already parsed arguments.

    Use this when arguments are already parsed (e.g., in subcommand handlers).

    Args:
        args: Already parsed command-line arguments (must include AppLauncher args).

    Returns:
        SimulationApp instance.
    """
    args.kit_args = (
        "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
    )
    app_launcher = AppLauncher(vars(args))
    simulation_app = app_launcher.app
    return simulation_app


def close_app(simulation_app: SimulationApp) -> None:
    """Close Isaac Sim app."""
    simulation_app.close()


def stabilize_garment_after_reset(
    env: "DirectRLEnv",
    args: argparse.Namespace,
    num_steps: int = 60,
) -> None:
    """Stabilize garment after environment reset by running physics steps.

    Moves robot to home position and lets garment settle naturally after reset,
    preventing floating or clipping. This is critical for garment physics to
    initialize properly, especially when using CUDA device.

    Args:
        env: Environment instance.
        args: Command-line arguments containing task name.
        num_steps: Number of stabilization steps to run.
    """
    if num_steps <= 0:
        return

    is_bimanual = "Bi" in args.task or "bi" in args.task.lower()

    action_dim = None
    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None and hasattr(action_manager, "total_action_dim"):
        try:
            action_dim = int(action_manager.total_action_dim)
        except Exception:
            action_dim = None

    if action_dim is None:
        try:
            initial_obs = env._get_observations()
            action_dim = (
                len(initial_obs["action"])
                if "action" in initial_obs
                else (12 if is_bimanual else 6)
            )
        except Exception:
            action_dim = 12 if is_bimanual else 6

    single_arm_home_position, dual_arm_home_position = _get_cached_home_positions()
    home_joints = dual_arm_home_position if is_bimanual else single_arm_home_position

    home_action: torch.Tensor | None = None
    if len(home_joints) == action_dim:
        home_action = torch.from_numpy(home_joints).float().to(env.device).unsqueeze(0)
    elif (
        is_bimanual
        and action_dim == 16
        and hasattr(env, "_compute_target_pose_from_joint_targets")
        and hasattr(env, "target_eef_pose_to_action")
    ):
        try:
            home_joint_tensor = torch.from_numpy(home_joints).float().to(env.device).unsqueeze(0)
            left_joint_targets = home_joint_tensor[:, :6]
            right_joint_targets = home_joint_tensor[:, 6:12]
            target_eef_pose_dict = {
                "left_arm": env._compute_target_pose_from_joint_targets("left_arm", left_joint_targets),
                "right_arm": env._compute_target_pose_from_joint_targets("right_arm", right_joint_targets),
            }
            gripper_action_dict = {
                "left_arm": left_joint_targets[:, 5:6],
                "right_arm": right_joint_targets[:, 5:6],
            }
            home_action = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_eef_pose_dict,
                gripper_action_dict=gripper_action_dict,
                env_id=0,
            )
        except Exception:
            home_action = None

    if home_action is None:
        # Use warning from logger if available, otherwise print
        try:
            from lehome.utils.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(
                f"Home position dimension mismatch: got {len(home_joints)}, "
                f"expected {action_dim}. Using zeros."
            )
        except Exception:
            pass
        home_action = torch.zeros(1, action_dim, dtype=torch.float32, device=env.device)

    force_cuda_render_sync = None
    cuda_visual_sync_enabled = None
    with torch.inference_mode():
        try:
            from scripts.mimicgen.core.cuda_visual_sync import cuda_visual_sync_enabled as _cuda_visual_sync_enabled
            from scripts.mimicgen.core.cuda_visual_sync import force_cuda_render_sync as _force_cuda_render_sync

            cuda_visual_sync_enabled = _cuda_visual_sync_enabled
            force_cuda_render_sync = _force_cuda_render_sync
        except Exception:
            force_cuda_render_sync = None
            cuda_visual_sync_enabled = None

    for step_idx in range(num_steps):
        env.step(home_action)
        if (
            force_cuda_render_sync is not None
            and cuda_visual_sync_enabled is not None
            and cuda_visual_sync_enabled(env)
        ):
            force_cuda_render_sync(env)
        elif (step_idx + 1) % 10 == 0 or step_idx == num_steps - 1:
            env.render()
