"""Teleoperation runtime helpers for MimicGen HDF5 recording."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

import torch

from isaaclab.envs import DirectRLEnv

from lehome.devices import BiKeyboard, BiSO101Leader, SO101Leader, Se3Keyboard
from lehome.utils.logger import get_logger
from lehome.utils.record import get_next_experiment_path_with_gap

from ...utils.common import stabilize_garment_after_reset
from ...utils.garment_debug_markers import GarmentKeypointDebugMarkers
from .cuda_visual_sync import cuda_visual_sync_enabled
from .cuda_visual_sync import force_cuda_render_sync
from .cuda_visual_sync import post_reset_cuda_visual_sync
from .data_utils import as_numpy
from .env_runtime import get_env_garment_metadata
from .record_debug import (
    DEBUG_POSE_LOG_INTERVAL,
    SUCCESS_LOG_INTERVAL,
    log_debug_pose_snapshot_if_enabled,
    log_episode_success_snapshot,
    log_success_result,
)
from .recording import DirectHDF5Recorder

logger = get_logger(__name__)


def create_debug_markers_if_needed(
    args: argparse.Namespace,
) -> GarmentKeypointDebugMarkers | None:
    """Create live garment semantic keypoint markers when enabled for teleop."""
    if not getattr(args, "debugging_markers", False):
        return None
    if getattr(args, "headless", False):
        logger.warning(
            "--debugging_markers was enabled in headless mode. Teleop will continue, "
            "but the markers will not be visible."
        )
        return None
    try:
        markers = GarmentKeypointDebugMarkers()
        logger.info("Enabled garment semantic keypoint debugging markers.")
        return markers
    except Exception as exc:
        logger.warning(
            f"Failed to initialize debugging markers. Continuing without them: {exc}",
            exc_info=True,
        )
        return None


def update_debug_markers_if_needed(
    env: DirectRLEnv,
    debug_markers: GarmentKeypointDebugMarkers | None,
    debug_marker_state: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> None:
    """Update marker overlay when live debugging markers are enabled."""
    if debug_markers is None:
        return
    debug_markers.update_from_env(env)
    if debug_marker_state is not None:
        debug_marker_state["step_count"] = int(debug_marker_state.get("step_count", 0)) + 1


def validate_task_and_device(args: argparse.Namespace) -> None:
    """Validate that task name matches the teleop device configuration."""
    if args.task is None:
        raise ValueError("Please specify --task.")
    if "Bi" in args.task:
        assert args.teleop_device in {"bi-so101leader", "bi-keyboard"}, (
            "Only support bi-so101leader or bi-keyboard for bi-arm task"
        )
    else:
        assert args.teleop_device in {"so101leader", "keyboard"}, (
            "Only support so101leader or keyboard for single-arm task"
        )


def _safe_get_all_pose(env: DirectRLEnv) -> dict[str, Any] | None:
    try:
        return env.get_all_pose()
    except Exception as exc:
        logger.error(f"[Recording] Failed to get initial pose: {exc}")
        traceback.print_exc()
        return None


def _get_or_build_maintain_action(
    env: DirectRLEnv,
    args: argparse.Namespace,
    control_state: dict[str, Any],
    current_obs: dict[str, Any],
) -> torch.Tensor:
    current_state = current_obs.get("observation.state")
    if current_state is not None:
        state_arr = as_numpy(current_state, dtype="float32").reshape(-1)
        action_dim = int(state_arr.shape[0])
    else:
        state_arr = None
        action_dim = 12 if "Bi" in (args.task or "") else 6

    maintain_action = control_state.get("maintain_action")
    if (
        maintain_action is None
        or not torch.is_tensor(maintain_action)
        or tuple(maintain_action.shape) != (1, action_dim)
        or str(maintain_action.device) != str(env.device)
    ):
        maintain_action = torch.zeros(1, action_dim, dtype=torch.float32, device=env.device)
        control_state["maintain_action"] = maintain_action

    if state_arr is None:
        maintain_action.zero_()
    else:
        maintain_action[0].copy_(torch.as_tensor(state_arr, dtype=torch.float32, device=env.device))
    return maintain_action


def create_teleop_interface(env: DirectRLEnv, args: argparse.Namespace) -> Any:
    """Create teleoperation interface based on device type."""
    if args.teleop_device == "keyboard":
        return Se3Keyboard(env, sensitivity=0.25 * args.sensitivity)
    if args.teleop_device == "so101leader":
        return SO101Leader(env, port=args.port, recalibrate=args.recalibrate)
    if args.teleop_device == "bi-so101leader":
        return BiSO101Leader(
            env,
            left_port=args.left_arm_port,
            right_port=args.right_arm_port,
            recalibrate=args.recalibrate,
        )
    if args.teleop_device == "bi-keyboard":
        return BiKeyboard(env, sensitivity=0.25 * args.sensitivity)
    raise ValueError(
        f"Invalid device interface '{args.teleop_device}'. "
        f"Supported: 'keyboard', 'so101leader', 'bi-so101leader', 'bi-keyboard'."
    )


def register_teleop_callbacks(
    teleop_interface: Any,
    *,
    recording_enabled: bool = False,
) -> dict[str, bool]:
    """Register callback functions for teleoperation control keys."""
    flags = {
        "start": False,
        "success": False,
        "remove": False,
        "abort": False,
    }

    def on_start():
        flags["start"] = True
        logger.info("[S] Recording started!")

    def on_success():
        if not recording_enabled or not flags["start"]:
            logger.debug("[N] Ignored (recording not started yet)")
            return
        flags["success"] = True
        logger.info("[N] Mark the current episode as successful.")

    def on_remove():
        if not recording_enabled or not flags["start"]:
            logger.debug("[D] Ignored (recording not started yet)")
            return
        flags["remove"] = True
        logger.info("[D] Discard the current episode and re-record.")

    def on_abort():
        flags["abort"] = True
        logger.warning("[ESC] Abort recording, clearing the current episode buffer...")

    teleop_interface.add_callback("S", on_start)
    teleop_interface.add_callback("N", on_success)
    teleop_interface.add_callback("D", on_remove)
    teleop_interface.add_callback("ESCAPE", on_abort)
    return flags


def _resolve_recording_output_paths(dataset_root: str) -> tuple[Path, Path]:
    """Resolve HDF5 output path and external garment-info json path."""
    root_path = Path(dataset_root)
    if root_path.suffix.lower() in {".hdf5", ".h5"}:
        hdf5_path = root_path
    else:
        run_dir = get_next_experiment_path_with_gap(root_path)
        hdf5_path = run_dir / "teleop_dataset.hdf5"
    json_path = hdf5_path.parent / "meta" / "garment_info.json"
    return hdf5_path, json_path


def create_dataset_if_needed(
    env: DirectRLEnv,
    args: argparse.Namespace,
) -> DirectHDF5Recorder | None:
    """Create direct HDF5 dataset writer if recording is enabled."""
    if not args.enable_record:
        return None

    is_bi_arm = ("Bi" in (args.task or "")) or (getattr(args, "teleop_device", "") or "").startswith("bi-")
    action_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    if is_bi_arm:
        joint_names = [f"left_{n}" for n in action_names] + [f"right_{n}" for n in action_names]
    else:
        joint_names = action_names

    hdf5_path, json_path = _resolve_recording_output_paths(getattr(args, "dataset_root", "Datasets/record"))

    env_args: dict[str, Any] = {"env_name": args.task or "", "type": 2}
    if getattr(args, "garment_name", None):
        env_args["garment_name"] = args.garment_name
    if getattr(args, "garment_version", None):
        env_args["garment_version"] = args.garment_version
    if getattr(args, "task_description", None):
        env_args["task_description"] = args.task_description
    env_args["joint_names"] = joint_names

    if getattr(args, "record_ee_pose", False):
        logger.warning(
            "--record_ee_pose is ignored by scripts/mimicgen/dataset_record_hdf5.py; "
            "the MimicGen pipeline consumes obs/*_ee_frame_state directly."
        )

    dataset = DirectHDF5Recorder(
        env=env,
        file_path=hdf5_path,
        env_args=env_args,
        fps=int(getattr(args, "step_hz", 30)),
        is_bi_arm=is_bi_arm,
        garment_info_json_path=json_path,
    )
    logger.info(f"Recording direct HDF5 file: {hdf5_path}")
    return dataset


def run_idle_phase(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    count_render: int,
    debug_pose_state: dict[str, Any],
    control_state: dict[str, Any],
    debug_markers: GarmentKeypointDebugMarkers | None = None,
    debug_marker_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, int]:
    """Run idle phase before recording starts."""
    actions = teleop_interface.advance()
    object_initial_pose = None

    if count_render == 0:
        logger.info("[Idle Phase] Initializing observations...")
        env.initialize_obs()
        update_debug_markers_if_needed(
            env,
            debug_markers,
            debug_marker_state,
            force=True,
        )
        count_render += 1

        logger.info("[Idle Phase] Stabilizing garment after initialization...")
        stabilize_garment_after_reset(env, args)
        post_reset_cuda_visual_sync(env)
        update_debug_markers_if_needed(
            env,
            debug_markers,
            debug_marker_state,
            force=True,
        )
        object_initial_pose = _safe_get_all_pose(env)
        if object_initial_pose is not None:
            control_state["cached_object_initial_pose"] = object_initial_pose
        logger.info("[Idle Phase] Ready for recording")

    if actions is None:
        current_obs = env._get_observations()
        maintain_action = _get_or_build_maintain_action(env, args, control_state, current_obs)
        env.step(maintain_action)
        if cuda_visual_sync_enabled(env):
            force_cuda_render_sync(env)
        else:
            env.render()
        if object_initial_pose is None:
            object_initial_pose = control_state.get("cached_object_initial_pose")
            if object_initial_pose is None:
                object_initial_pose = _safe_get_all_pose(env)
    else:
        env.step(actions)
        force_cuda_render_sync(env)
        object_initial_pose = _safe_get_all_pose(env)

    update_debug_markers_if_needed(env, debug_markers, debug_marker_state)

    if object_initial_pose is not None:
        control_state["cached_object_initial_pose"] = object_initial_pose

    log_debug_pose_snapshot_if_enabled(env, args, debug_pose_state)
    return object_initial_pose, count_render


def run_recording_phase(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    flags: dict[str, bool],
    dataset: DirectHDF5Recorder,
    initial_object_pose: dict[str, Any] | None,
    control_state: dict[str, Any] | None = None,
    debug_markers: GarmentKeypointDebugMarkers | None = None,
    debug_marker_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run the active episode recording loop after the user starts recording."""
    if control_state is None:
        control_state = {}

    episode_index = 0
    object_initial_pose = initial_object_pose

    if object_initial_pose is None:
        object_initial_pose = control_state.get("cached_object_initial_pose")
    if object_initial_pose is None:
        object_initial_pose = _safe_get_all_pose(env)
    if object_initial_pose is not None:
        control_state["cached_object_initial_pose"] = object_initial_pose

    while episode_index < args.num_episode:
        if flags["abort"]:
            dataset.discard_episode()
            dataset.finalize()
            logger.warning(f"Recording aborted, completed {episode_index} episodes")
            return object_initial_pose

        flags["success"] = False
        flags["remove"] = False
        episode_step_count = 0
        episode_discarded = False
        garment_name, scale = get_env_garment_metadata(env)
        if object_initial_pose is None:
            object_initial_pose = control_state.get("cached_object_initial_pose")
        if object_initial_pose is None:
            object_initial_pose = _safe_get_all_pose(env)
        if object_initial_pose is not None:
            control_state["cached_object_initial_pose"] = object_initial_pose

        dataset.begin_episode(
            episode_index=episode_index,
            object_initial_pose=object_initial_pose,
            garment_name=garment_name,
            scale=scale,
        )

        if args.log_success:
            log_success_result(
                env,
                episode_index=episode_index,
                step_in_episode=episode_step_count,
                context="episode_start",
            )

        while not flags["success"]:
            if flags["abort"]:
                dataset.discard_episode()
                dataset.finalize()
                logger.warning(f"Recording aborted, completed {episode_index} episodes")
                return object_initial_pose

            try:
                actions = teleop_interface.advance()
            except Exception as exc:
                logger.error(f"[Recording] Error in teleop interface: {exc}")
                traceback.print_exc()
                actions = None

            if actions is None:
                env.render()
            else:
                env.step(actions)
                force_cuda_render_sync(env)

            update_debug_markers_if_needed(env, debug_markers, debug_marker_state)

            episode_step_count += 1
            if args.log_success and episode_step_count % SUCCESS_LOG_INTERVAL == 0:
                log_success_result(
                    env,
                    episode_index=episode_index,
                    step_in_episode=episode_step_count,
                    context="periodic_check",
                )

            observations = env._get_observations()
            _, truncated = env._get_dones()
            frame: dict[str, Any] = {
                "action": observations["action"],
                "observation.state": observations["observation.state"],
            }
            for key in (
                "observation.images.top_rgb",
                "observation.images.left_rgb",
                "observation.images.right_rgb",
                "observation.images.wrist_rgb",
            ):
                if key in observations:
                    frame[key] = observations[key]

            dataset.append_frame(frame)

            if truncated or flags["remove"]:
                dataset.discard_episode()
                logger.info(f"Re-recording episode {episode_index}")
                try:
                    env.reset()
                    stabilize_garment_after_reset(env, args)
                    post_reset_cuda_visual_sync(env)
                    update_debug_markers_if_needed(
                        env,
                        debug_markers,
                        debug_marker_state,
                        force=True,
                    )
                    object_initial_pose = _safe_get_all_pose(env)
                    if object_initial_pose is not None:
                        control_state["cached_object_initial_pose"] = object_initial_pose
                except Exception as exc:
                    logger.error(f"[Recording] Failed to reset environment during re-recording: {exc}")
                    traceback.print_exc()
                    object_initial_pose = _safe_get_all_pose(env)
                flags["remove"] = False
                episode_discarded = True
                break

        if episode_discarded:
            continue

        log_episode_success_snapshot(env, episode_index, episode_step_count)

        logger.info(f"[Recording] Saving episode {episode_index}...")
        try:
            dataset.finalize_episode()
        except Exception as exc:
            logger.error(f"[Recording] Failed to save episode {episode_index}: {exc}")
            traceback.print_exc()
            dataset.discard_episode()
            raise

        episode_index += 1
        logger.info(f"Episode {episode_index - 1} completed, progress: {episode_index}/{args.num_episode}")

        try:
            env.reset()
            stabilize_garment_after_reset(env, args)
            post_reset_cuda_visual_sync(env)
            update_debug_markers_if_needed(
                env,
                debug_markers,
                debug_marker_state,
                force=True,
            )
        except Exception as exc:
            logger.error(f"[Recording] Failed to reset environment: {exc}")
            traceback.print_exc()

        object_initial_pose = _safe_get_all_pose(env)
        if object_initial_pose is not None:
            control_state["cached_object_initial_pose"] = object_initial_pose

    dataset.finalize()
    logger.info(f"All {args.num_episode} episodes recording completed!")
    return object_initial_pose


def run_live_control_without_record(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    debug_pose_state: dict[str, Any],
    control_state: dict[str, Any],
    debug_markers: GarmentKeypointDebugMarkers | None = None,
    debug_marker_state: dict[str, Any] | None = None,
) -> None:
    """Run live teleoperation control without recording."""
    actions = teleop_interface.advance()
    if actions is None:
        current_obs = env._get_observations()
        maintain_action = _get_or_build_maintain_action(env, args, control_state, current_obs)
        env.step(maintain_action)
        if cuda_visual_sync_enabled(env):
            force_cuda_render_sync(env)
        else:
            env.render()
    else:
        env.step(actions)
        force_cuda_render_sync(env)

    update_debug_markers_if_needed(env, debug_markers, debug_marker_state)

    log_debug_pose_snapshot_if_enabled(env, args, debug_pose_state)
    if args.log_success:
        _ = env._get_success()


__all__ = [
    "DEBUG_POSE_LOG_INTERVAL",
    "create_dataset_if_needed",
    "create_debug_markers_if_needed",
    "create_teleop_interface",
    "register_teleop_callbacks",
    "run_idle_phase",
    "run_live_control_without_record",
    "run_recording_phase",
    "update_debug_markers_if_needed",
    "validate_task_and_device",
]
