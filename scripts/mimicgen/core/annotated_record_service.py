"""Integrated teleop + online annotation service for generation-ready Mimic HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaacsim.simulation_app import SimulationApp

from lehome.utils.logger import get_logger
from lehome.utils.record import get_next_experiment_path_with_gap

from ...utils.common import stabilize_garment_after_reset
from .annotated_recording import AnnotatedMimicHDF5Recorder
from .annotated_runtime_snapshot import capture_annotated_runtime_snapshot
from .env_runtime import get_env_garment_metadata
from .online_annotation import OnlineAnnotationState
from .record_debug import (
    DEBUG_POSE_LOG_INTERVAL,
    log_debug_pose_snapshot_if_enabled,
    log_success_result,
)
from .teleop_runtime import (
    create_teleop_interface,
    register_teleop_callbacks,
    validate_task_and_device,
)

logger = get_logger(__name__)

_SIGNAL_LABELS = {
    "grasp_left_middle": "grasp left middle",
    "grasp_right_middle": "grasp right middle",
    "left_middle_to_lower": "move left middle to lower",
    "right_middle_to_lower": "move right middle to lower",
    "grasp_left_lower": "grasp left lower",
    "grasp_right_lower": "grasp right lower",
    "left_lower_to_upper": "move left lower to upper",
    "right_lower_to_upper": "move right lower to upper",
    "left_return_home": "return left arm home",
    "right_return_home": "return right arm home",
}
_ARM_LABELS = {
    "left_arm": "Left arm",
    "right_arm": "Right arm",
}


def _safe_get_all_pose(env: DirectRLEnv) -> dict[str, Any] | None:
    try:
        return env.get_all_pose()
    except Exception as exc:
        logger.error(f"[Annotated Recording] Failed to get initial pose: {exc}", exc_info=True)
        return None


def _resolve_output_hdf5_path(dataset_root: str) -> Path:
    root_path = Path(dataset_root)
    if root_path.suffix.lower() in {".hdf5", ".h5"}:
        return root_path
    run_dir = get_next_experiment_path_with_gap(root_path)
    return run_dir / "annotated_dataset.hdf5"


def _build_env_args(args: argparse.Namespace) -> dict[str, Any]:
    env_args: dict[str, Any] = {"env_name": args.task or "", "type": 2}
    if getattr(args, "garment_name", None):
        env_args["garment_name"] = args.garment_name
    if getattr(args, "garment_version", None):
        env_args["garment_version"] = args.garment_version
    if getattr(args, "task_description", None):
        env_args["task_description"] = args.task_description
    return env_args


def _resolve_runtime_task(task_name: str) -> str:
    task_name = str(task_name or "")
    if task_name.endswith("-Mimic-v0"):
        return task_name.replace("-Mimic-v0", "-v0")
    return task_name


def _normalize_action_tensor(env: DirectRLEnv, action: Any) -> torch.Tensor:
    if action is None:
        raise ValueError("Action cannot be None when normalizing teleop step.")
    action_tensor = torch.as_tensor(action, device=env.device, dtype=torch.float32)
    if action_tensor.ndim == 1:
        action_tensor = action_tensor.unsqueeze(0)
    if action_tensor.ndim != 2:
        raise ValueError(f"Expected action rank 2 after normalization, got {tuple(action_tensor.shape)}.")
    return action_tensor


def _get_hold_action(env: DirectRLEnv) -> torch.Tensor:
    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None and hasattr(action_manager, "action"):
        try:
            action = _normalize_action_tensor(env, action_manager.action)
            if action.shape[0] == 1:
                return action.clone()
        except Exception:
            pass

    action_dim = 16
    if action_manager is not None and hasattr(action_manager, "total_action_dim"):
        try:
            action_dim = int(action_manager.total_action_dim)
        except Exception:
            action_dim = 16
    return torch.zeros((1, action_dim), device=env.device, dtype=torch.float32)


def _format_signal_label(signal_name: str | None) -> str:
    if signal_name is None:
        return "complete"
    return _SIGNAL_LABELS.get(str(signal_name), str(signal_name).replace("_", " "))


def _bool_text(value: Any) -> str:
    if value is None:
        return "unknown"
    return "yes" if bool(torch.as_tensor(value).reshape(-1)[0].item()) else "no"


def _cm_text(distance_m: Any) -> str:
    value = float(torch.as_tensor(distance_m).reshape(-1)[0].item()) * 100.0
    return f"{value:.1f} cm"


def _describe_head_status(
    annotator: OnlineAnnotationState,
    arm_name: str,
    snapshot,
) -> str:
    head_signal = annotator.current_signal_heads().get(arm_name)
    queue = annotator.arm_queues.get(arm_name, [])
    queue_index = int(annotator.arm_queue_indices.get(arm_name, 0))
    arm_label = _ARM_LABELS.get(arm_name, arm_name)
    if head_signal is None:
        return f"{arm_label}: complete ({queue_index}/{len(queue)} subtasks)."

    context = snapshot.observation_context
    dwell = int(annotator.consecutive_true_counts.get(arm_name, 0))
    required = int(annotator._required_dwell(head_signal))
    signal_label = _format_signal_label(head_signal)

    if head_signal in {"grasp_left_middle", "grasp_right_middle", "grasp_left_lower", "grasp_right_lower"}:
        checkpoint_name = (
            "garment_left_middle"
            if head_signal == "grasp_left_middle"
            else "garment_right_middle"
            if head_signal == "grasp_right_middle"
            else "garment_left_lower"
            if head_signal == "grasp_left_lower"
            else "garment_right_lower"
        )
        eef_pos = context.eef_world_positions[arm_name]
        kp_pos = context.semantic_keypoints_world[checkpoint_name]
        distance = torch.linalg.norm(eef_pos - kp_pos, dim=-1, keepdim=True)
        distance_limit_cm = context.grasp_eef_to_keypoint_threshold_m * 100.0
        return (
            f"{arm_label}: waiting to {signal_label}. "
            f"Opened at keypoint first: {'yes' if annotator.grasp_open_ready_by_arm.get(arm_name, False) else 'no'}. "
            f"Gripper closed: {_bool_text(context.gripper_closed_by_arm[arm_name])}. "
            f"EEF to keypoint: {_cm_text(distance)} (need <= {distance_limit_cm:.1f} cm). "
            f"Dwell: {dwell}/{required}."
        )

    if head_signal in {"left_middle_to_lower", "right_middle_to_lower"}:
        point_a = "garment_left_middle" if head_signal == "left_middle_to_lower" else "garment_right_middle"
        point_b = "garment_left_lower" if head_signal == "left_middle_to_lower" else "garment_right_lower"
        distance = torch.linalg.norm(
            context.semantic_keypoints_world[point_a] - context.semantic_keypoints_world[point_b],
            dim=-1,
            keepdim=True,
        )
        distance_limit_cm = context.middle_to_lower_threshold_m * 100.0
        return (
            f"{arm_label}: waiting to {signal_label}. "
            f"Keypoint distance: {_cm_text(distance)} (need <= {distance_limit_cm:.1f} cm). "
            f"Dwell: {dwell}/{required}."
        )

    if head_signal in {"left_lower_to_upper", "right_lower_to_upper"}:
        point_a = "garment_left_lower" if head_signal == "left_lower_to_upper" else "garment_right_lower"
        point_b = "garment_left_upper" if head_signal == "left_lower_to_upper" else "garment_right_upper"
        distance = torch.linalg.norm(
            context.semantic_keypoints_world[point_a] - context.semantic_keypoints_world[point_b],
            dim=-1,
            keepdim=True,
        )
        distance_limit_cm = context.lower_to_upper_threshold_m * 100.0
        return (
            f"{arm_label}: waiting to {signal_label}. "
            f"Keypoint distance: {_cm_text(distance)} (need <= {distance_limit_cm:.1f} cm). "
            f"Dwell: {dwell}/{required}."
        )

    if head_signal in {"left_return_home", "right_return_home"}:
        return (
            f"{arm_label}: waiting to {signal_label}. "
            f"Fold success: {_bool_text(context.fold_success)}. "
            f"Arm at rest: {_bool_text(context.arm_at_rest_by_arm[arm_name])}. "
            f"Dwell: {dwell}/{required}."
        )

    return (
        f"{arm_label}: waiting for {_format_signal_label(head_signal)}. "
        f"Dwell: {dwell}/{required}."
    )


def _log_annotation_progress(
    annotator: OnlineAnnotationState,
    episode_index: int,
    episode_step_count: int,
) -> None:
    """Log compact queue progress for operator-facing annotation updates."""
    prefix = f"[Annotated Recording][Episode {episode_index}][step {episode_step_count}]"
    for arm_name in sorted(annotator.current_signal_heads().keys()):
        queue = annotator.arm_queues.get(arm_name, [])
        completed = int(annotator.arm_queue_indices.get(arm_name, 0))
        total = len(queue)
        next_signal = annotator.current_signal_heads().get(arm_name)
        arm_label = _ARM_LABELS.get(arm_name, arm_name)
        next_label = _format_signal_label(next_signal) if next_signal is not None else "complete"
        logger.info(
            "%s %s: %d/%d complete | next: %s",
            prefix,
            arm_label,
            completed,
            total,
            next_label,
        )


def _log_incomplete_episode_summary(
    annotator: OnlineAnnotationState,
    episode_index: int,
    episode_step_count: int,
    snapshot=None,
) -> None:
    """Log pending queue items before discarding an incomplete episode."""
    prefix = f"[Annotated Recording][Episode {episode_index}][step {episode_step_count}]"
    pending_parts: list[str] = []
    for arm_name, queue in annotator.arm_queues.items():
        queue_index = int(annotator.arm_queue_indices.get(arm_name, 0))
        pending = queue[queue_index:]
        pending_labels = ", ".join(_format_signal_label(signal) for signal in pending) if pending else "none"
        pending_parts.append(
            f"{_ARM_LABELS.get(arm_name, arm_name)} pending: {pending_labels}"
        )
    logger.warning("%s Episode discarded because annotation queues were incomplete.", prefix)
    for part in pending_parts:
        logger.warning("%s %s", prefix, part)
    if snapshot is not None:
        for arm_name in sorted(annotator.current_signal_heads().keys()):
            logger.warning("%s %s", prefix, _describe_head_status(annotator, arm_name, snapshot))


def _reset_environment_for_next_attempt(
    env: DirectRLEnv,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    env.reset()
    stabilize_garment_after_reset(env, args)
    return _safe_get_all_pose(env)


def record_dataset(args: argparse.Namespace, simulation_app: SimulationApp) -> None:
    """Record online-annotated generation-ready Mimic demonstrations."""
    validate_task_and_device(args)

    device = getattr(args, "device", "cpu")
    runtime_task = _resolve_runtime_task(args.task)
    if runtime_task != args.task:
        logger.info(
            "Annotated teleop uses the base fold-cloth env for 12D control. "
            f"Runtime task switched from {args.task} to {runtime_task}."
        )

    env_cfg = parse_env_cfg(runtime_task, device=device)
    env_cfg.task_type = args.teleop_device
    env_cfg.garment_name = args.garment_name
    env_cfg.garment_version = args.garment_version
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path
    if not bool(getattr(args, "enable_cameras", False)) and hasattr(env_cfg, "scene"):
        for camera_name in ("left_camera", "right_camera", "top_camera"):
            if hasattr(env_cfg.scene, camera_name):
                setattr(env_cfg.scene, camera_name, None)
        logger.info(
            "[Annotated Recording] Cameras disabled by default for record_annotated. "
            "Pass --enable_cameras to opt in."
        )
    elif bool(getattr(args, "enable_cameras", False)):
        logger.info(
            "[Annotated Recording] Cameras explicitly enabled. "
            "This mode does not save camera data and FPS may drop."
        )
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
        logger.info(
            "[Annotated Recording] Disabled success-based auto-reset so return-home "
            "annotation can finish after the cloth is folded."
        )

    if args.use_random_seed:
        env_cfg.use_random_seed = True
        logger.info("Using random seed (no fixed seed)")
    else:
        env_cfg.use_random_seed = False
        env_cfg.random_seed = args.seed
        logger.info(f"Using fixed random seed: {args.seed}")

    env: DirectRLEnv | None = gym.make(runtime_task, cfg=env_cfg).unwrapped
    try:
        action_manager = getattr(env, "action_manager", None)
        action_dim = int(getattr(action_manager, "total_action_dim", 0))
        if action_dim not in {12, 16}:
            raise ValueError(
                f"Annotated teleop expected a fold-cloth action contract of 12 or 16, got {action_dim}."
            )

        teleop_interface = create_teleop_interface(env, args)
        flags = register_teleop_callbacks(teleop_interface, recording_enabled=True)
        teleop_interface.reset()

        if getattr(args, "debugging_markers", False):
            logger.info(
                "[Annotated Recording] Ignoring --debugging_markers for record_annotated "
                "to preserve teleop FPS."
            )

        recorder = AnnotatedMimicHDF5Recorder(
            file_path=_resolve_output_hdf5_path(getattr(args, "dataset_root", "Datasets/record")),
            env_args=_build_env_args(args),
            fps=int(getattr(args, "step_hz", 90)),
        )
        annotator = OnlineAnnotationState.from_env(env)
        debug_pose_state: dict[str, Any] = {"step_count": 0, "eef_body_idx_cache": {}}
        cached_object_initial_pose: dict[str, Any] | None = None
        initialized = False
        episode_index = 0

        if getattr(args, "debugging_log_pose", False):
            logger.info(
                "[Debug Pose] Enabled. Logging EEF and garment checkpoint positions in cm "
                f"at step 0 and every {DEBUG_POSE_LOG_INTERVAL} sim steps."
            )

        while simulation_app.is_running():
            with torch.inference_mode():
                if not initialized:
                    logger.info("[Idle Phase] Initializing observations...")
                    env.initialize_obs()
                    logger.info("[Idle Phase] Stabilizing garment after initialization...")
                    stabilize_garment_after_reset(env, args)
                    cached_object_initial_pose = _safe_get_all_pose(env)
                    logger.info("[Idle Phase] Ready for annotated recording")
                    initialized = True
                    continue

                if episode_index >= int(args.num_episode):
                    logger.info(
                        f"All {int(args.num_episode)} annotated episodes recording completed!"
                    )
                    break

                if not flags["start"]:
                    input_action = teleop_interface.advance()
                    action = (
                        _get_hold_action(env)
                        if input_action is None
                        else _normalize_action_tensor(env, input_action)
                    )
                    env.step(action)
                    log_debug_pose_snapshot_if_enabled(env, args, debug_pose_state)
                    continue

                flags["success"] = False
                flags["remove"] = False
                annotator.reset()
                annotation_complete_logged = False
                garment_name, scale = get_env_garment_metadata(env)
                if cached_object_initial_pose is None:
                    cached_object_initial_pose = _safe_get_all_pose(env)

                recorder.begin_episode(
                    episode_index=episode_index,
                    object_initial_pose=cached_object_initial_pose,
                    garment_name=garment_name,
                    scale=scale,
                )

                episode_step_count = 0
                _log_annotation_progress(annotator, episode_index, episode_step_count)
                if args.log_success:
                    log_success_result(
                        env,
                        episode_index=episode_index,
                        step_in_episode=episode_step_count,
                        context="episode_start",
                    )

                while simulation_app.is_running():
                    if flags["abort"]:
                        recorder.discard_episode()
                        recorder.finalize()
                        logger.warning(
                            f"Annotated recording aborted, completed {episode_index} episodes"
                        )
                        return

                    if flags["remove"]:
                        recorder.discard_episode()
                        logger.info(f"Re-recording annotated episode {episode_index}")
                        cached_object_initial_pose = _reset_environment_for_next_attempt(
                            env,
                            args,
                        )
                        flags["remove"] = False
                        break

                    if flags["success"]:
                        if annotator.is_complete():
                            if args.log_success:
                                log_success_result(
                                    env,
                                    episode_index=episode_index,
                                    step_in_episode=episode_step_count,
                                    context="save_requested",
                                )
                            logger.info(f"[Annotated Recording] Saving episode {episode_index}...")
                            recorder.finalize_episode(env)
                            episode_index += 1
                            logger.info(
                                f"Annotated episode saved, progress: {episode_index}/{int(args.num_episode)}"
                            )
                        else:
                            if args.log_success:
                                log_success_result(
                                    env,
                                    episode_index=episode_index,
                                    step_in_episode=episode_step_count,
                                    context="discard_incomplete",
                                )
                            snapshot = capture_annotated_runtime_snapshot(
                                env,
                                _get_hold_action(env),
                                include_fold_success=annotator.needs_fold_success(),
                            )
                            _log_incomplete_episode_summary(
                                annotator,
                                episode_index=episode_index,
                                episode_step_count=episode_step_count,
                                snapshot=snapshot,
                            )
                            logger.warning(
                                "[Annotated Recording] Episode ended before all subtasks were latched. "
                                "Discarding incomplete episode."
                            )
                            recorder.discard_episode()

                        cached_object_initial_pose = _reset_environment_for_next_attempt(
                            env,
                            args,
                        )
                        flags["success"] = False
                        break

                    try:
                        input_action = teleop_interface.advance()
                    except Exception as exc:
                        logger.error(
                            f"[Annotated Recording] Error in teleop interface: {exc}",
                            exc_info=True,
                        )
                        input_action = None

                    action = (
                        _get_hold_action(env)
                        if input_action is None
                        else _normalize_action_tensor(env, input_action)
                    )
                    env.step(action)
                    episode_step_count += 1
                    log_debug_pose_snapshot_if_enabled(env, args, debug_pose_state)
                    snapshot = capture_annotated_runtime_snapshot(
                        env,
                        action,
                        include_fold_success=annotator.needs_fold_success(),
                    )
                    newly_latched = annotator.advance_from_context(snapshot.observation_context)
                    if newly_latched:
                        if annotator.needs_fold_success() and snapshot.observation_context.fold_success is None:
                            snapshot = capture_annotated_runtime_snapshot(
                                env,
                                action,
                                include_fold_success=True,
                            )
                        logger.info(
                            "[Annotated Recording][Episode %d][step %d] Latched: %s",
                            episode_index,
                            episode_step_count,
                            ", ".join(_format_signal_label(signal_name) for signal_name in newly_latched),
                        )
                        _log_annotation_progress(annotator, episode_index, episode_step_count)

                    if annotator.is_complete() and not annotation_complete_logged:
                        logger.info(
                            "[Annotated Recording] All subtask queues completed. "
                            "Press N to save the episode or D to re-record."
                        )
                        annotation_complete_logged = True

                    recorder.append_step(
                        checkpoint_positions=snapshot.checkpoint_positions,
                        eef_pose=snapshot.eef_pose,
                        subtask_term_signals=annotator.as_bool_dict(),
                        gripper_actions=snapshot.gripper_actions,
                        joint_actions=snapshot.joint_actions,
                        joint_pos=snapshot.joint_pos,
                        joint_vel=snapshot.joint_vel,
                    )

                    terminated, truncated = env._get_dones()
                    done = bool(torch.any(terminated).item() or torch.any(truncated).item())
                    if done:
                        if args.log_success:
                            log_success_result(
                                env,
                                episode_index=episode_index,
                                step_in_episode=episode_step_count,
                                context="terminated",
                            )
                        _log_incomplete_episode_summary(
                            annotator,
                            episode_index=episode_index,
                            episode_step_count=episode_step_count,
                            snapshot=snapshot,
                        )
                        recorder.discard_episode()
                        logger.warning(
                            "[Annotated Recording] Episode terminated/truncated before save. "
                            "Discarding current attempt."
                        )
                        cached_object_initial_pose = _reset_environment_for_next_attempt(
                            env,
                            args,
                        )
                        break
    except KeyboardInterrupt:
        logger.warning("\n[Ctrl+C] Interrupt signal detected")
    finally:
        if "recorder" in locals() and recorder is not None:
            recorder.finalize()
        if env is not None:
            env.close()


__all__ = ["record_dataset"]
