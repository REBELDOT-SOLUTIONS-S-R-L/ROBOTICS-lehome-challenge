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

from lehome.tasks.fold_cloth.fold_cloth_bi_arm_mimic_env_cfg import (
    configure_subtasks_from_garment_cfg,
)
from lehome.utils.logger import get_logger
from lehome.utils.record import RateLimiter, get_next_experiment_path_with_gap

from ...utils.common import stabilize_garment_after_reset
from .annotated_recording import AnnotatedMimicHDF5Recorder
from .annotated_runtime_snapshot import (
    capture_annotated_runtime_snapshot,
    ensure_return_home_snapshot_fields,
)
from .env_runtime import get_env_garment_metadata
from .online_annotation import OnlineAnnotationState
from .record_debug import (
    DEBUG_POSE_LOG_INTERVAL,
    log_debug_pose_snapshot_if_enabled,
    log_success_result,
)
from .cuda_visual_sync import apply_cuda_fabric_render_settings
from .cuda_visual_sync import force_cuda_render_sync
from .cuda_visual_sync import post_reset_cuda_visual_sync
from .teleop_runtime import (
    create_debug_markers_if_needed,
    create_teleop_interface,
    register_teleop_callbacks,
    update_debug_markers_if_needed,
    validate_task_and_device,
)

logger = get_logger(__name__)


class PoseSequence:
    """Deterministic Halton low-discrepancy sequence over 4D pose space.

    Generates *n* points in (pos_x, pos_y, rot_x, rot_y) using Halton
    sequences with prime bases 2, 3, 5, 7.  Position Z and rotation Z
    are read once from the garment config and held fixed.
    """

    _BASES = (2, 3, 5, 7)

    def __init__(
        self,
        n: int,
        pos_range: list[float],
        rot_range: list[float],
        max_failures: int = 0,
    ) -> None:
        if n < 1:
            raise ValueError(f"--pose_sequence must be >= 1, got {n}")
        self.n = n
        self._max_failures = int(max_failures)
        self._pos_min = (pos_range[0], pos_range[1])
        self._pos_max = (pos_range[3], pos_range[4])
        self._pos_z = pos_range[2]
        self._rot_min = (rot_range[0], rot_range[1])
        self._rot_max = (rot_range[3], rot_range[4])
        self._rot_z = rot_range[2]

        self._sequence: list[tuple[list[float], list[float]]] = []
        for i in range(n):
            h = self._halton_point(i + 1)  # skip index 0 (all-zeros corner)
            pos_x = round(self._pos_min[0] + h[0] * (self._pos_max[0] - self._pos_min[0]), 4)
            pos_y = round(self._pos_min[1] + h[1] * (self._pos_max[1] - self._pos_min[1]), 4)
            rot_x = round(self._rot_min[0] + h[2] * (self._rot_max[0] - self._rot_min[0]), 1)
            rot_y = round(self._rot_min[1] + h[3] * (self._rot_max[1] - self._rot_min[1]), 1)
            self._sequence.append(
                ([pos_x, pos_y, self._pos_z], [rot_x, rot_y, self._rot_z])
            )
        self._index = 0
        self._failures_per_index: list[int] = [0] * n

    @staticmethod
    def _halton_value(index: int, base: int) -> float:
        """Compute the *index*-th element of the van der Corput sequence in *base*."""
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def _halton_point(self, index: int) -> tuple[float, ...]:
        """Return a 4D point in [0,1]^4 for the given 1-based index."""
        return tuple(self._halton_value(index, b) for b in self._BASES)

    @property
    def total(self) -> int:
        return self.n

    @property
    def index(self) -> int:
        return self._index

    @property
    def exhausted(self) -> bool:
        return self._index >= self.n

    def current(self) -> tuple[list[float], list[float]]:
        """Return ``(pos, ori_euler_degrees)`` for the current sequence position.

        If the sequence is exhausted (all indices consumed), return the
        last pose so in-flight worker resets don't IndexError before the
        main loop observes exhaustion and breaks.
        """
        if self._index >= self.n:
            return self._sequence[-1]
        return self._sequence[self._index]

    def advance(self) -> None:
        """Move to the next sequence position (call after a successful save)."""
        self._index += 1

    def record_failure(self, reason: str) -> None:
        """Record a failure against the current Halton index and log it.

        If ``max_failures`` is set and exceeded, skip to the next Halton
        pose so generation doesn't spin forever on an unreachable pose.
        """
        if self.exhausted:
            return
        current_index = self._index
        self._failures_per_index[current_index] += 1
        count = self._failures_per_index[current_index]
        logger.info(
            "[PoseSequence] Halton index %d/%d failure #%d (%s)",
            current_index,
            self.n,
            count,
            reason,
        )
        if self._max_failures > 0 and count >= self._max_failures:
            logger.warning(
                "[PoseSequence] Halton index %d/%d hit failure cap (%d); skipping to next pose.",
                current_index,
                self.n,
                self._max_failures,
            )
            self.advance()
            self.log_status()

    def failures_at(self, index: int) -> int:
        return self._failures_per_index[index]

    def log_status(self) -> None:
        if self.exhausted:
            logger.info("[PoseSequence] All %d poses exhausted.", self.n)
            return
        pos, ori = self._sequence[self._index]
        logger.info(
            "[PoseSequence] Index %d/%d | pos=(%.2f, %.2f) rot=(%.2f, %.2f) | prior failures at this index: %d",
            self._index, self.n, pos[0], pos[1], ori[0], ori[1],
            self._failures_per_index[self._index],
        )


def _install_pose_sequence_override(env: DirectRLEnv, seq: PoseSequence) -> None:
    """Monkey-patch the garment's ``_sample_reset_pose`` to return deterministic poses."""
    garment_obj = getattr(env, "object", None)
    if garment_obj is None:
        raise RuntimeError("Cannot install pose sequence: env has no garment object.")

    def _sequence_sample_reset_pose():
        pos, ori = seq.current()
        logger.debug("[PoseSequence] Overriding reset pose to pos=%s ori=%s", pos, ori)
        return pos, ori

    garment_obj._sample_reset_pose = _sequence_sample_reset_pose
    logger.info("[PoseSequence] Installed deterministic pose override on garment object.")


def _build_pose_sequence(args: argparse.Namespace, env: DirectRLEnv) -> PoseSequence | None:
    """Build a PoseSequence from CLI args if --pose_sequence is set."""
    n = getattr(args, "pose_sequence", None)
    if n is None:
        return None

    garment_obj = getattr(env, "object", None)
    if garment_obj is None:
        raise RuntimeError("Cannot build pose sequence: env has no garment object.")

    pos_range, _ = garment_obj._get_config_value("soft_reset_pos_range", "common")
    rot_range, _ = garment_obj._get_config_value("soft_reset_rot_range", "common")
    max_failures = int(getattr(args, "pose_sequence_max_failures", 0) or 0)
    seq = PoseSequence(int(n), pos_range, rot_range, max_failures=max_failures)
    logger.info("[PoseSequence] Created %d-point Halton sequence (bases 2,3,5,7)", n)
    if max_failures > 0:
        logger.info("[PoseSequence] Per-index failure cap: %d (skips pose on exceed).", max_failures)
    else:
        logger.info("[PoseSequence] Per-index failure cap: disabled (retry forever).")
    logger.info(
        "[PoseSequence] Pos range: x=[%.2f, %.2f] y=[%.2f, %.2f] z=%.2f (fixed)",
        pos_range[0], pos_range[3], pos_range[1], pos_range[4], pos_range[2],
    )
    logger.info(
        "[PoseSequence] Rot range: x=[%.2f, %.2f] y=[%.2f, %.2f] z=%.2f (fixed)",
        rot_range[0], rot_range[3], rot_range[1], rot_range[4], rot_range[2],
    )
    _install_pose_sequence_override(env, seq)
    return seq


_SIGNAL_LABELS = {
    "prepare_for_grasp_left_middle": "approach left middle",
    "prepare_for_grasp_right_middle": "approach right middle",
    "grasp_left_middle": "grasp left middle",
    "grasp_right_middle": "grasp right middle",
    "left_middle_to_lower": "move left middle to lower",
    "right_middle_to_lower": "move right middle to lower",
    "left_at_waiting_pos": "left arm to waiting position",
    "right_at_waiting_pos": "right arm to waiting position",
    "prepare_for_grasp_left_lower": "approach left lower",
    "prepare_for_grasp_right_lower": "approach right lower",
    "grasp_left_lower": "grasp left lower",
    "grasp_right_lower": "grasp right lower",
    "left_lower_to_upper": "move left lower to upper",
    "right_lower_to_upper": "move right lower to upper",
    "prepare_for_grasp_left_upper": "approach left upper",
    "prepare_for_grasp_left_on_right_upper": "approach right upper with left arm",
    "prepare_for_grasp_right_on_left_lower": "approach left lower with right arm",
    "grasp_left_upper": "grasp left upper",
    "grasp_left_on_right_upper": "grasp right upper with left arm",
    "grasp_right_on_left_lower": "grasp left lower with right arm",
    "left_upper_to_right_upper": "move left upper to right upper",
    "left_lower_to_right_lower": "move left lower to right lower",
    "right_upper_to_left_upper": "move right upper to left upper",
    "right_lower_to_left_lower": "move right lower to left lower",
    "release_left_upper_at_right_upper": "release left upper at right upper",
    "release_left_lower_at_right_lower": "release left lower at right lower",
    "release_right_upper_at_left_upper": "release right upper at left upper",
    "release_right_lower_at_left_lower": "release right lower at left lower",
    "left_return_home": "return left arm home",
    "right_return_home": "return right arm home",
}
_ARM_LABELS = {
    "left_arm": "Left arm",
    "right_arm": "Right arm",
}
_RETURN_HOME_SIGNALS = {"left_return_home", "right_return_home"}


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


def _capture_camera_frames(env: DirectRLEnv) -> dict[str, torch.Tensor] | None:
    camera_frames: dict[str, torch.Tensor] = {}

    def _maybe_add(sensor_name: str, target_key: str) -> None:
        try:
            sensor = env.scene[sensor_name]
            rgb = sensor.data.output["rgb"]
        except Exception:
            return
        if rgb is None:
            return
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
            return
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        camera_frames[target_key] = rgb.clone()

    _maybe_add("top_camera", "top")
    _maybe_add("left_camera", "left_wrist")
    _maybe_add("right_camera", "right_wrist")
    return camera_frames or None


def _create_rate_limiter(args: argparse.Namespace) -> RateLimiter | None:
    step_hz = int(getattr(args, "step_hz", 0) or 0)
    if step_hz <= 0:
        return None
    return RateLimiter(step_hz)


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

    if head_signal in {
        "grasp_left_middle",
        "grasp_right_middle",
        "grasp_left_upper",
        "grasp_left_on_right_upper",
        "grasp_right_on_left_lower",
        "grasp_left_lower",
        "grasp_right_lower",
    }:
        checkpoint_name = (
            "garment_left_middle"
            if head_signal == "grasp_left_middle"
            else "garment_right_middle"
            if head_signal == "grasp_right_middle"
            else "garment_left_upper"
            if head_signal == "grasp_left_upper"
            else "garment_right_upper"
            if head_signal == "grasp_left_on_right_upper"
            else "garment_left_lower"
            if head_signal == "grasp_right_on_left_lower"
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
        point_name = "garment_left_middle" if head_signal == "left_middle_to_lower" else "garment_right_middle"
        middle_z = context.semantic_keypoints_world[point_name][..., 2:3]
        z_limit_cm = context.middle_to_lower_middle_keypoint_max_z_m * 100.0
        return (
            f"{arm_label}: waiting to {signal_label}. "
            f"Gripper open: {'no' if _bool_text(context.gripper_closed_by_arm[arm_name]) == 'yes' else 'yes'}. "
            f"Middle keypoint z: {_cm_text(middle_z)} (need < {z_limit_cm:.1f} cm). "
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
            f"Gripper open: {'no' if _bool_text(context.gripper_closed_by_arm[arm_name]) == 'yes' else 'yes'}. "
            f"Keypoint distance: {_cm_text(distance)} (need <= {distance_limit_cm:.1f} cm). "
            f"Dwell: {dwell}/{required}."
        )

    if head_signal in {"left_at_waiting_pos", "right_at_waiting_pos"}:
        return (
            f"{arm_label}: waiting to {signal_label}. "
            f"At waiting pos: {_bool_text(context.arm_at_waiting_pos_by_arm[arm_name])}. "
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


def _current_return_home_arms(annotator: OnlineAnnotationState) -> tuple[str, ...]:
    return tuple(
        arm_name
        for arm_name, signal_name in annotator.current_signal_heads().items()
        if signal_name in _RETURN_HOME_SIGNALS
    )


def _log_return_home_success_if_both_arms_at_rest(
    env: DirectRLEnv,
    annotator: OnlineAnnotationState,
    snapshot,
    episode_index: int,
    episode_step_count: int,
    already_logged: bool,
) -> bool:
    """Emit one fold-success snapshot once both arms are at rest during return-home."""
    if already_logged:
        return True

    current_heads = annotator.current_signal_heads()
    if current_heads.get("left_arm") != "left_return_home":
        return False
    if current_heads.get("right_arm") != "right_return_home":
        return False

    context = snapshot.observation_context
    left_at_rest = bool(
        torch.as_tensor(context.arm_at_rest_by_arm.get("left_arm", False)).reshape(-1)[0].item()
    )
    right_at_rest = bool(
        torch.as_tensor(context.arm_at_rest_by_arm.get("right_arm", False)).reshape(-1)[0].item()
    )
    if not (left_at_rest and right_at_rest):
        return False

    logger.info(
        "[Annotated Recording][Episode %d][step %d] Both arms are at rest in the final subtask. "
        "Running fold success check once.",
        episode_index,
        episode_step_count,
    )
    log_success_result(
        env,
        episode_index=episode_index,
        step_in_episode=episode_step_count,
        context="both_arms_at_rest_in_return_home",
    )
    return True


def _reset_environment_for_next_attempt(
    env: DirectRLEnv,
    args: argparse.Namespace,
    debug_markers=None,
    debug_marker_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    env.reset()
    stabilize_garment_after_reset(env, args)
    post_reset_cuda_visual_sync(env)
    update_debug_markers_if_needed(
        env,
        debug_markers,
        debug_marker_state,
        force=True,
    )
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
    apply_cuda_fabric_render_settings(env_cfg, device, context="annotated teleop")
    env_cfg.task_type = args.teleop_device
    env_cfg.garment_name = args.garment_name
    env_cfg.garment_version = args.garment_version
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path
    garment_type = configure_subtasks_from_garment_cfg(env_cfg)
    logger.info(
        "[Annotated Recording] Loaded %s subtask builder for garment %s.",
        garment_type,
        args.garment_name,
    )
    subtask_cfgs = getattr(env_cfg, "subtask_configs", {})
    if isinstance(subtask_cfgs, dict):
        for arm_name, cfgs in sorted(subtask_cfgs.items()):
            queue = [
                str(getattr(cfg, "subtask_term_signal", ""))
                for cfg in cfgs
                if getattr(cfg, "subtask_term_signal", None) is not None
            ]
            logger.info(
                "[Annotated Recording] %s queue: %s",
                arm_name,
                " -> ".join(queue) if queue else "<empty>",
            )
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
            "Camera frames will be saved in the annotated HDF5 and FPS may drop."
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
    debug_markers = None
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
        debug_markers = create_debug_markers_if_needed(args)

        pose_seq = _build_pose_sequence(args, env)

        recorder = AnnotatedMimicHDF5Recorder(
            file_path=_resolve_output_hdf5_path(getattr(args, "dataset_root", "Datasets/record")),
            env_args=_build_env_args(args),
            fps=int(getattr(args, "step_hz", 90)),
            episode_capacity=int(env.max_episode_length) + 1,
            record_camera_streams=bool(getattr(args, "enable_cameras", False)),
        )
        rate_limiter = _create_rate_limiter(args)
        annotator = OnlineAnnotationState.from_env(env)
        debug_pose_state: dict[str, Any] = {"step_count": 0, "eef_body_idx_cache": {}}
        debug_marker_state: dict[str, Any] = {"step_count": 0}
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
                    update_debug_markers_if_needed(
                        env,
                        debug_markers,
                        debug_marker_state,
                        force=True,
                    )
                    logger.info("[Idle Phase] Stabilizing garment after initialization...")
                    stabilize_garment_after_reset(env, args)
                    post_reset_cuda_visual_sync(env)
                    update_debug_markers_if_needed(
                        env,
                        debug_markers,
                        debug_marker_state,
                        force=True,
                    )
                    cached_object_initial_pose = _safe_get_all_pose(env)
                    logger.info("[Idle Phase] Ready for annotated recording")
                    if pose_seq is not None:
                        pose_seq.log_status()
                    initialized = True
                    continue

                if pose_seq is not None:
                    if pose_seq.exhausted:
                        logger.info(
                            f"[PoseSequence] All {pose_seq.total} poses completed!"
                        )
                        break
                elif episode_index >= int(args.num_episode):
                    logger.info(
                        f"All {int(args.num_episode)} annotated episodes recording completed!"
                    )
                    break

                if not flags["start"]:
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
                    force_cuda_render_sync(env)
                    update_debug_markers_if_needed(env, debug_markers, debug_marker_state)
                    log_debug_pose_snapshot_if_enabled(env, args, debug_pose_state)
                    if rate_limiter is not None:
                        rate_limiter.sleep(env)
                    continue

                flags["success"] = False
                flags["remove"] = False
                annotator.reset()
                annotation_complete_logged = False
                return_home_success_logged = False
                garment_name, scale = get_env_garment_metadata(env)
                if cached_object_initial_pose is None:
                    cached_object_initial_pose = _safe_get_all_pose(env)

                recorder.begin_episode(
                    episode_index=episode_index,
                    object_initial_pose=cached_object_initial_pose,
                    garment_name=garment_name,
                    scale=scale,
                    signal_names=tuple(annotator.latched_signals.keys()),
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
                            debug_markers,
                            debug_marker_state,
                        )
                        flags["remove"] = False
                        break

                    if flags["success"]:
                        if annotator.is_complete():
                            if args.log_success and not return_home_success_logged:
                                log_success_result(
                                    env,
                                    episode_index=episode_index,
                                    step_in_episode=episode_step_count,
                                    context="save_requested",
                                )
                            logger.info(f"[Annotated Recording] Saving episode {episode_index}...")
                            recorder.finalize_episode(env)
                            episode_index += 1
                            if pose_seq is not None:
                                pose_seq.advance()
                                total = pose_seq.total
                            else:
                                total = int(args.num_episode)
                            logger.info(
                                f"Annotated episode saved, progress: {episode_index}/{total}"
                            )
                            if pose_seq is not None and not pose_seq.exhausted:
                                pose_seq.log_status()
                        else:
                            if args.log_success and not return_home_success_logged:
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
                                rest_pose_arms=_current_return_home_arms(annotator),
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

                        if pose_seq is not None and pose_seq.exhausted:
                            cached_object_initial_pose = None
                        else:
                            cached_object_initial_pose = _reset_environment_for_next_attempt(
                                env,
                                args,
                                debug_markers,
                                debug_marker_state,
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
                    force_cuda_render_sync(env)
                    update_debug_markers_if_needed(env, debug_markers, debug_marker_state)
                    episode_step_count += 1
                    log_debug_pose_snapshot_if_enabled(env, args, debug_pose_state)
                    snapshot = capture_annotated_runtime_snapshot(
                        env,
                        action,
                        include_fold_success=annotator.needs_fold_success(),
                        rest_pose_arms=_current_return_home_arms(annotator),
                    )
                    return_home_success_logged = _log_return_home_success_if_both_arms_at_rest(
                        env,
                        annotator,
                        snapshot,
                        episode_index,
                        episode_step_count,
                        return_home_success_logged,
                    )
                    newly_latched = annotator.advance_from_context(snapshot.observation_context)
                    if newly_latched:
                        if annotator.needs_fold_success() and snapshot.observation_context.fold_success is None:
                            snapshot = ensure_return_home_snapshot_fields(
                                env,
                                snapshot,
                                _current_return_home_arms(annotator),
                        )
                        logger.info(
                            "[Annotated Recording][Episode %d][step %d] Latched: %s",
                            episode_index,
                            episode_step_count,
                            ", ".join(_format_signal_label(signal_name) for signal_name in newly_latched),
                        )
                        _log_annotation_progress(annotator, episode_index, episode_step_count)

                    if annotator.is_complete() and not annotation_complete_logged:
                        if not return_home_success_logged:
                            log_success_result(
                                env,
                                episode_index=episode_index,
                                step_in_episode=episode_step_count,
                                context="annotation_complete",
                            )
                        logger.info(
                            "[Annotated Recording] All subtask queues completed. "
                            "Press N to save the episode or D to re-record."
                        )
                        annotation_complete_logged = True

                    recorder.append_step(
                        checkpoint_positions=snapshot.checkpoint_positions,
                        eef_pose_components=snapshot.eef_pose_components,
                        subtask_term_signals=annotator.as_bool_dict(),
                        gripper_actions=snapshot.gripper_actions,
                        joint_actions=snapshot.joint_actions,
                        joint_pos=snapshot.joint_pos,
                        camera_frames=_capture_camera_frames(env)
                        if bool(getattr(args, "enable_cameras", False))
                        else None,
                    )
                    if rate_limiter is not None:
                        rate_limiter.sleep(env)

                    terminated, truncated = env._get_dones()
                    done = bool(torch.any(terminated).item() or torch.any(truncated).item())
                    if done:
                        if args.log_success and not return_home_success_logged:
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
                            debug_markers,
                            debug_marker_state,
                        )
                        break
    except KeyboardInterrupt:
        logger.warning("\n[Ctrl+C] Interrupt signal detected")
    finally:
        if debug_markers is not None:
            debug_markers.close()
        if "recorder" in locals() and recorder is not None:
            recorder.finalize()
        if env is not None:
            env.close()


__all__ = ["record_dataset"]
