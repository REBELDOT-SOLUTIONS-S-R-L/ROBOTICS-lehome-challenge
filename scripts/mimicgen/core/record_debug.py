"""Success and pose-debug helpers for MimicGen teleoperation recording."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv

from lehome.tasks.fold_cloth.checkpoint_mappings import ARM_KEYPOINT_GROUPS, CHECKPOINT_LABELS
from lehome.utils.logger import get_logger

from .data_utils import as_numpy
from .recording import (
    _get_scene_articulation,
    _get_single_arm_candidates,
    _resolve_eef_body_idx,
)

logger = get_logger(__name__)

SUCCESS_LOG_INTERVAL = 50
DEBUG_POSE_LOG_INTERVAL = 50
GARMENT_CHECKPOINT_LABELS = CHECKPOINT_LABELS
ARM_KEYPOINT_DISTANCE_LABELS = ARM_KEYPOINT_GROUPS


def _evaluate_success_result(env: DirectRLEnv) -> dict[str, Any] | None:
    """Evaluate garment success without relying on the environment's internal log throttle."""
    if (
        hasattr(env, "object")
        and env.object is not None
        and hasattr(env.object, "_cloth_prim_view")
        and hasattr(env, "garment_loader")
        and hasattr(env, "cfg")
        and hasattr(env.cfg, "garment_name")
    ):
        from lehome.utils.success_checker_chanllege import evaluate_garment_fold_success

        garment_type = env.garment_loader.get_garment_type(env.cfg.garment_name)
        return evaluate_garment_fold_success(env.object, garment_type)

    if not hasattr(env, "_get_success"):
        return None

    success_value = env._get_success()
    if torch.is_tensor(success_value):
        success = bool(success_value.reshape(-1)[0].item()) if success_value.numel() > 0 else False
    else:
        success = bool(success_value)
    return {"success": success, "garment_type": "unknown", "thresholds": [], "details": {}}


def log_success_result(
    env: DirectRLEnv,
    episode_index: int,
    step_in_episode: int | None = None,
    context: str = "progress",
) -> bool | None:
    """Log a deterministic success breakdown from the recorder."""
    try:
        result = _evaluate_success_result(env)
    except Exception as exc:
        logger.warning(
            f"[Recording][Episode {episode_index}] Failed to evaluate success during {context}: {exc}"
        )
        return None

    if result is None:
        logger.warning(
            f"[Recording][Episode {episode_index}] Success evaluation unavailable during {context}."
        )
        return None

    prefix = f"[Recording][Episode {episode_index}]"
    if step_in_episode is not None:
        prefix += f"[step {step_in_episode}]"

    logger.info(
        f"{prefix} [Success Check] Garment type: {result.get('garment_type', 'unknown')}, "
        f"Thresholds: {result.get('thresholds', [])}"
    )
    details = result.get("details", {})
    for condition_info in details.values():
        status = "✓" if condition_info.get("passed", False) else "✗"
        logger.info(f"{prefix}   {condition_info.get('description', '')} -> {status}")

    success = bool(result.get("success", False))
    logger.info(f"{prefix} [Success Check] Final result: {'Success ✓' if success else 'Failed ✗'}")
    return success


def _get_arm_eef_world_position_cm(
    env: DirectRLEnv,
    arm_name: str,
    eef_body_idx_cache: dict[str, int],
) -> np.ndarray | None:
    arm = _get_scene_articulation(env, arm_name)
    if arm is None:
        return None

    eef_body_idx = _resolve_eef_body_idx(env, arm_name, arm, eef_body_idx_cache)
    if eef_body_idx is None:
        return None

    body_pos_w = getattr(arm.data, "body_link_pos_w", None)
    if body_pos_w is None:
        body_pos_w = getattr(arm.data, "body_pos_w", None)
    if body_pos_w is None:
        return None

    return as_numpy(body_pos_w[0, eef_body_idx], dtype=np.float32).reshape(-1) * 100.0


def _get_debug_arm_names(env: DirectRLEnv) -> list[str]:
    left_arm = _get_scene_articulation(env, "left_arm")
    right_arm = _get_scene_articulation(env, "right_arm")
    if left_arm is not None or right_arm is not None:
        names = []
        if left_arm is not None:
            names.append("left_arm")
        if right_arm is not None:
            names.append("right_arm")
        return names

    for arm_name in _get_single_arm_candidates(env):
        if _get_scene_articulation(env, arm_name) is not None:
            return [arm_name]
    return []


def _get_garment_checkpoint_positions_world_cm(
    particle_object: Any,
    check_points: Sequence[int],
) -> list[list[float]] | None:
    try:
        checkpoint_positions = particle_object.get_checkpoint_world_positions(
            check_points,
            as_numpy=True,
        )
        checkpoint_positions = _as_numpy(checkpoint_positions, dtype=np.float32)
        return (checkpoint_positions * 100.0).tolist()
    except Exception:
        pass

    try:
        world_points, _, _, _ = particle_object.get_current_mesh_points()
        world_points = as_numpy(world_points, dtype=np.float32)
        return (world_points[check_points] * 100.0).tolist()
    except Exception:
        pass

    try:
        world_points = (
            particle_object._cloth_prim_view.get_world_positions().squeeze(0).detach().cpu().numpy()
        )
        world_points = as_numpy(world_points, dtype=np.float32)
        return (world_points[check_points] * 100.0).tolist()
    except Exception:
        return None


def log_debug_pose_snapshot(
    env: DirectRLEnv,
    step_count: int,
    eef_body_idx_cache: dict[str, int],
) -> None:
    """Print EEF and garment checkpoint positions plus same-side distances."""
    prefix = f"[Debug Pose][step {step_count}]"

    arm_names = _get_debug_arm_names(env)
    eef_positions_by_arm: dict[str, np.ndarray] = {}
    if arm_names:
        logger.info(f"{prefix} EEF world positions (cm):")
        for arm_name in arm_names:
            eef_pos_cm = _get_arm_eef_world_position_cm(env, arm_name, eef_body_idx_cache)
            if eef_pos_cm is None or eef_pos_cm.size < 3:
                logger.info(f"{prefix}   {arm_name}: unavailable")
                continue
            eef_positions_by_arm[arm_name] = eef_pos_cm
            logger.info(f"{prefix}   {arm_name}: [{eef_pos_cm[0]:.2f}, {eef_pos_cm[1]:.2f}, {eef_pos_cm[2]:.2f}]")
    else:
        logger.warning(f"{prefix} Could not resolve any robot arm articulation for EEF logging.")

    particle_object = getattr(env, "object", None)
    check_points = getattr(particle_object, "check_points", None)
    if particle_object is None or check_points is None:
        logger.warning(f"{prefix} Garment checkpoints unavailable.")
        return

    garment_type = None
    if hasattr(env, "garment_loader") and hasattr(env, "cfg") and hasattr(env.cfg, "garment_name"):
        try:
            garment_type = env.garment_loader.get_garment_type(env.cfg.garment_name)
        except Exception:
            garment_type = None

    from lehome.utils.success_checker_chanllege import get_object_particle_position

    garment_positions_cm = get_object_particle_position(particle_object, check_points)
    if garment_positions_cm is None:
        logger.warning(f"{prefix} Failed to fetch garment checkpoint positions.")
        return
    garment_world_positions_cm = _get_garment_checkpoint_positions_world_cm(particle_object, check_points)

    if garment_type is not None:
        logger.info(f"{prefix} Garment checkpoints used by success checker (cm) for garment_type={garment_type}:")
    else:
        logger.info(f"{prefix} Garment checkpoints used by success checker (cm):")

    for point_idx, (mesh_idx, point_pos_cm) in enumerate(zip(check_points, garment_positions_cm)):
        point_arr = as_numpy(point_pos_cm, dtype=np.float32).reshape(-1)
        checkpoint_name = (
            GARMENT_CHECKPOINT_LABELS[point_idx]
            if point_idx < len(GARMENT_CHECKPOINT_LABELS)
            else f"checkpoint_{point_idx}"
        )
        if point_arr.size < 3:
            logger.info(f"{prefix}   p[{point_idx}] {checkpoint_name} mesh_idx={mesh_idx}: unavailable")
            continue
        logger.info(
            f"{prefix}   p[{point_idx}] {checkpoint_name} mesh_idx={mesh_idx}: "
            f"[{point_arr[0]:.2f}, {point_arr[1]:.2f}, {point_arr[2]:.2f}]"
        )

    if eef_positions_by_arm:
        garment_world_positions_by_label: dict[str, np.ndarray] = {}
        if garment_world_positions_cm is not None:
            for point_idx, point_pos_cm in enumerate(garment_world_positions_cm):
                checkpoint_name = (
                    GARMENT_CHECKPOINT_LABELS[point_idx]
                    if point_idx < len(GARMENT_CHECKPOINT_LABELS)
                    else f"checkpoint_{point_idx}"
                )
                point_arr = as_numpy(point_pos_cm, dtype=np.float32).reshape(-1)
                if point_arr.size >= 3:
                    garment_world_positions_by_label[checkpoint_name] = point_arr

        logger.info(f"{prefix} Same-side EEF to garment checkpoint distances (world frame, cm):")
        for arm_name, keypoint_names in ARM_KEYPOINT_DISTANCE_LABELS.items():
            eef_pos_cm = eef_positions_by_arm.get(arm_name)
            if eef_pos_cm is None:
                continue
            for keypoint_name in keypoint_names:
                keypoint_pos_cm = garment_world_positions_by_label.get(keypoint_name)
                if keypoint_pos_cm is None or keypoint_pos_cm.size < 3:
                    logger.info(f"{prefix}   {arm_name} -> {keypoint_name}: unavailable")
                    continue
                distance_cm = float(np.linalg.norm(eef_pos_cm[:3] - keypoint_pos_cm[:3]))
                logger.info(f"{prefix}   {arm_name} -> {keypoint_name}: {distance_cm:.2f} cm")


def log_debug_pose_snapshot_if_enabled(
    env: DirectRLEnv,
    args: argparse.Namespace,
    debug_pose_state: dict[str, Any],
) -> None:
    """Log one pose snapshot on step 0 and then at the configured interval."""
    if not getattr(args, "debugging_log_pose", False):
        return

    step_count = int(debug_pose_state.get("step_count", 0))
    if step_count == 0 or step_count % DEBUG_POSE_LOG_INTERVAL == 0:
        eef_body_idx_cache = debug_pose_state.setdefault("eef_body_idx_cache", {})
        log_debug_pose_snapshot(env, step_count, eef_body_idx_cache)

    debug_pose_state["step_count"] = step_count + 1


def log_episode_success_snapshot(
    env: DirectRLEnv,
    episode_index: int,
    episode_step_count: int | None = None,
) -> None:
    """Emit one explicit success summary for the current episode state."""
    success = log_success_result(
        env,
        episode_index=episode_index,
        step_in_episode=episode_step_count,
        context="episode_end",
    )
    if success is None:
        return
    logger.info(
        f"[Recording] Episode {episode_index} success snapshot: "
        f"{'Success ✓' if success else 'Failed ✗'}"
    )


__all__ = [
    "DEBUG_POSE_LOG_INTERVAL",
    "SUCCESS_LOG_INTERVAL",
    "log_debug_pose_snapshot_if_enabled",
    "log_episode_success_snapshot",
    "log_success_result",
]
