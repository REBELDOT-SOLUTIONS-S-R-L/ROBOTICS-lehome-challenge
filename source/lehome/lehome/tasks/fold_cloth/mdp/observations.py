"""Observation functions for the garment folding environment."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

from isaaclab.managers import SceneEntityCfg

from ..checkpoint_mappings import (
    CHECKPOINT_LABELS,
    semantic_keypoints_from_positions as map_semantic_keypoints_from_positions,
)
from lehome.utils.robot_utils import is_so101_at_rest_pose
from lehome.utils.success_checker_chanllege import (
    evaluate_garment_fold_success,
    success_checker_garment_fold,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_DEFAULT_GRIPPER_JOINT_IDX = 5
_DEFAULT_GRIPPER_CLOSE_THRESHOLD = 0.20
_DEFAULT_GRASP_EEF_TO_KEYPOINT_THRESHOLD_M = 0.05
_DEFAULT_MIDDLE_TO_LOWER_THRESHOLD_M = 0.10
_DEFAULT_MIDDLE_TO_LOWER_MIDDLE_KEYPOINT_MAX_Z_M = 0.53
_DEFAULT_LOWER_TO_UPPER_THRESHOLD_M = 0.12
# Narrow release-zone geometry (in the garment's own in-plane frame).
# The "big square" is the quad formed by the 4 corner keypoints
# (garment_{left,right}_{lower,upper}).  Inside that square we carve out a
# narrow rectangle centered on the left<->right midline, covering the lower
# half along the upper<->lower axis and 60% of the span along the
# left<->right axis.
_DEFAULT_RELEASE_ZONE_WIDTH_FRACTION = 0.60
_DEFAULT_RELEASE_ZONE_LOWER_FRACTION = 0.50
_RETURN_HOME_SIGNALS = {"left_return_home", "right_return_home"}

_RELEASE_ZONE_LOG = logging.getLogger("lehome.fold_cloth.release_zone")
_RELEASE_ZONE_DEBUG_ENV_VAR = "LEHOME_DEBUG_RELEASE_ZONE"
# One log block per second at most; set to 0 via the env var to log every call.
_RELEASE_ZONE_DEBUG_MIN_INTERVAL_S = 1.0
_release_zone_debug_last_ts: list[float] = [0.0]
_release_zone_debug_handler_installed: list[bool] = [False]


def _release_zone_debug_enabled() -> bool:
    raw = os.environ.get(_RELEASE_ZONE_DEBUG_ENV_VAR, "").strip().lower()
    enabled = raw not in ("", "0", "false", "no", "off")
    if enabled and not _release_zone_debug_handler_installed[0]:
        # Self-configure so users can just ``export LEHOME_DEBUG_RELEASE_ZONE=1``
        # without also tweaking the root logger level.
        if not _RELEASE_ZONE_LOG.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            _RELEASE_ZONE_LOG.addHandler(handler)
            _RELEASE_ZONE_LOG.propagate = False
        _RELEASE_ZONE_LOG.setLevel(logging.INFO)
        _release_zone_debug_handler_installed[0] = True
    return enabled


def _release_zone_debug_should_log() -> bool:
    if not _release_zone_debug_enabled():
        return False
    min_interval = _RELEASE_ZONE_DEBUG_MIN_INTERVAL_S
    try:
        min_interval = float(os.environ.get("LEHOME_DEBUG_RELEASE_ZONE_INTERVAL_S", min_interval))
    except Exception:
        pass
    now = time.monotonic()
    if now - _release_zone_debug_last_ts[0] < max(0.0, min_interval):
        return False
    _release_zone_debug_last_ts[0] = now
    return True


@dataclass
class FoldClothSubtaskObservationContext:
    """Precomputed state used to evaluate fold-cloth subtask predicates."""

    device: torch.device | str
    num_envs: int
    semantic_keypoints_world: dict[str, torch.Tensor]
    eef_world_positions: dict[str, torch.Tensor]
    gripper_closed_by_arm: dict[str, torch.Tensor]
    arm_at_rest_by_arm: dict[str, torch.Tensor]
    arm_at_waiting_pos_by_arm: dict[str, torch.Tensor]
    grasp_eef_to_keypoint_threshold_m: float
    middle_to_lower_threshold_m: float
    middle_to_lower_middle_keypoint_max_z_m: float
    lower_to_upper_threshold_m: float
    # Per-arm ``(num_envs, 1)`` bool tensor: EEF's world XY is inside the narrow
    # release zone built from the 4 garment corner keypoints.  Empty dict when
    # the corners are not available (degenerate cloth state), in which case the
    # signals depending on this fall back to False.
    eef_in_release_zone_by_arm: dict[str, torch.Tensor] | None = None
    fold_success: torch.Tensor | None = None


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: Sequence[int] | None) -> tuple[Sequence[int] | slice, int]:
    if hasattr(env, "_resolve_env_ids"):
        return env._resolve_env_ids(env_ids)
    if env_ids is None:
        return slice(None), int(env.num_envs)
    if isinstance(env_ids, slice):
        return env_ids, int(env.num_envs)
    return env_ids, len(env_ids)


def _resolve_arm_name(arm: str | SceneEntityCfg) -> str:
    if isinstance(arm, SceneEntityCfg):
        return arm.name
    return str(arm)


def _resolve_checkpoint_name(checkpoint_name: str) -> str:
    checkpoint_name = str(checkpoint_name)
    raise KeyError(
            f"Unsupported garment checkpoint name {checkpoint_name!r}. "

        )
    return checkpoint_name


def _cfg_float(env: ManagerBasedEnv, attr_name: str, default: float) -> float:
    cfg = getattr(env, "cfg", None)
    value = getattr(cfg, attr_name, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _false_bool_column(env: ManagerBasedEnv, num_envs: int) -> torch.Tensor:
    return torch.zeros((num_envs, 1), dtype=torch.bool, device=env.device)


def _true_bool_column(env: ManagerBasedEnv, num_envs: int) -> torch.Tensor:
    return torch.ones((num_envs, 1), dtype=torch.bool, device=env.device)


def _full_float_column(env: ManagerBasedEnv, num_envs: int, value: float) -> torch.Tensor:
    return torch.full((num_envs, 1), float(value), dtype=torch.float32, device=env.device)


def _context_false_bool_column(context: FoldClothSubtaskObservationContext) -> torch.Tensor:
    return torch.zeros((context.num_envs, 1), dtype=torch.bool, device=context.device)


def _context_true_bool_column(context: FoldClothSubtaskObservationContext) -> torch.Tensor:
    return torch.ones((context.num_envs, 1), dtype=torch.bool, device=context.device)


def _context_full_float_column(
    context: FoldClothSubtaskObservationContext,
    value: float,
) -> torch.Tensor:
    return torch.full(
        (context.num_envs, 1),
        float(value),
        dtype=torch.float32,
        device=context.device,
    )


def _get_scene_arm(env: ManagerBasedEnv, arm: str | SceneEntityCfg):
    arm_name = _resolve_arm_name(arm)
    return arm_name, env.scene[arm_name]


def _get_gripper_joint_index(arm: Any) -> int:
    joint_names = list(getattr(arm.data, "joint_names", []) or [])
    if "gripper" in joint_names:
        return int(joint_names.index("gripper"))
    return _DEFAULT_GRIPPER_JOINT_IDX


def _get_eef_world_position(
    env: ManagerBasedEnv,
    arm: str | SceneEntityCfg,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    env_ids, num_envs = _resolve_env_ids(env, env_ids)
    arm_name, arm_entity = _get_scene_arm(env, arm)
    if hasattr(env, "get_robot_eef_pose"):
        pose = env.get_robot_eef_pose(arm_name, env_ids=env_ids)
        pose = torch.as_tensor(pose, device=env.device, dtype=torch.float32)
        if pose.ndim == 2 and pose.shape == (4, 4):
            pose = pose.unsqueeze(0)
        return pose[:, :3, 3].reshape(num_envs, 3)

    if not hasattr(env, "_get_eef_body_idx"):
        raise AttributeError(f"{type(env).__name__} does not expose get_robot_eef_pose() or _get_eef_body_idx().")
    eef_body_idx = int(env._get_eef_body_idx(arm_name))
    return arm_entity.data.body_link_pos_w[env_ids, eef_body_idx].reshape(num_envs, 3)


def _get_semantic_keypoint_positions_world(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None = None,
) -> dict[str, torch.Tensor] | None:
    _, num_envs = _resolve_env_ids(env, env_ids)
    garment_obj = getattr(env, "object", None)
    if garment_obj is None or not hasattr(garment_obj, "check_points"):
        return None

    check_points = getattr(garment_obj, "check_points", None)
    if not check_points or len(check_points) < len(CHECKPOINT_LABELS):
        return None

    try:
        kp_positions = garment_obj.get_checkpoint_world_positions(
            tuple(check_points[: len(CHECKPOINT_LABELS)]),
            as_numpy=True,
        )
    except Exception:
        return None

    kp_positions = np.asarray(kp_positions, dtype=np.float32)
    try:
        semantic_points = map_semantic_keypoints_from_positions(kp_positions)
    except Exception:
        return None

    points_world: dict[str, torch.Tensor] = {}
    for name, point in semantic_points.items():
        points_world[name] = (
            torch.as_tensor(point, dtype=torch.float32, device=env.device)
            .reshape(1, 3)
            .expand(num_envs, 3)
            .clone()
        )
    return points_world


def _fold_success_scalar(env: ManagerBasedEnv) -> bool:
    garment_object = getattr(env, "object", None)
    garment_loader = getattr(env, "garment_loader", None)
    cfg = getattr(env, "cfg", None)
    garment_name = getattr(cfg, "garment_name", None)
    if garment_object is None or garment_loader is None or garment_name is None:
        return False

    try:
        garment_type = garment_loader.get_garment_type(garment_name)
        result = evaluate_garment_fold_success(garment_object, garment_type)
    except Exception:
        try:
            result = success_checker_garment_fold(garment_object, garment_type)
        except Exception:
            return False

    if isinstance(result, dict):
        return bool(result.get("success", False))
    return bool(result)


def _normalize_optional_bool_column(
    env: ManagerBasedEnv,
    num_envs: int,
    value: Any | None,
) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=env.device)
    if tensor.ndim == 0:
        scalar = bool(tensor.item())
        return _true_bool_column(env, num_envs) if scalar else _false_bool_column(env, num_envs)
    tensor = tensor.to(dtype=torch.bool).reshape(-1)
    if tensor.numel() == 1:
        scalar = bool(tensor[0].item())
        return _true_bool_column(env, num_envs) if scalar else _false_bool_column(env, num_envs)
    return tensor[:num_envs].reshape(num_envs, 1)


def _compute_eef_in_release_zone(
    env: ManagerBasedEnv,
    num_envs: int,
    semantic_points: dict[str, torch.Tensor],
    eef_positions: dict[str, torch.Tensor],
    width_fraction: float,
    lower_fraction: float,
) -> dict[str, torch.Tensor]:
    """Return per-arm ``(num_envs, 1)`` bool: EEF XY in the narrow release zone.

    The zone is built in the garment's own in-plane frame:

    * ``u`` is the unit vector along the left->right corner axis, taken as the
      mean of (left_lower -> right_lower) and (left_upper -> right_upper).
    * ``v_perp`` is Gram-Schmidt orthogonalized from ``v`` (upper->lower axis)
      against ``u``, so ``(u, v_perp)`` is a right-handed orthonormal basis
      lying in the world XY plane.
    * ``W`` / ``H`` are the full spans between the corner midpoints along
      ``u`` / ``v`` respectively.  The narrow zone is the lower
      ``lower_fraction`` along ``v_perp`` intersected with the central
      ``width_fraction`` along ``u``.

    If any corner keypoint is missing or the span collapses, every arm's
    membership is reported as False (safe / conservative).
    """
    device = next(
        (t.device for t in list(semantic_points.values()) + list(eef_positions.values())),
        env.device,
    )

    def _false_zone() -> dict[str, torch.Tensor]:
        return {
            arm_name: torch.zeros((num_envs, 1), dtype=torch.bool, device=device)
            for arm_name in eef_positions
        }

    corners = (
        "garment_left_lower",
        "garment_left_upper",
        "garment_right_lower",
        "garment_right_upper",
    )
    missing = [name for name in corners if name not in semantic_points]
    if missing:
        if _release_zone_debug_should_log():
            _RELEASE_ZONE_LOG.info(
                "[release_zone] corners missing -> zone=False for all arms. missing=%s available=%s",
                missing,
                sorted(semantic_points.keys()),
            )
        return _false_zone()

    # (num_envs, 2) tensors in world XY.
    kp_ll = semantic_points["garment_left_lower"][..., :2]
    kp_lu = semantic_points["garment_left_upper"][..., :2]
    kp_rl = semantic_points["garment_right_lower"][..., :2]
    kp_ru = semantic_points["garment_right_upper"][..., :2]

    left_mid = 0.5 * (kp_ll + kp_lu)
    right_mid = 0.5 * (kp_rl + kp_ru)
    upper_mid = 0.5 * (kp_lu + kp_ru)
    lower_mid = 0.5 * (kp_ll + kp_rl)
    center = 0.25 * (kp_ll + kp_lu + kp_rl + kp_ru)

    u_vec = right_mid - left_mid
    v_vec = lower_mid - upper_mid

    eps = 1e-6
    w_full = torch.linalg.norm(u_vec, dim=-1, keepdim=True)
    h_full = torch.linalg.norm(v_vec, dim=-1, keepdim=True)
    valid_span = (w_full > eps) & (h_full > eps)
    if not bool(valid_span.all()):
        # Any env with a degenerate span falls back to False; we still compute
        # the rest for the valid envs below.  Mask is applied at the end.
        pass

    u_hat = u_vec / torch.clamp(w_full, min=eps)
    # Gram-Schmidt: v_perp = normalize(v - (v . u) u).  ``v_norm`` is therefore
    # the orthogonal component of the upper->lower vector projected against
    # ``u_hat``; this is the correct span to use for the lower-half bound
    # when the quad isn't perfectly rectangular.  For a rectangle it equals
    # ``h_full`` exactly.
    v_dot_u = (v_vec * u_hat).sum(dim=-1, keepdim=True)
    v_orth = v_vec - v_dot_u * u_hat
    v_norm = torch.linalg.norm(v_orth, dim=-1, keepdim=True)
    valid_span = valid_span & (v_norm > eps)
    v_hat = v_orth / torch.clamp(v_norm, min=eps)

    # ``half_*`` describes half-spans in the orthonormal (u_hat, v_hat) basis.
    half_u = 0.5 * w_full
    half_v = 0.5 * v_norm
    width_half = half_u * float(width_fraction)
    # Lower half along v_hat corresponds to s_v in [0, +||v_orth||/2].  With
    # ``lower_fraction=0.5`` this is exactly the lower half of the square,
    # measured in the same orthonormal frame as ``s_v``.
    v_upper_bound = float(lower_fraction) * v_norm
    v_lower_bound = torch.zeros_like(v_upper_bound)

    result: dict[str, torch.Tensor] = {}
    per_arm_diag: dict[str, dict[str, float]] = {}
    for arm_name, eef_xyz in eef_positions.items():
        eef_xy = eef_xyz[..., :2]
        d = eef_xy - center
        s_u = (d * u_hat).sum(dim=-1, keepdim=True)
        s_v = (d * v_hat).sum(dim=-1, keepdim=True)
        inside_width = s_u.abs() <= width_half
        inside_lower = (s_v >= v_lower_bound) & (s_v <= v_upper_bound)
        inside = inside_width & inside_lower & valid_span
        result[arm_name] = inside.reshape(num_envs, 1)

        if _release_zone_debug_enabled():
            per_arm_diag[arm_name] = {
                "eef_x": float(eef_xy[0, 0].item()),
                "eef_y": float(eef_xy[0, 1].item()),
                "s_u": float(s_u[0, 0].item()),
                "s_v": float(s_v[0, 0].item()),
                "s_u_abs_le_width_half": bool(inside_width[0, 0].item()),
                "s_v_in_lower": bool(inside_lower[0, 0].item()),
                "valid_span": bool(valid_span[0, 0].item()),
                "inside": bool(inside[0, 0].item()),
            }

    if _release_zone_debug_enabled() and _release_zone_debug_should_log():
        try:
            corners_xy = {name: semantic_points[name][0, :2].tolist() for name in corners}
            u_hat_vec = u_hat[0].tolist()
            v_hat_vec = v_hat[0].tolist()
            _RELEASE_ZONE_LOG.info(
                "[release_zone] corners=%s center=%s u_hat=%s v_hat=%s W=%.4f ||v_orth||=%.4f "
                "width_half=%.4f v_upper_bound=%.4f | %s",
                corners_xy,
                center[0].tolist(),
                u_hat_vec,
                v_hat_vec,
                float(w_full[0, 0].item()),
                float(v_norm[0, 0].item()),
                float(width_half[0, 0].item()),
                float(v_upper_bound[0, 0].item()),
                per_arm_diag,
            )
        except Exception as exc:
            _RELEASE_ZONE_LOG.debug("release-zone diagnostic formatting failed: %s", exc)

    return result


def build_subtask_observation_context(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None = None,
    *,
    semantic_keypoints_world: dict[str, torch.Tensor] | None = None,
    eef_world_positions: dict[str, torch.Tensor] | None = None,
    gripper_closed_by_arm: dict[str, torch.Tensor] | None = None,
    arm_at_rest_by_arm: dict[str, torch.Tensor] | None = None,
    fold_success_value: Any | None = None,
    include_fold_success: bool = False,
) -> FoldClothSubtaskObservationContext:
    """Build a shared context for evaluating one or more subtask predicates."""
    _, num_envs = _resolve_env_ids(env, env_ids)
    semantic_points = (
        semantic_keypoints_world
        if semantic_keypoints_world is not None
        else _get_semantic_keypoint_positions_world(env, env_ids=env_ids)
    )
    if semantic_points is None:
        semantic_points = {}

    eef_positions = eef_world_positions or {
        "left_arm": _get_eef_world_position(env, "left_arm", env_ids=env_ids),
        "right_arm": _get_eef_world_position(env, "right_arm", env_ids=env_ids),
    }
    gripper_closed_map = gripper_closed_by_arm or {
        "left_arm": gripper_closed(env, "left_arm", env_ids=env_ids),
        "right_arm": gripper_closed(env, "right_arm", env_ids=env_ids),
    }
    arm_rest_map = arm_at_rest_by_arm or {
        "left_arm": arm_at_rest(env, "left_arm", env_ids=env_ids),
        "right_arm": arm_at_rest(env, "right_arm", env_ids=env_ids),
    }
    arm_waiting_map = {
        "left_arm": arm_at_waiting_pos(env, "left_arm", env_ids=env_ids),
        "right_arm": arm_at_waiting_pos(env, "right_arm", env_ids=env_ids),
    }

    fold_success_column = _normalize_optional_bool_column(
        env,
        num_envs,
        fold_success_value,
    )
    if fold_success_column is None and include_fold_success:
        fold_success_column = (
            _true_bool_column(env, num_envs)
            if _fold_success_scalar(env)
            else _false_bool_column(env, num_envs)
        )

    zone_width_fraction = _cfg_float(
        env,
        "subtask_release_zone_width_fraction",
        _DEFAULT_RELEASE_ZONE_WIDTH_FRACTION,
    )
    zone_lower_fraction = _cfg_float(
        env,
        "subtask_release_zone_lower_fraction",
        _DEFAULT_RELEASE_ZONE_LOWER_FRACTION,
    )
    eef_in_release_zone_map = _compute_eef_in_release_zone(
        env,
        num_envs,
        semantic_points,
        eef_positions,
        width_fraction=zone_width_fraction,
        lower_fraction=zone_lower_fraction,
    )

    return FoldClothSubtaskObservationContext(
        device=env.device,
        num_envs=num_envs,
        semantic_keypoints_world=semantic_points,
        eef_world_positions=eef_positions,
        gripper_closed_by_arm=gripper_closed_map,
        arm_at_rest_by_arm=arm_rest_map,
        arm_at_waiting_pos_by_arm=arm_waiting_map,
        grasp_eef_to_keypoint_threshold_m=_cfg_float(
            env,
            "subtask_grasp_eef_to_keypoint_threshold_m",
            _DEFAULT_GRASP_EEF_TO_KEYPOINT_THRESHOLD_M,
        ),
        middle_to_lower_threshold_m=_cfg_float(
            env,
            "subtask_middle_to_lower_threshold_m",
            _DEFAULT_MIDDLE_TO_LOWER_THRESHOLD_M,
        ),
        middle_to_lower_middle_keypoint_max_z_m=_cfg_float(
            env,
            "subtask_middle_to_lower_middle_keypoint_max_z_m",
            _DEFAULT_MIDDLE_TO_LOWER_MIDDLE_KEYPOINT_MAX_Z_M,
        ),
        lower_to_upper_threshold_m=_cfg_float(
            env,
            "subtask_lower_to_upper_threshold_m",
            _DEFAULT_LOWER_TO_UPPER_THRESHOLD_M,
        ),
        eef_in_release_zone_by_arm=eef_in_release_zone_map,
        fold_success=fold_success_column,
    )


def _context_keypoint_pair_distance(
    context: FoldClothSubtaskObservationContext,
    checkpoint_a: str,
    checkpoint_b: str,
) -> torch.Tensor:
    pos_a = context.semantic_keypoints_world.get(checkpoint_a)
    pos_b = context.semantic_keypoints_world.get(checkpoint_b)
    if pos_a is None or pos_b is None:
        return _context_full_float_column(context, float("inf"))
    return torch.linalg.norm(pos_a - pos_b, dim=-1, keepdim=True)


def _context_keypoint_z(
    context: FoldClothSubtaskObservationContext,
    checkpoint_name: str,
) -> torch.Tensor:
    kp_pos = context.semantic_keypoints_world.get(checkpoint_name)
    if kp_pos is None:
        return _context_full_float_column(context, float("inf"))
    return kp_pos[..., 2:3]


def _context_eef_to_keypoint_distance(
    context: FoldClothSubtaskObservationContext,
    arm_name: str,
    checkpoint_name: str,
) -> torch.Tensor:
    eef_pos = context.eef_world_positions.get(arm_name)
    kp_pos = context.semantic_keypoints_world.get(checkpoint_name)
    if eef_pos is None or kp_pos is None:
        return _context_full_float_column(context, float("inf"))
    return torch.linalg.norm(eef_pos - kp_pos, dim=-1, keepdim=True)


def get_subtask_signal_observation_from_context(
    context: FoldClothSubtaskObservationContext,
    signal_name: str,
) -> torch.Tensor:
    """Evaluate a single instantaneous subtask predicate from precomputed state."""
    signal_name = str(signal_name)
    if signal_name == "grasp_left_middle":
        return context.gripper_closed_by_arm.get("left_arm", _context_false_bool_column(context)) & (
            _context_eef_to_keypoint_distance(context, "left_arm", "garment_left_middle")
            <= context.grasp_eef_to_keypoint_threshold_m
        )
    if signal_name == "grasp_right_middle":
        return context.gripper_closed_by_arm.get("right_arm", _context_false_bool_column(context)) & (
            _context_eef_to_keypoint_distance(context, "right_arm", "garment_right_middle")
            <= context.grasp_eef_to_keypoint_threshold_m
        )
    if signal_name == "left_middle_to_lower":
        # Arm still holding the middle keypoint and has moved its EEF into the
        # narrow drop zone above the lower-corner half of the garment.
        in_zone_map = context.eef_in_release_zone_by_arm or {}
        in_zone = in_zone_map.get("left_arm", _context_false_bool_column(context))
        return context.gripper_closed_by_arm.get("left_arm", _context_false_bool_column(context)) & in_zone
    if signal_name == "right_middle_to_lower":
        in_zone_map = context.eef_in_release_zone_by_arm or {}
        in_zone = in_zone_map.get("right_arm", _context_false_bool_column(context))
        return context.gripper_closed_by_arm.get("right_arm", _context_false_bool_column(context)) & in_zone
    if signal_name == "release_left_middle":
        # Arm's EEF is inside the zone, gripper has opened, and the tracked
        # middle keypoint has dropped below the configured max Z (i.e. the
        # cloth actually detached and is resting on the lower garment half).
        in_zone_map = context.eef_in_release_zone_by_arm or {}
        in_zone = in_zone_map.get("left_arm", _context_false_bool_column(context))
        return (
            in_zone
            & (~context.gripper_closed_by_arm.get("left_arm", _context_false_bool_column(context)))
            & (
                _context_keypoint_z(context, "garment_left_middle")
                < context.middle_to_lower_middle_keypoint_max_z_m
            )
        )
    if signal_name == "release_right_middle":
        in_zone_map = context.eef_in_release_zone_by_arm or {}
        in_zone = in_zone_map.get("right_arm", _context_false_bool_column(context))
        return (
            in_zone
            & (~context.gripper_closed_by_arm.get("right_arm", _context_false_bool_column(context)))
            & (
                _context_keypoint_z(context, "garment_right_middle")
                < context.middle_to_lower_middle_keypoint_max_z_m
            )
        )
    if signal_name == "grasp_left_lower":
        return context.gripper_closed_by_arm.get("left_arm", _context_false_bool_column(context)) & (
            _context_eef_to_keypoint_distance(context, "left_arm", "garment_left_lower")
            <= context.grasp_eef_to_keypoint_threshold_m
        )
    if signal_name == "grasp_right_lower":
        return context.gripper_closed_by_arm.get("right_arm", _context_false_bool_column(context)) & (
            _context_eef_to_keypoint_distance(context, "right_arm", "garment_right_lower")
            <= context.grasp_eef_to_keypoint_threshold_m
        )
    if signal_name == "left_lower_to_upper":
        return (
            ~context.gripper_closed_by_arm.get("left_arm", _context_false_bool_column(context))
        ) & (
            _context_keypoint_pair_distance(
                context,
                "garment_left_lower",
                "garment_left_upper",
            ) <= context.lower_to_upper_threshold_m
        )
    if signal_name == "right_lower_to_upper":
        return (
            ~context.gripper_closed_by_arm.get("right_arm", _context_false_bool_column(context))
        ) & (
            _context_keypoint_pair_distance(
                context,
                "garment_right_lower",
                "garment_right_upper",
            ) <= context.lower_to_upper_threshold_m
        )
    if signal_name == "left_at_waiting_pos":
        return context.arm_at_waiting_pos_by_arm.get("left_arm", _context_false_bool_column(context))
    if signal_name == "right_at_waiting_pos":
        return context.arm_at_waiting_pos_by_arm.get("right_arm", _context_false_bool_column(context))
    if signal_name == "left_return_home":
        fold_success_value = context.fold_success
        if fold_success_value is None:
            return _context_false_bool_column(context)
        return fold_success_value & context.arm_at_rest_by_arm.get(
            "left_arm",
            _context_false_bool_column(context),
        )
    if signal_name == "right_return_home":
        fold_success_value = context.fold_success
        if fold_success_value is None:
            return _context_false_bool_column(context)
        return fold_success_value & context.arm_at_rest_by_arm.get(
            "right_arm",
            _context_false_bool_column(context),
        )
    raise KeyError(
        f"Unsupported subtask signal {signal_name!r}. "
        f"Expected one of {tuple(SUBTASK_SIGNAL_OBSERVATION_FNS)}."
    )


def get_subtask_signal_observations_from_context(
    context: FoldClothSubtaskObservationContext,
) -> dict[str, torch.Tensor]:
    """Evaluate all instantaneous subtask predicates from precomputed state."""
    return {
        signal_name: get_subtask_signal_observation_from_context(context, signal_name)
        for signal_name in SUBTASK_SIGNAL_OBSERVATION_FNS
    }


def robot_rest_pose(
    env: ManagerBasedEnv,
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
) -> torch.Tensor:
    """Return per-arm rest-pose indicators for the bimanual SO101 setup.

    The observation is a 2D vector per environment:
      [left_arm_at_rest, right_arm_at_rest]
    represented as float values in {0.0, 1.0}.
    """

    left_arm = env.scene[left_arm_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]

    left_at_rest = is_so101_at_rest_pose(
        left_arm.data.joint_pos,
        left_arm.data.joint_names,
        arm_name=left_arm_cfg.name,
    )
    right_at_rest = is_so101_at_rest_pose(
        right_arm.data.joint_pos,
        right_arm.data.joint_names,
        arm_name=right_arm_cfg.name,
    )

    return torch.stack(
        (
            left_at_rest.to(dtype=torch.float32),
            right_at_rest.to(dtype=torch.float32),
        ),
        dim=-1,
    )


def gripper_closed(
    env: ManagerBasedEnv,
    arm: str | SceneEntityCfg,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return whether the given arm gripper is closed."""
    env_ids, num_envs = _resolve_env_ids(env, env_ids)
    _, arm_entity = _get_scene_arm(env, arm)
    gripper_joint_idx = _get_gripper_joint_index(arm_entity)
    threshold = _cfg_float(env, "subtask_gripper_close_threshold", _DEFAULT_GRIPPER_CLOSE_THRESHOLD)
    joint_pos = arm_entity.data.joint_pos[env_ids, gripper_joint_idx].reshape(num_envs, 1)
    return joint_pos < threshold


def arm_at_rest(
    env: ManagerBasedEnv,
    arm: str | SceneEntityCfg,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return whether the given arm is at the rest pose."""
    env_ids, num_envs = _resolve_env_ids(env, env_ids)
    arm_name, arm_entity = _get_scene_arm(env, arm)
    at_rest = is_so101_at_rest_pose(
        arm_entity.data.joint_pos[env_ids],
        arm_entity.data.joint_names,
        arm_name=arm_name,
    )
    return at_rest.reshape(num_envs, 1)


_WAITING_POS_EEF_X_THRESHOLD = 0.20


def arm_at_waiting_pos(
    env: ManagerBasedEnv,
    arm: str | SceneEntityCfg,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return whether the arm EEF has retracted past the X threshold.

    Left arm: eef_x < -0.20.  Right arm: eef_x > 0.20.
    """
    env_ids, num_envs = _resolve_env_ids(env, env_ids)
    arm_name, _ = _get_scene_arm(env, arm)
    eef_pos = _get_eef_world_position(env, arm_name, env_ids=env_ids)
    eef_x = eef_pos[..., 0:1]  # (num_envs, 1)
    normalized = str(arm_name).strip().lower()
    if "left" in normalized:
        return eef_x < -_WAITING_POS_EEF_X_THRESHOLD
    return eef_x > _WAITING_POS_EEF_X_THRESHOLD


def fold_success(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return whether the garment is folded successfully, without requiring rest-pose."""
    _, num_envs = _resolve_env_ids(env, env_ids)
    if _fold_success_scalar(env):
        return _true_bool_column(env, num_envs)
    return _false_bool_column(env, num_envs)


def eef_to_keypoint_distance(
    env: ManagerBasedEnv,
    arm: str | SceneEntityCfg,
    checkpoint_name: str,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return the world-frame distance between an arm EEF and a garment checkpoint."""
    _, num_envs = _resolve_env_ids(env, env_ids)
    checkpoint_name = _resolve_checkpoint_name(checkpoint_name)
    semantic_points = _get_semantic_keypoint_positions_world(env, env_ids=env_ids)
    if semantic_points is None or checkpoint_name not in semantic_points:
        return _full_float_column(env, num_envs, float("inf"))

    eef_pos = _get_eef_world_position(env, arm, env_ids=env_ids)
    kp_pos = semantic_points[checkpoint_name]
    return torch.linalg.norm(eef_pos - kp_pos, dim=-1, keepdim=True)


def keypoint_pair_distance(
    env: ManagerBasedEnv,
    checkpoint_a: str,
    checkpoint_b: str,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return the world-frame distance between two garment checkpoints."""
    _, num_envs = _resolve_env_ids(env, env_ids)
    checkpoint_a = _resolve_checkpoint_name(checkpoint_a)
    checkpoint_b = _resolve_checkpoint_name(checkpoint_b)
    semantic_points = _get_semantic_keypoint_positions_world(env, env_ids=env_ids)
    if semantic_points is None or checkpoint_a not in semantic_points or checkpoint_b not in semantic_points:
        return _full_float_column(env, num_envs, float("inf"))

    pos_a = semantic_points[checkpoint_a]
    pos_b = semantic_points[checkpoint_b]
    return torch.linalg.norm(pos_a - pos_b, dim=-1, keepdim=True)


def grasp_left_middle(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_grasp_eef_to_keypoint_threshold_m",
        _DEFAULT_GRASP_EEF_TO_KEYPOINT_THRESHOLD_M,
    )
    return gripper_closed(env, "left_arm", env_ids=env_ids) & (
        eef_to_keypoint_distance(env, "left_arm", "garment_left_middle", env_ids=env_ids) <= threshold
    )


def grasp_right_middle(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_grasp_eef_to_keypoint_threshold_m",
        _DEFAULT_GRASP_EEF_TO_KEYPOINT_THRESHOLD_M,
    )
    return gripper_closed(env, "right_arm", env_ids=env_ids) & (
        eef_to_keypoint_distance(env, "right_arm", "garment_right_middle", env_ids=env_ids) <= threshold
    )


def _release_zone_signal(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None,
    arm_name: str,
) -> torch.Tensor:
    """Compute ``(num_envs, 1)`` bool: ``arm_name`` EEF XY in the narrow release zone."""
    _, num_envs = _resolve_env_ids(env, env_ids)
    semantic_points = _get_semantic_keypoint_positions_world(env, env_ids=env_ids)
    if semantic_points is None:
        return _false_bool_column(env, num_envs)
    eef_positions = {arm_name: _get_eef_world_position(env, arm_name, env_ids=env_ids)}
    width_fraction = _cfg_float(
        env,
        "subtask_release_zone_width_fraction",
        _DEFAULT_RELEASE_ZONE_WIDTH_FRACTION,
    )
    lower_fraction = _cfg_float(
        env,
        "subtask_release_zone_lower_fraction",
        _DEFAULT_RELEASE_ZONE_LOWER_FRACTION,
    )
    in_zone = _compute_eef_in_release_zone(
        env,
        num_envs,
        semantic_points,
        eef_positions,
        width_fraction=width_fraction,
        lower_fraction=lower_fraction,
    )
    return in_zone.get(arm_name, _false_bool_column(env, num_envs))


def left_middle_to_lower(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    # New semantics: gripper still CLOSED (holding the middle keypoint) and
    # the EEF has moved into the narrow drop zone derived from the 4 corner
    # keypoints.  The release itself is handled by ``release_left_middle``.
    return gripper_closed(env, "left_arm", env_ids=env_ids) & _release_zone_signal(env, env_ids, "left_arm")


def right_middle_to_lower(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    return gripper_closed(env, "right_arm", env_ids=env_ids) & _release_zone_signal(env, env_ids, "right_arm")


def _release_middle_signal(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None,
    arm_name: str,
    middle_keypoint: str,
) -> torch.Tensor:
    _, num_envs = _resolve_env_ids(env, env_ids)
    threshold_z = _cfg_float(
        env,
        "subtask_middle_to_lower_middle_keypoint_max_z_m",
        _DEFAULT_MIDDLE_TO_LOWER_MIDDLE_KEYPOINT_MAX_Z_M,
    )
    semantic_points = _get_semantic_keypoint_positions_world(env, env_ids=env_ids)
    if semantic_points is None or middle_keypoint not in semantic_points:
        return _false_bool_column(env, num_envs)
    in_zone = _release_zone_signal(env, env_ids, arm_name)
    gripper_open = ~gripper_closed(env, arm_name, env_ids=env_ids)
    kp_z_below = semantic_points[middle_keypoint][..., 2:3] < threshold_z
    return in_zone & gripper_open & kp_z_below


def release_left_middle(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    return _release_middle_signal(env, env_ids, "left_arm", "garment_left_middle")


def release_right_middle(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    return _release_middle_signal(env, env_ids, "right_arm", "garment_right_middle")


def grasp_left_lower(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_grasp_eef_to_keypoint_threshold_m",
        _DEFAULT_GRASP_EEF_TO_KEYPOINT_THRESHOLD_M,
    )
    return gripper_closed(env, "left_arm", env_ids=env_ids) & (
        eef_to_keypoint_distance(env, "left_arm", "garment_left_lower", env_ids=env_ids) <= threshold
    )


def grasp_right_lower(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_grasp_eef_to_keypoint_threshold_m",
        _DEFAULT_GRASP_EEF_TO_KEYPOINT_THRESHOLD_M,
    )
    return gripper_closed(env, "right_arm", env_ids=env_ids) & (
        eef_to_keypoint_distance(env, "right_arm", "garment_right_lower", env_ids=env_ids) <= threshold
    )


def left_lower_to_upper(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_lower_to_upper_threshold_m",
        _DEFAULT_LOWER_TO_UPPER_THRESHOLD_M,
    )
    return (~gripper_closed(env, "left_arm", env_ids=env_ids)) & (
        keypoint_pair_distance(
            env,
            "garment_left_lower",
            "garment_left_upper",
            env_ids=env_ids,
        ) <= threshold
    )


def right_lower_to_upper(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    threshold = _cfg_float(
        env,
        "subtask_lower_to_upper_threshold_m",
        _DEFAULT_LOWER_TO_UPPER_THRESHOLD_M,
    )
    return (~gripper_closed(env, "right_arm", env_ids=env_ids)) & (
        keypoint_pair_distance(
            env,
            "garment_right_lower",
            "garment_right_upper",
            env_ids=env_ids,
        ) <= threshold
    )


def left_at_waiting_pos(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    return arm_at_waiting_pos(env, "left_arm", env_ids=env_ids)


def right_at_waiting_pos(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    return arm_at_waiting_pos(env, "right_arm", env_ids=env_ids)


def left_return_home(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    return fold_success(env, env_ids=env_ids) & arm_at_rest(env, "left_arm", env_ids=env_ids)


def right_return_home(env: ManagerBasedEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    return fold_success(env, env_ids=env_ids) & arm_at_rest(env, "right_arm", env_ids=env_ids)


SUBTASK_SIGNAL_OBSERVATION_FNS = {
    "grasp_left_middle": grasp_left_middle,
    "grasp_right_middle": grasp_right_middle,
    "left_middle_to_lower": left_middle_to_lower,
    "right_middle_to_lower": right_middle_to_lower,
    "release_left_middle": release_left_middle,
    "release_right_middle": release_right_middle,
    "left_at_waiting_pos": left_at_waiting_pos,
    "right_at_waiting_pos": right_at_waiting_pos,
    "grasp_left_lower": grasp_left_lower,
    "grasp_right_lower": grasp_right_lower,
    "left_lower_to_upper": left_lower_to_upper,
    "right_lower_to_upper": right_lower_to_upper,
    "left_return_home": left_return_home,
    "right_return_home": right_return_home,
}


def get_subtask_signal_observation(
    env: ManagerBasedEnv,
    signal_name: str,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Evaluate a single instantaneous subtask predicate by exact signal name."""
    signal_name = str(signal_name)
    context = build_subtask_observation_context(
        env,
        env_ids=env_ids,
        include_fold_success=signal_name in _RETURN_HOME_SIGNALS,
    )
    return get_subtask_signal_observation_from_context(context, signal_name)


def get_subtask_signal_observations(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate all instantaneous subtask predicates."""
    context = build_subtask_observation_context(
        env,
        env_ids=env_ids,
        include_fold_success=True,
    )
    return get_subtask_signal_observations_from_context(context)


__all__ = [
    "FoldClothSubtaskObservationContext",
    "SUBTASK_SIGNAL_OBSERVATION_FNS",
    "arm_at_rest",
    "build_subtask_observation_context",
    "eef_to_keypoint_distance",
    "fold_success",
    "get_subtask_signal_observation",
    "get_subtask_signal_observation_from_context",
    "get_subtask_signal_observations",
    "get_subtask_signal_observations_from_context",
    "grasp_left_lower",
    "grasp_left_middle",
    "grasp_right_lower",
    "grasp_right_middle",
    "arm_at_waiting_pos",
    "gripper_closed",
    "keypoint_pair_distance",
    "left_at_waiting_pos",
    "left_lower_to_upper",
    "left_middle_to_lower",
    "left_return_home",
    "release_left_middle",
    "release_right_middle",
    "right_at_waiting_pos",
    "right_lower_to_upper",
    "right_middle_to_lower",
    "right_return_home",
    "robot_rest_pose",
]
