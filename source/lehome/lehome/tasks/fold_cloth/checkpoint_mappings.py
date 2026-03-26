"""Shared garment checkpoint semantics and per-garment checkpoint loading."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_GARMENT_CFG_BASE_PATH = "Assets/objects/Challenge_Garment"
_REPO_ROOT = Path(__file__).resolve().parents[5]
_GARMENT_TYPE_DIR_BY_PREFIX = {
    "Top_Long": "Top_Long",
    "Top_Short": "Top_Short",
    "Pant_Long": "Pant_Long",
    "Pant_Short": "Pant_Short",
}


CHECKPOINT_LABELS = (
    "garment_left_upper",
    "garment_right_upper",
    "garment_left_middle",
    "garment_right_middle",
    "garment_left_lower",
    "garment_right_lower",
)
CSV_TRACE_KEYPOINT_NAMES = CHECKPOINT_LABELS
ARM_KEYPOINT_GROUPS = {
    "left_arm": tuple(name for name in CHECKPOINT_LABELS if "_left_" in name),
    "right_arm": tuple(name for name in CHECKPOINT_LABELS if "_right_" in name),
}
CENTER_GROUPS = {
    "garment_upper_center": (
        "garment_left_upper",
        "garment_right_upper",
    ),
    "garment_lower_center": (
        "garment_left_lower",
        "garment_right_lower",
    ),
    "garment_kp_left": ARM_KEYPOINT_GROUPS["left_arm"],
    "garment_kp_right": ARM_KEYPOINT_GROUPS["right_arm"],
    "garment_center": CHECKPOINT_LABELS,
}
SUCCESS_DISTANCE_SPECS = (
    ("left_middle_to_lower", "garment_left_middle", "garment_left_lower", 0.10),
    ("right_middle_to_lower", "garment_right_middle", "garment_right_lower", 0.10),
    ("left_lower_to_upper", "garment_left_lower", "garment_left_upper", 0.12),
    ("right_lower_to_upper", "garment_right_lower", "garment_right_upper", 0.12),
)
REQUIRED_MIMIC_OBJECT_REFS = CHECKPOINT_LABELS


class ClothObjectPoseUnavailableError(RuntimeError):
    """Raised when live garment object poses cannot be queried from the environment."""


class ClothObjectPoseValidationError(ValueError):
    """Raised when garment object poses exist structurally but are semantically invalid."""


def _resolve_garment_base_path(base_path: str | Path) -> Path:
    base_path = Path(base_path)
    if base_path.is_absolute():
        return base_path
    return _REPO_ROOT / base_path


def _get_garment_type_dir(garment_name: str) -> str:
    parts = garment_name.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid garment name format: {garment_name!r}. "
            "Expected format like 'Top_Long_Unseen_0'."
        )
    garment_prefix = f"{parts[0]}_{parts[1]}"
    try:
        return _GARMENT_TYPE_DIR_BY_PREFIX[garment_prefix]
    except KeyError as exc:
        raise ValueError(
            f"Unknown garment type prefix {garment_prefix!r} for garment {garment_name!r}."
        ) from exc


def _find_garment_config_path(
    garment_name: str,
    version: str,
    base_path: str | Path,
) -> Path:
    garment_dir = _resolve_garment_base_path(base_path) / version / _get_garment_type_dir(garment_name) / garment_name
    if not garment_dir.is_dir():
        raise FileNotFoundError(f"Garment directory not found: {garment_dir}")

    candidates = sorted(garment_dir.glob("*_obj_exp.json"))
    if not candidates:
        candidates = sorted(garment_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No garment JSON configuration found under: {garment_dir}")
    return candidates[0]


@lru_cache(maxsize=None)
def _load_garment_config_raw(
    garment_name: str,
    version: str = "Release",
    base_path: str = DEFAULT_GARMENT_CFG_BASE_PATH,
) -> dict[str, Any]:
    config_path = _find_garment_config_path(
        garment_name=garment_name,
        version=version,
        base_path=base_path,
    )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(
            f"Expected garment JSON object at {config_path}, got {type(config).__name__}."
        )
    return config


def _extract_checkpoint_indices(
    garment_name: str,
    version: str = "Release",
    base_path: str = DEFAULT_GARMENT_CFG_BASE_PATH,
) -> tuple[int, ...]:
    config = _load_garment_config_raw(
        garment_name=garment_name,
        version=version,
        base_path=base_path,
    )
    checkpoint_indices = tuple(int(idx) for idx in config.get("check_point", []))
    if len(checkpoint_indices) < len(CHECKPOINT_LABELS):
        raise ValueError(
            f"Garment {garment_name!r} ({version}) exposes {len(checkpoint_indices)} checkpoints, "
            f"expected at least {len(CHECKPOINT_LABELS)}."
        )
    return checkpoint_indices


def semantic_keypoints_from_positions(kp_positions: np.ndarray) -> dict[str, np.ndarray]:
    """Map six checkpoint positions into shared semantic garment names."""
    kp_positions = np.asarray(kp_positions, dtype=np.float32)
    if kp_positions.ndim != 2 or kp_positions.shape[0] < len(CHECKPOINT_LABELS) or kp_positions.shape[1] < 3:
        raise ValueError(
            f"Expected kp_positions with shape (>=6, 3), got {tuple(kp_positions.shape)}"
        )

    semantic_points = {
        checkpoint_name: kp_positions[idx]
        for idx, checkpoint_name in enumerate(CHECKPOINT_LABELS)
    }
    for group_name, member_names in CENTER_GROUPS.items():
        semantic_points[group_name] = np.mean(
            np.stack([semantic_points[name] for name in member_names], axis=0),
            axis=0,
        )
    return semantic_points


def semantic_keypoints_from_positions_torch(kp_positions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Torch-native variant of semantic garment checkpoint mapping."""
    kp_positions = torch.as_tensor(kp_positions, dtype=torch.float32)
    if kp_positions.ndim != 2 or kp_positions.shape[0] < len(CHECKPOINT_LABELS) or kp_positions.shape[1] < 3:
        raise ValueError(
            f"Expected kp_positions with shape (>=6, 3), got {tuple(kp_positions.shape)}"
        )

    semantic_points = {
        checkpoint_name: kp_positions[idx, :3]
        for idx, checkpoint_name in enumerate(CHECKPOINT_LABELS)
    }
    for group_name, member_names in CENTER_GROUPS.items():
        semantic_points[group_name] = torch.stack(
            [semantic_points[name] for name in member_names],
            dim=0,
        ).mean(dim=0)
    return semantic_points


def _pose_value_to_numpy(pose_value: Any) -> np.ndarray:
    """Convert tensor-like pose values into numpy arrays without changing structure."""
    if hasattr(pose_value, "detach"):
        pose_value = pose_value.detach()
    if hasattr(pose_value, "cpu"):
        pose_value = pose_value.cpu()
    return np.asarray(pose_value, dtype=np.float32)


def validate_semantic_object_pose_dict(
    pose_dict: Any,
    *,
    context: str = "object_pose",
    required_refs: tuple[str, ...] = REQUIRED_MIMIC_OBJECT_REFS,
    identity_translation_tol_m: float = 1e-4,
    collapsed_translation_tol_m: float = 5e-3,
) -> None:
    """Validate that semantic garment object poses are present and not degenerate."""
    if not isinstance(pose_dict, dict) or len(pose_dict) == 0:
        raise ClothObjectPoseValidationError(f"{context}: expected a non-empty pose dict.")

    missing_refs = [name for name in required_refs if name not in pose_dict]
    if missing_refs:
        raise ClothObjectPoseValidationError(
            f"{context}: missing required garment refs: {missing_refs}."
        )

    translation_series_by_ref: dict[str, np.ndarray] = {}
    for ref_name in required_refs:
        try:
            pose_array = _pose_value_to_numpy(pose_dict[ref_name])
        except Exception as exc:
            raise ClothObjectPoseValidationError(
                f"{context}: failed to convert pose for {ref_name!r}: {exc}"
            ) from exc

        if pose_array.ndim == 2 and pose_array.shape == (4, 4):
            pose_array = pose_array[np.newaxis, ...]
        if pose_array.ndim != 3 or pose_array.shape[-2:] != (4, 4):
            raise ClothObjectPoseValidationError(
                f"{context}: invalid pose shape for {ref_name!r}: {tuple(pose_array.shape)}."
            )
        if pose_array.shape[0] == 0:
            raise ClothObjectPoseValidationError(f"{context}: empty pose horizon for {ref_name!r}.")
        if not np.all(np.isfinite(pose_array)):
            raise ClothObjectPoseValidationError(
                f"{context}: non-finite values detected in pose for {ref_name!r}."
            )
        translation_series_by_ref[ref_name] = pose_array[:, :3, 3]

    try:
        stacked_series = np.stack([translation_series_by_ref[name] for name in required_refs], axis=0)
    except ValueError as exc:
        raise ClothObjectPoseValidationError(
            f"{context}: required garment pose horizons are inconsistent across refs."
        ) from exc
    if np.all(np.abs(stacked_series) <= float(identity_translation_tol_m)):
        raise ClothObjectPoseValidationError(
            f"{context}: all required garment pose translations are near the origin/identity."
        )

    first_step_translations = stacked_series[:, 0, :]
    per_axis_spread = np.ptp(first_step_translations, axis=0)
    max_spread = float(np.max(per_axis_spread))
    if max_spread <= float(collapsed_translation_tol_m):
        raise ClothObjectPoseValidationError(
            f"{context}: required garment keypoints collapse to the same location "
            f"(max spread {max_spread:.6f} m)."
        )


def get_garment_checkpoint_config(
    garment_name: str,
    version: str = "Release",
    base_path: str = DEFAULT_GARMENT_CFG_BASE_PATH,
) -> dict[str, Any]:
    """Return per-garment checkpoint metadata derived from the raw garment JSON."""
    config_path = _find_garment_config_path(
        garment_name=garment_name,
        version=version,
        base_path=base_path,
    )
    checkpoint_indices = _extract_checkpoint_indices(
        garment_name=garment_name,
        version=version,
        base_path=base_path,
    )
    return {
        "garment_type_dir": _get_garment_type_dir(garment_name),
        "config_path": str(config_path.relative_to(_resolve_garment_base_path(base_path)).as_posix()),
        "checkpoint_indices": checkpoint_indices,
        "checkpoint_name_to_index": {
            checkpoint_name: checkpoint_indices[idx]
            for idx, checkpoint_name in enumerate(CHECKPOINT_LABELS)
        },
    }


def get_garment_checkpoint_indices(
    garment_name: str,
    version: str = "Release",
    base_path: str = DEFAULT_GARMENT_CFG_BASE_PATH,
) -> tuple[int, ...]:
    """Return the raw check_point indices for a garment."""
    return _extract_checkpoint_indices(
        garment_name=garment_name,
        version=version,
        base_path=base_path,
    )
