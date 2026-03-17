"""Shared garment checkpoint semantics loaded from checkpoint_mappings.json."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


_MAPPINGS_PATH = Path(__file__).with_name("checkpoint_mappings.json")


@lru_cache(maxsize=1)
def _load_checkpoint_mappings() -> dict[str, Any]:
    with _MAPPINGS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_MAPPINGS = _load_checkpoint_mappings()

CHECKPOINT_LABELS = tuple(str(name) for name in _MAPPINGS["checkpoint_names_by_index"])
ARM_KEYPOINT_GROUPS = {
    str(arm_name): tuple(str(name) for name in keypoint_names)
    for arm_name, keypoint_names in _MAPPINGS["arm_keypoint_groups"].items()
}
CSV_TRACE_KEYPOINT_NAMES = tuple(
    str(name) for name in _MAPPINGS["csv_trace_keypoint_names"]
)
CENTER_GROUPS = {
    str(group_name): tuple(str(name) for name in members)
    for group_name, members in _MAPPINGS["center_groups"].items()
}
SUCCESS_DISTANCE_SPECS = tuple(
    (
        str(item["name"]),
        str(item["src"]),
        str(item["dst"]),
        float(item["threshold_m"]),
    )
    for item in _MAPPINGS["success_distance_specs"]
)
GARMENT_CHECKPOINT_CONFIGS = {
    str(version): {
        str(garment_name): dict(config)
        for garment_name, config in garments.items()
    }
    for version, garments in _MAPPINGS.get("garments", {}).items()
}
REQUIRED_MIMIC_OBJECT_REFS = (
    "garment_left_sleeve",
    "garment_right_sleeve",
    "garment_left_bottom",
    "garment_right_bottom",
    "garment_left_top",
    "garment_right_top",
)


class ClothObjectPoseUnavailableError(RuntimeError):
    """Raised when live garment object poses cannot be queried from the environment."""


class ClothObjectPoseValidationError(ValueError):
    """Raised when garment object poses exist structurally but are semantically invalid."""


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


def get_garment_checkpoint_config(garment_name: str, version: str = "Release") -> dict[str, Any]:
    """Return per-garment checkpoint metadata loaded from checkpoint_mappings.json."""
    try:
        return GARMENT_CHECKPOINT_CONFIGS[version][garment_name]
    except KeyError as exc:
        raise KeyError(
            f"No checkpoint mapping found for garment_name={garment_name!r}, version={version!r}"
        ) from exc


def get_garment_checkpoint_indices(garment_name: str, version: str = "Release") -> tuple[int, ...]:
    """Return the raw check_point indices for a garment."""
    config = get_garment_checkpoint_config(garment_name, version)
    return tuple(int(idx) for idx in config.get("checkpoint_indices", []))
