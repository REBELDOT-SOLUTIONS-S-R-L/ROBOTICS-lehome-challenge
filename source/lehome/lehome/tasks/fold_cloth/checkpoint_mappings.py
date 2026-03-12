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
