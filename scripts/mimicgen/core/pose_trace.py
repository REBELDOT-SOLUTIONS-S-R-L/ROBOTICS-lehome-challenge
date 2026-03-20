"""Shared pose-trace CSV helpers for MimicGen generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_trace_constants():
    try:
        from lehome.tasks.fold_cloth.checkpoint_mappings import (
            ARM_KEYPOINT_GROUPS,
            CSV_TRACE_KEYPOINT_NAMES,
            SUCCESS_DISTANCE_SPECS,
        )

        return ARM_KEYPOINT_GROUPS, CSV_TRACE_KEYPOINT_NAMES, SUCCESS_DISTANCE_SPECS
    except Exception:
        mappings_path = (
            Path(__file__).resolve().parents[3]
            / "source/lehome/lehome/tasks/fold_cloth/checkpoint_mappings.json"
        )
        with mappings_path.open("r", encoding="utf-8") as handle:
            mappings = json.load(handle)

        arm_keypoint_groups = {
            str(arm_name): tuple(str(name) for name in keypoint_names)
            for arm_name, keypoint_names in mappings["arm_keypoint_groups"].items()
        }
        csv_trace_keypoint_names = tuple(
            str(name) for name in mappings["csv_trace_keypoint_names"]
        )
        success_distance_specs = tuple(
            (
                str(item["name"]),
                str(item["src"]),
                str(item["dst"]),
                float(item["threshold_m"]),
            )
            for item in mappings["success_distance_specs"]
        )
        return arm_keypoint_groups, csv_trace_keypoint_names, success_distance_specs


ARM_KEYPOINT_GROUPS, CSV_TRACE_KEYPOINT_NAMES, SUCCESS_DISTANCE_SPECS = _load_trace_constants()


TRACE_EEF_NAMES = ("left_arm", "right_arm")
TRACE_KEYPOINT_NAMES = CSV_TRACE_KEYPOINT_NAMES
TRACE_EEF_KEYPOINT_GROUPS = ARM_KEYPOINT_GROUPS
TRACE_SUCCESS_DISTANCE_SPECS = SUCCESS_DISTANCE_SPECS


def extract_first_xyz(pose: Any) -> list[float] | None:
    """Extract xyz position from a transform-like tensor or array."""
    if pose is None:
        return None
    try:
        tensor = torch.as_tensor(pose)
    except Exception:
        return None

    if tensor.ndim == 3 and tensor.shape[-2:] == (4, 4):
        xyz = tensor[0, :3, 3]
    elif tensor.ndim == 2 and tensor.shape == (4, 4):
        xyz = tensor[:3, 3]
    elif tensor.ndim >= 1 and tensor.shape[-1] >= 3:
        xyz = tensor.reshape(-1, tensor.shape[-1])[0, :3]
    else:
        return None

    return [round(float(v), 6) for v in xyz.detach().cpu().tolist()]


def distance_xyz(a: list[float] | None, b: list[float] | None) -> float | None:
    """Compute Euclidean distance between two xyz points."""
    if a is None or b is None:
        return None
    return round(
        float(
            np.linalg.norm(
                np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
            )
        ),
        6,
    )


def resolve_pose_output_path(output_file: str, pose_output_file: str | None) -> Path:
    """Resolve CSV path for pose trace output."""
    if pose_output_file:
        return Path(pose_output_file).expanduser()
    output_path = Path(output_file).expanduser()
    return output_path.with_name(f"{output_path.stem}_pose_trace.csv")


def build_pose_snapshot(
    env: Any,
    *,
    step_count: int,
    env_id: int = 0,
    episode_index: int | None = None,
    episode_step: int | None = None,
    completed_attempts: int | None = None,
    completed_successes: int | None = None,
) -> dict[str, Any]:
    """Build a flat pose snapshot row suitable for CSV logging."""
    eef_positions: dict[str, list[float]] = {}
    keypoint_positions: dict[str, list[float]] = {}

    for eef_name in TRACE_EEF_NAMES:
        try:
            eef_pose = env.get_robot_eef_pose(eef_name=eef_name, env_ids=[env_id])
        except Exception:
            continue
        xyz = extract_first_xyz(eef_pose)
        if xyz is not None:
            eef_positions[eef_name] = xyz

    try:
        object_poses = env.get_object_poses(env_ids=[env_id])
        if isinstance(object_poses, dict):
            for key_name, key_pose in object_poses.items():
                xyz = extract_first_xyz(key_pose)
                if xyz is not None:
                    keypoint_positions[key_name] = xyz
    except Exception as exc:
        print(f"[pose] step={step_count} failed to read object poses: {exc}")

    row: dict[str, Any] = {
        "step": int(step_count),
        "env_id": int(env_id),
        "episode_index": episode_index,
        "episode_step": episode_step,
        "completed_attempts": completed_attempts,
        "completed_successes": completed_successes,
    }
    for eef_name in TRACE_EEF_NAMES:
        xyz = eef_positions.get(eef_name)
        row[f"eef_{eef_name}_x"] = None if xyz is None else xyz[0]
        row[f"eef_{eef_name}_y"] = None if xyz is None else xyz[1]
        row[f"eef_{eef_name}_z"] = None if xyz is None else xyz[2]
    for keypoint_name in TRACE_KEYPOINT_NAMES:
        xyz = keypoint_positions.get(keypoint_name)
        row[f"keypoint_{keypoint_name}_x"] = None if xyz is None else xyz[0]
        row[f"keypoint_{keypoint_name}_y"] = None if xyz is None else xyz[1]
        row[f"keypoint_{keypoint_name}_z"] = None if xyz is None else xyz[2]

    for eef_name, keypoint_names in TRACE_EEF_KEYPOINT_GROUPS.items():
        eef_xyz = eef_positions.get(eef_name)
        for keypoint_name in keypoint_names:
            row[f"dist_{eef_name}_to_{keypoint_name}_m"] = distance_xyz(
                eef_xyz,
                keypoint_positions.get(keypoint_name),
            )

    for metric_name, src_name, dst_name, threshold in TRACE_SUCCESS_DISTANCE_SPECS:
        distance = distance_xyz(keypoint_positions.get(src_name), keypoint_positions.get(dst_name))
        row[f"dist_{metric_name}_m"] = distance
        row[f"threshold_{metric_name}_m"] = threshold
        row[f"pass_{metric_name}"] = None if distance is None else int(distance <= threshold)

    return row


class PoseTraceCsvWriter:
    """Append flat pose snapshots to CSV for later plotting."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None

    def write(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def write_pose_snapshot(
    env: Any,
    *,
    step_count: int,
    env_id: int = 0,
    pose_writer: PoseTraceCsvWriter | None = None,
    episode_index: int | None = None,
    episode_step: int | None = None,
    completed_attempts: int | None = None,
    completed_successes: int | None = None,
) -> dict[str, Any]:
    """Persist one pose snapshot row to the CSV trace and return it."""
    row = build_pose_snapshot(
        env,
        step_count=step_count,
        env_id=env_id,
        episode_index=episode_index,
        episode_step=episode_step,
        completed_attempts=completed_attempts,
        completed_successes=completed_successes,
    )
    if pose_writer is not None:
        pose_writer.write(row)
    return row
