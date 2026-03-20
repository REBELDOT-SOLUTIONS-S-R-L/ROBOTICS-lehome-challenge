"""Shared dataset and garment metadata helpers for MimicGen scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset_io import (
    decode_attr,
    demo_sort_key,
    normalize_hdf5_scalar,
    read_hdf5_node,
)

try:
    import h5py
except ImportError:
    h5py = None


def find_pose_list_in_obj(obj: Any) -> list[float] | None:
    """Find an initial-pose-like numeric list recursively."""
    if isinstance(obj, dict):
        for key in ("object_initial_pose", "initial_pose", "garment_initial_pose"):
            value = obj.get(key)
            if isinstance(value, list) and len(value) >= 6:
                return value
        for value in obj.values():
            pose = find_pose_list_in_obj(value)
            if pose is not None:
                return pose
    elif isinstance(obj, list):
        if len(obj) >= 6 and all(isinstance(v, (int, float)) for v in obj[:6]):
            return [float(v) for v in obj]
        for item in obj:
            pose = find_pose_list_in_obj(item)
            if pose is not None:
                return pose
    return None


def extract_garment_name_from_episode_meta(meta: dict[str, Any]) -> str | None:
    """Extract garment name from per-episode meta payload."""
    for key in ("garment_name", "garment", "asset_name"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value

    garment_info = meta.get("garment_info")
    if garment_info is None:
        garment_info = meta.get("garment_info.json")
    if isinstance(garment_info, dict) and garment_info:
        first_key = next(iter(garment_info.keys()))
        if isinstance(first_key, str) and first_key:
            return first_key

    for key, value in meta.items():
        if key in {"garment_info", "garment_info.json", "episode_index", "seed"}:
            continue
        if isinstance(key, str) and key.endswith(".json"):
            continue
        if isinstance(value, dict) and find_pose_list_in_obj(value) is not None:
            if isinstance(key, str) and key:
                return key
    return None


def extract_initial_pose_from_episode_meta(
    meta: dict[str, Any],
    *,
    source_episode_index: int | None = None,
) -> dict[str, Any] | None:
    """Extract an env.set_all_pose-compatible garment initial pose from episode metadata."""
    garment_info = meta.get("garment_info")
    if garment_info is None:
        garment_info = meta.get("garment_info.json")

    if isinstance(garment_info, dict):
        for _, episodes in garment_info.items():
            if not isinstance(episodes, dict):
                continue

            preferred_keys: list[str] = []
            if source_episode_index is not None:
                preferred_keys.append(str(source_episode_index))
            preferred_keys.extend(sorted(episodes.keys()))

            seen: set[str] = set()
            for episode_key in preferred_keys:
                if episode_key in seen:
                    continue
                seen.add(episode_key)
                episode_meta = episodes.get(episode_key)
                if not isinstance(episode_meta, dict):
                    continue
                pose = episode_meta.get("object_initial_pose")
                if isinstance(pose, list) and len(pose) >= 6:
                    return {"Garment": pose}

    pose = find_pose_list_in_obj(meta)
    if pose is None:
        return None
    return {"Garment": pose}


def load_garment_info_json(
    garment_info_path: str | Path | None,
    *,
    warning_prefix: str = "Warning",
) -> dict[str, Any] | None:
    """Load garment_info.json for initial pose replay."""
    if garment_info_path is None:
        return None
    path = Path(garment_info_path)
    if not path.exists():
        print(f"{warning_prefix}: garment_info_json not found: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"{warning_prefix}: failed to parse garment_info_json ({path}): {exc}")
    return None


def get_first_garment_name(garment_info: dict[str, Any] | None) -> str | None:
    """Return the first garment key from garment_info data."""
    if not garment_info or not isinstance(garment_info, dict):
        return None
    for garment_name, episodes in garment_info.items():
        if isinstance(garment_name, str) and garment_name and isinstance(episodes, dict):
            return garment_name
    return None


def try_read_garment_name_from_json(json_path: Path) -> str | None:
    """Read the top-level garment name key from garment_info.json."""
    if not json_path.exists():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and data:
            first_key = next(iter(data.keys()))
            if isinstance(first_key, str) and first_key:
                return first_key
    except (OSError, json.JSONDecodeError):
        return None
    return None


def find_garment_info_json(hdf5_path: Path) -> Path | None:
    """Find garment_info.json associated with an HDF5 file when available."""
    candidates = [
        hdf5_path.parent / f"{hdf5_path.stem}.garment_info.json",
        hdf5_path.parent / "garment_info.json",
        hdf5_path.parent / "meta" / "garment_info.json",
        hdf5_path.parent / hdf5_path.stem / "meta" / "garment_info.json",
        hdf5_path.parent / "record" / hdf5_path.stem / "meta" / "garment_info.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_initial_pose_from_json(
    pose_file: Path | None,
    source_episode_index: int,
) -> dict[str, Any] | None:
    """Load an initial pose from garment_info.json using source episode index."""
    if pose_file is None or not pose_file.exists():
        return None

    try:
        with pose_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    episode_key = str(source_episode_index)
    for _, episodes in data.items():
        if not isinstance(episodes, dict):
            continue
        if episode_key not in episodes:
            continue
        pose_list = episodes[episode_key].get("object_initial_pose")
        if pose_list is not None:
            return {"Garment": pose_list}
    return None


def load_dataset_env_args(input_file: str) -> dict[str, Any]:
    """Read dataset-level env_args from /data attrs."""
    if h5py is None:
        return {}
    try:
        with h5py.File(input_file, "r") as file:
            data_group = file.get("data")
            if data_group is None:
                return {}
            raw = data_group.attrs.get("env_args")
            if raw is None:
                return {}
            raw = decode_attr(raw)
            if isinstance(raw, str):
                return json.loads(raw)
    except Exception:
        return {}
    return {}


def _iter_demo_groups(input_file: str):
    if h5py is None:
        return []
    try:
        with h5py.File(input_file, "r") as file:
            data_group = file.get("data")
            if data_group is None:
                return []
            items = []
            for demo_name in sorted(
                [name for name in data_group.keys() if name.startswith("demo_")],
                key=demo_sort_key,
            ):
                items.append((demo_name, data_group[demo_name]))
            return items
    except Exception:
        return []


def get_first_demo_garment_name(input_file: str) -> str | None:
    """Infer garment name from the first demo metadata payload available."""
    if h5py is None:
        return None
    try:
        with h5py.File(input_file, "r") as file:
            data_group = file.get("data")
            if data_group is None:
                return None
            demo_names = sorted(
                [name for name in data_group.keys() if name.startswith("demo_")],
                key=demo_sort_key,
            )
            for demo_name in demo_names:
                demo_group = data_group[demo_name]
                if "meta" not in demo_group:
                    continue
                meta = read_hdf5_node(demo_group["meta"])
                if isinstance(meta, dict):
                    garment_name = extract_garment_name_from_episode_meta(meta)
                    if garment_name:
                        return garment_name
    except Exception:
        return None
    return None


def get_first_demo_object_pose_keys(input_file: str) -> set[str] | None:
    """Read the union of source datagen object_pose keys across available demos."""
    if h5py is None:
        return None
    try:
        with h5py.File(input_file, "r") as file:
            data_group = file.get("data")
            if data_group is None:
                return None
            all_keys: set[str] = set()
            for demo_name in sorted(
                [name for name in data_group.keys() if name.startswith("demo_")],
                key=demo_sort_key,
            ):
                demo_group = data_group[demo_name]
                obs_group = demo_group.get("obs")
                if obs_group is None:
                    continue
                datagen_group = obs_group.get("datagen_info")
                if datagen_group is None:
                    continue
                object_pose_group = datagen_group.get("object_pose")
                if object_pose_group is None:
                    continue
                all_keys.update(object_pose_group.keys())
            if all_keys:
                return all_keys
    except Exception:
        return None
    return None


def get_source_actions_mode(input_file: str) -> str | None:
    """Read /data attrs['actions_mode'] when present."""
    if h5py is None:
        return None
    try:
        with h5py.File(input_file, "r") as file:
            data_group = file.get("data")
            if data_group is None:
                return None
            mode = data_group.attrs.get("actions_mode")
            if mode is None:
                return None
            mode = decode_attr(mode)
            return str(mode) if mode is not None else None
    except Exception:
        return None


def get_first_demo_action_dim(input_file: str) -> int | None:
    """Read top-level /data/demo_*/actions second dimension from the first demo."""
    if h5py is None:
        return None
    try:
        with h5py.File(input_file, "r") as file:
            data_group = file.get("data")
            if data_group is None:
                return None
            for demo_name in sorted(
                [name for name in data_group.keys() if name.startswith("demo_")],
                key=demo_sort_key,
            ):
                demo_group = data_group[demo_name]
                if "actions" not in demo_group:
                    continue
                shape = tuple(np.asarray(demo_group["actions"]).shape)
                if len(shape) != 2:
                    continue
                return int(shape[1])
    except Exception:
        return None
    return None


def get_first_demo_pose_frame_stats(
    input_file: str,
    *,
    max_steps: int = 128,
) -> dict[str, float] | None:
    """Extract simple source-frame stats to detect mixed object/eef pose frames."""
    if h5py is None:
        return None
    try:
        with h5py.File(input_file, "r") as file:
            data_group = file.get("data")
            if data_group is None:
                return None
            for demo_name in sorted(
                [name for name in data_group.keys() if name.startswith("demo_")],
                key=demo_sort_key,
            ):
                demo_group = data_group[demo_name]
                obs_group = demo_group.get("obs")
                if obs_group is None:
                    continue
                datagen_group = obs_group.get("datagen_info")
                if datagen_group is None:
                    continue
                object_pose_group = datagen_group.get("object_pose")
                target_pose_group = datagen_group.get("target_eef_pose")
                eef_pose_group = datagen_group.get("eef_pose")
                if object_pose_group is None:
                    continue

                object_positions = []
                for key in object_pose_group.keys():
                    arr = np.asarray(object_pose_group[key])
                    if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                        n = min(max_steps, arr.shape[0])
                        object_positions.append(arr[:n, :3, 3].reshape(-1, 3))
                if not object_positions:
                    continue
                src_object_center = np.concatenate(object_positions, axis=0).mean(axis=0)

                pose_group = target_pose_group if target_pose_group is not None else eef_pose_group
                if pose_group is None:
                    continue
                eef_positions = []
                for key in pose_group.keys():
                    arr = np.asarray(pose_group[key])
                    if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                        n = min(max_steps, arr.shape[0])
                        eef_positions.append(arr[:n, :3, 3].reshape(-1, 3))
                if not eef_positions:
                    continue
                src_target_center = np.concatenate(eef_positions, axis=0).mean(axis=0)

                return {
                    "src_object_center_x": float(src_object_center[0]),
                    "src_object_center_y": float(src_object_center[1]),
                    "src_object_center_z": float(src_object_center[2]),
                    "src_target_center_x": float(src_target_center[0]),
                    "src_target_center_y": float(src_target_center[1]),
                    "src_target_center_z": float(src_target_center[2]),
                    "src_target_object_z_gap": float(abs(src_target_center[2] - src_object_center[2])),
                }
    except Exception:
        return None
    return None


def should_auto_fix_mixed_pose_frames(
    input_file: str,
    runtime_object_center: torch.Tensor | None,
) -> tuple[bool, dict[str, float] | None]:
    """Heuristic detection of mixed-frame source data."""
    if runtime_object_center is None:
        return False, None
    stats = get_first_demo_pose_frame_stats(input_file)
    if stats is None:
        return False, None
    runtime_z = float(runtime_object_center[2].item())
    source_z_shift = abs(runtime_z - stats["src_object_center_z"])
    should_fix = (source_z_shift > 0.20) and (stats["src_target_object_z_gap"] > 0.25)
    stats = dict(stats)
    stats["runtime_object_center_z"] = runtime_z
    stats["source_runtime_object_z_shift"] = source_z_shift
    return should_fix, stats


def build_dataset_metadata_index(
    data_group: Any,
    *,
    index_cls: type,
    normalize_scalar=normalize_hdf5_scalar,
    warning_printer=print,
):
    """Scan an already-open /data group once and cache demo metadata."""
    if data_group is None:
        return index_cls(
            garment_info=None,
            actions_frame=None,
            ik_quat_order=None,
            source_episode_indices={},
            episode_groups={},
        )

    def _normalize_attr(raw_value, valid_values: set[str] | None = None) -> str | None:
        raw_value = normalize_scalar(raw_value)
        if raw_value is None:
            return None
        value = str(raw_value).strip().lower()
        if valid_values is not None and value not in valid_values:
            return None
        return value

    def _merge_garment_info(dst: dict[str, Any], src: dict[str, Any]) -> None:
        for garment_name, episodes in src.items():
            if not isinstance(episodes, dict):
                continue
            dst.setdefault(str(garment_name), {})
            for episode_idx, payload in episodes.items():
                dst[str(garment_name)][str(episode_idx)] = payload

    merged_garment_info: dict[str, Any] = {}
    source_episode_indices: dict[str, int] = {}
    episode_groups: dict[str, Any] = {}

    try:
        actions_frame = _normalize_attr(data_group.attrs.get("actions_frame", None), {"base", "world"})
        ik_quat_order = _normalize_attr(data_group.attrs.get("ik_quat_order", None), {"xyzw", "wxyz"})

        for demo_name in sorted(data_group.keys(), key=demo_sort_key):
            if not demo_name.startswith("demo_"):
                continue

            demo_group = data_group[demo_name]
            episode_groups[demo_name] = demo_group

            try:
                if "source_episode_index" in demo_group.attrs:
                    source_episode_indices[demo_name] = int(demo_group.attrs["source_episode_index"])
            except Exception:
                pass

            meta_group = demo_group.get("meta")
            if meta_group is not None:
                for key in ("garment_info", "garment_info.json"):
                    if key not in meta_group:
                        continue
                    raw = normalize_scalar(meta_group[key][()])
                    if isinstance(raw, str):
                        try:
                            parsed = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(parsed, dict):
                            _merge_garment_info(merged_garment_info, parsed)

            initial_state_group = demo_group.get("initial_state")
            garment_group = None if initial_state_group is None else initial_state_group.get("garment")
            demo_suffix = demo_name.split("_", maxsplit=1)[1] if "_" in demo_name else None
            if garment_group is None or demo_suffix is None or not demo_suffix.isdigit():
                continue

            for garment_name in garment_group.keys():
                garment_entry = garment_group[garment_name]
                if "initial_pose" not in garment_entry:
                    continue

                pose = normalize_scalar(garment_entry["initial_pose"][()])
                if not isinstance(pose, list):
                    continue

                payload: dict[str, Any] = {"object_initial_pose": pose}
                if "scale" in garment_entry:
                    payload["scale"] = normalize_scalar(garment_entry["scale"][()])

                merged_garment_info.setdefault(str(garment_name), {})[demo_suffix] = payload
    except Exception as exc:
        warning_printer(f"Warning: failed to read dataset metadata index from HDF5: {exc}")
        return index_cls(
            garment_info=None,
            actions_frame=None,
            ik_quat_order=None,
            source_episode_indices=source_episode_indices,
            episode_groups=episode_groups,
        )

    return index_cls(
        garment_info=merged_garment_info or None,
        actions_frame=actions_frame,
        ik_quat_order=ik_quat_order,
        source_episode_indices=source_episode_indices,
        episode_groups=episode_groups,
    )
