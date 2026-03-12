# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Main mimic data generation script with LeHome HDF5 compatibility."""


"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import json

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument("--task_type", type=str, default=None, help="Specify task type. If your annotated dataset is recorded with keyboard, you should set it to 'keyboard', otherwise not to set it and keep default value None.")
parser.add_argument(
    "--garment_name",
    type=str,
    default=None,
    help="Garment name (for garment tasks), e.g. Top_Long_Unseen_0.",
)
parser.add_argument(
    "--garment_version",
    type=str,
    default=None,
    help="Garment version (for garment tasks), e.g. Release or Holdout.",
)
parser.add_argument(
    "--garment_cfg_base_path",
    type=str,
    default="Assets/objects/Challenge_Garment",
    help="Base path of garment assets (for garment tasks).",
)
parser.add_argument(
    "--particle_cfg_path",
    type=str,
    default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
    help="Path to particle garment config yaml (for garment tasks).",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--pose_output_interval",
    type=int,
    default=1,
    help="Pose print interval in env steps (used only with --print_poses). Set <=0 to disable.",
)
parser.add_argument(
    "--print_poses",
    action="store_true",
    default=False,
    help="Print EEF and keypoint positions during generation.",
)
parser.add_argument(
    "--use_eef_pose_as_target",
    action="store_true",
    default=False,
    help=(
        "Use measured datagen_info.eef_pose as source target trajectory "
        "(instead of recorded target_eef_pose)."
    ),
)
parser.add_argument(
    "--source_target_z_offset",
    type=float,
    default=0.0,
    help=(
        "Meters added to source target EEF z before generation. "
        "Use negative values to lower grasp trajectories (e.g. -0.02)."
    ),
)
parser.add_argument(
    "--align_object_pose_to_runtime",
    action="store_true",
    default=False,
    help=(
        "Enable legacy source pose alignment to runtime frame. "
        "Keep disabled unless you are repairing old datasets with known global frame offsets."
    ),
)
parser.add_argument(
    "--object_pose_alignment_mode",
    type=str,
    default="object_only",
    choices=["object_only", "all_poses"],
    help=(
        "Alignment behavior when --align_object_pose_to_runtime is enabled: "
        "'object_only' shifts source object poses only (for mixed-frame datasets), "
        "'all_poses' shifts object/eef/target together (for pure global frame shifts)."
    ),
)
parser.add_argument(
    "--auto_fix_mixed_pose_frames",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Auto-detect mixed source pose frames and enable object-only source object alignment when needed. "
        "Disabled by default in strict pipeline mode."
    ),
)
parser.add_argument(
    "--strict_preflight",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Run strict source dataset contract checks before generation "
        "(recommended; enabled by default)."
    ),
)
parser.add_argument(
    "--expected_source_action_dim",
    type=int,
    default=16,
    help="Expected top-level source action dimension for strict preflight checks.",
)
parser.add_argument(
    "--require_source_actions_mode",
    type=str,
    default="ee_pose",
    help="Required /data attrs['actions_mode'] value in strict preflight mode (set empty to skip).",
)
parser.add_argument(
    "--disable_object_pose_alignment",
    action="store_true",
    default=False,
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--mimic_ik_orientation_weight",
    type=float,
    default=0.01,
    help=(
        "Orientation weight forwarded to env IK conversion (target_eef_pose_to_action). "
        "Higher values enforce source wrist orientation more strongly; 0 disables orientation tracking."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import gymnasium as gym
import inspect
import numpy as np
import random
import torch
from typing import Any

import omni

from isaaclab.envs import ManagerBasedRLMimicEnv

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
import isaaclab_mimic.datagen.generation as mimic_generation
from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool
from isaaclab_mimic.datagen.generation import setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import lehome.tasks  # noqa: F401

from lehome.utils.env_utils import get_task_type

try:
    import h5py
except ImportError:
    h5py = None


def _decode_attr(value: Any) -> Any:
    """Decode HDF5 scalar attribute values to plain python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_dataset_env_args(input_file: str) -> dict[str, Any]:
    """Read dataset-level env_args from /data attrs."""
    if h5py is None:
        return {}
    try:
        with h5py.File(input_file, "r") as f:
            data_group = f.get("data")
            if data_group is None:
                return {}
            raw = data_group.attrs.get("env_args")
            if raw is None:
                return {}
            raw = _decode_attr(raw)
            if isinstance(raw, str):
                return json.loads(raw)
    except Exception:
        return {}
    return {}


def _find_pose_list_in_obj(obj: Any) -> list[float] | None:
    """Find an initial-pose-like numeric list recursively."""
    if isinstance(obj, dict):
        for key in ("object_initial_pose", "initial_pose", "garment_initial_pose"):
            value = obj.get(key)
            if isinstance(value, list) and len(value) >= 6:
                return value
        for value in obj.values():
            pose = _find_pose_list_in_obj(value)
            if pose is not None:
                return pose
    elif isinstance(obj, list):
        if len(obj) >= 6 and all(isinstance(v, (int, float)) for v in obj[:6]):
            return [float(v) for v in obj]
        for item in obj:
            pose = _find_pose_list_in_obj(item)
            if pose is not None:
                return pose
    return None


def _extract_garment_name_from_episode_meta(meta: dict[str, Any]) -> str | None:
    """Extract garment name from /data/demo_*/meta payload."""
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
        if isinstance(value, dict) and _find_pose_list_in_obj(value) is not None:
            if isinstance(key, str) and key:
                return key
    return None


def _parse_json_if_possible(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _read_hdf5_node(node: Any) -> Any:
    """Recursively read HDF5 node into python values for metadata parsing."""
    if h5py is None:
        return None
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if isinstance(value, np.ndarray):
            value = value.item() if value.ndim == 0 else value.tolist()
        value = _decode_attr(value)
        return _parse_json_if_possible(value)
    if isinstance(node, h5py.Group):
        out: dict[str, Any] = {}
        for key in node.keys():
            out[key] = _read_hdf5_node(node[key])
        return out
    return None


def _demo_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("demo_"):
        suffix = name.split("demo_", maxsplit=1)[1]
        if suffix.isdigit():
            return int(suffix), name
    return 10**9, name


def _get_first_demo_garment_name(input_file: str) -> str | None:
    """Infer garment name from first demo /meta payload."""
    if h5py is None:
        return None
    try:
        with h5py.File(input_file, "r") as f:
            data_group = f.get("data")
            if data_group is None:
                return None
            demo_names = sorted(
                [name for name in data_group.keys() if name.startswith("demo_")],
                key=_demo_sort_key,
            )
            for demo_name in demo_names:
                demo_group = data_group[demo_name]
                if "meta" not in demo_group:
                    continue
                meta = _read_hdf5_node(demo_group["meta"])
                if isinstance(meta, dict):
                    garment_name = _extract_garment_name_from_episode_meta(meta)
                    if garment_name:
                        return garment_name
    except Exception:
        return None
    return None


def _get_first_demo_object_pose_keys(input_file: str) -> set[str] | None:
    """Read first demo datagen object_pose keys from source HDF5, if available."""
    if h5py is None:
        return None
    try:
        with h5py.File(input_file, "r") as f:
            data_group = f.get("data")
            if data_group is None:
                return None
            demo_names = sorted(
                [name for name in data_group.keys() if name.startswith("demo_")],
                key=_demo_sort_key,
            )
            for demo_name in demo_names:
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
                return set(object_pose_group.keys())
    except Exception:
        return None
    return None


def _get_source_actions_mode(input_file: str) -> str | None:
    """Read /data attrs['actions_mode'] when present."""
    if h5py is None:
        return None
    try:
        with h5py.File(input_file, "r") as f:
            data_group = f.get("data")
            if data_group is None:
                return None
            mode = data_group.attrs.get("actions_mode")
            if mode is None:
                return None
            mode = _decode_attr(mode)
            return str(mode) if mode is not None else None
    except Exception:
        return None


def _get_first_demo_action_dim(input_file: str) -> int | None:
    """Read top-level /data/demo_*/actions second dimension from first demo."""
    if h5py is None:
        return None
    try:
        with h5py.File(input_file, "r") as f:
            data_group = f.get("data")
            if data_group is None:
                return None
            demo_names = sorted(
                [name for name in data_group.keys() if name.startswith("demo_")],
                key=_demo_sort_key,
            )
            for demo_name in demo_names:
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


def _validate_source_dataset_contract(
    input_file: str,
    expected_action_dim: int,
    required_actions_mode: str | None,
) -> None:
    """Validate source dataset action/datagen contract for strict generation."""
    action_dim = _get_first_demo_action_dim(input_file)
    if action_dim is None:
        raise ValueError("Strict preflight failed: unable to read top-level source action dimension.")
    if int(action_dim) != int(expected_action_dim):
        raise ValueError(
            "Strict preflight failed: unexpected source action dimension. "
            f"Expected {expected_action_dim}, found {action_dim}. "
            "Run scripts/utils/eef_action_process.py --to_ik before annotation/generation."
        )

    required_mode = (required_actions_mode or "").strip()
    if required_mode:
        actions_mode = _get_source_actions_mode(input_file)
        if actions_mode is None:
            raise ValueError(
                "Strict preflight failed: source dataset has no /data attrs['actions_mode']. "
                f"Expected '{required_mode}'."
            )
        if str(actions_mode).strip().lower() != required_mode.lower():
            raise ValueError(
                "Strict preflight failed: unexpected source actions_mode. "
                f"Expected '{required_mode}', found '{actions_mode}'."
            )


def _validate_first_demo_datagen_contract(input_file: str) -> None:
    """Validate first demo datagen_info structure used by Mimic generation."""
    if h5py is None:
        raise ImportError("Strict preflight requires h5py.")

    with h5py.File(input_file, "r") as f:
        data_group = f.get("data")
        if data_group is None:
            raise ValueError("Strict preflight failed: missing /data group.")
        demo_names = sorted(
            [name for name in data_group.keys() if name.startswith("demo_")],
            key=_demo_sort_key,
        )
        if not demo_names:
            raise ValueError("Strict preflight failed: no demo_* groups found.")

        demo_group = data_group[demo_names[0]]
        if "actions" not in demo_group:
            raise ValueError("Strict preflight failed: first demo missing top-level actions.")
        action_shape = tuple(np.asarray(demo_group["actions"]).shape)
        if len(action_shape) != 2:
            raise ValueError(
                f"Strict preflight failed: first demo actions must be [T,D], got {action_shape}."
            )
        num_steps = int(action_shape[0])

        obs_group = demo_group.get("obs")
        if obs_group is None:
            raise ValueError("Strict preflight failed: first demo missing /obs.")
        datagen_group = obs_group.get("datagen_info")
        if datagen_group is None:
            raise ValueError("Strict preflight failed: missing /obs/datagen_info.")

        for key in ("object_pose", "eef_pose", "target_eef_pose", "subtask_term_signals"):
            if key not in datagen_group:
                raise ValueError(f"Strict preflight failed: missing /obs/datagen_info/{key}.")

        for key in ("object_pose", "eef_pose", "target_eef_pose"):
            pose_group = datagen_group[key]
            if not isinstance(pose_group, h5py.Group) or len(pose_group.keys()) == 0:
                raise ValueError(f"Strict preflight failed: /obs/datagen_info/{key} is empty.")
            for pose_name in pose_group.keys():
                pose_shape = tuple(np.asarray(pose_group[pose_name]).shape)
                if len(pose_shape) != 3 or tuple(pose_shape[-2:]) != (4, 4):
                    raise ValueError(
                        "Strict preflight failed: invalid pose shape at "
                        f"/obs/datagen_info/{key}/{pose_name}: {pose_shape}."
                    )
                if int(pose_shape[0]) != num_steps:
                    raise ValueError(
                        "Strict preflight failed: horizon mismatch between actions and "
                        f"/obs/datagen_info/{key}/{pose_name} ({num_steps} vs {pose_shape[0]})."
                    )


def _pose_dict_first_center_cpu(pose_dict: Any) -> torch.Tensor | None:
    """Compute mean xyz over first timestep for a pose dict on CPU."""
    if not isinstance(pose_dict, dict) or len(pose_dict) == 0:
        return None
    points = []
    for pose_value in pose_dict.values():
        try:
            pose = torch.as_tensor(pose_value, dtype=torch.float32)
        except Exception:
            continue
        if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
            points.append(pose[0, :3, 3])
        elif pose.ndim == 2 and pose.shape == (4, 4):
            points.append(pose[:3, 3])
    if not points:
        return None
    return torch.stack(points, dim=0).mean(dim=0)


def _get_runtime_object_center(env: Any) -> torch.Tensor | None:
    """Get runtime object center from env.get_object_poses()."""
    try:
        object_poses = env.get_object_poses(env_ids=[0])
    except Exception:
        return None
    return _pose_dict_first_center_cpu(object_poses)


def _get_first_demo_pose_frame_stats(input_file: str, max_steps: int = 128) -> dict[str, float] | None:
    """Extract simple source-frame stats to detect mixed object/eef pose frames."""
    if h5py is None:
        return None
    try:
        with h5py.File(input_file, "r") as f:
            data_group = f.get("data")
            if data_group is None:
                return None
            demo_names = sorted(
                [name for name in data_group.keys() if name.startswith("demo_")],
                key=_demo_sort_key,
            )
            for demo_name in demo_names:
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


def _should_auto_fix_mixed_pose_frames(
    input_file: str, runtime_object_center: torch.Tensor | None
) -> tuple[bool, dict[str, float] | None]:
    """Heuristic detection of mixed-frame source data (object local, eef/target world)."""
    if runtime_object_center is None:
        return False, None
    stats = _get_first_demo_pose_frame_stats(input_file)
    if stats is None:
        return False, None
    runtime_z = float(runtime_object_center[2].item())
    source_z_shift = abs(runtime_z - stats["src_object_center_z"])
    # Mixed-frame signature:
    # 1) source object Z is far from runtime object Z, and
    # 2) source target/object Z gap is also large.
    should_fix = (source_z_shift > 0.20) and (stats["src_target_object_z_gap"] > 0.25)
    stats = dict(stats)
    stats["runtime_object_center_z"] = runtime_z
    stats["source_runtime_object_z_shift"] = source_z_shift
    return should_fix, stats


def _validate_subtask_object_refs(env: Any, input_file: str) -> None:
    """Ensure all mimic subtask object_refs are available at runtime and in source datagen_info."""
    subtask_cfgs = getattr(env.cfg, "subtask_configs", {})
    if not isinstance(subtask_cfgs, dict):
        return

    required_refs = sorted(
        {
            cfg.object_ref
            for cfgs in subtask_cfgs.values()
            for cfg in cfgs
            if getattr(cfg, "object_ref", None) is not None
        }
    )
    if not required_refs:
        return

    # Runtime check: env must expose these refs through get_object_poses().
    runtime_keys: set[str] = set()
    try:
        runtime_object_poses = env.get_object_poses(env_ids=[0])
        if isinstance(runtime_object_poses, dict):
            runtime_keys = set(runtime_object_poses.keys())
    except Exception as e:
        raise ValueError(f"Failed to query runtime object poses for mimic validation: {e}") from e

    missing_runtime = [name for name in required_refs if name not in runtime_keys]
    if missing_runtime:
        raise ValueError(
            "Mimic object_ref validation failed at runtime. Missing refs in env.get_object_poses(): "
            f"{missing_runtime}. Available keys: {sorted(runtime_keys)}"
        )

    # Source check: annotated dataset should include object_pose entries for all refs.
    source_keys = _get_first_demo_object_pose_keys(input_file)
    if source_keys is not None:
        missing_source = [name for name in required_refs if name not in source_keys]
        if missing_source:
            raise ValueError(
                "Mimic object_ref validation failed in source dataset datagen_info.object_pose. "
                f"Missing refs: {missing_source}. Available keys: {sorted(source_keys)}"
            )

    print(f"Validated Mimic object_refs: {required_refs}")


def _resolve_task_type(task_id: str, explicit_task_type: str | None) -> str:
    """Resolve task_type robustly for LeHome naming variants."""
    if explicit_task_type is not None:
        return explicit_task_type
    lowered = task_id.lower()
    if "biso101" in lowered or "biarm" in lowered or "bimanual" in lowered or lowered.startswith("lehome-bi"):
        return "bi-so101leader"
    return get_task_type(task_id, None)


def _normalize_last_subtask_offsets_for_generation(env_cfg: Any) -> None:
    """Mimic generation requires the last subtask term offset range to be (0, 0)."""
    subtask_cfgs = getattr(env_cfg, "subtask_configs", None)
    if not isinstance(subtask_cfgs, dict):
        return

    for eef_name, configs in subtask_cfgs.items():
        if not configs:
            continue
        last_cfg = configs[-1]
        current = getattr(last_cfg, "subtask_term_offset_range", None)
        if current is None:
            continue
        if tuple(current) != (0, 0):
            print(
                f"Warning: overriding final subtask_term_offset_range for '{eef_name}' "
                f"from {tuple(current)} to (0, 0) for Mimic generation compatibility."
            )
            last_cfg.subtask_term_offset_range = (0, 0)


def _is_supported_numeric_array(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)


def _load_episode_with_numeric_datasets_only(input_file: str, episode_name: str, device: str) -> EpisodeData:
    """Load episode while skipping non-numeric payloads (e.g., /meta string blobs)."""
    if h5py is None:
        raise ImportError(
            "h5py is required for numeric-only HDF5 loading. Install it with `pip install h5py`."
        )

    def _load_group(group: Any) -> dict:
        out = {}
        for key in group.keys():
            if key == "meta":
                continue
            node = group[key]
            if isinstance(node, h5py.Group):
                loaded = _load_group(node)
                if loaded:
                    out[key] = loaded
                continue
            arr = np.array(node)
            if not _is_supported_numeric_array(arr):
                continue
            out[key] = torch.tensor(arr, device=device)
        return out

    with h5py.File(input_file, "r") as file:
        data_group = file.get("data", None)
        if data_group is None or episode_name not in data_group:
            raise KeyError(f"Episode '{episode_name}' not found in {input_file}")

        h5_episode_group = data_group[episode_name]
        episode = EpisodeData()
        episode.data = _load_group(h5_episode_group)

        if "seed" in h5_episode_group.attrs:
            try:
                episode.seed = int(h5_episode_group.attrs["seed"])
            except Exception:
                episode.seed = h5_episode_group.attrs["seed"]
        if "success" in h5_episode_group.attrs:
            episode.success = bool(h5_episode_group.attrs["success"])

    return episode


def _load_episode_compat(
    dataset_file_handler: HDF5DatasetFileHandler,
    input_file: str,
    episode_name: str,
    device: str,
) -> EpisodeData:
    """Load episode robustly across mixed numeric/string HDF5 schemas."""
    try:
        episode = dataset_file_handler.load_episode(episode_name, device)
        if episode is None:
            raise ValueError(f"Episode '{episode_name}' not found.")
        return episode
    except TypeError as e:
        if "numpy.object_" not in str(e):
            raise
        print(
            "Info: default episode loader failed on non-numeric datasets"
            f" ({e}). Using numeric-only fallback loader."
        )
        return _load_episode_with_numeric_datasets_only(input_file, episode_name, device)


class RobustDataGenInfoPool(DataGenInfoPool):
    """Datagen info pool with fallback episode loading for mixed HDF5 schemas."""

    def __init__(
        self,
        *args,
        prefer_eef_pose_as_target: bool = False,
        source_target_z_offset: float = 0.0,
        align_object_pose_to_runtime: bool = False,
        align_object_pose_mode: str = "object_only",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._prefer_eef_pose_as_target = bool(prefer_eef_pose_as_target)
        self._source_target_z_offset = float(source_target_z_offset)
        self._align_object_pose_to_runtime = bool(align_object_pose_to_runtime)
        self._align_object_pose_mode = str(align_object_pose_mode).strip().lower()
        if self._align_object_pose_mode not in {"object_only", "all_poses"}:
            raise ValueError(
                f"Invalid align_object_pose_mode '{align_object_pose_mode}'. "
                "Expected one of: object_only, all_poses."
            )
        self._runtime_object_center = self._get_runtime_object_center()

    def _get_runtime_object_center(self) -> torch.Tensor | None:
        """Get current env object center from runtime world-frame object poses."""
        try:
            object_poses = self.env.get_object_poses(env_ids=[0])
        except Exception:
            return None
        return self._pose_dict_first_center(object_poses)

    def _pose_dict_first_center(self, pose_dict: Any) -> torch.Tensor | None:
        """Compute mean xyz over first timestep for a pose dict."""
        if not isinstance(pose_dict, dict) or len(pose_dict) == 0:
            return None
        points = []
        for pose_value in pose_dict.values():
            try:
                pose = torch.as_tensor(pose_value, device=self.device, dtype=torch.float32)
            except Exception:
                continue
            if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
                points.append(pose[0, :3, 3])
            elif pose.ndim == 2 and pose.shape == (4, 4):
                points.append(pose[:3, 3])
        if not points:
            return None
        return torch.stack(points, dim=0).mean(dim=0)

    def _apply_translation_to_pose_dict(self, pose_dict: Any, delta_xyz: torch.Tensor) -> Any:
        """Shift all pose translations by delta_xyz."""
        if not isinstance(pose_dict, dict):
            return pose_dict
        out = {}
        for key, value in pose_dict.items():
            try:
                pose = torch.as_tensor(value, device=self.device, dtype=torch.float32).clone()
            except Exception:
                out[key] = value
                continue
            if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
                pose[:, :3, 3] += delta_xyz.view(1, 3)
            elif pose.ndim == 2 and pose.shape == (4, 4):
                pose[:3, 3] += delta_xyz
            out[key] = pose
        return out

    def _apply_z_offset_to_pose_dict(self, pose_dict: Any) -> Any:
        """Apply z offset to a dict of 4x4 pose tensors."""
        if abs(self._source_target_z_offset) < 1e-9:
            return pose_dict
        if not isinstance(pose_dict, dict):
            return pose_dict

        out = {}
        for key, value in pose_dict.items():
            try:
                pose = torch.as_tensor(value, device=self.device, dtype=torch.float32).clone()
            except Exception:
                out[key] = value
                continue

            if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
                pose[:, 2, 3] += self._source_target_z_offset
            elif pose.ndim == 2 and pose.shape == (4, 4):
                pose[2, 3] += self._source_target_z_offset
            out[key] = pose
        return out

    def _maybe_override_target_eef_pose(self, episode: EpisodeData) -> None:
        """Prepare target_eef_pose used by Mimic source segments."""
        try:
            obs = episode.data.get("obs", {})
            datagen = obs.get("datagen_info", {})
            eef_pose = datagen.get("eef_pose")
            target_eef_pose = datagen.get("target_eef_pose")

            if self._prefer_eef_pose_as_target and eef_pose is not None:
                target_eef_pose = eef_pose
            if target_eef_pose is None:
                return
            datagen["target_eef_pose"] = self._apply_z_offset_to_pose_dict(target_eef_pose)
        except Exception:
            return

    def _maybe_align_object_pose_to_runtime(self, episode: EpisodeData, episode_name: str) -> None:
        """Align source pose frames to runtime world frame if a large global offset is detected."""
        if not self._align_object_pose_to_runtime:
            return
        if self._runtime_object_center is None:
            return
        try:
            obs = episode.data.get("obs", {})
            datagen = obs.get("datagen_info", {})
            object_pose = datagen.get("object_pose")
            if object_pose is None:
                return

            src_center = self._pose_dict_first_center(object_pose)
            if src_center is None:
                return
            delta = self._runtime_object_center - src_center
            # Only apply when offset is clearly non-trivial (legacy local-frame datagen_info).
            if float(torch.linalg.norm(delta).item()) < 0.15:
                return
            datagen["object_pose"] = self._apply_translation_to_pose_dict(object_pose, delta)
            if self._align_object_pose_mode == "all_poses":
                if "eef_pose" in datagen:
                    datagen["eef_pose"] = self._apply_translation_to_pose_dict(datagen.get("eef_pose"), delta)
                if "target_eef_pose" in datagen:
                    datagen["target_eef_pose"] = self._apply_translation_to_pose_dict(
                        datagen.get("target_eef_pose"), delta
                    )
            print(
                "Info: aligned source datagen poses to runtime frame for "
                f"{episode_name} with delta xyz=({float(delta[0]):+.4f}, {float(delta[1]):+.4f}, {float(delta[2]):+.4f}), "
                f"mode={self._align_object_pose_mode}"
            )
        except Exception:
            return

    def load_from_dataset_file(self, file_path, select_demo_keys: str | None = None):
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(file_path)
        try:
            episode_names = sorted(list(dataset_file_handler.get_episode_names()), key=_demo_sort_key)
            for episode_name in episode_names:
                if select_demo_keys is not None and episode_name not in select_demo_keys:
                    continue
                episode = _load_episode_compat(dataset_file_handler, file_path, episode_name, self.device)
                self._maybe_override_target_eef_pose(episode)
                self._maybe_align_object_pose_to_runtime(episode, episode_name)
                self._add_episode(episode)
        finally:
            dataset_file_handler.close()


def setup_async_generation_compat(
    env: Any,
    num_envs: int,
    input_file: str,
    success_term: Any,
    prefer_eef_pose_as_target: bool = False,
    source_target_z_offset: float = 0.0,
    align_object_pose_to_runtime: bool = False,
    align_object_pose_mode: str = "object_only",
    pause_subtask: bool = False,
    motion_planners: Any = None,
) -> dict[str, Any]:
    """Setup async generation with robust HDF5 datagen pool loading."""
    asyncio_event_loop = asyncio.get_event_loop()
    env_reset_queue = asyncio.Queue()
    env_action_queue = asyncio.Queue()
    shared_datagen_info_pool_lock = asyncio.Lock()
    shared_datagen_info_pool = RobustDataGenInfoPool(
        env,
        env.cfg,
        env.device,
        asyncio_lock=shared_datagen_info_pool_lock,
        prefer_eef_pose_as_target=prefer_eef_pose_as_target,
        source_target_z_offset=source_target_z_offset,
        align_object_pose_to_runtime=align_object_pose_to_runtime,
        align_object_pose_mode=align_object_pose_mode,
    )
    shared_datagen_info_pool.load_from_dataset_file(input_file)
    print(f"Loaded {shared_datagen_info_pool.num_datagen_infos} to datagen info pool")
    if prefer_eef_pose_as_target:
        print("Using measured datagen_info.eef_pose as source target trajectory (override enabled).")
    if abs(source_target_z_offset) > 1e-9:
        print(f"Applying source target z offset: {source_target_z_offset:+.4f} m")
    if align_object_pose_to_runtime:
        print(f"Applying source object-pose runtime alignment in mode: {align_object_pose_mode}")

    data_generator = DataGenerator(env=env, src_demo_datagen_info_pool=shared_datagen_info_pool)
    data_generator_asyncio_tasks = []
    for i in range(num_envs):
        env_motion_planner = motion_planners[i] if motion_planners else None
        task = asyncio_event_loop.create_task(
            mimic_generation.run_data_generator(
                env,
                i,
                env_reset_queue,
                env_action_queue,
                data_generator,
                success_term,
                pause_subtask=pause_subtask,
                motion_planner=env_motion_planner,
            )
        )
        data_generator_asyncio_tasks.append(task)

    return {
        "tasks": data_generator_asyncio_tasks,
        "event_loop": asyncio_event_loop,
        "reset_queue": env_reset_queue,
        "action_queue": env_action_queue,
        "info_pool": shared_datagen_info_pool,
    }


def _extract_first_xyz(pose: Any) -> list[float] | None:
    """Extract xyz position from a transform-like tensor/array."""
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


def _print_pose_snapshot(env: Any, step_count: int, env_id: int = 0) -> None:
    """Print positions for EEFs and garment keypoints."""
    eef_positions: dict[str, list[float]] = {}
    keypoint_positions: dict[str, list[float]] = {}

    subtask_cfgs = getattr(env.cfg, "subtask_configs", {})
    if isinstance(subtask_cfgs, dict) and len(subtask_cfgs) > 0:
        eef_names = list(subtask_cfgs.keys())
    else:
        eef_names = ["left_arm", "right_arm"]

    for eef_name in eef_names:
        try:
            eef_pose = env.get_robot_eef_pose(eef_name=eef_name, env_ids=[env_id])
        except Exception:
            continue
        xyz = _extract_first_xyz(eef_pose)
        if xyz is not None:
            eef_positions[eef_name] = xyz

    try:
        object_poses = env.get_object_poses(env_ids=[env_id])
        if isinstance(object_poses, dict):
            for key_name, key_pose in object_poses.items():
                xyz = _extract_first_xyz(key_pose)
                if xyz is not None:
                    keypoint_positions[key_name] = xyz
    except Exception as e:
        print(f"[pose] step={step_count} failed to read object poses: {e}")

    def _fmt_xyz(xyz: list[float]) -> str:
        return f"x={xyz[0]: .4f}  y={xyz[1]: .4f}  z={xyz[2]: .4f}"

    print(f"[pose] step={step_count}")
    if eef_positions:
        print("  eef:")
        for name in sorted(eef_positions.keys()):
            print(f"    - {name:<20} {_fmt_xyz(eef_positions[name])}")
    if keypoint_positions:
        print("  keypoints:")
        for name in sorted(keypoint_positions.keys()):
            print(f"    - {name:<20} {_fmt_xyz(keypoint_positions[name])}")


def env_loop_with_pose_output(
    env: ManagerBasedRLMimicEnv,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    asyncio_event_loop: asyncio.AbstractEventLoop,
    print_poses: bool = False,
    pose_output_interval: int = 1,
) -> None:
    """Main async loop for generation with optional pose output."""
    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    prev_num_attempts = 0
    step_count = 0
    action_dim = int(env.single_action_space.shape[0])

    if print_poses and pose_output_interval > 0:
        _print_pose_snapshot(env, step_count=0, env_id=0)

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while True:
            while env_action_queue.qsize() != env.num_envs:
                asyncio_event_loop.run_until_complete(asyncio.sleep(0))
                while not env_reset_queue.empty():
                    env_id_tensor[0] = env_reset_queue.get_nowait()
                    env.reset(env_ids=env_id_tensor)
                    env_reset_queue.task_done()

            # Keep action tensor shape explicit as [num_envs, action_dim].
            actions = torch.zeros((env.num_envs, action_dim), device=env.device)
            for _ in range(env.num_envs):
                env_id, action = asyncio_event_loop.run_until_complete(env_action_queue.get())
                action_tensor = torch.as_tensor(action, device=env.device).reshape(-1)
                if action_tensor.numel() != action_dim:
                    raise ValueError(
                        "Invalid action size from generator: "
                        f"expected {action_dim}, received {action_tensor.numel()}."
                    )
                actions[env_id] = action_tensor

            env.step(actions)
            for _ in range(env.num_envs):
                env_action_queue.task_done()

            step_count += 1
            if print_poses and pose_output_interval > 0 and (step_count % pose_output_interval == 0):
                _print_pose_snapshot(env, step_count=step_count, env_id=0)

            if prev_num_attempts != mimic_generation.num_attempts:
                prev_num_attempts = mimic_generation.num_attempts
                generated_sucess_rate = (
                    100 * mimic_generation.num_success / mimic_generation.num_attempts
                    if mimic_generation.num_attempts > 0
                    else 0.0
                )
                print("")
                print("*" * 50, "\033[K")
                print(
                    f"{mimic_generation.num_success}/{mimic_generation.num_attempts}"
                    f" ({generated_sucess_rate:.1f}%) successful demos generated by mimic\033[K"
                )
                print("*" * 50, "\033[K")

                generation_guarantee = env.cfg.datagen_config.generation_guarantee
                generation_num_trials = env.cfg.datagen_config.generation_num_trials
                check_val = mimic_generation.num_success if generation_guarantee else mimic_generation.num_attempts
                if check_val >= generation_num_trials:
                    print(f"Reached {generation_num_trials} successes/attempts. Exiting.")
                    break

            if env.sim.is_stopped():
                break

    env.close()


def main():
    num_envs = args_cli.num_envs

    # Setup output paths and get env name
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task.split(":")[-1] if args_cli.task else None
    dataset_env_args = _load_dataset_env_args(args_cli.input_file)
    env_name = task_name or dataset_env_args.get("env_name")
    if env_name is None:
        env_name = get_env_name_from_dataset(args_cli.input_file)

    # Configure environment
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
    )
    # Use env_name if task_name is None (env_name is guaranteed to have a value)
    task_id = task_name or env_name
    setattr(env_cfg, "task_type", _resolve_task_type(task_id, args_cli.task_type))
    setattr(env_cfg, "mimic_ik_orientation_weight", float(args_cli.mimic_ik_orientation_weight))
    _normalize_last_subtask_offsets_for_generation(env_cfg)
    print(f"Using mimic IK orientation_weight={float(args_cli.mimic_ik_orientation_weight):.4f}")

    # Keep garment task resource paths configurable from CLI.
    if hasattr(env_cfg, "garment_cfg_base_path"):
        env_cfg.garment_cfg_base_path = args_cli.garment_cfg_base_path
    if hasattr(env_cfg, "particle_cfg_path"):
        env_cfg.particle_cfg_path = args_cli.particle_cfg_path

    # Configure garment metadata for tasks that require explicit garment loading.
    if hasattr(env_cfg, "garment_name"):
        resolved_garment_name = (
            args_cli.garment_name
            or dataset_env_args.get("garment_name")
            or _get_first_demo_garment_name(args_cli.input_file)
            or getattr(env_cfg, "garment_name", None)
        )
        resolved_garment_version = (
            args_cli.garment_version
            or dataset_env_args.get("garment_version")
            or getattr(env_cfg, "garment_version", None)
        )

        if resolved_garment_name is None or (
            isinstance(resolved_garment_name, str) and not resolved_garment_name.strip()
        ):
            raise ValueError(
                "This task requires a garment_name, but none was provided. "
                "Pass --garment_name (e.g., Top_Long_Unseen_0) or include garment_name in data/env_args or"
                " /data/demo_*/meta."
            )

        env_cfg.garment_name = (
            resolved_garment_name.strip()
            if isinstance(resolved_garment_name, str)
            else resolved_garment_name
        )
        if hasattr(env_cfg, "garment_version") and resolved_garment_version is not None:
            env_cfg.garment_version = resolved_garment_version

        print(
            f"Using garment config: name={env_cfg.garment_name}, "
            f"version={getattr(env_cfg, 'garment_version', 'N/A')}"
        )

    # create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if hasattr(env, "_init_ik_solver_if_needed"):
        try:
            if not bool(env._init_ik_solver_if_needed()):
                raise RuntimeError("environment IK solver initialization returned False")
        except Exception as e:
            raise RuntimeError(
                "Generation requires a working IK solver in the environment, "
                f"but initialization failed: {e}"
            ) from e

    # check if the mimic API from this environment contains decprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # reset before starting
    env.reset()
    if hasattr(env, "initialize_obs"):
        try:
            env.initialize_obs()
        except Exception as e:
            print(f"Warning: initialize_obs failed during generation reset: {e}")

    if bool(args_cli.strict_preflight):
        _validate_source_dataset_contract(
            args_cli.input_file,
            expected_action_dim=int(args_cli.expected_source_action_dim),
            required_actions_mode=str(args_cli.require_source_actions_mode),
        )
        _validate_first_demo_datagen_contract(args_cli.input_file)
    _validate_subtask_object_refs(env, args_cli.input_file)

    explicit_object_alignment = bool(args_cli.align_object_pose_to_runtime) and (not args_cli.disable_object_pose_alignment)
    object_alignment_mode = str(args_cli.object_pose_alignment_mode).strip().lower()
    auto_object_alignment = False
    if (not explicit_object_alignment) and bool(args_cli.auto_fix_mixed_pose_frames):
        runtime_object_center = _get_runtime_object_center(env)
        should_fix, frame_stats = _should_auto_fix_mixed_pose_frames(args_cli.input_file, runtime_object_center)
        if should_fix:
            auto_object_alignment = True
            object_alignment_mode = "object_only"
            print(
                "Detected mixed source pose frames "
                f"(src_obj_z={frame_stats['src_object_center_z']:+.4f}, "
                f"src_target_z={frame_stats['src_target_center_z']:+.4f}, "
                f"runtime_obj_z={frame_stats['runtime_object_center_z']:+.4f}, "
                f"src_target_obj_z_gap={frame_stats['src_target_object_z_gap']:+.4f}). "
                "Auto-enabling object-only source object alignment."
            )

    # Keep default generation path identical to LeIsaac unless explicit source-pose
    # overrides are requested or mixed-frame source data requires compatibility correction.
    use_pose_overrides = (
        bool(args_cli.use_eef_pose_as_target)
        or abs(float(args_cli.source_target_z_offset)) > 1e-9
        or explicit_object_alignment
        or auto_object_alignment
    )
    if use_pose_overrides:
        print("Using compatibility datagen pipeline with source pose overrides.")
        async_components = setup_async_generation_compat(
            env=env,
            num_envs=args_cli.num_envs,
            input_file=args_cli.input_file,
            success_term=success_term,
            prefer_eef_pose_as_target=bool(args_cli.use_eef_pose_as_target),
            source_target_z_offset=args_cli.source_target_z_offset,
            align_object_pose_to_runtime=(explicit_object_alignment or auto_object_alignment),
            align_object_pose_mode=object_alignment_mode,
            pause_subtask=args_cli.pause_subtask,
        )
    else:
        async_components = mimic_generation.setup_async_generation(
            env=env,
            num_envs=args_cli.num_envs,
            input_file=args_cli.input_file,
            success_term=success_term,
            pause_subtask=args_cli.pause_subtask,
        )

    try:
        mimic_generation.num_success = 0
        mimic_generation.num_failures = 0
        mimic_generation.num_attempts = 0
        asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        if args_cli.print_poses:
            env_loop_with_pose_output(
                env,
                async_components["reset_queue"],
                async_components["action_queue"],
                async_components["event_loop"],
                print_poses=True,
                pose_output_interval=args_cli.pose_output_interval,
            )
        else:
            mimic_generation.env_loop(
                env,
                async_components["reset_queue"],
                async_components["action_queue"],
                async_components["info_pool"],
                async_components["event_loop"],
            )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # close sim app
    simulation_app.close()
