"""
Convert a local LeRobot dataset folder (v3.0 parquet-based layout) to a single HDF5 file
with the full IsaacLab recorder structure expected by annotate_demos.py.

Output HDF5 structure:
    data/
        demo_N/
            meta/
                garment_info.json  (UTF-8 JSON string for this episode only)
            actions              (T, 12) float32  — raw radians from source dataset
            processed_actions    (T-1, 12) float32
            obs/
                left_joint_pos          (T, 6)
                left_joint_pos_rel      (T, 6)
                left_joint_pos_target   (T, 6)
                left_joint_vel          (T, 6)
                left_joint_vel_rel      (T, 6)
                left_wrist              (T, 480, 640, 3) uint8  [placeholder zeros]
                right_joint_pos         (T, 6)
                right_joint_pos_rel     (T, 6)
                right_joint_pos_target  (T, 6)
                right_joint_vel         (T, 6)
                right_joint_vel_rel     (T, 6)
                right_wrist             (T, 480, 640, 3) uint8  [placeholder zeros]
                top                     (T, 480, 640, 3) uint8  [placeholder zeros]
                actions                 (T, 12)
            initial_state/
                articulation/
                    left_arm/           {root_pose(1,7), root_velocity(1,6),
                                         joint_position(1,6), joint_velocity(1,6)}
                    right_arm/          ...
                    StorageFurniture131/{root_pose(1,7), root_velocity(1,6),
                                         joint_position(1,16), joint_velocity(1,16)}
                rigid_object/
                    Bed002/             {root_pose(1,7), root_velocity(1,6)}
                    ...
            states/
                articulation/           same layout, shape (T-1, ...)
                rigid_object/           same layout, shape (T-1, ...)
    attrs:
        data.env_args  (JSON string)
        data.total     (int)
        demo_N.num_samples (int)
        demo_N.seed    (int)
        demo_N.success (bool)

Usage:
    python scripts/utils/convert_to_hdf5.py \\
        --lerobot_dir Datasets/record/001 \\
        --output_hdf5 Datasets/hdf5/001.hdf5

    # With custom scene config
    python scripts/utils/convert_to_hdf5.py \\
        --lerobot_dir Datasets/record/001 \\
        --output_hdf5 Datasets/hdf5/001.hdf5 \\
        --scene_config my_scene.json

    # Copy the reference structure from an existing HDF5
    python scripts/utils/convert_to_hdf5.py \\
        --lerobot_dir Datasets/record/001 \\
        --output_hdf5 Datasets/hdf5/001.hdf5 \\
        --reference_hdf5 Datasets/dataset.hdf5
"""

import argparse
import datetime as dt
import json
import os
import random
from contextlib import suppress
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Disable HDF5 file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ---------------------------------------------------------------------------
# Default scene entity configuration
# ---------------------------------------------------------------------------
# Keep defaults minimal to avoid injecting entities that are not present
# in the target scene. Override with --scene_config JSON or --reference_hdf5
# when explicit scene entities are needed.

DEFAULT_ARTICULATIONS = {
    "left_arm": {"num_joints": 6},
    "right_arm": {"num_joints": 6},
}

DEFAULT_RIGID_OBJECTS: list[str] = []

# Observation keys that are camera images (uint8, 480x640x3)
OBS_IMAGE_KEYS = ["left_wrist", "right_wrist", "top"]

# Observation keys that are float32 (T, 6) joint data
OBS_JOINT_KEYS = [
    "left_joint_pos",
    "left_joint_pos_rel",
    "left_joint_pos_target",
    "left_joint_vel",
    "left_joint_vel_rel",
    "right_joint_pos",
    "right_joint_pos_rel",
    "right_joint_pos_target",
    "right_joint_vel",
    "right_joint_vel_rel",
]


# ---------------------------------------------------------------------------
# Stats generation (kept for compatibility)
# ---------------------------------------------------------------------------


def generate_stats_if_missing(dataset_root: Path):
    """Check if stats files exist. If not, attempt to generate them.

    Accepts common LeRobot layouts:
    - meta/episodes_stats.jsonl
    - meta/stats.json or meta/stats.safetensors
    """

    meta_dir = dataset_root / "meta"
    episodes_stats_candidates = [
        meta_dir / "episodes_stats.jsonl",
        meta_dir / "episodes" / "episodes_stats.jsonl",
    ]
    stats_candidates = [
        meta_dir / "stats.json",
        meta_dir / "stats.safetensors",
    ]

    existing_episode_stats = next((p for p in episodes_stats_candidates if p.exists()), None)
    existing_stats = next((p for p in stats_candidates if p.exists()), None)

    if existing_episode_stats is not None and existing_stats is not None:
        print(
            "Stats files found, skipping generation:\n"
            f"  - {existing_episode_stats}\n"
            f"  - {existing_stats}"
        )
        return

    print("Missing stats files! Attempting to generate them...")
    print(
        "  Checked episode-stats paths: "
        + ", ".join(str(p) for p in episodes_stats_candidates)
    )
    print("  Checked stats paths: " + ", ".join(str(p) for p in stats_candidates))

    try:
        from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
    except ImportError:
        print("Warning: Could not import stats computation functions. Skipping stats generation.")
        return

    info_path = dataset_root / "meta/info.json"
    if not info_path.exists():
        print(f"Warning: meta/info.json not found in {dataset_root}, skipping stats.")
        return

    with open(info_path) as info_file:
        info = json.load(info_file)
    features = info.get("features", {})

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        print(f"Warning: No parquet files found in {data_dir}, skipping stats generation.")
        return

    try:
        all_stats = []
        episode_stats_list = []

        for parquet_path in tqdm(parquet_files, desc="Computing stats"):
            dataframe = pd.read_parquet(parquet_path)
            episode_data = {}
            for column_name in dataframe.columns:
                if column_name not in features:
                    continue
                column_values = dataframe[column_name].values
                if not isinstance(column_values, np.ndarray):
                    try:
                        column_values = np.array(column_values.tolist())
                    except Exception:
                        column_values = np.asarray(column_values)
                if (
                    column_values.dtype == object
                    and len(column_values) > 0
                    and isinstance(column_values[0], (list, tuple, np.ndarray))
                ):
                    with suppress(Exception):
                        column_values = np.stack(column_values)
                episode_data[column_name] = column_values

            stats = compute_episode_stats(episode_data, features)
            all_stats.append(stats)
            stats_with_index = stats.copy()
            stats_with_index["episode_index"] = int(
                parquet_path.stem.split("_")[-1]
                if "_" in parquet_path.stem
                else parquet_path.stem.split("-")[-1]
            )
            episode_stats_list.append(stats_with_index)

        print("Aggregating statistics...")
        aggregated = aggregate_stats(all_stats)

        def numpy_converter(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        stats_path = meta_dir / "stats.json"
        episodes_stats_path = meta_dir / "episodes_stats.jsonl"

        with open(stats_path, "w") as stats_file:
            json.dump(aggregated, stats_file, indent=4, default=numpy_converter)

        with open(episodes_stats_path, "w") as episodes_stats_file:
            for stat in episode_stats_list:
                clean_stat = json.loads(json.dumps(stat, default=numpy_converter))
                output_item = {
                    "episode_index": clean_stat["episode_index"],
                    "stats": {k: v for k, v in clean_stat.items() if k != "episode_index"},
                }
                episodes_stats_file.write(json.dumps(output_item) + "\n")

        print(f"Stats generated at {episodes_stats_path}")
    except Exception as e:
        print(f"Warning: Stats generation failed ({e}). Continuing without stats...")


# ---------------------------------------------------------------------------
# Scene config helpers
# ---------------------------------------------------------------------------


def load_scene_config_from_reference_hdf5(reference_path: str) -> tuple[dict, list]:
    """Extract articulation and rigid object names/shapes from a reference HDF5 file.

    Returns:
        (articulations_dict, rigid_objects_list)
    """
    articulations = {}
    rigid_objects = []

    with h5py.File(reference_path, "r") as ref_file:
        # Find the first demo
        data_group = ref_file["data"]
        demo_names = sorted(data_group.keys())
        if not demo_names:
            raise ValueError(f"No demos found in reference HDF5: {reference_path}")

        demo = data_group[demo_names[0]]

        # Extract articulations
        if "initial_state/articulation" in demo:
            art_group = demo["initial_state/articulation"]
            for name in art_group:
                num_joints = art_group[name]["joint_position"].shape[1]
                articulations[name] = {"num_joints": num_joints}

        # Extract rigid objects
        if "initial_state/rigid_object" in demo:
            rigid_objects = list(demo["initial_state/rigid_object"].keys())

    return articulations, rigid_objects


def load_scene_config_from_json(json_path: str) -> tuple[dict, list]:
    """Load scene config from a JSON file.

    Expected format:
    {
        "articulations": {
            "left_arm": {"num_joints": 6},
            "right_arm": {"num_joints": 6},
            "StorageFurniture131": {"num_joints": 16}
        },
        "rigid_objects": ["Bed002", "ClotheRack001", ...]
    }
    """
    with open(json_path) as f:
        config = json.load(f)
    return config["articulations"], config["rigid_objects"]


def detect_garment_name_from_meta(dataset_root: Path) -> str | None:
    """Auto-detect garment name from garment_info.json if available."""
    garment_info_path = resolve_garment_info_path(dataset_root)
    if not garment_info_path.exists():
        return None

    try:
        with open(garment_info_path, "r", encoding="utf-8") as file:
            garment_info = json.load(file)
        if isinstance(garment_info, dict) and garment_info:
            return next(iter(garment_info.keys()))
    except Exception as e:
        print(f"Warning: failed to parse {garment_info_path}: {e}")

    return None


def resolve_garment_info_path(dataset_root: Path) -> Path:
    """Resolve garment_info.json path from known dataset locations."""
    candidates = [
        dataset_root / "garment_info.json",
        dataset_root / "meta" / "garment_info.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def load_garment_info(dataset_root: Path) -> dict | None:
    """Load garment_info.json from dataset root/meta locations."""
    garment_info_path = resolve_garment_info_path(dataset_root)
    if not garment_info_path.exists():
        return None

    try:
        with open(garment_info_path, "r", encoding="utf-8") as file:
            garment_info = json.load(file)
    except Exception as e:
        print(f"Warning: failed to read/parse {garment_info_path}: {e}")
        return None

    if not isinstance(garment_info, dict):
        print(f"Warning: unexpected garment info format in {garment_info_path}; expected dict.")
        return None

    print(f"  Loaded garment metadata from: {garment_info_path}")
    return garment_info


def extract_episode_garment_info(
    garment_info: dict | None,
    episode_index: int,
    fallback_episode_index: int | None = None,
) -> dict:
    """Extract garment_info for one episode, preserving original top-level garment names."""
    if not garment_info:
        return {}

    target_keys = {str(episode_index)}
    if fallback_episode_index is not None:
        target_keys.add(str(fallback_episode_index))

    episode_payload = {}
    for garment_name, episodes in garment_info.items():
        if not isinstance(episodes, dict):
            continue
        matched = {}
        for key in target_keys:
            if key in episodes:
                matched[key] = episodes[key]
        if matched:
            episode_payload[garment_name] = matched

    return episode_payload


def write_episode_garment_info(
    episode_group: h5py.Group,
    garment_info: dict | None,
    source_episode_index: int,
    demo_index: int,
):
    """Write per-demo garment metadata to /data/demo_*/meta/garment_info.json."""
    episode_payload = extract_episode_garment_info(
        garment_info=garment_info,
        episode_index=source_episode_index,
        fallback_episode_index=demo_index,
    )

    content = json.dumps(episode_payload, ensure_ascii=False)
    meta_group = episode_group.require_group("meta")
    meta_group.create_dataset(
        "garment_info.json",
        data=content,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


# ---------------------------------------------------------------------------
# HDF5 structure builders
# ---------------------------------------------------------------------------


def create_articulation_state(
    group: h5py.Group,
    articulations: dict,
    num_frames: int,
    obs_state: np.ndarray | None = None,
):
    """Create articulation state groups with datasets.

    Args:
        group: HDF5 group to write into (e.g. initial_state or states).
        articulations: Dict of {name: {"num_joints": N}}.
        num_frames: Number of frames (1 for initial_state, T-1 for states).
        obs_state: Optional observation.state array (num_frames, 12) to populate
                   left_arm and right_arm joint positions.
    """
    art_group = group.require_group("articulation")

    for art_name, art_info in articulations.items():
        nj = art_info["num_joints"]
        entity_group = art_group.require_group(art_name)

        # Joint position
        jp = np.zeros((num_frames, nj), dtype=np.float32)
        # Populate arm joint positions from observation.state if available
        if obs_state is not None:
            if art_name == "left_arm" and nj == 6 and obs_state.shape[1] >= 6:
                jp[:, :6] = obs_state[:num_frames, :6]
            elif art_name == "right_arm" and nj == 6 and obs_state.shape[1] >= 12:
                jp[:, :6] = obs_state[:num_frames, 6:12]

        entity_group.create_dataset("joint_position", data=jp, compression="gzip")
        entity_group.create_dataset(
            "joint_velocity", data=np.zeros((num_frames, nj), dtype=np.float32), compression="gzip"
        )

        # Root pose: [x, y, z, qw, qx, qy, qz] — identity quaternion
        root_pose = np.zeros((num_frames, 7), dtype=np.float32)
        root_pose[:, 3] = 1.0  # qw = 1
        entity_group.create_dataset("root_pose", data=root_pose, compression="gzip")

        entity_group.create_dataset(
            "root_velocity", data=np.zeros((num_frames, 6), dtype=np.float32), compression="gzip"
        )


def create_rigid_object_state(
    group: h5py.Group,
    rigid_objects: list,
    num_frames: int,
):
    """Create rigid object state groups with datasets.

    Args:
        group: HDF5 group to write into (e.g. initial_state or states).
        rigid_objects: List of rigid object names.
        num_frames: Number of frames (1 for initial_state, T-1 for states).
    """
    # Don't create rigid_object group when there are no rigid objects configured.
    if not rigid_objects:
        return

    ro_group = group.require_group("rigid_object")

    for obj_name in rigid_objects:
        entity_group = ro_group.require_group(obj_name)

        root_pose = np.zeros((num_frames, 7), dtype=np.float32)
        root_pose[:, 3] = 1.0  # qw = 1
        entity_group.create_dataset("root_pose", data=root_pose, compression="gzip")

        entity_group.create_dataset(
            "root_velocity", data=np.zeros((num_frames, 6), dtype=np.float32), compression="gzip"
        )


def create_obs_group(
    episode_group: h5py.Group,
    joint_actions_array: np.ndarray,
    obs_state: np.ndarray | None,
    obs_ee_pose: np.ndarray | None,
    num_obs_frames: int,
    include_images: bool = False,
):
    """Create the obs/ group with named observation datasets.

    Args:
        episode_group: The demo group to write into.
        joint_actions_array: Joint-space actions array (T, 12) for obs/actions and target joints.
        obs_state: Observation state array (T, 12) — joint positions.
        obs_ee_pose: Optional observation.ee_pose array.
        num_obs_frames: Number of observation frames (same as T for actions).
        include_images: Whether to include placeholder image datasets.
    """
    obs_group = episode_group.require_group("obs")

    # Joint observation keys (T, 6) float32
    for key in OBS_JOINT_KEYS:
        if obs_state is not None:
            if key == "left_joint_pos":
                data = obs_state[:num_obs_frames, :6].copy()
            elif key == "right_joint_pos":
                data = obs_state[:num_obs_frames, 6:12].copy()
            elif key == "left_joint_pos_target":
                # Target positions = actions for left arm
                data = joint_actions_array[:num_obs_frames, :6].copy()
            elif key == "right_joint_pos_target":
                # Target positions = actions for right arm
                data = joint_actions_array[:num_obs_frames, 6:12].copy()
            elif key == "left_joint_pos_rel":
                # Relative = target - current
                data = (
                    joint_actions_array[:num_obs_frames, :6]
                    - obs_state[:num_obs_frames, :6]
                )
            elif key == "right_joint_pos_rel":
                data = (
                    joint_actions_array[:num_obs_frames, 6:12]
                    - obs_state[:num_obs_frames, 6:12]
                )
            else:
                # Velocity keys — zeros (not available from parquet)
                data = np.zeros((num_obs_frames, 6), dtype=np.float32)
        else:
            data = np.zeros((num_obs_frames, 6), dtype=np.float32)

        obs_group.create_dataset(key, data=data.astype(np.float32), compression="gzip")

    # obs/actions stays in joint-space for compatibility with existing replay/annotation flows.
    obs_group.create_dataset(
        "actions", data=joint_actions_array[:num_obs_frames], compression="gzip"
    )

    # Optional EE pose observations (from observation.ee_pose, if present).
    if obs_ee_pose is not None:
        ee_data = obs_ee_pose[:num_obs_frames].astype(np.float32, copy=False)
        obs_group.create_dataset("ee_pose", data=ee_data, compression="gzip")

        # Single-arm convention used in LeIsaac processed datasets:
        # ee_frame_state = [x, y, z, qx, qy, qz, qw] (gripper excluded).
        if ee_data.ndim == 2 and ee_data.shape[1] == 8:
            obs_group.create_dataset(
                "ee_frame_state", data=ee_data[:, :7], compression="gzip"
            )
        # Dual-arm convenience split for downstream processing.
        elif ee_data.ndim == 2 and ee_data.shape[1] == 16:
            obs_group.create_dataset(
                "left_ee_frame_state", data=ee_data[:, :7], compression="gzip"
            )
            obs_group.create_dataset(
                "right_ee_frame_state", data=ee_data[:, 8:15], compression="gzip"
            )

    # Image observations — placeholder zeros (very large, skip by default)
    if include_images:
        for img_key in OBS_IMAGE_KEYS:
            obs_group.create_dataset(
                img_key,
                data=np.zeros((num_obs_frames, 480, 640, 3), dtype=np.uint8),
                compression="gzip",
                compression_opts=1,
            )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert_lerobot_folder_to_hdf5(
    lerobot_dir: str,
    output_hdf5_path: str,
    task_name: str = "",
    reference_hdf5: str | None = None,
    scene_config: str | None = None,
    include_images: bool = False,
    garment_name: str | None = None,
    garment_version: str = "Release",
    actions_mode: str = "joint",
):
    """Convert a LeRobot dataset to HDF5 format matching the IsaacLab recorder structure.

    Args:
        lerobot_dir: Path to the LeRobot dataset root.
        output_hdf5_path: Path for the output HDF5 file.
        task_name: Task/env name to embed in HDF5.
        reference_hdf5: Path to a reference HDF5 file to extract scene entity names.
        scene_config: Path to a JSON scene config file.
        include_images: Whether to include placeholder image observations.
        garment_name: Garment name to store in HDF5 env_args (for garment tasks).
        garment_version: Garment version to store in HDF5 env_args.
        actions_mode: Action representation for top-level demo/actions:
            - "joint": use raw joint action (default)
            - "ee_pose": use action.ee_pose when available
    """
    dataset_root = Path(lerobot_dir).expanduser().resolve()
    output_path = Path(output_hdf5_path).expanduser().resolve()
    resolved_garment_name = garment_name or detect_garment_name_from_meta(dataset_root)

    # Determine scene entity configuration
    if reference_hdf5:
        print(f"Loading scene config from reference HDF5: {reference_hdf5}")
        articulations, rigid_objects = load_scene_config_from_reference_hdf5(reference_hdf5)
    elif scene_config:
        print(f"Loading scene config from JSON: {scene_config}")
        articulations, rigid_objects = load_scene_config_from_json(scene_config)
    else:
        print("Using default minimal scene config (arms only, no rigid objects).")
        articulations = DEFAULT_ARTICULATIONS
        rigid_objects = DEFAULT_RIGID_OBJECTS

    print(f"  Articulations: {list(articulations.keys())}")
    print(f"  Rigid objects: {rigid_objects}")

    # 1. Ensure stats exist
    generate_stats_if_missing(dataset_root)

    print(f"Loading LeRobotDataset from {dataset_root}...")
    dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)

    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  Frames:   {dataset.num_frames}")
    print(f"  FPS:      {dataset.fps}")
    if resolved_garment_name:
        print(f"  Garment:  {resolved_garment_name} ({garment_version})")
    else:
        print("  Garment:  not specified (annotation may require --garment_name)")

    # Discover available columns
    hf_columns = list(dataset.hf_dataset.column_names)
    has_obs_state = "observation.state" in hf_columns
    has_action = "action" in hf_columns
    has_obs_ee_pose = "observation.ee_pose" in hf_columns
    has_action_ee_pose = "action.ee_pose" in hf_columns

    if not has_action:
        raise ValueError("Dataset must contain 'action' column.")
    if not has_obs_state:
        print("Warning: 'observation.state' not found — obs and state groups will use zeros.")
    if not has_obs_ee_pose:
        print("Info: 'observation.ee_pose' not found — EE observation datasets will be skipped.")
    if actions_mode == "ee_pose" and not has_action_ee_pose:
        raise ValueError(
            "actions_mode=ee_pose requested, but dataset has no 'action.ee_pose'. "
            "Run `python -m scripts.dataset augment ...` first."
        )

    # Group global rows by episode index so each HDF5 demo maps to exactly one source episode.
    episode_column = [
        idx.item() if isinstance(idx, torch.Tensor) else idx
        for idx in dataset.hf_dataset["episode_index"]
    ]
    episode_to_rows: dict[int, list[int]] = {}
    for row_idx, episode_idx in enumerate(episode_column):
        episode_to_rows.setdefault(int(episode_idx), []).append(row_idx)

    episode_indices = sorted(episode_to_rows.keys())
    print(f"  Episode indices found: {episode_indices}")
    print(
        "  Episode lengths: "
        + ", ".join(f"{ep}:{len(episode_to_rows[ep])}" for ep in episode_indices)
    )

    # 2. Setup output HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_samples = 0
    garment_info = load_garment_info(dataset_root)
    if garment_info is None:
        print(
            f"Info: garment_info.json not found at {dataset_root / 'garment_info.json'} "
            f"or {dataset_root / 'meta' / 'garment_info.json'}; per-demo meta will be empty."
        )

    with h5py.File(output_path, "w") as hdf5:
        # Create the 'data' group with env_args
        data_group = hdf5.create_group("data")
        env_args = {"env_name": task_name, "type": 2}
        if resolved_garment_name:
            env_args["garment_name"] = resolved_garment_name
            env_args["garment_version"] = garment_version
        data_group.attrs["env_args"] = json.dumps(env_args)
        data_group.attrs["total"] = 0
        data_group.attrs["actions_mode"] = actions_mode

        # 3. Iterate episodes
        for demo_idx, ep_idx in enumerate(tqdm(episode_indices, desc="Converting to HDF5")):
            ep_row_indices = episode_to_rows.get(ep_idx, [])
            if not ep_row_indices:
                raise RuntimeError(
                    f"Episode {ep_idx} has no rows after grouping. Conversion aborted."
                )

            T = len(ep_row_indices)  # total frames in this episode

            # --- Extract action column ---
            action_col = dataset.hf_dataset["action"]
            action_list = [action_col[i] for i in ep_row_indices]
            if isinstance(action_list[0], torch.Tensor):
                actions_raw = torch.stack(action_list).numpy()
            else:
                actions_raw = np.array(action_list)

            # Keep raw source values (already stored in radians for this project).
            joint_actions_array = actions_raw.astype(np.float32, copy=False)

            # Optional action.ee_pose column
            action_ee_pose = None
            if has_action_ee_pose:
                action_ee_col = dataset.hf_dataset["action.ee_pose"]
                action_ee_list = [action_ee_col[i] for i in ep_row_indices]
                if isinstance(action_ee_list[0], torch.Tensor):
                    action_ee_pose = torch.stack(action_ee_list).numpy().astype(
                        np.float32, copy=False
                    )
                else:
                    action_ee_pose = np.array(action_ee_list, dtype=np.float32)

            # Top-level demo/actions can be kept as joint-space or switched to ee_pose.
            if actions_mode == "ee_pose":
                actions_array = action_ee_pose
            else:
                actions_array = joint_actions_array

            # --- Extract observation.state column ---
            obs_state = None
            if has_obs_state:
                obs_col = dataset.hf_dataset["observation.state"]
                obs_list = [obs_col[i] for i in ep_row_indices]
                if isinstance(obs_list[0], torch.Tensor):
                    obs_state_raw = torch.stack(obs_list).numpy()
                else:
                    obs_state_raw = np.array(obs_list)
                # Keep raw source values (already stored in radians for this project).
                obs_state = obs_state_raw.astype(np.float32, copy=False)

            # --- Extract observation.ee_pose column ---
            obs_ee_pose = None
            if has_obs_ee_pose:
                obs_ee_col = dataset.hf_dataset["observation.ee_pose"]
                obs_ee_list = [obs_ee_col[i] for i in ep_row_indices]
                if isinstance(obs_ee_list[0], torch.Tensor):
                    obs_ee_pose = torch.stack(obs_ee_list).numpy().astype(
                        np.float32, copy=False
                    )
                else:
                    obs_ee_pose = np.array(obs_ee_list, dtype=np.float32)

            # --- Create episode group ---
            ep_group = data_group.create_group(f"demo_{demo_idx}")
            ep_group.attrs["num_samples"] = T
            ep_group.attrs["source_episode_index"] = int(ep_idx)
            ep_group.attrs["seed"] = random.randint(0, 2**31 - 1)
            ep_group.attrs["success"] = True

            # --- per-demo garment metadata ---
            write_episode_garment_info(ep_group, garment_info, int(ep_idx), int(demo_idx))

            total_samples += T

            # --- actions (T, 12) ---
            ep_group.create_dataset("actions", data=actions_array, compression="gzip")

            # --- processed_actions (T-1, 12) ---
            # Post-step recorder captures processed actions after each step,
            # so there are T-1 entries (one per step after the first observation)
            if T > 1:
                ep_group.create_dataset(
                    "processed_actions", data=joint_actions_array[:-1], compression="gzip"
                )
            else:
                ep_group.create_dataset(
                    "processed_actions", data=joint_actions_array, compression="gzip"
                )

            # --- obs/ group (T frames) ---
            create_obs_group(
                ep_group,
                joint_actions_array,
                obs_state,
                obs_ee_pose,
                T,
                include_images=include_images,
            )

            # --- initial_state/ (1 frame — first frame) ---
            initial_state_group = ep_group.create_group("initial_state")
            initial_obs = obs_state[:1] if obs_state is not None else None
            create_articulation_state(initial_state_group, articulations, 1, initial_obs)
            create_rigid_object_state(initial_state_group, rigid_objects, 1)

            # --- states/ (T-1 frames — post-step states) ---
            states_group = ep_group.create_group("states")
            num_state_frames = max(T - 1, 1)
            states_obs = obs_state[1:] if obs_state is not None and T > 1 else obs_state
            create_articulation_state(states_group, articulations, num_state_frames, states_obs)
            create_rigid_object_state(states_group, rigid_objects, num_state_frames)

        # Update total sample count
        data_group.attrs["total"] = total_samples
        data_group.attrs["num_episodes"] = len(episode_indices)

        written_demos = len(data_group.keys())
        if written_demos != len(episode_indices):
            raise RuntimeError(
                f"Episode split mismatch: expected {len(episode_indices)} demos, wrote {written_demos}."
            )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Wrote HDF5 to: {output_path}")
    print(f"  File size:  {file_size_mb:.1f} MB")
    print(f"  Episodes:   {len(episode_indices)}")
    print(f"  Total frames: {total_samples}")
    print(f"  Images:     {'included (zeros)' if include_images else 'skipped'}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset (v3.0) to IsaacLab-compatible HDF5 format."
    )
    parser.add_argument(
        "--lerobot_dir",
        type=str,
        required=True,
        help="Path to local LeRobot dataset root directory.",
    )
    parser.add_argument(
        "--output_hdf5",
        type=str,
        required=True,
        help="Path to output HDF5 file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Task/env name to embed in HDF5 (for annotate_demos.py).",
    )
    parser.add_argument(
        "--reference_hdf5",
        type=str,
        default=None,
        help="Path to a reference HDF5 file to copy scene entity names from.",
    )
    parser.add_argument(
        "--scene_config",
        type=str,
        default=None,
        help="Path to a JSON file defining scene articulations and rigid objects.",
    )
    parser.add_argument(
        "--include_images",
        action="store_true",
        help="Include placeholder image observation datasets (large!).",
    )
    parser.add_argument(
        "--garment_name",
        type=str,
        default=None,
        help="Garment name to store in HDF5 env_args (auto-detected from meta/garment_info.json if omitted).",
    )
    parser.add_argument(
        "--garment_version",
        type=str,
        default="Release",
        help="Garment version to store in HDF5 env_args (default: Release).",
    )
    parser.add_argument(
        "--actions_mode",
        type=str,
        choices=["joint", "ee_pose"],
        default="joint",
        help=(
            "Representation for top-level data/demo_*/actions. "
            "'joint' keeps original joint actions; "
            "'ee_pose' uses action.ee_pose (similar to LeIsaac --to_ik stage)."
        ),
    )

    args = parser.parse_args()

    convert_lerobot_folder_to_hdf5(
        lerobot_dir=args.lerobot_dir,
        output_hdf5_path=args.output_hdf5,
        task_name=args.task,
        reference_hdf5=args.reference_hdf5,
        scene_config=args.scene_config,
        include_images=args.include_images,
        garment_name=args.garment_name,
        garment_version=args.garment_version,
        actions_mode=args.actions_mode,
    )


if __name__ == "__main__":
    main()
