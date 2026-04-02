"""Dataset processing utilities for augmenting and merging LeRobot datasets."""

from pathlib import Path
from typing import Optional, List, TYPE_CHECKING
import json
import shutil
import traceback
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.dataset_tools import merge_datasets as lerobot_merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.utils import RobotKinematics, compute_ee_pose_single_arm
from lehome.utils.logger import get_logger
from .data_formatting import normalize_depth_column_schema

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def compute_ee_pose_batch(
    solver: RobotKinematics,
    joint_batch: np.ndarray,
    state_unit: str,
    is_bimanual: bool,
) -> np.ndarray:
    """Compute end-effector poses for a batch of joint configurations.

    Returns:
        - Single-arm: shape (N, 8) - [x, y, z, qx, qy, qz, qw, gripper]
        - Dual-arm: shape (N, 16) - [left_8D, right_8D]
    """
    poses = []
    for idx, joints in enumerate(joint_batch):
        joints = np.asarray(joints, dtype=np.float32)
        try:
            if is_bimanual:
                left_joints = joints[:6]
                right_joints = joints[6:12]
                left_pose = compute_ee_pose_single_arm(solver, left_joints, state_unit)
                right_pose = compute_ee_pose_single_arm(
                    solver, right_joints, state_unit
                )
                poses.append(np.concatenate([left_pose, right_pose], axis=0))
            else:
                poses.append(compute_ee_pose_single_arm(solver, joints, state_unit))
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute EE pose for frame {idx} (joints: {joints}): {e}"
            ) from e

    return np.stack(poses, axis=0)


def add_ee_pose_to_parquet(
    parquet_path: Path,
    solver: RobotKinematics,
    state_unit: str,
    is_bimanual: bool,
    output_path: Path,
) -> None:
    """Add end-effector pose columns to a Parquet file."""
    table = pq.read_table(parquet_path)
    if "observation.state" not in table.column_names:
        raise KeyError(f"'observation.state' not found in {parquet_path}")
    if "action" not in table.column_names:
        raise KeyError(f"'action' not found in {parquet_path}")

    pose_dim = 16 if is_bimanual else 8

    obs_joint_batch = np.stack(table["observation.state"].to_pylist(), axis=0)
    obs_ee_pose = compute_ee_pose_batch(
        solver, obs_joint_batch, state_unit, is_bimanual
    )
    obs_ee_col = pa.array(obs_ee_pose.tolist(), type=pa.list_(pa.float32(), pose_dim))

    action_joint_batch = np.stack(table["action"].to_pylist(), axis=0)
    action_ee_pose = compute_ee_pose_batch(
        solver, action_joint_batch, state_unit, is_bimanual
    )
    action_ee_col = pa.array(
        action_ee_pose.tolist(), type=pa.list_(pa.float32(), pose_dim)
    )

    new_table = table.append_column("observation.ee_pose", obs_ee_col)
    new_table = new_table.append_column("action.ee_pose", action_ee_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, output_path)


def update_info_json(meta_path: Path, is_bimanual: bool, overwrite: bool) -> None:
    """Update dataset metadata (info.json) to include ee_pose feature definitions."""
    info_path = meta_path / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)
    feats = info.get("features", {})

    if ("observation.ee_pose" in feats or "action.ee_pose" in feats) and not overwrite:
        raise RuntimeError(
            "ee_pose features already exist; use --overwrite to replace."
        )

    if is_bimanual:
        ee_pose_feature = {
            "dtype": "float32",
            "shape": [16],
            "names": [
                "left_x",
                "left_y",
                "left_z",
                "left_qx",
                "left_qy",
                "left_qz",
                "left_qw",
                "left_gripper",
                "right_x",
                "right_y",
                "right_z",
                "right_qx",
                "right_qy",
                "right_qz",
                "right_qw",
                "right_gripper",
            ],
        }
    else:
        ee_pose_feature = {
            "dtype": "float32",
            "shape": [8],
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        }

    feats["observation.ee_pose"] = ee_pose_feature
    feats["action.ee_pose"] = ee_pose_feature

    info["features"] = feats
    with info_path.open("w") as f:
        json.dump(info, f, indent=4)


def augment_ee_pose(
    dataset_root: Path,
    urdf_path: Path,
    state_unit: str = "rad",
    output_root: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    """Add end-effector pose to existing datasets."""
    dataset_root = dataset_root.resolve()
    urdf_path = urdf_path.resolve()
    output_root = output_root.resolve() if output_root else dataset_root

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF path not found: {urdf_path}")

    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    joint_names = info["features"]["observation.state"]["names"]
    num_joints = len(joint_names)

    if num_joints == 6:
        is_bimanual = False
        solver_joint_names = joint_names[:5]
        print("✓ Detected single-arm dataset (6 DoF)")
    elif num_joints == 12:
        is_bimanual = True
        solver_joint_names = [n.replace("left_", "") for n in joint_names[:5]]
        print("✓ Detected dual-arm dataset (12 DoF)")
    else:
        raise ValueError(
            f"Unsupported number of joints: {num_joints}. "
            f"Only 6 (single-arm) or 12 (dual-arm) are supported."
        )

    solver = RobotKinematics(
        str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=solver_joint_names,
    )

    data_root = dataset_root / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_root}")

    total_files = len(parquet_files)
    print(f"📦 Processing {total_files} parquet file(s)...")

    for idx, src in enumerate(parquet_files, 1):
        rel = src.relative_to(dataset_root)
        dst = output_root / rel

        if dst.exists() and not overwrite:
            raise FileExistsError(
                f"{dst} exists; use --overwrite or set --output_root to new dir."
            )

        print(f"  [{idx}/{total_files}] {src.name}")
        try:
            add_ee_pose_to_parquet(src, solver, state_unit, is_bimanual, dst)
        except Exception as e:
            raise RuntimeError(f"Failed to process {src}: {e}") from e

    if output_root != dataset_root:
        print("📋 Copying meta, videos, and images...")
        for sub in ["meta", "videos", "images"]:
            src_dir = dataset_root / sub
            dst_dir = output_root / sub
            if src_dir.exists():
                if dst_dir.exists() and not overwrite:
                    raise FileExistsError(f"{dst_dir} exists; use --overwrite.")
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    update_info_json(output_root / "meta", is_bimanual, overwrite=overwrite)

    pose_dim = 16 if is_bimanual else 8
    print(f"✅ Done! Added ee_pose features ({pose_dim}D) to dataset.")


def merge_datasets(
    source_roots: List[Path],
    output_root: Path,
    output_repo_id: str = "merged_dataset",
    merge_custom_meta: bool = True,
) -> None:
    """Merge multiple LeRobot datasets including custom meta files.

    Args:
        source_roots: List of source dataset root directories
        output_root: Output dataset root directory
        output_repo_id: Repository ID for the merged dataset
        merge_custom_meta: Whether to merge custom meta files (garment_info.json)
    """
    # Validate source datasets
    for source_root in source_roots:
        if not source_root.exists():
            raise ValueError(f"Source dataset not found: {source_root}")
        if not (source_root / "meta").exists():
            raise ValueError(f"Meta directory not found in {source_root}")

    logger.info(f"Merging {len(source_roots)} datasets:")
    for i, root in enumerate(source_roots, 1):
        logger.info(f"  {i}. {root}")
    logger.info(f"Output: {output_root}")

    # Normalize depth column schema if needed (to avoid ArrowTypeError)
    for source_root in source_roots:
        try:
            normalize_depth_column_schema(source_root)
        except Exception as e:
            logger.warning(f"Depth format normalization failed for {source_root}: {e}")

    # Load all source datasets
    datasets = []
    for source_root in source_roots:
        repo_id = source_root.name
        try:
            dataset = LeRobotDataset(repo_id=repo_id, root=source_root)
            datasets.append(dataset)
            logger.info(
                f"Loaded dataset: {repo_id} ({dataset.meta.total_episodes} episodes) from {dataset.root}"
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {repo_id}: {e}")
            logger.error(f"  Source root: {source_root}")
            traceback.print_exc()
            raise

    # Merge datasets
    logger.info("Starting dataset merge...")
    merged_dataset = lerobot_merge_datasets(
        datasets=datasets,
        output_repo_id=output_repo_id,
        output_dir=output_root,
    )

    logger.info(f"Merged dataset created:")
    logger.info(f"  Total episodes: {merged_dataset.meta.total_episodes}")
    logger.info(f"  Total frames: {merged_dataset.meta.total_frames}")
    logger.info(f"  Location: {output_root}")

    # Merge custom meta files
    if merge_custom_meta:
        logger.info("Merging custom meta files...")
        merge_garment_info(source_roots, output_root)
        logger.info("Custom meta files merged successfully")

    logger.info("Dataset merge completed!")


def merge_garment_info(source_roots: List[Path], output_root: Path) -> int:
    """Merge garment_info.json files from multiple datasets.

    Format:
    {
      "Top_Long_Unseen_0": {
        "0": {"object_initial_pose": [...], "scale": [...]},
        "1": {"object_initial_pose": [...], "scale": [...]}
      }
    }

    Args:
        source_roots: List of source dataset root directories
        output_root: Output dataset root directory

    Returns:
        Total number of episodes merged
    """
    output_path = output_root / "meta" / "garment_info.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_data = {}
    episode_offset = 0
    total_merged = 0

    for source_root in source_roots:
        source_path = source_root / "meta" / "garment_info.json"

        if not source_path.exists():
            logger.warning(f"garment_info.json not found in {source_root}, skipping...")
            info_path = source_root / "meta" / "info.json"
            if info_path.exists():
                with open(info_path, "r") as f:
                    episode_offset += json.load(f).get("total_episodes", 0)
            continue

        logger.info(f"Merging garment_info.json from {source_root}")
        count = 0

        try:
            with open(source_path, "r", encoding="utf-8") as f:
                source_data = json.load(f)

            for garment_name, episodes in source_data.items():
                if garment_name not in merged_data:
                    merged_data[garment_name] = {}

                for episode_key, episode_data in episodes.items():
                    try:
                        old_idx = int(episode_key)
                        new_key = str(old_idx + episode_offset)
                        merged_data[garment_name][new_key] = episode_data.copy()
                        count += 1
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Invalid episode key '{episode_key}' in {source_path}: {e}"
                        )
                        continue

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to parse {source_path}: {e}")
            continue

        total_merged += count
        logger.info(f"  Merged {count} episodes from {source_root}")

        # Update episode offset for next dataset
        info_path = source_root / "meta" / "info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                episode_offset += json.load(f).get("total_episodes", count)
        else:
            episode_offset += count

    # Sort by garment_name and episode indices
    sorted_data = {}
    for garment_name in sorted(merged_data.keys()):
        episodes = merged_data[garment_name]
        sorted_episodes = {
            str(k): episodes[str(k)]
            for k in sorted(int(key) for key in episodes.keys())
        }
        sorted_data[garment_name] = sorted_episodes

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Total merged {total_merged} episodes from {len(sorted_data)} garments to {output_path}"
    )
    return total_merged
