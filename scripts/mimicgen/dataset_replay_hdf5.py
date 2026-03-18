"""Dataset replay utility functions for replaying recorded HDF5 episodes."""

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.tasks.fold_cloth.checkpoint_mappings import (
    CSV_TRACE_KEYPOINT_NAMES,
    semantic_keypoints_from_positions,
)
from lehome.utils.record import RateLimiter
from lehome.utils.logger import get_logger

from ..utils.common import stabilize_garment_after_reset

logger = get_logger(__name__)

try:
    import h5py
except ImportError:
    h5py = None


def _require_h5py() -> None:
    """Ensure h5py is available before running HDF5 operations."""
    if h5py is None:
        raise ImportError(
            "h5py is required for HDF5 replay. "
            "Install it in your environment (e.g. `pip install h5py`)."
        )


def _decode_attr(value: Any) -> Any:
    """Decode HDF5 attribute values to plain Python types when needed."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_python_scalar_or_list(value: Any) -> Any:
    """Convert h5/ndarray values to JSON-friendly python types."""
    value = _decode_attr(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode_attr(value.item())
        return value.tolist()
    return value


def _parse_json_if_possible(value: Any) -> Any:
    """Parse JSON string values when they are serialized dict/list content."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, (dict, list)):
            return parsed
    return value


def _read_hdf5_node(node: Any) -> Any:
    """Recursively read HDF5 groups/datasets to plain Python objects."""
    if h5py is None:
        return None

    if isinstance(node, h5py.Dataset):
        value = _to_python_scalar_or_list(node[()])
        return _parse_json_if_possible(value)

    if isinstance(node, h5py.Group):
        out: Dict[str, Any] = {}
        for key, val in node.attrs.items():
            out[key] = _parse_json_if_possible(_to_python_scalar_or_list(val))
        for key in node.keys():
            out[key] = _read_hdf5_node(node[key])
        return out

    return node


def _demo_sort_key(name: str) -> Tuple[int, int, str]:
    """Sort numbered demo groups before any non-standard dataset entries."""
    if name.startswith("demo_"):
        suffix = name.split("demo_", maxsplit=1)[1]
        if suffix.isdigit():
            return 0, int(suffix), name
    return 1, 0, name


class GarmentKeypointDebugMarkers:
    """Viewport-only garment keypoint marker overlay for replay debugging."""

    _SEMANTIC_KEYPOINT_NAMES = CSV_TRACE_KEYPOINT_NAMES
    _MARKER_COLORS = (
        (0.0, 1.0, 1.0),
        (1.0, 0.5, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    )
    _MARKER_INDICES = np.arange(len(_SEMANTIC_KEYPOINT_NAMES), dtype=np.int32)

    def __init__(self):
        markers = {}
        for name, color in zip(self._SEMANTIC_KEYPOINT_NAMES, self._MARKER_COLORS):
            markers[name] = sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        cfg = VisualizationMarkersCfg(
            prim_path="/World/Debug/GarmentKeypoints",
            markers=markers,
        )
        self._markers = VisualizationMarkers(cfg)
        self._markers.set_visibility(False)
        self._disabled = False
        self._warned_missing_keypoints = False
        self._warned_update_failure = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    def update_from_env(self, env: DirectRLEnv) -> None:
        """Update marker positions from the current cloth state."""
        if self._disabled:
            return

        try:
            translations = self._extract_semantic_keypoint_positions(env)
        except Exception as exc:
            if not self._warned_update_failure:
                logger.warning(
                    f"Disabling debugging markers after update failure: {exc}",
                    exc_info=True,
                )
                self._warned_update_failure = True
            self.disable()
            return

        if translations is None:
            if not self._warned_missing_keypoints:
                logger.warning(
                    "Garment semantic keypoints are unavailable in the current replay env. "
                    "Hiding debugging markers."
                )
                self._warned_missing_keypoints = True
            self._markers.set_visibility(False)
            return

        self._markers.set_visibility(True)
        self._markers.visualize(
            translations=translations,
            marker_indices=self._MARKER_INDICES,
        )

    def disable(self) -> None:
        """Permanently disable the marker overlay for the current run."""
        if self._disabled:
            return
        self._disabled = True
        try:
            self._markers.set_visibility(False)
        except Exception:
            pass

    def _extract_semantic_keypoint_positions(self, env: DirectRLEnv) -> Optional[np.ndarray]:
        garment_obj = getattr(env, "object", None)
        if garment_obj is None or not hasattr(garment_obj, "check_points"):
            return None

        check_points = getattr(garment_obj, "check_points", None)
        if check_points is None or len(check_points) < len(self._SEMANTIC_KEYPOINT_NAMES):
            return None

        mesh_points = None
        try:
            mesh_points_world, _, _, _ = garment_obj.get_current_mesh_points()
            mesh_points = np.asarray(mesh_points_world, dtype=np.float32)
        except Exception:
            cloth_prim_view = getattr(garment_obj, "_cloth_prim_view", None)
            if cloth_prim_view is None:
                return None
            try:
                mesh_points = (
                    cloth_prim_view.get_world_positions()
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            except Exception:
                return None

        kp_positions = mesh_points[list(check_points)]
        semantic_points = semantic_keypoints_from_positions(kp_positions)
        return np.stack(
            [np.asarray(semantic_points[name], dtype=np.float32) for name in self._SEMANTIC_KEYPOINT_NAMES],
            axis=0,
        )


class HDF5ReplaySource:
    """Reader for IsaacLab-style HDF5 replay datasets."""

    def __init__(self, hdf5_path: str):
        _require_h5py()
        self.path = Path(hdf5_path)
        self.file = h5py.File(self.path, "r")

        if "data" not in self.file:
            self.file.close()
            raise ValueError(f"HDF5 missing required '/data' group: {self.path}")

        self.data_group = self.file["data"]
        self.demo_names = sorted(
            [name for name in self.data_group.keys() if name.startswith("demo_")],
            key=_demo_sort_key,
        )

        if len(self.demo_names) == 0:
            self.file.close()
            raise ValueError(f"No demo groups found under '/data' in {self.path}")

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "HDF5ReplaySource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def num_episodes(self) -> int:
        return len(self.demo_names)

    @property
    def fps(self) -> int:
        raw = self.data_group.attrs.get("fps", 30)
        try:
            return int(_decode_attr(raw))
        except (TypeError, ValueError):
            return 30

    def get_demo_group(self, episode_index: int) -> Any:
        return self.data_group[self.demo_names[episode_index]]

    def get_source_episode_index(self, episode_index: int) -> int:
        demo = self.get_demo_group(episode_index)
        if "source_episode_index" in demo.attrs:
            try:
                return int(_decode_attr(demo.attrs["source_episode_index"]))
            except (TypeError, ValueError):
                return episode_index
        return episode_index

    def get_env_args(self) -> Dict[str, Any]:
        raw_env_args = self.data_group.attrs.get("env_args")
        if raw_env_args is None:
            return {}
        try:
            value = _decode_attr(raw_env_args)
            if isinstance(value, str):
                return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return {}

    def get_episode_meta(self, episode_index: int) -> Dict[str, Any]:
        """Read demo-level metadata from /data/demo_*/meta or synthesize it from initial_state."""
        demo = self.get_demo_group(episode_index)
        parsed: Dict[str, Any] = {}
        if "meta" in demo:
            meta_node = demo["meta"]
            meta_value = _read_hdf5_node(meta_node)
            if isinstance(meta_value, dict):
                parsed.update(meta_value)

        initial_state = demo.get("initial_state")
        garment_group = None if initial_state is None else initial_state.get("garment")
        if garment_group is None:
            return parsed

        for garment_name in garment_group.keys():
            garment_entry = garment_group[garment_name]
            garment_meta: Dict[str, Any] = {}

            if "initial_pose" in garment_entry:
                pose_value = _to_python_scalar_or_list(garment_entry["initial_pose"][()])
                if isinstance(pose_value, list) and len(pose_value) == 1 and isinstance(pose_value[0], list):
                    pose_value = pose_value[0]
                garment_meta["initial_pose"] = _parse_json_if_possible(pose_value)

            if "scale" in garment_entry:
                scale_value = _to_python_scalar_or_list(garment_entry["scale"][()])
                if isinstance(scale_value, list) and len(scale_value) == 1 and isinstance(scale_value[0], list):
                    scale_value = scale_value[0]
                garment_meta["scale"] = _parse_json_if_possible(scale_value)

            if garment_meta:
                parsed.setdefault(str(garment_name), {}).update(garment_meta)

        return parsed

    def get_garment_name_from_env_args(self) -> Optional[str]:
        env_args = self.get_env_args()
        garment_name = env_args.get("garment_name")
        if isinstance(garment_name, str) and garment_name:
            return garment_name
        return None

    def has_ee_pose(self) -> bool:
        demo = self.get_demo_group(0)
        if "obs" in demo and "ee_pose" in demo["obs"]:
            return True

        if "actions" in demo:
            actions = np.asarray(demo["actions"][:])
            if actions.ndim == 2 and actions.shape[1] in (8, 16):
                has_joint_actions = "obs" in demo and "actions" in demo["obs"]
                actions_mode = _decode_attr(self.data_group.attrs.get("actions_mode", ""))
                actions_mode = str(actions_mode).lower()
                return has_joint_actions or actions_mode == "ee_pose"

        return False

    def get_state_dim(self) -> int:
        demo = self.get_demo_group(0)
        joint_actions = self._extract_joint_actions(demo)
        obs_state = self._extract_observation_state(demo, joint_actions)
        return int(obs_state.shape[1])

    def get_episode_frames(
        self, episode_index: int, require_ee_pose: bool = False
    ) -> List[Dict[str, torch.Tensor]]:
        """Load one episode into replay frame format used by replay_episode()."""
        demo = self.get_demo_group(episode_index)

        joint_actions = self._extract_joint_actions(demo)
        obs_state = self._extract_observation_state(demo, joint_actions)
        ee_pose = self._extract_action_ee_pose(demo)

        if require_ee_pose and ee_pose is None:
            raise ValueError(
                f"{self.demo_names[episode_index]} does not contain action ee_pose data."
            )

        total_frames = min(joint_actions.shape[0], obs_state.shape[0])
        if ee_pose is not None:
            total_frames = min(total_frames, ee_pose.shape[0])

        if total_frames <= 0:
            return []

        frames: List[Dict[str, torch.Tensor]] = []
        for idx in range(total_frames):
            frame = {
                "action": torch.from_numpy(joint_actions[idx]).float(),
                "observation.state": torch.from_numpy(obs_state[idx]).float(),
            }
            if ee_pose is not None:
                frame["action.ee_pose"] = torch.from_numpy(ee_pose[idx]).float()
            frames.append(frame)

        return frames

    def _extract_joint_actions(self, demo: Any) -> np.ndarray:
        """Prefer joint-space actions from obs/actions, fallback to top-level actions."""
        if "obs" in demo and "actions" in demo["obs"]:
            actions = np.asarray(demo["obs"]["actions"][:], dtype=np.float32)
        elif "actions" in demo:
            actions = np.asarray(demo["actions"][:], dtype=np.float32)
        else:
            raise ValueError("Demo missing both 'obs/actions' and top-level 'actions'.")

        if actions.ndim != 2:
            raise ValueError(f"Expected 2D actions array, got shape={actions.shape}")

        return actions

    def _extract_action_ee_pose(self, demo: Any) -> Optional[np.ndarray]:
        """Load ee_pose actions when available."""
        if "obs" in demo and "ee_pose" in demo["obs"]:
            ee_pose = np.asarray(demo["obs"]["ee_pose"][:], dtype=np.float32)
            if ee_pose.ndim == 2 and ee_pose.shape[1] in (8, 16):
                return ee_pose
            return None

        if "actions" not in demo:
            return None

        top_level_actions = np.asarray(demo["actions"][:], dtype=np.float32)
        if top_level_actions.ndim != 2 or top_level_actions.shape[1] not in (8, 16):
            return None

        has_joint_actions = "obs" in demo and "actions" in demo["obs"]
        actions_mode = _decode_attr(self.data_group.attrs.get("actions_mode", ""))
        actions_mode = str(actions_mode).lower()

        if has_joint_actions or actions_mode == "ee_pose":
            return top_level_actions

        return None

    def _extract_observation_state(
        self, demo: Any, joint_actions: np.ndarray
    ) -> np.ndarray:
        """Build observation.state from available HDF5 observation/state fields."""
        if "obs" in demo:
            obs = demo["obs"]

            if "left_joint_pos" in obs and "right_joint_pos" in obs:
                left = np.asarray(obs["left_joint_pos"][:], dtype=np.float32)
                right = np.asarray(obs["right_joint_pos"][:], dtype=np.float32)
                num_frames = min(left.shape[0], right.shape[0])
                if num_frames > 0:
                    return np.concatenate([left[:num_frames], right[:num_frames]], axis=1)

            if "left_joint_pos" in obs:
                left = np.asarray(obs["left_joint_pos"][:], dtype=np.float32)
                if left.ndim == 2 and left.shape[0] > 0:
                    return left

        initial_state = self._collect_articulation_joint_positions(
            demo.get("initial_state")
        )
        states = self._collect_articulation_joint_positions(demo.get("states"))
        if initial_state is not None and states is not None:
            return np.concatenate([initial_state[:1], states], axis=0)

        # Fallback: use the joint actions as current-state proxy.
        return joint_actions.copy()

    def _collect_articulation_joint_positions(self, group: Any) -> Optional[np.ndarray]:
        """Collect and concatenate articulation joint positions from a group."""
        if group is None or "articulation" not in group:
            return None

        articulation_group = group["articulation"]
        part_arrays: List[np.ndarray] = []
        num_frames: Optional[int] = None

        for name in sorted(articulation_group.keys()):
            entity_group = articulation_group[name]
            if "joint_position" not in entity_group:
                continue

            arr = np.asarray(entity_group["joint_position"][:], dtype=np.float32)
            if arr.ndim != 2:
                continue

            part_arrays.append(arr)
            num_frames = arr.shape[0] if num_frames is None else min(num_frames, arr.shape[0])

        if len(part_arrays) == 0 or num_frames is None or num_frames <= 0:
            return None

        trimmed = [arr[:num_frames] for arr in part_arrays]
        return np.concatenate(trimmed, axis=1)


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments for HDF5 dataset replay."""
    dataset_path = Path(args.dataset_root)
    if not dataset_path.exists():
        raise ValueError(f"HDF5 dataset file does not exist: {args.dataset_root}")
    if not dataset_path.is_file():
        raise ValueError(f"dataset_root must be an HDF5 file path, got: {args.dataset_root}")
    if dataset_path.suffix.lower() not in {".hdf5", ".h5"}:
        raise ValueError(
            f"dataset_root must point to a .hdf5/.h5 file, got: {args.dataset_root}"
        )

    with HDF5ReplaySource(args.dataset_root) as dataset:
        if dataset.num_episodes == 0:
            raise ValueError(f"HDF5 dataset has no episodes: {args.dataset_root}")

        if args.start_episode >= dataset.num_episodes:
            raise ValueError(
                f"start_episode ({args.start_episode}) is out of range. "
                f"Dataset has {dataset.num_episodes} episodes."
            )

    if args.num_replays < 1:
        raise ValueError(f"num_replays must be >= 1, got {args.num_replays}")

    if args.start_episode < 0:
        raise ValueError(f"start_episode must be >= 0, got {args.start_episode}")

    if args.end_episode is not None:
        if args.end_episode < 0:
            raise ValueError(f"end_episode must be >= 0, got {args.end_episode}")
        if args.end_episode <= args.start_episode:
            raise ValueError(
                f"end_episode ({args.end_episode}) must be > start_episode ({args.start_episode})"
            )

    if getattr(args, "debugging_markers", False) and args.output_root is not None:
        raise ValueError(
            "--debugging_markers is viewport-only and cannot be used together with --output_root."
        )

    if getattr(args, "debugging_markers", False) and getattr(args, "headless", False):
        logger.warning(
            "--debugging_markers was enabled in headless mode. Replay will continue, "
            "but the markers will not be visible."
        )


def _try_read_garment_name_from_json(json_path: Path) -> Optional[str]:
    """Read garment name (top-level key) from garment_info.json."""
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and len(data) > 0:
            first_key = next(iter(data.keys()))
            if isinstance(first_key, str) and first_key:
                return first_key
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _find_pose_list_in_obj(obj: Any) -> Optional[List[float]]:
    """Find a pose list in a generic metadata object."""
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


def _extract_garment_name_from_episode_meta(meta: Dict[str, Any]) -> Optional[str]:
    """Extract garment name from per-episode /meta content."""
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


def _extract_initial_pose_from_episode_meta(
    meta: Dict[str, Any],
    source_episode_index: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Extract initial pose in env.set_all_pose format from episode /meta."""
    garment_info = meta.get("garment_info")
    if garment_info is None:
        garment_info = meta.get("garment_info.json")

    if isinstance(garment_info, dict):
        for _, episodes in garment_info.items():
            if not isinstance(episodes, dict):
                continue

            preferred_keys: List[str] = []
            if source_episode_index is not None:
                preferred_keys.append(str(source_episode_index))
            preferred_keys.extend(sorted(episodes.keys()))

            seen = set()
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

    pose = _find_pose_list_in_obj(meta)
    if pose is None:
        return None
    return {"Garment": pose}


def find_garment_info_json(hdf5_path: Path) -> Optional[Path]:
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
    pose_file: Optional[Path], source_episode_index: int
) -> Optional[Dict[str, Any]]:
    """Load initial pose from garment_info.json using source episode index."""
    if pose_file is None:
        return None
    if not pose_file.exists():
        return None

    try:
        with open(pose_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to parse {pose_file}: {e}")
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

    logger.warning(
        f"Initial pose not found in {pose_file} for source episode {source_episode_index}"
    )
    return None


def create_replay_dataset(
    args: argparse.Namespace, source_hdf5_path: Path, fps: int
) -> Tuple[Optional[LeRobotDataset], Optional[Path]]:
    """Create a dataset for saving replayed episodes."""
    if args.output_root is None:
        return None, None

    output_path = Path(args.output_root)
    root = output_path / source_hdf5_path.stem
    if root.exists():
        logger.warning(
            f"Target path {root} already exists. Deleting it to create a fresh replay dataset."
        )
        shutil.rmtree(root)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating replay dataset at: {root}")
    replay_dataset = LeRobotDataset.create(
        repo_id="replay_output",
        fps=fps,
        root=root,
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=None,
    )

    json_path = replay_dataset.root / "meta" / "garment_info.json"
    return replay_dataset, json_path


def compute_action_from_ee_pose(
    env: DirectRLEnv,
    frame_data: Dict[str, torch.Tensor],
    ik_solver: Any,
    is_bimanual: bool,
    args: argparse.Namespace,
    ik_stats: Dict[str, Any],
    device: str = "cpu",
) -> Optional[torch.Tensor]:
    """Compute joint angles from action.ee_pose using inverse kinematics."""
    from lehome.utils import compute_joints_from_ee_pose

    try:
        if "action.ee_pose" not in frame_data:
            logger.warning(
                "action.ee_pose not found in frame data, falling back to original action"
            )
            ik_stats["total"] += 1
            ik_stats["fallback"] += 1
            return None

        action_ee_pose = frame_data["action.ee_pose"].cpu().numpy()
        current_state = frame_data["observation.state"].cpu().numpy().flatten()

        if is_bimanual:
            left_ee = action_ee_pose[:8]
            right_ee = action_ee_pose[8:16]
            current_left = current_state[:6]
            current_right = current_state[6:12]

            left_joints = compute_joints_from_ee_pose(
                ik_solver,
                current_left,
                left_ee,
                args.ee_state_unit,
                orientation_weight=1.0,
            )
            right_joints = compute_joints_from_ee_pose(
                ik_solver,
                current_right,
                right_ee,
                args.ee_state_unit,
                orientation_weight=1.0,
            )

            if left_joints is None or right_joints is None:
                ik_stats["fallback"] += 1
                return None

            action_joints = np.concatenate([left_joints, right_joints], axis=0)
        else:
            action_joints = compute_joints_from_ee_pose(
                ik_solver,
                current_state,
                action_ee_pose,
                args.ee_state_unit,
                orientation_weight=1.0,
            )
            if action_joints is None:
                ik_stats["fallback"] += 1
                return None

        action_tensor = torch.from_numpy(action_joints).float().to(device).unsqueeze(0)

        ik_stats["total"] += 1
        ik_stats["success"] += 1

        original_action = frame_data["action"].cpu().numpy()
        error = np.max(np.abs(action_joints - original_action))
        ik_stats["errors"].append(error)

        return action_tensor

    except Exception as e:
        logger.warning(f"IK computation failed: {e}", exc_info=True)
        ik_stats["total"] += 1
        ik_stats["fallback"] += 1
        return None


def replay_episode(
    env: DirectRLEnv,
    episode_data: List[Dict[str, torch.Tensor]],
    rate_limiter: Optional[RateLimiter],
    initial_pose: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    replay_dataset: Optional[LeRobotDataset] = None,
    disable_depth: bool = False,
    ik_solver: Optional[Any] = None,
    is_bimanual: bool = False,
    ik_stats: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    task_description: str = "fold the garment on the table",
    debug_markers: Optional[GarmentKeypointDebugMarkers] = None,
) -> bool:
    """Replay a single episode from recorded data."""
    try:
        env.reset()
        if debug_markers is not None:
            debug_markers.update_from_env(env)

        if initial_pose is not None:
            env.set_all_pose(initial_pose)
            logger.debug(f"Set initial pose from recorded data: {initial_pose}")
            if debug_markers is not None:
                debug_markers.update_from_env(env)
        else:
            logger.warning("No initial pose found in recorded data, using default pose")

        stabilize_garment_after_reset(env, args)
        if debug_markers is not None:
            debug_markers.update_from_env(env)

        success_achieved = False

        for idx in range(len(episode_data)):
            if rate_limiter:
                rate_limiter.sleep(env)

            if args.use_ee_pose and ik_solver is not None:
                action = compute_action_from_ee_pose(
                    env,
                    episode_data[idx],
                    ik_solver,
                    is_bimanual,
                    args,
                    ik_stats,
                    device,
                )
                if action is None:
                    action = episode_data[idx]["action"].to(device).unsqueeze(0)
            else:
                action = episode_data[idx]["action"].to(device).unsqueeze(0)

            env.step(action)
            if debug_markers is not None:
                debug_markers.update_from_env(env)

            if replay_dataset is not None:
                observations = env._get_observations()

                if disable_depth and "observation.top_depth" in observations:
                    observations = {
                        k: v
                        for k, v in observations.items()
                        if k != "observation.top_depth"
                    }
                frame = {**observations, "task": task_description}
                replay_dataset.add_frame(frame)

            success = env._get_success().item()
            if success:
                success_achieved = True

        return success_achieved
    except Exception as e:
        logger.error(f"Error during episode replay: {e}", exc_info=True)
        return False


def append_episode_initial_pose(
    json_path: Path,
    episode_idx: int,
    object_initial_pose: Dict[str, Any],
    garment_name: Optional[str] = None,
    scale: Optional[Any] = None,
) -> None:
    """Append initial pose information to output garment_info.json."""
    from lehome.utils.record import append_episode_initial_pose as append_pose

    append_pose(
        json_path,
        episode_idx,
        object_initial_pose,
        garment_name=garment_name,
        scale=scale,
    )


def replay(args: argparse.Namespace) -> None:
    """Replay HDF5 recorded datasets for visualization and verification."""
    validate_args(args)

    source_hdf5_path = Path(args.dataset_root)
    garment_info_path = find_garment_info_json(source_hdf5_path)
    if garment_info_path is not None:
        logger.info(f"Using garment info from: {garment_info_path}")
    else:
        logger.warning(
            "No external garment_info.json found for this HDF5 file. "
            "Replay will rely on /data/demo_*/meta when available."
        )

    device = getattr(args, "device", "cpu")
    task_description = getattr(args, "task_description", "fold the garment on the table")

    ik_solver: Optional[Any] = None
    is_bimanual = False
    ik_stats: Dict[str, Any] = {"total": 0, "success": 0, "fallback": 0, "errors": []}
    debug_markers: Optional[GarmentKeypointDebugMarkers] = None

    with HDF5ReplaySource(args.dataset_root) as dataset:
        if args.use_ee_pose and not dataset.has_ee_pose():
            raise ValueError(
                "HDF5 dataset does not contain ee_pose actions. "
                "Expected one of: 'data/demo_*/obs/ee_pose' or top-level "
                "'data/demo_*/actions' in ee_pose mode."
            )

        if args.use_ee_pose:
            urdf_path = Path(args.ee_urdf_path)
            if not urdf_path.exists():
                raise FileNotFoundError(f"URDF file not found: {urdf_path}")

            from lehome.utils import RobotKinematics

            state_dim = dataset.get_state_dim()
            is_bimanual = state_dim >= 12

            solver_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ]
            ik_solver = RobotKinematics(
                str(urdf_path),
                target_frame_name="gripper_frame_link",
                joint_names=solver_names,
            )
            arm_mode = "dual-arm" if is_bimanual else "single-arm"
            logger.info(f"IK solver loaded ({arm_mode} mode)")
            logger.warning(
                "Using action.ee_pose + IK control, which may differ from original action"
            )

        logger.info(f"Creating environment: {args.task}")
        env_cfg = parse_env_cfg(args.task, device=device)

        garment_name = getattr(args, "garment_name", None)
        if isinstance(garment_name, str):
            garment_name = garment_name.strip() or None
        if garment_name is not None:
            logger.info(f"Using garment name from CLI: {garment_name}")

        first_episode_meta = dataset.get_episode_meta(0)
        if garment_name is None and first_episode_meta:
            garment_name = _extract_garment_name_from_episode_meta(first_episode_meta)
            if garment_name is not None:
                logger.info("Using garment name from /data/demo_0/meta or initial_state/garment")

        if garment_name is None:
            garment_name = dataset.get_garment_name_from_env_args()
        if garment_name is None and garment_info_path is not None:
            garment_name = _try_read_garment_name_from_json(garment_info_path)

        default_garment_name = getattr(env_cfg, "garment_name", None)
        if isinstance(default_garment_name, str):
            default_garment_name = default_garment_name.strip() or None
        if garment_name is None:
            garment_name = default_garment_name

        if garment_name is not None:
            env_cfg.garment_name = garment_name
            logger.info(f"Using garment name: {garment_name}")
        else:
            raise ValueError(
                "Could not determine garment name for replay. "
                "This HDF5 file does not include garment metadata in /data/demo_*/meta, "
                "initial_state/garment, data/env_args, or garment_info.json. "
                "Pass --garment_name (for example, Top_Long_Unseen_0)."
            )

        env_cfg.garment_version = args.garment_version
        env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
        env_cfg.particle_cfg_path = args.particle_cfg_path

        env: DirectRLEnv = gym.make(args.task, cfg=env_cfg).unwrapped

        try:
            logger.info("Initializing observations...")
            env.initialize_obs()
            logger.info("Observations initialized successfully")

            if getattr(args, "debugging_markers", False):
                try:
                    debug_markers = GarmentKeypointDebugMarkers()
                    logger.info("Enabled garment semantic keypoint debugging markers.")
                except Exception as exc:
                    logger.warning(
                        f"Failed to initialize debugging markers. Continuing without them: {exc}",
                        exc_info=True,
                    )
                    debug_markers = None

            rate_limiter = RateLimiter(args.step_hz) if args.step_hz > 0 else None
            replay_dataset, json_path = create_replay_dataset(args, source_hdf5_path, dataset.fps)

            start_idx = args.start_episode
            end_idx = args.end_episode if args.end_episode is not None else dataset.num_episodes
            end_idx = min(end_idx, dataset.num_episodes)
            total_episodes = end_idx - start_idx

            logger.info(
                f"Replaying episodes {start_idx} to {end_idx - 1} "
                f"(displayed as 1 to {total_episodes})"
            )

            total_attempts = 0
            total_successes = 0
            saved_episodes = 0

            try:
                for episode_idx in range(start_idx, end_idx):
                    display_episode_num = episode_idx - start_idx + 1
                    source_episode_index = dataset.get_source_episode_index(episode_idx)

                    logger.info("")
                    logger.info(f"{'=' * 60}")
                    logger.info(f"Episode {display_episode_num}/{total_episodes}")
                    logger.info(
                        f"Demo: {dataset.demo_names[episode_idx]} "
                        f"(source episode {source_episode_index})"
                    )
                    logger.info(f"{'=' * 60}")

                    episode_meta = dataset.get_episode_meta(episode_idx)
                    initial_pose = _extract_initial_pose_from_episode_meta(
                        episode_meta, source_episode_index=source_episode_index
                    )
                    if initial_pose is None:
                        initial_pose = load_initial_pose_from_json(
                            garment_info_path, source_episode_index
                        )

                    try:
                        episode_data = dataset.get_episode_frames(
                            episode_idx, require_ee_pose=args.use_ee_pose
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to load {dataset.demo_names[episode_idx]}: {e}",
                            exc_info=True,
                        )
                        continue

                    if len(episode_data) == 0:
                        logger.warning(
                            f"Episode {display_episode_num} has no frame data, skipping..."
                        )
                        continue

                    logger.info(f"Episode length: {len(episode_data)} frames")

                    for replay_idx in range(args.num_replays):
                        total_attempts += 1

                        if replay_dataset is not None:
                            replay_dataset.clear_episode_buffer()

                        success = replay_episode(
                            env=env,
                            episode_data=episode_data,
                            rate_limiter=rate_limiter,
                            initial_pose=initial_pose,
                            args=args,
                            replay_dataset=replay_dataset,
                            disable_depth=args.disable_depth,
                            ik_solver=ik_solver,
                            is_bimanual=is_bimanual,
                            ik_stats=ik_stats,
                            device=device,
                            task_description=task_description,
                            debug_markers=debug_markers,
                        )

                        if success:
                            total_successes += 1
                            logger.info(f"  [Replay {replay_idx + 1}/{args.num_replays}] Success")
                        else:
                            logger.info(f"  [Replay {replay_idx + 1}/{args.num_replays}] Failed")

                        should_save = replay_dataset is not None and (
                            not args.save_successful_only or success
                        )

                        if should_save:
                            try:
                                replay_dataset.save_episode()
                                if json_path is not None and initial_pose is not None:
                                    append_episode_initial_pose(
                                        json_path, saved_episodes, initial_pose
                                    )
                                saved_episodes += 1
                                logger.info(f"  Saved as episode {saved_episodes - 1}")
                            except Exception as e:
                                logger.error(f"Failed to save episode: {e}", exc_info=True)
                        elif replay_dataset is not None:
                            replay_dataset.clear_episode_buffer()

            finally:
                if replay_dataset is not None:
                    try:
                        replay_dataset.clear_episode_buffer()
                        replay_dataset.finalize()
                    except Exception as e:
                        logger.error(f"Error finalizing dataset: {e}", exc_info=True)

            logger.info("")
            logger.info(f"{'=' * 60}")
            logger.info("Replay Statistics")
            logger.info(f"{'=' * 60}")
            logger.info(f"  Total attempts: {total_attempts}")
            logger.info(f"  Total successes: {total_successes}")
            if total_attempts > 0:
                logger.info(f"  Success rate: {100.0 * total_successes / total_attempts:.1f}%")
            if replay_dataset is not None:
                logger.info(f"  Saved episodes: {saved_episodes}")

            if args.use_ee_pose and ik_stats["total"] > 0:
                logger.info("")
                logger.info("IK Statistics")
                logger.info(f"  Total IK attempts: {ik_stats['total']}")
                logger.info(f"  IK successes: {ik_stats['success']}")
                logger.info(f"  IK fallbacks: {ik_stats['fallback']}")
                if ik_stats["total"] > 0:
                    logger.info(
                        f"  IK success rate: "
                        f"{100.0 * ik_stats['success'] / ik_stats['total']:.1f}%"
                    )
                if ik_stats["errors"]:
                    errors = np.array(ik_stats["errors"])
                    unit = "rad" if args.ee_state_unit == "rad" else "deg"
                    logger.info(f"  Joint angle error vs original action ({unit}):")
                    logger.info(f"    mean = {np.mean(errors):.6f}")
                    logger.info(f"    max  = {np.max(errors):.6f}")

            logger.info(f"{'=' * 60}")

        finally:
            env.close()
