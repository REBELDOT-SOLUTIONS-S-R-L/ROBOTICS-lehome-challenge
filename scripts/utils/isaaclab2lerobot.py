"""
Convert multiple Isaac Lab HDF5 synthetic-data files into a single merged
LeRobot **v2.1** dataset (independent of the installed lerobot version).

Source HDF5 layout (per episode, under data/demo_N):
    obs/left_joint_pos    (T, 6)   float32  ┐ concat -> `action`
    obs/right_joint_pos   (T, 6)   float32  ┘
    obs/actions           (T, 12)  float32  -> `observation.state`
    obs/left_wrist        (T, 480, 640, 3) uint8 -> observation.images.left_wrist
    obs/right_wrist       (T, 480, 640, 3) uint8 -> observation.images.right_wrist
    obs/top               (T, 480, 640, 3) uint8 -> observation.images.top

Output layout (v2.1):
    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/episodes_stats.jsonl
    meta/stats.json
    meta/modality.json                      (copied from configs/gr00t)
    data/chunk-000/episode_{N:06d}.parquet
    videos/chunk-000/<video_key>/episode_{N:06d}.mp4

Videos are encoded directly from numpy arrays with h264_nvenc (GPU).
"""

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASK_STRING = "Fold the Long Sleeve Top on the table"
FPS = 30
IMG_H, IMG_W = 480, 640
ACTION_DIM = 12
CHUNK_SIZE = 1000
CODEBASE_VERSION = "v2.1"
GPU_VCODEC = "h264_nvenc"
PIX_FMT = "yuv420p"

DEFAULT_INPUT_FILES = [
    "Top_Long_Seen_0-generated-HALTON_64-run_2.hdf5",
    "Top_Long_Seen_1-generated-HALTON_64.hdf5",
    "Top_Long_Seen_2-generated-HALTON_64-run_2.hdf5",
    "Top_Long_Seen_5-generated-HALTON_64.hdf5",
    "Top_Long_Seen_7-generated-HALTON_64-run_2.hdf5",
    "Top_Long_Seen_9-generated-HALTON_64-run_2.hdf5",
]

CAM_KEYS = {
    "observation.images.left_wrist": "obs/left_wrist",
    "observation.images.right_wrist": "obs/right_wrist",
    "observation.images.top": "obs/top",
}

JOINT_NAMES = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
]

DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def build_features() -> dict:
    video_info = {
        "video.fps": float(FPS),
        "video.height": IMG_H,
        "video.width": IMG_W,
        "video.channels": 3,
        "video.codec": "h264",
        "video.pix_fmt": PIX_FMT,
        "video.is_depth_map": False,
        "has_audio": False,
    }
    cam_feature = {
        "dtype": "video",
        "shape": [IMG_H, IMG_W, 3],
        "names": ["height", "width", "channel"],
        "info": video_info,
    }
    state_feature = {"dtype": "float32", "shape": [ACTION_DIM], "names": JOINT_NAMES}
    return {
        "action": state_feature,
        "observation.state": state_feature,
        "observation.images.left_wrist": cam_feature,
        "observation.images.right_wrist": cam_feature,
        "observation.images.top": cam_feature,
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


# ---------------------------------------------------------------------------
# Stats (vendored from lerobot 0.3.3 compute_stats.py)
# ---------------------------------------------------------------------------


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_batch(imgs_nchw: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    _, _, height, width = imgs_nchw.shape
    if max(width, height) < max_size_threshold:
        return imgs_nchw
    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return imgs_nchw[:, :, ::downsample_factor, ::downsample_factor]


def sample_images_from_thwc(imgs_thwc: np.ndarray) -> np.ndarray:
    indices = sample_indices(imgs_thwc.shape[0])
    sampled = imgs_thwc[indices]  # (N, H, W, 3)
    sampled = np.transpose(sampled, (0, 3, 1, 2))  # (N, 3, H, W)
    return auto_downsample_batch(sampled)


def get_feature_stats(array: np.ndarray, axis, keepdims: bool) -> dict:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def compute_episode_stats(episode_data: dict, features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        ftype = features[key]["dtype"]
        if ftype in ("image", "video"):
            ep_ft_array = sample_images_from_thwc(data)  # (N, 3, H, W) uint8
            stats = get_feature_stats(ep_ft_array, axis=(0, 2, 3), keepdims=True)
            # normalize and squeeze batch dim for images: (1,3,1,1) -> (3,1,1)
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v.astype(np.float64) / 255.0, axis=0)
                for k, v in stats.items()
            }
        else:
            arr = np.asarray(data)
            keepdims = arr.ndim == 1
            ep_stats[key] = get_feature_stats(arr, axis=0, keepdims=keepdims)
    return ep_stats


def aggregate_feature_stats(stats_ft_list: list[dict]) -> dict:
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(stats_list: list[dict]) -> dict:
    data_keys = {key for stats in stats_list for key in stats}
    return {key: aggregate_feature_stats([s[key] for s in stats_list if key in s]) for key in data_keys}


# ---------------------------------------------------------------------------
# NVENC video encoder (direct from numpy, no PNG intermediates)
# ---------------------------------------------------------------------------


def encode_mp4_from_array(frames_thwc: np.ndarray, out_path: Path, fps: int) -> None:
    """Encode a (T, H, W, 3) uint8 RGB array to mp4 using h264_nvenc."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_thwc.shape[1:3]
    options = {
        "preset": "p1",      # NVENC fastest preset
        "tune": "ull",       # ultra-low latency
        "rc": "vbr",
        "cq": "30",
        "bf": "0",           # no B-frames
        "g": "2",            # keyframe interval
    }
    with av.open(str(out_path), mode="w") as container:
        stream = container.add_stream(GPU_VCODEC, rate=fps, options=options)
        stream.width = w
        stream.height = h
        stream.pix_fmt = PIX_FMT
        for frame in frames_thwc:
            vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(vf):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------


def write_episode_parquet(
    action: np.ndarray,
    state: np.ndarray,
    ep_idx: int,
    global_start: int,
    fps: int,
    task_idx: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = len(action)
    table = pa.table(
        {
            "action": pa.array([action[t].tolist() for t in range(T)], type=pa.list_(pa.float32())),
            "observation.state": pa.array(
                [state[t].tolist() for t in range(T)], type=pa.list_(pa.float32())
            ),
            "timestamp": pa.array([t / fps for t in range(T)], type=pa.float32()),
            "frame_index": pa.array(list(range(T)), type=pa.int64()),
            "episode_index": pa.array([ep_idx] * T, type=pa.int64()),
            "index": pa.array(list(range(global_start, global_start + T)), type=pa.int64()),
            "task_index": pa.array([task_idx] * T, type=pa.int64()),
        }
    )
    pq.write_table(table, out_path)


# ---------------------------------------------------------------------------
# Meta writers
# ---------------------------------------------------------------------------


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    out = {}
    for key, value in d.items():
        parts = key.split(sep)
        cur = out
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out


def serialize_stats(stats: dict) -> dict:
    serialized = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, np.generic):
            serialized[key] = value.item()
        else:
            serialized[key] = value
    return unflatten_dict(serialized)


def write_meta(
    output_root: Path,
    features: dict,
    total_episodes: int,
    total_frames: int,
    total_videos: int,
    episode_entries: list[dict],
    episode_stats: list[dict],
    aggregated_stats: dict,
    robot_type: str | None,
) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = max(1, (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
        "features": features,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_STRING}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episode_entries:
            f.write(json.dumps(ep) + "\n")

    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for idx, stats in enumerate(episode_stats):
            serialized = serialize_stats(stats)
            f.write(json.dumps({"episode_index": idx, "stats": serialized}) + "\n")

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(serialize_stats(aggregated_stats), f, indent=4)


# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------


def sorted_demo_names(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda n: int(n.split("_")[-1]))


def load_episode(ep: h5py.Group):
    left = ep["obs/left_joint_pos"][:].astype(np.float32)
    right = ep["obs/right_joint_pos"][:].astype(np.float32)
    action = np.concatenate([left, right], axis=-1)
    if action.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}D action, got {action.shape}")

    state = ep["obs/actions"][:].astype(np.float32)
    if state.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}D obs/actions, got {state.shape}")

    images = {}
    for feat_key, h5_key in CAM_KEYS.items():
        arr = ep[h5_key][:]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.shape[1:3] != (IMG_H, IMG_W):
            raise ValueError(f"{h5_key} expected (T, {IMG_H}, {IMG_W}, 3), got {arr.shape}")
        images[feat_key] = arr

    T = len(action)
    if not all(len(images[k]) == T for k in images) or len(state) != T:
        raise ValueError("Length mismatch between action/state/images")

    return action, state, images, T


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(
    input_dir: Path,
    filenames: list[str],
    output_root: Path,
    modality_json: Path | None,
    skip_failed: bool,
) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output {output_root} exists and is non-empty — remove it first.")
    output_root.mkdir(parents=True, exist_ok=True)

    features = build_features()

    ep_idx = 0
    global_start = 0
    episode_entries: list[dict] = []
    episode_stats: list[dict] = []

    for filename in filenames:
        hdf5_path = input_dir / filename
        if not hdf5_path.exists():
            print(f"[WARN] {hdf5_path} not found, skipping.")
            continue

        print(f"\n=== {filename} ===")
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                print(f"[WARN] No /data group in {filename}, skipping.")
                continue
            demos = sorted_demo_names(f["data"])
            for demo_name in tqdm(demos, desc=filename):
                ep = f["data"][demo_name]

                if skip_failed and not bool(ep.attrs.get("success", True)):
                    continue

                try:
                    action, state, images, T = load_episode(ep)
                except (KeyError, ValueError) as e:
                    print(f"[WARN] Skipping {filename}/{demo_name}: {e}")
                    continue

                chunk_index = ep_idx // CHUNK_SIZE
                parquet_path = output_root / DATA_PATH_TEMPLATE.format(
                    episode_chunk=chunk_index, episode_index=ep_idx
                )
                write_episode_parquet(
                    action=action,
                    state=state,
                    ep_idx=ep_idx,
                    global_start=global_start,
                    fps=FPS,
                    task_idx=0,
                    out_path=parquet_path,
                )

                for feat_key, frames in images.items():
                    video_path = output_root / VIDEO_PATH_TEMPLATE.format(
                        episode_chunk=chunk_index,
                        video_key=feat_key,
                        episode_index=ep_idx,
                    )
                    encode_mp4_from_array(frames, video_path, FPS)

                ep_stats_dict = compute_episode_stats(
                    {
                        "action": action,
                        "observation.state": state,
                        **images,
                    },
                    features,
                )
                episode_stats.append(ep_stats_dict)
                episode_entries.append(
                    {"episode_index": ep_idx, "tasks": [TASK_STRING], "length": T}
                )

                ep_idx += 1
                global_start += T

    if ep_idx == 0:
        raise RuntimeError("No episodes written — aborting before meta write.")

    print("\nAggregating dataset stats...")
    aggregated = aggregate_stats(episode_stats)

    write_meta(
        output_root=output_root,
        features=features,
        total_episodes=ep_idx,
        total_frames=global_start,
        total_videos=ep_idx * len(CAM_KEYS),
        episode_entries=episode_entries,
        episode_stats=episode_stats,
        aggregated_stats=aggregated,
        robot_type="so101_bimanual",
    )

    if modality_json is not None and modality_json.exists():
        shutil.copy(modality_json, output_root / "meta" / "modality.json")
        print(f"[OK] Copied modality.json -> {output_root / 'meta' / 'modality.json'}")

    print(f"\n[OK] LeRobot v2.1 dataset written to: {output_root}")
    print(f"     Episodes: {ep_idx}, Frames: {global_start}")
    print(f"     Duration: {global_start / FPS / 60:.1f} min @ {FPS} fps")
    print(f"     Finished at: {datetime.now().isoformat(timespec='seconds')}")


def main():
    parser = argparse.ArgumentParser(description="Merge Isaac Lab HDF5s into a LeRobot v2.1 dataset.")
    parser.add_argument(
        "--input_dir",
        default="/media/alexluci/480eeb06-1ed9-4099-af71-85b9cc90b82b/synthetic_data_garment",
    )
    parser.add_argument(
        "--output_root",
        default="/workspace/IsaacTools/ROBOTICS-lehome-challenge/Datasets/lerobot_v21",
    )
    parser.add_argument(
        "--modality_json",
        default="/workspace/IsaacTools/ROBOTICS-lehome-challenge/configs/gr00t/modality.json",
    )
    parser.add_argument("--files", nargs="+", default=DEFAULT_INPUT_FILES)
    parser.add_argument("--skip_failed", action="store_true", help="Skip demos where attrs['success'] is False")
    args = parser.parse_args()

    logging.getLogger("libav").setLevel(logging.ERROR)

    convert(
        input_dir=Path(args.input_dir),
        filenames=list(args.files),
        output_root=Path(args.output_root),
        modality_json=Path(args.modality_json) if args.modality_json else None,
        skip_failed=args.skip_failed,
    )


if __name__ == "__main__":
    main()
