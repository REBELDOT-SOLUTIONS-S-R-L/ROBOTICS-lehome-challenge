"""
Convert multiple Isaac Lab HDF5 synthetic-data files into a single merged
LeRobot **v2.1** dataset (independent of the installed lerobot version).

Inputs are configured in `GARMENT_DATASETS` near the top of this file. Each
garment-type entry binds a unique task string (one row in `tasks.jsonl`) to a
list of root directories scanned recursively for `*.hdf5` files. Every episode
in the merged dataset carries the `task_index` of its source garment, which is
how GR00T's language modality switches between fold programs at train/eval.

Source HDF5 layout (per episode, under data/demo_N):
    obs/left_joint_pos    (T, 6)   float32  ┐ concat -> `observation.state`
    obs/right_joint_pos   (T, 6)   float32  ┘   (measured joint angles)
    obs/actions           (T, 12)  float32  -> `action`
                                                (commanded target joint angles)
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
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FPS = 30
IMG_H, IMG_W = 480, 640
ACTION_DIM = 12
CHUNK_SIZE = 1000
CODEBASE_VERSION = "v2.1"
GPU_VCODEC = "h264_nvenc"
PIX_FMT = "yuv420p"

# Per-garment-type sources. Each entry maps a unique task string (one row in
# tasks.jsonl) to one or more directories scanned for *.hdf5 files. All entries
# are merged into a single LeRobot v2.1 dataset; each episode's task_index
# points back to the entry it came from. Add/remove garment keys freely.
GARMENT_DATASETS: dict[str, dict] = {
    "top_long": {
        "task": "Fold the long-sleeve top on the table",
        "roots": [
            "/media/alexluci/480eeb06-1ed9-4099-af71-85b9cc90b82b/synthetic_data_garment/Top_Long",
            "/workspace/IsaacTools/ROBOTICS-lehome-challenge/Datasets/hdf5_mimicgen_pipeline/2_generated/Top_Long",
        ],
    },
    "top_short": {
        "task": "Fold the short-sleeve top on the table",
        "roots": [
            "/media/alexluci/480eeb06-1ed9-4099-af71-85b9cc90b82b/synthetic_data_garment/Top_Short",
            "/workspace/IsaacTools/ROBOTICS-lehome-challenge/Datasets/hdf5_mimicgen_pipeline/2_generated/Top_Short",
        ],
    },
    "pant_long": {
        "task": "Fold the long pants on the table",
        "roots": [
            "/media/alexluci/480eeb06-1ed9-4099-af71-85b9cc90b82b/synthetic_data_garment/Pant_Long",
            "/workspace/IsaacTools/ROBOTICS-lehome-challenge/Datasets/hdf5_mimicgen_pipeline/2_generated/Pant_Long",
        ],
    },
    "pant_short": {
        "task": "Fold the short pants on the table",
        "roots": [
            "/media/alexluci/480eeb06-1ed9-4099-af71-85b9cc90b82b/synthetic_data_garment/Pant_Short",
            "/workspace/IsaacTools/ROBOTICS-lehome-challenge/Datasets/hdf5_mimicgen_pipeline/2_generated/Pant_Short",
        ],
    },
}

# Skip any *.hdf5 whose filename contains one of these substrings.
# MimicGen writes failed rollouts to a companion "..._failed.hdf5" file we never want.
EXCLUDE_NAME_SUBSTRINGS: tuple[str, ...] = ("_failed",)

# Pipeline depth: a background reader thread loads up to this many episode
# payloads ahead of the encoder. 2 keeps RAM bounded (~3 episodes in flight ≈
# a few GB worst case for 1k-frame demos) while still hiding I/O latency.
PREFETCH_DEPTH = 2

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
        "names": ["height", "width", "channels"],
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
        "task": {"dtype": "string", "shape": [1], "names": None},
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
# NVENC video encoder (raw RGB piped to ffmpeg, no per-frame Python overhead)
# ---------------------------------------------------------------------------


def encode_mp4_from_array(frames_thwc: np.ndarray, out_path: Path, fps: int) -> None:
    """Encode a (T, H, W, 3) uint8 RGB array to mp4 using h264_nvenc.

    Pipes raw RGB bytes through stdin to ffmpeg so the colorspace conversion
    runs once via SIMD-optimized libswscale inside ffmpeg, instead of looping
    through PyAV's `VideoFrame.from_ndarray` per frame.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frames_thwc.dtype != np.uint8:
        frames_thwc = frames_thwc.astype(np.uint8)
    if not frames_thwc.flags["C_CONTIGUOUS"]:
        frames_thwc = np.ascontiguousarray(frames_thwc)
    h, w = frames_thwc.shape[1:3]

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-framerate", str(fps),
        "-i", "-",
        "-c:v", GPU_VCODEC,        # h264_nvenc
        "-preset", "p1",           # fastest NVENC preset
        "-tune", "ull",            # ultra-low latency
        "-rc", "vbr",
        "-cq", "30",
        "-bf", "0",                # no B-frames
        "-g", "2",                 # keyframe interval
        "-pix_fmt", PIX_FMT,       # output yuv420p
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(frames_thwc.tobytes())
    finally:
        proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed encoding {out_path}: {err.strip()}")
    proc.stderr.close()


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
    task: str,
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
            "task": pa.array([task] * T, type=pa.string()),
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
    task_strings: list[str],
) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = max(1, (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_strings),
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
        for task_idx, task in enumerate(task_strings):
            f.write(json.dumps({"task_index": task_idx, "task": task}) + "\n")

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
    state = np.concatenate([left, right], axis=-1)
    if state.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}D observation.state, got {state.shape}")

    action = ep["obs/actions"][:].astype(np.float32)
    if action.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}D action, got {action.shape}")

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


def discover_sources() -> tuple[list[str], list[tuple[str, int, Path]]]:
    """Resolve `GARMENT_DATASETS` into a flat work list.

    Returns:
        task_strings: ordered list mapping task_index -> task description.
        sources: list of (garment_key, task_index, hdf5_path), preserving the
            order of `GARMENT_DATASETS` so output episode_index is deterministic.
    """
    task_strings: list[str] = []
    for entry in GARMENT_DATASETS.values():
        task = entry["task"]
        if task not in task_strings:
            task_strings.append(task)

    sources: list[tuple[str, int, Path]] = []
    for garment_key, entry in GARMENT_DATASETS.items():
        task_idx = task_strings.index(entry["task"])
        for root in entry["roots"]:
            root_path = Path(root)
            if not root_path.is_dir():
                print(f"[WARN] [{garment_key}] root not found: {root_path}")
                continue
            matched = sorted(p for p in root_path.glob("*.hdf5"))
            for hdf5_path in matched:
                if any(s in hdf5_path.name for s in EXCLUDE_NAME_SUBSTRINGS):
                    continue
                sources.append((garment_key, task_idx, hdf5_path))
    return task_strings, sources


def convert(
    output_root: Path,
    modality_json: Path | None,
    skip_failed: bool,
    max_episodes: int | None = None,
) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output {output_root} exists and is non-empty — remove it first.")
    output_root.mkdir(parents=True, exist_ok=True)

    features = build_features()
    task_strings, sources = discover_sources()
    if not sources:
        raise RuntimeError("No HDF5 files discovered from GARMENT_DATASETS roots.")

    print(f"Discovered {len(sources)} HDF5 file(s) across {len(task_strings)} task(s):")
    for task_idx, task in enumerate(task_strings):
        n = sum(1 for _, t, _ in sources if t == task_idx)
        print(f"  [task_index={task_idx}] {n:3d} files — {task}")
    if max_episodes is not None:
        print(f"Limiting conversion to first {max_episodes} episode(s).")

    ep_idx = 0
    global_start = 0
    episode_entries: list[dict] = []
    episode_stats: list[dict] = []

    # Producer/consumer pipeline:
    #   reader thread ──► queue ──► main thread (parquet + parallel-camera mp4 + stats)
    # The reader hides HDF5 read+gzip-decompress latency behind encoding, and the
    # camera ThreadPoolExecutor runs the 3 NVENC encodes for an episode in parallel.
    queue: Queue = Queue(maxsize=PREFETCH_DEPTH)
    SENTINEL = None

    def reader() -> None:
        queued_episodes = 0
        try:
            for garment_key, task_idx, hdf5_path in sources:
                if max_episodes is not None and queued_episodes >= max_episodes:
                    return
                print(f"\n=== [{garment_key}] {hdf5_path.name} ===")
                try:
                    fh = h5py.File(hdf5_path, "r")
                except Exception as e:
                    print(f"[WARN] Failed to open {hdf5_path.name}: {e}")
                    continue
                try:
                    if "data" not in fh:
                        print(f"[WARN] No /data group in {hdf5_path.name}, skipping.")
                        continue
                    for demo_name in sorted_demo_names(fh["data"]):
                        ep = fh["data"][demo_name]
                        if skip_failed and not bool(ep.attrs.get("success", True)):
                            continue
                        try:
                            action, state, images, T = load_episode(ep)
                        except (KeyError, ValueError) as e:
                            print(f"[WARN] Skipping {hdf5_path.name}/{demo_name}: {e}")
                            continue
                        queue.put((task_idx, hdf5_path.name, demo_name,
                                   action, state, images, T))
                        queued_episodes += 1
                        if max_episodes is not None and queued_episodes >= max_episodes:
                            return
                finally:
                    fh.close()
        finally:
            queue.put(SENTINEL)

    reader_thread = Thread(target=reader, name="hdf5-reader", daemon=True)
    reader_thread.start()

    # Rough demo-count estimate for the progress bar (most MimicGen files emit ~64-100 demos).
    total_demos_estimate = len(sources) * 100
    if max_episodes is not None:
        total_demos_estimate = min(total_demos_estimate, max_episodes)

    with ThreadPoolExecutor(max_workers=len(CAM_KEYS), thread_name_prefix="nvenc") as cam_pool, \
         tqdm(total=total_demos_estimate, desc="Encoding episodes", unit="ep") as pbar:
        while True:
            item = queue.get()
            if item is SENTINEL:
                break
            task_idx, hdf5_name, demo_name, action, state, images, T = item
            task_string = task_strings[task_idx]

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
                task_idx=task_idx,
                task=task_string,
                out_path=parquet_path,
            )

            cam_futures = []
            for feat_key, frames in images.items():
                video_path = output_root / VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=chunk_index,
                    video_key=feat_key,
                    episode_index=ep_idx,
                )
                cam_futures.append(cam_pool.submit(encode_mp4_from_array, frames, video_path, FPS))
            for fut in cam_futures:
                fut.result()

            timestamps = (np.arange(T, dtype=np.float32) / FPS)
            frame_indices = np.arange(T, dtype=np.int64)
            episode_indices = np.full(T, ep_idx, dtype=np.int64)
            global_indices = np.arange(global_start, global_start + T, dtype=np.int64)
            task_indices = np.full(T, task_idx, dtype=np.int64)

            ep_stats_dict = compute_episode_stats(
                {
                    "action": action,
                    "observation.state": state,
                    **images,
                    "timestamp": timestamps,
                    "frame_index": frame_indices,
                    "episode_index": episode_indices,
                    "index": global_indices,
                    "task_index": task_indices,
                },
                features,
            )
            episode_stats.append(ep_stats_dict)
            episode_entries.append(
                {"episode_index": ep_idx, "tasks": [task_string], "length": T}
            )

            ep_idx += 1
            global_start += T
            pbar.update(1)

    reader_thread.join()

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
        task_strings=task_strings,
    )

    if modality_json is not None and modality_json.exists():
        shutil.copy(modality_json, output_root / "meta" / "modality.json")
        print(f"[OK] Copied modality.json -> {output_root / 'meta' / 'modality.json'}")

    print(f"\n[OK] LeRobot v2.1 dataset written to: {output_root}")
    print(f"     Episodes: {ep_idx}, Frames: {global_start}")
    print(f"     Duration: {global_start / FPS / 60:.1f} min @ {FPS} fps")
    print(f"     Finished at: {datetime.now().isoformat(timespec='seconds')}")


def upload_dataset_to_databricks(
    output_root: Path,
    volume_path: str | None,
    upload_script: Path | None = None,
) -> None:
    """Upload the completed LeRobot dataset by invoking databricks_upload_dataset.py."""
    script = upload_script or Path(__file__).with_name("databricks_upload_dataset.py")
    if not script.is_file():
        raise FileNotFoundError(f"Databricks upload script not found: {script}")

    cmd = [sys.executable, str(script), str(output_root)]
    if volume_path:
        cmd.append(volume_path)

    destination = volume_path or "default Databricks volume path"
    print(f"\nUploading dataset to Databricks ({destination})...")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Merge Isaac Lab HDF5s into a LeRobot v2.1 dataset. "
                    "Source files and per-garment task strings are configured "
                    "via GARMENT_DATASETS at the top of this file.",
    )
    parser.add_argument(
        "--output_root",
        default="/workspace/IsaacTools/ROBOTICS-lehome-challenge/Datasets/lerobot_v21",
    )
    parser.add_argument(
        "--modality_json",
        default="/workspace/IsaacTools/ROBOTICS-lehome-challenge/configs/gr00t/modality.json",
    )
    parser.add_argument("--skip_failed", action="store_true", help="Skip demos where attrs['success'] is False")
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Convert at most this many valid episodes, useful for smoke tests (e.g. 1).",
    )
    upload_group = parser.add_mutually_exclusive_group()
    upload_group.add_argument(
        "--upload_to_databricks",
        dest="upload_to_databricks",
        action="store_true",
        default=True,
        help="Upload the converted dataset to Databricks after conversion (default).",
    )
    upload_group.add_argument(
        "--no_databricks_upload",
        dest="upload_to_databricks",
        action="store_false",
        help="Skip the post-conversion Databricks upload.",
    )
    parser.add_argument(
        "--databricks_volume_path",
        default=None,
        help=(
            "Destination Databricks volume path. If omitted, "
            "scripts/utils/databricks_upload_dataset.py uses its default."
        ),
    )
    parser.add_argument(
        "--databricks_upload_script",
        default=None,
        help="Path to the Databricks upload helper script.",
    )
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max_episodes must be >= 1")

    output_root = Path(args.output_root)
    convert(
        output_root=output_root,
        modality_json=Path(args.modality_json) if args.modality_json else None,
        skip_failed=args.skip_failed,
        max_episodes=args.max_episodes,
    )
    if args.upload_to_databricks:
        upload_dataset_to_databricks(
            output_root=output_root,
            volume_path=args.databricks_volume_path,
            upload_script=Path(args.databricks_upload_script) if args.databricks_upload_script else None,
        )


if __name__ == "__main__":
    main()
