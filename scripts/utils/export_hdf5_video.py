#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _try_reexec_with_repo_python() -> None:
    if os.environ.get("_HDF5_VIDEO_REEXEC") == "1":
        return

    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        candidate = parent / ".venv" / "bin" / "python3"
        if candidate.exists():
            env = os.environ.copy()
            env["_HDF5_VIDEO_REEXEC"] = "1"
            os.execve(
                str(candidate),
                [str(candidate), str(script_path), *sys.argv[1:]],
                env,
            )


try:
    import h5py
    import numpy as np
except ImportError:
    _try_reexec_with_repo_python()
    raise SystemExit(
        "Missing Python dependencies. Install `h5py` and `numpy`, or run this "
        "script with the repo virtualenv Python."
    )


VIDEO_SUFFIXES = (".h5", ".hdf5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export playable MP4 videos from camera streams in an HDF5 demo."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset path or unique HDF5 filename under the current directory.",
    )
    parser.add_argument(
        "--demo",
        required=True,
        help="Demo index or name, for example `0` or `demo_0`.",
    )
    parser.add_argument(
        "--stream",
        help=(
            "Optional video stream to export. Accepts a basename like `top` or a "
            "path relative to the demo group like `obs/top`. Defaults to all streams."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for exported MP4 files. Defaults to the dataset file directory.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Output FPS. Defaults to the dataset `data/fps` attribute, else 30.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Frames to stream to ffmpeg at once. Default: 32.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MP4 outputs.",
    )
    parser.add_argument(
        "--list-streams",
        action="store_true",
        help="List the detected video streams for the selected demo and exit.",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_arg: str) -> Path:
    candidate = Path(dataset_arg).expanduser()
    if candidate.exists():
        return candidate.resolve()

    search_root = Path.cwd()
    names_to_match = {dataset_arg}
    if not Path(dataset_arg).suffix:
        names_to_match.update(f"{dataset_arg}{suffix}" for suffix in VIDEO_SUFFIXES)

    matches: list[Path] = []
    for suffix in VIDEO_SUFFIXES:
        matches.extend(
            path.resolve()
            for path in search_root.rglob(f"*{suffix}")
            if path.name in names_to_match
        )
    matches = sorted(set(matches))
    if not matches:
        raise SystemExit(f"Dataset not found: {dataset_arg}")
    if len(matches) > 1:
        joined = "\n".join(f"  - {path}" for path in matches)
        raise SystemExit(
            f"Dataset name `{dataset_arg}` is ambiguous. Use a path instead.\n{joined}"
        )
    return matches[0]


def normalize_demo_name(demo_arg: str) -> str:
    demo_arg = demo_arg.strip()
    if demo_arg.startswith("/"):
        demo_arg = demo_arg.rstrip("/").split("/")[-1]
    if demo_arg.startswith("demo_"):
        return demo_arg
    try:
        index = int(demo_arg)
    except ValueError as exc:
        raise SystemExit(
            f"Invalid demo `{demo_arg}`. Use an integer like `0` or a name like `demo_0`."
        ) from exc
    return f"demo_{index}"


def list_demo_names(data_group: h5py.Group) -> list[str]:
    return sorted(name for name in data_group.keys() if name.startswith("demo_"))


def is_video_dataset(dataset: h5py.Dataset) -> bool:
    if dataset.dtype != np.uint8:
        return False

    shape = dataset.shape
    if len(shape) == 4:
        _, height, width, channels = shape
        return height > 1 and width > 1 and channels in (1, 3, 4)
    if len(shape) == 3:
        _, height, width = shape
        return height > 1 and width > 1
    return False


def discover_video_streams(demo_group: h5py.Group) -> list[str]:
    streams: list[str] = []

    def visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and is_video_dataset(obj):
            streams.append(name)

    demo_group.visititems(visitor)
    return sorted(streams)


def select_streams(streams: list[str], requested_stream: str | None) -> list[str]:
    if requested_stream is None:
        return streams

    needle = requested_stream.strip("/")
    matches = [
        stream
        for stream in streams
        if stream == needle
        or stream.endswith(f"/{needle}")
        or Path(stream).name == needle
    ]
    if not matches:
        available = "\n".join(f"  - {stream}" for stream in streams)
        raise SystemExit(
            f"Stream `{requested_stream}` was not found.\nAvailable streams:\n{available}"
        )
    if len(matches) > 1:
        available = "\n".join(f"  - {stream}" for stream in matches)
        raise SystemExit(
            f"Stream `{requested_stream}` is ambiguous. Use a more specific path.\n{available}"
        )
    return matches


def infer_fps(data_group: h5py.Group, requested_fps: float | None) -> float:
    if requested_fps is not None:
        if requested_fps <= 0:
            raise SystemExit("--fps must be positive.")
        return requested_fps

    dataset_fps = data_group.attrs.get("fps")
    if dataset_fps is None:
        return 30.0
    return float(dataset_fps)


def resolve_output_path(
    dataset_path: Path,
    demo_name: str,
    stream_name: str,
    output_dir: Path | None,
) -> Path:
    target_dir = dataset_path.parent if output_dir is None else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_stream_name = stream_name.replace("/", "_")
    filename = f"{dataset_path.stem}.{demo_name}.{safe_stream_name}.mp4"
    return target_dir / filename


def ffmpeg_input_spec(dataset: h5py.Dataset) -> tuple[str, int, int]:
    if dataset.ndim == 4:
        _, height, width, channels = dataset.shape
        pix_fmt = {1: "gray", 3: "rgb24", 4: "rgba"}[channels]
        return pix_fmt, int(width), int(height)

    _, height, width = dataset.shape
    return "gray", int(width), int(height)


def iter_frame_batches(dataset: h5py.Dataset, batch_size: int) -> Iterable[np.ndarray]:
    num_frames = dataset.shape[0]
    for start in range(0, num_frames, batch_size):
        yield dataset[start : start + batch_size]


def encode_stream(
    dataset: h5py.Dataset,
    output_path: Path,
    fps: float,
    batch_size: int,
    overwrite: bool,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("`ffmpeg` is required but was not found in PATH.")

    pix_fmt, width, height = ffmpeg_input_spec(dataset)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        assert process.stdin is not None
        for batch in iter_frame_batches(dataset, batch_size):
            process.stdin.write(batch.tobytes(order="C"))
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except Exception as exc:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read().decode("utf-8", errors="replace")
        process.kill()
        process.wait()
        raise SystemExit(
            f"ffmpeg failed while writing {output_path}:\n{stderr.strip() or str(exc)}"
        ) from exc

    if return_code != 0:
        raise SystemExit(
            f"ffmpeg failed while writing {output_path}:\n{stderr.strip() or 'no error output'}"
        )


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    dataset_path = resolve_dataset_path(args.dataset)
    demo_name = normalize_demo_name(args.demo)

    with h5py.File(dataset_path, "r") as hdf:
        if "data" not in hdf:
            raise SystemExit(f"`{dataset_path}` does not contain a `/data` group.")

        data_group = hdf["data"]
        available_demos = list_demo_names(data_group)
        if demo_name not in data_group:
            joined = "\n".join(f"  - {name}" for name in available_demos)
            raise SystemExit(
                f"Demo `{demo_name}` was not found in `{dataset_path}`.\nAvailable demos:\n{joined}"
            )

        demo_group = data_group[demo_name]
        video_streams = discover_video_streams(demo_group)
        if not video_streams:
            raise SystemExit(
                f"No video-like datasets were found under `/data/{demo_name}` in `{dataset_path}`."
            )

        if args.list_streams:
            for stream in video_streams:
                print(stream)
            return 0

        selected_streams = select_streams(video_streams, args.stream)
        fps = infer_fps(data_group, args.fps)

        for stream_name in selected_streams:
            stream_dataset = demo_group[stream_name]
            output_path = resolve_output_path(
                dataset_path=dataset_path,
                demo_name=demo_name,
                stream_name=stream_name,
                output_dir=args.output_dir,
            )
            print(
                f"Encoding `{dataset_path.name}` {demo_name} {stream_name} "
                f"({stream_dataset.shape[0]} frames @ {fps:g} fps) -> {output_path}"
            )
            encode_stream(
                dataset=stream_dataset,
                output_path=output_path,
                fps=fps,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
            print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
