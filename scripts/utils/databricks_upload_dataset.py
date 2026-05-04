import argparse
import time
from pathlib import Path

from databricks.sdk import WorkspaceClient

DEFAULT_VOLUME_PATH = "/Volumes/workspace/default/finetune_lerobot_datasets/"


class ProgressReader:
    def __init__(self, file_obj, total_size: int, label: str = ""):
        self.file_obj = file_obj
        self.total_size = total_size
        self.bytes_read = 0
        self.last_percent_printed = -1
        self.start_time = time.time()
        self.label = label

    def read(self, size: int = -1) -> bytes:
        data = self.file_obj.read(size)

        if data:
            self.bytes_read += len(data)
            percent = int(self.bytes_read * 100 / self.total_size) if self.total_size else 100

            if percent != self.last_percent_printed:
                elapsed = max(time.time() - self.start_time, 0.001)
                speed_mb_s = (self.bytes_read / 1024 / 1024) / elapsed
                remaining = self.total_size - self.bytes_read
                eta_s = remaining / (self.bytes_read / elapsed) if self.bytes_read > 0 else 0

                prefix = f"[{self.label}] " if self.label else ""
                print(
                    f"{prefix}Upload progress: {percent}% "
                    f"({self.bytes_read / 1024 / 1024:.1f} MB / {self.total_size / 1024 / 1024:.1f} MB) | "
                    f"{speed_mb_s:.1f} MB/s | ETA {eta_s:.0f}s"
                )
                self.last_percent_printed = percent

        elif self.last_percent_printed != 100:
            elapsed = max(time.time() - self.start_time, 0.001)
            speed_mb_s = (self.bytes_read / 1024 / 1024) / elapsed
            prefix = f"[{self.label}] " if self.label else ""
            print(
                f"{prefix}Upload progress: 100% "
                f"({self.bytes_read / 1024 / 1024:.1f} MB / {self.total_size / 1024 / 1024:.1f} MB) | "
                f"{speed_mb_s:.1f} MB/s"
            )
            self.last_percent_printed = 100

        return data

    def seek(self, offset, whence=0):
        return self.file_obj.seek(offset, whence)

    def tell(self):
        return self.file_obj.tell()

    def seekable(self):
        return self.file_obj.seekable()

    def readable(self):
        return self.file_obj.readable()

    def close(self):
        return self.file_obj.close()

    @property
    def closed(self):
        return self.file_obj.closed

    def __getattr__(self, name):
        return getattr(self.file_obj, name)


def _join_volume_path(volume_root: str, relative: Path) -> str:
    root = volume_root.rstrip("/")
    rel = relative.as_posix().lstrip("/")
    return f"{root}/{rel}"


def _upload_file(client: WorkspaceClient, local_file: Path, remote_path: str, label: str) -> None:
    total_size = local_file.stat().st_size
    with local_file.open("rb") as f:
        wrapped = ProgressReader(f, total_size, label=label)
        client.files.upload(remote_path, wrapped, overwrite=True)


def main():
    parser = argparse.ArgumentParser(
        description="Upload a file or folder (e.g. a LeRobot-format dataset) to Databricks volumes."
    )
    parser.add_argument("source_path", help="Local file or folder path to upload.")
    parser.add_argument(
        "volume_path",
        nargs="?",
        default=DEFAULT_VOLUME_PATH,
        help=(
            "Destination Databricks volume path. For folder uploads, this is "
            "the destination directory under which the folder contents are placed."
        ),
    )
    args = parser.parse_args()

    source_path = Path(args.source_path)
    volume_path = args.volume_path

    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    client = WorkspaceClient()

    if source_path.is_file():
        total_size = source_path.stat().st_size
        print(f"Preparing upload: {source_path}")
        print(f"File size: {total_size / 1024 / 1024 / 1024:.2f} GB")
        print(f"Destination: {volume_path}")
        print("Starting upload...")
        _upload_file(client, source_path, volume_path, label=source_path.name)
        print("Upload finished successfully.")
        return

    # Folder upload: walk recursively, preserve relative structure under
    # {volume_path}/{source_path.name}/...
    files = sorted(p for p in source_path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"No files found under folder: {source_path}")

    total_bytes = sum(p.stat().st_size for p in files)
    remote_root = _join_volume_path(volume_path, Path(source_path.name))

    print(f"Preparing folder upload: {source_path}")
    print(f"Files: {len(files)} | Total size: {total_bytes / 1024 / 1024 / 1024:.2f} GB")
    print(f"Destination root: {remote_root}")

    uploaded_bytes = 0
    overall_start = time.time()
    for idx, local_file in enumerate(files, start=1):
        relative = local_file.relative_to(source_path)
        remote_path = _join_volume_path(remote_root, relative)
        file_size = local_file.stat().st_size

        print(
            f"[{idx}/{len(files)}] {relative.as_posix()} "
            f"({file_size / 1024 / 1024:.1f} MB) -> {remote_path}"
        )
        _upload_file(client, local_file, remote_path, label=relative.as_posix())

        uploaded_bytes += file_size
        overall_elapsed = max(time.time() - overall_start, 0.001)
        overall_speed = (uploaded_bytes / 1024 / 1024) / overall_elapsed
        overall_percent = int(uploaded_bytes * 100 / total_bytes) if total_bytes else 100
        print(
            f"Overall: {overall_percent}% "
            f"({uploaded_bytes / 1024 / 1024 / 1024:.2f} / {total_bytes / 1024 / 1024 / 1024:.2f} GB) | "
            f"{overall_speed:.1f} MB/s"
        )

    print("Folder upload finished successfully.")


if __name__ == "__main__":
    main()
