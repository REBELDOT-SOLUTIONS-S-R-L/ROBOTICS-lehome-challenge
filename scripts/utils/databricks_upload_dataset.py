import os
import time
from databricks.sdk import WorkspaceClient

FILE_PATH = "Datasets/hdf5_mimicgen_pipeline/1_annotated_teleop/Top_Long_Unseen_0-annotated-1.hdf5"
VOLUME_PATH = "/Volumes/workspace/default/mimicgen_annotated_hdf5_datasets/Top_Long_Unseen_0-annotated-1.hdf5"


class ProgressReader:
    def __init__(self, file_obj, total_size: int):
        self.file_obj = file_obj
        self.total_size = total_size
        self.bytes_read = 0
        self.last_percent_printed = -1
        self.start_time = time.time()

    def read(self, size: int = -1) -> bytes:
        data = self.file_obj.read(size)

        if data:
            self.bytes_read += len(data)
            percent = int(self.bytes_read * 100 / self.total_size)

            if percent != self.last_percent_printed:
                elapsed = max(time.time() - self.start_time, 0.001)
                speed_mb_s = (self.bytes_read / 1024 / 1024) / elapsed
                remaining = self.total_size - self.bytes_read
                eta_s = remaining / (self.bytes_read / elapsed) if self.bytes_read > 0 else 0

                print(
                    f"Upload progress: {percent}% "
                    f"({self.bytes_read / 1024 / 1024:.1f} MB / {self.total_size / 1024 / 1024:.1f} MB) | "
                    f"{speed_mb_s:.1f} MB/s | ETA {eta_s:.0f}s"
                )
                self.last_percent_printed = percent

        elif self.last_percent_printed != 100:
            elapsed = max(time.time() - self.start_time, 0.001)
            speed_mb_s = (self.bytes_read / 1024 / 1024) / elapsed
            print(
                f"Upload progress: 100% "
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


def main():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"File not found: {FILE_PATH}")

    total_size = os.path.getsize(FILE_PATH)
    print(f"Preparing upload: {FILE_PATH}")
    print(f"File size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"Destination: {VOLUME_PATH}")

    client = WorkspaceClient()

    with open(FILE_PATH, "rb") as f:
        wrapped = ProgressReader(f, total_size)
        print("Starting upload...")
        client.files.upload(VOLUME_PATH, wrapped, overwrite=True)

    print("Upload finished successfully.")


if __name__ == "__main__":
    main()
