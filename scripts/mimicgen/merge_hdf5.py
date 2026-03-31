"""Merge two MimicGen HDF5 datasets by appending demos from a second file into the first."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


def _copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    """Recursively copy all datasets, groups, and attributes from src into dst."""
    for attr_name, attr_value in src.attrs.items():
        dst.attrs[attr_name] = attr_value
    for key in src.keys():
        if isinstance(src[key], h5py.Group):
            child = dst.create_group(key)
            _copy_group(src[key], child)
        else:
            src.copy(src[key], dst, name=key)


def merge(target_path: str, source_path: str) -> None:
    """Append all demos from *source_path* into *target_path*."""
    with h5py.File(target_path, "a") as target, h5py.File(source_path, "r") as source:
        target_data = target["data"]
        source_data = source["data"]

        existing = [k for k in target_data.keys() if k.startswith("demo_")]
        next_index = max((int(k.split("_", 1)[1]) for k in existing), default=-1) + 1

        source_demos = sorted(
            [k for k in source_data.keys() if k.startswith("demo_")],
            key=lambda k: int(k.split("_", 1)[1]),
        )

        total_added_samples = 0
        for demo_name in source_demos:
            new_name = f"demo_{next_index}"
            new_group = target_data.create_group(new_name)
            _copy_group(source_data[demo_name], new_group)
            num_samples = int(new_group.attrs.get("num_samples", 0))
            total_added_samples += num_samples
            print(f"  {demo_name} -> {new_name}  ({num_samples} samples)")
            next_index += 1

        if "total" in target_data.attrs:
            target_data.attrs["total"] = int(target_data.attrs["total"]) + total_added_samples

        print(
            f"\nDone — appended {len(source_demos)} demos from {source_path} "
            f"into {target_path} (now {next_index} demos, "
            f"{int(target_data.attrs.get('total', 0))} total samples)."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Append demos from a source HDF5 dataset into a target HDF5 dataset."
    )
    parser.add_argument("target", type=str, help="Path to the target HDF5 file (modified in place).")
    parser.add_argument("source", type=str, help="Path to the source HDF5 file (read only).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    for path in (args.target, args.source):
        if not Path(path).is_file():
            parser.error(f"File not found: {path}")

    merge(args.target, args.source)


if __name__ == "__main__":
    main()
