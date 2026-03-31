"""Merge multiple MimicGen HDF5 datasets by appending demos from source files into a target."""

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


def merge(target_path: str, source_paths: list[str]) -> None:
    """Append all demos from each source file into *target_path*."""
    with h5py.File(target_path, "a") as target:
        target_data = target["data"]

        existing = [k for k in target_data.keys() if k.startswith("demo_")]
        next_index = max((int(k.split("_", 1)[1]) for k in existing), default=-1) + 1
        print(f"Target has {len(existing)} existing demos (next index: {next_index})")

        for source_path in source_paths:
            print(f"\nAppending from {source_path}:")
            with h5py.File(source_path, "r") as source:
                source_data = source["data"]
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

                print(f"  Added {len(source_demos)} demos ({total_added_samples} samples)")

        print(
            f"\nDone — target now has {next_index} demos, "
            f"{int(target_data.attrs.get('total', 0))} total samples."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Append demos from one or more source HDF5 datasets into a target."
    )
    parser.add_argument("target", type=str, help="Path to the target HDF5 file (modified in place).")
    parser.add_argument("sources", type=str, nargs="+", help="Paths to source HDF5 files (read only).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    for path in [args.target] + args.sources:
        if not Path(path).is_file():
            parser.error(f"File not found: {path}")

    merge(args.target, args.sources)


if __name__ == "__main__":
    main()
