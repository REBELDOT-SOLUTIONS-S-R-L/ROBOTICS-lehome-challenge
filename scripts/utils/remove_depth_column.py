#!/usr/bin/env python3
"""Compatibility wrapper for dataset column removal utilities."""

import argparse
from pathlib import Path

try:
    from scripts.utils.data_formatting import remove_dataset_column
except ImportError:
    from data_formatting import remove_dataset_column


def main():
    parser = argparse.ArgumentParser(
        description="[Internal] Remove a column from a parquet-based dataset copy."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--column_to_remove", type=str, default="observation.top_depth")
    args = parser.parse_args()

    remove_dataset_column(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        column_name=args.column_to_remove,
        overwrite=True,
    )


if __name__ == "__main__":
    main()
