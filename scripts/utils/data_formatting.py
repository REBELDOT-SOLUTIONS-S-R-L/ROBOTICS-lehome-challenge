"""Data formatting utilities for LeRobot datasets."""

from pathlib import Path
import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_depth_column_schema(
    dataset_root: Path, column_name: str = "observation.top_depth"
) -> None:
    """Ensure a depth column has a stable Arrow schema for dataset merging."""
    dataset_root = dataset_root.resolve()
    data_root = dataset_root / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))

    if not parquet_files:
        return

    try:
        first_table = pq.read_table(parquet_files[0])
    except Exception as exc:
        logger.warning(f"Failed to read parquet file {parquet_files[0]}: {exc}")
        return

    if column_name not in first_table.column_names:
        return

    logger.info(
        f"Found {column_name} in {dataset_root.name}, "
        f"normalizing schema in {len(parquet_files)} parquet file(s)..."
    )

    for parquet_file in parquet_files:
        try:
            table = pq.read_table(parquet_file)
            if column_name not in table.column_names:
                continue

            fixed_values = []
            for item in table[column_name].to_pylist():
                if item is None:
                    fixed_values.append(None)
                    continue

                if isinstance(item, np.ndarray):
                    item = item.tolist()

                if isinstance(item, list):
                    rows = []
                    for row in item:
                        if isinstance(row, np.ndarray):
                            rows.append(row.astype(np.float32).tolist())
                        elif isinstance(row, list):
                            rows.append([float(value) for value in row])
                        else:
                            rows.append([float(row)])
                    fixed_values.append(rows)
                else:
                    fixed_values.append([[float(item)]])

            height = width = None
            for item in fixed_values:
                if item is not None and item and isinstance(item[0], list):
                    height = len(item)
                    width = len(item[0])
                    break

            if height is None or width is None:
                logger.warning(
                    f"Skipped depth normalization for {parquet_file}: cannot infer shape."
                )
                continue

            sample_value = None
            for item in fixed_values:
                if item is not None and item and isinstance(item[0], list) and item[0]:
                    sample_value = item[0][0]
                    break

            if sample_value is not None and isinstance(sample_value, (int, np.integer)):
                depth_type = pa.list_(pa.list_(pa.uint16(), width), height)
            else:
                depth_type = pa.list_(pa.list_(pa.float32(), width), height)

            new_column = pa.array(fixed_values, type=depth_type)
            column_index = table.column_names.index(column_name)
            table = table.remove_column(column_index)
            table = table.add_column(column_index, column_name, new_column)

            pq.write_table(table, parquet_file)
        except Exception as exc:
            logger.warning(
                f"Failed to normalize depth column in {parquet_file}: {exc}"
            )

    logger.info(f"Depth column normalization completed for {dataset_root.name}.")


def _clean_episodes_table(table: pa.Table, column_name: str) -> pa.Table:
    """Remove column-related episode stats and reset merged file indices."""
    columns_to_drop = [name for name in table.column_names if column_name in name]
    if columns_to_drop:
        logger.info(
            f"Removing {len(columns_to_drop)} episode statistics column(s) for {column_name}."
        )
        table = table.drop(columns_to_drop)

    index_columns = [
        name
        for name in table.column_names
        if name.endswith("/file_index") or name.endswith("/chunk_index")
    ]

    for column_name_in_table in index_columns:
        column_index = table.column_names.index(column_name_in_table)
        zero_array = pa.array([0] * table.num_rows, type=pa.int64())
        table = table.remove_column(column_index)
        table = table.add_column(column_index, column_name_in_table, zero_array)

    return table


def _sync_metadata_without_column(
    source_root: Path,
    output_root: Path,
    column_name: str,
    total_rows: int,
) -> None:
    """Copy metadata while removing the feature definition for a column."""
    for item in (source_root / "meta").glob("*"):
        if item.name == "episodes":
            continue

        destination = output_root / "meta" / item.name

        if item.name == "info.json":
            info = json.loads(item.read_text())
            info.get("features", {}).pop(column_name, None)
            info["chunks"] = 1
            info["total_frames"] = total_rows
            destination.write_text(json.dumps(info, indent=4))
        elif item.name == "stats.json":
            stats = json.loads(item.read_text())
            stats.pop(column_name, None)
            destination.write_text(json.dumps(stats, indent=4))
        elif item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def remove_dataset_column(
    dataset_root: Path,
    output_root: Path,
    column_name: str = "observation.top_depth",
    overwrite: bool = False,
) -> None:
    """Create a dataset copy with one data column removed and metadata resynced."""
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()

    if dataset_root == output_root:
        raise ValueError("output_root must be different from dataset_root.")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use overwrite=True to replace it."
            )
        shutil.rmtree(output_root)

    logger.info(f"Processing dataset copy: {dataset_root.name} -> {output_root.name}")

    output_data_chunk = output_root / "data" / "chunk-000"
    output_data_chunk.mkdir(parents=True, exist_ok=True)

    data_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    data_tables = []
    for parquet_file in data_files:
        table = pq.read_table(parquet_file)
        if column_name in table.column_names:
            table = table.drop([column_name])
        data_tables.append(table)

    merged_data_table = pa.concat_tables(data_tables)
    pq.write_table(merged_data_table, output_data_chunk / "file-000.parquet")

    output_episode_chunk = output_root / "meta" / "episodes" / "chunk-000"
    output_episode_chunk.mkdir(parents=True, exist_ok=True)

    episode_files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if episode_files:
        episode_tables = [pq.read_table(path) for path in episode_files]
        merged_episode_table = _clean_episodes_table(
            pa.concat_tables(episode_tables), column_name
        )
        pq.write_table(merged_episode_table, output_episode_chunk / "file-000.parquet")
    else:
        logger.warning(
            f"No episode parquet files found under {dataset_root / 'meta' / 'episodes'}."
        )

    _sync_metadata_without_column(
        dataset_root,
        output_root,
        column_name,
        total_rows=merged_data_table.num_rows,
    )

    videos_root = dataset_root / "videos"
    if videos_root.exists():
        shutil.copytree(
            videos_root,
            output_root / "videos",
            ignore=shutil.ignore_patterns(f"{column_name}.mp4"),
        )

    logger.info("Dataset column removal completed.")
