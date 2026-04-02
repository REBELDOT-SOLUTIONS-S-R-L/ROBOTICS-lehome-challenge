"""Shared HDF5 and dataset-loading helpers for MimicGen scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import h5py
except ImportError:
    h5py = None


def require_h5py(context: str = "HDF5 operations") -> None:
    """Raise a consistent error when h5py is unavailable."""
    if h5py is None:
        raise ImportError(
            f"h5py is required for {context}. Install it in your environment "
            "(for example, `pip install h5py`)."
        )


def decode_attr(value: Any) -> Any:
    """Decode HDF5 scalar attribute values to plain python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_json_if_possible(value: Any) -> Any:
    """Parse JSON strings opportunistically while leaving plain text untouched."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def to_python_scalar_or_list(value: Any) -> Any:
    """Convert h5py and numpy values into plain Python scalars or lists."""
    value = decode_attr(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return decode_attr(value.item())
        return value.tolist()
    return value


def normalize_hdf5_scalar(value: Any) -> Any:
    """Normalize a scalar-or-small-array HDF5 value into Python-native types."""
    value = to_python_scalar_or_list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def read_hdf5_node(node: Any) -> Any:
    """Recursively read an HDF5 node into plain Python structures."""
    if h5py is None:
        return None

    if isinstance(node, h5py.Dataset):
        return parse_json_if_possible(to_python_scalar_or_list(node[()]))

    if isinstance(node, h5py.Group):
        out: dict[str, Any] = {}
        for key, value in node.attrs.items():
            out[key] = parse_json_if_possible(to_python_scalar_or_list(value))
        for key in node.keys():
            out[key] = read_hdf5_node(node[key])
        return out

    return node


def demo_sort_key(name: str) -> tuple[int, int, str]:
    """Sort demo_N groups numerically before any non-standard entries."""
    if name.startswith("demo_"):
        suffix = name.split("demo_", maxsplit=1)[1]
        if suffix.isdigit():
            return 0, int(suffix), name
    return 1, 0, name


def is_supported_numeric_array(arr: np.ndarray) -> bool:
    """Return whether a numpy array can be losslessly converted to torch."""
    return np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)


def load_numeric_hdf5_group(
    group: Any,
    device: str,
    *,
    skip_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Load only numeric datasets from an HDF5 group recursively."""
    require_h5py("numeric-only HDF5 loading")

    skip_keys = skip_keys or set()
    out: dict[str, Any] = {}
    for key in group.keys():
        if key in skip_keys:
            continue
        node = group[key]
        if isinstance(node, h5py.Group):
            nested = load_numeric_hdf5_group(node, device=device, skip_keys=skip_keys)
            if nested:
                out[key] = nested
            continue
        array = np.array(node)
        if not is_supported_numeric_array(array):
            continue
        out[key] = torch.as_tensor(array, device=device)
    return out


def load_episode_with_numeric_datasets_only_from_group(
    h5_episode_group: Any,
    device: str,
    skip_keys: set[str] | None = None,
):
    """Load an EpisodeData while skipping non-numeric datasets such as /meta."""
    require_h5py("numeric-only episode loading")
    if h5_episode_group is None:
        raise KeyError("Missing HDF5 episode group for numeric-only fallback loading.")

    from isaaclab.utils.datasets import EpisodeData

    if skip_keys is None:
        skip_keys = {"meta"}

    episode = EpisodeData()
    episode.data = load_numeric_hdf5_group(h5_episode_group, device=device, skip_keys=skip_keys)

    if "seed" in h5_episode_group.attrs:
        try:
            episode.seed = int(h5_episode_group.attrs["seed"])
        except Exception:
            episode.seed = h5_episode_group.attrs["seed"]
    if "success" in h5_episode_group.attrs:
        episode.success = bool(h5_episode_group.attrs["success"])
    return episode


def load_episode_with_numeric_datasets_only_from_file(
    input_file: str | Path,
    episode_name: str,
    device: str,
    skip_keys: set[str] | None = None,
):
    """Open a dataset file and load one episode using numeric-only fallback logic."""
    require_h5py("numeric-only episode loading")
    with h5py.File(str(input_file), "r") as file:
        data_group = file.get("data", None)
        if data_group is None or episode_name not in data_group:
            raise KeyError(f"Episode {episode_name!r} not found in {input_file}")
        return load_episode_with_numeric_datasets_only_from_group(
            data_group[episode_name], device=device, skip_keys=skip_keys
        )


def load_episode_compat(
    dataset_file_handler: Any,
    episode_name: str,
    device: str,
    *,
    input_file: str | Path | None = None,
    h5_episode_group: Any | None = None,
    info_prefix: str = "Info",
    obs_skip_keys: set[str] | frozenset[str] | None = None,
):
    """Load an episode robustly across mixed numeric/string HDF5 schemas.

    Args:
        obs_skip_keys: Observation keys to exclude (e.g. camera image streams).
            Skipped at the I/O level so large tensors are never read from disk.
    """
    # When obs_skip_keys is requested we bypass the default loader entirely and
    # use the selective numeric loader so camera images are never read from disk.
    if obs_skip_keys:
        skip = {"meta"} | set(obs_skip_keys)
        # Try the h5py group from the handler's internal state first.
        h5_group = h5_episode_group
        if h5_group is None:
            data_group = getattr(dataset_file_handler, "_hdf5_data_group", None)
            if data_group is not None and episode_name in data_group:
                h5_group = data_group[episode_name]
        if h5_group is not None:
            return load_episode_with_numeric_datasets_only_from_group(
                h5_group, device=device, skip_keys=skip,
            )
        if input_file is not None:
            return load_episode_with_numeric_datasets_only_from_file(
                input_file, episode_name, device=device, skip_keys=skip,
            )

    try:
        episode = dataset_file_handler.load_episode(episode_name, device)
        if episode is None:
            raise ValueError(f"Episode {episode_name!r} not found.")
        return episode
    except TypeError as exc:
        if "numpy.object_" not in str(exc):
            raise
        print(
            f"{info_prefix}: default episode loader failed on non-numeric datasets ({exc}). "
            "Using numeric-only fallback loader."
        )
        skip = {"meta"}
        if h5_episode_group is not None:
            return load_episode_with_numeric_datasets_only_from_group(
                h5_episode_group,
                device=device,
                skip_keys=skip,
            )
        if input_file is not None:
            return load_episode_with_numeric_datasets_only_from_file(
                input_file,
                episode_name,
                device=device,
                skip_keys=skip,
            )
        raise
