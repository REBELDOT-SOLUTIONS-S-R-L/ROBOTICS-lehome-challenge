"""Shared data-formatting helpers for MimicGen runtime modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


def as_tensor(data: Any, squeeze_second_dim: bool = False) -> torch.Tensor | None:
    """Convert arrays or recorder list[tensor] buffers into a dense tensor."""
    if data is None:
        return None

    if torch.is_tensor(data):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        if len(data) == 0:
            return None
        if any(torch.is_tensor(item) or isinstance(item, np.ndarray) for item in data):
            elems = []
            for item in data:
                if item is None:
                    return None
                elem = item if torch.is_tensor(item) else torch.as_tensor(item)
                elems.append(elem)
            try:
                tensor = torch.stack(elems, dim=0)
            except RuntimeError:
                return None
        else:
            try:
                tensor = torch.as_tensor(data)
            except (TypeError, ValueError):
                return None
    else:
        try:
            tensor = torch.as_tensor(data)
        except (TypeError, ValueError):
            return None

    if squeeze_second_dim and tensor.ndim >= 3 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    return tensor


def as_2d_tensor(data: Any) -> torch.Tensor | None:
    """Convert array-like input to a 2D tensor, or return None if unavailable."""
    tensor = as_tensor(data, squeeze_second_dim=True)
    if tensor is None:
        return None
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        return None
    return tensor


def as_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert tensor or array-like values to a numpy array."""
    if isinstance(value, np.ndarray):
        arr = value
    elif torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def to_json_compatible(value: Any) -> Any:
    """Recursively convert tensors and arrays into JSON-serializable structures."""
    if isinstance(value, dict):
        return {str(k): to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_compatible(v) for v in value]
    if isinstance(value, tuple):
        return [to_json_compatible(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, Mapping):
        return {str(k): to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_compatible(v) for v in value]
    return value


def quat_wxyz_to_xyzw(quat: Any) -> np.ndarray:
    """Convert quaternion values from (w, x, y, z) to (x, y, z, w)."""
    quat_arr = as_numpy(quat, dtype=np.float32).reshape(-1)
    if quat_arr.shape[0] != 4:
        return quat_arr
    return np.array([quat_arr[1], quat_arr[2], quat_arr[3], quat_arr[0]], dtype=np.float32)


def flatten_nested_leaves(
    node: Any,
    prefix: str = "",
    skip_root_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Flatten nested dict leaves into slash-delimited key paths."""
    leaves: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if not prefix and skip_root_keys and key in skip_root_keys:
                continue
            next_prefix = f"{prefix}/{key}" if prefix else key
            leaves.update(flatten_nested_leaves(value, prefix=next_prefix, skip_root_keys=None))
        return leaves

    if prefix:
        leaves[prefix] = node
    return leaves


__all__ = [
    "as_2d_tensor",
    "as_numpy",
    "as_tensor",
    "flatten_nested_leaves",
    "quat_wxyz_to_xyzw",
    "to_json_compatible",
]
