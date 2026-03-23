"""Shared data-formatting helpers for MimicGen runtime modules."""

from __future__ import annotations

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


__all__ = ["as_2d_tensor", "as_tensor", "flatten_nested_leaves"]
