"""Shared environment-configuration helpers for MimicGen scripts."""

from __future__ import annotations

from typing import Any


def resolve_task_type(task_id: str, explicit_task_type: str | None) -> str:
    """Resolve task_type robustly for LeHome naming variants."""
    if explicit_task_type is not None:
        return explicit_task_type

    lowered = task_id.lower()
    if (
        "biso101" in lowered
        or "biarm" in lowered
        or "bimanual" in lowered
        or lowered.startswith("lehome-bi")
    ):
        return "bi-so101leader"

    from lehome.utils.env_utils import get_task_type

    return get_task_type(task_id, None)


def apply_common_mimic_env_overrides(env_cfg: Any, args: Any) -> None:
    """Apply the common CLI-driven Mimic overrides shared across flows."""
    setattr(env_cfg, "mimic_ik_orientation_weight", float(args.mimic_ik_orientation_weight))

    if hasattr(env_cfg, "garment_cfg_base_path"):
        env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    if hasattr(env_cfg, "particle_cfg_path"):
        env_cfg.particle_cfg_path = args.particle_cfg_path


def resolve_env_garment_metadata(
    env_cfg: Any,
    args: Any,
    dataset_env_args: dict[str, Any],
    *,
    fallback_garment_name: str | None = None,
    fallback_garment_version: str | None = None,
) -> tuple[Any, Any]:
    """Resolve garment name and version to assign onto an env config."""
    resolved_garment_name = (
        args.garment_name
        or dataset_env_args.get("garment_name")
        or fallback_garment_name
        or getattr(env_cfg, "garment_name", None)
    )
    resolved_garment_version = (
        args.garment_version
        or dataset_env_args.get("garment_version")
        or fallback_garment_version
        or getattr(env_cfg, "garment_version", None)
    )
    return resolved_garment_name, resolved_garment_version


def assign_env_garment_metadata(
    env_cfg: Any,
    garment_name: Any,
    garment_version: Any,
    *,
    missing_error_message: str,
) -> None:
    """Assign resolved garment metadata to an env config."""
    if garment_name is None or (
        isinstance(garment_name, str) and not garment_name.strip()
    ):
        raise ValueError(missing_error_message)

    env_cfg.garment_name = garment_name.strip() if isinstance(garment_name, str) else garment_name
    if hasattr(env_cfg, "garment_version") and garment_version is not None:
        env_cfg.garment_version = garment_version


def normalize_last_subtask_offsets_for_generation(env_cfg: Any) -> None:
    """Mimic generation requires the last subtask term offset range to be (0, 0)."""
    subtask_cfgs = getattr(env_cfg, "subtask_configs", None)
    if not isinstance(subtask_cfgs, dict):
        return

    for eef_name, configs in subtask_cfgs.items():
        if not configs:
            continue
        last_cfg = configs[-1]
        current = getattr(last_cfg, "subtask_term_offset_range", None)
        if current is None:
            continue
        if tuple(current) != (0, 0):
            print(
                f"Warning: overriding final subtask_term_offset_range for '{eef_name}' "
                f"from {tuple(current)} to (0, 0) for Mimic generation compatibility."
            )
            last_cfg.subtask_term_offset_range = (0, 0)
