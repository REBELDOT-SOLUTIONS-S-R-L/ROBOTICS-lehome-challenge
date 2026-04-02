from __future__ import annotations

import argparse
import warnings
from collections.abc import Mapping, Sequence


_CONFIG_HELP = (
    "Optional config file that expands to CLI args before parsing. "
    "Supports .json, .yaml, .yml, .csv, .txt, and .args."
)


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help=_CONFIG_HELP)


def add_garment_override_arguments(group: argparse._ArgumentGroup) -> None:
    group.add_argument(
        "--garment_name",
        type=str,
        default=None,
        help="Garment name override, for example Top_Long_Unseen_0.",
    )
    group.add_argument(
        "--garment_version",
        type=str,
        default=None,
        help="Garment split/version override, for example Release or Holdout.",
    )
    group.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path of garment assets.",
    )
    group.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path to the particle cloth config YAML.",
    )


def add_task_type_argument(group: argparse._ArgumentGroup) -> None:
    group.add_argument(
        "--task_type",
        type=str,
        default=None,
        help=(
            "Explicit task-type override for compatibility flows. "
            "For keyboard-recorded source datasets, set this to 'keyboard'."
        ),
    )


def add_mimic_ik_orientation_weight_argument(group: argparse._ArgumentGroup) -> None:
    group.add_argument(
        "--mimic_ik_orientation_weight",
        type=float,
        default=0.01,
        help=(
            "Orientation weight forwarded to env IK conversion (target_eef_pose_to_action). "
            "Higher values enforce source wrist orientation more strongly; 0 disables orientation tracking."
        ),
    )


def warn_on_deprecated_flags(argv: Sequence[str], deprecated_flags: Mapping[str, str]) -> None:
    emitted: set[str] = set()
    for flag, message in deprecated_flags.items():
        if flag in emitted:
            continue
        if any(arg == flag or arg.startswith(f"{flag}=") for arg in argv):
            warnings.warn(message, UserWarning, stacklevel=2)
            emitted.add(flag)
