# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Thin CLI wrapper for MimicGen demo annotation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

try:
    from scripts.mimicgen.core.cli_parser import (
        add_config_argument,
        add_garment_override_arguments,
        add_mimic_ik_orientation_weight_argument,
        add_task_type_argument,
        warn_on_deprecated_flags,
    )
    from scripts.utils.arg_config import expand_cli_args_with_config
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from scripts.mimicgen.core.cli_parser import (
        add_config_argument,
        add_garment_override_arguments,
        add_mimic_ik_orientation_weight_argument,
        add_task_type_argument,
        warn_on_deprecated_flags,
    )
    from scripts.utils.arg_config import expand_cli_args_with_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="[Support Tool] Annotate demonstrations for Isaac Lab environments.",
        fromfile_prefix_chars="@",
    )
    add_config_argument(parser)

    core_group = parser.add_argument_group("core workflow")
    core_group.add_argument("--task", type=str, default=None, help="Name of the task.")
    core_group.add_argument(
        "--input_file", type=str, default="./datasets/dataset.hdf5", help="File name of the dataset to be annotated."
    )
    core_group.add_argument(
        "--output_file",
        type=str,
        default="./datasets/dataset_annotated.hdf5",
        help="File name of the annotated output dataset file.",
    )
    core_group.add_argument("--auto", action="store_true", default=False, help="Automatically annotate subtasks.")
    core_group.add_argument(
        "--step_hz",
        type=int,
        default=30,
        help="Replay speed in Hz for annotation (set <=0 to disable throttling).",
    )

    garment_group = parser.add_argument_group("garment and environment overrides")
    add_garment_override_arguments(garment_group)

    runtime_group = parser.add_argument_group("runtime and debugging")
    runtime_group.add_argument(
        "--garment_info_json",
        type=str,
        default=None,
        help="Path to garment_info.json for per-episode initial cloth pose replay.",
    )
    runtime_group.add_argument(
        "--ignore_replay_success",
        action="store_true",
        default=False,
        help=(
            "Allow annotation/export even when replayed episode does not satisfy success term. "
            "Useful for stochastic cloth scenes where deterministic replay can fail."
        ),
    )
    runtime_group.add_argument(
        "--sanitize_datagen_poses",
        action="store_true",
        default=False,
        help=(
            "Sanitize datagen pose rotations to valid SO(3) before export. "
            "Disabled by default to expose upstream pose issues."
        ),
    )

    compatibility_group = parser.add_argument_group("compatibility and legacy")
    parser.add_argument(
        "--enable_pinocchio",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    compatibility_group.add_argument(
        "--require_ik_actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require top-level demo/actions to be IK actions before annotation (LeIsaac-style strict pipeline). "
            "Disable with --no-require-ik-actions to allow joint-action replay."
        ),
    )
    compatibility_group.add_argument(
        "--ik_action_dim",
        type=int,
        default=16,
        help="Expected IK action dimension (16 for bimanual cloth, 8 for single-arm).",
    )
    compatibility_group.add_argument(
        "--ik_quat_order",
        type=str,
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order used inside IK actions.",
    )
    compatibility_group.add_argument(
        "--legacy_joint_replay",
        action="store_true",
        default=False,
        help=(
            "Allow legacy fallback replay from joint action streams when strict IK action replay cannot be used. "
            "Disabled by default."
        ),
    )
    compatibility_group.add_argument(
        "--strict_pose_z_gap_threshold",
        type=float,
        default=0.55,
        help=(
            "Maximum allowed mean z-gap between target_eef_pose and object_pose in recorded datagen_info. "
            "Used only when --require-ik-actions is enabled."
        ),
    )
    compatibility_group.add_argument(
        "--ik_action_frame",
        type=str,
        choices=["auto", "base", "world"],
        default="auto",
        help=(
            "Frame used by IK top-level actions. "
            "'base' means action poses are in each arm base frame and will be transformed to world for replay. "
            "'world' means action poses are already in world frame. "
            "'auto' tries dataset metadata and then a numeric heuristic."
        ),
    )
    compatibility_group.add_argument(
        "--ik_auto_base_z_threshold",
        type=float,
        default=0.35,
        help=(
            "Auto frame heuristic threshold for IK actions: if mean z in IK action poses is below this value, "
            "actions are treated as base-frame."
        ),
    )
    add_task_type_argument(compatibility_group)
    add_mimic_ik_orientation_weight_argument(compatibility_group)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw_argv = sys.argv[1:] if argv is None else argv
    expanded_argv = expand_cli_args_with_config(raw_argv, parser)
    warn_on_deprecated_flags(
        expanded_argv,
        {
            "--enable_pinocchio": (
                "`--enable_pinocchio` is deprecated and kept only for compatibility."
            ),
        },
    )
    args_cli = parser.parse_args(expanded_argv)

    if args_cli.enable_pinocchio:
        import pinocchio  # noqa: F401

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        from scripts.mimicgen.core.annotate_runner import run_annotation

        run_annotation(args_cli, simulation_app, cli_argv=list(raw_argv))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
