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
    from scripts.utils.arg_config import expand_cli_args_with_config
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from scripts.utils.arg_config import expand_cli_args_with_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate demonstrations for Isaac Lab environments.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional config file that expands to CLI args before parsing. "
            "Supports .json, .yaml, .yml, .csv, .txt, and .args."
        ),
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--garment_name",
        type=str,
        default=None,
        help="Garment name (for garment tasks), e.g. Top_Long_Unseen_0.",
    )
    parser.add_argument(
        "--garment_version",
        type=str,
        default=None,
        help="Garment version (for garment tasks), e.g. Release or Holdout.",
    )
    parser.add_argument(
        "--input_file", type=str, default="./datasets/dataset.hdf5", help="File name of the dataset to be annotated."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./datasets/dataset_annotated.hdf5",
        help="File name of the annotated output dataset file.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help=(
            "Specify task type. If your dataset is recorded with keyboard, you should set it to 'keyboard', otherwise not"
            " to set it and keep default value None."
        ),
    )
    parser.add_argument("--auto", action="store_true", default=False, help="Automatically annotate subtasks.")
    parser.add_argument(
        "--enable_pinocchio",
        action="store_true",
        default=False,
        help="Enable Pinocchio.",
    )
    parser.add_argument(
        "--garment_info_json",
        type=str,
        default=None,
        help="Path to garment_info.json for per-episode initial cloth pose replay.",
    )
    parser.add_argument(
        "--step_hz",
        type=int,
        default=30,
        help="Replay speed in Hz for annotation (set <=0 to disable throttling).",
    )
    parser.add_argument(
        "--ignore_replay_success",
        action="store_true",
        default=False,
        help=(
            "Allow annotation/export even when replayed episode does not satisfy success term. "
            "Useful for stochastic cloth scenes where deterministic replay can fail."
        ),
    )
    parser.add_argument(
        "--sanitize_datagen_poses",
        action="store_true",
        default=False,
        help=(
            "Sanitize datagen pose rotations to valid SO(3) before export. "
            "Disabled by default to expose upstream pose issues."
        ),
    )
    parser.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path of garment assets (for garment tasks).",
    )
    parser.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path to particle garment config yaml (for garment tasks).",
    )
    parser.add_argument(
        "--require_ik_actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require top-level demo/actions to be IK actions before annotation (LeIsaac-style strict pipeline). "
            "Disable with --no-require-ik-actions to allow joint-action replay."
        ),
    )
    parser.add_argument(
        "--ik_action_dim",
        type=int,
        default=16,
        help="Expected IK action dimension (16 for bimanual cloth, 8 for single-arm).",
    )
    parser.add_argument(
        "--ik_quat_order",
        type=str,
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order used inside IK actions.",
    )
    parser.add_argument(
        "--legacy_joint_replay",
        action="store_true",
        default=False,
        help=(
            "Allow legacy fallback replay from joint action streams when strict IK action replay cannot be used. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--strict_pose_z_gap_threshold",
        type=float,
        default=0.55,
        help=(
            "Maximum allowed mean z-gap between target_eef_pose and object_pose in recorded datagen_info. "
            "Used only when --require-ik-actions is enabled."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
        "--ik_auto_base_z_threshold",
        type=float,
        default=0.35,
        help=(
            "Auto frame heuristic threshold for IK actions: if mean z in IK action poses is below this value, "
            "actions are treated as base-frame."
        ),
    )
    parser.add_argument(
        "--mimic_ik_orientation_weight",
        type=float,
        default=0.01,
        help=(
            "Orientation weight forwarded to env IK conversion (target_eef_pose_to_action). "
            "Higher values enforce source wrist orientation more strongly; 0 disables orientation tracking."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw_argv = sys.argv[1:] if argv is None else argv
    args_cli = parser.parse_args(expand_cli_args_with_config(raw_argv, parser))

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
