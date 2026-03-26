# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Thin CLI wrapper for MimicGen dataset generation."""

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
    parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
    add_config_argument(parser)

    core_group = parser.add_argument_group("core workflow")
    core_group.add_argument("--task", type=str, default=None, help="Name of the task.")
    core_group.add_argument(
        "--generation_num_trials", type=int, help="Number of demos to be generated.", default=None
    )
    core_group.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
    )
    core_group.add_argument("--input_file", type=str, default=None, help="File path to the source dataset file.")
    core_group.add_argument(
        "--output_file",
        type=str,
        default="./datasets/output_dataset.hdf5",
        help="File path to export recorded and generated episodes.",
    )
    core_group.add_argument(
        "--garment_settle_steps",
        type=int,
        default=20,
        help=(
            "Number of post-reset hold-action steps used to let garment cloth settle before "
            "Mimic samples runtime object poses."
        ),
    )
    core_group.add_argument(
        "--logging_interval",
        type=int,
        default=1,
        help="CSV logging interval in env steps. Must be > 0.",
    )
    core_group.add_argument(
        "--log_success",
        action="store_true",
        default=False,
        help="Log garment success-term distances for env 0 at episode start and every 50 env steps.",
    )

    garment_group = parser.add_argument_group("garment and environment overrides")
    add_garment_override_arguments(garment_group)

    runtime_group = parser.add_argument_group("runtime and debugging")
    runtime_group.add_argument(
        "--pause_subtask",
        action="store_true",
        help="Pause after every subtask during generation for debugging; only useful with rendering.",
    )
    parser.add_argument(
        "--enable_pinocchio",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pose_output_interval",
        dest="logging_interval",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--print_poses",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    runtime_group.add_argument(
        "--save_pose_trace",
        action="store_true",
        default=False,
        help="Write a pose-trace CSV alongside generation for offline debugging.",
    )
    runtime_group.add_argument(
        "--pose_output_file",
        type=str,
        default=None,
        help=(
            "Optional CSV path for pose trace output. "
            "Defaults to <output_file stem>_pose_trace.csv."
        ),
    )

    compatibility_group = parser.add_argument_group("compatibility and legacy")
    compatibility_group.add_argument(
        "--use_eef_pose_as_target",
        action="store_true",
        default=False,
        help=(
            "Use measured datagen_info.eef_pose as source target trajectory "
            "(instead of recorded target_eef_pose)."
        ),
    )
    compatibility_group.add_argument(
        "--source_target_z_offset",
        type=float,
        default=0.0,
        help=(
            "Meters added to source target EEF z before generation. "
            "Use negative values to lower grasp trajectories (e.g. -0.02)."
        ),
    )
    compatibility_group.add_argument(
        "--align_object_pose_to_runtime",
        action="store_true",
        default=False,
        help=(
            "Enable legacy source pose alignment to runtime frame. "
            "Keep disabled unless you are repairing old datasets with known global frame offsets."
        ),
    )
    compatibility_group.add_argument(
        "--object_pose_alignment_mode",
        type=str,
        default="object_only",
        choices=["object_only", "all_poses"],
        help=(
            "Alignment behavior when --align_object_pose_to_runtime is enabled: "
            "'object_only' shifts source object poses only (for mixed-frame datasets), "
            "'all_poses' shifts object/eef/target together (for pure global frame shifts)."
        ),
    )
    compatibility_group.add_argument(
        "--auto_fix_mixed_pose_frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Auto-detect mixed source pose frames and enable object-only source object alignment when needed. "
            "Disabled by default in strict pipeline mode."
        ),
    )
    compatibility_group.add_argument(
        "--strict_preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run strict source dataset contract checks before generation "
            "(recommended; enabled by default)."
        ),
    )
    compatibility_group.add_argument(
        "--expected_source_action_dim",
        type=int,
        default=16,
        help="Expected top-level source action dimension for strict preflight checks.",
    )
    compatibility_group.add_argument(
        "--require_source_actions_mode",
        type=str,
        default="ee_pose",
        help="Required /data attrs['actions_mode'] value in strict preflight mode (set empty to skip).",
    )
    parser.add_argument(
        "--disable_object_pose_alignment",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    add_task_type_argument(compatibility_group)
    add_mimic_ik_orientation_weight_argument(compatibility_group)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw_argv = sys.argv[1:] if argv is None else argv
    if any(arg in {"-h", "--help"} for arg in raw_argv):
        parser.print_help()
        return
    expanded_argv = expand_cli_args_with_config(raw_argv, parser)
    warn_on_deprecated_flags(
        expanded_argv,
        {
            "--pose_output_interval": (
                "`--pose_output_interval` is deprecated; use `--logging_interval` instead."
            ),
            "--disable_object_pose_alignment": (
                "`--disable_object_pose_alignment` is deprecated and kept only for compatibility."
            ),
            "--enable_pinocchio": (
                "`--enable_pinocchio` is deprecated and kept only for compatibility."
            ),
            "--print_poses": "`--print_poses` is deprecated and no longer has any effect.",
        },
    )
    args_cli = parser.parse_args(expanded_argv)
    if not args_cli.input_file:
        parser.error("the following arguments are required: --input_file")

    if args_cli.enable_pinocchio:
        import pinocchio  # noqa: F401

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        from scripts.mimicgen.core.generate_runner import run_generation

        run_generation(args_cli, simulation_app)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
