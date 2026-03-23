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
    from scripts.utils.arg_config import expand_cli_args_with_config
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from scripts.utils.arg_config import expand_cli_args_with_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
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
    parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
    )
    parser.add_argument("--input_file", type=str, default=None, help="File path to the source dataset file.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="./datasets/output_dataset.hdf5",
        help="File path to export recorded and generated episodes.",
    )
    parser.add_argument("--task_type", type=str, default=None, help="Specify task type. If your annotated dataset is recorded with keyboard, you should set it to 'keyboard', otherwise not to set it and keep default value None.")
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
        "--pause_subtask",
        action="store_true",
        help="pause after every subtask during generation for debugging - only useful with render flag",
    )
    parser.add_argument(
        "--garment_settle_steps",
        type=int,
        default=20,
        help=(
            "Number of post-reset hold-action steps used to let garment cloth settle before "
            "Mimic samples runtime object poses."
        ),
    )
    parser.add_argument(
        "--enable_pinocchio",
        action="store_true",
        default=False,
        help="Enable Pinocchio.",
    )
    parser.add_argument(
        "--logging_interval",
        type=int,
        default=1,
        help="CSV logging interval in env steps. Must be > 0.",
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
    parser.add_argument(
        "--save_pose_trace",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pose_output_file",
        type=str,
        default=None,
        help=(
            "Optional CSV path for pose trace output. "
            "Defaults to <output_file stem>_pose_trace.csv."
        ),
    )
    parser.add_argument(
        "--log_success",
        action="store_true",
        default=False,
        help="Log garment success-term distances for env 0 at episode start and every 50 env steps.",
    )
    parser.add_argument(
        "--use_eef_pose_as_target",
        action="store_true",
        default=False,
        help=(
            "Use measured datagen_info.eef_pose as source target trajectory "
            "(instead of recorded target_eef_pose)."
        ),
    )
    parser.add_argument(
        "--source_target_z_offset",
        type=float,
        default=0.0,
        help=(
            "Meters added to source target EEF z before generation. "
            "Use negative values to lower grasp trajectories (e.g. -0.02)."
        ),
    )
    parser.add_argument(
        "--align_object_pose_to_runtime",
        action="store_true",
        default=False,
        help=(
            "Enable legacy source pose alignment to runtime frame. "
            "Keep disabled unless you are repairing old datasets with known global frame offsets."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
        "--auto_fix_mixed_pose_frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Auto-detect mixed source pose frames and enable object-only source object alignment when needed. "
            "Disabled by default in strict pipeline mode."
        ),
    )
    parser.add_argument(
        "--strict_preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run strict source dataset contract checks before generation "
            "(recommended; enabled by default)."
        ),
    )
    parser.add_argument(
        "--expected_source_action_dim",
        type=int,
        default=16,
        help="Expected top-level source action dimension for strict preflight checks.",
    )
    parser.add_argument(
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
    if any(arg in {"-h", "--help"} for arg in raw_argv):
        parser.print_help()
        return
    args_cli = parser.parse_args(expand_cli_args_with_config(raw_argv, parser))
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
