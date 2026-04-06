import argparse


def setup_record_parser(
    subparsers: argparse.ArgumentParser, parent_parsers: list[argparse.ArgumentParser]
) -> argparse.ArgumentParser:
    """Setup parser for 'record' subcommand."""
    parser = subparsers.add_parser(
        "record",
        help="Record teleoperation data",
        parents=parent_parsers,
        conflict_handler="resolve",
    )

    core_group = parser.add_argument_group("core workflow")
    core_group.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
    core_group.add_argument(
        "--task",
        type=str,
        default="LeHome-BiSO101-Direct-Garment-v2",
        help="Name of the task.",
    )
    core_group.add_argument(
        "--num_episode",
        type=int,
        default=20,
        help="Maximum number of episodes to record",
    )
    core_group.add_argument(
        "--dataset_root",
        type=str,
        default="Datasets/record",
        help="Root directory for saving recorded datasets (default: Datasets/record)",
    )
    core_group.add_argument(
        "--enable_record",
        action="store_true",
        default=False,
        help="Enable dataset recording function",
    )

    teleop_group = parser.add_argument_group("teleoperation")
    teleop_group.add_argument(
        "--teleop_device",
        type=str,
        default="keyboard",
        choices=["keyboard", "bi-keyboard", "so101leader", "bi-so101leader"],
        help="Device for interacting with environment",
    )
    teleop_group.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Port for the teleop device:so101leader, default is /dev/ttyACM0",
    )
    teleop_group.add_argument(
        "--left_arm_port",
        type=str,
        default="/dev/ttyUSB0",
        help="Port for the left teleop device:bi-so101leader, default is /dev/ttyUSB0",
    )
    teleop_group.add_argument(
        "--right_arm_port",
        type=str,
        default="/dev/ttyUSB1",
        help="Port for the right teleop device:bi-so101leader, default is /dev/ttyUSB1",
    )
    teleop_group.add_argument(
        "--recalibrate",
        action="store_true",
        default=False,
        help="recalibrate SO101-Leader or Bi-SO101Leader",
    )
    teleop_group.add_argument(
        "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
    )

    garment_group = parser.add_argument_group("garment and environment overrides")
    garment_group.add_argument(
        "--garment_name",
        type=str,
        default="Top_Long_Unseen_0",
        help="Name of the garment.",
    )
    garment_group.add_argument(
        "--garment_version", type=str, default="Release", help="Version of the garment."
    )
    garment_group.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path of the garment configuration.",
    )
    garment_group.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path of the particle configuration.",
    )

    runtime_group = parser.add_argument_group("runtime and debugging")
    runtime_group.add_argument(
        "--use_random_seed",
        action="store_true",
        default=False,
        help="Use random seed for the environment.",
    )
    runtime_group.add_argument(
        "--seed", type=int, default=42, help="Seed for the environment."
    )
    runtime_group.add_argument(
        "--log_success",
        action="store_true",
        default=False,
        help="Log success information.",
    )
    runtime_group.add_argument(
        "--debugging_log_pose",
        action="store_true",
        default=False,
        help="Print EEF and garment checkpoint positions during teleoperation.",
    )
    runtime_group.add_argument(
        "--debugging_markers",
        action="store_true",
        default=False,
        help="Show live garment semantic keypoint markers during teleoperation.",
    )
    runtime_group.add_argument(
        "--step_hz", type=int, default=120, help="Environment stepping rate in Hz."
    )

    recording_group = parser.add_argument_group("recording and dataset options")
    recording_group.add_argument(
        "--disable_depth",
        action="store_true",
        default=False,
        help="Disable using top depth observation in env and dataset.",
    )
    recording_group.add_argument(
        "--enable_pointcloud",
        action="store_true",
        default=False,
        help="Whether to enable pointcloud observation in env and dataset.",
    )
    recording_group.add_argument(
        "--task_description",
        type=str,
        default="fold the garment on the table",
        help=" Description of the task to be performed.",
    )
    recording_group.add_argument(
        "--record_ee_pose",
        action="store_true",
        default=False,
        help="Record end-effector pose online (requires Pinocchio and scipy)",
    )
    recording_group.add_argument(
        "--ee_urdf_path",
        type=str,
        default=None,
        help="URDF file path (required only when using --record_ee_pose)",
    )
    recording_group.add_argument(
        "--ee_state_unit",
        type=str,
        default="rad",
        choices=["deg", "rad"],
        help="Joint angle unit for kinematic solver (default: rad)",
    )

    return parser


def setup_record_annotated_parser(
    subparsers: argparse.ArgumentParser, parent_parsers: list[argparse.ArgumentParser]
) -> argparse.ArgumentParser:
    """Setup parser for 'record_annotated' subcommand."""
    parser = subparsers.add_parser(
        "record_annotated",
        help="Record generation-ready Mimic teleoperation data with online annotation",
        parents=parent_parsers,
        conflict_handler="resolve",
    )

    core_group = parser.add_argument_group("core workflow")
    core_group.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
    core_group.add_argument(
        "--task",
        type=str,
        default="LeHome-BiSO101-ManagerBased-Garment-v0",
        help="Name of the task.",
    )
    core_group.add_argument(
        "--num_episode",
        type=int,
        default=20,
        help="Maximum number of episodes to record",
    )
    core_group.add_argument(
        "--dataset_root",
        type=str,
        default="Datasets/hdf5_datasets/3_annotated_datasets/annotated_dataset.hdf5",
        help="Output HDF5 path or root directory for annotated datasets.",
    )

    teleop_group = parser.add_argument_group("teleoperation")
    teleop_group.add_argument(
        "--teleop_device",
        type=str,
        default="bi-so101leader",
        choices=["keyboard", "bi-keyboard", "so101leader", "bi-so101leader"],
        help="Device for interacting with environment",
    )
    teleop_group.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Port for the teleop device:so101leader, default is /dev/ttyACM0",
    )
    teleop_group.add_argument(
        "--left_arm_port",
        type=str,
        default="/dev/ttyUSB0",
        help="Port for the left teleop device:bi-so101leader, default is /dev/ttyUSB0",
    )
    teleop_group.add_argument(
        "--right_arm_port",
        type=str,
        default="/dev/ttyUSB1",
        help="Port for the right teleop device:bi-so101leader, default is /dev/ttyUSB1",
    )
    teleop_group.add_argument(
        "--recalibrate",
        action="store_true",
        default=False,
        help="recalibrate SO101-Leader or Bi-SO101Leader",
    )
    teleop_group.add_argument(
        "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
    )

    garment_group = parser.add_argument_group("garment and environment overrides")
    garment_group.add_argument(
        "--garment_name",
        type=str,
        default="Top_Long_Unseen_0",
        help="Name of the garment.",
    )
    garment_group.add_argument(
        "--garment_version", type=str, default="Release", help="Version of the garment."
    )
    garment_group.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path of the garment configuration.",
    )
    garment_group.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path of the particle configuration.",
    )

    runtime_group = parser.add_argument_group("runtime and debugging")
    runtime_group.add_argument(
        "--use_random_seed",
        action="store_true",
        default=False,
        help="Use random seed for the environment.",
    )
    runtime_group.add_argument(
        "--seed", type=int, default=42, help="Seed for the environment."
    )
    runtime_group.add_argument(
        "--log_success",
        action="store_true",
        default=False,
        help="Log success information.",
    )
    runtime_group.add_argument(
        "--debugging_log_pose",
        action="store_true",
        default=False,
        help="Print EEF and garment checkpoint positions during teleoperation.",
    )
    runtime_group.add_argument(
        "--debugging_markers",
        action="store_true",
        default=False,
        help="Show live garment semantic keypoint markers during annotated teleoperation.",
    )
    runtime_group.add_argument(
        "--step_hz", type=int, default=90, help="Environment stepping rate in Hz."
    )

    recording_group = parser.add_argument_group("recording and dataset options")
    recording_group.add_argument(
        "--task_description",
        type=str,
        default="fold the garment on the table",
        help="Description of the task to be performed.",
    )

    return parser


def setup_replay_parser(
    subparsers: argparse.ArgumentParser, parent_parsers: list[argparse.ArgumentParser]
) -> argparse.ArgumentParser:
    """Setup parser for 'replay' subcommand."""
    parser = subparsers.add_parser(
        "replay",
        help="Replay dataset",
        parents=parent_parsers,
        conflict_handler="resolve",
    )

    core_group = parser.add_argument_group("core workflow")
    core_group.add_argument(
        "--task",
        type=str,
        default="LeHome-BiSO101-Direct-Garment-v2",
        help="Name of the task environment.",
    )
    core_group.add_argument(
        "--step_hz", type=int, default=60, help="Environment stepping rate in Hz."
    )
    core_group.add_argument(
        "--dataset_root",
        type=str,
        default="Datasets/record/example/record_top_long_release_10/001",
        help="Root directory of the dataset to replay.",
    )
    core_group.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory to save replayed episodes (if None, replay only without saving).",
    )
    core_group.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="Starting episode index (inclusive).",
    )
    core_group.add_argument(
        "--end_episode",
        type=int,
        default=None,
        help="Ending episode index (exclusive). If None, replay all episodes.",
    )
    core_group.add_argument(
        "--save_successful_only",
        action="store_true",
        default=False,
        help="Only save episodes that achieve success during replay.",
    )

    runtime_group = parser.add_argument_group("runtime and debugging")
    runtime_group.add_argument(
        "--num_replays",
        type=int,
        default=1,
        help="Number of times to replay each episode.",
    )
    runtime_group.add_argument(
        "--task_description",
        type=str,
        default="fold the garment on the table",
        help="Description of the task to be performed.",
    )
    runtime_group.add_argument(
        "--disable_depth",
        action="store_true",
        default=False,
        help="Disable depth observation during replay.",
    )
    runtime_group.add_argument(
        "--debugging_markers",
        action="store_true",
        default=False,
        help="Show live garment semantic keypoint markers during HDF5 replay.",
    )

    garment_group = parser.add_argument_group("garment and environment overrides")
    garment_group.add_argument(
        "--garment_name",
        type=str,
        default=None,
        help="Garment name override for tasks that require an explicit cloth asset.",
    )
    garment_group.add_argument(
        "--garment_version", type=str, default="Release", help="Version of the garment."
    )
    garment_group.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path of the garment configuration.",
    )
    garment_group.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path of the particle configuration.",
    )

    compatibility_group = parser.add_argument_group("compatibility and legacy")
    compatibility_group.add_argument(
        "--use_ee_pose",
        action="store_true",
        default=False,
        help="Use action.ee_pose (Cartesian space) control, converted to joint angles via IK.",
    )
    compatibility_group.add_argument(
        "--ee_urdf_path",
        type=str,
        default="Assets/robots/so101_new_calib.urdf",
        help="URDF file path (required when using --use_ee_pose).",
    )
    compatibility_group.add_argument(
        "--ee_state_unit",
        type=str,
        default="rad",
        choices=["deg", "rad"],
        help="Joint angle unit for kinematic solver (default: rad).",
    )

    return parser


def setup_inspect_parser(
    subparsers: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Setup parser for 'inspect' subcommand."""
    parser = subparsers.add_parser("inspect", help="Inspect dataset metadata")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--show_frames", type=int, default=None, help="Display first N frames"
    )
    parser.add_argument("--show_stats", action="store_true", help="Display statistics")
    return parser


def setup_read_parser(subparsers: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Setup parser for 'read' subcommand."""
    parser = subparsers.add_parser("read", help="Read dataset states")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--num_frames", type=int, default=None, help="Number of frames to read"
    )
    parser.add_argument(
        "--episode", type=int, default=None, help="Specific episode index"
    )
    parser.add_argument("--output_csv", type=str, default=None, help="Export to CSV")
    parser.add_argument("--show_stats", action="store_true", help="Display statistics")
    return parser


def setup_augment_parser(
    subparsers: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Setup parser for 'augment' subcommand."""
    parser = subparsers.add_parser("augment", help="Add end-effector pose to dataset")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument("--urdf_path", type=str, required=True, help="URDF file path")
    parser.add_argument(
        "--state_unit",
        type=str,
        default="rad",
        choices=["rad", "deg"],
        help="Joint angle unit",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output directory (default: in-place)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing EE pose data"
    )
    return parser


def setup_merge_parser(subparsers: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Setup parser for 'merge' subcommand."""
    parser = subparsers.add_parser("merge", help="Merge multiple datasets")
    parser.add_argument(
        "--source_roots",
        type=str,
        required=True,
        help="List of source dataset directories (as Python list string)",
    )
    parser.add_argument(
        "--output_root", type=str, required=True, help="Output dataset directory"
    )
    parser.add_argument(
        "--output_repo_id", type=str, default="merged_dataset", help="Repository ID"
    )
    parser.add_argument(
        "--merge_custom_meta",
        action="store_true",
        default=True,
        help="Merge custom meta files",
    )
    return parser


def setup_eval_parser() -> argparse.ArgumentParser:
    """Setup parser for evaluation script.

    Returns:
        The parser with evaluation arguments added.
    """
    parser = argparse.ArgumentParser(
        description="A script for evaluating policy in lehome manipulation environments."
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

    core_group = parser.add_argument_group("core workflow")
    core_group.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
    core_group.add_argument(
        "--max_steps",
        type=int,
        default=600,
        help="Maximum number of steps per evaluation episode.",
    )
    core_group.add_argument(
        "--task",
        type=str,
        default="LeHome-BiSO101-Direct-Garment-v2",
        help="Name of the task.",
    )
    core_group.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run for each garment.",
    )
    core_group.add_argument(
        "--step_hz", type=int, default=120, help="Environment stepping rate in Hz."
    )

    runtime_group = parser.add_argument_group("runtime and debugging")
    runtime_group.add_argument(
        "--use_random_seed",
        action="store_true",
        default=False,
        help="Use random seed for the environment.",
    )
    runtime_group.add_argument(
        "--seed", type=int, default=42, help="Seed for the environment."
    )

    garment_group = parser.add_argument_group("garment and environment overrides")
    garment_group.add_argument(
        "--garment_type",
        type=str,
        default="top_long",
        choices=["top_long", "top_short", "pant_long", "pant_short", "custom"],
        help="Type of garments to evaluate.",
    )
    garment_group.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path to the garment configuration files.",
    )
    garment_group.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path to the particle configuration file.",
    )
    garment_group.add_argument(
        "--task_description",
        type=str,
        default="fold the garment on the table",
        help="Task description for VLA models (used in complementary_data).",
    )

    outputs_group = parser.add_argument_group("outputs")
    outputs_group.add_argument(
        "--save_video",
        action="store_true",
        help="If set, save evaluation episodes as video.",
    )
    outputs_group.add_argument(
        "--video_dir",
        type=str,
        default="outputs/eval_videos",
        help="Directory to save evaluation videos.",
    )
    outputs_group.add_argument(
        "--save_datasets",
        action="store_true",
        help="If set, save evaluation episodes dataset(only success).",
    )
    outputs_group.add_argument(
        "--eval_dataset_path",
        type=str,
        default="Datasets/eval",
        help="Path to save evaluation datasets.",
    )

    policy_group = parser.add_argument_group("policy")
    policy_group.add_argument(
        "--policy_type",
        type=str,
        default="lerobot",
        help=(
            "Type of policy to use. Available policies are registered in PolicyRegistry. "
            "Built-in options: 'lerobot', 'custom'. "
            "Participants can register their own policies using @PolicyRegistry.register('my_policy')."
        ),
    )
    policy_group.add_argument(
        "--policy_path",
        type=str,
        default="outputs/train/diffusion_fold_1/checkpoints/100000/pretrained_model",
        help="Path to the pretrained IL policy checkpoint.",
    )
    policy_group.add_argument(
        "--dataset_root",
        type=str,
        help="Path of the train dataset (for metadata).",
    )

    compatibility_group = parser.add_argument_group("compatibility and legacy")
    compatibility_group.add_argument(
        "--use_ee_pose",
        action="store_true",
        help="If set, policy outputs end-effector poses instead of joint angles. IK will be used to convert to joint angles.",
    )
    compatibility_group.add_argument(
        "--ee_urdf_path",
        type=str,
        default="Assets/robots/so101_new_calib.urdf",
        help="URDF path for IK solver (required when --use_ee_pose is set).",
    )

    return parser
