# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dataset annotation orchestration for MimicGen demos."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch

try:
    import h5py
except ImportError:
    h5py = None

with contextlib.suppress(Exception):
    import isaaclab_mimic.envs  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import lehome.tasks  # noqa: F401

from .annotation_runtime import (
    annotate_episode_in_auto_mode,
    annotate_episode_in_manual_mode,
    build_replay_runtime_context,
    get_robot_eef_pose_world,
    normalize_manual_subtask_signal_name,
    prepare_replay_plan,
    recover_from_object_pose_capture_failure,
    stabilize_garment_after_reset_for_annotation,
)
from .annotation_session import AnnotationSessionController
from .dataset_io import load_episode_compat as _load_episode_compat
from .dataset_meta import build_dataset_metadata_index as _build_dataset_metadata_index
from .dataset_meta import get_first_garment_name as _get_first_garment_name
from .dataset_meta import load_dataset_env_args as _load_dataset_env_args
from .dataset_meta import load_garment_info_json as _load_garment_info
from .env_setup import apply_common_mimic_env_overrides
from .env_setup import assign_env_garment_metadata
from .env_setup import resolve_env_garment_metadata
from .env_setup import resolve_task_type as _resolve_task_type

try:
    from scripts.utils.annotate_utils import (
        DatagenObjectPoseCaptureError,
        DatasetMetadataIndex,
        resolve_valid_annotation_object_pose as _resolve_valid_annotation_object_pose,
        sanitize_pose_dict as _sanitize_pose_dict,
    )
except ImportError:
    scripts_dir = Path(__file__).resolve().parents[2]
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    from utils.annotate_utils import (
        DatagenObjectPoseCaptureError,
        DatasetMetadataIndex,
        resolve_valid_annotation_object_pose as _resolve_valid_annotation_object_pose,
        sanitize_pose_dict as _sanitize_pose_dict,
    )


OBJECT_POSE_CAPTURE_MAX_ATTEMPTS = 3


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records the datagen info data in each step."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            try:
                eef_pose = self._env.get_robot_eef_pose(eef_name=eef_name)
            except Exception:
                eef_pose = get_robot_eef_pose_world(self._env, eef_name)
            eef_pose_dict[eef_name] = eef_pose

        object_pose = _resolve_valid_annotation_object_pose(self._env)

        runtime_args = getattr(self._env, "_annotation_runtime_args", None)
        sanitize_poses = bool(getattr(runtime_args, "sanitize_datagen_poses", False))
        if sanitize_poses:
            object_pose = _sanitize_pose_dict(object_pose)
            eef_pose_dict = _sanitize_pose_dict(eef_pose_dict)
            target_eef_pose = _sanitize_pose_dict(
                self._env.action_to_target_eef_pose(self._env.action_manager.action)
            )
        else:
            target_eef_pose = self._env.action_to_target_eef_pose(self._env.action_manager.action)

        datagen_info = {
            "object_pose": object_pose,
            "eef_pose": eef_pose_dict,
            "target_eef_pose": target_eef_pose,
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for the datagen info recorder term."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Mimic specific recorder terms."""

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


def _build_subtask_term_signal_names(env: ManagerBasedRLMimicEnv) -> dict[str, list[str]]:
    """Resolve manual-annotation signal names from environment subtask configs."""
    subtask_term_signal_names: dict[str, list[str]] = {}
    for eef_name, eef_subtask_configs in env.cfg.subtask_configs.items():
        subtask_term_signal_names[eef_name] = [
            normalize_manual_subtask_signal_name(
                subtask_config.subtask_term_signal,
                eef_name,
                subtask_index,
            )
            for subtask_index, subtask_config in enumerate(eef_subtask_configs)
        ]
    return subtask_term_signal_names


def _create_keyboard_interface(args_cli, session: AnnotationSessionController):
    """Create and configure the keyboard interface when running with a GUI."""
    if args_cli.headless or os.environ.get("HEADLESS", 0):
        return None

    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

    try:
        keyboard_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    except TypeError:
        keyboard_interface = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1)
    session.register_callbacks(keyboard_interface, auto_mode=bool(args_cli.auto))
    keyboard_interface.reset()
    return keyboard_interface


def run_annotation(parsed_args, simulation_app_instance, *, cli_argv: list[str] | None = None):
    """Add Isaac Lab Mimic annotations to the given demo dataset file."""
    args_cli = parsed_args
    raw_cli_argv = list(cli_argv or [])

    if bool(args_cli.enable_pinocchio):
        with contextlib.suppress(Exception):
            import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_file} does not exist.")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    env = None
    keyboard_interface = None
    try:
        dataset_env_args = _load_dataset_env_args(args_cli.input_file)
        env_name = dataset_file_handler.get_env_name()
        episode_count = dataset_file_handler.get_num_episodes()
        dataset_metadata = _build_dataset_metadata_index(
            getattr(dataset_file_handler, "_hdf5_data_group", None),
            index_cls=DatasetMetadataIndex,
            warning_printer=print,
        )

        garment_info = dataset_metadata.garment_info
        if garment_info is not None:
            print("Using garment initial poses from HDF5 demo data (/data/demo_*/meta or initial_state/garment).")
        else:
            garment_info_path = args_cli.garment_info_json
            if garment_info_path is None:
                candidate = Path(args_cli.input_file).parent / "meta" / "garment_info.json"
                if candidate.exists():
                    garment_info_path = str(candidate)
            garment_info = _load_garment_info(garment_info_path)
            if garment_info is not None:
                print(f"Using garment initial poses from: {garment_info_path}")

        source_actions_frame_hint = dataset_metadata.actions_frame
        if source_actions_frame_hint is not None:
            print(f"Source actions frame hint from dataset: {source_actions_frame_hint}")

        source_ik_quat_order_hint = dataset_metadata.ik_quat_order
        explicit_ik_quat_order = any(
            arg == "--ik_quat_order" or arg.startswith("--ik_quat_order=") for arg in raw_cli_argv
        )
        if source_ik_quat_order_hint is not None and not explicit_ik_quat_order:
            args_cli.ik_quat_order = source_ik_quat_order_hint
            print(f"Source IK quaternion order from dataset: {source_ik_quat_order_hint}")
        elif source_ik_quat_order_hint is not None:
            print(
                "Ignoring dataset IK quaternion order "
                f"({source_ik_quat_order_hint}) because --ik_quat_order was set explicitly."
            )

        if episode_count == 0:
            print("No episodes found in the dataset.")
            return

        output_dir = os.path.dirname(args_cli.output_file)
        output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        task_id = args_cli.task or env_name
        if task_id is None:
            raise ValueError("Task/env name was not specified nor found in the dataset.")

        env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=1)
        task_type = _resolve_task_type(task_id, args_cli.task_type)
        setattr(env_cfg, "task_type", task_type)
        env_cfg.env_name = task_id
        apply_common_mimic_env_overrides(env_cfg, args_cli)
        print(f"Using mimic IK orientation_weight={float(args_cli.mimic_ik_orientation_weight):.4f}")

        if hasattr(env_cfg, "garment_name"):
            resolved_garment_name, resolved_garment_version = resolve_env_garment_metadata(
                env_cfg,
                args_cli,
                dataset_env_args,
                fallback_garment_name=_get_first_garment_name(garment_info),
            )
            assign_env_garment_metadata(
                env_cfg,
                resolved_garment_name,
                resolved_garment_version,
                missing_error_message=(
                    "This task requires a garment_name, but none was provided. "
                    "Pass --garment_name (e.g., Top_Long_Unseen_0) or include garment_name in data/env_args."
                ),
            )
            print(
                f"Using garment config: name={env_cfg.garment_name}, "
                f"version={getattr(env_cfg, 'garment_version', 'N/A')}"
            )

        success_term = None
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        else:
            raise NotImplementedError("No success termination term was found in the environment.")

        env_cfg.terminations = None
        env_cfg.recorders = MimicRecorderManagerCfg()
        if hasattr(env_cfg.recorders, "record_pre_step_flat_policy_observations"):
            env_cfg.recorders.record_pre_step_flat_policy_observations = None
        if not args_cli.auto:
            env_cfg.recorders.record_pre_step_subtask_term_signals = None

        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name

        env = gym.make(task_id, cfg=env_cfg).unwrapped
        if not isinstance(env, ManagerBasedRLMimicEnv):
            raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")
        setattr(env, "_annotation_runtime_args", args_cli)

        if args_cli.auto:
            if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
                raise NotImplementedError(
                    "The environment does not implement the get_subtask_term_signals method required "
                    "to run automatic annotations."
                )
            subtask_term_signal_names = None
        else:
            subtask_term_signal_names = _build_subtask_term_signal_names(env)

        env.reset()
        if hasattr(env, "initialize_obs"):
            try:
                env.initialize_obs()
                stabilize_garment_after_reset_for_annotation(env)
            except Exception as exc:
                print(f"Warning: initial garment initialization/stabilization failed: {exc}")

        replay_runtime = build_replay_runtime_context(env, args_cli)
        session = AnnotationSessionController()
        keyboard_interface = _create_keyboard_interface(args_cli, session)

        exported_episode_count = 0
        processed_episode_count = 0
        with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
            while simulation_app_instance.is_running() and not simulation_app_instance.is_exiting():
                for episode_index, episode_name in enumerate(dataset_file_handler.get_episode_names()):
                    processed_episode_count += 1
                    print(f"\nAnnotating episode #{episode_index} ({episode_name})")
                    episode = _load_episode_compat(
                        dataset_file_handler,
                        episode_name,
                        env.device,
                        input_file=args_cli.input_file,
                        h5_episode_group=dataset_metadata.episode_groups.get(episode_name),
                        info_prefix="\tInfo",
                    )
                    source_episode_index = dataset_metadata.source_episode_indices.get(
                        episode_name,
                        episode_index,
                    )
                    replay_plan = prepare_replay_plan(
                        env,
                        episode,
                        replay_runtime,
                        args_cli,
                        episode_index=source_episode_index,
                        garment_info=garment_info,
                        frame_hint=source_actions_frame_hint,
                    )

                    is_episode_annotated_successfully = False
                    object_pose_failure = None
                    for attempt_index in range(OBJECT_POSE_CAPTURE_MAX_ATTEMPTS):
                        session.reset_attempt_state()
                        try:
                            if args_cli.auto:
                                is_episode_annotated_successfully = annotate_episode_in_auto_mode(
                                    env,
                                    episode,
                                    replay_plan,
                                    replay_runtime,
                                    session,
                                    args_cli,
                                    task_type,
                                    success_term,
                                )
                            else:
                                is_episode_annotated_successfully = annotate_episode_in_manual_mode(
                                    env,
                                    episode,
                                    replay_plan,
                                    replay_runtime,
                                    session,
                                    args_cli,
                                    task_type,
                                    success_term,
                                    subtask_term_signal_names,
                                )
                            object_pose_failure = None
                            break
                        except DatagenObjectPoseCaptureError as exc:
                            object_pose_failure = exc
                            print(
                                "\tAnnotation object-pose capture failed on "
                                f"attempt {attempt_index + 1}/{OBJECT_POSE_CAPTURE_MAX_ATTEMPTS}: {exc}"
                            )
                            recover_from_object_pose_capture_failure(env, session)
                            if attempt_index + 1 < OBJECT_POSE_CAPTURE_MAX_ATTEMPTS:
                                print("\tReset complete. Replaying the episode again from the start.")

                    if object_pose_failure is not None:
                        raise RuntimeError(
                            "Aborting annotation after repeated invalid garment object-pose capture for "
                            f"{episode_name}: {object_pose_failure}"
                        ) from object_pose_failure

                    if is_episode_annotated_successfully and not session.state.skip_episode:
                        env.recorder_manager.set_success_to_episodes(
                            None,
                            torch.tensor([[True]], dtype=torch.bool, device=env.device),
                        )
                        env.recorder_manager.export_episodes()
                        exported_episode_count += 1
                        print("\tExported the annotated episode.")
                    else:
                        print("\tSkipped exporting the episode due to incomplete subtask annotations.")
                break

        print(
            f"\nExported {exported_episode_count} (out of {processed_episode_count}) annotated"
            f" episode{'s' if exported_episode_count > 1 else ''}."
        )

        if h5py is not None and os.path.exists(args_cli.output_file):
            try:
                with h5py.File(args_cli.output_file, "r+") as out_file:
                    data_group = out_file.get("data")
                    if data_group is not None:
                        data_group.attrs["actions_mode"] = "ee_pose" if bool(args_cli.require_ik_actions) else "joint"
            except Exception as exc:
                print(f"Warning: failed to write actions_mode attribute to annotated dataset: {exc}")

        print("Exiting the app.")
    finally:
        with contextlib.suppress(Exception):
            dataset_file_handler.close()
        if keyboard_interface is not None:
            with contextlib.suppress(Exception):
                keyboard_interface.reset()
        if env is not None:
            env.close()
