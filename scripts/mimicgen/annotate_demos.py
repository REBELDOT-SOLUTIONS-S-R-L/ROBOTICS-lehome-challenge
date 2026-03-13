# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to add mimic annotations to demos to be used as source demos for mimic dataset generation.
"""

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# Launching Isaac Sim Simulator first.


# add argparse arguments
parser = argparse.ArgumentParser(description="Annotate demonstrations for Isaac Lab environments.")
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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
from copy import deepcopy

import gymnasium as gym
import isaaclab_mimic.envs  # noqa: F401
import torch
try:
    import h5py
except ImportError:
    h5py = None

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

# Only enables inputs if this script is NOT headless mode
if not args_cli.headless and not os.environ.get("HEADLESS", 0):
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import lehome.tasks  # noqa: F401

from lehome.utils.env_utils import (
    get_task_type,
    dynamic_reset_gripper_effort_limit_sim,
)

is_paused = False
current_action_index = 0
marked_subtask_action_indices = []
skip_episode = False
task_type = None


def stabilize_garment_after_reset_for_annotation(env: ManagerBasedRLMimicEnv, num_steps: int = 20):
    """Run a short physics-only settle phase without adding recorder entries."""
    if num_steps <= 0:
        return

    try:
        left_joint_pos = env.scene["left_arm"].data.joint_pos
        right_joint_pos = env.scene["right_arm"].data.joint_pos
        hold_action = torch.cat([left_joint_pos, right_joint_pos], dim=-1).to(env.device)
    except Exception:
        return

    try:
        env.action_manager.process_action(hold_action)
    except Exception:
        # If action manager cannot process, continue with pure sim stepping.
        pass

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    render_interval = max(1, int(getattr(env.cfg.sim, "render_interval", 1)))
    for step_index in range(num_steps):
        try:
            env.action_manager.apply_action()
        except Exception:
            pass

        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if is_rendering and (step_index + 1) % render_interval == 0:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)


def _rebuild_initial_state_from_episode(episode: EpisodeData) -> dict:
    """Build a robust initial_state, filling arm joints from obs if needed."""
    initial_state = deepcopy(episode.data.get("initial_state", {}))

    if "articulation" not in initial_state:
        initial_state["articulation"] = {}

    obs = episode.data.get("obs", {})
    actions = episode.data.get("actions", None)
    left_obs = obs.get("left_joint_pos", None)
    right_obs = obs.get("right_joint_pos", None)
    left_action_fallback = None
    right_action_fallback = None
    if actions is not None and getattr(actions, "ndim", 0) >= 2 and actions.shape[0] > 0:
        if actions.shape[-1] >= 6:
            left_action_fallback = actions[0:1, :6].clone()
        if actions.shape[-1] >= 12:
            right_action_fallback = actions[0:1, 6:12].clone()

    def _select_initial_joint_row(obs_tensor, fallback_tensor):
        if obs_tensor is None or getattr(obs_tensor, "ndim", 0) < 2 or obs_tensor.shape[0] == 0:
            return fallback_tensor
        norms = torch.sum(torch.abs(obs_tensor), dim=-1)
        nz = torch.nonzero(norms > 1e-6, as_tuple=False)
        if nz.numel() > 0:
            idx = int(nz[0, 0].item())
            return obs_tensor[idx:idx + 1].clone()
        return obs_tensor[0:1].clone() if fallback_tensor is None else fallback_tensor

    def _ensure_arm_state(arm_name: str, obs_tensor, fallback_tensor):
        obs_first = _select_initial_joint_row(obs_tensor, fallback_tensor)
        if obs_first is None:
            return

        arm_state = initial_state["articulation"].get(arm_name, {})
        if not isinstance(arm_state, dict):
            arm_state = {}

        jp = arm_state.get("joint_position", None)
        needs_fill = (
            jp is None
            or (hasattr(jp, "numel") and jp.numel() == 0)
            or (hasattr(jp, "shape") and jp.shape[-1] != obs_first.shape[-1])
            or (hasattr(jp, "abs") and float(jp.abs().sum().item()) < 1e-6)
        )
        if needs_fill:
            arm_state["joint_position"] = obs_first

        if "joint_velocity" not in arm_state or arm_state["joint_velocity"] is None:
            arm_state["joint_velocity"] = torch.zeros_like(obs_first)
        if "root_pose" not in arm_state or arm_state["root_pose"] is None:
            root_pose = torch.zeros((1, 7), dtype=obs_first.dtype, device=obs_first.device)
            root_pose[:, 3] = 1.0
            arm_state["root_pose"] = root_pose
        if "root_velocity" not in arm_state or arm_state["root_velocity"] is None:
            arm_state["root_velocity"] = torch.zeros((1, 6), dtype=obs_first.dtype, device=obs_first.device)

        initial_state["articulation"][arm_name] = arm_state

    _ensure_arm_state("left_arm", left_obs, left_action_fallback)
    _ensure_arm_state("right_arm", right_obs, right_action_fallback)
    return initial_state


def _set_arm_joint_state_from_initial_state(env: ManagerBasedRLMimicEnv, initial_state: dict):
    """Apply arm joint states from initial_state directly to sim."""
    articulation_state = initial_state.get("articulation", {})
    for arm_name in ("left_arm", "right_arm"):
        try:
            arm = env.scene[arm_name]
        except Exception:
            continue
        arm_state = articulation_state.get(arm_name, {})
        if not isinstance(arm_state, dict):
            continue

        joint_pos = arm_state.get("joint_position", None)
        if joint_pos is not None and getattr(joint_pos, "ndim", 0) == 2 and joint_pos.shape[0] > 0:
            arm.write_joint_position_to_sim(joint_pos[:1], env_ids=None)

        joint_vel = arm_state.get("joint_velocity", None)
        if joint_vel is not None and getattr(joint_vel, "ndim", 0) == 2 and joint_vel.shape[0] > 0:
            arm.write_joint_velocity_to_sim(joint_vel[:1], env_ids=None)


def _as_2d_tensor(data) -> torch.Tensor | None:
    """Convert array-like input to a 2D tensor, or return None if unavailable."""
    if data is None:
        return None
    tensor = data if torch.is_tensor(data) else torch.tensor(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        return None
    return tensor


def _get_replay_actions_for_episode(episode: EpisodeData, expected_action_dim: int | None = None) -> torch.Tensor:
    """Choose the best action trajectory for replay, matching env action dimension when possible."""
    obs = episode.data.get("obs", {})
    actions = _as_2d_tensor(episode.data.get("actions", None))
    processed_actions = _as_2d_tensor(episode.data.get("processed_actions", None))
    obs_actions = _as_2d_tensor(obs.get("actions", None))

    left_obs = _as_2d_tensor(obs.get("left_joint_pos", None))
    right_obs = _as_2d_tensor(obs.get("right_joint_pos", None))
    concat_joint_actions = None
    if left_obs is not None and right_obs is not None:
        if (
            left_obs.shape[0] == right_obs.shape[0]
            and left_obs.shape[1] >= 6
            and right_obs.shape[1] >= 6
        ):
            concat_joint_actions = torch.cat([left_obs[:, :6], right_obs[:, :6]], dim=-1)

    if actions is None:
        raise ValueError("Episode does not contain 'actions' for replay.")

    # Prefer source actions when dimensions already match the environment.
    if expected_action_dim is None or actions.shape[1] == expected_action_dim:
        replay_actions = actions
        source_name = "actions"
    else:
        # Priority for mismatched source actions:
        # 1) obs/actions (same horizon, joint-space in converted datasets)
        # 2) processed_actions
        # 3) left/right_joint_pos concat
        replay_actions = None
        source_name = None
        for name, candidate in (
            ("obs/actions", obs_actions),
            ("processed_actions", processed_actions),
            ("obs/left+right_joint_pos", concat_joint_actions),
        ):
            if candidate is not None and candidate.shape[1] == expected_action_dim:
                replay_actions = candidate
                source_name = name
                break

        if replay_actions is None:
            available_shapes = []
            for name, candidate in (
                ("actions", actions),
                ("obs/actions", obs_actions),
                ("processed_actions", processed_actions),
                ("obs/left+right_joint_pos", concat_joint_actions),
            ):
                if candidate is not None:
                    available_shapes.append(f"{name}={tuple(candidate.shape)}")
            raise ValueError(
                f"Could not find replay actions with expected dimension {expected_action_dim}. "
                f"Available: {', '.join(available_shapes)}"
            )

        print(
            f"\tReplay fallback: using {source_name} with shape {tuple(replay_actions.shape)} "
            f"instead of actions with shape {tuple(actions.shape)}."
        )

    actions_std = float(actions.float().std().item()) if actions.numel() > 0 else 0.0
    replay_std = float(replay_actions.float().std().item()) if replay_actions.numel() > 0 else 0.0
    if actions_std < 1e-3 and replay_std > 1e-2 and source_name != "actions":
        print(
            f"\tReplay note: source actions are near-constant (std={actions_std:.6f}), "
            f"selected replay stream std={replay_std:.6f}."
        )

    return replay_actions


def _load_garment_info(garment_info_path: str | None) -> dict | None:
    """Load garment_info.json for initial pose replay."""
    if garment_info_path is None:
        return None
    path = Path(garment_info_path)
    if not path.exists():
        print(f"Warning: garment_info_json not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"Warning: failed to parse garment_info_json ({path}): {e}")
    return None


def _load_garment_info_from_hdf5(input_file: str) -> dict | None:
    """Load merged garment_info from /data/demo_*/meta/(garment_info|garment_info.json)."""
    if h5py is None:
        return None

    def _normalize_scalar(value):
        if hasattr(value, "shape") and getattr(value, "shape", ()) == ():
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value

    def _merge(dst: dict, src: dict):
        for garment_name, episodes in src.items():
            if not isinstance(episodes, dict):
                continue
            dst.setdefault(garment_name, {})
            for episode_idx, payload in episodes.items():
                dst[garment_name][str(episode_idx)] = payload

    merged: dict = {}
    try:
        with h5py.File(input_file, "r") as file:
            data_group = file.get("data", None)
            if data_group is None:
                return None

            for demo_name in sorted(data_group.keys()):
                if not demo_name.startswith("demo_"):
                    continue
                demo_group = data_group[demo_name]
                if "meta" not in demo_group:
                    continue
                meta_group = demo_group["meta"]

                for key in ("garment_info", "garment_info.json"):
                    if key not in meta_group:
                        continue
                    raw = _normalize_scalar(meta_group[key][()])
                    if isinstance(raw, str):
                        try:
                            parsed = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                    elif isinstance(raw, dict):
                        parsed = raw
                    else:
                        continue

                    if isinstance(parsed, dict):
                        _merge(merged, parsed)
    except Exception as e:
        print(f"Warning: failed to read garment info from HDF5 metadata: {e}")
        return None

    return merged if merged else None


def _get_episode_initial_pose(garment_info: dict | None, episode_index: int) -> dict | None:
    """Extract {"Garment": [...]} pose for the given episode index."""
    if not garment_info:
        return None
    episode_key = str(episode_index)
    for _, episodes in garment_info.items():
        if isinstance(episodes, dict) and episode_key in episodes:
            pose = episodes[episode_key].get("object_initial_pose")
            if pose is not None:
                return {"Garment": pose}
    return None


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def skip_episode_cb():
    global skip_episode
    skip_episode = True


def mark_subtask_cb():
    global current_action_index, marked_subtask_action_indices
    marked_subtask_action_indices.append(current_action_index)
    print(f"Marked a subtask signal at action index: {current_action_index}")


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records the datagen info data in each step."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
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


def main():
    """Add Isaac Lab Mimic annotations to the given demo dataset file."""
    global is_paused, current_action_index, marked_subtask_action_indices, task_type

    # Load input dataset to be annotated
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    dataset_env_args = {}
    try:
        raw_env_args = dataset_file_handler._hdf5_data_group.attrs.get("env_args", "{}")
        if isinstance(raw_env_args, bytes):
            raw_env_args = raw_env_args.decode("utf-8")
        dataset_env_args = json.loads(raw_env_args) if isinstance(raw_env_args, str) else {}
    except Exception as e:
        print(f"Warning: failed to parse dataset env_args: {e}")

    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    garment_info = _load_garment_info_from_hdf5(args_cli.input_file)
    if garment_info is not None:
        print("Using garment initial poses from HDF5 demo metadata (/data/demo_*/meta).")
    else:
        garment_info_path = args_cli.garment_info_json
        if garment_info_path is None:
            candidate = Path(args_cli.input_file).parent / "meta" / "garment_info.json"
            if candidate.exists():
                garment_info_path = str(candidate)
        garment_info = _load_garment_info(garment_info_path)
        if garment_info is not None:
            print(f"Using garment initial poses from: {garment_info_path}")

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    # get output directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task, args_cli.task_type)
    setattr(env_cfg, "task_type", task_type)

    env_cfg.env_name = env_name

    # Configure garment metadata for tasks that require explicit garment loading.
    if hasattr(env_cfg, "garment_name"):
        resolved_garment_name = (
            args_cli.garment_name
            or dataset_env_args.get("garment_name")
            or getattr(env_cfg, "garment_name", None)
        )
        resolved_garment_version = (
            args_cli.garment_version
            or dataset_env_args.get("garment_version")
            or getattr(env_cfg, "garment_version", None)
        )

        if resolved_garment_name is None or (
            isinstance(resolved_garment_name, str) and not resolved_garment_name.strip()
        ):
            raise ValueError(
                "This task requires a garment_name, but none was provided. "
                "Pass --garment_name (e.g., Top_Long_Unseen_0) or include garment_name in data/env_args."
            )

        env_cfg.garment_name = (
            resolved_garment_name.strip()
            if isinstance(resolved_garment_name, str)
            else resolved_garment_name
        )
        if hasattr(env_cfg, "garment_version") and resolved_garment_version is not None:
            env_cfg.garment_version = resolved_garment_version

        print(
            f"Using garment config: name={env_cfg.garment_name}, "
            f"version={getattr(env_cfg, 'garment_version', 'N/A')}"
        )

    # extract success checking function to invoke manually
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    # Disable all termination terms
    env_cfg.terminations = None

    # Set up recorder terms for mimic annotations
    env_cfg.recorders: MimicRecorderManagerCfg = MimicRecorderManagerCfg()
    # ActionStateRecorder's flat policy observation term writes to key "obs" as a tensor.
    # Mimic annotations write nested keys under "obs/datagen_info/...".
    # Keeping both causes type conflicts in EpisodeData (list vs dict under "obs").
    if hasattr(env_cfg.recorders, "record_pre_step_flat_policy_observations"):
        env_cfg.recorders.record_pre_step_flat_policy_observations = None
    if not args_cli.auto:
        # disable subtask term signals recorder term if in manual mode
        env_cfg.recorders.record_pre_step_subtask_term_signals = None

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # create environment from loaded config
    env: ManagerBasedRLMimicEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if args_cli.auto:
        # check if the mimic API env.get_subtask_term_signals() is implemented
        if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
            raise NotImplementedError(
                "The environment does not implement the get_subtask_term_signals method required "
                "to run automatic annotations."
            )
    else:
        # get subtask termination signal names for each eef from the environment configs
        subtask_term_signal_names = {}
        for eef_name, eef_subtask_configs in env.cfg.subtask_configs.items():
            subtask_term_signal_names[eef_name] = [
                subtask_config.subtask_term_signal for subtask_config in eef_subtask_configs
            ]
            # no need to annotate the last subtask term signal, so remove it from the list
            subtask_term_signal_names[eef_name].pop()

    # reset environment
    env.reset()
    if hasattr(env, "initialize_obs"):
        try:
            env.initialize_obs()
            stabilize_garment_after_reset_for_annotation(env)
        except Exception as e:
            print(f"Warning: initial garment initialization/stabilization failed: {e}")

    # Only enables inputs if this script is NOT headless mode
    if not args_cli.headless and not os.environ.get("HEADLESS", 0):
        try:
            keyboard_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        except TypeError:
            # Backward compatibility for older keyboard API variants.
            keyboard_interface = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1)
        keyboard_interface.add_callback("N", play_cb)
        keyboard_interface.add_callback("B", pause_cb)
        keyboard_interface.add_callback("Q", skip_episode_cb)
        if not args_cli.auto:
            keyboard_interface.add_callback("S", mark_subtask_cb)
        keyboard_interface.reset()

    # simulate environment -- run everything in inference mode
    exported_episode_count = 0
    processed_episode_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            # Iterate over the episodes in the loaded dataset file
            for episode_index, episode_name in enumerate(dataset_file_handler.get_episode_names()):
                processed_episode_count += 1
                print(f"\nAnnotating episode #{episode_index} ({episode_name})")
                episode = dataset_file_handler.load_episode(episode_name, env.device)

                is_episode_annotated_successfully = False
                if args_cli.auto:
                    is_episode_annotated_successfully = annotate_episode_in_auto_mode(
                        env, episode, success_term, episode_index, garment_info
                    )
                else:
                    is_episode_annotated_successfully = annotate_episode_in_manual_mode(
                        env, episode, success_term, subtask_term_signal_names, episode_index, garment_info
                    )

                if is_episode_annotated_successfully and not skip_episode:
                    # set success to the recorded episode data and export to file
                    env.recorder_manager.set_success_to_episodes(
                        None, torch.tensor([[True]], dtype=torch.bool, device=env.device)
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
    print("Exiting the app.")

    # Close environment after annotation is complete
    env.close()


def replay_episode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    episode_index: int | None = None,
    garment_info: dict | None = None,
) -> bool:
    """Replays an episode in the environment.

    This function replays the given recorded episode in the environment. It can optionally check if the task
    was successfully completed using a success termination condition input.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.

    Returns:
        True if the episode was successfully replayed and the success condition was met (if provided),
        False otherwise.
    """
    global current_action_index, skip_episode, is_paused, task_type
    # read initial state and actions from the loaded episode
    initial_state = _rebuild_initial_state_from_episode(episode)
    expected_action_dim = None
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "total_action_dim"):
        expected_action_dim = int(env.action_manager.total_action_dim)
    elif hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
        shape = getattr(env.action_space, "shape", ())
        if len(shape) > 0:
            expected_action_dim = int(shape[0])
    actions = _get_replay_actions_for_episode(episode, expected_action_dim=expected_action_dim)
    env.seed(int(episode.seed))
    # Hard simulator resets can destabilize particle-cloth scenes.
    # For garment environments, prefer environment-level reset only.
    is_cloth_env = hasattr(env, "object")
    if not is_cloth_env:
        env.sim.reset()
        try:
            env.sim.play()
        except Exception:
            pass
    env.recorder_manager.reset()
    env.reset()
    try:
        env.sim.play()
    except Exception:
        pass
    if hasattr(env, "initialize_obs"):
        try:
            env.initialize_obs()
        except Exception as e:
            print(f"Warning: initialize_obs failed during replay reset: {e}")
    _set_arm_joint_state_from_initial_state(env, initial_state)
    if episode_index is not None and hasattr(env, "set_all_pose"):
        initial_pose = _get_episode_initial_pose(garment_info, episode_index)
        if initial_pose is not None:
            try:
                env.set_all_pose(initial_pose)
            except Exception as e:
                print(f"Warning: failed to set initial garment pose for episode {episode_index}: {e}")
    env.scene.write_data_to_sim()
    env.sim.forward()
    try:
        env.sim.play()
    except Exception:
        pass
    stabilize_garment_after_reset_for_annotation(env)
    first_action = True
    step_period = (1.0 / args_cli.step_hz) if args_cli.step_hz and args_cli.step_hz > 0 else 0.0
    for action_index, action in enumerate(actions):
        step_start_time = time.perf_counter()
        current_action_index = action_index
        if first_action:
            first_action = False
        else:
            while is_paused or skip_episode:
                env.sim.render()
                if skip_episode:
                    return False
                continue
        try:
            if env.sim.is_stopped():
                env.sim.play()
        except Exception:
            pass
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=env.device).reshape(1, -1)
        if getattr(env.cfg, "dynamic_reset_gripper_effort_limit", False):
            dynamic_reset_gripper_effort_limit_sim(env, task_type)
        env.step(action_tensor)
        # Force viewport updates in GUI mode so replay is visually observable.
        if env.sim.has_gui() or env.sim.has_rtx_sensors():
            env.sim.render()
        if step_period > 0.0:
            elapsed = time.perf_counter() - step_start_time
            if elapsed < step_period:
                time.sleep(step_period - elapsed)
    if success_term is not None:
        if not bool(success_term.func(env, **success_term.params)[0]):
            return False
    return True


def annotate_episode_in_auto_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    episode_index: int | None = None,
    garment_info: dict | None = None,
) -> bool:
    """Annotates an episode in automatic mode.

    This function replays the given episode in the environment and checks if the task was successfully completed.
    If the task was not completed, it will print a message and return False. Otherwise, it will check if all the
    subtask term signals are annotated and return True if they are, False otherwise.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.

    Returns:
        True if the episode was successfully annotated, False otherwise.
    """
    global skip_episode
    skip_episode = False
    is_episode_annotated_successfully = replay_episode(
        env, episode, success_term, episode_index=episode_index, garment_info=garment_info
    )
    if skip_episode:
        print("\tSkipping the episode.")
        return False
    if not is_episode_annotated_successfully:
        print("\tThe final task was not completed.")
    else:
        # check if all the subtask term signals are annotated
        annotated_episode = env.recorder_manager.get_episode(0)
        subtask_term_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_term_signals"]
        for signal_name, signal_flags in subtask_term_signal_dict.items():
            if not torch.any(signal_flags):
                is_episode_annotated_successfully = False
                print(f'\tDid not detect completion for the subtask "{signal_name}".')
    return is_episode_annotated_successfully


def annotate_episode_in_manual_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    subtask_term_signal_names: dict[str, list[str]] = {},
    episode_index: int | None = None,
    garment_info: dict | None = None,
) -> bool:
    """Annotates an episode in manual mode.

    This function replays the given episode in the environment and allows for manual marking of subtask term signals.
    It iterates over each eef and prompts the user to mark the subtask term signals for that eef.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.
        subtask_term_signal_names: Dictionary mapping eef names to lists of subtask term signal names.

    Returns:
        True if the episode was successfully annotated, False otherwise.
    """
    global is_paused, marked_subtask_action_indices, skip_episode
    # iterate over the eefs for marking subtask term signals
    subtask_term_signal_action_indices = {}
    for eef_name, eef_subtask_term_signal_names in subtask_term_signal_names.items():
        # skip if no subtask annotation is needed for this eef
        if len(eef_subtask_term_signal_names) == 0:
            continue

        while True:
            is_paused = False
            skip_episode = False
            print(f'\tPlaying the episode for subtask annotations for eef "{eef_name}".')
            print("\tSubtask signals to annotate:")
            print(f"\t\t- Termination:\t{eef_subtask_term_signal_names}")

            print('\n\tReplay starts immediately.')
            print('\tPress "B" to pause.')
            print('\tPress "N" to resume.')
            print('\tPress "S" to annotate subtask signals.')
            print('\tPress "Q" to skip the episode.\n')
            marked_subtask_action_indices = []
            task_success_result = replay_episode(
                env, episode, success_term, episode_index=episode_index, garment_info=garment_info
            )
            if skip_episode:
                print("\tSkipping the episode.")
                return False

            print(f"\tSubtasks marked at action indices: {marked_subtask_action_indices}")
            expected_subtask_signal_count = len(eef_subtask_term_signal_names)
            if task_success_result and expected_subtask_signal_count == len(marked_subtask_action_indices):
                print(f'\tAll {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were annotated.')
                for marked_signal_index in range(expected_subtask_signal_count):
                    # collect subtask term signal action indices
                    subtask_term_signal_action_indices[eef_subtask_term_signal_names[marked_signal_index]] = (
                        marked_subtask_action_indices[marked_signal_index]
                    )
                break

            if not task_success_result:
                print("\tThe final task was not completed.")
                return False

            if expected_subtask_signal_count != len(marked_subtask_action_indices):
                print(
                    f"\tOnly {len(marked_subtask_action_indices)} out of"
                    f' {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were'
                    " annotated."
                )

            print(f'\tThe episode will be replayed again for re-marking subtask signals for the eef "{eef_name}".\n')

    annotated_episode = env.recorder_manager.get_episode(0)
    for (
        subtask_term_signal_name,
        subtask_term_signal_action_index,
    ) in subtask_term_signal_action_indices.items():
        # subtask termination signal is false until subtask is complete, and true afterwards
        subtask_signals = torch.ones(len(episode.data["actions"]), dtype=torch.bool)
        subtask_signals[:subtask_term_signal_action_index] = False
        annotated_episode.add(f"obs/datagen_info/subtask_term_signals/{subtask_term_signal_name}", subtask_signals)
    return True


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
