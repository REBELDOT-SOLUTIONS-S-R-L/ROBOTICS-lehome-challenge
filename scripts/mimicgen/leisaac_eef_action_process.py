"""Script to run EEF action processing for MimicGen recorded demos."""

import argparse
import json
import os
from collections.abc import Iterable
from copy import deepcopy

import h5py
import numpy as np
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="[Support Tool] EEF action processing for MimicGen recorded demos.")
parser.add_argument("--input_file", type=str, default="./datasets/mimic-lift-cube-example.hdf5", help="File path to load MimicGen recorded demos.")
parser.add_argument("--output_file", type=str, default="./datasets/processed_mimic-lift-cube-example.hdf5", help="File path to save processed MimicGen recorded demos.")
parser.add_argument("--to_ik", action="store_true", help="Whether to convert the action to ik action.")
parser.add_argument("--to_joint", action="store_true", help="Whether to convert the action to joint action.")
parser.add_argument("--gr00t_format", action="store_true", help="When converting to joint action, convert to GR00T format (degrees + sign flips for shoulder_lift and wrist_roll).")
parser.add_argument("--device", type=str, default="cpu", help="Device to use for tensor operations (e.g. 'cpu', 'cuda:0').")
args_cli = parser.parse_args()


# ---------------------------------------------------------------------------
# Minimal EpisodeData and HDF5DatasetFileHandler (no Isaac Sim dependency)
# ---------------------------------------------------------------------------

class EpisodeData:
    """Lightweight container for a single episode."""

    def __init__(self) -> None:
        self._data = {}
        self._seed = None
        self._env_id = None
        self._success = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: dict):
        self._data = data

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property
    def env_id(self):
        return self._env_id

    @env_id.setter
    def env_id(self, env_id):
        self._env_id = env_id

    @property
    def success(self):
        return self._success

    @success.setter
    def success(self, success):
        self._success = success

    def is_empty(self) -> bool:
        return not bool(self._data)


class HDF5DatasetFileHandler:
    """HDF5 dataset file handler (Isaac Sim-free)."""

    def __init__(self):
        self._hdf5_file_stream = None
        self._hdf5_data_group = None
        self._demo_count = 0
        self._env_args = {}

    def open(self, file_path: str, mode: str = "r"):
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        self._hdf5_file_stream = h5py.File(file_path, mode)
        self._hdf5_data_group = self._hdf5_file_stream["data"]
        self._demo_count = len(self._hdf5_data_group)

    def create(self, file_path: str, env_name: str = None):
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        self._hdf5_file_stream = h5py.File(file_path, "w")
        self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
        self._hdf5_data_group.attrs["total"] = 0
        self._demo_count = 0
        env_name = env_name if env_name is not None else ""
        self._add_env_args({"env_name": env_name, "type": 2})

    def __del__(self):
        self.close()

    def _add_env_args(self, env_args: dict):
        self._env_args.update(env_args)
        self._hdf5_data_group.attrs["env_args"] = json.dumps(self._env_args)

    def get_env_name(self) -> str | None:
        env_args = json.loads(self._hdf5_data_group.attrs.get("env_args", "{}"))
        return env_args.get("env_name")

    def get_episode_names(self) -> Iterable[str]:
        return self._hdf5_data_group.keys()

    def load_episode(self, episode_name: str, device: str) -> EpisodeData | None:
        if episode_name not in self._hdf5_data_group:
            return None
        episode = EpisodeData()
        h5_episode_group = self._hdf5_data_group[episode_name]

        def _load(group):
            data = {}
            for key in group:
                if isinstance(group[key], h5py.Group):
                    data[key] = _load(group[key])
                else:
                    data[key] = torch.tensor(np.array(group[key]), device=device)
            return data

        episode.data = _load(h5_episode_group)
        if "seed" in h5_episode_group.attrs:
            episode.seed = h5_episode_group.attrs["seed"]
        if "success" in h5_episode_group.attrs:
            episode.success = h5_episode_group.attrs["success"]
        episode.env_id = self.get_env_name()
        return episode

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        if episode.is_empty():
            return
        episode_group_name = f"demo_{demo_id if demo_id is not None else self._demo_count}"
        if episode_group_name in self._hdf5_data_group:
            raise ValueError(f"Episode group '{episode_group_name}' already exists")
        h5_episode_group = self._hdf5_data_group.create_group(episode_group_name)
        h5_episode_group.attrs["num_samples"] = len(episode.data["actions"]) if "actions" in episode.data else 0
        if episode.seed is not None:
            h5_episode_group.attrs["seed"] = episode.seed
        if episode.success is not None:
            h5_episode_group.attrs["success"] = episode.success

        def _write(group, key, value):
            if isinstance(value, dict):
                sub = group.create_group(key)
                for k, v in value.items():
                    _write(sub, k, v)
            else:
                arr = value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                group.create_dataset(key, data=arr, compression="gzip")

        for key, value in episode.data.items():
            _write(h5_episode_group, key, value)

        self._hdf5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]
        if demo_id is None:
            self._demo_count += 1

    def flush(self):
        if self._hdf5_file_stream is not None:
            self._hdf5_file_stream.flush()

    def close(self):
        if self._hdf5_file_stream is not None:
            self._hdf5_file_stream.close()
            self._hdf5_file_stream = None


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def convert_usd_to_motor_degrees(usd_degrees: np.ndarray) -> np.ndarray:
    """Convert USD joint degrees to motor degrees with SO101 sign conventions."""
    motor_degrees = usd_degrees.copy()
    if motor_degrees.ndim != 2:
        raise ValueError(
            f"Expected [T, D] joint array for GR00T conversion, got shape {tuple(motor_degrees.shape)}."
        )
    if motor_degrees.shape[1] not in (6, 12):
        raise ValueError(
            "GR00T conversion only supports single-arm [T,6] or bimanual [T,12] joint arrays, "
            f"got shape {tuple(motor_degrees.shape)}."
        )

    arm_offsets = (0,) if motor_degrees.shape[1] == 6 else (0, 6)
    for arm_offset in arm_offsets:
        motor_degrees[:, arm_offset + 1] = -usd_degrees[:, arm_offset + 1]
        motor_degrees[:, arm_offset + 4] = -usd_degrees[:, arm_offset + 4]
        motor_degrees[:, arm_offset + 5] = (usd_degrees[:, arm_offset + 5] + 10.0) / 110.0 * 100.0
        motor_degrees[:, arm_offset + 5] = np.clip(motor_degrees[:, arm_offset + 5], 0.0, 100.0)

    return motor_degrees


def _is_bimanual_obs(obs: dict) -> bool:
    return "left_ee_frame_state" in obs and "right_ee_frame_state" in obs


def _has_bimanual_joint_obs(obs: dict) -> bool:
    return (
        ("left_joint_pos_target" in obs and "right_joint_pos_target" in obs)
        or ("left_joint_pos" in obs and "right_joint_pos" in obs)
    )


def _get_bimanual_joint_tensor_from_obs(obs: dict) -> torch.Tensor | np.ndarray | None:
    if "left_joint_pos_target" in obs and "right_joint_pos_target" in obs:
        return torch.cat([obs["left_joint_pos_target"], obs["right_joint_pos_target"]], dim=-1)
    if "left_joint_pos" in obs and "right_joint_pos" in obs:
        return torch.cat([obs["left_joint_pos"], obs["right_joint_pos"]], dim=-1)
    if "actions" in obs:
        actions = obs["actions"]
        if actions.shape[-1] == 12:
            return actions
    return None


def _convert_joint_data_to_gr00t(joint_data):
    """Convert joint data from radians to GR00T format (degrees with sign flips)."""
    joint_data_np = (
        joint_data.cpu().numpy() if isinstance(joint_data, torch.Tensor) else np.asarray(joint_data)
    )
    original_shape = joint_data_np.shape
    if joint_data_np.ndim == 0:
        return joint_data
    if joint_data_np.ndim == 1:
        joint_data_np = joint_data_np.reshape(1, -1)

    if joint_data_np.shape[-1] not in (6, 12):
        return joint_data

    joint_data_usd_degrees = np.degrees(joint_data_np)
    joint_data_motor_degrees = convert_usd_to_motor_degrees(joint_data_usd_degrees)

    if len(original_shape) == 1:
        joint_data_motor_degrees = joint_data_motor_degrees.flatten()

    if isinstance(joint_data, torch.Tensor):
        return torch.from_numpy(joint_data_motor_degrees).to(joint_data.device).to(joint_data.dtype)
    return joint_data_motor_degrees


def joint_action_to_ik(episode_data: EpisodeData) -> EpisodeData:
    """Convert the action to ik action."""
    obs = episode_data.data["obs"]
    action = episode_data.data['actions']
    if _is_bimanual_obs(obs):
        left_eef_state = obs["left_ee_frame_state"]
        right_eef_state = obs["right_ee_frame_state"]
        left_gripper_action = action[:, 5:6]
        right_gripper_action = action[:, 11:12]
        new_actions = torch.cat(
            [left_eef_state, left_gripper_action, right_eef_state, right_gripper_action],
            dim=1,
        )
    else:
        eef_state = obs['ee_frame_state']
        gripper_action = action[:, -1:]
        new_actions = torch.cat([eef_state, gripper_action], dim=1)
    episode_data.data['actions'] = new_actions

    return episode_data


def ik_action_to_joint(episode_data: EpisodeData, gr00t_format: bool = False) -> EpisodeData:
    """
    Convert the action from IK format (EEF pose) to joint action.

    Args:
        episode_data: Episode data containing observations and actions
        gr00t_format: If True, convert to GR00T format (degrees + sign flips)
    """
    obs = episode_data.data.get('obs', {})

    actions = episode_data.data['actions']

    if actions.shape[-1] >= 7:
        is_bimanual_joint_stream = _is_bimanual_obs(obs) or _has_bimanual_joint_obs(obs)
        if is_bimanual_joint_stream:
            joint_pos = _get_bimanual_joint_tensor_from_obs(obs)
            if joint_pos is None:
                raise ValueError(
                    "Cannot convert bimanual IK actions to joint actions: "
                    "missing obs/actions or left/right joint targets/positions."
                )
        elif 'joint_pos_target' in obs:
            joint_pos = obs['joint_pos_target']
        elif 'joint_pos' in obs:
            joint_pos = obs['joint_pos']
        elif 'actions' in obs:
            joint_pos = obs['actions']
        else:
            raise ValueError(
                "Cannot convert IK actions to joint actions: "
                "No joint positions found in observations. "
                "Available observation keys: " + str(list(obs.keys()))
            )
    else:
        joint_pos = actions

    if gr00t_format:
        joint_pos_np = joint_pos.cpu().numpy() if isinstance(joint_pos, torch.Tensor) else np.asarray(joint_pos)

        if joint_pos_np.ndim == 1:
            joint_pos_np = joint_pos_np.reshape(1, -1)

        if len(joint_pos_np) > 0:
            print(f"Before conversion (radians): {joint_pos_np[0]}")
            print(f"Before conversion (degrees): {np.degrees(joint_pos_np[0])}")

        joint_pos_usd_degrees = np.degrees(joint_pos_np)
        joint_pos_motor_degrees = convert_usd_to_motor_degrees(joint_pos_usd_degrees)

        if len(joint_pos_motor_degrees) > 0:
            print(f"After conversion (Motor degrees): {joint_pos_motor_degrees[0]}")

        if isinstance(joint_pos, torch.Tensor):
            new_actions = torch.from_numpy(joint_pos_motor_degrees).to(joint_pos.device).to(joint_pos.dtype)
        else:
            new_actions = joint_pos_motor_degrees

        for obs_key in [
            'joint_pos',
            'joint_pos_target',
            'joint_vel',
            'left_joint_pos',
            'right_joint_pos',
            'left_joint_pos_target',
            'right_joint_pos_target',
            'left_joint_vel',
            'right_joint_vel',
        ]:
            if obs_key not in obs:
                continue
            obs_joint_data = obs[obs_key]
            obs_joint_data_np = obs_joint_data.cpu().numpy() if isinstance(obs_joint_data, torch.Tensor) else np.asarray(obs_joint_data)

            if obs_joint_data_np.ndim == 1:
                obs_joint_data_np = obs_joint_data_np.reshape(1, -1)

            if obs_key == 'joint_vel' and obs_joint_data_np.shape[-1] != 6:
                continue

            obs_joint_data_usd_degrees = np.degrees(obs_joint_data_np)
            obs_joint_data_motor_degrees = convert_usd_to_motor_degrees(obs_joint_data_usd_degrees)

            if isinstance(obs_joint_data, torch.Tensor):
                obs[obs_key] = torch.from_numpy(obs_joint_data_motor_degrees).to(obs_joint_data.device).to(obs_joint_data.dtype)
            else:
                obs[obs_key] = obs_joint_data_motor_degrees

        if 'initial_state' in episode_data.data:
            initial_state = episode_data.data['initial_state']
            if isinstance(initial_state, dict) and 'articulation' in initial_state:
                articulation = initial_state['articulation']
                if isinstance(articulation, dict):
                    for arm_name in ("robot", "left_arm", "right_arm"):
                        if arm_name in articulation:
                            arm_data = articulation[arm_name]
                            if isinstance(arm_data, dict) and 'joint_position' in arm_data:
                                print(f"DEBUG: Converting initial_state/articulation/{arm_name}/joint_position")
                                arm_data['joint_position'] = _convert_joint_data_to_gr00t(arm_data['joint_position'])

        if 'states' in episode_data.data:
            states = episode_data.data['states']
            if isinstance(states, dict) and 'articulation' in states:
                articulation = states['articulation']
                if isinstance(articulation, dict):
                    for arm_name in ("robot", "left_arm", "right_arm"):
                        if arm_name in articulation:
                            arm_data = articulation[arm_name]
                            if isinstance(arm_data, dict) and 'joint_position' in arm_data:
                                print(f"DEBUG: Converting states/articulation/{arm_name}/joint_position")
                                arm_data['joint_position'] = _convert_joint_data_to_gr00t(arm_data['joint_position'])

    else:
        new_actions = joint_pos

    if 'actions' in obs:
        obs['actions'] = new_actions.clone() if isinstance(new_actions, torch.Tensor) else new_actions.copy()
    episode_data.data['obs'] = obs

    episode_data.data['actions'] = new_actions
    if 'processed_actions' in episode_data.data:
        episode_data.data['processed_actions'] = (
            new_actions.clone() if isinstance(new_actions, torch.Tensor) else new_actions.copy()
        )

    return episode_data


def main():
    """Process the EEF action stream of MimicGen annotated recorded demos."""
    if args_cli.to_ik and args_cli.to_joint:
        raise ValueError("Cannot convert to both ik and joint action at the same time.")
    if not args_cli.to_ik and not args_cli.to_joint:
        raise ValueError("Must convert to either ik or joint action.")
    if args_cli.gr00t_format and not args_cli.to_joint:
        raise ValueError("--gr00t_format can only be used with --to_joint.")
    if os.path.abspath(args_cli.input_file) == os.path.abspath(args_cli.output_file):
        raise ValueError(
            "--input_file and --output_file must be different files. "
            "Create a new output HDF5 when converting actions."
        )

    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The dataset file {args_cli.input_file} does not exist.")
    input_dataset_handler = HDF5DatasetFileHandler()
    input_dataset_handler.open(args_cli.input_file)

    output_dataset_handler = HDF5DatasetFileHandler()
    output_dataset_handler.create(args_cli.output_file)

    episode_names = list(input_dataset_handler.get_episode_names())
    for episode_name in tqdm(episode_names):
        episode_data = input_dataset_handler.load_episode(episode_name, device=args_cli.device)
        if episode_data.success is not None and not episode_data.success:
            continue
        process_episode_data = deepcopy(episode_data)
        if args_cli.to_ik:
            process_episode_data = joint_action_to_ik(process_episode_data)
        elif args_cli.to_joint:
            process_episode_data = ik_action_to_joint(process_episode_data, gr00t_format=args_cli.gr00t_format)
        output_dataset_handler.write_episode(process_episode_data)

    input_data_group = input_dataset_handler._hdf5_data_group
    data_group = output_dataset_handler._hdf5_data_group
    if input_data_group is not None and data_group is not None:
        preserved_total = int(data_group.attrs.get("total", 0))
        for attr_name, attr_value in input_data_group.attrs.items():
            data_group.attrs[attr_name] = attr_value
        data_group.attrs["total"] = preserved_total
        data_group.attrs["num_episodes"] = len(list(data_group.keys()))
        data_group.attrs["actions_mode"] = "ee_pose" if args_cli.to_ik else "joint"
        if args_cli.to_ik:
            data_group.attrs["actions_frame"] = "base"
            data_group.attrs["ik_quat_order"] = "wxyz"
        else:
            if "actions_frame" in data_group.attrs:
                del data_group.attrs["actions_frame"]
            if "ik_quat_order" in data_group.attrs:
                del data_group.attrs["ik_quat_order"]

    output_dataset_handler.flush()
    input_dataset_handler.close()
    output_dataset_handler.close()


if __name__ == "__main__":
    main()