"""Script to run EEF action processing for MimicGen recorded demos."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="[Support Tool] EEF action processing for MimicGen recorded demos.")
parser.add_argument("--input_file", type=str, default="./datasets/mimic-lift-cube-example.hdf5", help="File path to load MimicGen recorded demos.")
parser.add_argument("--output_file", type=str, default="./datasets/processed_mimic-lift-cube-example.hdf5", help="File path to save processed MimicGen recorded demos.")
parser.add_argument("--to_ik", action="store_true", help="Whether to convert the action to ik action.")
parser.add_argument("--to_joint", action="store_true", help="Whether to convert the action to joint action.")
parser.add_argument("--gr00t_format", action="store_true", help="When converting to joint action, convert to GR00T format (degrees + sign flips for shoulder_lift and wrist_roll).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from isaaclab.utils.datasets import HDF5DatasetFileHandler, EpisodeData


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

    # Check if actions are in IK format (EEF pose: pos + quat + gripper = 7 values)
    # or already in joint format (6 values)
    actions = episode_data.data['actions']
    
    # Determine if actions are IK format (shape [T, 7] or [T, 8]) or joint format (shape [T, 6])
    if actions.shape[-1] >= 7:
        # IK format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, gripper] or [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        # Need to convert IK to joint positions
        # Try to get joint positions from observations if available
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
        # Already in joint format (shape [T, 6])
        joint_pos = actions

    if gr00t_format:
        # Convert from radians (USD) to degrees (Motor) with sign flips
        # joint_pos is in radians, shape [T, 6] where T is number of timesteps
        joint_pos_np = joint_pos.cpu().numpy() if isinstance(joint_pos, torch.Tensor) else np.asarray(joint_pos)
        
        # Ensure shape is [T, 6]
        if joint_pos_np.ndim == 1:
            joint_pos_np = joint_pos_np.reshape(1, -1)
        
        # Debug: print first frame before conversion
        if len(joint_pos_np) > 0:
            print(f"Before conversion (radians): {joint_pos_np[0]}")
            print(f"Before conversion (degrees): {np.degrees(joint_pos_np[0])}")
        
        # Convert radians to degrees (USD degrees)
        joint_pos_usd_degrees = np.degrees(joint_pos_np)
        
        # Convert USD degrees to Motor degrees (applies sign flips)
        joint_pos_motor_degrees = convert_usd_to_motor_degrees(joint_pos_usd_degrees)
        
        # Debug: print first frame after conversion
        if len(joint_pos_motor_degrees) > 0:
            print(f"After conversion (Motor degrees): {joint_pos_motor_degrees[0]}")
        
        # Convert back to tensor if needed
        if isinstance(joint_pos, torch.Tensor):
            new_actions = torch.from_numpy(joint_pos_motor_degrees).to(joint_pos.device).to(joint_pos.dtype)
        else:
            new_actions = joint_pos_motor_degrees
        
        # Also convert observations (joint_pos, joint_pos_target) to GR00T format
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
            if obs_key in obs:
                obs_joint_data = obs[obs_key]
                obs_joint_data_np = obs_joint_data.cpu().numpy() if isinstance(obs_joint_data, torch.Tensor) else np.asarray(obs_joint_data)
                
                # Ensure shape is [T, 6] or [T, 5] (for joint_vel which might not include gripper)
                if obs_joint_data_np.ndim == 1:
                    obs_joint_data_np = obs_joint_data_np.reshape(1, -1)
                
                # For joint_vel, only convert if it has 6 dimensions (includes gripper)
                # Otherwise, it might be velocities which don't need conversion
                if obs_key == 'joint_vel' and obs_joint_data_np.shape[-1] != 6:
                    # Skip conversion for velocities that don't match joint format
                    continue
                
                # Convert radians to degrees (USD degrees)
                obs_joint_data_usd_degrees = np.degrees(obs_joint_data_np)
                
                # Convert USD degrees to Motor degrees (applies sign flips)
                obs_joint_data_motor_degrees = convert_usd_to_motor_degrees(obs_joint_data_usd_degrees)
                
                # Convert back to tensor if needed
                if isinstance(obs_joint_data, torch.Tensor):
                    obs[obs_key] = torch.from_numpy(obs_joint_data_motor_degrees).to(obs_joint_data.device).to(obs_joint_data.dtype)
                else:
                    obs[obs_key] = obs_joint_data_motor_degrees
        
        # Convert specific known joint position paths (avoid converting root_velocity and other 6-element arrays)
        # Convert initial_state/articulation/robot/joint_position
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
        
        # Convert states/articulation/robot/joint_position
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
        # Standard conversion: keep in radians (USD format)
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
    # check arguments
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

    # Load dataset
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
    # run the main function
    main()
