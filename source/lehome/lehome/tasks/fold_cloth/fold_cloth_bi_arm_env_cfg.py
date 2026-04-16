"""Manager-based environment configuration for garment folding with dual SO101 arms.

Replaces all LeIsaac dependencies with lehome resources and IsaacLab core classes.
The GarmentObject (particle cloth) is managed manually by the env subclass,
not via the scene config, because it uses particle physics not supported by
IsaacLab's standard asset types.
"""
from __future__ import annotations

from typing import Any

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab.envs.mdp import time_out as mdp_time_out
from isaaclab.envs.mdp import reset_scene_to_default as mdp_reset_scene_to_default
from isaaclab.envs.mdp import joint_pos as mdp_joint_pos
from isaaclab.envs.mdp import last_action as mdp_last_action

from .mdp import garment_folded
from .mdp import robot_rest_pose as mdp_robot_rest_pose

from lehome.assets.robots.lerobot import (
    SO101_FOLLOWER_CFG,
    SO101_LEFT_ARM_HOME_JOINT_POS,
    SO101_RIGHT_ARM_HOME_JOINT_POS,
)
from lehome.assets.scenes.bedroom import MARBLE_BEDROOM_CFG
from lehome.devices.action_process import init_action_cfg, preprocess_device_action as _preprocess_device_action

##
# Scene definition
##


@configclass
class GarmentFoldSceneCfg(InteractiveSceneCfg):
    """Scene configuration for garment folding with dual SO101 arms.

    Contains the bedroom scene, two SO101 robots, three cameras, and lighting.
    The garment (particle cloth) is NOT part of the scene config — it is created
    manually by the GarmentFoldEnv subclass.
    """

    # Bedroom scene USD
    scene_usd: AssetBaseCfg = MARBLE_BEDROOM_CFG.replace(
        prim_path="/World/Scene",
    )

    # Left robot arm
    left_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Left_Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(-0.23, -0.25, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=dict(SO101_LEFT_ARM_HOME_JOINT_POS),
        ),
    )

    # Right robot arm
    right_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Right_Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(0.23, -0.25, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=dict(SO101_RIGHT_ARM_HOME_JOINT_POS),
        ),
    )

    # Left wrist camera
    left_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Left_Robot/gripper/left_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    # Right wrist camera
    right_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/gripper/right_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    # Top camera (mounted on right arm base)
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.245, -0.44, 0.56),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
    )

    # Dome light
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1200, color=(0.75, 0.75, 0.75)),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for bimanual control.

    Default (non-mimic): 12D joint-space actions
      [left_arm(5), left_gripper(1), right_arm(5), right_gripper(1)].
    Mimic mode can override these terms to native IK via ``use_teleop_device``.
    """

    left_arm_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="left_arm",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        use_default_offset=False,
    )
    left_gripper_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="left_arm",
        joint_names=["gripper"],
        use_default_offset=False,
    )
    right_arm_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="right_arm",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        use_default_offset=False,
    )
    right_gripper_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="right_arm",
        joint_names=["gripper"],
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications — joint positions, last action, and rest pose."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        left_joint_pos = ObsTerm(
            func=mdp_joint_pos,
            params={"asset_cfg": SceneEntityCfg("left_arm")},
        )
        right_joint_pos = ObsTerm(
            func=mdp_joint_pos,
            params={"asset_cfg": SceneEntityCfg("right_arm")},
        )
        actions = ObsTerm(func=mdp_last_action)
        robot_rest_pose = ObsTerm(func=mdp_robot_rest_pose)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event configuration — reset to default on episode reset."""

    reset_all = EventTerm(func=mdp_reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    """Termination terms — episode timeout only.

    Success-based termination is handled by the env subclass since it requires
    direct access to the GarmentObject particle system.
    """

    time_out = DoneTerm(func=mdp_time_out, time_out=True)
    success = DoneTerm(func=garment_folded, time_out=False)


@configclass
class RewardsCfg:
    """Reward terms — placeholder for future reward shaping.

    Dense reward computation requires direct access to GarmentObject particle
    positions, which is handled in the env subclass.
    """

    pass


##
# Environment configuration
##


@configclass
class GarmentFoldEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the manager-based garment folding environment."""

    # Scene settings
    scene: GarmentFoldSceneCfg = GarmentFoldSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    # Simulation
    render_cfg = sim_utils.RenderCfg(
        rendering_mode="quality", antialiasing_mode="FXAA"
    )
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 90,
        render_interval=1,
        render=render_cfg,
        use_fabric=False,
        device="cuda:0",
    )

    # Garment configuration
    garment_name: str = None
    garment_version: str = "Release"
    garment_cfg_base_path: str = "Assets/objects/Challenge_Garment"
    particle_cfg_path: str = (
        "source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml"
    )

    # Random seed
    use_random_seed: bool = True
    random_seed: int = 42

    # Task description
    task_description: str = "Fold the garment on the table."
    # Teleop / action contract name (updated by scripts)
    task_type: str = "bi-so101leader"
    # Generation-only override: keep 12D joint-action contract and use env IK solver.
    force_pinocchio_generation: bool = False
    # Online subtask-observation thresholds
    subtask_grasp_eef_to_keypoint_threshold_m: float = 0.15
    subtask_gripper_close_threshold: float = 0.20
    subtask_middle_to_lower_threshold_m: float = 0.10
    subtask_middle_to_lower_middle_keypoint_max_z_m: float = 0.60
    subtask_lower_to_upper_threshold_m: float = 0.15
    subtask_signal_min_consecutive_steps: int = 3
    return_home_min_consecutive_steps: int = 10

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 1
        self.episode_length_s = 60
        self.sim.dt = 1 / 90
        self.sim.render_interval = self.decimation

        self.viewer.eye = (0, -1.2, 1.3)
        self.viewer.lookat = (0, 6.4, -2.8)

    def use_teleop_device(self, teleop_device: str) -> None:
        """Switch action-term contract based on teleop/mimic device mode."""
        self.task_type = teleop_device
        self.actions = init_action_cfg(self.actions, device=teleop_device)

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        """Convert incoming teleop payload to env action tensor."""
        return _preprocess_device_action(action, teleop_device)
