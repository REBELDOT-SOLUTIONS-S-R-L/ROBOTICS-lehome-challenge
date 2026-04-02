import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from lehome.utils.constant import ASSETS_ROOT


"""Configuration for the SO101 Follower Robot."""
SO101_FOLLOWER_ASSET_PATH = (
    Path(ASSETS_ROOT) / "robots" / "lerobot" / "so101_follower_eef.usd"
)
SO101_KINFE_ASSET_PATH = Path(ASSETS_ROOT) / "robots" / "lerobot" / "so101_knife.usd"

ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


SO101_FOLLOWER_HOME_JOINT_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": math.radians(-100.0),
    "elbow_flex": math.radians(85.0),
    "wrist_flex": math.radians(60.0),
    "wrist_roll": 0.0,
    "gripper": 0.0,
}

SO101_FOLLOWER_HOME_POSE_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 85.0,
    "wrist_flex": 60.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}

SO101_LEFT_ARM_HOME_JOINT_POS = dict(SO101_FOLLOWER_HOME_JOINT_POS)
SO101_LEFT_ARM_HOME_JOINT_POS["shoulder_pan"] = math.radians(-70.0)

SO101_RIGHT_ARM_HOME_JOINT_POS = dict(SO101_FOLLOWER_HOME_JOINT_POS)
SO101_RIGHT_ARM_HOME_JOINT_POS["shoulder_pan"] = math.radians(70.0)


def _build_rest_pose_range(home_pose_deg: dict[str, float]) -> dict[str, tuple[float, float]]:
    return {
        "shoulder_pan": (
            home_pose_deg["shoulder_pan"] - 20.0,
            home_pose_deg["shoulder_pan"] + 20.0,
        ),
        "shoulder_lift": (
            home_pose_deg["shoulder_lift"] - 20.0,
            home_pose_deg["shoulder_lift"] + 20.0,
        ),
        "elbow_flex": (
            home_pose_deg["elbow_flex"] - 20.0,
            home_pose_deg["elbow_flex"] + 20.0,
        ),
        "wrist_flex": (
            home_pose_deg["wrist_flex"] - 25.0,
            home_pose_deg["wrist_flex"] + 25.0,
        ),
        "wrist_roll": (
            home_pose_deg["wrist_roll"] - 20.0,
            home_pose_deg["wrist_roll"] + 20.0,
        ),
        "gripper": (
            home_pose_deg["gripper"] - 20.0,
            home_pose_deg["gripper"] + 20.0,
        ),
    }

SO101_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_FOLLOWER_ASSET_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=16,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.4, -2.3, 0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=SO101_FOLLOWER_HOME_JOINT_POS,
    ),
    actuators={
        "sts3215-gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "sts3215-arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

SO101_KINFE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_KINFE_ASSET_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.4, -2.3, 0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=SO101_FOLLOWER_HOME_JOINT_POS,
    ),
    actuators={
        "sts3215-gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "sts3215-arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
# joint limit written in USD (degree)
SO101_FOLLOWER_USD_JOINT_LIMLITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10, 100.0),
}

# motor limit written in real device (normalized to related range)
SO101_FOLLOWER_MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}


SO101_FOLLOWER_REST_POSE_RANGE = _build_rest_pose_range(SO101_FOLLOWER_HOME_POSE_DEG)
