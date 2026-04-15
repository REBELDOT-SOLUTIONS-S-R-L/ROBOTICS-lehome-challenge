"""MimicGen-compatible environment configuration for garment folding.

Defines subtask decomposition for bimanual garment folding with dual SO101 arms.
Uses garment cloth keypoints as virtual object references so MimicGen can
determine spatial relationships between the robot and grasp targets.

Subtask phases:
  1. Reach & grasp middle keypoints
  2. Bring middle keypoints to lower keypoints
  3. Re-grasp lower keypoints
  4. Bring lower keypoints to upper keypoints
  5. Return both arms to the home pose

Task success in this environment is measured by:
  - Garment fold success from 6 garment check_points and garment-specific
    distance thresholds (`success_checker_garment_fold`).
  - Both SO101 arms returning to rest pose (`garment_folded` termination).
"""
from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintType,
)
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTermCfg
from isaaclab.utils import configclass

from .fold_cloth_bi_arm_env_cfg import GarmentFoldEnvCfg
from .mdp.recorders import GarmentDatagenRecorder


# ---------------------------------------------------------------------------
# Recorder config: records garment keypoint datagen_info each step
# ---------------------------------------------------------------------------

@configclass
class GarmentDatagenRecorderCfg(RecorderTermCfg):
    """Configuration for the garment datagen recorder term."""
    class_type: type = GarmentDatagenRecorder


@configclass
class GarmentFoldRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder manager that captures datagen_info with garment keypoints."""
    garment_datagen = GarmentDatagenRecorderCfg()


# ---------------------------------------------------------------------------
# Mimic env config
# ---------------------------------------------------------------------------

@configclass
class GarmentFoldMimicEnvCfg(GarmentFoldEnvCfg, MimicEnvCfg):
    """MimicGen configuration for bimanual garment folding.

    Inherits from GarmentFoldEnvCfg (scene + managers) and MimicEnvCfg (datagen).
    Uses garment keypoints as virtual object references for MimicGen's
    source demo selection and trajectory transformation.

    Virtual objects (from garment check_points):
      - garment_left_middle, garment_right_middle
      - garment_left_lower, garment_right_lower
      - garment_left_upper, garment_right_upper
      - garment_upper_center, garment_lower_center
      - garment_kp_left, garment_kp_right, garment_center
    """

    def __post_init__(self):
        super().__post_init__()

        # -----------------------------------------------------------------
        # Recorder config
        # -----------------------------------------------------------------
        self.mimic_recorder_config = GarmentFoldRecorderManagerCfg()

        # -----------------------------------------------------------------
        # Data generation settings
        # -----------------------------------------------------------------
        self.datagen_config.name = "garment_fold_bi_so101_v0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        # Keep generation deterministic/stable by default.
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_select_src_per_arm = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        # Keep generated waypoints object-relative where supported by the env.
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 42

        # -----------------------------------------------------------------
        # Left arm subtasks
        # -----------------------------------------------------------------
        left_subtask_configs = []

        # Subtask 0: Reach & grasp left middle keypoint
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_left_middle",
                subtask_term_signal="grasp_left_middle",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_left_middle", "garment_left_upper"],
                    "nn_k": 1,
                },
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm reaches and grasps left middle keypoint",
                next_subtask_description="Bring left middle keypoint to left lower keypoint",
            )
        )

        # Subtask 1: Bring left middle keypoint to left lower keypoint
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_left_lower",
                subtask_term_signal="left_middle_to_lower",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_left_middle", "garment_left_lower"],
                    "nn_k": 1,
                },
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm brings left middle keypoint to left lower keypoint",
                next_subtask_description="Move left arm to waiting position",
            )
        )

        # Subtask 2: Move left arm to waiting position (home pose)
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal="left_at_waiting_pos",
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                action_noise=0.0,
                num_interpolation_steps=8,
                num_fixed_steps=10,
                apply_noise_during_interpolation=False,
                description="Left arm moves to waiting position",
                next_subtask_description="Re-grasp left lower keypoint",
            )
        )

        # Subtask 3: Re-grasp left lower keypoint
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_left_lower",
                subtask_term_signal="grasp_left_lower",
                subtask_term_offset_range=(3, 8),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_left_lower", "garment_left_upper"],
                    "nn_k": 1,
                },
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm re-grasps left lower keypoint",
                next_subtask_description="Bring left lower keypoint to left upper keypoint",
            )
        )

        # Subtask 4: Bring left lower keypoint to left upper keypoint
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_left_upper",
                subtask_term_signal="left_lower_to_upper",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_left_lower", "garment_left_upper"],
                    "nn_k": 1,
                },
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm brings left lower keypoint to left upper keypoint",
                next_subtask_description="Return left arm to home position",
            )
        )

        # Subtask 5: Return left arm to home position
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal="left_return_home",
                # Last subtask runs to episode end; keep offsets fixed.
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                action_noise=0.0,
                num_interpolation_steps=8,
                num_fixed_steps=10,
                apply_noise_during_interpolation=False,
                description="Left arm returns to home position",
            )
        )
        self.subtask_configs["left_arm"] = left_subtask_configs

        # -----------------------------------------------------------------
        # Right arm subtasks
        # -----------------------------------------------------------------
        right_subtask_configs = []

        # Subtask 0: Reach & grasp right middle keypoint
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_right_middle",
                subtask_term_signal="grasp_right_middle",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_right_middle", "garment_right_upper"],
                    "nn_k": 1,
                },
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm reaches and grasps right middle keypoint",
                next_subtask_description="Bring right middle keypoint to right lower keypoint",
            )
        )

        # Subtask 1: Bring right middle keypoint to right lower keypoint
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_right_lower",
                subtask_term_signal="right_middle_to_lower",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_right_middle", "garment_right_lower"],
                    "nn_k": 1,
                },
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm brings right middle keypoint to right lower keypoint",
                next_subtask_description="Move right arm to waiting position",
            )
        )

        # Subtask 2: Move right arm to waiting position (home pose)
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal="right_at_waiting_pos",
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                action_noise=0.0,
                num_interpolation_steps=8,
                num_fixed_steps=10,
                apply_noise_during_interpolation=False,
                description="Right arm moves to waiting position",
                next_subtask_description="Re-grasp right lower keypoint",
            )
        )

        # Subtask 3: Re-grasp right lower keypoint
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_right_lower",
                subtask_term_signal="grasp_right_lower",
                subtask_term_offset_range=(3, 8),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_right_lower", "garment_right_upper"],
                    "nn_k": 1,
                },
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm re-grasps right lower keypoint",
                next_subtask_description="Bring right lower keypoint to right upper keypoint",
            )
        )

        # Subtask 4: Bring right lower keypoint to right upper keypoint
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_right_upper",
                subtask_term_signal="right_lower_to_upper",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_right_lower", "garment_right_upper"],
                    "nn_k": 1,
                },
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm brings right lower keypoint to right upper keypoint",
                next_subtask_description="Return right arm to home position",
            )
        )

        # Subtask 5: Return right arm to home position
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal="right_return_home",
                # Last subtask runs to episode end; keep offsets fixed.
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                action_noise=0.0,
                num_interpolation_steps=8,
                num_fixed_steps=10,
                apply_noise_during_interpolation=False,
                description="Right arm returns to home position",
            )
        )
        self.subtask_configs["right_arm"] = right_subtask_configs

        # -----------------------------------------------------------------
        # Arm synchronization constraints
        # -----------------------------------------------------------------
        # Subtask numbering: 0=grasp_middle, 1=middle_to_lower,
        # 2=move_to_waiting_pos, 3=grasp_lower, 4=lower_to_upper,
        # 5=return_home.
        #
        # Sync point 1: Both arms finish waiting_pos before grasp_lower.
        # Sync point 2: Both arms finish grasp_lower before lower_to_upper.
        # Sync point 3: Both arms finish lower_to_upper before return_home.
        self.task_constraint_configs = [
            # Sync before subtask 3 (grasp_lower)
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 2), ("right_arm", 3)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
            ),
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("right_arm", 2), ("left_arm", 3)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
            ),
            # Sync before subtask 4 (lower_to_upper)
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 3), ("right_arm", 4)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
            ),
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("right_arm", 3), ("left_arm", 4)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
            ),
            # Sync before subtask 5 (return_home)
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 4), ("right_arm", 5)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
            ),
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("right_arm", 4), ("left_arm", 5)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
            ),
        ]
