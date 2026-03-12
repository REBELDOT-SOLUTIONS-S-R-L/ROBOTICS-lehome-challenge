"""MimicGen-compatible environment configuration for garment folding.

Defines subtask decomposition for bimanual garment folding with dual SO101 arms.
Uses garment cloth keypoints as virtual object references so MimicGen can
determine spatial relationships between the robot and grasp targets.

Subtask phases:
  1. Reach & grasp sleeves (both arms, coordinated)
  2. Bring sleeves to bottom corners
  3. Re-grasp bottom corners
  4. Bring bottom corners to top corners (both arms, coordinated)

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
    SubTaskConstraintCoordinationScheme,
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
      - garment_left_sleeve, garment_right_sleeve
      - garment_left_bottom, garment_right_bottom
      - garment_left_top, garment_right_top
      - garment_top_center, garment_bottom_center
      - garment_kp_left, garment_kp_right, garment_center (backward compatible)
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
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
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

        # Subtask 0: Reach & grasp left sleeve
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_left_sleeve",
                subtask_term_signal="grasp_left_sleeve",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm reaches and grasps left sleeve",
                next_subtask_description="Bring left sleeve to left bottom corner",
            )
        )

        # Subtask 1: Bring left sleeve to left bottom corner
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_left_bottom",
                subtask_term_signal="left_sleeve_to_bottom",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm brings left sleeve to left bottom corner",
                next_subtask_description="Re-grasp left bottom corner",
            )
        )

        # Subtask 2: Re-grasp left bottom corner
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_left_bottom",
                subtask_term_signal="grasp_left_bottom",
                subtask_term_offset_range=(3, 8),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm re-grasps left bottom corner",
                next_subtask_description="Bring left bottom corner to left top corner",
            )
        )

        # Subtask 3: Bring left bottom corner to left top corner
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_left_top",
                subtask_term_signal="left_bottom_to_top",
                # MimicGen requires last subtask offset start to be 0.
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm brings left bottom corner to left top corner",
            )
        )
        self.subtask_configs["left_arm"] = left_subtask_configs

        # -----------------------------------------------------------------
        # Right arm subtasks
        # -----------------------------------------------------------------
        right_subtask_configs = []

        # Subtask 0: Reach & grasp right sleeve
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_right_sleeve",
                subtask_term_signal="grasp_right_sleeve",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm reaches and grasps right sleeve",
                next_subtask_description="Bring right sleeve to right bottom corner",
            )
        )

        # Subtask 1: Bring right sleeve to right bottom corner
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_right_bottom",
                subtask_term_signal="right_sleeve_to_bottom",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm brings right sleeve to right bottom corner",
                next_subtask_description="Re-grasp right bottom corner",
            )
        )

        # Subtask 2: Re-grasp right bottom corner
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_right_bottom",
                subtask_term_signal="grasp_right_bottom",
                subtask_term_offset_range=(3, 8),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm re-grasps right bottom corner",
                next_subtask_description="Bring right bottom corner to right top corner",
            )
        )

        # Subtask 3: Bring right bottom corner to right top corner
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_right_top",
                subtask_term_signal="right_bottom_to_top",
                # MimicGen requires last subtask offset start to be 0.
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm brings right bottom corner to right top corner",
            )
        )
        self.subtask_configs["right_arm"] = right_subtask_configs

        # -----------------------------------------------------------------
        # Bimanual coordination constraints
        # -----------------------------------------------------------------
        self.task_constraint_configs = [
            # Grasp phase: both arms coordinate to grasp simultaneously
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 0), ("right_arm", 0)],
                constraint_type=SubTaskConstraintType.COORDINATION,
                # Use object-relative transform so grasp adapts to current sleeve pose
                # instead of replaying source-space targets.
                coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
                coordination_synchronize_start=True,
            ),
            # Sleeve-to-bottom phase
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 1), ("right_arm", 1)],
                constraint_type=SubTaskConstraintType.COORDINATION,
                coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
                coordination_synchronize_start=True,
            ),
            # Bottom re-grasp phase
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 2), ("right_arm", 2)],
                constraint_type=SubTaskConstraintType.COORDINATION,
                coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
                coordination_synchronize_start=True,
            ),
            # Bottom-to-top fold phase
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 3), ("right_arm", 3)],
                constraint_type=SubTaskConstraintType.COORDINATION,
                coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
                coordination_synchronize_start=True,
            ),
        ]
