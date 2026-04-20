"""MimicGen-compatible environment configuration for garment folding.

Defines subtask decomposition for bimanual garment folding with dual SO101 arms.
Uses garment cloth keypoints as virtual object references so MimicGen can
determine spatial relationships between the robot and grasp targets.

Subtask phases:
  1. Reach & grasp middle keypoints
  2. Bring middle keypoints over the release zone (gripper still closed)
  3. Open gripper and release middle keypoints inside the drop zone
  4. Move both arms to the waiting pose
  5. Re-grasp lower keypoints
  6. Bring lower keypoints to upper keypoints
  7. Return both arms to the home pose

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
        # Include the source demo's first measured EEF pose in the transformed
        # segment so interpolation goes to where the robot *actually was* at
        # subtask start, not just to the first target pose.  This smooths
        # cross-subtask transitions and reduces early-subtask IK spikes.
        self.datagen_config.generation_transform_first_robot_pose = True
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
                # Zero noise during a grasp: pose jitter at the jaw misses the ridge.
                action_noise=0.0,
                num_interpolation_steps=5,
                # Hold the approach pose for ~110 ms (10 steps @ 90 Hz) so the
                # gripper physically closes on the cloth before the arm lifts.
                num_fixed_steps=10,
                apply_noise_during_interpolation=False,
                description="Left arm reaches and grasps left middle keypoint",
                next_subtask_description="Bring left middle keypoint to left lower keypoint",
            )
        )

        # Subtask 1: Bring left middle keypoint over the release zone
        #   New termination semantics: gripper still closed AND left EEF's XY
        #   is inside the narrow drop zone derived from the 4 garment corner
        #   keypoints.  The actual cloth release is its own subtask (2).
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
                # Small noise widens the training distribution on transfer motions.
                action_noise=0.01,
                num_interpolation_steps=5,
                # Short hold at subtask entry so contacts settle after interp.
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                description="Left arm carries left middle keypoint over the release zone",
                next_subtask_description="Release left middle keypoint inside the drop zone",
            )
        )

        # Subtask 2: Release left middle keypoint inside the drop zone
        #   Termination: gripper OPEN + EEF still inside the narrow drop zone +
        #   tracked middle keypoint has fallen below the configured max Z.
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_left_lower",
                subtask_term_signal="release_left_middle",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_left_middle", "garment_left_lower"],
                    "nn_k": 1,
                },
                # Zero noise during the release: jitter at the moment the jaws
                # open can drag the cloth off the target lower corner.
                action_noise=0.0,
                num_interpolation_steps=5,
                # Hold open long enough for the cloth to detach and settle.
                num_fixed_steps=10,
                apply_noise_during_interpolation=False,
                description="Left arm opens gripper and releases middle keypoint in the drop zone",
                next_subtask_description="Move left arm to waiting position",
            )
        )

        # Subtask 3: Move left arm to waiting position (home pose)
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal="left_at_waiting_pos",
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                action_noise=0.01,
                num_interpolation_steps=8,
                num_fixed_steps=10,
                apply_noise_during_interpolation=False,
                description="Left arm moves to waiting position",
                next_subtask_description="Re-grasp left lower keypoint",
            )
        )

        # Subtask 4: Re-grasp left lower keypoint
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_left_lower",
                subtask_term_signal="grasp_left_lower",
                subtask_term_offset_range=(3, 8),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_left_lower", "garment_left_upper", "garment_right_lower"],
                    "nn_k": 1,
                },
                # Zero noise during a grasp.
                action_noise=0.0,
                num_interpolation_steps=5,
                # Hold the approach pose so the gripper fully closes before
                # the arm begins the fold motion.
                num_fixed_steps=10,
                apply_noise_during_interpolation=False,
                description="Left arm re-grasps left lower keypoint",
                next_subtask_description="Bring left lower keypoint to left upper keypoint",
            )
        )

        # Subtask 5: Bring left lower keypoint to left upper keypoint
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_left_upper",
                subtask_term_signal="left_lower_to_upper",
                # Give the cloth 110-280 ms to drape and settle after the
                # fold signal fires, before the arm retracts to home.
                subtask_term_offset_range=(10, 25),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_left_lower", "garment_left_upper", "garment_right_upper"],
                    "nn_k": 1,
                },
                action_noise=0.01,
                num_interpolation_steps=5,
                # Short hold at subtask entry.
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                description="Left arm brings left lower keypoint to left upper keypoint",
                next_subtask_description="Return left arm to home position",
            )
        )

        # Subtask 6: Return left arm to home position
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal="left_return_home",
                # Last subtask runs to episode end; Mimic requires offsets = 0.
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                action_noise=0.01,
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
                # Zero noise during a grasp: pose jitter at the jaw misses the ridge.
                action_noise=0.0,
                num_interpolation_steps=5,
                # Hold the approach pose for ~110 ms so the gripper closes
                # before the arm lifts.
                num_fixed_steps=10,
                apply_noise_during_interpolation=False,
                description="Right arm reaches and grasps right middle keypoint",
                next_subtask_description="Bring right middle keypoint to right lower keypoint",
            )
        )

        # Subtask 1: Bring right middle keypoint over the release zone
        #   See left-arm subtask 1 for the new termination semantics.
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
                action_noise=0.01,
                num_interpolation_steps=5,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                description="Right arm carries right middle keypoint over the release zone",
                next_subtask_description="Release right middle keypoint inside the drop zone",
            )
        )

        # Subtask 2: Release right middle keypoint inside the drop zone
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_right_lower",
                subtask_term_signal="release_right_middle",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_right_middle", "garment_right_lower"],
                    "nn_k": 1,
                },
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=10,
                apply_noise_during_interpolation=False,
                description="Right arm opens gripper and releases middle keypoint in the drop zone",
                next_subtask_description="Move right arm to waiting position",
            )
        )

        # Subtask 3: Move right arm to waiting position (home pose)
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal="right_at_waiting_pos",
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                action_noise=0.01,
                num_interpolation_steps=8,
                num_fixed_steps=10,
                apply_noise_during_interpolation=False,
                description="Right arm moves to waiting position",
                next_subtask_description="Re-grasp right lower keypoint",
            )
        )

        # Subtask 4: Re-grasp right lower keypoint
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_right_lower",
                subtask_term_signal="grasp_right_lower",
                subtask_term_offset_range=(3, 8),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_right_lower", "garment_right_upper", "garment_left_lower"],
                    "nn_k": 1,
                },
                # Zero noise during a grasp.
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=10,
                apply_noise_during_interpolation=False,
                description="Right arm re-grasps right lower keypoint",
                next_subtask_description="Bring right lower keypoint to right upper keypoint",
            )
        )

        # Subtask 5: Bring right lower keypoint to right upper keypoint
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="garment_right_upper",
                subtask_term_signal="right_lower_to_upper",
                # Give the cloth time to drape and settle after the fold
                # signal fires, before the arm retracts.
                subtask_term_offset_range=(10, 25),
                selection_strategy="nearest_neighbor_multi_keypoint",
                selection_strategy_kwargs={
                    "keypoint_names": ["garment_right_lower", "garment_right_upper", "garment_left_upper"],
                    "nn_k": 1,
                },
                action_noise=0.01,
                num_interpolation_steps=5,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                description="Right arm brings right lower keypoint to right upper keypoint",
                next_subtask_description="Return right arm to home position",
            )
        )

        # Subtask 6: Return right arm to home position
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal="right_return_home",
                # Last subtask runs to episode end; Mimic requires offsets = 0.
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                action_noise=0.01,
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
        # 2=release_middle, 3=move_to_waiting_pos, 4=grasp_lower,
        # 5=lower_to_upper, 6=return_home.
        #
        # Sync point 1: Both arms finish waiting_pos before grasp_lower.
        # Sync point 2: Both arms finish grasp_lower before lower_to_upper.
        # Sync point 3: Both arms finish lower_to_upper before return_home.
        self.task_constraint_configs = [
            # Sync before subtask 4 (grasp_lower)
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 3), ("right_arm", 4)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
            ),
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("right_arm", 3), ("left_arm", 4)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
            ),
            # Sync before subtask 5 (lower_to_upper)
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 4), ("right_arm", 5)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
            ),
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("right_arm", 4), ("left_arm", 5)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
            ),
            # Sync before subtask 6 (return_home)
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 5), ("right_arm", 6)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
            ),
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("right_arm", 5), ("left_arm", 6)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
            ),
        ]
