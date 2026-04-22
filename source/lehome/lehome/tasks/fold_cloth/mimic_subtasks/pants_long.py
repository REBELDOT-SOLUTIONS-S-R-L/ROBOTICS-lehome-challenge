"""MimicGen subtask decomposition for long pants.

Long pants use an asymmetric mirrored sequence:

1. Both arms perform the lateral right-to-left fold.
2. The right arm retracts home.
3. The left arm re-grasps the stacked lower edge and performs the final
   lower-to-upper fold.

The first lateral fold stacks ``garment_right_lower`` onto
``garment_left_lower``. The left arm's second-phase re-grasp therefore
pulls the stacked lower layer upward without needing a separate right-arm
re-grasp sequence.
"""
from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import (
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintType,
)


def build(cfg):
    """Return (subtask_configs, task_constraint_configs) for pants_long."""
    del cfg

    # -----------------------------------------------------------------
    # Left arm subtasks
    # -----------------------------------------------------------------
    left_subtask_configs = [
        # Subtask 0: descend into approach pose above garment_right_upper.
        SubTaskConfig(
            object_ref="garment_right_upper",
            subtask_term_signal="prepare_for_grasp_left_on_right_upper",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": [
                    "garment_right_upper",
                    "garment_left_upper",
                    "garment_right_lower",
                    "garment_left_lower",
                ],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Left arm descends into approach pose above right_upper",
            next_subtask_description="Left arm grasps right_upper",
        ),
        # Subtask 1: grasp garment_right_upper with the left arm.
        SubTaskConfig(
            object_ref="garment_right_upper",
            subtask_term_signal="grasp_left_on_right_upper",
            subtask_term_offset_range=(5, 15),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "prepare_for_grasp_left_on_right_upper"},
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm grasps right_upper",
            next_subtask_description="Left arm carries right_upper over left_upper",
        ),
        # Subtask 2: carry right_upper across to left_upper.
        SubTaskConfig(
            object_ref="garment_left_upper",
            subtask_term_signal="right_upper_to_left_upper",
            subtask_term_offset_range=(5, 10),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "prepare_for_grasp_left_on_right_upper"},
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Left arm carries right_upper over left_upper",
            next_subtask_description="Left arm releases right_upper at left_upper",
        ),
        # Subtask 3: release right_upper at left_upper.
        SubTaskConfig(
            object_ref="garment_left_upper",
            subtask_term_signal="release_right_upper_at_left_upper",
            subtask_term_offset_range=(5, 10),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "right_upper_to_left_upper"},
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm opens gripper and releases right_upper at left_upper",
            next_subtask_description="Left arm moves to waiting position",
        ),
        # Subtask 4: move to waiting position before the second fold.
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="left_at_waiting_pos",
            subtask_term_offset_range=(0, 0),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "right_upper_to_left_upper"},
            action_noise=0.01,
            num_interpolation_steps=8,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm moves to waiting/home position",
            next_subtask_description="Left arm prepares to grasp left_lower",
        ),
        # Subtask 5: descend into approach pose above garment_left_lower.
        SubTaskConfig(
            object_ref="garment_left_lower",
            subtask_term_signal="prepare_for_grasp_left_lower",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": [
                    "garment_left_lower",
                    "garment_left_upper",
                    "garment_right_lower",
                ],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Left arm descends into approach pose above left_lower",
            next_subtask_description="Left arm grasps left_lower",
        ),
        # Subtask 6: grasp garment_left_lower.
        SubTaskConfig(
            object_ref="garment_left_lower",
            subtask_term_signal="grasp_left_lower",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": [
                    "garment_left_lower",
                    "garment_left_upper",
                    "garment_right_lower",
                ],
                "nn_k": 1,
            },
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm grasps left_lower",
            next_subtask_description="Left arm brings left_lower to left_upper",
        ),
        # Subtask 7: bring left_lower to left_upper.
        SubTaskConfig(
            object_ref="garment_left_upper",
            subtask_term_signal="left_lower_to_upper",
            subtask_term_offset_range=(10, 25),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": [
                    "garment_left_lower",
                    "garment_left_upper",
                    "garment_right_upper",
                ],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Left arm brings left_lower to left_upper",
            next_subtask_description="Left arm returns home",
        ),
        # Subtask 8: return home.
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="left_return_home",
            subtask_term_offset_range=(0, 0),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "left_lower_to_upper"},
            action_noise=0.01,
            num_interpolation_steps=8,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm returns to home position",
        ),
    ]

    # -----------------------------------------------------------------
    # Right arm subtasks: 5-step first fold only
    # -----------------------------------------------------------------
    right_subtask_configs = [
        # Subtask 0: descend into approach pose above garment_right_lower.
        SubTaskConfig(
            object_ref="garment_right_lower",
            subtask_term_signal="prepare_for_grasp_right_lower",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": [
                    "garment_right_lower",
                    "garment_left_lower",
                    "garment_right_upper",
                    "garment_left_upper",
                ],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Right arm descends into approach pose above right_lower",
            next_subtask_description="Right arm grasps right_lower",
        ),
        # Subtask 1: grasp garment_right_lower.
        SubTaskConfig(
            object_ref="garment_right_lower",
            subtask_term_signal="grasp_right_lower",
            subtask_term_offset_range=(5, 15),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={
                "source_subtask": "prepare_for_grasp_right_lower",
            },
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm grasps right_lower",
            next_subtask_description="Right arm carries right_lower over left_lower",
        ),
        # Subtask 2: carry right_lower to left_lower.
        SubTaskConfig(
            object_ref="garment_left_lower",
            subtask_term_signal="right_lower_to_left_lower",
            subtask_term_offset_range=(5, 10),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={
                "source_subtask": "prepare_for_grasp_right_lower",
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Right arm carries right_lower over left_lower",
            next_subtask_description="Right arm releases right_lower at left_lower",
        ),
        # Subtask 3: release right_lower at left_lower.
        SubTaskConfig(
            object_ref="garment_left_lower",
            subtask_term_signal="release_right_lower_at_left_lower",
            subtask_term_offset_range=(5, 10),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "right_lower_to_left_lower"},
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm opens gripper and releases right_lower at left_lower",
            next_subtask_description="Right arm moves to home position",
        ),
        # Subtask 4: return to waiting/home position after the first fold.
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="right_at_waiting_pos",
            subtask_term_offset_range=(0, 0),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "right_lower_to_left_lower"},
            action_noise=0.01,
            num_interpolation_steps=8,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm moves to waiting/home position",
        ),
    ]

    # -----------------------------------------------------------------
    # Arm synchronization
    # -----------------------------------------------------------------
    # Left subtask numbering:
    #   0=prep_first_fold, 1=grasp_first_fold, 2=move_first_fold,
    #   3=release_first_fold, 4=waiting_pos, 5=prep_lower_regrasp,
    #   6=grasp_lower_regrasp, 7=lower_to_upper, 8=return_home.
    # Right subtask numbering: 0=prep, 1=grasp, 2=move, 3=release, 4=home.
    #
    # Keep both arms synchronized through the first release, then require
    # the right arm to get home before the left arm begins the second fold.
    task_constraint_configs = [
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("left_arm", 3), ("right_arm", 4)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("right_arm", 3), ("left_arm", 4)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("right_arm", 4), ("left_arm", 5)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
    ]

    return (
        {"left_arm": left_subtask_configs, "right_arm": right_subtask_configs},
        task_constraint_configs,
    )
