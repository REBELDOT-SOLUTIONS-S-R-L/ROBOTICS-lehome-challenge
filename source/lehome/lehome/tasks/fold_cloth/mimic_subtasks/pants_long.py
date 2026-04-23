"""MimicGen subtask decomposition for long pants.

Long pants use an asymmetric mirrored sequence (right->left fold):

1. Left arm performs the upper right-to-left fold and retracts home.
2. Right arm performs the lower right-to-left fold.
3. Right arm re-grasps the stacked lower edge and performs the final
   lower-to-upper fold.

After the first fold, ``garment_right_upper`` is stacked on
``garment_left_upper`` and ``garment_right_lower`` is stacked on
``garment_left_lower``.  The right arm's second-phase re-grasp therefore
pulls the stacked lower layer upward (toward the stacked upper layer)
without needing a separate left-arm re-grasp sequence.

Arm assignment (both arms reach to the right edge of the pant):
    * left_arm  picks up  garment_right_upper  ->  garment_left_upper  ->  home
    * right_arm picks up  garment_right_lower  ->  garment_left_lower,
                 then     re-grasps stacked lowers -> stacked uppers -> home
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
    # Left arm subtasks: 5-step first fold (upper right -> upper left)
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
            next_subtask_description="Left arm returns home",
        ),
        # Subtask 4: return home.
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="left_return_home",
            subtask_term_offset_range=(0, 0),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "right_upper_to_left_upper"},
            action_noise=0.01,
            num_interpolation_steps=8,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm returns to home position",
        ),
    ]

    # -----------------------------------------------------------------
    # Right arm subtasks: lower right->left fold, then re-grasp and
    # fold stacked lowers up onto stacked uppers.
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
            selection_strategy_kwargs={"source_subtask": "prepare_for_grasp_right_lower"},
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
            selection_strategy_kwargs={"source_subtask": "prepare_for_grasp_right_lower"},
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
            next_subtask_description="Right arm prepares to re-grasp stacked lowers",
        ),
        # Subtask 4: descend into approach pose above the stacked lowers
        # (both garment_right_lower and garment_left_lower now sit at the
        # left_lower position; target keypoint is garment_left_lower).
        SubTaskConfig(
            object_ref="garment_left_lower",
            subtask_term_signal="prepare_for_grasp_right_on_left_lower",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": [
                    "garment_left_lower",
                    "garment_right_lower",
                    "garment_left_upper",
                    "garment_right_upper",
                ],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Right arm descends into approach pose above stacked lowers",
            next_subtask_description="Right arm grasps stacked lowers",
        ),
        # Subtask 5: grasp the stacked lowers (keypoint target: left_lower).
        SubTaskConfig(
            object_ref="garment_left_lower",
            subtask_term_signal="grasp_right_on_left_lower",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": [
                    "garment_left_lower",
                    "garment_right_lower",
                    "garment_left_upper",
                    "garment_right_upper",
                ],
                "nn_k": 1,
            },
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm grasps stacked lowers at left_lower",
            next_subtask_description="Right arm brings stacked lowers up to stacked uppers",
        ),
        # Subtask 6: bring stacked lowers up to the stacked uppers.
        # Signal checks distance between garment_right_lower (now at the
        # stacked-lower position) and garment_right_upper (now at the
        # stacked-upper position), with right arm gripper open.
        SubTaskConfig(
            object_ref="garment_right_upper",
            subtask_term_signal="right_lower_to_upper",
            subtask_term_offset_range=(10, 25),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": [
                    "garment_right_lower",
                    "garment_right_upper",
                    "garment_left_upper",
                ],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Right arm brings stacked lowers up to stacked uppers",
            next_subtask_description="Right arm returns home",
        ),
        # Subtask 7: return home.
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="right_return_home",
            subtask_term_offset_range=(0, 0),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "right_lower_to_upper"},
            action_noise=0.01,
            num_interpolation_steps=8,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm returns to home position",
        ),
    ]

    # -----------------------------------------------------------------
    # Arm synchronization
    # -----------------------------------------------------------------
    # Left subtask numbering: 0=prep, 1=grasp, 2=move, 3=release, 4=home.
    # Right subtask numbering:
    #   0=prep_first_fold, 1=grasp_first_fold, 2=move_first_fold,
    #   3=release_first_fold, 4=prep_lower_regrasp, 5=grasp_lower_regrasp,
    #   6=lower_to_upper, 7=return_home.
    #
    # Keep both arms synchronized through their first-fold releases, then
    # require the left arm to get home before the right arm begins the
    # second (lower-to-upper) fold.
    task_constraint_configs = [
        # Right arm finishes its first-fold release before the left arm
        # retracts home.
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("right_arm", 3), ("left_arm", 4)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
        # Left arm is home before the right arm starts its re-grasp, so
        # the left arm is out of the workspace for the second fold.
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("left_arm", 4), ("right_arm", 4)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
    ]

    return (
        {"left_arm": left_subtask_configs, "right_arm": right_subtask_configs},
        task_constraint_configs,
    )
