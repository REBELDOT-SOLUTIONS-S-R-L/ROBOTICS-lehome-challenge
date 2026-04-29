"""MimicGen subtask decomposition for long pants.

Long pants use an asymmetric mirrored sequence (left->right fold):

1. Left arm performs the upper left-to-right fold and retracts home.
2. Right arm performs the lower left-to-right fold.
3. Right arm re-grasps the stacked lower edge and performs the final
   lower-to-upper fold.

After the first fold, ``garment_left_upper`` is stacked on
``garment_right_upper`` and ``garment_left_lower`` is stacked on
``garment_right_lower``.  The right arm's second-phase re-grasp therefore
pulls the stacked lower layer upward (toward the stacked upper layer)
without needing a separate left-arm re-grasp sequence.

Arm assignment (both arms reach to the left edge of the pant):
    * left_arm  picks up  garment_left_upper  ->  garment_right_upper  ->  home
    * right_arm picks up  garment_left_lower  ->  garment_right_lower,
                 then     re-grasps stacked lowers -> stacked uppers -> home
"""
from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import SubTaskConfig


def build(cfg):
    """Return (subtask_configs, task_constraint_configs) for pants_long."""
    del cfg

    # -----------------------------------------------------------------
    # Left arm subtasks: 5-step first fold (upper left -> upper right)
    # -----------------------------------------------------------------
    left_subtask_configs = [
        # Subtask 0: descend into approach pose above garment_left_upper.
        SubTaskConfig(
            object_ref="garment_left_upper",
            subtask_term_signal="prepare_for_grasp_left_upper",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": [
                    "garment_left_upper",
                    "garment_left_lower",
                ],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Left arm descends into approach pose above left_upper",
            next_subtask_description="Left arm grasps left_upper",
        ),
        # Subtask 1: grasp garment_left_upper with the left arm.
        # Offset range kept tight because the annotated carry signal fires
        # only ~3 frames after grasp in this dataset; a wider range would
        # violate MimicGen's subtask sanity check.
        SubTaskConfig(
            object_ref="garment_left_upper",
            subtask_term_signal="grasp_left_upper",
            subtask_term_offset_range=(3, 5),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "prepare_for_grasp_left_upper"},
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm grasps left_upper",
            next_subtask_description="Left arm carries left_upper over right_upper",
        ),
        # Subtask 2: carry left_upper across to right_upper.
        SubTaskConfig(
            object_ref="garment_right_upper",
            subtask_term_signal="left_upper_to_right_upper",
            subtask_term_offset_range=(5, 10),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "prepare_for_grasp_left_upper"},
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Left arm carries left_upper over right_upper",
            next_subtask_description="Left arm releases left_upper at right_upper",
        ),
        # Subtask 3: release left_upper at right_upper.
        SubTaskConfig(
            object_ref="garment_right_upper",
            subtask_term_signal="release_left_upper_at_right_upper",
            subtask_term_offset_range=(5, 10),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "left_upper_to_right_upper"},
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm opens gripper and releases left_upper at right_upper",
            next_subtask_description="Left arm returns home",
        ),
        # Subtask 4: return home.
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="left_return_home",
            subtask_term_offset_range=(0, 0),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "left_upper_to_right_upper"},
            action_noise=0.01,
            num_interpolation_steps=8,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm returns to home position",
        ),
    ]

    # -----------------------------------------------------------------
    # Right arm subtasks: lower left->right fold, then re-grasp and
    # fold stacked lowers up onto stacked uppers.
    # -----------------------------------------------------------------
    right_subtask_configs = [
        # Subtask 0: descend into approach pose above garment_left_lower.
        SubTaskConfig(
            object_ref="garment_left_lower",
            subtask_term_signal="prepare_for_grasp_right_on_left_lower",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": [
                    "garment_left_lower",
                    "garment_left_upper",
                ],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Right arm descends into approach pose above left_lower",
            next_subtask_description="Right arm grasps left_lower",
        ),
        # Subtask 1: grasp garment_left_lower.
        # Tight offset (see note on left-arm grasp subtask) — carry signal
        # fires ~3 frames after grasp in the annotated teleop data.
        SubTaskConfig(
            object_ref="garment_left_lower",
            subtask_term_signal="grasp_right_on_left_lower",
            subtask_term_offset_range=(3, 5),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "prepare_for_grasp_right_on_left_lower"},
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm grasps left_lower",
            next_subtask_description="Right arm carries left_lower over right_lower",
        ),
        # Subtask 2: carry left_lower to right_lower.
        SubTaskConfig(
            object_ref="garment_right_lower",
            subtask_term_signal="left_lower_to_right_lower",
            subtask_term_offset_range=(5, 10),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "prepare_for_grasp_right_on_left_lower"},
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Right arm carries left_lower over right_lower",
            next_subtask_description="Right arm releases left_lower at right_lower",
        ),
        # Subtask 3: release left_lower at right_lower.
        SubTaskConfig(
            object_ref="garment_right_lower",
            subtask_term_signal="release_left_lower_at_right_lower",
            subtask_term_offset_range=(0, 5),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "left_lower_to_right_lower"},
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm opens gripper and releases left_lower at right_lower",
            next_subtask_description="Right arm prepares to re-grasp stacked lowers",
        ),
        # Subtask 4: descend into approach pose above the stacked lowers
        # (both garment_left_lower and garment_right_lower now sit at the
        # right_lower position; target keypoint is garment_right_lower).
        SubTaskConfig(
            object_ref="garment_right_lower",
            subtask_term_signal="prepare_for_grasp_right_lower",
            subtask_term_offset_range=(0, 5),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": [
                    "garment_right_lower",
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
        # Subtask 5: grasp the stacked lowers (keypoint target: right_lower).
        SubTaskConfig(
            object_ref="garment_right_lower",
            subtask_term_signal="grasp_right_lower",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": [
                    "garment_right_lower",
                    "garment_right_upper",
                ],
                "nn_k": 1,
            },
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm grasps stacked lowers at right_lower",
            next_subtask_description="Right arm brings stacked lowers up to stacked uppers",
        ),
        # Subtask 6: bring stacked lowers up to the stacked uppers.
        # Signal checks distance between garment_right_lower (now lifted
        # from the stacked-lower position) and garment_right_upper (at the
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
    # No cross-arm constraints: both arms run their own subtask queues
    # independently. Adding SEQUENTIAL constraints here caused one arm to
    # idle in mid-air while waiting for the other to latch a signal.
    task_constraint_configs = []

    return (
        {"left_arm": left_subtask_configs, "right_arm": right_subtask_configs},
        task_constraint_configs,
    )
