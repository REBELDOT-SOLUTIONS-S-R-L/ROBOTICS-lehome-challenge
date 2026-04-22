"""MimicGen subtask decomposition for short pants.

Pants use only 4 of the 6 semantic keypoints for success:
``garment_{left,right}_{upper,lower}`` (waist corners and ankle corners).
``*_middle`` is unused — the success check ignores p[2] and p[3]
(see ``check_pant_short`` in ``utils/success_checker_chanllege.py``).

Fold strategy — single bimanual LATERAL fold, left edge onto right edge:

    left_upper ---- right_upper             left_upper \\
        |                |                    (moved) ----> right_upper
        |                |       ------->                   |
    left_lower ---- right_lower             left_lower \\
                                              (moved) ----> right_lower

Arm assignment (both arms move their keypoints left->right, in parallel):
    * left_arm  picks up  garment_left_upper  ->  drops near garment_right_upper
    * right_arm picks up  garment_left_lower  ->  drops near garment_right_lower

The right arm reaches across the pant width to grab ``garment_left_lower``;
this is the physically natural choice for same-direction bimanual folds
(both arms carry the left edge to the right edge in parallel).  Flip the
``object_ref`` / keypoint assignments below if the teleop convention ends
up doing right->left instead.

Subtask sequence (5 per arm, no re-grasp):
    0. prepare_for_grasp
    1. grasp
    2. carry_to_destination     (gripper closed, kp pair close)
    3. release_at_destination   (gripper open, kp pair close, kp z low)
    4. return_home
"""
from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import (
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintType,
)


def build(cfg):
    """Return (subtask_configs, task_constraint_configs) for pant_short."""
    del cfg

    # -----------------------------------------------------------------
    # Left arm: garment_left_upper -> garment_right_upper
    # -----------------------------------------------------------------
    left_subtask_configs = [
        # Subtask 0: descend into approach pose above garment_left_upper
        SubTaskConfig(
            object_ref="garment_left_upper",
            subtask_term_signal="prepare_for_grasp_left_upper",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                # Pants only have 4 meaningful corner keypoints; use all of
                # them for source-demo selection so the pick is driven by
                # the full garment layout.
                "keypoint_names": [
                    "garment_left_upper",
                    "garment_right_upper",
                    "garment_left_lower",
                    "garment_right_lower",
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
        # Subtask 1: grasp garment_left_upper
        SubTaskConfig(
            object_ref="garment_left_upper",
            subtask_term_signal="grasp_left_upper",
            subtask_term_offset_range=(5, 15),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "prepare_for_grasp_left_upper"},
            # Zero noise during a grasp.
            action_noise=0.0,
            num_interpolation_steps=5,
            # Hold the approach pose so the gripper fully closes before lift.
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm grasps left_upper",
            next_subtask_description="Left arm carries left_upper over right_upper",
        ),
        # Subtask 2: carry left_upper across to right_upper (still closed).
        # ``object_ref`` anchors the transformed trajectory to the
        # destination corner so the drop lands correctly under runtime
        # garment pose variation.
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
        # Subtask 3: release.  Gripper opens, carried corner settles onto
        # the destination corner.
        SubTaskConfig(
            object_ref="garment_right_upper",
            subtask_term_signal="release_left_upper_at_right_upper",
            subtask_term_offset_range=(5, 10),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "left_upper_to_right_upper"},
            action_noise=0.0,
            num_interpolation_steps=5,
            # Hold open long enough for the cloth to detach and settle.
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm opens gripper and releases left_upper at right_upper",
            next_subtask_description="Left arm returns home",
        ),
        # Subtask 4: return home.
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="left_return_home",
            # Last subtask runs to episode end; Mimic requires offsets = 0.
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
    # Right arm: garment_left_lower -> garment_right_lower (cross-reach)
    # -----------------------------------------------------------------
    right_subtask_configs = [
        # Subtask 0: descend into approach pose above garment_left_lower
        # (right arm reaches across the garment width).
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
            description="Right arm descends into approach pose above left_lower",
            next_subtask_description="Right arm grasps left_lower",
        ),
        # Subtask 1: grasp garment_left_lower with right arm
        SubTaskConfig(
            object_ref="garment_left_lower",
            subtask_term_signal="grasp_right_on_left_lower",
            subtask_term_offset_range=(5, 15),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={
                "source_subtask": "prepare_for_grasp_right_on_left_lower"
            },
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
            selection_strategy_kwargs={
                "source_subtask": "prepare_for_grasp_right_on_left_lower"
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Right arm carries left_lower over right_lower",
            next_subtask_description="Right arm releases left_lower at right_lower",
        ),
        # Subtask 3: release.
        SubTaskConfig(
            object_ref="garment_right_lower",
            subtask_term_signal="release_left_lower_at_right_lower",
            subtask_term_offset_range=(5, 10),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "left_lower_to_right_lower"},
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm opens gripper and releases left_lower at right_lower",
            next_subtask_description="Right arm returns home",
        ),
        # Subtask 4: return home.
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="right_return_home",
            subtask_term_offset_range=(0, 0),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={"source_subtask": "left_lower_to_right_lower"},
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
    # Subtask numbering: 0=prep, 1=grasp, 2=carry, 3=release, 4=return_home.
    # Keep arms loosely synchronized: both must finish the release before
    # either returns home.  No earlier sync is needed because the two
    # arms' pickups are on the SAME edge (left side) — parallel motion is
    # fine and additional sync would just reduce throughput.
    task_constraint_configs = [
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("left_arm", 3), ("right_arm", 4)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("right_arm", 3), ("left_arm", 4)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
    ]

    return (
        {"left_arm": left_subtask_configs, "right_arm": right_subtask_configs},
        task_constraint_configs,
    )
