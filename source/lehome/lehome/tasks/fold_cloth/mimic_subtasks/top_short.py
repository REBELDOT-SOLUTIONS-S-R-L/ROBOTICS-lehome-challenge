"""MimicGen subtask decomposition for short-sleeve tops.

9-subtask bimanual fold with the drop zone in the UPPER half of the
garment (collar side) instead of the lower half (hem side):

    0. prepare_for_grasp (approach over the middle corners)
    1. grasp_middle
    2. middle_to_lower     (carry middle keypoint over the upper release zone)
    3. release_middle
    4. at_waiting_pos
    5. prepare_for_grasp   (approach over the lower corners — re-grasp)
    6. grasp_lower
    7. lower_to_upper
    8. return_home

The termination signals ``*_middle_to_lower`` and ``release_*_middle``
are reused by name.  Their bodies read the release-zone geometry from
the env cfg, so ``subtask_release_zone_upper_fraction`` (set in
:class:`GarmentFoldMimicEnvCfg` when garment_type is top-short-sleeve)
is what flips the drop location from hem to collar.
"""
from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import (
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintType,
)


def build(cfg):
    """Return (subtask_configs, task_constraint_configs) for top_short."""
    # ``left_middle_to_lower`` / ``release_*_middle`` read the release-zone
    # geometry from env cfg at runtime.  Set the top-half zone explicit here
    # so callers do not need to remember a separate cfg-side override.
    #
    # Short-sleeve tops have the middle keypoint near the sleeve tip, which
    # often sits at or past the ``garment_*_upper`` corners.  An ``upper_fraction``
    # of 1.0 makes the zone reach all the way from the garment center to the
    # upper corners (and the corner-derived span includes the sleeves), giving
    # the carry/release signals a forgiving target without straying outside
    # the garment footprint.
    _TOP_SHORT_UPPER_FRACTION = 1.0
    if hasattr(cfg, "subtask_release_zone_upper_fraction"):
        if getattr(cfg, "subtask_release_zone_upper_fraction", None) is None:
            cfg.subtask_release_zone_upper_fraction = _TOP_SHORT_UPPER_FRACTION

    # -----------------------------------------------------------------
    # Left arm subtasks
    # -----------------------------------------------------------------
    left_subtask_configs = []

    # Subtask 0: Descend into approach pose above left middle keypoint
    left_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_left_middle",
            subtask_term_signal="prepare_for_grasp_left_middle",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": ["garment_left_middle", "garment_left_upper", "garment_left_lower"],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Left arm descends into approach pose above left middle keypoint",
            next_subtask_description="Left arm reaches and grasps left middle keypoint",
        )
    )

    # Subtask 1: Reach & grasp left middle keypoint
    left_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_left_middle",
            subtask_term_signal="grasp_left_middle",
            subtask_term_offset_range=(5, 15),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": ["garment_left_middle", "garment_left_upper", "garment_left_lower"],
                "nn_k": 1,
            },
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm reaches and grasps left middle keypoint",
            next_subtask_description="Bring left middle keypoint to left upper keypoint",
        )
    )

    # Subtask 2: Bring left middle keypoint over the upper release zone
    #   object_ref + keypoint frame anchor on the UPPER corners so the
    #   delta-transform places the drop near the collar instead of the hem.
    left_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_left_upper",
            subtask_term_signal="left_middle_to_lower",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": ["garment_left_upper", "garment_right_upper", "garment_left_middle"],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Left arm carries left middle keypoint over the upper release zone",
            next_subtask_description="Release left middle keypoint inside the upper drop zone",
        )
    )

    # Subtask 3: Release left middle keypoint inside the upper drop zone
    left_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_left_upper",
            subtask_term_signal="release_left_middle",
            subtask_term_offset_range=(5, 10),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={
                "source_subtask": "left_middle_to_lower",
            },
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm opens gripper and releases middle keypoint in the upper drop zone",
            next_subtask_description="Move left arm to waiting position",
        )
    )

    # Subtask 4: Move left arm to waiting position (home pose)
    left_subtask_configs.append(
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="left_at_waiting_pos",
            subtask_term_offset_range=(0, 0),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={
                "source_subtask": "left_middle_to_lower",
            },
            action_noise=0.01,
            num_interpolation_steps=8,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm moves to waiting position",
            next_subtask_description="Descend into approach pose above left lower keypoint",
        )
    )

    # Subtask 5: Descend into approach pose above left lower keypoint
    left_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_left_lower",
            subtask_term_signal="prepare_for_grasp_left_lower",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": ["garment_left_lower", "garment_left_upper", "garment_right_lower"],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Left arm descends into approach pose above left lower keypoint",
            next_subtask_description="Left arm re-grasps left lower keypoint",
        )
    )

    # Subtask 6: Re-grasp left lower keypoint
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
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm re-grasps left lower keypoint",
            next_subtask_description="Bring left lower keypoint to left upper keypoint",
        )
    )

    # Subtask 7: Bring left lower keypoint to left upper keypoint
    left_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_left_upper",
            subtask_term_signal="left_lower_to_upper",
            subtask_term_offset_range=(10, 25),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": ["garment_left_lower", "garment_left_upper", "garment_right_upper"],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Left arm brings left lower keypoint to left upper keypoint",
            next_subtask_description="Return left arm to home position",
        )
    )

    # Subtask 8: Return left arm to home position
    left_subtask_configs.append(
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="left_return_home",
            subtask_term_offset_range=(0, 0),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={
                "source_subtask": "left_lower_to_upper",
            },
            action_noise=0.01,
            num_interpolation_steps=8,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Left arm returns to home position",
        )
    )

    # -----------------------------------------------------------------
    # Right arm subtasks
    # -----------------------------------------------------------------
    right_subtask_configs = []

    # Subtask 0: Descend into approach pose above right middle keypoint
    right_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_right_middle",
            subtask_term_signal="prepare_for_grasp_right_middle",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": ["garment_right_middle", "garment_right_upper", "garment_right_lower"],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Right arm descends into approach pose above right middle keypoint",
            next_subtask_description="Right arm reaches and grasps right middle keypoint",
        )
    )

    # Subtask 1: Reach & grasp right middle keypoint
    right_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_right_middle",
            subtask_term_signal="grasp_right_middle",
            subtask_term_offset_range=(5, 15),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": ["garment_right_middle", "garment_right_upper", "garment_right_lower"],
                "nn_k": 1,
            },
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm reaches and grasps right middle keypoint",
            next_subtask_description="Bring right middle keypoint to right upper keypoint",
        )
    )

    # Subtask 2: Bring right middle keypoint over the upper release zone
    right_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_right_upper",
            subtask_term_signal="right_middle_to_lower",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": ["garment_right_upper", "garment_left_upper", "garment_right_middle"],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Right arm carries right middle keypoint over the upper release zone",
            next_subtask_description="Release right middle keypoint inside the upper drop zone",
        )
    )

    # Subtask 3: Release right middle keypoint inside the upper drop zone
    right_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_right_upper",
            subtask_term_signal="release_right_middle",
            subtask_term_offset_range=(5, 10),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={
                "source_subtask": "right_middle_to_lower",
            },
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm opens gripper and releases middle keypoint in the upper drop zone",
            next_subtask_description="Move right arm to waiting position",
        )
    )

    # Subtask 4: Move right arm to waiting position (home pose)
    right_subtask_configs.append(
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="right_at_waiting_pos",
            subtask_term_offset_range=(0, 0),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={
                "source_subtask": "right_middle_to_lower",
            },
            action_noise=0.01,
            num_interpolation_steps=8,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm moves to waiting position",
            next_subtask_description="Descend into approach pose above right lower keypoint",
        )
    )

    # Subtask 5: Descend into approach pose above right lower keypoint
    right_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_right_lower",
            subtask_term_signal="prepare_for_grasp_right_lower",
            subtask_term_offset_range=(3, 8),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": ["garment_right_lower", "garment_right_upper", "garment_left_lower"],
                "nn_k": 1,
            },
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Right arm descends into approach pose above right lower keypoint",
            next_subtask_description="Right arm re-grasps right lower keypoint",
        )
    )

    # Subtask 6: Re-grasp right lower keypoint
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
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm re-grasps right lower keypoint",
            next_subtask_description="Bring right lower keypoint to right upper keypoint",
        )
    )

    # Subtask 7: Bring right lower keypoint to right upper keypoint
    right_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_right_upper",
            subtask_term_signal="right_lower_to_upper",
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

    # Subtask 8: Return right arm to home position
    right_subtask_configs.append(
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="right_return_home",
            subtask_term_offset_range=(0, 0),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={
                "source_subtask": "right_lower_to_upper",
            },
            action_noise=0.01,
            num_interpolation_steps=8,
            num_fixed_steps=10,
            apply_noise_during_interpolation=False,
            description="Right arm returns to home position",
        )
    )

    # -----------------------------------------------------------------
    # Arm synchronization constraints
    # -----------------------------------------------------------------
    # Subtask numbering: 0=prepare_for_grasp_middle, 1=grasp_middle,
    # 2=middle_to_lower, 3=release_middle, 4=move_to_waiting_pos,
    # 5=prepare_for_grasp_lower, 6=grasp_lower, 7=lower_to_upper,
    # 8=return_home.
    task_constraint_configs = [
        # Sync before subtask 5 (prepare_for_grasp_lower)
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("left_arm", 4), ("right_arm", 5)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("right_arm", 4), ("left_arm", 5)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
        # Sync before subtask 6 (grasp_lower): both arms hold at the
        # prepare_for_grasp pose until both have arrived, then close
        # their grippers together.
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("left_arm", 5), ("right_arm", 6)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("right_arm", 5), ("left_arm", 6)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
        # Sync before subtask 8 (return_home)
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("left_arm", 7), ("right_arm", 8)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("right_arm", 7), ("left_arm", 8)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
    ]

    return (
        {"left_arm": left_subtask_configs, "right_arm": right_subtask_configs},
        task_constraint_configs,
    )
