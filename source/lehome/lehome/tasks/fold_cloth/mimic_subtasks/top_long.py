"""MimicGen subtask decomposition for long-sleeve tops.

9-subtask bimanual fold with the drop zone in the LOWER half of the garment:

    0. prepare_for_grasp (approach over the middle corners)
    1. grasp_middle
    2. middle_to_lower     (carry middle keypoint over the lower release zone)
    3. release_middle
    4. at_waiting_pos
    5. prepare_for_grasp   (approach over the lower corners — re-grasp)
    6. grasp_lower
    7. lower_to_upper
    8. return_home

Subtasks are expressed as a pure Python builder so other garment types can
import and selectively override them (see ``top_short.py``).
"""
from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import (
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintType,
)


def build(cfg):
    """Return (subtask_configs, task_constraint_configs) for top_long.

    Args:
        cfg: the ``GarmentFoldMimicEnvCfg`` instance under construction.  Not
             currently read, but passed so future per-garment tweaks (e.g.
             noise magnitudes driven by cfg) can access it.

    Returns:
        (
            {"left_arm": [SubTaskConfig, ...], "right_arm": [SubTaskConfig, ...]},
            [SubTaskConstraintConfig, ...],
        )
    """
    del cfg  # unused for top_long

    # -----------------------------------------------------------------
    # Left arm subtasks
    # -----------------------------------------------------------------
    left_subtask_configs = []

    # Subtask 0: Descend into approach pose above left middle keypoint
    #   Termination: gripper still open, EEF z below the configured
    #   ``subtask_prep_for_grasp_eef_z_m`` cutoff, and EEF horizontally
    #   near the left middle keypoint.  With
    #   ``generation_select_src_per_subtask = False`` this is the only
    #   subtask that runs selection; every later subtask inherits the
    #   chosen demo, so we match against the full 6-keypoint garment
    #   layout instead of just the 3 keypoints near this subtask.
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
            # Small noise during the approach broadens the training
            # distribution without compromising the grasp itself.
            action_noise=0.01,
            num_interpolation_steps=5,
            num_fixed_steps=5,
            apply_noise_during_interpolation=False,
            description="Left arm descends into approach pose above left middle keypoint",
            next_subtask_description="Left arm reaches and grasps left middle keypoint",
        )
    )

    # Subtask 1: Reach & grasp left middle keypoint
    #   Reuses the source demo chosen for prepare_for_grasp_left_middle so
    #   the approach and the grasp come from the same human trajectory.
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

    # Subtask 2: Bring left middle keypoint over the release zone
    #   Termination: gripper still closed AND left EEF's XY is inside the
    #   narrow drop zone derived from the 4 garment corner keypoints.
    #   The actual cloth release is its own subtask (3).
    left_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_left_lower",
            subtask_term_signal="left_middle_to_lower",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": ["garment_left_lower", "garment_right_lower", "garment_left_middle"],
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

    # Subtask 3: Release left middle keypoint inside the drop zone
    #   Termination: gripper OPEN + EEF still inside the narrow drop zone +
    #   tracked middle keypoint has fallen below the configured max Z.
    #   Reuses the source demo chosen for left_middle_to_lower so carry
    #   and release come from the same human trajectory.
    left_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_left_lower",
            subtask_term_signal="release_left_middle",
            subtask_term_offset_range=(5, 10),
            selection_strategy="source_from_subtask",
            selection_strategy_kwargs={
                "source_subtask": "left_middle_to_lower",
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

    # Subtask 4: Move left arm to waiting position (home pose)
    #   Reuses the source demo chosen for left_middle_to_lower so the
    #   waiting-pos motion matches the same human trajectory.
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
    #   Mirror of subtask 0 but targeted at the re-grasp keypoint.
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

    # Subtask 7: Bring left lower keypoint to left upper keypoint
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

    # Subtask 8: Return left arm to home position
    #   Reuses the source demo chosen for left_lower_to_upper so the
    #   retract motion matches the preceding fold trajectory.
    left_subtask_configs.append(
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="left_return_home",
            # Last subtask runs to episode end; Mimic requires offsets = 0.
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
    #   With ``generation_select_src_per_subtask = False`` this is the
    #   only subtask that runs selection; later right-arm subtasks
    #   inherit the chosen demo, so we match on the full 6-keypoint
    #   garment layout.
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
    #   Reuses the source demo chosen for prepare_for_grasp_right_middle so
    #   the approach and the grasp come from the same human trajectory.
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

    # Subtask 2: Bring right middle keypoint over the release zone
    #   See left-arm subtask 2 for the termination semantics.
    right_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_right_lower",
            subtask_term_signal="right_middle_to_lower",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_multi_keypoint",
            selection_strategy_kwargs={
                "keypoint_names": ["garment_right_lower", "garment_left_lower", "garment_right_middle"],
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

    # Subtask 3: Release right middle keypoint inside the drop zone
    #   Reuses the source demo chosen for right_middle_to_lower so carry
    #   and release come from the same human trajectory.
    right_subtask_configs.append(
        SubTaskConfig(
            object_ref="garment_right_lower",
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
            description="Right arm opens gripper and releases middle keypoint in the drop zone",
            next_subtask_description="Move right arm to waiting position",
        )
    )

    # Subtask 4: Move right arm to waiting position (home pose)
    #   Reuses the source demo chosen for right_middle_to_lower so the
    #   waiting-pos motion matches the same human trajectory.
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
            # Zero noise during a grasp.
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

    # Subtask 8: Return right arm to home position
    #   Reuses the source demo chosen for right_lower_to_upper so the
    #   retract motion matches the preceding fold trajectory.
    right_subtask_configs.append(
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal="right_return_home",
            # Last subtask runs to episode end; Mimic requires offsets = 0.
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
    #
    # Sync point 1: Both arms finish waiting_pos before the prep_for_grasp_lower
    # approach, so the two arms re-engage the garment together.
    # Sync point 2: Both arms finish grasp_lower before lower_to_upper.
    # Sync point 3: Both arms finish lower_to_upper before return_home.
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
        # Sync before subtask 7 (lower_to_upper)
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("left_arm", 6), ("right_arm", 7)],
            constraint_type=SubTaskConstraintType.SEQUENTIAL,
        ),
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("right_arm", 6), ("left_arm", 7)],
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
