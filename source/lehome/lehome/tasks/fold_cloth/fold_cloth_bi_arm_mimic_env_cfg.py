"""MimicGen-compatible environment configuration for garment folding.

Scene, managers, action contract, and datagen settings are shared across
all garment types.  The subtask decomposition itself lives per-garment in
:mod:`mimic_subtasks`; this class dispatches to the right builder based on
``garment_name`` -> ``ChallengeGarmentLoader.get_garment_type(...)``.

Virtual object references (from garment check_points) used by the builders:
    - garment_left_middle, garment_right_middle
    - garment_left_lower, garment_right_lower
    - garment_left_upper, garment_right_upper
    - garment_upper_center, garment_lower_center
    - garment_kp_left, garment_kp_right, garment_center

Task success in this environment is measured by:
    - Garment fold success from 6 garment check_points and garment-specific
      distance thresholds (``success_checker_garment_fold``).
    - Both SO101 arms returning to rest pose (``garment_folded`` termination).
"""
from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTermCfg
from isaaclab.utils import configclass

from lehome.tasks.bedroom.challenge_garment_loader import ChallengeGarmentLoader

from .fold_cloth_bi_arm_env_cfg import GarmentFoldEnvCfg
from .mdp.recorders import GarmentDatagenRecorder
from .mimic_subtasks import BUILDERS


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


def configure_subtasks_from_garment_cfg(cfg) -> str:
    """Populate garment-specific subtask config on any fold-cloth env cfg.

    This helper is shared by the Mimic env config and annotated teleop,
    which uses the base runtime env but still needs garment-specific
    annotation queues.

    Returns:
        Canonical garment type string resolved from ``cfg.garment_name``.
    """
    garment_name = getattr(cfg, "garment_name", None)
    if not garment_name:
        raise ValueError(
            "configure_subtasks_from_garment_cfg() requires 'garment_name' "
            "to be set on the cfg before env construction."
        )

    garment_cfg_base_path = getattr(cfg, "garment_cfg_base_path", "Assets/objects/Challenge_Garment")
    garment_type = ChallengeGarmentLoader(garment_cfg_base_path).get_garment_type(garment_name)
    try:
        builder = BUILDERS[garment_type]
    except KeyError as exc:
        raise KeyError(
            f"No mimic subtask builder registered for garment_type "
            f"'{garment_type}'. Known types: {sorted(BUILDERS.keys())}."
        ) from exc

    subtask_configs, task_constraint_configs = builder(cfg)

    # The Mimic cfg initializes ``subtask_configs`` itself, but the base
    # runtime cfg used by annotated teleop does not. Support both.
    existing_subtask_configs = getattr(cfg, "subtask_configs", None)
    if not isinstance(existing_subtask_configs, dict):
        existing_subtask_configs = {}
        setattr(cfg, "subtask_configs", existing_subtask_configs)
    existing_subtask_configs.update(subtask_configs)
    setattr(cfg, "task_constraint_configs", task_constraint_configs)

    if (
        garment_type == "top-short-sleeve"
        and getattr(cfg, "subtask_release_zone_upper_fraction", None) is None
    ):
        setattr(
            cfg,
            "subtask_release_zone_upper_fraction",
            getattr(cfg, "subtask_release_zone_lower_fraction"),
        )

    return str(garment_type)


# ---------------------------------------------------------------------------
# Mimic env config
# ---------------------------------------------------------------------------

@configclass
class GarmentFoldMimicEnvCfg(GarmentFoldEnvCfg, MimicEnvCfg):
    """MimicGen configuration for bimanual garment folding.

    Inherits from GarmentFoldEnvCfg (scene + managers) and MimicEnvCfg (datagen).
    Subtask decomposition is garment-type-specific and looked up from
    :data:`mimic_subtasks.BUILDERS` at ``__post_init__`` time.
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
        # Select a single source demo per arm on the first subtask and reuse
        # it for the whole episode.  Subtask 0 uses
        # ``nearest_neighbor_all_keypoints`` so the pick is driven by the full
        # garment layout (all six garment check_points), not the three
        # keypoints relevant to one subtask.
        self.datagen_config.generation_select_src_per_subtask = False
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

        # ``garment_name`` is populated by the calling script *after*
        # ``parse_env_cfg`` returns (see ``annotated_record_service.py``),
        # so we can't dispatch the subtask config here.  The env calls
        # :meth:`configure_subtasks_from_garment` during ``__init__`` once
        # the cfg is fully populated.

    def configure_subtasks_from_garment(self) -> None:
        """Populate ``subtask_configs`` / ``task_constraint_configs`` from ``garment_name``.

        Invoked by :class:`GarmentFoldMimicEnv` right before the base env
        ``__init__`` runs.  Uses :class:`ChallengeGarmentLoader` to resolve
        the canonical garment-type string, then looks up the matching
        builder in :data:`mimic_subtasks.BUILDERS`.

        For short-sleeve tops this also flips the release-zone geometry to
        the upper half of the garment (``subtask_release_zone_upper_fraction``),
        since the subtask-signal observation functions read the zone from
        cfg rather than from any per-subtask override.
        """
        configure_subtasks_from_garment_cfg(self)
