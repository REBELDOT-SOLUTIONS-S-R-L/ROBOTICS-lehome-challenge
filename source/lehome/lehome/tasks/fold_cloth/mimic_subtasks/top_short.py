"""MimicGen subtask decomposition for short-sleeve tops.

Same 9-subtask structure as :mod:`top_long`, but the first-fold drop zone
is in the UPPER half of the garment (collar side) instead of the lower
half (hem side).  Concretely this means:

* Subtask 2 (carry middle keypoint to the release zone) targets the UPPER
  corner keypoints instead of the lower ones — the object_ref and the
  selection keypoints switch from ``*_lower`` to ``*_upper``.
* Subtask 3 (release) also retargets to the upper corners.
* The termination signals ``*_middle_to_lower`` and ``release_*_middle``
  are reused by name.  Their bodies read the release-zone geometry from
  the env cfg, so ``subtask_release_zone_upper_fraction`` (set in
  :class:`GarmentFoldMimicEnvCfg` when garment_type is top-short-sleeve)
  is what actually flips the drop location from hem to collar.
* Subtasks 0-1 and 4-8 are reused verbatim from top_long.
* Arm-sync constraints are identical.
"""
from __future__ import annotations

from copy import deepcopy

from . import top_long


_LOWER_TO_UPPER = {
    "garment_left_lower": "garment_left_upper",
    "garment_right_lower": "garment_right_upper",
}


def _retarget_subtask_to_upper(subtask, side: str) -> None:
    """Swap lower-corner keypoint references to upper-corner equivalents.

    ``object_ref`` flips only for the matching ``side`` (subtasks are
    anchored to *this* arm's drop target).  ``keypoint_names`` in the
    selection strategy flip on BOTH sides so the release zone is framed
    by the two upper corners plus the carried middle keypoint, matching
    the symmetry of the corresponding lower-zone frame in top_long.
    """
    side_lower = f"garment_{side}_lower"
    side_upper = f"garment_{side}_upper"

    if subtask.object_ref == side_lower:
        subtask.object_ref = side_upper

    kwargs = subtask.selection_strategy_kwargs or {}
    keypoint_names = kwargs.get("keypoint_names")
    if keypoint_names:
        kwargs["keypoint_names"] = [_LOWER_TO_UPPER.get(name, name) for name in keypoint_names]
        subtask.selection_strategy_kwargs = kwargs


def build(cfg):
    """Return (subtask_configs, task_constraint_configs) for top_short.

    Starts from ``top_long.build(cfg)`` and retargets subtasks 2 and 3 on
    both arms from the lower corners to the upper corners.
    """
    # ``left_middle_to_lower`` / ``release_*_middle`` read the release-zone
    # geometry from env cfg at runtime.  Make the top-half zone explicit here
    # so callers do not need to remember a separate cfg-side override.
    if hasattr(cfg, "subtask_release_zone_upper_fraction") and hasattr(
        cfg, "subtask_release_zone_lower_fraction"
    ):
        if getattr(cfg, "subtask_release_zone_upper_fraction", None) is None:
            cfg.subtask_release_zone_upper_fraction = cfg.subtask_release_zone_lower_fraction

    subtask_configs, task_constraint_configs = top_long.build(cfg)
    subtask_configs = deepcopy(subtask_configs)
    task_constraint_configs = deepcopy(task_constraint_configs)

    for side, key in (("left", "left_arm"), ("right", "right_arm")):
        arm_subtasks = subtask_configs[key]
        # Subtask 2: middle keypoint carried to the upper release zone.
        _retarget_subtask_to_upper(arm_subtasks[2], side)
        # Subtask 3: release_middle, keyed by object_ref to the upper corner
        # so the transformed trajectory is anchored to where the drop lands.
        _retarget_subtask_to_upper(arm_subtasks[3], side)

    return subtask_configs, task_constraint_configs
