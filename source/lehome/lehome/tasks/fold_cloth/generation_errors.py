"""Exceptions raised during MimicGen generation for the garment-folding task.

These are thrown from within ``GarmentFoldEnv`` (or from the MimicGen
``DataGenerator.generate`` loop via hooks on the env) to signal conditions
that should abort the current trial without propagating as generic errors.
"""

from __future__ import annotations

from collections.abc import Mapping


class SubtaskVerificationError(RuntimeError):
    """Raised when a just-finished subtask fails its post-completion check.

    The generation loop catches this, discards the per-step data accumulated
    for the current episode, and writes a minimal "failed" row containing
    only the information needed to reproduce and diagnose the failure:
    ``initial_state/garment_initial_pose``, ``source_demo_indices`` (per arm,
    only the subtasks that were actually selected before the failure), and a
    free-form ``fail_reason`` string.

    Attributes:
        arm_name: Name of the end-effector whose subtask verification failed
            (e.g. ``"left_arm"`` / ``"right_arm"``).
        subtask_index: Zero-based index of the failing subtask within that
            arm's ``subtask_configs`` list.
        fail_reason: Short, human-readable explanation written to the
            failed-episode dataset.
        source_demo_selections: Mapping from arm name to the list of source
            demo indices selected up to (and including) the failing subtask.
    """

    def __init__(
        self,
        *,
        arm_name: str,
        subtask_index: int,
        fail_reason: str,
        source_demo_selections: Mapping[str, list[int]] | None = None,
    ) -> None:
        message = (
            f"subtask verification failed for {arm_name} "
            f"(subtask_index={subtask_index}): {fail_reason}"
        )
        super().__init__(message)
        self.arm_name = str(arm_name)
        self.subtask_index = int(subtask_index)
        self.fail_reason = str(fail_reason)
        self.source_demo_selections: dict[str, list[int]] = {
            str(k): [int(v) for v in vs]
            for k, vs in (source_demo_selections or {}).items()
        }
