"""Online subtask annotation state for integrated fold-cloth teleoperation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lehome.tasks.fold_cloth.mdp.observations import (
    FoldClothSubtaskObservationContext,
    build_subtask_observation_context,
    get_subtask_signal_observation,
    get_subtask_signal_observation_from_context,
)


_RETURN_HOME_SIGNALS = {"left_return_home", "right_return_home"}
_GRASP_SIGNALS = {
    "grasp_left_middle",
    "grasp_right_middle",
    "grasp_left_lower",
    "grasp_right_lower",
}
_DEFAULT_ARM_QUEUES = {
    "left_arm": [
        "prepare_for_grasp_left_middle",
        "grasp_left_middle",
        "left_middle_to_lower",
        "release_left_middle",
        "left_at_waiting_pos",
        "prepare_for_grasp_left_lower",
        "grasp_left_lower",
        "left_lower_to_upper",
        "left_return_home",
    ],
    "right_arm": [
        "prepare_for_grasp_right_middle",
        "grasp_right_middle",
        "right_middle_to_lower",
        "release_right_middle",
        "right_at_waiting_pos",
        "prepare_for_grasp_right_lower",
        "grasp_right_lower",
        "right_lower_to_upper",
        "right_return_home",
    ],
}


@dataclass
class OnlineAnnotationState:
    """Per-episode online subtask annotation state."""

    arm_queues: dict[str, list[str]]
    device: torch.device | str
    subtask_signal_min_consecutive_steps: int = 3
    return_home_min_consecutive_steps: int = 10
    latched_signals: dict[str, bool] = field(init=False)
    arm_queue_indices: dict[str, int] = field(init=False)
    consecutive_true_counts: dict[str, int] = field(init=False)
    grasp_open_ready_by_arm: dict[str, bool] = field(init=False)
    _completion_announced: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        all_signals = [signal for queue in self.arm_queues.values() for signal in queue]
        self.latched_signals = {signal: False for signal in all_signals}
        self.arm_queue_indices = {arm_name: 0 for arm_name in self.arm_queues}
        self.consecutive_true_counts = {arm_name: 0 for arm_name in self.arm_queues}
        self.grasp_open_ready_by_arm = {arm_name: False for arm_name in self.arm_queues}

    @classmethod
    def from_env(cls, env: Any) -> "OnlineAnnotationState":
        subtask_cfgs = getattr(getattr(env, "cfg", None), "subtask_configs", {})
        arm_queues: dict[str, list[str]] = {}
        if isinstance(subtask_cfgs, dict) and subtask_cfgs:
            for arm_name, cfgs in subtask_cfgs.items():
                queue: list[str] = []
                for cfg in cfgs:
                    signal_name = getattr(cfg, "subtask_term_signal", None)
                    if signal_name is None:
                        continue
                    queue.append(str(signal_name))
                if queue:
                    arm_queues[str(arm_name)] = queue

        if not arm_queues:
            arm_queues = {arm_name: list(queue) for arm_name, queue in _DEFAULT_ARM_QUEUES.items()}

        cfg = getattr(env, "cfg", None)
        return cls(
            arm_queues=arm_queues,
            device=env.device,
            subtask_signal_min_consecutive_steps=int(
                getattr(cfg, "subtask_signal_min_consecutive_steps", 3)
            ),
            return_home_min_consecutive_steps=int(
                getattr(cfg, "return_home_min_consecutive_steps", 10)
            ),
        )

    def reset(self) -> None:
        for signal_name in self.latched_signals:
            self.latched_signals[signal_name] = False
        for arm_name in self.arm_queue_indices:
            self.arm_queue_indices[arm_name] = 0
            self.consecutive_true_counts[arm_name] = 0
            self.grasp_open_ready_by_arm[arm_name] = False
        self._completion_announced = False

    def _required_dwell(self, signal_name: str) -> int:
        dwell = (
            self.return_home_min_consecutive_steps
            if signal_name in _RETURN_HOME_SIGNALS
            else self.subtask_signal_min_consecutive_steps
        )
        return max(1, int(dwell))

    def _head_signal_for_arm(self, arm_name: str) -> str | None:
        queue = self.arm_queues.get(arm_name, [])
        index = int(self.arm_queue_indices.get(arm_name, 0))
        if index >= len(queue):
            return None
        return queue[index]

    def _grasp_target_name(self, signal_name: str) -> str | None:
        mapping = {
            "grasp_left_middle": "garment_left_middle",
            "grasp_right_middle": "garment_right_middle",
            "grasp_left_lower": "garment_left_lower",
            "grasp_right_lower": "garment_right_lower",
        }
        return mapping.get(signal_name)

    def _grasp_proximity_true(
        self,
        context: FoldClothSubtaskObservationContext,
        arm_name: str,
        signal_name: str,
    ) -> bool:
        target_name = self._grasp_target_name(signal_name)
        if target_name is None:
            return False
        eef_pos = context.eef_world_positions.get(arm_name)
        keypoint_pos = context.semantic_keypoints_world.get(target_name)
        if eef_pos is None or keypoint_pos is None:
            return False
        distance = torch.linalg.norm(eef_pos - keypoint_pos, dim=-1, keepdim=True)
        return bool(
            (distance <= float(context.grasp_eef_to_keypoint_threshold_m))
            .reshape(-1)[0]
            .item()
        )

    def needs_fold_success(self) -> bool:
        return any(
            (signal_name in _RETURN_HOME_SIGNALS)
            for signal_name in self.current_signal_heads().values()
            if signal_name is not None
        )

    def _advance_joint_return_home_from_context(
        self,
        context: FoldClothSubtaskObservationContext,
    ) -> list[str] | None:
        left_signal = self._head_signal_for_arm("left_arm")
        right_signal = self._head_signal_for_arm("right_arm")
        # Only take over when BOTH arms are at return_home.
        # Otherwise return None so normal per-arm signal evaluation runs.
        if left_signal != "left_return_home" or right_signal != "right_return_home":
            return None

        fold_success = bool(
            torch.as_tensor(context.fold_success).reshape(-1)[0].item()
        ) if context.fold_success is not None else False
        left_at_rest = bool(
            torch.as_tensor(context.arm_at_rest_by_arm.get("left_arm", False)).reshape(-1)[0].item()
        )
        right_at_rest = bool(
            torch.as_tensor(context.arm_at_rest_by_arm.get("right_arm", False)).reshape(-1)[0].item()
        )
        if not (fold_success and left_at_rest and right_at_rest):
            self.consecutive_true_counts["left_arm"] = 0
            self.consecutive_true_counts["right_arm"] = 0
            return []

        newly_latched = []
        for arm_name, signal_name in (
            ("left_arm", "left_return_home"),
            ("right_arm", "right_return_home"),
        ):
            self.latched_signals[signal_name] = True
            self.arm_queue_indices[arm_name] += 1
            self.consecutive_true_counts[arm_name] = 0
            self.grasp_open_ready_by_arm[arm_name] = False
            newly_latched.append(signal_name)
        return newly_latched

    def _advance_from_signal_reader(self, signal_reader) -> list[str]:
        newly_latched: list[str] = []
        for arm_name in self.arm_queues:
            signal_name = self._head_signal_for_arm(arm_name)
            if signal_name is None:
                self.consecutive_true_counts[arm_name] = 0
                self.grasp_open_ready_by_arm[arm_name] = False
                continue

            observation = signal_reader(signal_name)
            is_true = bool(torch.as_tensor(observation).reshape(-1)[0].item())
            if is_true:
                self.consecutive_true_counts[arm_name] += 1
            else:
                self.consecutive_true_counts[arm_name] = 0

            if self.consecutive_true_counts[arm_name] < self._required_dwell(signal_name):
                continue

            self.latched_signals[signal_name] = True
            self.arm_queue_indices[arm_name] += 1
            self.consecutive_true_counts[arm_name] = 0
            self.grasp_open_ready_by_arm[arm_name] = False
            newly_latched.append(signal_name)
        return newly_latched

    def advance(self, env: Any) -> list[str]:
        """Advance one step of online annotation and return newly latched signals."""
        return self.advance_from_context(
            build_subtask_observation_context(
                env,
                env_ids=[0],
                include_fold_success=self.needs_fold_success(),
            )
        )

    def advance_from_context(
        self,
        context: FoldClothSubtaskObservationContext,
    ) -> list[str]:
        """Advance one step of online annotation using precomputed predicate inputs."""
        joint_return_home_latched = self._advance_joint_return_home_from_context(context)
        if joint_return_home_latched is not None:
            return joint_return_home_latched

        newly_latched: list[str] = []
        for arm_name in self.arm_queues:
            signal_name = self._head_signal_for_arm(arm_name)
            if signal_name is None:
                self.consecutive_true_counts[arm_name] = 0
                self.grasp_open_ready_by_arm[arm_name] = False
                continue

            if signal_name in _GRASP_SIGNALS:
                gripper_closed = bool(
                    torch.as_tensor(context.gripper_closed_by_arm.get(arm_name, False))
                    .reshape(-1)[0]
                    .item()
                )
                is_near_target = self._grasp_proximity_true(context, arm_name, signal_name)
                if is_near_target and not gripper_closed:
                    self.grasp_open_ready_by_arm[arm_name] = True
                is_true = bool(
                    is_near_target
                    and gripper_closed
                    and self.grasp_open_ready_by_arm.get(arm_name, False)
                )
            else:
                self.grasp_open_ready_by_arm[arm_name] = False
                observation = get_subtask_signal_observation_from_context(context, signal_name)
                is_true = bool(torch.as_tensor(observation).reshape(-1)[0].item())

            if is_true:
                self.consecutive_true_counts[arm_name] += 1
            else:
                self.consecutive_true_counts[arm_name] = 0

            if self.consecutive_true_counts[arm_name] < self._required_dwell(signal_name):
                continue

            self.latched_signals[signal_name] = True
            self.arm_queue_indices[arm_name] += 1
            self.consecutive_true_counts[arm_name] = 0
            self.grasp_open_ready_by_arm[arm_name] = False
            newly_latched.append(signal_name)
        return newly_latched

    def as_bool_dict(self) -> dict[str, bool]:
        """Return current latched state as plain booleans for recorder packing."""
        return {
            signal_name: bool(latched)
            for signal_name, latched in self.latched_signals.items()
        }

    def is_complete(self) -> bool:
        return all(
            int(self.arm_queue_indices.get(arm_name, 0)) >= len(queue)
            for arm_name, queue in self.arm_queues.items()
        )

    def current_signal_heads(self) -> dict[str, str | None]:
        return {
            arm_name: self._head_signal_for_arm(arm_name)
            for arm_name in self.arm_queues
        }


__all__ = ["OnlineAnnotationState"]
