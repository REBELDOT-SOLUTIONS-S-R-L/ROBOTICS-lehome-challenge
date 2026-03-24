"""Online subtask annotation state for integrated fold-cloth teleoperation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lehome.tasks.fold_cloth.mdp.observations import (
    FoldClothSubtaskObservationContext,
    get_subtask_signal_observation,
    get_subtask_signal_observation_from_context,
)


_RETURN_HOME_SIGNALS = {"left_return_home", "right_return_home"}
_DEFAULT_ARM_QUEUES = {
    "left_arm": [
        "grasp_left_middle",
        "left_middle_to_lower",
        "grasp_left_lower",
        "left_lower_to_upper",
        "left_return_home",
    ],
    "right_arm": [
        "grasp_right_middle",
        "right_middle_to_lower",
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
    _completion_announced: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        all_signals = [signal for queue in self.arm_queues.values() for signal in queue]
        self.latched_signals = {signal: False for signal in all_signals}
        self.arm_queue_indices = {arm_name: 0 for arm_name in self.arm_queues}
        self.consecutive_true_counts = {arm_name: 0 for arm_name in self.arm_queues}

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

    def needs_fold_success(self) -> bool:
        return any(
            (signal_name in _RETURN_HOME_SIGNALS)
            for signal_name in self.current_signal_heads().values()
            if signal_name is not None
        )

    def _advance_from_signal_reader(self, signal_reader) -> list[str]:
        newly_latched: list[str] = []
        for arm_name in self.arm_queues:
            signal_name = self._head_signal_for_arm(arm_name)
            if signal_name is None:
                self.consecutive_true_counts[arm_name] = 0
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
            newly_latched.append(signal_name)
        return newly_latched

    def advance(self, env: Any) -> list[str]:
        """Advance one step of online annotation and return newly latched signals."""
        return self._advance_from_signal_reader(
            lambda signal_name: get_subtask_signal_observation(env, signal_name, env_ids=[0])
        )

    def advance_from_context(
        self,
        context: FoldClothSubtaskObservationContext,
    ) -> list[str]:
        """Advance one step of online annotation using precomputed predicate inputs."""
        return self._advance_from_signal_reader(
            lambda signal_name: get_subtask_signal_observation_from_context(context, signal_name)
        )

    def as_tensor_dict(self) -> dict[str, torch.Tensor]:
        """Return current latched state as float32 tensors shaped (1, 1)."""
        return {
            signal_name: torch.tensor(
                [[1.0 if latched else 0.0]],
                dtype=torch.float32,
                device=self.device,
            )
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
