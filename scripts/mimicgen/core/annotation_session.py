"""Interactive session state for MimicGen demo annotation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnnotationSessionState:
    """Mutable annotation session state shared by runtime and keyboard callbacks."""

    paused: bool = False
    current_action_index: int = 0
    marked_subtask_action_indices: list[int] = field(default_factory=list)
    skip_episode: bool = False
    expected_subtask_mark_count: int | None = None
    last_marked_action_index: int = -10**9
    last_mark_wall_time: float = 0.0
    active_mark_eef_name: str | None = None
    active_mark_signal_names: list[str] = field(default_factory=list)


class AnnotationSessionController:
    """Own keyboard-driven annotation state transitions and mark bookkeeping."""

    def __init__(
        self,
        *,
        mark_debounce_sec: float = 0.2,
        mark_min_action_gap: int = 3,
    ) -> None:
        self.state = AnnotationSessionState()
        self._mark_debounce_sec = float(mark_debounce_sec)
        self._mark_min_action_gap = int(mark_min_action_gap)

    def register_callbacks(self, keyboard_interface: Any, *, auto_mode: bool) -> None:
        """Bind keyboard callbacks to the provided interface."""
        keyboard_interface.add_callback("N", self.play)
        keyboard_interface.add_callback("B", self.pause)
        keyboard_interface.add_callback("Q", self.skip_episode)
        if not auto_mode:
            keyboard_interface.add_callback("S", self.mark_subtask)

    def play(self) -> None:
        self.state.paused = False

    def pause(self) -> None:
        self.state.paused = True

    def skip_episode(self) -> None:
        self.state.skip_episode = True

    def set_current_action_index(self, action_index: int) -> None:
        self.state.current_action_index = int(action_index)

    def configure_manual_marking(self, eef_name: str, signal_names: list[str]) -> None:
        """Prepare one manual marking pass for a specific EEF."""
        self.state.paused = False
        self.state.skip_episode = False
        self.state.marked_subtask_action_indices = []
        self.state.expected_subtask_mark_count = len(signal_names)
        self.state.active_mark_eef_name = eef_name
        self.state.active_mark_signal_names = list(signal_names)
        self.state.last_marked_action_index = -10**9
        self.state.last_mark_wall_time = 0.0

    def clear_manual_marking(self) -> None:
        self.state.expected_subtask_mark_count = None
        self.state.active_mark_eef_name = None
        self.state.active_mark_signal_names = []

    def reset_attempt_state(self) -> None:
        """Clear replay and marking state before replaying an episode."""
        self.state = AnnotationSessionState()

    def mark_subtask(self) -> None:
        """Record the current replay action index as a manual subtask boundary."""
        now = time.perf_counter()
        state = self.state

        if (
            state.expected_subtask_mark_count is not None
            and len(state.marked_subtask_action_indices) >= state.expected_subtask_mark_count
        ):
            return
        if (now - state.last_mark_wall_time) < self._mark_debounce_sec:
            return
        if (state.current_action_index - state.last_marked_action_index) < self._mark_min_action_gap:
            return

        marked_signal_index = len(state.marked_subtask_action_indices)
        state.marked_subtask_action_indices.append(state.current_action_index)
        state.last_mark_wall_time = now
        state.last_marked_action_index = state.current_action_index

        signal_name = None
        if marked_signal_index < len(state.active_mark_signal_names):
            signal_name = state.active_mark_signal_names[marked_signal_index]
        if signal_name is not None:
            print(
                f'Marked subtask signal "{signal_name}"'
                f' for eef "{state.active_mark_eef_name}" at action index: {state.current_action_index}'
            )
        else:
            print(f"Marked a subtask signal at action index: {state.current_action_index}")


__all__ = ["AnnotationSessionController", "AnnotationSessionState"]
