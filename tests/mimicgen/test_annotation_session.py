from __future__ import annotations

import unittest
from unittest import mock

from scripts.mimicgen.core.annotation_session import AnnotationSessionController


class AnnotationSessionControllerTests(unittest.TestCase):
    def test_reset_attempt_state_restores_defaults(self) -> None:
        controller = AnnotationSessionController()
        controller.configure_manual_marking("left_arm", ["a", "b"])
        controller.state.paused = True
        controller.state.skip_episode = True
        controller.state.current_action_index = 7

        controller.reset_attempt_state()

        self.assertFalse(controller.state.paused)
        self.assertFalse(controller.state.skip_episode)
        self.assertEqual(controller.state.current_action_index, 0)
        self.assertEqual(controller.state.marked_subtask_action_indices, [])
        self.assertIsNone(controller.state.expected_subtask_mark_count)
        self.assertEqual(controller.state.active_mark_signal_names, [])

    def test_mark_subtask_respects_gap_and_debounce(self) -> None:
        controller = AnnotationSessionController(mark_debounce_sec=0.5, mark_min_action_gap=3)
        controller.configure_manual_marking("left_arm", ["first", "second"])

        with mock.patch("builtins.print"):
            with mock.patch("scripts.mimicgen.core.annotation_session.time.perf_counter", side_effect=[1.0, 1.1, 2.0]):
                controller.set_current_action_index(10)
                controller.mark_subtask()
                controller.set_current_action_index(11)
                controller.mark_subtask()
                controller.set_current_action_index(14)
                controller.mark_subtask()

        self.assertEqual(controller.state.marked_subtask_action_indices, [10, 14])

    def test_register_callbacks_omits_manual_mark_in_auto_mode(self) -> None:
        controller = AnnotationSessionController()

        class FakeKeyboard:
            def __init__(self) -> None:
                self.keys: list[str] = []

            def add_callback(self, key: str, callback) -> None:
                del callback
                self.keys.append(key)

        keyboard = FakeKeyboard()
        controller.register_callbacks(keyboard, auto_mode=True)
        self.assertEqual(keyboard.keys, ["N", "B", "Q"])


if __name__ == "__main__":
    unittest.main()
