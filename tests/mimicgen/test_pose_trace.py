from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.mimicgen.core.pose_trace import (
    PoseTraceCsvWriter,
    TRACE_EEF_KEYPOINT_GROUPS,
    TRACE_KEYPOINT_NAMES,
    TRACE_SUCCESS_DISTANCE_SPECS,
    build_pose_snapshot,
    resolve_pose_output_path,
    write_pose_snapshot,
)


def _pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [x, y, z]
    return pose


class FakeEnv:
    def __init__(self) -> None:
        self._eef_poses = {
            "left_arm": _pose(0.0, 0.0, 0.0)[None, ...],
            "right_arm": _pose(1.0, 0.0, 0.0)[None, ...],
        }
        required_keypoints = set(TRACE_KEYPOINT_NAMES)
        for _, src_name, dst_name, _ in TRACE_SUCCESS_DISTANCE_SPECS:
            required_keypoints.add(src_name)
            required_keypoints.add(dst_name)

        self._object_poses = {}
        for idx, keypoint_name in enumerate(sorted(required_keypoints)):
            self._object_poses[keypoint_name] = _pose(float(idx), 0.0, 1.0)[None, ...]

        left_keypoint = TRACE_EEF_KEYPOINT_GROUPS["left_arm"][0]
        right_keypoint = TRACE_EEF_KEYPOINT_GROUPS["right_arm"][0]
        self._object_poses[left_keypoint] = _pose(0.0, 0.0, 1.0)[None, ...]
        self._object_poses[right_keypoint] = _pose(1.0, 0.0, 1.0)[None, ...]

    def get_robot_eef_pose(self, *, eef_name: str, env_ids):
        return self._eef_poses[eef_name]

    def get_object_poses(self, *, env_ids):
        return self._object_poses


class PoseTraceTests(unittest.TestCase):
    def test_trace_constants_match_current_garment_semantics(self) -> None:
        self.assertEqual(
            TRACE_KEYPOINT_NAMES,
            (
                "garment_left_upper",
                "garment_right_upper",
                "garment_left_middle",
                "garment_right_middle",
                "garment_left_lower",
                "garment_right_lower",
            ),
        )
        self.assertEqual(
            TRACE_EEF_KEYPOINT_GROUPS,
            {
                "left_arm": (
                    "garment_left_upper",
                    "garment_left_middle",
                    "garment_left_lower",
                ),
                "right_arm": (
                    "garment_right_upper",
                    "garment_right_middle",
                    "garment_right_lower",
                ),
            },
        )
        self.assertEqual(
            tuple(metric_name for metric_name, _, _, _ in TRACE_SUCCESS_DISTANCE_SPECS),
            (
                "left_middle_to_lower",
                "right_middle_to_lower",
                "left_lower_to_upper",
                "right_lower_to_upper",
            ),
        )

    def test_resolve_pose_output_path_defaults_next_to_output_file(self) -> None:
        path = resolve_pose_output_path("/tmp/output_dataset.hdf5", None)
        self.assertEqual(path, Path("/tmp/output_dataset_pose_trace.csv"))

    def test_build_pose_snapshot_populates_expected_fields(self) -> None:
        env = FakeEnv()
        left_keypoint = TRACE_EEF_KEYPOINT_GROUPS["left_arm"][0]

        row = build_pose_snapshot(env, step_count=7, env_id=0, episode_index=2, episode_step=3)

        self.assertEqual(row["step"], 7)
        self.assertEqual(row["episode_index"], 2)
        self.assertEqual(row["eef_left_arm_x"], 0.0)
        self.assertEqual(row[f"keypoint_{left_keypoint}_z"], 1.0)
        self.assertEqual(row[f"dist_left_arm_to_{left_keypoint}_m"], 1.0)

    def test_write_pose_snapshot_writes_csv_row(self) -> None:
        env = FakeEnv()
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "pose_trace.csv"
            writer = PoseTraceCsvWriter(csv_path)
            try:
                row = write_pose_snapshot(
                    env,
                    step_count=1,
                    env_id=0,
                    pose_writer=writer,
                    episode_index=0,
                    episode_step=1,
                )
            finally:
                writer.close()

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], "1")
        self.assertIn("eef_left_arm_x", rows[0])
        self.assertEqual(float(rows[0]["eef_left_arm_x"]), row["eef_left_arm_x"])


if __name__ == "__main__":
    unittest.main()
