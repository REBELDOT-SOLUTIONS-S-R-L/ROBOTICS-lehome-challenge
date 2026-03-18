"""Shared garment semantic keypoint debug markers for live viewport overlays."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from lehome.tasks.fold_cloth.checkpoint_mappings import (
    CSV_TRACE_KEYPOINT_NAMES,
    semantic_keypoints_from_positions,
)
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


class GarmentKeypointDebugMarkers:
    """Viewport-only garment keypoint marker overlay for live cloth debugging."""

    _MARKER_RADIUS_M = 0.008
    _SEMANTIC_KEYPOINT_NAMES = CSV_TRACE_KEYPOINT_NAMES
    _MARKER_COLORS = (
        (0.0, 1.0, 1.0),
        (1.0, 0.5, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    )
    _MARKER_INDICES = np.arange(len(_SEMANTIC_KEYPOINT_NAMES), dtype=np.int32)

    def __init__(self):
        markers = {}
        for name, color in zip(self._SEMANTIC_KEYPOINT_NAMES, self._MARKER_COLORS):
            markers[name] = sim_utils.SphereCfg(
                radius=self._MARKER_RADIUS_M,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        cfg = VisualizationMarkersCfg(
            prim_path="/World/Debug/GarmentKeypoints",
            markers=markers,
        )
        self._markers = VisualizationMarkers(cfg)
        self._markers.set_visibility(False)
        self._disabled = False
        self._warned_missing_keypoints = False
        self._warned_update_failure = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    def update_from_env(self, env: Any) -> None:
        """Update marker positions from the current cloth state."""
        if self._disabled:
            return

        try:
            translations = self._extract_semantic_keypoint_positions(env)
        except Exception as exc:
            if not self._warned_update_failure:
                logger.warning(
                    f"Disabling debugging markers after update failure: {exc}",
                    exc_info=True,
                )
                self._warned_update_failure = True
            self.disable()
            return

        if translations is None:
            if not self._warned_missing_keypoints:
                logger.warning(
                    "Garment semantic keypoints are unavailable in the current environment. "
                    "Hiding debugging markers."
                )
                self._warned_missing_keypoints = True
            self._markers.set_visibility(False)
            return

        self._markers.set_visibility(True)
        self._markers.visualize(
            translations=translations,
            marker_indices=self._MARKER_INDICES,
        )

    def disable(self) -> None:
        """Permanently disable the marker overlay for the current run."""
        if self._disabled:
            return
        self._disabled = True
        try:
            self._markers.set_visibility(False)
        except Exception:
            pass

    def _extract_semantic_keypoint_positions(self, env: Any) -> Optional[np.ndarray]:
        garment_obj = getattr(env, "object", None)
        if garment_obj is None or not hasattr(garment_obj, "check_points"):
            return None

        check_points = getattr(garment_obj, "check_points", None)
        if check_points is None or len(check_points) < len(self._SEMANTIC_KEYPOINT_NAMES):
            return None

        mesh_points = None
        try:
            mesh_points_world, _, _, _ = garment_obj.get_current_mesh_points()
            mesh_points = np.asarray(mesh_points_world, dtype=np.float32)
        except Exception:
            cloth_prim_view = getattr(garment_obj, "_cloth_prim_view", None)
            if cloth_prim_view is None:
                return None
            try:
                mesh_points = (
                    cloth_prim_view.get_world_positions()
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            except Exception:
                return None

        kp_positions = mesh_points[list(check_points)]
        semantic_points = semantic_keypoints_from_positions(kp_positions)
        return np.stack(
            [np.asarray(semantic_points[name], dtype=np.float32) for name in self._SEMANTIC_KEYPOINT_NAMES],
            axis=0,
        )
