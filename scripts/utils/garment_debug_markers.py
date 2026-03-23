"""Shared garment semantic keypoint debug markers for live viewport overlays."""

from __future__ import annotations

import importlib
from typing import Any, Optional

import numpy as np

from lehome.tasks.fold_cloth.checkpoint_mappings import CSV_TRACE_KEYPOINT_NAMES
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


class GarmentKeypointDebugMarkers:
    """Viewport-only garment keypoint overlay backed by Isaac debug draw."""

    _DEBUG_DRAW_EXTENSION = "isaacsim.util.debug_draw"
    _SEMANTIC_KEYPOINT_NAMES = CSV_TRACE_KEYPOINT_NAMES
    _MARKER_COLORS = (
        (0.0, 1.0, 1.0, 1.0),
        (1.0, 0.5, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
    )
    _MARKER_POINT_SIZE = 14
    _MARKER_SIZES = [_MARKER_POINT_SIZE] * len(_SEMANTIC_KEYPOINT_NAMES)

    def __init__(self):
        self._draw, self._release_draw_interface = self._create_debug_draw_interface()
        self._disabled = False
        self._has_drawn_points = False
        self._warned_missing_keypoints = False
        self._warned_update_failure = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @classmethod
    def _create_debug_draw_interface(cls) -> tuple[Any, Any | None]:
        try:
            omni_kit_app = importlib.import_module("omni.kit.app")
        except ImportError as exc:
            raise RuntimeError("Isaac debug draw is unavailable before Kit startup.") from exc

        app = omni_kit_app.get_app()
        if app is None:
            raise RuntimeError("Isaac Kit app is not running.")

        extension_manager = app.get_extension_manager()
        if extension_manager is None:
            raise RuntimeError("Isaac extension manager is unavailable.")

        enable_immediate = getattr(extension_manager, "set_extension_enabled_immediate", None)
        if callable(enable_immediate):
            enabled = enable_immediate(cls._DEBUG_DRAW_EXTENSION, True)
            if enabled is False:
                raise RuntimeError(f"Failed to enable extension '{cls._DEBUG_DRAW_EXTENSION}'.")
        else:
            enable = getattr(extension_manager, "set_extension_enabled", None)
            if callable(enable):
                enable(cls._DEBUG_DRAW_EXTENSION, True)
            else:
                raise RuntimeError("Isaac extension manager cannot enable debug-draw extensions.")

        try:
            debug_draw_module = importlib.import_module("isaacsim.util.debug_draw._debug_draw")
        except ImportError as exc:
            raise RuntimeError(
                "Isaac debug draw module could not be imported after enabling the extension."
            ) from exc

        draw = debug_draw_module.acquire_debug_draw_interface()
        if draw is None:
            raise RuntimeError("Isaac debug draw interface acquisition returned None.")
        release = getattr(debug_draw_module, "release_debug_draw_interface", None)
        return draw, release

    def _clear_drawn_points(self) -> None:
        if not self._has_drawn_points or self._draw is None:
            return
        try:
            self._draw.clear_points()
        except Exception:
            pass
        self._has_drawn_points = False

    def _draw_points(self, translations: np.ndarray) -> None:
        point_list = [tuple(float(v) for v in point[:3]) for point in translations]
        if self._has_drawn_points:
            self._draw.clear_points()
        self._draw.draw_points(point_list, list(self._MARKER_COLORS), list(self._MARKER_SIZES))
        self._has_drawn_points = True

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
                    "Clearing debugging markers until they become available again."
                )
                self._warned_missing_keypoints = True
            self._clear_drawn_points()
            return

        try:
            self._draw_points(translations)
        except Exception as exc:
            if not self._warned_update_failure:
                logger.warning(
                    f"Disabling debugging markers after draw failure: {exc}",
                    exc_info=True,
                )
                self._warned_update_failure = True
            self.disable()

    def close(self) -> None:
        """Clear the overlay and release the backend interface if it was acquired."""
        self._clear_drawn_points()
        if self._draw is not None and callable(self._release_draw_interface):
            try:
                self._release_draw_interface(self._draw)
            except Exception:
                pass
        self._draw = None
        self._release_draw_interface = None
        self._disabled = True

    def disable(self) -> None:
        """Permanently disable the marker overlay for the current run."""
        if self._disabled:
            return
        self.close()

    def _extract_semantic_keypoint_positions(self, env: Any) -> Optional[np.ndarray]:
        garment_obj = getattr(env, "object", None)
        if garment_obj is None or not hasattr(garment_obj, "check_points"):
            return None

        check_points = getattr(garment_obj, "check_points", None)
        if check_points is None or len(check_points) < len(self._SEMANTIC_KEYPOINT_NAMES):
            return None

        try:
            checkpoint_positions = garment_obj.get_checkpoint_world_positions(
                tuple(check_points[: len(self._SEMANTIC_KEYPOINT_NAMES)]),
                as_numpy=True,
            )
        except Exception:
            return None

        checkpoint_positions = np.asarray(checkpoint_positions, dtype=np.float32)
        if (
            checkpoint_positions.ndim != 2
            or checkpoint_positions.shape[0] < len(self._SEMANTIC_KEYPOINT_NAMES)
            or checkpoint_positions.shape[1] < 3
        ):
            return None
        return checkpoint_positions[: len(self._SEMANTIC_KEYPOINT_NAMES)]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass