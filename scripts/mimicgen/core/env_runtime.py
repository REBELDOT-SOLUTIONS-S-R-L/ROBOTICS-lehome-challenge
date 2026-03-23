"""Shared environment runtime helpers for MimicGen replay and annotation."""

from __future__ import annotations

import sys
from pathlib import Path

from isaaclab.envs import ManagerBasedRLMimicEnv

try:
    from scripts.utils.annotate_utils import ReplayRuntimeContext
except ImportError:
    scripts_dir = Path(__file__).resolve().parents[2]
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    from utils.annotate_utils import ReplayRuntimeContext


def ensure_ik_solver_ready(
    env: ManagerBasedRLMimicEnv,
    replay_runtime: ReplayRuntimeContext,
) -> None:
    """Initialize the environment IK solver once when strict IK replay needs it."""
    if replay_runtime.native_ik_action_contract:
        return
    if replay_runtime.ik_solver_checked:
        if not replay_runtime.ik_solver_ready:
            raise RuntimeError("environment IK solver initialization previously failed")
        return

    replay_runtime.ik_solver_checked = True
    replay_runtime.ik_solver_ready = True
    if not hasattr(env, "_init_ik_solver_if_needed"):
        return
    try:
        if not bool(env._init_ik_solver_if_needed()):
            raise RuntimeError("environment IK solver initialization returned False")
    except Exception as exc:
        replay_runtime.ik_solver_ready = False
        raise RuntimeError(
            "Strict IK replay requires a working IK solver in the environment, "
            f"but initialization failed: {exc}"
        ) from exc


__all__ = ["ensure_ik_solver_ready"]
