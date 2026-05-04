"""Garment-specific MimicGen subtask decompositions.

Each submodule exposes ``build(cfg) -> (subtask_configs, task_constraint_configs)``
returning the data consumed by :class:`GarmentFoldMimicEnvCfg`.  The
``BUILDERS`` mapping dispatches on the canonical garment-type string
returned by :meth:`ChallengeGarmentLoader.get_garment_type` — one of
``{"top-long-sleeve", "top-short-sleeve", "short-pant", "long-pant"}``.
"""
from __future__ import annotations

from . import pant_short, pants_long, top_long, top_short

BUILDERS = {
    "top-long-sleeve": top_long.build,
    "top-short-sleeve": top_short.build,
    "long-pant": pants_long.build,
    "short-pant": pant_short.build,
}

__all__ = ["BUILDERS", "pant_short", "pants_long", "top_long", "top_short"]
