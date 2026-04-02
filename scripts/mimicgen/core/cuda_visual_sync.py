"""Shared CUDA render-sync helpers for teleop and generation workflows."""

from __future__ import annotations

import contextlib
from typing import Any

import carb
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from pxr import Gf, Sdf, Usd, UsdGeom


def cuda_runtime_enabled(device_or_env: Any) -> bool:
    """Return whether the given env or device string is CUDA-backed."""
    if hasattr(device_or_env, "device"):
        return "cuda" in str(getattr(device_or_env, "device"))
    return "cuda" in str(device_or_env)


def cuda_visual_sync_enabled(env: Any) -> bool:
    """Return whether CUDA render-facing USD sync is needed for this env."""
    if not cuda_runtime_enabled(env):
        return False
    with contextlib.suppress(Exception):
        return bool(env.sim.has_gui() or env.sim.has_rtx_sensors())
    return False


def apply_cuda_fabric_render_settings(env_cfg: Any, device: Any, *, context: str) -> bool:
    """Enable fabric-backed transform updates for CUDA environments."""
    if not cuda_runtime_enabled(device):
        return False

    with contextlib.suppress(Exception):
        env_cfg.sim.use_fabric = True

    carb_settings = carb.settings.get_settings()
    carb_settings.set_bool("/physics/fabricUpdateTransformations", True)
    carb_settings.set_bool("/physics/fabricUpdateVelocities", True)
    carb_settings.set_bool("/physics/fabricUpdateJointStates", True)
    print(f"Enabled fabric-backed transform sync for CUDA {context}.")
    return True


def _ensure_cuda_robot_visual_sync_initialized(
    env: Any,
) -> dict[str, list[tuple[Any, Any, Any]]]:
    cache = getattr(env, "_cuda_robot_visual_sync_cache", None)
    if cache is not None:
        return cache

    stage = env.scene.stage
    cache = {}
    for arm_name in ("left_arm", "right_arm"):
        with contextlib.suppress(Exception):
            arm = env.scene[arm_name]
            prim_entries = []
            for prim_path in arm.root_physx_view.link_paths[0]:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                with contextlib.suppress(Exception):
                    sim_utils.standardize_xform_ops(prim)
                translate_attr = prim.GetAttribute("xformOp:translate")
                orient_attr = prim.GetAttribute("xformOp:orient")
                if not translate_attr.IsValid() or not orient_attr.IsValid():
                    continue
                parent_prim = prim.GetParent() if prim.GetParent().IsValid() else None
                prim_entries.append((translate_attr, orient_attr, parent_prim))
            if prim_entries:
                cache[arm_name] = prim_entries

    setattr(env, "_cuda_robot_visual_sync_cache", cache)
    return cache


def sync_cuda_robot_visuals_to_usd(env: Any) -> None:
    """Mirror live articulation link transforms into USD for CUDA rendering."""
    cache = _ensure_cuda_robot_visual_sync_initialized(env)
    if not cache:
        return

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    with Sdf.ChangeBlock():
        for arm_name, prim_entries in cache.items():
            arm = env.scene[arm_name]
            poses = arm.root_physx_view.get_link_transforms()[0]
            count = min(len(prim_entries), int(poses.shape[0]))
            poses_cpu = poses[:count].detach().cpu()
            positions_cpu = poses_cpu[:, :3].numpy()
            orientations_cpu = math_utils.convert_quat(poses_cpu[:, 3:7], to="wxyz").numpy()
            for body_idx in range(count):
                translate_attr, orient_attr, parent_prim = prim_entries[body_idx]
                world_pos = Gf.Vec3d(*positions_cpu[body_idx].tolist())
                world_quat = Gf.Quatd(
                    float(orientations_cpu[body_idx, 0]),
                    Gf.Vec3d(*orientations_cpu[body_idx, 1:].tolist()),
                )

                if parent_prim is not None and parent_prim.GetPath() != Sdf.Path.absoluteRootPath:
                    prim_tf = Gf.Matrix4d()
                    prim_tf.SetTranslateOnly(world_pos)
                    prim_tf.SetRotateOnly(world_quat)
                    parent_world_tf = xform_cache.GetLocalToWorldTransform(parent_prim)
                    local_tf = prim_tf * parent_world_tf.GetInverse()
                    local_pos = local_tf.ExtractTranslation()
                    local_quat = local_tf.ExtractRotationQuat()
                else:
                    local_pos = world_pos
                    local_quat = world_quat

                translate_attr.Set(local_pos)
                orient_attr.Set(local_quat)


def force_cuda_render_sync(env: Any) -> None:
    """Force a render refresh for CUDA envs and mirror robot visuals to USD."""
    if not cuda_visual_sync_enabled(env):
        return
    with contextlib.suppress(Exception):
        sync_cuda_robot_visuals_to_usd(env)
    with contextlib.suppress(Exception):
        env.sim.render()


def restore_cuda_cloth_visual_pose_to_initial(env: Any) -> None:
    """No-op for CUDA cloth.

    On CUDA the fabric renderer drives the cloth visual directly from
    the solver's particle buffer.  Calling set_world_pose() on the
    cloth prim disrupts this and causes the visual mesh to freeze at
    the XForm position instead of tracking the live particles.
    """


def post_reset_cuda_visual_sync(env: Any) -> None:
    """Apply the reset-time CUDA cloth/robot visual sync."""
    if not cuda_visual_sync_enabled(env):
        return
    restore_cuda_cloth_visual_pose_to_initial(env)
    force_cuda_render_sync(env)
