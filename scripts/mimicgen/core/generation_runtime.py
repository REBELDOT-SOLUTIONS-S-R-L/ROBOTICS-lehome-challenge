"""Generation runtime helpers for MimicGen workflows."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import traceback
from typing import Any

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

import isaaclab_mimic.datagen.generation as mimic_generation
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import TerminationTermCfg
from isaaclab_mimic.datagen.data_generator import DataGenerator
from pxr import Gf, Sdf, Usd, UsdGeom

from lehome.tasks.fold_cloth.checkpoint_mappings import (
    ClothObjectPoseUnavailableError,
    ClothObjectPoseValidationError,
)
from lehome.utils.logger import get_logger

from .generation_source import RobustDataGenInfoPool
from .pose_trace import PoseTraceCsvWriter
from .pose_trace import build_pose_snapshot as _build_pose_snapshot
from .pose_trace import resolve_pose_output_path as _resolve_pose_output_path
from .pose_trace import write_pose_snapshot as _write_pose_snapshot

logger = get_logger(__name__)
SUCCESS_LOG_INTERVAL = 50


async def _wait_for_next_generation_item(
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
) -> tuple[str, Any, tuple[int, Any] | None]:
    """Wait for either a reset request or an action item."""
    reset_task = asyncio.create_task(env_reset_queue.get())
    action_task = asyncio.create_task(env_action_queue.get())
    done, pending = await asyncio.wait(
        {reset_task, action_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()
    for task in pending:
        with contextlib.suppress(asyncio.CancelledError):
            await task

    if reset_task in done:
        extra_action = action_task.result() if action_task in done else None
        return "reset", reset_task.result(), extra_action
    return "action", action_task.result(), None


def _coerce_action_tensor(
    action: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    action_dim: int,
) -> torch.Tensor:
    """Convert a queued action into a flat device tensor with the expected size."""
    if torch.is_tensor(action):
        action_tensor = action
        if action_tensor.device != torch.device(device):
            action_tensor = action_tensor.to(device=device)
        if action_tensor.dtype != dtype:
            action_tensor = action_tensor.to(dtype=dtype)
    else:
        action_tensor = torch.as_tensor(action, device=device, dtype=dtype)

    action_tensor = action_tensor.reshape(-1)
    if int(action_tensor.numel()) != int(action_dim):
        raise ValueError(
            "Invalid action size from generator: "
            f"expected {action_dim}, received {action_tensor.numel()}."
        )
    return action_tensor


def _cuda_runtime_enabled(env: ManagerBasedRLMimicEnv) -> bool:
    """Return whether generation is running on a CUDA-backed env device."""
    return "cuda" in str(env.device)


def _cuda_visual_sync_enabled(env: ManagerBasedRLMimicEnv) -> bool:
    """Return whether CUDA generation currently needs render-facing USD sync."""
    if not _cuda_runtime_enabled(env):
        return False
    with contextlib.suppress(Exception):
        return bool(env.sim.has_gui() or env.sim.has_rtx_sensors())
    return False


def _force_cuda_render_sync(env: ManagerBasedRLMimicEnv) -> None:
    """Force a render refresh for CUDA generation and mirror robot visuals to USD."""
    if not _cuda_visual_sync_enabled(env):
        return
    with contextlib.suppress(Exception):
        _sync_cuda_robot_visuals_to_usd(env)
    with contextlib.suppress(Exception):
        env.sim.render()


def _ensure_cuda_robot_visual_sync_initialized(
    env: ManagerBasedRLMimicEnv,
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


def _sync_cuda_robot_visuals_to_usd(env: ManagerBasedRLMimicEnv) -> None:
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


def _restore_cuda_cloth_visual_pose_to_initial(env: ManagerBasedRLMimicEnv) -> None:
    """Keep CUDA cloth visuals aligned with the initial particle-buffer frame after env.reset()."""
    if not _cuda_visual_sync_enabled(env):
        return
    obj = getattr(env, "object", None)
    if obj is None:
        return
    init_pos = getattr(obj, "init_pos", None)
    init_ori = getattr(obj, "init_ori", None)
    if init_pos is None or init_ori is None:
        return
    with contextlib.suppress(Exception):
        obj.set_world_pose(position=init_pos, orientation=init_ori)


async def run_data_generator_with_object_pose_failures(
    env: ManagerBasedRLMimicEnv,
    env_id: int,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    data_generator: DataGenerator,
    success_term: TerminationTermCfg,
    pause_subtask: bool = False,
    motion_planner: Any = None,
):
    """Run Mimic generation while treating cloth object-pose failures as failed trials."""
    while True:
        try:
            results = await data_generator.generate(
                env_id=env_id,
                success_term=success_term,
                env_reset_queue=env_reset_queue,
                env_action_queue=env_action_queue,
                pause_subtask=pause_subtask,
                motion_planner=motion_planner,
            )
        except (ClothObjectPoseUnavailableError, ClothObjectPoseValidationError) as exc:
            mimic_generation.num_failures += 1
            mimic_generation.num_attempts += 1
            print(
                f"Warning: generation trial for env {env_id} failed due to invalid cloth object poses: {exc}"
            )
            continue
        except Exception as exc:
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()
            raise exc

        if bool(results["success"]):
            mimic_generation.num_success += 1
        else:
            mimic_generation.num_failures += 1
        mimic_generation.num_attempts += 1


def setup_async_generation(
    env: Any,
    num_envs: int,
    input_file: str,
    success_term: Any,
    prefer_eef_pose_as_target: bool = False,
    source_target_z_offset: float = 0.0,
    align_object_pose_to_runtime: bool = False,
    align_object_pose_mode: str = "object_only",
    pause_subtask: bool = False,
    post_reset_settle_steps: int = 0,
    post_reset_hold_action: torch.Tensor | None = None,
    motion_planners: Any = None,
) -> dict[str, Any]:
    """Setup async generation with robust HDF5 datagen pool loading."""
    asyncio_event_loop = asyncio.get_event_loop()
    env_reset_queue = asyncio.Queue()
    env_action_queue = asyncio.Queue()
    shared_datagen_info_pool_lock = asyncio.Lock()
    shared_datagen_info_pool = RobustDataGenInfoPool(
        env,
        env.cfg,
        env.device,
        asyncio_lock=shared_datagen_info_pool_lock,
        prefer_eef_pose_as_target=prefer_eef_pose_as_target,
        source_target_z_offset=source_target_z_offset,
        align_object_pose_to_runtime=align_object_pose_to_runtime,
        align_object_pose_mode=align_object_pose_mode,
    )
    shared_datagen_info_pool.load_from_dataset_file(input_file)
    print(f"Loaded {shared_datagen_info_pool.num_datagen_infos} to datagen info pool")
    if shared_datagen_info_pool.invalid_episode_names:
        print(
            "Skipped invalid source episodes due to cloth object-pose validation: "
            f"{shared_datagen_info_pool.invalid_episode_names}"
        )
    if prefer_eef_pose_as_target:
        print("Using measured datagen_info.eef_pose as source target trajectory (override enabled).")
    if abs(source_target_z_offset) > 1e-9:
        print(f"Applying source target z offset: {source_target_z_offset:+.4f} m")
    if align_object_pose_to_runtime:
        print(f"Applying source object-pose runtime alignment in mode: {align_object_pose_mode}")

    data_generator = DataGenerator(
        env=env,
        src_demo_datagen_info_pool=shared_datagen_info_pool,
        post_reset_settle_steps=post_reset_settle_steps,
        post_reset_hold_action=post_reset_hold_action,
    )
    data_generator_asyncio_tasks = []
    for i in range(num_envs):
        env_motion_planner = motion_planners[i] if motion_planners else None
        task = asyncio_event_loop.create_task(
            run_data_generator_with_object_pose_failures(
                env,
                i,
                env_reset_queue,
                env_action_queue,
                data_generator,
                success_term,
                pause_subtask=pause_subtask,
                motion_planner=env_motion_planner,
            )
        )
        data_generator_asyncio_tasks.append(task)

    return {
        "tasks": data_generator_asyncio_tasks,
        "event_loop": asyncio_event_loop,
        "reset_queue": env_reset_queue,
        "action_queue": env_action_queue,
        "info_pool": shared_datagen_info_pool,
    }


def _evaluate_generation_success_result(env: ManagerBasedRLMimicEnv) -> dict[str, Any] | None:
    """Evaluate success using the same garment checker used by direct recording."""
    if (
        hasattr(env, "object")
        and env.object is not None
        and hasattr(env.object, "_cloth_prim_view")
        and hasattr(env, "garment_loader")
        and hasattr(env, "cfg")
        and hasattr(env.cfg, "garment_name")
    ):
        from lehome.utils.success_checker_chanllege import evaluate_garment_fold_success

        garment_type = env.garment_loader.get_garment_type(env.cfg.garment_name)
        return evaluate_garment_fold_success(env.object, garment_type)
    return None


def recording_style_success_tensor(env: ManagerBasedRLMimicEnv) -> torch.Tensor:
    """Return the same garment-only success signal used by direct recording."""
    result = _evaluate_generation_success_result(env)
    success = bool(result.get("success", False)) if result is not None else False
    return torch.full((int(env.num_envs),), success, dtype=torch.bool, device=env.device)


def _log_success_snapshot(env: ManagerBasedRLMimicEnv, row: dict[str, Any]) -> bool:
    """Log garment success using the same checker and thresholds as recording."""
    prefix = (
        f"[Generation][Episode {row.get('episode_index', 'N/A')}]"
        f"[step {row.get('episode_step', row.get('step', 'N/A'))}]"
    )
    result = _evaluate_generation_success_result(env)

    if result is None:
        logger.warning(f"{prefix} [Success Check] Success evaluation unavailable.")
        return False

    logger.info(
        f"{prefix} [Success Check] Garment type: {result.get('garment_type', 'unknown')}, "
        f"Thresholds: {result.get('thresholds', [])}"
    )
    details = result.get("details", {})
    for condition_info in details.values():
        status = "✓" if condition_info.get("passed", False) else "✗"
        logger.info(f"{prefix}   {condition_info.get('description', '')} -> {status}")

    success = bool(result.get("success", False))
    logger.info(f"{prefix} [Success Check] Final result: {'Success ✓' if success else 'Failed ✗'}")
    return success


def env_loop_with_pose_output(
    env: ManagerBasedRLMimicEnv,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    asyncio_event_loop: asyncio.AbstractEventLoop,
    output_file: str,
    pose_output_file: str | None,
    enable_pose_trace: bool = False,
    logging_interval: int = 1,
    log_success: bool = False,
) -> None:
    """Main async loop for generation with CSV logging and optional success logging."""
    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    prev_num_attempts = 0
    step_count = 0
    action_dim = int(env.single_action_space.shape[0])
    pose_writer: PoseTraceCsvWriter | None = None
    if enable_pose_trace:
        pose_output_path = _resolve_pose_output_path(output_file, pose_output_file)
        pose_writer = PoseTraceCsvWriter(pose_output_path)
        print(f"Pose trace CSV: {pose_output_path}")
    episode_indices = {env_id: -1 for env_id in range(int(env.num_envs))}
    episode_steps = {env_id: -1 for env_id in range(int(env.num_envs))}
    actions = torch.empty((env.num_envs, action_dim), device=env.device, dtype=torch.float32)

    try:
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            stashed_action_items: list[tuple[int, Any]] = []
            while True:
                actions.zero_()
                ready_env_ids: set[int] = set()
                consumed_action_items = 0

                while len(ready_env_ids) < env.num_envs:
                    if not env_reset_queue.empty():
                        item_type = "reset"
                        payload = env_reset_queue.get_nowait()
                        extra_action = None
                    elif stashed_action_items:
                        item_type = "action"
                        payload = stashed_action_items.pop(0)
                        extra_action = None
                    else:
                        item_type, payload, extra_action = asyncio_event_loop.run_until_complete(
                            _wait_for_next_generation_item(env_reset_queue, env_action_queue)
                        )
                        if extra_action is not None:
                            stashed_action_items.append(extra_action)

                    if item_type == "reset":
                        reset_env_id = int(payload)
                        stashed_action_items = [
                            item for item in stashed_action_items if int(item[0]) != reset_env_id
                        ]
                        env_id_tensor[0] = reset_env_id
                        env.reset(env_ids=env_id_tensor)
                        _restore_cuda_cloth_visual_pose_to_initial(env)
                        _force_cuda_render_sync(env)
                        ready_env_ids.discard(reset_env_id)
                        actions[reset_env_id].zero_()
                        episode_indices[reset_env_id] += 1
                        episode_steps[reset_env_id] = 0
                        env_reset_queue.task_done()
                        if reset_env_id == 0 and (enable_pose_trace or log_success):
                            row = _write_pose_snapshot(
                                env,
                                step_count=step_count,
                                env_id=0,
                                pose_writer=pose_writer,
                                episode_index=episode_indices[0],
                                episode_step=episode_steps[0],
                                completed_attempts=int(mimic_generation.num_attempts),
                                completed_successes=int(mimic_generation.num_success),
                            ) if enable_pose_trace else _build_pose_snapshot(
                                env,
                                step_count=step_count,
                                env_id=0,
                                episode_index=episode_indices[0],
                                episode_step=episode_steps[0],
                                completed_attempts=int(mimic_generation.num_attempts),
                                completed_successes=int(mimic_generation.num_success),
                            )
                            if log_success:
                                _log_success_snapshot(env, row)
                        continue

                    env_id, action = payload
                    action_tensor = _coerce_action_tensor(
                        action,
                        device=env.device,
                        dtype=torch.float32,
                        action_dim=action_dim,
                    )
                    actions[int(env_id)].copy_(action_tensor)
                    ready_env_ids.add(int(env_id))
                    consumed_action_items += 1

                env.step(actions)
                _force_cuda_render_sync(env)
                for _ in range(consumed_action_items):
                    env_action_queue.task_done()

                step_count += 1
                for env_id in range(int(env.num_envs)):
                    if episode_steps[env_id] >= 0:
                        episode_steps[env_id] += 1
                row: dict[str, Any] | None = None
                if enable_pose_trace and step_count % logging_interval == 0:
                    row = _write_pose_snapshot(
                        env,
                        step_count=step_count,
                        env_id=0,
                        pose_writer=pose_writer,
                        episode_index=episode_indices.get(0),
                        episode_step=episode_steps.get(0),
                        completed_attempts=int(mimic_generation.num_attempts),
                        completed_successes=int(mimic_generation.num_success),
                    )
                if log_success and step_count % SUCCESS_LOG_INTERVAL == 0:
                    if row is None:
                        row = _build_pose_snapshot(
                            env,
                            step_count=step_count,
                            env_id=0,
                            episode_index=episode_indices.get(0),
                            episode_step=episode_steps.get(0),
                            completed_attempts=int(mimic_generation.num_attempts),
                            completed_successes=int(mimic_generation.num_success),
                        )
                    _log_success_snapshot(env, row)

                if prev_num_attempts != mimic_generation.num_attempts:
                    prev_num_attempts = mimic_generation.num_attempts
                    generated_success_rate = (
                        100 * mimic_generation.num_success / mimic_generation.num_attempts
                        if mimic_generation.num_attempts > 0
                        else 0.0
                    )
                    print("")
                    print("*" * 50, "\033[K")
                    print(
                        f"{mimic_generation.num_success}/{mimic_generation.num_attempts}"
                        f" ({generated_success_rate:.1f}%) successful demos generated by mimic\033[K"
                    )
                    print("*" * 50, "\033[K")

                    generation_guarantee = env.cfg.datagen_config.generation_guarantee
                    generation_num_trials = env.cfg.datagen_config.generation_num_trials
                    check_val = (
                        mimic_generation.num_success
                        if generation_guarantee
                        else mimic_generation.num_attempts
                    )
                    if check_val >= generation_num_trials:
                        print(f"Reached {generation_num_trials} successes/attempts. Exiting.")
                        break

                if env.sim.is_stopped():
                    break
    finally:
        if pose_writer is not None:
            pose_writer.close()
        env.close()


__all__ = [
    "env_loop_with_pose_output",
    "recording_style_success_tensor",
    "setup_async_generation",
]
