from __future__ import annotations

import contextlib
import inspect
import random
from typing import Any
import gymnasium as gym
import numpy as np
import omni
import torch
import carb

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.datagen.generation import setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401
import lehome.tasks  # noqa: F401

from lehome.assets.robots.lerobot import (
    ACTION_NAMES,
    SO101_FOLLOWER_HOME_JOINT_POS,
    SO101_LEFT_ARM_HOME_JOINT_POS,
    SO101_RIGHT_ARM_HOME_JOINT_POS,
)
from lehome.tasks.fold_cloth.mdp.recorders import GenerationPoseRecorder

from .dataset_meta import get_first_demo_action_dim as _get_first_demo_action_dim
from .dataset_meta import get_first_demo_garment_name as _get_first_demo_garment_name
from .dataset_meta import get_first_demo_object_pose_keys as _get_first_demo_object_pose_keys
from .dataset_meta import get_source_actions_mode as _get_source_actions_mode
from .dataset_meta import load_dataset_env_args as _load_dataset_env_args
from .dataset_meta import should_auto_fix_mixed_pose_frames as _should_auto_fix_mixed_pose_frames
from .env_setup import apply_common_mimic_env_overrides
from .env_setup import assign_env_garment_metadata
from .env_setup import normalize_last_subtask_offsets_for_generation as _normalize_last_subtask_offsets_for_generation
from .env_setup import resolve_env_garment_metadata
from .env_setup import resolve_task_type as _resolve_task_type
from .annotated_record_service import _build_pose_sequence
from .generation_runtime import (
    env_loop_with_pose_output,
    recording_style_success_tensor,
    setup_async_generation,
)
from .generation_source import get_runtime_object_center as _get_runtime_object_center

try:
    import h5py
except ImportError:
    h5py = None

from lehome.tasks.fold_cloth.checkpoint_mappings import validate_semantic_object_pose_dict


SINGLE_ARM_SETTLE_ACTION = np.array(
    [SO101_FOLLOWER_HOME_JOINT_POS[action_name] for action_name in ACTION_NAMES],
    dtype=np.float32,
)
DUAL_ARM_SETTLE_ACTION = np.concatenate(
    [
        np.array(
            [SO101_LEFT_ARM_HOME_JOINT_POS[action_name] for action_name in ACTION_NAMES],
            dtype=np.float32,
        ),
        np.array(
            [SO101_RIGHT_ARM_HOME_JOINT_POS[action_name] for action_name in ACTION_NAMES],
            dtype=np.float32,
        ),
    ]
).astype(np.float32)

MIN_GENERATION_GARMENT_SETTLE_STEPS = 60


class PreStepCameraObservationsRecorder(RecorderTerm):
    """Record camera observations into the generated HDF5 obs group."""

    def record_pre_step(self):
        camera_obs: dict[str, torch.Tensor] = {}

        def _maybe_add(sensor_name: str, target_key: str) -> None:
            try:
                sensor = self._env.scene[sensor_name]
                rgb = sensor.data.output["rgb"]
            except Exception:
                return

            if rgb is None or rgb.ndim != 4:
                return
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            camera_obs[target_key] = rgb.clone()

        _maybe_add("top_camera", "top")
        _maybe_add("left_camera", "left_wrist")
        _maybe_add("right_camera", "right_wrist")
        _maybe_add("wrist_camera", "wrist")

        if not camera_obs:
            return None, None
        return "obs", camera_obs


class PreStepGenerationActionsRecorder(RecorderTerm):
    """Record generated top-level actions, preserving 16D ee-pose export in Pinocchio mode."""

    def record_pre_step(self):
        env = self._env
        export_actions = None
        with contextlib.suppress(Exception):
            if hasattr(env, "get_generation_export_actions"):
                export_actions = env.get_generation_export_actions()
        if export_actions is None:
            export_actions = env.action_manager.action
        if export_actions is None:
            return None, None
        return "actions", export_actions.clone()


@configclass
class PreStepCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for camera observation recording during generation export."""

    class_type: type[RecorderTerm] = PreStepCameraObservationsRecorder


@configclass
class PreStepGenerationActionsRecorderCfg(RecorderTermCfg):
    """Configuration for generation action export during HDF5 recording."""

    class_type: type[RecorderTerm] = PreStepGenerationActionsRecorder


@configclass
class PreStepGenerationPoseRecorderCfg(RecorderTermCfg):
    """Configuration for fold-cloth pose recording during generation export."""

    class_type: type[RecorderTerm] = GenerationPoseRecorder


@configclass
class GenerationRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Default action/state recorder plus generated pose and camera observations."""

    record_pre_step_actions = PreStepGenerationActionsRecorderCfg()
    record_pre_step_generation_pose = PreStepGenerationPoseRecorderCfg()
    record_pre_step_camera_observations = PreStepCameraObservationsRecorderCfg()


def _set_generated_actions_mode(env: ManagerBasedRLMimicEnv, actions_mode: str) -> None:
    """Stamp generated HDF5 outputs with an explicit action representation mode."""
    recorder_manager = getattr(env, "recorder_manager", None)
    if recorder_manager is None:
        return
    for attr_name in ("_dataset_file_handler", "_failed_episode_dataset_file_handler"):
        handler = getattr(recorder_manager, attr_name, None)
        data_group = getattr(handler, "_hdf5_data_group", None)
        if data_group is not None:
            data_group.attrs["actions_mode"] = actions_mode


def _build_post_reset_hold_action(env: ManagerBasedRLMimicEnv) -> torch.Tensor:
    """Build a safe per-env hold action used while garment cloth settles after reset."""
    if hasattr(env, "single_action_space") and hasattr(env.single_action_space, "shape"):
        action_dim = int(env.single_action_space.shape[0])
    else:
        action_dim = int(env.action_space.shape[-1])

    if action_dim == int(DUAL_ARM_SETTLE_ACTION.shape[0]):
        values = DUAL_ARM_SETTLE_ACTION
    elif action_dim == int(SINGLE_ARM_SETTLE_ACTION.shape[0]):
        values = SINGLE_ARM_SETTLE_ACTION
    else:
        values = np.zeros(action_dim, dtype=np.float32)

    return torch.tensor(values, dtype=torch.float32, device=env.device)


def _stabilize_after_initial_reset(
    env: ManagerBasedRLMimicEnv,
    hold_action: torch.Tensor,
    num_steps: int,
) -> None:
    """Let cloth settle once before preflight checks and generation startup."""
    if num_steps <= 0 or not hasattr(env, "object"):
        return

    batched_action = hold_action.reshape(1, -1).repeat(env.num_envs, 1)
    for _ in range(int(num_steps)):
        env.step(batched_action)


def _resolve_generation_garment_settle_steps(requested_steps: int) -> int:
    """Clamp generation settle steps to a cloth-safe minimum."""
    requested_steps = int(requested_steps)
    effective_steps = max(requested_steps, MIN_GENERATION_GARMENT_SETTLE_STEPS)
    if effective_steps != requested_steps:
        print(
            "Increasing garment_settle_steps from "
            f"{requested_steps} to {effective_steps} for generation cloth stability."
        )
    return effective_steps


def _validate_source_dataset_contract(
    input_file: str,
    expected_action_dim: int,
    required_actions_mode: str | None,
) -> None:
    """Validate source dataset action/datagen contract for strict generation."""
    action_dim = _get_first_demo_action_dim(input_file)
    if action_dim is None:
        raise ValueError("Strict preflight failed: unable to read top-level source action dimension.")
    if int(action_dim) != int(expected_action_dim):
        raise ValueError(
            "Strict preflight failed: unexpected source action dimension. "
            f"Expected {expected_action_dim}, found {action_dim}. "
            "Run scripts/mimicgen/eef_action_process.py --to_ik before annotation/generation."
        )

    required_mode = (required_actions_mode or "").strip()
    if required_mode:
        actions_mode = _get_source_actions_mode(input_file)
        if actions_mode is None:
            raise ValueError(
                "Strict preflight failed: source dataset has no /data attrs['actions_mode']. "
                f"Expected '{required_mode}'."
            )
        if str(actions_mode).strip().lower() != required_mode.lower():
            raise ValueError(
                "Strict preflight failed: unexpected source actions_mode. "
                f"Expected '{required_mode}', found '{actions_mode}'."
            )


def _validate_first_demo_datagen_contract(input_file: str) -> None:
    """Validate first demo datagen_info structure used by Mimic generation."""
    if h5py is None:
        raise ImportError("Strict preflight requires h5py.")

    with h5py.File(input_file, "r") as file:
        data_group = file.get("data")
        if data_group is None:
            raise ValueError("Strict preflight failed: missing /data group.")
        demo_names = sorted(
            [name for name in data_group.keys() if name.startswith("demo_")],
            key=lambda name: int(name.split("_", 1)[1]) if name.startswith("demo_") and name.split("_", 1)[1].isdigit() else name,
        )
        if not demo_names:
            raise ValueError("Strict preflight failed: no demo_* groups found.")

        demo_group = data_group[demo_names[0]]
        if "actions" not in demo_group:
            raise ValueError("Strict preflight failed: first demo missing top-level actions.")
        action_shape = tuple(np.asarray(demo_group["actions"]).shape)
        if len(action_shape) != 2:
            raise ValueError(
                f"Strict preflight failed: first demo actions must be [T,D], got {action_shape}."
            )
        num_steps = int(action_shape[0])

        obs_group = demo_group.get("obs")
        if obs_group is None:
            raise ValueError("Strict preflight failed: first demo missing /obs.")
        datagen_group = obs_group.get("datagen_info")
        if datagen_group is None:
            raise ValueError("Strict preflight failed: missing /obs/datagen_info.")

        for key in ("object_pose", "eef_pose", "target_eef_pose", "subtask_term_signals"):
            if key not in datagen_group:
                raise ValueError(f"Strict preflight failed: missing /obs/datagen_info/{key}.")

        for key in ("object_pose", "eef_pose", "target_eef_pose"):
            pose_group = datagen_group[key]
            if not isinstance(pose_group, h5py.Group) or len(pose_group.keys()) == 0:
                raise ValueError(f"Strict preflight failed: /obs/datagen_info/{key} is empty.")
            for pose_name in pose_group.keys():
                pose_shape = tuple(np.asarray(pose_group[pose_name]).shape)
                if len(pose_shape) != 3 or tuple(pose_shape[-2:]) != (4, 4):
                    raise ValueError(
                        "Strict preflight failed: invalid pose shape at "
                        f"/obs/datagen_info/{key}/{pose_name}: {pose_shape}."
                    )
                if int(pose_shape[0]) != num_steps:
                    raise ValueError(
                        "Strict preflight failed: horizon mismatch between actions and "
                        f"/obs/datagen_info/{key}/{pose_name} ({num_steps} vs {pose_shape[0]})."
                    )


def _validate_subtask_object_refs(env: Any, input_file: str) -> None:
    """Ensure all mimic subtask object_refs are available at runtime and in source datagen_info."""
    subtask_cfgs = getattr(env.cfg, "subtask_configs", {})
    if not isinstance(subtask_cfgs, dict):
        return

    required_refs = sorted(
        {
            cfg.object_ref
            for cfgs in subtask_cfgs.values()
            for cfg in cfgs
            if getattr(cfg, "object_ref", None) is not None
        }
    )
    if not required_refs:
        return

    runtime_keys: set[str] = set()
    try:
        runtime_object_poses = env.get_object_poses(env_ids=[0])
        if isinstance(runtime_object_poses, dict):
            validate_semantic_object_pose_dict(
                runtime_object_poses,
                context="generation runtime object_ref validation",
            )
            runtime_keys = set(runtime_object_poses.keys())
    except Exception as exc:
        raise ValueError(f"Failed to query runtime object poses for mimic validation: {exc}") from exc

    missing_runtime = [name for name in required_refs if name not in runtime_keys]
    if missing_runtime:
        raise ValueError(
            "Mimic object_ref validation failed at runtime. Missing refs in env.get_object_poses(): "
            f"{missing_runtime}. Available keys: {sorted(runtime_keys)}"
        )

    source_keys = _get_first_demo_object_pose_keys(input_file)
    if source_keys is not None:
        missing_source = [name for name in required_refs if name not in source_keys]
        if missing_source:
            raise ValueError(
                "Mimic object_ref validation failed in source dataset datagen_info.object_pose. "
                f"Missing refs: {missing_source}. Available keys: {sorted(source_keys)}"
            )

    print(f"Validated Mimic object_refs: {required_refs}")


def run_generation(parsed_args, simulation_app_instance) -> None:
    if bool(parsed_args.enable_pinocchio):
        with contextlib.suppress(Exception):
            import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

    if int(parsed_args.logging_interval) <= 0:
        raise ValueError("--logging_interval must be > 0.")

    num_envs = parsed_args.num_envs

    output_dir, output_file_name = setup_output_paths(parsed_args.output_file)
    task_name = parsed_args.task.split(":")[-1] if parsed_args.task else None
    dataset_env_args = _load_dataset_env_args(parsed_args.input_file)
    env_name = task_name or dataset_env_args.get("env_name")
    if env_name is None:
        env_name = get_env_name_from_dataset(parsed_args.input_file)

    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=parsed_args.device,
        generation_num_trials=parsed_args.generation_num_trials,
    )
    if str(parsed_args.device).startswith("cuda"):
        # CUDA cloth rendering needs the render delegate to consume live fabric transforms.
        env_cfg.sim.use_fabric = True
        carb_settings = carb.settings.get_settings()
        carb_settings.set_bool("/physics/fabricUpdateTransformations", True)
        carb_settings.set_bool("/physics/fabricUpdateVelocities", True)
        carb_settings.set_bool("/physics/fabricUpdateJointStates", True)
        print("Enabled fabric-backed transform sync for CUDA generation.")
    dataset_export_mode = env_cfg.recorders.dataset_export_mode
    env_cfg.recorders = GenerationRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = dataset_export_mode

    task_id = task_name or env_name
    setattr(env_cfg, "task_type", _resolve_task_type(task_id, parsed_args.task_type))
    setattr(env_cfg, "force_pinocchio_generation", bool(parsed_args.enable_pinocchio))
    apply_common_mimic_env_overrides(env_cfg, parsed_args)
    _normalize_last_subtask_offsets_for_generation(env_cfg)
    print(f"Using mimic IK orientation_weight={float(parsed_args.mimic_ik_orientation_weight):.4f}")

    if hasattr(env_cfg, "garment_name"):
        resolved_garment_name, resolved_garment_version = resolve_env_garment_metadata(
            env_cfg,
            parsed_args,
            dataset_env_args,
            fallback_garment_name=_get_first_demo_garment_name(parsed_args.input_file),
        )
        assign_env_garment_metadata(
            env_cfg,
            resolved_garment_name,
            resolved_garment_version,
            missing_error_message=(
                "This task requires a garment_name, but none was provided. "
                "Pass --garment_name (e.g., Top_Long_Unseen_0) or include garment_name in data/env_args or /data/demo_*/meta."
            ),
        )
        print(
            f"Using garment config: name={env_cfg.garment_name}, "
            f"version={getattr(env_cfg, 'garment_version', 'N/A')}"
        )

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    _set_generated_actions_mode(env, "ee_pose")

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if hasattr(env, "garment_loader") and hasattr(env, "cfg") and hasattr(env.cfg, "garment_name"):
        success_term = TerminationTermCfg(func=recording_style_success_tensor, params={}, time_out=False)
        print("Using recording-style garment success checker for generation.")

    force_pinocchio_generation = bool(parsed_args.enable_pinocchio)
    requires_env_ik_solver = force_pinocchio_generation
    if (
        not force_pinocchio_generation
        and hasattr(env, "_is_native_mimic_ik_action_contract")
        and hasattr(env, "action_manager")
        and hasattr(env.action_manager, "total_action_dim")
    ):
        try:
            requires_env_ik_solver = (
                not bool(env._is_native_mimic_ik_action_contract())
                and int(env.action_manager.total_action_dim) != 16
            )
        except Exception:
            requires_env_ik_solver = True

    if requires_env_ik_solver:
        if not hasattr(env, "_init_ik_solver_if_needed"):
            raise RuntimeError(
                "Generation requires a working IK solver in the environment, "
                "but this environment does not expose _init_ik_solver_if_needed()."
            )
        try:
            if not bool(env._init_ik_solver_if_needed()):
                raise RuntimeError("environment IK solver initialization returned False")
        except Exception as exc:
            raise RuntimeError(
                "Generation requires a working IK solver in the environment, "
                f"but initialization failed: {exc}"
            ) from exc
        if force_pinocchio_generation:
            print("Using Pinocchio pose->joint conversion for generation.")
    else:
        print("Using native env IK action contract for generation (no Pinocchio pose->joint conversion).")

    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # If --pose_sequence is set, monkey-patch the garment's _sample_reset_pose
    # so every reset pulls the current Halton pose.  Advance happens once per
    # successful episode (see run_data_generator_with_object_pose_failures).
    pose_sequence = _build_pose_sequence(parsed_args, env)
    if pose_sequence is not None:
        # The Halton sequence is authoritative over the trial count: generate
        # exactly as many successful demos as there are poses in the sequence.
        env.cfg.datagen_config.generation_num_trials = pose_sequence.total
        pose_sequence.log_status()

    env.reset()
    if hasattr(env, "initialize_obs"):
        try:
            env.initialize_obs()
        except Exception as exc:
            print(f"Warning: initialize_obs failed during generation reset: {exc}")
    garment_settle_steps = _resolve_generation_garment_settle_steps(parsed_args.garment_settle_steps)
    post_reset_hold_action = _build_post_reset_hold_action(env)
    _stabilize_after_initial_reset(
        env,
        hold_action=post_reset_hold_action,
        num_steps=garment_settle_steps,
    )

    if bool(parsed_args.strict_preflight):
        _validate_source_dataset_contract(
            parsed_args.input_file,
            expected_action_dim=int(parsed_args.expected_source_action_dim),
            required_actions_mode=str(parsed_args.require_source_actions_mode),
        )
        _validate_first_demo_datagen_contract(parsed_args.input_file)
    _validate_subtask_object_refs(env, parsed_args.input_file)

    explicit_object_alignment = bool(parsed_args.align_object_pose_to_runtime) and (
        not parsed_args.disable_object_pose_alignment
    )
    object_alignment_mode = str(parsed_args.object_pose_alignment_mode).strip().lower()
    auto_object_alignment = False
    if (not explicit_object_alignment) and bool(parsed_args.auto_fix_mixed_pose_frames):
        runtime_object_center = _get_runtime_object_center(env)
        should_fix, frame_stats = _should_auto_fix_mixed_pose_frames(
            parsed_args.input_file,
            runtime_object_center,
        )
        if should_fix:
            auto_object_alignment = True
            object_alignment_mode = "object_only"
            print(
                "Detected mixed source pose frames "
                f"(src_obj_z={frame_stats['src_object_center_z']:+.4f}, "
                f"src_target_z={frame_stats['src_target_center_z']:+.4f}, "
                f"runtime_obj_z={frame_stats['runtime_object_center_z']:+.4f}, "
                f"src_target_obj_z_gap={frame_stats['src_target_object_z_gap']:+.4f}). "
                "Auto-enabling object-only source object alignment."
            )

    use_pose_overrides = (
        bool(parsed_args.use_eef_pose_as_target)
        or abs(float(parsed_args.source_target_z_offset)) > 1e-9
        or explicit_object_alignment
        or auto_object_alignment
    )
    if use_pose_overrides:
        print("Using compatibility datagen pipeline with source pose overrides.")
    else:
        print("Using compatibility datagen pipeline with strict cloth object-pose validation.")

    async_components = setup_async_generation(
        env=env,
        num_envs=parsed_args.num_envs,
        input_file=parsed_args.input_file,
        success_term=success_term,
        prefer_eef_pose_as_target=bool(parsed_args.use_eef_pose_as_target),
        source_target_z_offset=parsed_args.source_target_z_offset,
        align_object_pose_to_runtime=(explicit_object_alignment or auto_object_alignment),
        align_object_pose_mode=object_alignment_mode,
        pause_subtask=parsed_args.pause_subtask,
        log_success=bool(parsed_args.log_success),
        post_reset_settle_steps=garment_settle_steps,
        post_reset_hold_action=post_reset_hold_action,
        save_failed_full=bool(getattr(parsed_args, "save_failed", False)),
        pose_sequence=pose_sequence,
    )

    import isaaclab_mimic.datagen.generation as mimic_generation

    mimic_generation.num_success = 0
    mimic_generation.num_failures = 0
    mimic_generation.num_attempts = 0
    enable_pose_trace = bool(parsed_args.save_pose_trace or parsed_args.pose_output_file)
    env_loop_with_pose_output(
        env,
        async_components["reset_queue"],
        async_components["action_queue"],
        async_components["event_loop"],
        output_file=parsed_args.output_file,
        pose_output_file=parsed_args.pose_output_file,
        enable_pose_trace=enable_pose_trace,
        logging_interval=int(parsed_args.logging_interval),
        log_success=bool(parsed_args.log_success),
        worker_tasks=async_components["tasks"],
        pose_sequence=pose_sequence,
    )


__all__ = ["run_generation"]
