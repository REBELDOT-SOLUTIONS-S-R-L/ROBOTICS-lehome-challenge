"""Runtime helpers for MimicGen demo annotation."""

from __future__ import annotations

import contextlib
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.datasets import EpisodeData

from lehome.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

from .annotation_session import AnnotationSessionController
from .data_utils import as_2d_tensor, as_tensor, flatten_nested_leaves
from .env_runtime import ensure_ik_solver_ready
from .robot_utils import are_arms_at_rest, arm_root_pose_world, decode_ik_action_trajectory

try:
    from scripts.utils.annotate_utils import (
        DatagenObjectPoseCaptureError,
        ReplayPlan,
        ReplayRuntimeContext,
    )
except ImportError:
    scripts_dir = Path(__file__).resolve().parents[2]
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    from utils.annotate_utils import (
        DatagenObjectPoseCaptureError,
        ReplayPlan,
        ReplayRuntimeContext,
    )


def normalize_manual_subtask_signal_name(signal_name: str | None, eef_name: str, subtask_index: int) -> str:
    """Return a display/store-safe signal name for manual annotation."""
    if isinstance(signal_name, str) and signal_name.strip():
        return signal_name
    return f"{eef_name}_subtask_{subtask_index}_complete"


def stabilize_garment_after_reset_for_annotation(env: ManagerBasedRLMimicEnv, num_steps: int = 20) -> None:
    """Run a short physics-only settle phase without adding recorder entries."""
    if num_steps <= 0:
        return

    try:
        left_joint_pos = env.scene["left_arm"].data.joint_pos
        right_joint_pos = env.scene["right_arm"].data.joint_pos
        hold_action = torch.cat([left_joint_pos, right_joint_pos], dim=-1).to(env.device)
    except Exception:
        return

    try:
        env.action_manager.process_action(hold_action)
    except Exception:
        pass

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    render_interval = max(1, int(getattr(env.cfg.sim, "render_interval", 1)))
    for step_index in range(num_steps):
        with contextlib.suppress(Exception):
            env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if is_rendering and (step_index + 1) % render_interval == 0:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)


def _rebuild_initial_state_from_episode(episode: EpisodeData) -> dict:
    """Build a robust initial_state, filling arm joints from obs if needed."""
    initial_state = deepcopy(episode.data.get("initial_state", {}))

    if "articulation" not in initial_state:
        initial_state["articulation"] = {}

    obs = episode.data.get("obs", {})
    actions = episode.data.get("actions", None)
    left_obs = obs.get("left_joint_pos", None)
    right_obs = obs.get("right_joint_pos", None)
    left_action_fallback = None
    right_action_fallback = None
    if actions is not None and getattr(actions, "ndim", 0) >= 2 and actions.shape[0] > 0:
        if actions.shape[-1] >= 6:
            left_action_fallback = actions[0:1, :6].clone()
        if actions.shape[-1] >= 12:
            right_action_fallback = actions[0:1, 6:12].clone()

    def _select_initial_joint_row(obs_tensor, fallback_tensor):
        if obs_tensor is None or getattr(obs_tensor, "ndim", 0) < 2 or obs_tensor.shape[0] == 0:
            return fallback_tensor
        norms = torch.sum(torch.abs(obs_tensor), dim=-1)
        nz = torch.nonzero(norms > 1e-6, as_tuple=False)
        if nz.numel() > 0:
            idx = int(nz[0, 0].item())
            return obs_tensor[idx : idx + 1].clone()
        return obs_tensor[0:1].clone() if fallback_tensor is None else fallback_tensor

    def _ensure_arm_state(arm_name: str, obs_tensor, fallback_tensor):
        obs_first = _select_initial_joint_row(obs_tensor, fallback_tensor)
        if obs_first is None:
            return

        arm_state = initial_state["articulation"].get(arm_name, {})
        if not isinstance(arm_state, dict):
            arm_state = {}

        jp = arm_state.get("joint_position", None)
        needs_fill = (
            jp is None
            or (hasattr(jp, "numel") and jp.numel() == 0)
            or (hasattr(jp, "shape") and jp.shape[-1] != obs_first.shape[-1])
            or (hasattr(jp, "abs") and float(jp.abs().sum().item()) < 1e-6)
        )
        if needs_fill:
            arm_state["joint_position"] = obs_first

        if "joint_velocity" not in arm_state or arm_state["joint_velocity"] is None:
            arm_state["joint_velocity"] = torch.zeros_like(obs_first)
        if "root_pose" not in arm_state or arm_state["root_pose"] is None:
            root_pose = torch.zeros((1, 7), dtype=obs_first.dtype, device=obs_first.device)
            root_pose[:, 3] = 1.0
            arm_state["root_pose"] = root_pose
        if "root_velocity" not in arm_state or arm_state["root_velocity"] is None:
            arm_state["root_velocity"] = torch.zeros((1, 6), dtype=obs_first.dtype, device=obs_first.device)

        initial_state["articulation"][arm_name] = arm_state

    _ensure_arm_state("left_arm", left_obs, left_action_fallback)
    _ensure_arm_state("right_arm", right_obs, right_action_fallback)
    return initial_state


def _set_arm_joint_state_from_initial_state(env: ManagerBasedRLMimicEnv, initial_state: dict) -> None:
    """Apply arm joint states from initial_state directly to sim."""
    articulation_state = initial_state.get("articulation", {})
    for arm_name in ("left_arm", "right_arm"):
        try:
            arm = env.scene[arm_name]
        except Exception:
            continue
        arm_state = articulation_state.get(arm_name, {})
        if not isinstance(arm_state, dict):
            continue

        joint_pos = arm_state.get("joint_position", None)
        if joint_pos is not None and getattr(joint_pos, "ndim", 0) == 2 and joint_pos.shape[0] > 0:
            arm.write_joint_position_to_sim(joint_pos[:1], env_ids=None)

        joint_vel = arm_state.get("joint_velocity", None)
        if joint_vel is not None and getattr(joint_vel, "ndim", 0) == 2 and joint_vel.shape[0] > 0:
            arm.write_joint_velocity_to_sim(joint_vel[:1], env_ids=None)


def _get_replay_actions_for_episode(episode: EpisodeData, expected_action_dim: int | None = None) -> torch.Tensor:
    """Choose the best action trajectory for replay, matching env action dimension when possible."""
    obs = episode.data.get("obs", {})
    actions = as_2d_tensor(episode.data.get("actions", None))
    processed_actions = as_2d_tensor(episode.data.get("processed_actions", None))
    obs_actions = as_2d_tensor(obs.get("actions", None))

    left_obs = as_2d_tensor(obs.get("left_joint_pos", None))
    right_obs = as_2d_tensor(obs.get("right_joint_pos", None))
    concat_joint_actions = None
    if left_obs is not None and right_obs is not None:
        if left_obs.shape[0] == right_obs.shape[0] and left_obs.shape[1] >= 6 and right_obs.shape[1] >= 6:
            concat_joint_actions = torch.cat([left_obs[:, :6], right_obs[:, :6]], dim=-1)

    if actions is None:
        raise ValueError("Episode does not contain 'actions' for replay.")

    if expected_action_dim is None or actions.shape[1] == expected_action_dim:
        replay_actions = actions
        source_name = "actions"
    else:
        replay_actions = None
        source_name = None
        for name, candidate in (
            ("obs/actions", obs_actions),
            ("processed_actions", processed_actions),
            ("obs/left+right_joint_pos", concat_joint_actions),
        ):
            if candidate is not None and candidate.shape[1] == expected_action_dim:
                replay_actions = candidate
                source_name = name
                break

        if replay_actions is None:
            available_shapes = []
            for name, candidate in (
                ("actions", actions),
                ("obs/actions", obs_actions),
                ("processed_actions", processed_actions),
                ("obs/left+right_joint_pos", concat_joint_actions),
            ):
                if candidate is not None:
                    available_shapes.append(f"{name}={tuple(candidate.shape)}")
            raise ValueError(
                f"Could not find replay actions with expected dimension {expected_action_dim}. "
                f"Available: {', '.join(available_shapes)}"
            )

        print(
            f"\tReplay fallback: using {source_name} with shape {tuple(replay_actions.shape)} "
            f"instead of actions with shape {tuple(actions.shape)}."
        )

    actions_std = float(actions.float().std().item()) if actions.numel() > 0 else 0.0
    replay_std = float(replay_actions.float().std().item()) if replay_actions.numel() > 0 else 0.0
    if actions_std < 1e-3 and replay_std > 1e-2 and source_name != "actions":
        print(
            f"\tReplay note: source actions are near-constant (std={actions_std:.6f}), "
            f"selected replay stream std={replay_std:.6f}."
        )

    return replay_actions
def _first_pose_translation_mean(pose_dict: dict[str, torch.Tensor], max_steps: int = 128) -> torch.Tensor | None:
    """Compute mean xyz over first timesteps from a dict of [T,4,4] tensors."""
    if not isinstance(pose_dict, dict) or len(pose_dict) == 0:
        return None
    xyz_list = []
    for pose in pose_dict.values():
        pose_t = as_tensor(pose, squeeze_second_dim=True)
        if pose_t is None:
            continue
        if pose_t.ndim != 3 or tuple(pose_t.shape[-2:]) != (4, 4):
            continue
        n = min(int(max_steps), int(pose_t.shape[0]))
        xyz_list.append(pose_t[:n, :3, 3].reshape(-1, 3))
    if not xyz_list:
        return None
    return torch.cat(xyz_list, dim=0).mean(dim=0)


def validate_recorded_datagen_pose_contract(
    annotated_episode: EpisodeData,
    z_gap_threshold: float,
) -> None:
    """Fail fast if recorder-exported datagen_info violates strict frame sanity checks."""
    obs = annotated_episode.data.get("obs", {})
    datagen = obs.get("datagen_info", {})
    if not isinstance(datagen, dict):
        raise ValueError("Annotated episode missing obs/datagen_info.")

    required = ("object_pose", "eef_pose", "target_eef_pose")
    for key in required:
        if key not in datagen:
            raise ValueError(f"Annotated episode missing obs/datagen_info/{key}.")

    for section_name in required:
        pose_dict = datagen[section_name]
        if not isinstance(pose_dict, dict) or len(pose_dict) == 0:
            raise ValueError(f"obs/datagen_info/{section_name} is empty or invalid.")
        for pose_name, pose_value in pose_dict.items():
            pose = as_tensor(pose_value, squeeze_second_dim=True)
            if pose is None:
                raise ValueError(
                    f"Invalid pose value for {section_name}/{pose_name}: unsupported type {type(pose_value)}."
                )
            if pose.ndim != 3 or tuple(pose.shape[-2:]) != (4, 4):
                raise ValueError(
                    f"Invalid pose shape for {section_name}/{pose_name}: expected [T,4,4], got {tuple(pose.shape)}."
                )
            last_row = pose[..., 3, :]
            target_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=last_row.device, dtype=last_row.dtype)
            max_err = float((last_row - target_row).abs().max().item())
            if max_err > 1e-3:
                raise ValueError(
                    f"Invalid homogeneous row in {section_name}/{pose_name}: max error={max_err:.6f}."
                )

    obj_center = _first_pose_translation_mean(datagen["object_pose"])
    tgt_center = _first_pose_translation_mean(datagen["target_eef_pose"])
    if obj_center is None or tgt_center is None:
        raise ValueError("Failed to compute strict datagen pose centers.")
    z_gap = float(abs(tgt_center[2] - obj_center[2]).item())
    if z_gap > float(z_gap_threshold):
        raise ValueError(
            "Strict datagen pose frame check failed: "
            f"mean |target_z - object_z|={z_gap:.4f} exceeds threshold {float(z_gap_threshold):.4f}."
        )


def _get_episode_initial_pose(garment_info: dict | None, episode_index: int) -> dict | None:
    """Extract {'Garment': [...]} pose for the given episode index."""
    if not garment_info:
        return None
    episode_key = str(episode_index)
    for _, episodes in garment_info.items():
        if isinstance(episodes, dict) and episode_key in episodes:
            pose = episodes[episode_key].get("object_initial_pose")
            if pose is not None:
                return {"Garment": pose}
    return None


def _infer_ik_action_frame(
    actions: torch.Tensor,
    eef_names: list[str],
    frame_hint: str | None,
    base_z_threshold: float,
) -> str:
    """Infer IK action frame from metadata hint or pose statistics."""
    if frame_hint in {"base", "world"}:
        return frame_hint

    if actions.ndim != 2 or actions.shape[0] == 0:
        return "world"

    sample = actions[: min(32, int(actions.shape[0]))]
    sample_dim = int(sample.shape[1])
    if len(eef_names) >= 2 and sample_dim >= 16:
        z_values = torch.cat([sample[:, 2], sample[:, 10]], dim=0)
    elif sample_dim >= 8:
        z_values = sample[:, 2]
    else:
        return "world"

    mean_z = float(z_values.float().mean().item())
    return "base" if mean_z < float(base_z_threshold) else "world"
def merge_source_obs_into_annotated_episode(source_episode: EpisodeData, annotated_episode: EpisodeData) -> None:
    """Preserve non-datagen observation keys from source episode in annotated export."""
    source_obs = source_episode.data.get("obs", None)
    if not isinstance(source_obs, dict):
        return

    annotated_obs = annotated_episode.data.get("obs", {})
    existing_paths = {f"obs/{path}" for path in flatten_nested_leaves(annotated_obs).keys()}
    source_leaves = flatten_nested_leaves(source_obs, skip_root_keys={"datagen_info"})

    for rel_path, value in source_leaves.items():
        full_path = f"obs/{rel_path}"
        if full_path in existing_paths:
            continue

        if isinstance(value, torch.Tensor):
            annotated_episode.add(full_path, value.clone())
        elif isinstance(value, np.ndarray):
            annotated_episode.add(full_path, torch.from_numpy(value.copy()))
        elif isinstance(value, (int, float, bool)):
            annotated_episode.add(full_path, torch.tensor(value))


def overwrite_annotated_actions_with_source_actions(
    source_episode: EpisodeData,
    annotated_episode: EpisodeData,
) -> None:
    """Keep top-level actions from source episode (strict IK pipeline contract)."""
    source_actions = as_2d_tensor(source_episode.data.get("actions", None))
    if source_actions is None:
        raise ValueError("Source episode does not contain top-level actions.")

    recorded_actions = as_2d_tensor(annotated_episode.data.get("actions", None))
    if recorded_actions is not None and recorded_actions.shape[0] != source_actions.shape[0]:
        raise ValueError(
            "Cannot preserve source actions: horizon mismatch between source and recorder output "
            f"({source_actions.shape[0]} vs {recorded_actions.shape[0]})."
        )

    annotated_episode.data["actions"] = source_actions.clone()


def build_replay_runtime_context(
    env: ManagerBasedRLMimicEnv,
    args,
) -> ReplayRuntimeContext:
    """Resolve replay configuration from the environment once."""
    expected_action_dim = None
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "total_action_dim"):
        expected_action_dim = int(env.action_manager.total_action_dim)
    elif hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
        shape = getattr(env.action_space, "shape", ())
        if len(shape) > 0:
            expected_action_dim = int(shape[0])

    eef_names = list(getattr(env.cfg, "subtask_configs", {}).keys())
    if not eef_names:
        eef_names = ["left_arm", "right_arm"]
    if set(eef_names) == {"left_arm", "right_arm"}:
        eef_names = ["left_arm", "right_arm"]

    native_ik_action_contract = False
    if hasattr(env, "_is_native_mimic_ik_action_contract"):
        try:
            native_ik_action_contract = bool(env._is_native_mimic_ik_action_contract())
        except Exception:
            native_ik_action_contract = False
    if (not native_ik_action_contract) and expected_action_dim is not None:
        native_ik_action_contract = int(expected_action_dim) == int(args.ik_action_dim)

    return ReplayRuntimeContext(
        expected_action_dim=expected_action_dim,
        eef_names=eef_names,
        native_ik_action_contract=native_ik_action_contract,
    )
def prepare_replay_plan(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    replay_runtime: ReplayRuntimeContext,
    args,
    *,
    episode_index: int | None = None,
    garment_info: dict | None = None,
    frame_hint: str | None = None,
) -> ReplayPlan:
    """Build replay inputs once per episode and reuse them across replay attempts."""
    initial_state = _rebuild_initial_state_from_episode(episode)
    source_actions = as_2d_tensor(episode.data.get("actions", None))
    if source_actions is None:
        raise ValueError("Episode does not contain top-level actions for replay.")

    replay_mode = "joint"
    replay_actions = source_actions
    expected_action_dim = replay_runtime.expected_action_dim
    if bool(args.require_ik_actions):
        if int(source_actions.shape[1]) != int(args.ik_action_dim):
            message = (
                "Strict IK replay is enabled, but source actions dimension does not match "
                f"--ik_action_dim ({source_actions.shape[1]} vs {args.ik_action_dim})."
            )
            if not bool(args.legacy_joint_replay):
                raise ValueError(message)
            print(f"\tWarning: {message} Falling back to legacy joint replay.")
            replay_actions = _get_replay_actions_for_episode(episode, expected_action_dim=expected_action_dim)
            replay_mode = "joint"
        else:
            replay_mode = "ik"
    else:
        if int(source_actions.shape[1]) == int(args.ik_action_dim):
            replay_mode = "ik"
        else:
            replay_actions = _get_replay_actions_for_episode(episode, expected_action_dim=expected_action_dim)
            replay_mode = "joint"

    replay_actions = replay_actions.to(device=env.device, dtype=torch.float32)
    garment_initial_pose = _get_episode_initial_pose(garment_info, episode_index)
    if replay_mode != "ik":
        return ReplayPlan(
            initial_state=initial_state,
            replay_actions=replay_actions,
            replay_mode="joint",
            ik_frame=None,
            seed=int(getattr(episode, "seed", 0)),
            episode_index=episode_index,
            garment_initial_pose=garment_initial_pose,
        )

    ik_frame = str(args.ik_action_frame).strip().lower()
    if ik_frame == "auto":
        ik_frame = _infer_ik_action_frame(
            replay_actions,
            eef_names=replay_runtime.eef_names,
            frame_hint=frame_hint,
            base_z_threshold=float(args.ik_auto_base_z_threshold),
        )
    if ik_frame not in {"base", "world"}:
        raise ValueError(f"Invalid IK action frame resolved: {ik_frame}")

    ensure_ik_solver_ready(env, replay_runtime)
    ik_pose_by_eef, ik_gripper_by_eef = decode_ik_action_trajectory(
        replay_actions,
        eef_names=replay_runtime.eef_names,
        quat_order=str(args.ik_quat_order),
        device=env.device,
    )
    return ReplayPlan(
        initial_state=initial_state,
        replay_actions=replay_actions,
        replay_mode="ik",
        ik_frame=ik_frame,
        seed=int(getattr(episode, "seed", 0)),
        episode_index=episode_index,
        garment_initial_pose=garment_initial_pose,
        ik_pose_by_eef=ik_pose_by_eef,
        ik_gripper_by_eef=ik_gripper_by_eef,
    )


def _print_success_debug_breakdown(env: ManagerBasedRLMimicEnv) -> None:
    """Print a breakdown of success components (fold + both arms at rest)."""
    try:
        fold_success = None
        if hasattr(env, "_get_success"):
            fold_success_tensor = env._get_success()
            fold_success = bool(fold_success_tensor[0].item())

        left_arm = env.scene["left_arm"]
        right_arm = env.scene["right_arm"]
        left_at_rest = bool(is_so101_at_rest_pose(left_arm.data.joint_pos, left_arm.data.joint_names)[0].item())
        right_at_rest = bool(is_so101_at_rest_pose(right_arm.data.joint_pos, right_arm.data.joint_names)[0].item())

        print(
            "\tSuccess breakdown:"
            f" fold_success={fold_success}, left_at_rest={left_at_rest}, right_at_rest={right_at_rest}"
        )

        def _report_joint_out_of_range(arm_name: str, arm) -> None:
            out_of_range = []
            joint_pos = arm.data.joint_pos[0].detach().cpu()
            for i, joint_name in enumerate(arm.data.joint_names):
                if joint_name not in SO101_FOLLOWER_REST_POSE_RANGE:
                    continue
                low_deg, high_deg = SO101_FOLLOWER_REST_POSE_RANGE[joint_name]
                low_rad = math.radians(low_deg)
                high_rad = math.radians(high_deg)
                value = float(joint_pos[i].item())
                if value < low_rad or value > high_rad:
                    out_of_range.append(
                        f"{joint_name}={math.degrees(value):.1f}deg (range {low_deg:.1f}..{high_deg:.1f})"
                    )
            if out_of_range:
                print(f"\t{arm_name} joints out of rest range: {', '.join(out_of_range)}")

        _report_joint_out_of_range("left_arm", left_arm)
        _report_joint_out_of_range("right_arm", right_arm)
    except Exception as exc:
        print(f"\tWarning: failed to compute success breakdown: {exc}")
def recover_from_object_pose_capture_failure(
    env: ManagerBasedRLMimicEnv,
    session: AnnotationSessionController,
) -> None:
    """Clear replay state and reset the environment before retrying an annotation attempt."""
    session.reset_attempt_state()
    with contextlib.suppress(Exception):
        env.recorder_manager.reset()
    with contextlib.suppress(Exception):
        env.reset()
        if hasattr(env, "initialize_obs"):
            env.initialize_obs()
            stabilize_garment_after_reset_for_annotation(env)


def replay_episode(
    env: ManagerBasedRLMimicEnv,
    replay_plan: ReplayPlan,
    replay_runtime: ReplayRuntimeContext,
    session: AnnotationSessionController,
    args,
    task_type: str | None,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    """Replay a prepared episode plan in the annotation environment."""
    actions = replay_plan.replay_actions
    replay_mode = replay_plan.replay_mode

    if replay_mode == "ik":
        print(
            "\tReplay mode: IK actions "
            f"(dim={actions.shape[1]}, quat_order={args.ik_quat_order}, frame={replay_plan.ik_frame})."
        )
        if replay_runtime.native_ik_action_contract:
            print("\tIK replay will use native env IK action contract (no Pinocchio pose->joint conversion).")
    else:
        print(f"\tReplay mode: joint actions (dim={actions.shape[1]}).")

    if replay_plan.seed is not None:
        env.seed(int(replay_plan.seed))
    is_cloth_env = hasattr(env, "object")
    if not is_cloth_env:
        env.sim.reset()
        with contextlib.suppress(Exception):
            env.sim.play()
    env.recorder_manager.reset()
    env.reset()
    with contextlib.suppress(Exception):
        env.sim.play()
    if hasattr(env, "initialize_obs"):
        try:
            env.initialize_obs()
        except Exception as exc:
            print(f"Warning: initialize_obs failed during replay reset: {exc}")
    _set_arm_joint_state_from_initial_state(env, replay_plan.initial_state)
    if replay_plan.episode_index is not None and hasattr(env, "set_all_pose") and replay_plan.garment_initial_pose is not None:
        try:
            env.set_all_pose(replay_plan.garment_initial_pose)
        except Exception as exc:
            print(f"Warning: failed to set initial garment pose for episode {replay_plan.episode_index}: {exc}")
    env.scene.write_data_to_sim()
    env.sim.forward()
    with contextlib.suppress(Exception):
        env.sim.play()
    stabilize_garment_after_reset_for_annotation(env)

    replay_pose_by_eef = None
    if replay_mode == "ik":
        replay_pose_by_eef = replay_plan.ik_pose_by_eef
        if replay_plan.ik_frame == "base":
            replay_pose_by_eef = {}
            for eef_name in replay_runtime.eef_names:
                base_in_world = arm_root_pose_world(env, eef_name).unsqueeze(0)
                replay_pose_by_eef[eef_name] = torch.matmul(base_in_world, replay_plan.ik_pose_by_eef[eef_name])

    first_action = True
    fold_success_seen = False
    step_period = (1.0 / args.step_hz) if args.step_hz and args.step_hz > 0 else 0.0
    for action_index, action in enumerate(actions):
        step_start_time = time.perf_counter()
        session.set_current_action_index(action_index)
        if first_action:
            first_action = False
        else:
            while session.state.paused or session.state.skip_episode:
                env.sim.render()
                if session.state.skip_episode:
                    return False
                continue
        with contextlib.suppress(Exception):
            if env.sim.is_stopped():
                env.sim.play()

        if replay_mode == "ik":
            target_eef_pose_dict = {
                eef_name: replay_pose_by_eef[eef_name][action_index : action_index + 1]
                for eef_name in replay_runtime.eef_names
            }
            gripper_action_dict = {
                eef_name: replay_plan.ik_gripper_by_eef[eef_name][action_index].reshape(1)
                for eef_name in replay_runtime.eef_names
            }
            action_tensor = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_eef_pose_dict,
                gripper_action_dict=gripper_action_dict,
                action_noise_dict=None,
                env_id=0,
            )
            action_tensor = torch.as_tensor(action_tensor, dtype=torch.float32, device=env.device).reshape(1, -1)
            if replay_runtime.expected_action_dim is not None and int(action_tensor.shape[1]) != int(replay_runtime.expected_action_dim):
                raise ValueError(
                    "Environment action conversion produced wrong dimension: "
                    f"expected {replay_runtime.expected_action_dim}, got {action_tensor.shape[1]}."
                )
        else:
            action_tensor = action.reshape(1, -1)

        if getattr(env.cfg, "dynamic_reset_gripper_effort_limit", False):
            dynamic_reset_gripper_effort_limit_sim(env, task_type)
        env.step(action_tensor)
        if hasattr(env, "_get_success"):
            try:
                fold_success_seen = fold_success_seen or bool(env._get_success()[0].item())
            except Exception:
                pass
        if env.sim.has_gui() or env.sim.has_rtx_sensors():
            env.sim.render()
        if step_period > 0.0:
            elapsed = time.perf_counter() - step_start_time
            if elapsed < step_period:
                time.sleep(step_period - elapsed)

    if success_term is not None:
        if getattr(success_term.func, "__name__", "") == "garment_folded":
            arms_rest_final = are_arms_at_rest(env)
            if not (fold_success_seen and arms_rest_final):
                _print_success_debug_breakdown(env)
                print(
                    "\tReplay success check:"
                    f" fold_success_seen={fold_success_seen}, arms_rest_final={arms_rest_final}"
                )
                return False
        else:
            if not bool(success_term.func(env, **success_term.params)[0]):
                _print_success_debug_breakdown(env)
                return False
    return True


def _finalize_annotated_episode(
    source_episode: EpisodeData,
    annotated_episode: EpisodeData,
    args,
) -> None:
    merge_source_obs_into_annotated_episode(source_episode, annotated_episode)
    if bool(args.require_ik_actions):
        overwrite_annotated_actions_with_source_actions(source_episode, annotated_episode)
        validate_recorded_datagen_pose_contract(
            annotated_episode,
            z_gap_threshold=float(args.strict_pose_z_gap_threshold),
        )


def annotate_episode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    replay_plan: ReplayPlan,
    replay_runtime: ReplayRuntimeContext,
    session: AnnotationSessionController,
    args,
    auto: bool,
    task_type: str | None,
    success_term: TerminationTermCfg | None = None,
    subtask_term_signal_names: dict[str, list[str]] | None = None,
) -> bool:
    """Replay one episode and annotate subtasks in auto or manual mode."""
    session.state.skip_episode = False
    if auto:
        is_episode_annotated_successfully = replay_episode(
            env,
            replay_plan,
            replay_runtime,
            session,
            args,
            task_type,
            success_term,
        )
        if session.state.skip_episode:
            print("\tSkipping the episode.")
            return False
        if not is_episode_annotated_successfully:
            print("\tThe final task was not completed.")
            if not args.ignore_replay_success:
                return False
            print("\tContinuing because --ignore_replay_success is enabled.")
            is_episode_annotated_successfully = True

        if is_episode_annotated_successfully:
            annotated_episode = env.recorder_manager.get_episode(0)
            _finalize_annotated_episode(episode, annotated_episode, args)
            subtask_term_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_term_signals"]
            for signal_name, signal_flags in subtask_term_signal_dict.items():
                if not torch.any(signal_flags):
                    is_episode_annotated_successfully = False
                    print(f'\tDid not detect completion for the subtask "{signal_name}".')
        return is_episode_annotated_successfully

    if subtask_term_signal_names is None:
        subtask_term_signal_names = {}

    subtask_term_signal_action_indices = {}
    episode_replay_success = True
    for eef_name, eef_subtask_term_signal_names in subtask_term_signal_names.items():
        if len(eef_subtask_term_signal_names) == 0:
            continue
        expected_subtask_signal_count = len(eef_subtask_term_signal_names)

        while True:
            session.configure_manual_marking(eef_name, eef_subtask_term_signal_names)
            print(f'\tPlaying the episode for subtask annotations for eef "{eef_name}".')
            print("\tSubtask signals to annotate:")
            print(f"\t\t- Termination:\t{eef_subtask_term_signal_names}")
            print('\n\tReplay starts immediately.')
            print('\tPress "B" to pause.')
            print('\tPress "N" to resume.')
            print('\tPress "S" to annotate subtask signals.')
            print('\tPress "Q" to skip the episode.\n')

            task_success_result = replay_episode(
                env,
                replay_plan,
                replay_runtime,
                session,
                args,
                task_type,
                success_term,
            )
            if session.state.skip_episode:
                print("\tSkipping the episode.")
                return False

            mark_summaries = []
            for marked_signal_index, action_index in enumerate(session.state.marked_subtask_action_indices):
                if marked_signal_index < len(eef_subtask_term_signal_names):
                    signal_name = eef_subtask_term_signal_names[marked_signal_index]
                else:
                    signal_name = f"signal_{marked_signal_index}"
                mark_summaries.append(f"{signal_name}@{action_index}")
            print(f"\tSubtasks marked: {mark_summaries}")
            marks_complete = expected_subtask_signal_count == len(session.state.marked_subtask_action_indices)

            if marks_complete:
                if task_success_result:
                    print(f'\tAll {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were annotated.')
                else:
                    episode_replay_success = False
                    print(
                        f'\tReplay did not reach final task success for eef "{eef_name}",'
                        " but all requested subtask marks were captured."
                    )
                for marked_signal_index in range(expected_subtask_signal_count):
                    subtask_term_signal_action_indices[eef_subtask_term_signal_names[marked_signal_index]] = (
                        session.state.marked_subtask_action_indices[marked_signal_index]
                    )
                break

            if not task_success_result:
                episode_replay_success = False
                print(
                    "\tThe final task was not completed in this replay attempt."
                    " Continue marking until all subtask signals are captured."
                )

            if expected_subtask_signal_count != len(session.state.marked_subtask_action_indices):
                print(
                    f"\tOnly {len(session.state.marked_subtask_action_indices)} out of"
                    f' {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were'
                    " annotated."
                )

            print(f'\tThe episode will be replayed again for re-marking subtask signals for the eef "{eef_name}".\n')

    session.clear_manual_marking()

    if not episode_replay_success and not args.ignore_replay_success:
        print(
            "\tThe final task was not completed in replay."
            " Add --ignore_replay_success to export annotations anyway."
        )
        return False
    if not episode_replay_success:
        print("\tContinuing because --ignore_replay_success is enabled.")

    annotated_episode = env.recorder_manager.get_episode(0)
    _finalize_annotated_episode(episode, annotated_episode, args)
    for subtask_term_signal_name, subtask_term_signal_action_index in subtask_term_signal_action_indices.items():
        subtask_signals = torch.ones(len(episode.data["actions"]), dtype=torch.bool)
        subtask_signals[:subtask_term_signal_action_index] = False
        annotated_episode.add(f"obs/datagen_info/subtask_term_signals/{subtask_term_signal_name}", subtask_signals)
    return True


__all__ = [
    "DatagenObjectPoseCaptureError",
    "annotate_episode",
    "build_replay_runtime_context",
    "normalize_manual_subtask_signal_name",
    "prepare_replay_plan",
    "recover_from_object_pose_capture_failure",
    "replay_episode",
    "stabilize_garment_after_reset_for_annotation",
    "validate_recorded_datagen_pose_contract",
]
