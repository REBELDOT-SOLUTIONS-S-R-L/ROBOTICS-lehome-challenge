"""Hybrid manager-based environment for garment folding.

Subclasses ManagerBasedRLEnv to add manual GarmentObject management.
The particle-based cloth cannot be declared in the scene config (no standard
Cfg type exists), so it is created/reset/deleted explicitly here.

All garment lifecycle logic is ported from:
    lehome.tasks.bedroom.garment_bi_v2.GarmentEnv
"""
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import OmegaConf

import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

from pxr import UsdShade, Sdf, UsdPhysics
import omni.kit.commands
from isaacsim.core.utils.prims import is_prim_path_valid

from lehome.assets.scenes.bedroom import MARBLE_BEDROOM_USD_PATH
from lehome.assets.object.Garment import GarmentObject
from lehome.tasks.bedroom.challenge_garment_loader import ChallengeGarmentLoader
from lehome.utils.success_checker_chanllege import success_checker_garment_fold
from lehome.utils.depth_to_pointcloud import generate_pointcloud_from_data
from lehome.devices.action_process import preprocess_device_action
from lehome.utils import RobotKinematics, compute_joints_from_ee_pose, mat_to_quat
from lehome.assets.robots.lerobot import ACTION_NAMES, SO101_FOLLOWER_USD_JOINT_LIMLITS
from lehome.utils.logger import get_logger

from .checkpoint_mappings import (
    ClothObjectPoseUnavailableError,
    semantic_keypoints_from_positions as map_semantic_keypoints_from_positions,
    validate_semantic_object_pose_dict,
)
from .fold_cloth_bi_arm_env_cfg import GarmentFoldEnvCfg
from .mdp.observations import get_subtask_signal_observations

logger = get_logger(__name__)

_SO101_URDF_REL_PATH = Path("Assets/robots/so101_new_calib.urdf")


class GarmentFoldEnv(ManagerBasedRLMimicEnv):
    """Hybrid manager-based environment for garment folding.

    Uses IsaacLab managers for actions, observations, terminations, and events,
    while manually managing the GarmentObject (particle cloth) lifecycle.
    Implements ManagerBasedRLMimicEnv methods for MimicGen compatibility.
    """

    cfg: GarmentFoldEnvCfg

    def __init__(self, cfg: GarmentFoldEnvCfg, render_mode: str | None = None, **kwargs):
        self.object = None  # Will be created in _setup_scene

        # Cache for distance-based reward
        self._last_computed_reward = 0.0
        # Mimic pose conversion caches.
        self._eef_body_idx_cache: dict[str, int] = {}
        self._ik_solver: RobotKinematics | None = None
        self._ik_solver_init_failed = False
        self._last_ik_input_eef_pose: dict[str, torch.Tensor] | None = None

        # IK telemetry: counters per arm, reset on each episode.  Used to
        # surface silent Pinocchio IK failures (L-BFGS-B returns the closest
        # reachable point without raising when the target is infeasible).
        self._ik_stats: dict[str, dict[str, float]] = {}
        self._ik_warn_cooldown: dict[str, int] = {}
        # Position/rotation residual thresholds above which a solve is
        # considered "bad" (reportable).  Calibrated for SO101 + cloth:
        # 1 cm is enough to miss the particle-attachment radius,
        # 15 deg is enough to misalign the gripper jaws with the cloth ridge.
        self._ik_pos_warn_m = 0.01
        self._ik_rot_warn_rad = float(np.deg2rad(15.0))
        # Hard-failure thresholds — treated as "unreachable" in logs.
        self._ik_pos_fail_m = 0.03
        self._ik_rot_fail_rad = float(np.deg2rad(30.0))

        # Per-env tracking of which subtasks each arm has completed so far.
        # Used to gate the (expensive) fold-success check to the final
        # subtask window — see ``is_final_fold_complete``.  Populated by
        # ``verify_subtask_completion`` every time a subtask trajectory
        # finishes and cleared on episode reset.
        self._completed_subtasks: dict[int, dict[str, set[int]]] = {}

        # Initialize garment loader and config
        self.garment_loader = ChallengeGarmentLoader(cfg.garment_cfg_base_path)
        self.garment_config = self.garment_loader.load_garment_config(
            cfg.garment_name, cfg.garment_version
        )
        self.particle_config = OmegaConf.load(cfg.particle_cfg_path)

        if cfg.use_random_seed:
            self.garment_rng = np.random.RandomState()
        else:
            self.garment_rng = np.random.RandomState(cfg.random_seed)

        super().__init__(cfg, render_mode, **kwargs)
        self._disable_gripper_capsule_collisions()
        self._filter_inter_arm_collisions()

        # Create garment object AFTER super().__init__() which sets up the scene.
        # Note: ManagerBasedEnv.__init__() creates the scene via InteractiveScene()
        # directly — there is no _setup_scene() override point, so we create
        # the garment here instead.
        self._create_garment_object()
        self._warmup_and_initialize_garment_object()

    def _warmup_and_initialize_garment_object(self, warmup_steps: int = 5):
        """Warm up physics and initialize garment observations/particle state."""
        if hasattr(self, "sim") and self.sim is not None:
            for i in range(warmup_steps):
                try:
                    self.sim.step(render=True)
                except Exception as e:
                    logger.warning(f"Error during garment warmup step {i + 1}: {e}")
                    break

        try:
            self.initialize_obs()
        except Exception as e:
            logger.warning(f"Failed to initialize garment observations after creation: {e}")

    def _create_garment_object(self):
        """Create a new GarmentObject with the currently selected asset."""
        if self.object is not None:
            self._delete_garment_object()

        garment_name = getattr(self.cfg, "garment_name", None)
        if garment_name and garment_name.strip():
            prim_name = garment_name.strip()
        else:
            prim_name = "Cloth"

        prim_path = f"/World/Object/{prim_name}"

        try:
            if is_prim_path_valid(prim_path):
                logger.debug(f"Prim path {prim_path} exists, deleting before creation")
                omni.kit.commands.execute("DeletePrims", paths=[prim_path])
                if hasattr(self, "sim") and self.sim is not None:
                    for _ in range(5):
                        self.sim.step(render=True)
                if is_prim_path_valid(prim_path):
                    logger.warning(f"Prim path {prim_path} still exists after deletion!")
        except Exception as e:
            logger.debug(f"Could not delete existing prim (may not exist): {e}")

        try:
            logger.debug(f"Creating GarmentObject at prim_path: {prim_path}")
            self.object = GarmentObject(
                prim_path=prim_path,
                particle_config=self.particle_config,
                garment_config=self.garment_config,
                rng=self.garment_rng,
            )
            logger.info("GarmentObject created successfully")
        except Exception as e:
            logger.error(f"Failed to create GarmentObject: {e}")
            raise RuntimeError(f"Failed to create GarmentObject: {e}") from e

        self._validate_created_object()

        self.texture_cfg = self.particle_config.objects.get("texture_randomization", {})
        self.light_cfg = self.particle_config.objects.get("light_randomization", {})

    def _validate_created_object(self):
        """Validate that the GarmentObject was created successfully."""
        if self.object is None:
            raise RuntimeError("GarmentObject creation returned None")

        required_attrs = [
            "usd_prim_path",
            "mesh_prim_path",
            "particle_system_path",
            "particle_material_path",
        ]
        for attr in required_attrs:
            if not hasattr(self.object, attr):
                raise RuntimeError(f"GarmentObject missing required attribute: {attr}")
            if getattr(self.object, attr) is None:
                raise RuntimeError(f"GarmentObject attribute {attr} is None")

        prim_paths_to_check = [
            ("usd_prim_path", self.object.usd_prim_path),
            ("mesh_prim_path", self.object.mesh_prim_path),
        ]
        for path_name, path_value in prim_paths_to_check:
            if not is_prim_path_valid(path_value):
                logger.warning(f"Prim path {path_name} '{path_value}' is not valid in stage.")
            else:
                logger.debug(f"Prim path {path_name} '{path_value}' is valid")

        logger.debug("GarmentObject validation passed")

    def _delete_garment_object(self):
        """Delete the current garment object from the stage."""
        if self.object is None:
            return

        from isaacsim.core.api import World
        world = World.instance()
        was_playing = world.is_playing()
        if was_playing:
            world.pause()

        try:
            if hasattr(self.object, "usd_prim_path") and self.object.usd_prim_path:
                prim_path = self.object.usd_prim_path
            else:
                garment_name = getattr(self.cfg, "garment_name", None)
                if garment_name and garment_name.strip():
                    prim_name = garment_name.strip()
                else:
                    prim_name = "Cloth"
                prim_path = f"/World/Object/{prim_name}"

            if hasattr(self.object, "particle_system_path"):
                particle_path = self.object.particle_system_path
                try:
                    if is_prim_path_valid(particle_path):
                        omni.kit.commands.execute("DeletePrims", paths=[particle_path])
                except Exception as e:
                    logger.warning(f"Failed to delete particle system: {e}")

            if is_prim_path_valid(prim_path):
                omni.kit.commands.execute("DeletePrims", paths=[prim_path])
        except Exception as e:
            logger.warning(f"Failed to delete garment object: {e}")
            import traceback
            traceback.print_exc()

        if was_playing:
            world.play()
        self.object = None

    def _get_rigid_body_prim_paths(self, root_prim_path: str) -> list[str]:
        """Return all rigid body prim paths below a robot root."""
        stage = self.scene.stage
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim.IsValid():
            logger.warning(
                f"[Collision Filter] Robot root prim not found while gathering rigid bodies: {root_prim_path}"
            )
            return []

        rigid_body_paths: list[str] = []
        stack = [root_prim]
        while stack:
            prim = stack.pop()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_paths.append(prim.GetPath().pathString)
            stack.extend(reversed(list(prim.GetChildren())))
        return rigid_body_paths

    def _apply_filtered_pairs(self, source_paths: Sequence[str], target_paths: Sequence[str]) -> None:
        """Apply USD filtered-pairs collision exclusions from each source to all targets."""
        stage = self.scene.stage
        for source_path in source_paths:
            source_prim = stage.GetPrimAtPath(source_path)
            if not source_prim.IsValid():
                continue
            filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(source_prim)
            rel = filtered_pairs_api.CreateFilteredPairsRel()
            existing_targets = set(rel.GetTargets())
            for target_path in target_paths:
                if target_path == source_path:
                    continue
                target_sdf_path = Sdf.Path(target_path)
                if target_sdf_path not in existing_targets:
                    rel.AddTarget(target_sdf_path)

    def _disable_gripper_capsule_collisions(self) -> None:
        """Disable collisions on capsule prims used only as gripper/jaw visuals."""
        stage = self.scene.stage
        robot_root_paths = (
            "/World/Robot/Left_Robot",
            "/World/Robot/Right_Robot",
        )
        disabled_paths: list[str] = []

        for root_path in robot_root_paths:
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                logger.warning(
                    "[Collision Filter] Robot root prim not found while disabling gripper capsules: %s",
                    root_path,
                )
                continue

            stack = [root_prim]
            while stack:
                prim = stack.pop()
                prim_path = prim.GetPath().pathString
                normalized_path = prim_path.lower()
                if (
                    prim.GetTypeName() == "Capsule"
                    and ("gripper" in normalized_path or "jaw" in normalized_path)
                ):
                    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
                    collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
                    if not collision_enabled_attr.IsValid():
                        collision_enabled_attr = collision_api.CreateCollisionEnabledAttr()
                    collision_enabled_attr.Set(False)
                    disabled_paths.append(prim_path)
                stack.extend(reversed(list(prim.GetChildren())))

        if disabled_paths:
            logger.info(
                "[Collision Filter] Disabled collisions on %d gripper/jaw capsule prims.",
                len(disabled_paths),
            )
        else:
            logger.warning(
                "[Collision Filter] No gripper/jaw capsule prims were found to disable collisions."
            )

    def _filter_inter_arm_collisions(self) -> None:
        """Disable collisions between left and right robot rigid bodies."""
        left_root_path = "/World/Robot/Left_Robot"
        right_root_path = "/World/Robot/Right_Robot"
        left_body_paths = self._get_rigid_body_prim_paths(left_root_path)
        right_body_paths = self._get_rigid_body_prim_paths(right_root_path)
        if not left_body_paths or not right_body_paths:
            logger.warning(
                "[Collision Filter] Skipping inter-arm collision filtering because rigid body paths "
                f"were incomplete. left={len(left_body_paths)} right={len(right_body_paths)}"
            )
            return

        self._apply_filtered_pairs(left_body_paths, right_body_paths)
        self._apply_filtered_pairs(right_body_paths, left_body_paths)
        logger.info(
            "[Collision Filter] Disabled inter-arm collisions between %d left-arm and %d right-arm rigid bodies.",
            len(left_body_paths),
            len(right_body_paths),
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset environments — resets arms to default and resets garment."""
        # Flush IK telemetry from the previous episode before super() wipes state.
        self._flush_ik_stats()

        # Reset per-env subtask-completion tracking for the success-check gate.
        self._clear_completed_subtasks(env_ids)

        super()._reset_idx(env_ids)

        # Reset cached reward
        self._last_computed_reward = 0.0

        # Reset the garment object
        if self.object is not None:
            self.object.reset()

        # Apply domain randomization
        if hasattr(self, "texture_cfg") and self.texture_cfg.get("enable", False):
            self._randomize_table038_texture()
        if hasattr(self, "light_cfg") and self.light_cfg.get("enable", False):
            self._randomize_light()

    # ------------------------------------------------------------------
    # IK telemetry helpers
    # ------------------------------------------------------------------

    def _ik_stats_entry(self, arm_name: str) -> dict[str, float]:
        """Lazily create the per-arm IK stats accumulator."""
        entry = self._ik_stats.get(arm_name)
        if entry is None:
            entry = {
                "n_calls": 0,
                "n_warn": 0,        # residual > warn thresholds
                "n_fail": 0,        # residual > fail thresholds (unreachable)
                "n_hit_bound": 0,   # solved joints pinned to a URDF bound
                "n_no_solve": 0,    # compute_joints_from_ee_pose returned None
                "pos_res_sum": 0.0,
                "pos_res_max": 0.0,
                "rot_res_sum": 0.0,
                "rot_res_max": 0.0,
            }
            self._ik_stats[arm_name] = entry
        return entry

    def _record_ik_diagnostics(self, arm_name: str, diag: dict | None) -> None:
        """Update per-arm IK counters and emit rate-limited warnings."""
        if diag is None:
            entry = self._ik_stats_entry(arm_name)
            entry["n_calls"] += 1
            entry["n_no_solve"] += 1
            # Warn once per cooldown window — IK code-path exceptions are
            # always noteworthy, unlike routine residual spikes.
            cooldown = self._ik_warn_cooldown.get(arm_name, 0)
            if cooldown <= 0:
                logger.warning(f"[IK][{arm_name}] compute_joints_from_ee_pose returned None (exception path)")
                self._ik_warn_cooldown[arm_name] = 90  # ~1 s @ 90 Hz
            else:
                self._ik_warn_cooldown[arm_name] = cooldown - 1
            return

        entry = self._ik_stats_entry(arm_name)
        entry["n_calls"] += 1
        pos_res = float(diag.get("pos_residual_m", 0.0))
        rot_res = float(diag.get("rot_residual_rad", 0.0))
        if not np.isfinite(pos_res):
            pos_res = 0.0
        if not np.isfinite(rot_res):
            rot_res = 0.0

        entry["pos_res_sum"] += pos_res
        entry["rot_res_sum"] += rot_res
        if pos_res > entry["pos_res_max"]:
            entry["pos_res_max"] = pos_res
        if rot_res > entry["rot_res_max"]:
            entry["rot_res_max"] = rot_res

        if bool(diag.get("hit_bound", False)):
            entry["n_hit_bound"] += 1

        is_warn = (pos_res > self._ik_pos_warn_m) or (rot_res > self._ik_rot_warn_rad)
        is_fail = (pos_res > self._ik_pos_fail_m) or (rot_res > self._ik_rot_fail_rad)
        if is_fail:
            entry["n_fail"] += 1
        elif is_warn:
            entry["n_warn"] += 1

        if is_fail:
            cooldown = self._ik_warn_cooldown.get(arm_name, 0)
            if cooldown <= 0:
                logger.warning(
                    f"[IK][{arm_name}] unreachable target: "
                    f"pos_residual={pos_res * 1000:.1f} mm  "
                    f"rot_residual={np.degrees(rot_res):.1f} deg  "
                    f"cost={float(diag.get('cost', 0.0)):.4g}  "
                    f"hit_bound={bool(diag.get('hit_bound', False))}"
                )
                self._ik_warn_cooldown[arm_name] = 90  # ~1 s @ 90 Hz
            else:
                self._ik_warn_cooldown[arm_name] = cooldown - 1

    def _flush_ik_stats(self) -> None:
        """Emit a per-arm summary of IK health for the episode that just ended."""
        if not self._ik_stats:
            return
        for arm_name, entry in self._ik_stats.items():
            n = int(entry.get("n_calls", 0))
            if n <= 0:
                continue
            pos_mean = entry["pos_res_sum"] / n
            rot_mean = entry["rot_res_sum"] / n
            logger.info(
                f"[IK][{arm_name}][episode end] "
                f"n={n}  "
                f"warn={int(entry['n_warn'])}  fail={int(entry['n_fail'])}  "
                f"bound_hits={int(entry['n_hit_bound'])}  no_solve={int(entry['n_no_solve'])}  "
                f"pos_res mean={pos_mean * 1000:.2f} mm max={entry['pos_res_max'] * 1000:.2f} mm  "
                f"rot_res mean={np.degrees(rot_mean):.2f} deg max={np.degrees(entry['rot_res_max']):.2f} deg"
            )
        self._ik_stats.clear()
        self._ik_warn_cooldown.clear()

    # ------------------------------------------------------------------
    # Observations (for compatibility with direct env record/replay)
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """Get observations compatible with the direct env format.

        This provides the same observation dict as garment_bi_v2.GarmentEnv
        for backward compatibility with record/replay scripts.
        """
        left_arm = self.scene["left_arm"]
        right_arm = self.scene["right_arm"]
        top_camera = self.scene["top_camera"]
        left_camera = self.scene["left_camera"]
        right_camera = self.scene["right_camera"]

        # Get current action (from action manager if available)
        action_dim = 12
        if hasattr(self, "action_manager") and hasattr(self.action_manager, "total_action_dim"):
            try:
                action_dim = int(self.action_manager.total_action_dim)
            except Exception:
                action_dim = 12
        if hasattr(self, "action_manager") and self.action_manager is not None:
            try:
                action = self.action_manager.action.squeeze(0)
            except Exception:
                action = torch.zeros(action_dim, device=self.device)
        else:
            action = torch.zeros(action_dim, device=self.device)

        left_joint_pos = torch.cat(
            [left_arm.data.joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        right_joint_pos = torch.cat(
            [right_arm.data.joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        joint_pos = torch.cat([left_joint_pos, right_joint_pos], dim=1).squeeze(0)

        top_camera_rgb = top_camera.data.output["rgb"]
        left_camera_rgb = left_camera.data.output["rgb"]
        right_camera_rgb = right_camera.data.output["rgb"]

        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu().detach().numpy().squeeze(),
            "observation.images.left_rgb": left_camera_rgb.cpu().detach().numpy().squeeze(),
            "observation.images.right_rgb": right_camera_rgb.cpu().detach().numpy().squeeze(),
        }
        return observations

    # ------------------------------------------------------------------
    # Rewards (dense, computed manually from garment particle state)
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        """Calculate distance-based reward for garment folding task."""
        if self.object is None or not hasattr(self.object, "_cloth_prim_view"):
            return torch.zeros(self.num_envs, device=self.device)

        garment_type = self.garment_loader.get_garment_type(self.cfg.garment_name)
        result = success_checker_garment_fold(self.object, garment_type)

        if not isinstance(result, dict):
            return torch.full(
                (self.num_envs,), self._last_computed_reward, device=self.device
            )

        success = result.get("success", False)
        details = result.get("details", {})

        if success:
            self._last_computed_reward = 1.0
            return torch.ones(self.num_envs, device=self.device)

        # Distance-based reward computation
        total_reward = 0.0
        num_conditions = len(details)
        if num_conditions == 0:
            return torch.zeros(self.num_envs, device=self.device)

        primary_rewards = []
        secondary_rewards = []

        for cond_key, cond_info in details.items():
            value = cond_info.get("value", 0.0)
            threshold = cond_info.get("threshold", 0.0)
            passed = cond_info.get("passed", False)
            description = cond_info.get("description", "")
            is_less_than = "<=" in description

            if passed:
                condition_reward = 1.0
            else:
                if is_less_than:
                    if threshold > 0:
                        excess_ratio = max(0.0, (value - threshold) / threshold)
                        condition_reward = np.exp(-3.0 * excess_ratio)
                    else:
                        condition_reward = 0.0
                else:
                    if threshold > 0:
                        ratio = value / threshold
                        condition_reward = max(0.0, 1.0 - np.exp(-1.5 * (1.0 - ratio)))
                    else:
                        condition_reward = 0.0

            if is_less_than:
                primary_rewards.append(condition_reward)
            else:
                secondary_rewards.append(condition_reward)

        if primary_rewards:
            avg_primary = sum(primary_rewards) / len(primary_rewards)
            min_primary = min(primary_rewards)
            primary_score = (avg_primary ** 0.7) * (min_primary ** 0.3)
        else:
            primary_score = 1.0

        if secondary_rewards:
            secondary_score = sum(secondary_rewards) / len(secondary_rewards)
        else:
            secondary_score = 1.0

        final_reward = (0.8 * primary_score + 0.2 * secondary_score) * 0.9
        self._last_computed_reward = float(final_reward)

        return torch.full((self.num_envs,), final_reward, device=self.device)

    # ------------------------------------------------------------------
    # Success checking
    # ------------------------------------------------------------------

    def _check_success(self) -> bool:
        """Check success based on garment type."""
        if self.object is None or not hasattr(self.object, "_cloth_prim_view"):
            return False

        garment_type = self.garment_loader.get_garment_type(self.cfg.garment_name)
        result = success_checker_garment_fold(self.object, garment_type)

        if isinstance(result, dict):
            return result.get("success", False)
        return bool(result)

    def _get_success(self) -> torch.Tensor:
        """Get success tensor for all environments."""
        if self.object is None or not hasattr(self.object, "_cloth_prim_view"):
            success = False
            result = None
        else:
            garment_type = self.garment_loader.get_garment_type(self.cfg.garment_name)
            result = success_checker_garment_fold(self.object, garment_type)

            if isinstance(result, dict):
                logger.info(
                    f"[Success Check] Garment type: {result.get('garment_type', 'unknown')}, "
                    f"Thresholds: {result.get('thresholds', [])}"
                )
                details = result.get("details", {})
                for key, cond_info in details.items():
                    status = "✓" if cond_info.get("passed", False) else "✗"
                    logger.info(f"  {cond_info.get('description', '')} -> {status}")
                success = result.get("success", False)
                logger.info(f"[Success Check] Final result: {'Success ✓' if success else 'Failed ✗'}")
            else:
                success = bool(result)

        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * self.num_envs, device=self.device, dtype=torch.bool
            )
        else:
            success_tensor = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return success_tensor

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get done signals — time out only."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------

    def _randomize_table038_texture(self):
        """Randomize Table038 texture based on config."""
        if not self.texture_cfg.get("enable", False):
            return

        folder = self.texture_cfg.get("folder", "")
        if not os.path.isabs(folder):
            folder = os.path.join(os.getcwd(), folder)

        min_id = int(self.texture_cfg.get("min_id", 1))
        max_id = int(self.texture_cfg.get("max_id", 1))
        shader_path = self.texture_cfg.get("prim_path", "")

        if not folder or not os.path.exists(folder):
            return
        if not shader_path:
            return

        stage = self.scene.stage
        shader_prim = stage.GetPrimAtPath(shader_path)
        if not shader_prim.IsValid():
            return

        shader = UsdShade.Shader(shader_prim)
        idx = random.randint(min_id, max_id)
        tex_path = os.path.join(folder, f"{idx}.png")

        tex_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
        if not tex_input:
            return
        tex_input.Set(Sdf.AssetPath(tex_path))

    def _randomize_light(self):
        """Randomize DomeLight attributes based on config."""
        if not self.light_cfg.get("enable", False):
            return

        prim_path = self.light_cfg.get("prim_path", "/World/Light")
        intensity_range = self.light_cfg.get("intensity_range", [800, 2000])
        color_range = self.light_cfg.get("color_range", [0.0, 1.0])

        stage = self.scene.stage
        light_prim = stage.GetPrimAtPath(prim_path)
        if not light_prim.IsValid():
            return

        intensity = random.uniform(*intensity_range)
        color = tuple(random.uniform(color_range[0], color_range[1]) for _ in range(3))

        light_prim.GetAttribute("inputs:intensity").Set(intensity)
        light_prim.GetAttribute("inputs:color").Set(color)

    # ------------------------------------------------------------------
    # Garment utilities (for record/replay/eval compatibility)
    # ------------------------------------------------------------------

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        """Process device action for teleoperation."""
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        """Initialize observations (particle cloth setup)."""
        self.object.initialize()

    def get_all_pose(self):
        """Get all garment pose data."""
        return self.object.get_all_pose()

    def set_all_pose(self, pose):
        """Set all garment pose data."""
        self.object.set_all_pose(pose)

    def switch_garment(self, garment_name: str, garment_version: str = None):
        """Switch to a different garment without recreating the environment."""
        logger.info(f"Switching garment to: {garment_name} (version: {garment_version})")

        if self.object is not None:
            self._delete_garment_object()

        if garment_version is None:
            garment_version = self.cfg.garment_version

        self.cfg.garment_name = garment_name
        self.cfg.garment_version = garment_version

        self.garment_config = self.garment_loader.load_garment_config(
            garment_name, garment_version
        )

        # Physics cleanup steps
        cleanup_steps = 20
        if hasattr(self, "sim") and self.sim is not None:
            for i in range(cleanup_steps):
                try:
                    self.sim.step(render=True)
                except Exception as e:
                    logger.warning(f"Error during cleanup step {i + 1}: {e}")
                    continue

        self._create_garment_object()

        initial_steps = 5
        if hasattr(self, "sim") and self.sim is not None:
            for i in range(initial_steps):
                try:
                    self.sim.step(render=True)
                except Exception as e:
                    logger.warning(f"Error during initial step {i + 1}: {e}")

        if hasattr(self, "render"):
            try:
                self.render()
            except Exception as e:
                logger.warning(f"Error during render: {e}")

        try:
            self.initialize_obs()
            if hasattr(self, "render"):
                try:
                    self.render()
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to initialize observations: {e}")

    def _get_workspace_pointcloud(
        self, env_index: int = 0, num_points: int = 2048, use_fps: bool = False
    ):
        """Retrieve workspace pointcloud from specified env_id."""
        top_camera = self.scene["top_camera"]
        top_camera_rgb_tensor = top_camera.data.output["rgb"]
        top_camera_depth_tensor = top_camera.data.output.get("depth")
        if top_camera_depth_tensor is None:
            raise RuntimeError("Top camera depth output is disabled for this environment.")

        depth_img = top_camera_depth_tensor[env_index].clone().cpu().numpy().squeeze()
        depth_img = depth_img.astype(np.float32) / 1000.0
        rgb_img = top_camera_rgb_tensor[env_index].clone().cpu().numpy()

        pointclouds = generate_pointcloud_from_data(
            rgb_image=rgb_img,
            depth_image=depth_img,
            num_points=num_points,
            use_fps=use_fps,
        )
        return pointclouds

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Cleanup method (defensive programming)."""
        logger.debug("Starting cleanup...")
        if self.object is not None:
            self._delete_garment_object()
        self.object = None
        logger.debug("Cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup on deletion."""
        try:
            if hasattr(self, "object") and self.object is not None:
                self.cleanup()
        except Exception:
            pass

    # =========================================================================
    # MimicGen Required Methods (ManagerBasedRLMimicEnv implementations)
    # =========================================================================

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> tuple[Sequence[int] | slice, int]:
        if env_ids is None:
            return slice(None), self.num_envs
        if isinstance(env_ids, slice):
            return env_ids, self.num_envs
        return env_ids, len(env_ids)

    def _get_eef_body_idx(self, eef_name: str) -> int:
        if eef_name in self._eef_body_idx_cache:
            return self._eef_body_idx_cache[eef_name]

        arm = self.scene[eef_name]
        candidate_patterns = (
            "^gripper_frame_link$",
            "^gripper_link$",
            ".*gripper_frame.*",
            ".*gripper.*",
        )
        for pattern in candidate_patterns:
            try:
                body_ids, _ = arm.find_bodies(pattern, preserve_order=True)
                if len(body_ids) > 0:
                    self._eef_body_idx_cache[eef_name] = int(body_ids[0])
                    try:
                        body_name = arm.body_names[int(body_ids[0])]
                    except Exception:
                        body_name = f"<index {int(body_ids[0])}>"
                    logger.info(
                        "Resolved end-effector body for %s: %s",
                        eef_name,
                        body_name,
                    )
                    return int(body_ids[0])
            except Exception:
                continue

        # Fallback: last articulation link.
        fallback_idx = int(arm.data.body_link_pos_w.shape[1] - 1)
        self._eef_body_idx_cache[eef_name] = fallback_idx
        logger.warning(
            "Could not resolve end-effector body for %s from names %s, using last link index %d",
            eef_name,
            arm.body_names,
            fallback_idx,
        )
        return fallback_idx

    def _init_ik_solver_if_needed(self) -> bool:
        if self._ik_solver is not None:
            return True
        if self._ik_solver_init_failed:
            return False
        try:
            repo_root = Path(__file__).resolve().parents[5]
            urdf_path = repo_root / _SO101_URDF_REL_PATH
            self._ik_solver = RobotKinematics(str(urdf_path), target_frame_name="gripper_frame_link")
            return True
        except Exception as exc:
            self._ik_solver_init_failed = True
            logger.warning(
                "Failed to initialize SO101 IK/FK solver from %s: %s. Falling back to current EEF pose only.",
                _SO101_URDF_REL_PATH,
                exc,
            )
            return False

    @staticmethod
    def _make_pose_from_pos_quat(pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
        rot_w = PoseUtils.matrix_from_quat(quat_w)
        return PoseUtils.make_pose(pos_w, rot_w)

    def _is_native_mimic_ik_action_contract(self) -> bool:
        """Return True when this env is configured to accept native 16D IK actions."""
        try:
            if hasattr(self, "action_manager") and hasattr(self.action_manager, "total_action_dim"):
                return int(self.action_manager.total_action_dim) == 16
        except Exception:
            pass
        return False

    def _use_pinocchio_generation(self) -> bool:
        """Return True when generation should force the 12D Pinocchio path."""
        return bool(getattr(self.cfg, "force_pinocchio_generation", False))

    @staticmethod
    def _sim_joints_to_pinocchio_convention(joint_pos: np.ndarray) -> np.ndarray:
        """Convert Isaac joint positions to the SO101 URDF convention.

        The so101_new_calib URDF and the Isaac USD share the same joint
        conventions, so no offset is needed — this is an identity copy.
        """
        return np.asarray(joint_pos, dtype=np.float64).copy()

    @staticmethod
    def _pinocchio_joints_to_sim_convention(joint_pos: np.ndarray) -> np.ndarray:
        """Convert SO101 URDF joint positions back to Isaac joint convention.

        Identity copy — see ``_sim_joints_to_pinocchio_convention``.
        """
        return np.asarray(joint_pos, dtype=np.float64).copy()

    def _get_arm_world_base_transform_np(self, arm_name: str, env_i: int) -> np.ndarray:
        """Get world<-base transform for one arm and env as a 4x4 matrix."""
        arm = self.scene[arm_name]
        base_pos_w = arm.data.root_pos_w[env_i].detach().cpu().numpy()
        base_quat_w = arm.data.root_quat_w[env_i].detach().cpu().numpy()
        T_world_base = np.eye(4, dtype=np.float64)
        T_world_base[:3, 3] = base_pos_w
        T_world_base[:3, :3] = PoseUtils.matrix_from_quat(
            torch.as_tensor(base_quat_w, dtype=torch.float32, device=self.device).unsqueeze(0)
        )[0].detach().cpu().numpy()
        return T_world_base

    def _world_pose_to_base_pose_np(self, arm_name: str, env_i: int, T_world_pose: np.ndarray) -> np.ndarray:
        """Transform one pose from world frame to arm base frame."""
        T_world_base = self._get_arm_world_base_transform_np(arm_name, env_i)
        return np.linalg.inv(T_world_base) @ T_world_pose

    def _base_pose_to_world_pose_np(self, arm_name: str, env_i: int, T_base_pose: np.ndarray) -> np.ndarray:
        """Transform one pose from arm base frame to world frame."""
        T_world_base = self._get_arm_world_base_transform_np(arm_name, env_i)
        return T_world_base @ T_base_pose

    def _ensure_ik_input_eef_pose_cache(self) -> dict[str, torch.Tensor]:
        """Allocate the cached base-frame IK input pose buffer on first use."""
        if self._last_ik_input_eef_pose is None:
            identity_pose = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1, 1)
            self._last_ik_input_eef_pose = {
                "left_arm": identity_pose.clone(),
                "right_arm": identity_pose.clone(),
            }
        return self._last_ik_input_eef_pose

    def _cache_ik_input_eef_pose(self, arm_name: str, env_i: int, T_base_pose: torch.Tensor | np.ndarray) -> None:
        """Cache the exact base-frame pose that is sent into IK / native IK actions."""
        cache = self._ensure_ik_input_eef_pose_cache()
        pose = torch.as_tensor(T_base_pose, device=self.device, dtype=torch.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"Expected IK input pose with shape (4, 4), got {tuple(pose.shape)}")
        cache[arm_name][env_i] = pose

    def get_last_ik_input_eef_pose(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Return the last cached per-arm base-frame IK input pose."""
        if self._last_ik_input_eef_pose is None:
            return None
        env_ids, _ = self._resolve_env_ids(env_ids)
        return {
            arm_name: pose[env_ids].clone()
            for arm_name, pose in self._last_ik_input_eef_pose.items()
        }

    def get_last_ik_input_eef_pose_world(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Return the last cached per-arm IK input pose in world frame."""
        base_pose_dict = self.get_last_ik_input_eef_pose(env_ids=env_ids)
        if base_pose_dict is None:
            return None

        env_ids_resolved, num_envs = self._resolve_env_ids(env_ids)
        if isinstance(env_ids_resolved, slice):
            env_index_list = list(range(self.num_envs))[env_ids_resolved]
        else:
            env_index_list = [int(idx) for idx in env_ids_resolved]

        world_pose_dict: dict[str, torch.Tensor] = {}
        for arm_name, base_pose in base_pose_dict.items():
            world_pose = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)
            for local_i, env_i in enumerate(env_index_list):
                T_world_pose = self._base_pose_to_world_pose_np(
                    arm_name,
                    env_i,
                    base_pose[local_i].detach().cpu().numpy(),
                )
                world_pose[local_i] = torch.as_tensor(T_world_pose, device=self.device, dtype=torch.float32)
            world_pose_dict[arm_name] = world_pose
        return world_pose_dict

    def get_generation_export_actions(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor | None:
        """Return the action tensor that should be exported into generated HDF5."""
        if not hasattr(self, "action_manager") or not hasattr(self.action_manager, "action"):
            return None

        runtime_action = self.action_manager.action
        if runtime_action.ndim == 1:
            runtime_action = runtime_action.unsqueeze(0)

        env_ids_resolved, num_envs = self._resolve_env_ids(env_ids)
        runtime_action = runtime_action[env_ids_resolved]
        if int(runtime_action.shape[-1]) == 16:
            return runtime_action.clone()
        if int(runtime_action.shape[-1]) != 12:
            return None

        base_pose_dict = self.get_last_ik_input_eef_pose(env_ids=env_ids)
        if base_pose_dict is None:
            base_pose_dict = {}
            target_pose_world = self.action_to_target_eef_pose(runtime_action)
            if isinstance(env_ids_resolved, slice):
                env_index_list = list(range(self.num_envs))[env_ids_resolved]
            else:
                env_index_list = [int(idx) for idx in env_ids_resolved]
            for arm_name, world_pose in target_pose_world.items():
                base_pose = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)
                for local_i, env_i in enumerate(env_index_list):
                    T_base_pose = self._world_pose_to_base_pose_np(
                        arm_name,
                        env_i,
                        world_pose[local_i].detach().cpu().numpy(),
                    )
                    base_pose[local_i] = torch.as_tensor(T_base_pose, device=self.device, dtype=torch.float32)
                base_pose_dict[arm_name] = base_pose

        export_action = torch.zeros((num_envs, 16), device=self.device, dtype=torch.float32)
        gripper_action = self.actions_to_gripper_actions(runtime_action)
        for arm_name, action_col_offset in (("left_arm", 0), ("right_arm", 8)):
            base_pose = base_pose_dict[arm_name]
            export_action[:, action_col_offset : action_col_offset + 3] = base_pose[:, :3, 3]
            export_action[:, action_col_offset + 3 : action_col_offset + 7] = PoseUtils.quat_from_matrix(
                base_pose[:, :3, :3]
            )
            export_action[:, action_col_offset + 7 : action_col_offset + 8] = gripper_action[arm_name]
        return export_action.clone()

    def action_to_ik_input_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """Decode native 16D IK actions into per-arm base-frame 4x4 poses."""
        if action.ndim == 1:
            action = action.unsqueeze(0)
        num_envs = int(action.shape[0])
        if int(action.shape[-1]) != 16:
            return None

        def _decode_arm(action_col_offset: int) -> torch.Tensor:
            decoded = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)
            for i in range(num_envs):
                pos_base = action[i, action_col_offset : action_col_offset + 3]
                quat_base_wxyz = action[i, action_col_offset + 3 : action_col_offset + 7]
                decoded[i, :3, 3] = pos_base
                decoded[i, :3, :3] = PoseUtils.matrix_from_quat(quat_base_wxyz.unsqueeze(0))[0]
            return decoded

        return {
            "left_arm": _decode_arm(0),
            "right_arm": _decode_arm(8),
        }

    def action_to_ik_input_joint_pos(self, action: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """Return per-arm joint targets sent to the robot for both action contracts."""
        if action.ndim == 1:
            action = action.unsqueeze(0)
        num_envs = int(action.shape[0])
        action_dim = int(action.shape[-1])
        if action_dim == 12:
            return {
                "left_arm": action[:, :6].clone(),
                "right_arm": action[:, 6:12].clone(),
            }
        if action_dim != 16:
            return None

        try:
            left_arm_term = self.action_manager.get_term("left_arm_action")
            right_arm_term = self.action_manager.get_term("right_arm_action")
            left_gripper_term = self.action_manager.get_term("left_gripper_action")
            right_gripper_term = self.action_manager.get_term("right_gripper_action")
            left_joint_targets = left_arm_term.compute_joint_position_target()[:num_envs]
            right_joint_targets = right_arm_term.compute_joint_position_target()[:num_envs]
            left_gripper = left_gripper_term.processed_actions[:num_envs]
            right_gripper = right_gripper_term.processed_actions[:num_envs]
        except Exception:
            return None

        return {
            "left_arm": torch.cat((left_joint_targets, left_gripper), dim=-1).clone(),
            "right_arm": torch.cat((right_joint_targets, right_gripper), dim=-1).clone(),
        }

    def _compute_target_pose_from_joint_targets(self, arm_name: str, joint_targets: torch.Tensor) -> torch.Tensor:
        if not self._init_ik_solver_if_needed() or self._ik_solver is None:
            return self.get_robot_eef_pose(arm_name)

        arm = self.scene[arm_name]
        num_envs = joint_targets.shape[0]
        target_pose = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)

        for i in range(num_envs):
            joints_np = joint_targets[i].detach().cpu().numpy()
            joints_np = self._sim_joints_to_pinocchio_convention(joints_np)
            # FK helper expects full 6D joint vector and returns [pos(xyz), quat(xyzw), gripper].
            ee_pose_base = np.asarray(self._ik_solver.forward_kinematics(np.rad2deg(joints_np)))
            pos_base = ee_pose_base[:3, 3]
            quat_base_xyzw = mat_to_quat(ee_pose_base[:3, :3])
            quat_base_wxyz = np.roll(quat_base_xyzw, 1)

            base_pos_w = arm.data.root_pos_w[i]
            base_quat_w = arm.data.root_quat_w[i]
            T_world_base = self._make_pose_from_pos_quat(base_pos_w.unsqueeze(0), base_quat_w.unsqueeze(0))[0]
            T_base_target = torch.eye(4, device=self.device, dtype=torch.float32)
            T_base_target[:3, 3] = torch.as_tensor(pos_base, device=self.device, dtype=torch.float32)
            T_base_target[:3, :3] = PoseUtils.matrix_from_quat(
                torch.as_tensor(quat_base_wxyz, device=self.device, dtype=torch.float32).unsqueeze(0)
            )[0]
            target_pose[i] = T_world_base @ T_base_target

        return target_pose

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose as 4x4 transform."""
        env_ids, num_envs = self._resolve_env_ids(env_ids)
        arm = self.scene[eef_name]
        eef_body_idx = self._get_eef_body_idx(eef_name)
        eef_pos_w = arm.data.body_link_pos_w[env_ids, eef_body_idx]
        eef_quat_w = arm.data.body_link_quat_w[env_ids, eef_body_idx]
        pose = self._make_pose_from_pos_quat(eef_pos_w, eef_quat_w)
        if pose.shape[0] != num_envs:
            pose = pose.reshape(num_envs, 4, 4)
        return pose

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Takes target poses and gripper actions and returns an environment action tensor."""
        def _normalize_target_pose(target: torch.Tensor | np.ndarray | None) -> torch.Tensor | None:
            if target is None:
                return None
            target = torch.as_tensor(target, device=self.device, dtype=torch.float32)
            if target.ndim == 2:
                if target.shape == (4, 4):
                    return target.unsqueeze(0)
                if target.shape[-1] == 16:
                    return target.reshape(-1, 4, 4)
                raise ValueError(f"Unsupported target_eef_pose shape {tuple(target.shape)}")
            if target.ndim == 3 and target.shape[-2:] == (4, 4):
                return target
            raise ValueError(f"Unsupported target_eef_pose rank/shape {tuple(target.shape)}")

        def _normalize_gripper_action(gripper: torch.Tensor | np.ndarray | None, num_envs: int) -> torch.Tensor | None:
            if gripper is None:
                return None
            gripper = torch.as_tensor(gripper, device=self.device, dtype=torch.float32).reshape(-1)
            if gripper.numel() == 1:
                return gripper.expand(num_envs)
            if gripper.numel() < num_envs:
                return gripper[-1:].expand(num_envs)
            return gripper[:num_envs]

        left_target = _normalize_target_pose(target_eef_pose_dict.get("left_arm"))
        right_target = _normalize_target_pose(target_eef_pose_dict.get("right_arm"))

        num_envs = 1
        if left_target is not None:
            num_envs = int(left_target.shape[0])
        elif right_target is not None:
            num_envs = int(right_target.shape[0])

        left_grip = _normalize_gripper_action(gripper_action_dict.get("left_arm"), num_envs)
        right_grip = _normalize_gripper_action(gripper_action_dict.get("right_arm"), num_envs)

        # Native mimic IK contract: action space is [left(pos+quat+grip=8), right(...=8)].
        if self._is_native_mimic_ik_action_contract():
            action = torch.zeros((num_envs, 16), device=self.device, dtype=torch.float32)

            def _fill_arm_native(
                arm_name: str,
                target_pose: torch.Tensor | None,
                gripper: torch.Tensor | None,
                action_col_offset: int,
            ) -> None:
                arm = self.scene[arm_name]
                if target_pose is None:
                    target_pose = self.get_robot_eef_pose(arm_name)
                if target_pose.shape[0] != num_envs:
                    target_pose = target_pose[:1].expand(num_envs, -1, -1).clone()

                for i in range(num_envs):
                    env_i = env_id if num_envs == 1 else min(i, self.num_envs - 1)
                    T_world_target = target_pose[i].detach().cpu().numpy()
                    T_base_target = self._world_pose_to_base_pose_np(arm_name, env_i, T_world_target)

                    pos_base = torch.as_tensor(T_base_target[:3, 3], device=self.device, dtype=torch.float32)
                    rot_base = torch.as_tensor(T_base_target[:3, :3], device=self.device, dtype=torch.float32)
                    quat_base_wxyz = PoseUtils.quat_from_matrix(rot_base.unsqueeze(0))[0]

                    current_grip = float(arm.data.joint_pos[env_i, 5].item())
                    grip_val = float(gripper[i].item()) if gripper is not None else current_grip
                    pose_action = torch.cat([pos_base, quat_base_wxyz], dim=0)

                    # Optional per-subtask action noise support used by MimicGen.
                    if action_noise_dict is not None and arm_name in action_noise_dict:
                        noise_scale = torch.as_tensor(
                            action_noise_dict[arm_name], device=self.device, dtype=torch.float32
                        ).reshape(-1)
                        if noise_scale.numel() == 1:
                            noise = noise_scale.expand(7) * torch.randn(7, device=self.device)
                        elif noise_scale.numel() >= 7:
                            noise = noise_scale[:7] * torch.randn(7, device=self.device)
                        else:
                            noise = noise_scale[-1:].expand(7) * torch.randn(7, device=self.device)
                        pose_action = pose_action + noise
                        quat = pose_action[3:7]
                        pose_action[3:7] = quat / torch.linalg.norm(quat).clamp_min(1e-12)

                    T_base_action = torch.eye(4, device=self.device, dtype=torch.float32)
                    T_base_action[:3, 3] = pose_action[:3]
                    T_base_action[:3, :3] = PoseUtils.matrix_from_quat(pose_action[3:7].unsqueeze(0))[0]
                    self._cache_ik_input_eef_pose(arm_name, env_i, T_base_action)

                    action[i, action_col_offset : action_col_offset + 7] = pose_action
                    action[i, action_col_offset + 7] = grip_val

            _fill_arm_native("left_arm", left_target, left_grip, 0)
            _fill_arm_native("right_arm", right_target, right_grip, 8)
            return action

        # Keep a small non-zero orientation term so IK does not collapse to degenerate
        # position-only solutions that often underuse elbow flexion.
        ik_orientation_weight = float(getattr(self.cfg, "mimic_ik_orientation_weight", 0.01))
        if not np.isfinite(ik_orientation_weight):
            ik_orientation_weight = 0.01
        ik_orientation_weight = max(0.0, ik_orientation_weight)

        action = torch.zeros((num_envs, 12), device=self.device)
        if left_grip is not None:
            action[:, 5] = left_grip.to(self.device)

        if right_grip is not None:
            action[:, 11] = right_grip.to(self.device)

        # Convert world-space target EEF poses to joint-space actions via IK.
        def _fill_arm_action(
            arm_name: str,
            target_pose: torch.Tensor | None,
            gripper: torch.Tensor | None,
            action_col_offset: int,
        ):
            arm = self.scene[arm_name]
            if target_pose is None:
                # Keep current joint target when no target pose is provided.
                action[:, action_col_offset:action_col_offset + 6] = arm.data.joint_pos[env_id:env_id + 1].expand(
                    num_envs, -1
                )
                if gripper is not None:
                    action[:, action_col_offset + 5] = gripper.to(self.device)
                return

            if not self._init_ik_solver_if_needed() or self._ik_solver is None:
                # IK unavailable: keep current joints but still apply gripper command.
                action[:, action_col_offset:action_col_offset + 6] = arm.data.joint_pos[env_id:env_id + 1].expand(
                    num_envs, -1
                )
                if gripper is not None:
                    action[:, action_col_offset + 5] = gripper.to(self.device)
                return

            for i in range(num_envs):
                env_i = env_id if num_envs == 1 else min(i, self.num_envs - 1)
                current_joints_sim = arm.data.joint_pos[env_i].detach().cpu().numpy()
                current_joints = self._sim_joints_to_pinocchio_convention(current_joints_sim)
                T_world_target = target_pose[i].detach().cpu().numpy()
                T_base_target = self._world_pose_to_base_pose_np(arm_name, env_i, T_world_target)
                self._cache_ik_input_eef_pose(arm_name, env_i, T_base_target)
                quat_base_xyzw = mat_to_quat(T_base_target[:3, :3])
                gripper_val = float(gripper[i].item()) if gripper is not None else float(current_joints[5])
                ee_pose = np.concatenate([T_base_target[:3, 3], quat_base_xyzw, [gripper_val]], axis=0)

                joint_targets, ik_diag = compute_joints_from_ee_pose(
                    self._ik_solver,
                    current_joints=current_joints,
                    ee_pose=ee_pose,
                    state_unit="rad",
                    orientation_weight=ik_orientation_weight,
                    return_diagnostics=True,
                )
                self._record_ik_diagnostics(arm_name, ik_diag if joint_targets is not None else None)
                if joint_targets is None:
                    joint_targets = current_joints.copy()
                    joint_targets[5] = gripper_val
                joint_targets = self._pinocchio_joints_to_sim_convention(joint_targets)
                # Clamp to USD joint limits to prevent physics instability at hard stops
                for j, name in enumerate(ACTION_NAMES[:6]):
                    lo_deg, hi_deg = SO101_FOLLOWER_USD_JOINT_LIMLITS[name]
                    joint_targets[j] = np.clip(joint_targets[j], np.radians(lo_deg), np.radians(hi_deg))
                action[i, action_col_offset:action_col_offset + 6] = torch.as_tensor(
                    joint_targets[:6], device=self.device, dtype=torch.float32
                )

        _fill_arm_action("left_arm", left_target, left_grip, 0)
        _fill_arm_action("right_arm", right_target, right_grip, 6)

        return action

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Converts step action to target pose dictionary for end effectors."""
        if action.ndim == 1:
            action = action.unsqueeze(0)
        num_envs = action.shape[0]
        if int(action.shape[-1]) == 16:
            left_target = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)
            right_target = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)

            def _decode_arm_target(arm_name: str, action_col_offset: int) -> torch.Tensor:
                target_pose = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)
                for i in range(num_envs):
                    env_i = min(i, self.num_envs - 1)
                    pos_base = action[i, action_col_offset : action_col_offset + 3]
                    quat_base_wxyz = action[i, action_col_offset + 3 : action_col_offset + 7]
                    T_base_target = torch.eye(4, device=self.device, dtype=torch.float32)
                    T_base_target[:3, 3] = pos_base
                    T_base_target[:3, :3] = PoseUtils.matrix_from_quat(quat_base_wxyz.unsqueeze(0))[0]
                    T_world_target = self._base_pose_to_world_pose_np(
                        arm_name, env_i, T_base_target.detach().cpu().numpy()
                    )
                    target_pose[i] = torch.as_tensor(T_world_target, device=self.device, dtype=torch.float32)
                return target_pose

            left_target = _decode_arm_target("left_arm", 0)
            right_target = _decode_arm_target("right_arm", 8)
            return {"left_arm": left_target, "right_arm": right_target}

        if action.shape[-1] < 12:
            return {
                "left_arm": self.get_robot_eef_pose("left_arm"),
                "right_arm": self.get_robot_eef_pose("right_arm"),
            }

        left_joint_targets = action[:, :6]
        right_joint_targets = action[:, 6:12]
        left_target = self._compute_target_pose_from_joint_targets("left_arm", left_joint_targets)
        right_target = self._compute_target_pose_from_joint_targets("right_arm", right_joint_targets)
        if left_target.shape[0] != num_envs:
            left_target = left_target[:1].expand(num_envs, -1, -1).clone()
        if right_target.shape[0] != num_envs:
            right_target = right_target[:1].expand(num_envs, -1, -1).clone()
        return {"left_arm": left_target, "right_arm": right_target}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extracts gripper actions from environment actions."""
        action_dim = int(actions.shape[-1])
        # Support both action contracts used by this project:
        # - 12D joint targets: [left(6), right(6)]
        # - 16D IK targets:    [left(pos+quat+grip=8), right(...=8)]
        if action_dim == 12:
            left_grip = actions[..., 5:6]
            right_grip = actions[..., 11:12]
        elif action_dim == 16:
            left_grip = actions[..., 7:8]
            right_grip = actions[..., 15:16]
        else:
            raise ValueError(
                f"Unsupported action dimension {action_dim} for gripper extraction. "
                "Expected 12 (joint) or 16 (ik)."
            )

        return {
            "left_arm": left_grip,
            "right_arm": right_grip,
        }

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Get virtual 4x4 object poses from garment checkpoints."""
        if env_ids is None:
            num_envs = self.num_envs
        elif isinstance(env_ids, slice):
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        def _semantic_keypoints_from_positions(kp_positions: np.ndarray) -> dict[str, np.ndarray]:
            return map_semantic_keypoints_from_positions(kp_positions)

        def _pos_to_4x4(pos: torch.Tensor) -> torch.Tensor:
            batch_shape = pos.shape[:-1]
            T = torch.eye(4, device=pos.device, dtype=pos.dtype).expand(*batch_shape, 4, 4).clone()
            T[..., :3, 3] = pos
            return T

        def _raise_unavailable(reason: str) -> None:
            raise ClothObjectPoseUnavailableError(
                f"Failed to query garment object poses for {type(self).__name__}: {reason}"
            )

        garment_obj = getattr(self, "object", None)
        if garment_obj is None or not hasattr(garment_obj, "check_points"):
            _raise_unavailable("garment object is missing or has no check_points.")

        check_points = garment_obj.check_points
        if not check_points or len(check_points) < 6:
            _raise_unavailable("garment check_points are missing or incomplete.")

        try:
            kp_positions = garment_obj.get_checkpoint_world_positions(
                check_points,
                as_numpy=True,
            )
        except Exception as exc:
            _raise_unavailable(
                f"unable to read garment checkpoint positions from GarmentObject: {exc}"
            )

        kp_positions = np.asarray(kp_positions, dtype=np.float32)
        semantic_points = _semantic_keypoints_from_positions(kp_positions)
        object_poses = {}
        for name, point in semantic_points.items():
            pos = (
                torch.tensor(point, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .expand(num_envs, -1)
            )
            object_poses[name] = _pos_to_4x4(pos)

        validate_semantic_object_pose_dict(
            object_poses,
            context=f"{type(self).__name__}.get_object_poses",
        )
        return object_poses

    def get_subtask_start_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        return {}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Compute instantaneous subtask termination signals from shared observation helpers."""
        signal_map = get_subtask_signal_observations(self, env_ids=env_ids)
        return {
            signal_name: signal.to(dtype=torch.float32)
            for signal_name, signal in signal_map.items()
        }

    # ------------------------------------------------------------------
    # Subtask verification hook (used by MimicGen DataGenerator)
    # ------------------------------------------------------------------

    # Only these subtasks carry information that, if missed, guarantees the
    # final fold will fail.  Other subtasks (``*_at_waiting_pos``,
    # ``*_lower_to_upper``, ``*_return_home``) either don't gate success
    # (waiting / home) or are re-checked by the final success term
    # (``*_lower_to_upper`` -> fold success).  Verifying them here would
    # discard otherwise-recoverable episodes because, for example, a brief
    # joint excursion leaves ``*_at_waiting_pos`` False at the exact moment
    # the offset range elapses, yet the eventual fold still lands.
    # Subtask numbering (prepare_for_grasp_* inserted before both grasps):
    #   0 -> prepare_for_grasp_*_middle
    #   1 -> grasp_*_middle
    #   2 -> *_middle_to_lower        (carry, gripper still closed)
    #   3 -> release_*_middle         (open, drop zone, middle kp low)
    #   4 -> *_at_waiting_pos
    #   5 -> prepare_for_grasp_*_lower
    #   6 -> grasp_*_lower
    #   7 -> *_lower_to_upper
    #   8 -> *_return_home
    #
    # Verified subtasks: only the two grasps.  The other critical signals
    # (carry, release) are sensitive to DataGenerator's subtask_term_offset
    # padding — they evaluate False at the verify moment even on correct
    # trajectories because the arm has usually moved on or the gripper has
    # opened/closed again by then.  The grasps are stable end-of-window
    # checkpoints and are enough to catch fundamentally broken rollouts;
    # the rest is re-validated by the final fold-success check.
    _VERIFIED_SUBTASK_INDICES: tuple[int, ...] = (1, 6)

    # Zero-based index of the ``*_lower_to_upper`` subtask (the actual fold).
    # Only after both arms have completed this subtask does the fold-success
    # check become meaningful — see ``is_final_fold_complete``.
    _LOWER_TO_UPPER_SUBTASK_INDEX: int = 7

    def verify_subtask_completion(
        self,
        *,
        arm_name: str,
        subtask_index: int,
        env_id: int,
    ) -> str | None:
        """Check whether a just-finished subtask achieved its semantic goal.

        The MimicGen data generator calls this after a subtask trajectory has
        fully run. Returning ``None`` means "ok, proceed to the next subtask";
        returning a non-empty string aborts the current trial with that string
        as the ``fail_reason`` written to the failed-episode dataset.

        Only the subtasks in ``_VERIFIED_SUBTASK_INDICES`` are verified — by
        default the two grasp subtasks (``grasp_*_middle``, ``grasp_*_lower``),
        the pre-release transfer (``*_middle_to_lower``), and the release
        itself (``release_*_middle``).  Failing any of these means the arm
        either never picked up the cloth, never reached the drop zone, or
        failed to actually detach the cloth in the zone — none of which the
        fold can recover from.  Other subtasks (``*_at_waiting_pos``,
        ``*_lower_to_upper``, ``*_return_home``) are left to the final
        success term so we don't discard trials for transient joint-range
        excursions that the episode recovers from.

        For the verified indices, the implementation reads the
        ``subtask_term_signal`` declared on the matching :class:`SubTaskConfig`
        and fails the trial when the signal is not True for ``env_id`` at
        trajectory end.

        Subclasses may override with richer checks (e.g. explicit EEF-to-
        keypoint distance, contact flags, per-arm joint tolerances).
        """
        # Bookkeeping runs for every subtask completion, even when the
        # signal check is skipped — it feeds ``is_final_fold_complete``
        # so the generation success term only fires after the fold.
        self._mark_subtask_completed(
            env_id=int(env_id),
            arm_name=arm_name,
            subtask_index=int(subtask_index),
        )

        if int(subtask_index) not in self._VERIFIED_SUBTASK_INDICES:
            return None

        arm_cfgs = getattr(self.cfg, "subtask_configs", {}).get(arm_name)
        if not arm_cfgs or subtask_index < 0 or subtask_index >= len(arm_cfgs):
            return None
        signal_name = getattr(arm_cfgs[subtask_index], "subtask_term_signal", None)
        if not signal_name:
            return None

        env_id_tensor = torch.tensor([int(env_id)], dtype=torch.int64, device=self.device)
        try:
            with self._verify_subtask_threshold_overrides():
                signals = self.get_subtask_term_signals(env_ids=env_id_tensor)
        except Exception as exc:
            logger.debug(
                "verify_subtask_completion: signal eval failed for %s[%d]: %s",
                arm_name,
                int(subtask_index),
                exc,
            )
            return None

        tensor = signals.get(signal_name)
        if tensor is None:
            return None
        try:
            value = float(tensor.reshape(-1)[0].item())
        except Exception:
            return None
        if value > 0.5:
            return None
        return (
            f"subtask_term_signal '{signal_name}' was False at subtask "
            f"{subtask_index} completion ({arm_name})"
        )

    # Map of cfg attributes that ``verify_subtask_completion`` may loosen.
    # Each entry points to the optional ``verify_*`` attribute whose value,
    # when not None, temporarily overrides the strict online threshold.
    _VERIFY_SUBTASK_THRESHOLD_ATTR_MAP: tuple[tuple[str, str], ...] = (
        (
            "subtask_grasp_eef_to_keypoint_threshold_m",
            "verify_subtask_grasp_eef_to_keypoint_threshold_m",
        ),
        (
            "subtask_release_zone_width_fraction",
            "verify_subtask_release_zone_width_fraction",
        ),
        (
            "subtask_release_zone_lower_fraction",
            "verify_subtask_release_zone_lower_fraction",
        ),
    )

    @contextmanager
    def _verify_subtask_threshold_overrides(self):
        """Temporarily swap strict online thresholds for looser verify values.

        The subtask-termination signals are driven by a handful of cfg floats
        (grasp radius, release-zone width/lower fractions).  During normal
        online evaluation we want the tight thresholds that define clean
        annotation boundaries; during MimicGen subtask verification we want
        the grace-margin values so the end-of-window snapshot still evaluates
        True after DataGenerator's offset padding.
        """
        previous: dict[str, Any] = {}
        try:
            for strict_attr, verify_attr in self._VERIFY_SUBTASK_THRESHOLD_ATTR_MAP:
                override = getattr(self.cfg, verify_attr, None)
                if override is None:
                    continue
                previous[strict_attr] = getattr(self.cfg, strict_attr)
                setattr(self.cfg, strict_attr, override)
            yield
        finally:
            for strict_attr, original in previous.items():
                setattr(self.cfg, strict_attr, original)

    # ------------------------------------------------------------------
    # Subtask-completion bookkeeping (drives the success-check gate)
    # ------------------------------------------------------------------

    def _mark_subtask_completed(
        self,
        *,
        env_id: int,
        arm_name: str,
        subtask_index: int,
    ) -> None:
        """Record that ``arm_name`` has finished ``subtask_index`` for ``env_id``."""
        per_env = self._completed_subtasks.setdefault(int(env_id), {})
        per_arm = per_env.setdefault(str(arm_name), set())
        per_arm.add(int(subtask_index))

    def _clear_completed_subtasks(self, env_ids: Sequence[int] | None) -> None:
        """Drop subtask-completion state for the given env ids (all if None)."""
        if env_ids is None:
            self._completed_subtasks.clear()
            return
        for eid in env_ids:
            try:
                self._completed_subtasks.pop(int(eid), None)
            except (TypeError, ValueError):
                continue

    def is_final_fold_complete(self, env_id: int) -> bool:
        """Return True when every configured arm has finished ``*_lower_to_upper``.

        The MimicGen success term (``recording_style_success_tensor``) uses
        this gate to skip the expensive garment-geometry checker until the
        fold motion itself is done.  Before that point the cloth is mid-flight
        and the success signal is meaningless; after that point, each step
        inside ``*_return_home`` is a legitimate opportunity for the fold to
        settle into a passing configuration.
        """
        completed = self._completed_subtasks.get(int(env_id))
        if not completed:
            return False
        arm_names = getattr(self.cfg, "subtask_configs", {}).keys()
        if not arm_names:
            return False
        target_index = int(self._LOWER_TO_UPPER_SUBTASK_INDEX)
        for arm_name in arm_names:
            if target_index not in completed.get(str(arm_name), set()):
                return False
        return True

    def record_failed_episode_minimal(
        self,
        *,
        env_id: int,
        source_demo_selections: Mapping[str, Sequence[int]] | None,
        fail_reason: str,
    ) -> None:
        """Overwrite the current episode buffer with a minimal failed record.

        Only the fields needed to re-try generation from the same initial
        conditions are kept:

        - ``initial_state/garment_initial_pose`` — garment pose at reset.
        - ``source_demo_indices/<arm>`` — source demo indices actually
          selected per arm, up to the failing subtask (inclusive).
        - ``fail_reason`` — utf-8 byte encoding of the reason string.

        All accumulated per-step observations, actions, and scene states from
        the aborted trajectory are discarded. The episode is marked as
        ``success=False`` and exported immediately via the recorder manager
        (which routes failed rows to the failed-episode handler when the
        ``EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES`` mode is active).
        """
        from isaaclab.utils.datasets import EpisodeData

        recorder = getattr(self, "recorder_manager", None)
        if recorder is None or not getattr(recorder, "active_terms", []):
            return

        minimal = EpisodeData()
        minimal.env_id = int(env_id)
        minimal.success = False

        new_data: Dict[str, Any] = {}

        garment_obj = getattr(self, "object", None)
        garment_reset_pose = (
            getattr(garment_obj, "reset_pose", None) if garment_obj is not None else None
        )
        if garment_reset_pose is not None:
            # Match the normal-path shape (1, 7) written by DataGenerator so
            # downstream readers don't have to special-case failed rows.
            pose_tensor = torch.as_tensor(
                garment_reset_pose, dtype=torch.float32, device=self.device,
            ).detach().clone().reshape(-1).unsqueeze(0)
            new_data["initial_state"] = {"garment_initial_pose": pose_tensor}

        src_group: Dict[str, torch.Tensor] = {}
        for arm_name, selections in (source_demo_selections or {}).items():
            selections_list = [int(v) for v in selections] if selections else []
            if not selections_list:
                continue
            # Match the normal-path shape (1, N) for consistency.
            src_group[str(arm_name)] = torch.tensor(
                [selections_list], dtype=torch.int64, device=self.device,
            )
        if src_group:
            new_data["source_demo_indices"] = src_group

        reason_str = str(fail_reason) if fail_reason else "unknown"
        reason_bytes = reason_str.encode("utf-8")
        if not reason_bytes:
            reason_bytes = b"unknown"
        new_data["fail_reason"] = torch.tensor(
            list(reason_bytes), dtype=torch.uint8, device=self.device,
        )

        minimal.data = new_data

        recorder._episodes[int(env_id)] = minimal
        try:
            recorder.export_episodes(env_ids=[int(env_id)])
        except Exception as exc:
            logger.error(
                "record_failed_episode_minimal: export_episodes failed for env %d: %s",
                int(env_id),
                exc,
            )


class GarmentFoldMimicEnv(GarmentFoldEnv):
    """Garment fold env variant that enables native mimic IK unless Pinocchio is forced."""

    def __init__(self, cfg: GarmentFoldEnvCfg, render_mode: str | None = None, **kwargs):
        task_type = str(getattr(cfg, "task_type", "bi-so101leader"))
        base_task_type = task_type[len("mimic_"):] if task_type.startswith("mimic_") else task_type
        if bool(getattr(cfg, "force_pinocchio_generation", False)):
            cfg.use_teleop_device(base_task_type)
        else:
            mimic_task_type = task_type if task_type.startswith("mimic_") else f"mimic_{task_type}"
            cfg.use_teleop_device(mimic_task_type)
        # Keep runtime task type for utility helpers that expect non-mimic labels.
        cfg.task_type = base_task_type
        super().__init__(cfg, render_mode=render_mode, **kwargs)
