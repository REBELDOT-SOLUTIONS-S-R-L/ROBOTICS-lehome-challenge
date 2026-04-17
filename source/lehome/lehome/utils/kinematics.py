# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class RobotKinematics:
    """Forward/Inverse kinematics tool based on Pinocchio (for lehome project)."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        """
        Initialize Pinocchio kinematics solver.

        Args:
            urdf_path: Path to robot URDF file
            target_frame_name: End-effector frame name
            joint_names: List of joint names for solving; None uses all movable joints in the model
        """
        self.urdf_path = urdf_path
        self.target_frame_name = target_frame_name
        self.joint_names = joint_names
        self.backend = "pinocchio"

        self._init_pinocchio()
    
    def _init_pinocchio(self):
        """Initialize Pinocchio-based kinematics solver."""
        try:
            import pinocchio as pin
        except ImportError:
            raise ImportError(
                "Pinocchio is required but not available. "
                "Please install it with: pip install pin"
            )
        
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError(
                "scipy is required for inverse kinematics with Pinocchio. "
                "Please install it with: pip install scipy"
            )

        def try_build_model(pin_module):
            """Try to build a Pinocchio model from multiple API variants."""
            model_local = None
            data_local = None

            # API variant 1: module function buildModelFromUrdf
            builder = getattr(pin_module, "buildModelFromUrdf", None)
            if callable(builder):
                model_local = builder(self.urdf_path)

            # API variant 2: module function buildModelFromURDF
            if model_local is None:
                builder = getattr(pin_module, "buildModelFromURDF", None)
                if callable(builder):
                    model_local = builder(self.urdf_path)

            # API variant 3: Model class static constructor
            if model_local is None:
                model_cls = getattr(pin_module, "Model", None)
                if model_cls is not None:
                    for method_name in ("BuildFromUrdf", "BuildFromURDF"):
                        method = getattr(model_cls, method_name, None)
                        if callable(method):
                            model_local = method(self.urdf_path)
                            break

            # API variant 4: RobotWrapper constructor
            if model_local is None:
                robot_wrapper_cls = getattr(pin_module, "RobotWrapper", None)
                if robot_wrapper_cls is not None:
                    for method_name in ("BuildFromURDF", "BuildFromUrdf"):
                        method = getattr(robot_wrapper_cls, method_name, None)
                        if not callable(method):
                            continue
                        try:
                            robot = method(self.urdf_path, [])
                        except TypeError:
                            robot = method(self.urdf_path)
                        model_local = getattr(robot, "model", None)
                        data_local = getattr(robot, "data", None)
                        if model_local is not None:
                            break

            return model_local, data_local

        # Try imported `pinocchio` first.
        model, data = try_build_model(pin)
        selected_pin_module = pin

        # Fallback: some environments expose robotics Pinocchio as `pin`.
        if model is None:
            try:
                import pin as pin_fallback
            except ImportError:
                pin_fallback = None
            if pin_fallback is not None:
                model, data = try_build_model(pin_fallback)
                if model is not None:
                    selected_pin_module = pin_fallback

        if model is None:
            module_path = getattr(pin, "__file__", "unknown")
            module_version = getattr(pin, "__version__", "unknown")
            available = ", ".join(
                name
                for name in (
                    "buildModelFromUrdf",
                    "buildModelFromURDF",
                    "Model",
                    "RobotWrapper",
                    "SE3",
                    "neutral",
                )
                if hasattr(pin, name)
            )
            raise RuntimeError(
                "Imported module 'pinocchio' does not expose a compatible robotics API. "
                f"module={module_path}, version={module_version}, available=[{available}]. "
                "This usually means a different PyPI package named 'pinocchio' is installed. "
                "Fix by reinstalling the robotics binding package: `pip uninstall -y pinocchio && pip install pin`."
            )

        self._pin = selected_pin_module

        self.model = model
        self.data = data if data is not None else self.model.createData()
        
        # Get end-effector frame ID
        try:
            self.ee_frame_id = self.model.getFrameId(self.target_frame_name)
        except Exception:
            # Try to find frame by name pattern
            frame_found = False
            for i in range(self.model.nframes):
                frame_name = self.model.frames[i].name
                if self.target_frame_name in frame_name or frame_name in self.target_frame_name:
                    self.ee_frame_id = i
                    frame_found = True
                    break
            if not frame_found:
                raise ValueError(
                    f"Frame '{self.target_frame_name}' not found in URDF. "
                    f"Available frames: {[self.model.frames[i].name for i in range(self.model.nframes)]}"
                )
        
        # Set joint names
        if self.joint_names is None:
            # Get all revolute and prismatic joints
            self.joint_names = []
            for i in range(1, self.model.njoints):  # Skip root joint
                joint_name = self.model.names[i]
                joint_model = self.model.joints[i]
                if joint_model.nq > 0:  # Has configuration space
                    self.joint_names.append(joint_name)
        else:
            self.joint_names = self.joint_names
        
        # Get joint indices and configuration indices
        self.joint_indices = []
        self.joint_q_indices = []  # Indices in the configuration vector
        for joint_name in self.joint_names:
            try:
                joint_id = self.model.getJointId(joint_name)
                self.joint_indices.append(joint_id)
                joint_model = self.model.joints[joint_id]
                if joint_model.nq > 0:
                    self.joint_q_indices.append(joint_model.idx_q)
            except Exception:
                raise ValueError(f"Joint '{joint_name}' not found in URDF model")
        
        self.nq = len(self.joint_indices)
        
        # Store minimize function for IK
        self._minimize = minimize

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics (Pinocchio) for given joint configuration.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """
        return self._forward_kinematics_pinocchio(joint_pos_deg)
    
    def _forward_kinematics_pinocchio(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Forward kinematics using Pinocchio."""
        pin = self._pin
        
        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_pos_deg[: self.nq])
        
        # Create full configuration vector (including fixed joints)
        q = pin.neutral(self.model)
        for i, q_idx in enumerate(self.joint_q_indices):
            q[q_idx] = joint_pos_rad[i]
        
        # Compute forward kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get end-effector pose (4x4 homogeneous matrix)
        ee_pose = self.data.oMf[self.ee_frame_id]
        return ee_pose.homogeneous

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        joint_bounds_rad: list[tuple[float, float]] | None = None,
        return_diagnostics: bool = False,
    ):
        """
        Compute inverse kinematics using Pinocchio + scipy.optimize.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK, set to 0.0 to only constrain position
            joint_bounds_rad: Optional per-joint (lower, upper) bounds in radians.
                When provided these replace the URDF model limits for the
                L-BFGS-B optimisation.  Length must equal ``self.nq``.
            return_diagnostics: When True, also return a dict with post-solve
                residuals and optimizer status.

        Returns:
            If ``return_diagnostics`` is False (default): joint positions in
            degrees that achieve (approximately) the desired end-effector pose.

            If ``return_diagnostics`` is True: tuple ``(joint_pos_deg, diag)``
            where ``diag`` contains:

              - ``pos_residual_m`` (float): ``||FK(q*) - target||`` in metres.
              - ``rot_residual_rad`` (float): ``||log3(R_FK.T @ R_target)||``
                in radians. Always computed even when ``orientation_weight``
                is small, so callers can flag orientation drift.
              - ``cost`` (float): final objective value.
              - ``converged`` (bool): whether L-BFGS-B reported convergence.
              - ``hit_bound`` (bool): whether any joint settled on a hard
                bound (within 1e-3 rad). A strong hint that the bound itself
                is what's keeping the optimizer from finding the target.
              - ``nit`` (int): number of L-BFGS-B iterations used.
        """
        return self._inverse_kinematics_pinocchio(
            current_joint_pos, desired_ee_pose, position_weight, orientation_weight,
            joint_bounds_rad=joint_bounds_rad,
            return_diagnostics=return_diagnostics,
        )

    def _inverse_kinematics_pinocchio(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float,
        orientation_weight: float,
        joint_bounds_rad: list[tuple[float, float]] | None = None,
        return_diagnostics: bool = False,
    ):
        """Inverse kinematics using Pinocchio with scipy.optimize."""
        pin = self._pin

        # Target pose as Pinocchio SE3
        target_pose = pin.SE3(desired_ee_pose[:3, :3], desired_ee_pose[:3, 3])

        # Initial guess (convert to radians)
        current_joint_rad = np.deg2rad(current_joint_pos[: self.nq])
        q0_controlled = np.array(current_joint_rad)

        # Optimization objective function
        def objective(q_controlled):
            # Reconstruct full configuration
            q = pin.neutral(self.model)
            for i, q_idx in enumerate(self.joint_q_indices):
                q[q_idx] = q_controlled[i]

            # Forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current_pose = self.data.oMf[self.ee_frame_id]

            # Position error
            pos_error = target_pose.translation - current_pose.translation
            pos_cost = position_weight * np.sum(pos_error ** 2)

            # Orientation error (logarithmic map)
            if orientation_weight > 0:
                rot_error = pin.log3(target_pose.rotation.T @ current_pose.rotation)
                rot_cost = orientation_weight * np.sum(rot_error ** 2)
            else:
                rot_cost = 0.0

            return pos_cost + rot_cost

        # Joint limits (bounds)
        if joint_bounds_rad is not None:
            bounds = list(joint_bounds_rad)
        else:
            bounds = []
            for q_idx in self.joint_q_indices:
                lower = self.model.lowerPositionLimit[q_idx]
                upper = self.model.upperPositionLimit[q_idx]
                # If limits are infinite, use reasonable defaults
                if not np.isfinite(lower):
                    lower = -np.pi
                if not np.isfinite(upper):
                    upper = np.pi
                bounds.append((lower, upper))

        # Solve IK using optimization
        result = self._minimize(
            objective,
            q0_controlled,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        # Convert result to degrees
        joint_pos_deg = np.rad2deg(result.x)

        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > self.nq:
            result_full = np.zeros_like(current_joint_pos)
            result_full[: self.nq] = joint_pos_deg
            result_full[self.nq :] = current_joint_pos[self.nq :]
            out_joints = result_full
        else:
            out_joints = joint_pos_deg

        if not return_diagnostics:
            return out_joints

        # Post-solve residuals: run FK on the solved joints and measure how
        # far off the achieved pose is from the target. L-BFGS-B does not
        # raise on infeasible targets; it just returns the best feasible
        # point in the bounded domain.  Callers use these residuals to
        # detect silently-unreachable targets.
        q_full = pin.neutral(self.model)
        for i, q_idx in enumerate(self.joint_q_indices):
            q_full[q_idx] = result.x[i]
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        achieved = self.data.oMf[self.ee_frame_id]

        pos_residual_m = float(np.linalg.norm(target_pose.translation - achieved.translation))
        try:
            rot_residual_rad = float(np.linalg.norm(pin.log3(target_pose.rotation.T @ achieved.rotation)))
        except Exception:
            rot_residual_rad = float("nan")

        hit_bound = False
        for q_val, (lo, hi) in zip(result.x, bounds):
            if (q_val - lo) < 1e-3 or (hi - q_val) < 1e-3:
                hit_bound = True
                break

        diag = {
            "pos_residual_m": pos_residual_m,
            "rot_residual_rad": rot_residual_rad,
            "cost": float(result.fun),
            "converged": bool(getattr(result, "success", False)),
            "hit_bound": hit_bound,
            "nit": int(getattr(result, "nit", 0) or 0),
        }
        return out_joints, diag
