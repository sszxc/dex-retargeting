from abc import abstractmethod
from typing import List, Optional

import nlopt
import numpy as np
import torch

from dex_retargeting.kinematics_adaptor import (
    KinematicAdaptor,
    MimicJointKinematicAdaptor,
)
from dex_retargeting.robot_wrapper import RobotWrapper


class Optimizer:
    """Base nonlinear optimizer for hand-to-robot retargeting."""
    retargeting_type = "BASE"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_human_indices: np.ndarray,
    ):
        """
        Args:
            robot: Kinematic wrapper for the robot.
            target_joint_names: Subset of DOF names to optimize.
            target_link_human_indices: Human landmark indices used by this optimizer.
        """
        self.robot = robot
        self.num_joints = robot.dof

        # Indices of optimized joints in pinocchio order
        joint_names = robot.dof_joint_names
        idx_pin2target = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(
                    f"Joint {target_joint_name} given does not appear to be in robot XML."
                )
            idx_pin2target.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.idx_pin2target = np.array(idx_pin2target)

        # Fixed (non-optimized) joint indices
        self.idx_pin2fixed = np.array(
            [i for i in range(robot.dof) if i not in idx_pin2target], dtype=int
        )
        # NLopt SLSQP on the target DOFs only
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_pin2target))
        self.opt_dof = len(idx_pin2target)  # This dof includes the mimic joints

        # Human indices for supervised links
        self.target_link_human_indices = target_link_human_indices

        # Heuristic: enough dummy links => floating base
        link_names = robot.link_names
        self.has_free_joint = len([name for name in link_names if "dummy" in name]) >= 6

        # Optional mimic / adaptor for nonstandard joints
        self.adaptor: Optional[KinematicAdaptor] = None

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        """Set box constraints on optimized DOFs.

        Args:
            joint_limits: Shape (opt_dof, 2), rows are [lower, upper].
            epsilon: Small slack added to bounds.
        """
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(
                f"Expect joint limits have shape: {(self.opt_dof, 2)}, but get {joint_limits.shape}"
            )
        # NLopt bounds with epsilon slack
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def get_link_indices(self, target_link_names):
        """Pinocchio frame ids for ``target_link_names``."""
        return [self.robot.get_link_index(link_name) for link_name in target_link_names]

    def set_kinematic_adaptor(self, adaptor: KinematicAdaptor):
        """Attach a kinematic adaptor (e.g. mimic joints)."""
        self.adaptor = adaptor

        # Remove mimic joints from fixed joint list
        # Mimic joints are driven by the adaptor, not held fixed
        if isinstance(adaptor, MimicJointKinematicAdaptor):
            fixed_idx = self.idx_pin2fixed
            mimic_idx = adaptor.idx_pin2mimic
            new_fixed_id = np.array(
                [x for x in fixed_idx if x not in mimic_idx], dtype=int
            )
            self.idx_pin2fixed = new_fixed_id

    def retarget(self, ref_value, fixed_qpos, last_qpos):
        """
        Run NLopt on the target joints.

        Args:
            ref_value: Task-specific human reference (positions, vectors, etc.).
            fixed_qpos: Values for ``self.fixed_joint_names`` (not optimized).
            last_qpos: Warm-start / fallback configuration for target joints.

        Returns:
            ``qpos`` aligned with ``self.target_joint_names``.
        """
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but non_target_qpos {fixed_qpos} is given"
            )
        # Build scalar objective for NLopt
        objective_fn = self.get_objective_function(
            ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32)
        )

        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            return np.array(qpos, dtype=np.float32)
        except RuntimeError as e:
            # On NLopt failure, return the previous solution
            print(e)
            return np.array(last_qpos, dtype=np.float32)

    @abstractmethod
    def get_objective_function(
        self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        """Return ``f(x, grad)`` callable for NLopt."""
        pass

    @property
    def fixed_joint_names(self):
        """Names of joints that are not optimized."""
        joint_names = self.robot.dof_joint_names
        return [joint_names[i] for i in self.idx_pin2fixed]


class PositionOptimizer(Optimizer):
    """Retarget by matching 3D link positions (smooth L1 / Huber on errors)."""
    retargeting_type = "POSITION"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
        scaling=1.0,
    ):
        """
        Args:
            robot: Robot wrapper.
            target_joint_names: Optimized joint names.
            target_link_names: Link names whose origins are supervised.
            target_link_human_indices: Human point indices.
            huber_delta: SmoothL1 beta.
            norm_delta: Quadratic penalty on deviation from ``last_qpos``.
            scaling: Human-space scale for the task positions.
        """
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        # Smooth L1 on Cartesian errors
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta
        self.scaling = scaling

        self.target_link_indices = self.get_link_indices(target_link_names)

        self.opt.set_ftol_abs(1e-5)

    def get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        """NLopt objective for 3D position matching."""
        # Full qpos with fixed joints filled in
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor((target_pos - target_pos[0, :]) * self.scaling + target_pos[0, :])
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            """Scalar loss and optional gradient for NLopt."""
            # Write optimized DOFs
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            # Forward kinematics with optional adaptor
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            # FK for supervised links
            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.target_link_indices
            ]
            # Link origins (translation columns)
            body_pos = np.stack(
                [pose[:3, 3] for pose in target_link_poses], axis=0
            )  # (n ,3)

            # Torch computation for accurate loss and grad
            # Torch autograd for loss + ∂loss/∂xyz
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            # Main retargeting term
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                # Stack position Jacobians
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    # Local Jacobian, position rows only
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    # World-frame translational Jacobian
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: joint order matches pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Jacobian columns → target DOF order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, jacobians)
                grad_qpos = grad_qpos.mean(1).sum(0)
                # Tikhonov-style pull toward previous qpos
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class VectorOptimizer(Optimizer):
    """Retarget by matching 3D vectors between link pairs (relative geometry)."""
    retargeting_type = "VECTOR"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
        scaling=1.0,
    ):
        """
        Args:
            robot: Robot wrapper.
            target_joint_names: Optimized joints.
            target_origin_link_names: Vector tail links.
            target_task_link_names: Vector tip links.
            target_link_human_indices: Human indices for the task.
            huber_delta: SmoothL1 on vector-error norms.
            norm_delta: Regularization vs. ``last_qpos``.
            scaling: Human vector scale.
        """
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta
        self.scaling = scaling

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        # Deduplicate links touched by multiple vectors
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        # Tensor indices into ``computed_link_names``
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Cache link indices that will involve in kinematics computation
        # Pinocchio indices for cached links
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        """NLopt objective for vector retargeting."""
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        # Human vectors as torch (scaled)
        torch_target_vec = torch.as_tensor(target_vector) * self.scaling
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            """Scalar loss for one NLopt iteration."""
            # Optimized DOFs
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            # Forward kinematics with optional adaptor
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            # Torch autograd for loss + ∂loss/∂xyz
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Gather origin/task 3D points
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            # Main retargeting term
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                # Stack position Jacobians
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    # Local translational Jacobian (first 3 rows)
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    # World-frame translational Jacobian
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: joint order matches pinocchio
                jacobians = np.stack(jacobians, axis=0)  # (n_links, 3, nv)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Columns → target DOFs
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class DexPilotOptimizer(Optimizer):
    """DexPilot-style retargeting (Tomasello et al., arXiv:1910.03135).

    Generalizes the original four-finger Allegro formulation to 2–5 fingers by
    projecting thumb–finger distances for stabler grasps.

    Args:
        robot: Robot wrapper.
        target_joint_names: Optimized joints.
        finger_tip_link_names: Fingertip link names.
        wrist_link_name: Palm / wrist link name.
        gamma: Unused legacy regularizer (commented in original DexPilot).
        project_dist: Enable projection when human vector length is below this.
        escape_dist: Disable projection above this length.
        eta1, eta2: Reference lengths for first/second projection tiers.
        scaling: Human vector scale.
    """

    retargeting_type = "DEXPILOT"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        target_link_human_indices: Optional[np.ndarray] = None,
        huber_delta=0.03,
        norm_delta=4e-3,
        # DexPilot parameters
        # gamma=2.5e-3,
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
    ):
        if len(finger_tip_link_names) < 2 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"DexPilot optimizer can only be applied to hands with 2 to 5 fingers, but got "
                f"{len(finger_tip_link_names)} fingers."
            )
        self.num_fingers = len(finger_tip_link_names)

        origin_link_index, task_link_index = self.generate_link_indices(
            self.num_fingers
        )

        if target_link_human_indices is None:
            target_link_human_indices = (
                np.stack([origin_link_index, task_link_index], axis=0) * 4
            ).astype(int)
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]

        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.scaling = scaling
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta

        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        # Deduplicate links touched by multiple vectors
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        # Tensor indices into ``computed_link_names``
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

        (
            self.projected,
            self.s2_project_index_origin,
            self.s2_project_index_task,
            self.projected_dist,
        ) = self.set_dexpilot_cache(self.num_fingers, eta1, eta2)

    @staticmethod
    def generate_link_indices(num_fingers):
        """DexPilot finger–finger and wrist–finger edge list indices into ``[wrist, tip_1, …]``.

        Example:
        >>> generate_link_indices(4)
        ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
        """
        origin_link_index = []
        task_link_index = []

        # Inter-finger chords
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)

        # Wrist (index 0) to each fingertip
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)

        return origin_link_index, task_link_index

    @staticmethod
    def set_dexpilot_cache(num_fingers, eta1, eta2):
        """Allocate boolean projection flags and reference lengths for DexPilot tiers."""
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

        # Second-tier projection pairs (indices into first-tier edges)
        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)

        projected_dist = np.array(
            [eta1] * (num_fingers - 1)
            + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2)
        )

        return projected, s2_project_index_origin, s2_project_index_task, projected_dist

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)

        # Up-weight wrist–tip residuals (more robust to bad hand poses)
        weight = torch.from_numpy(
            np.concatenate(
                [
                    weight,
                    np.ones(self.num_fingers, dtype=np.float32) * len_proj
                    + self.num_fingers,
                ]
            )
        )

        normal_vec = target_vector * self.scaling
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)
        projected_vec = dir_vec * self.projected_dist[:, None]

        reference_vec = np.where(
            self.projected[:, None], projected_vec, normal_vec[:len_proj]
        )
        reference_vec = np.concatenate(
            [reference_vec, normal_vec[len_proj:]], axis=0
        )
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            # Different from the original DexPilot, we use huber loss here instead of the squared dist
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = (
                self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
                * weight
                / (robot_vec.shape[0])
            ).sum()
            huber_distance = huber_distance.sum()
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                # In the original DexPilot, γ = 2.5 × 10−3 is a weight on regularizing the Allegro angles to zero
                # which is equivalent to fully opened the hand
                # In our implementation, we regularize the joint angles to the previous joint angles
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


def allegro_left_dummy_qpos_from_leap_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    """Map 21 Leap/MediaPipe points to Allegro left ``qpos`` (6 dummy + 16 finger joints = 22)."""
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    result = np.zeros(22, dtype=np.float64)
    p0, p5, p9, p13 = joint_pos[0], joint_pos[5], joint_pos[9], joint_pos[13]
    v1 = p5 - p0
    v2 = p9 - p0
    v3 = p13 - p0
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    n3 = np.cross(v3, v1)
    plane_normal = (n1 + n2 + n3) / 3.0
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    x_dir = p13 - p0
    x_dir = x_dir / np.linalg.norm(x_dir)
    y_dir = plane_normal
    z_dir = np.cross(x_dir, y_dir)
    z_dir = z_dir / np.linalg.norm(z_dir)
    y_dir = np.cross(z_dir, x_dir)
    y_dir = y_dir / np.linalg.norm(y_dir)
    rotmat = np.stack([x_dir, y_dir, z_dir], axis=1)
    SHIFTED = np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ],
        dtype=np.float64,
    )
    rotmat = rotmat @ SHIFTED
    import scipy.spatial.transform

    euler_xyz = scipy.spatial.transform.Rotation.from_matrix(rotmat).as_euler("XYZ")
    result[3:6] = euler_xyz
    result[0:3] = joint_pos[0] + 0.03 * x_dir

    def angle_between_vectors(v1, v2):
        if np.linalg.norm(v1) < 1e-8 or np.linalg.norm(v2) < 1e-8:
            return 0.0
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.arccos(dot)

    def angle_between_plane_and_vector(plane_vector1, plane_vector2, vector):
        if np.linalg.norm(plane_vector1) < 1e-8 or np.linalg.norm(plane_vector2) < 1e-8:
            return 0.0
        if np.linalg.norm(vector) < 1e-8:
            return 0.0
        plane_normal = np.cross(plane_vector1, plane_vector2)
        norm_normal = np.linalg.norm(plane_normal)
        if norm_normal < 1e-8:
            return 0.0
        plane_normal = plane_normal / norm_normal
        vector_norm = vector / np.linalg.norm(vector)
        dot = np.clip(np.dot(plane_normal, vector_norm), -1.0, 1.0)
        angle_with_normal = np.arccos(dot)
        return abs(np.pi / 2.0 - angle_with_normal)

    from dex_retargeting.thumb_retarget import calculate_thumb_angles

    vec_in_new_coord = rotmat.T @ (joint_pos[3] - joint_pos[2])
    _success, _angles, _error, _actual_vec = calculate_thumb_angles(vec_in_new_coord)
    result[10:13] = _angles

    result[13] = angle_between_vectors(
        joint_pos[3] - joint_pos[2], joint_pos[4] - joint_pos[3]
    )

    result[18] = angle_between_plane_and_vector(
        joint_pos[7] - joint_pos[6], joint_pos[8] - joint_pos[7], y_dir
    )
    result[21] = angle_between_vectors(
        joint_pos[7] - joint_pos[6], joint_pos[8] - joint_pos[7]
    )
    result[19] = angle_between_vectors(
        joint_pos[6] - joint_pos[5], joint_pos[7] - joint_pos[6]
    )
    result[20] = 0.5 * (result[19] + result[21])

    result[14] = angle_between_plane_and_vector(
        joint_pos[11] - joint_pos[10], joint_pos[12] - joint_pos[11], y_dir
    )
    result[17] = angle_between_vectors(
        joint_pos[11] - joint_pos[10], joint_pos[12] - joint_pos[11]
    )
    result[16] = angle_between_vectors(
        joint_pos[10] - joint_pos[9], joint_pos[11] - joint_pos[10]
    )
    result[15] = 0.5 * (result[14] + result[16])

    result[6] = angle_between_plane_and_vector(
        joint_pos[15] - joint_pos[14], joint_pos[16] - joint_pos[15], y_dir
    )
    result[9] = angle_between_vectors(
        joint_pos[15] - joint_pos[14], joint_pos[16] - joint_pos[15]
    )
    result[8] = angle_between_vectors(
        joint_pos[14] - joint_pos[13], joint_pos[15] - joint_pos[14]
    )
    result[7] = 0.5 * (result[6] + result[8])
    return result.astype(np.float32)


def hmf_proto5_left_dummy_qpos_from_leap_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    """Proto5 left hand: Allegro-style geometry, pinocchio order (dummy6 + WRZ/WRY + 4×4 finger joints = 24)."""
    # ---- Begin inlined copy of `allegro_left_dummy_qpos_from_leap_joint_pos` ----
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    a = np.zeros(22, dtype=np.float64)
    p0, p5, p9, p13 = joint_pos[0], joint_pos[5], joint_pos[9], joint_pos[13]
    v1 = p5 - p0
    v2 = p9 - p0
    v3 = p13 - p0
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    n3 = np.cross(v3, v1)
    plane_normal = (n1 + n2 + n3) / 3.0
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    x_dir = p13 - p0
    x_dir = x_dir / np.linalg.norm(x_dir)
    y_dir = plane_normal
    z_dir = np.cross(x_dir, y_dir)
    z_dir = z_dir / np.linalg.norm(z_dir)
    y_dir = np.cross(z_dir, x_dir)
    y_dir = y_dir / np.linalg.norm(y_dir)
    rotmat = np.stack([x_dir, y_dir, z_dir], axis=1)
    SHIFTED = np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ],
        dtype=np.float64,
    )
    rotmat = rotmat @ SHIFTED
    import scipy.spatial.transform

    euler_xyz = scipy.spatial.transform.Rotation.from_matrix(rotmat).as_euler("XYZ")
    a[3:6] = euler_xyz
    a[0:3] = joint_pos[0] + 0.03 * x_dir

    def angle_between_vectors(v1, v2):
        if np.linalg.norm(v1) < 1e-8 or np.linalg.norm(v2) < 1e-8:
            return 0.0
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.arccos(dot)

    def angle_between_plane_and_vector(plane_vector1, plane_vector2, vector):
        if np.linalg.norm(plane_vector1) < 1e-8 or np.linalg.norm(plane_vector2) < 1e-8:
            return 0.0
        if np.linalg.norm(vector) < 1e-8:
            return 0.0
        plane_normal = np.cross(plane_vector1, plane_vector2)
        norm_normal = np.linalg.norm(plane_normal)
        if norm_normal < 1e-8:
            return 0.0
        plane_normal = plane_normal / norm_normal
        vector_norm = vector / np.linalg.norm(vector)
        dot = np.clip(np.dot(plane_normal, vector_norm), -1.0, 1.0)
        angle_with_normal = np.arccos(dot)
        return abs(np.pi / 2.0 - angle_with_normal)

    # from dex_retargeting.thumb_retarget import calculate_thumb_angles

    # # thumb
    # vec_in_new_coord = rotmat.T @ (joint_pos[3] - joint_pos[2])
    # _success, _angles, _error, _actual_vec = calculate_thumb_angles(vec_in_new_coord)
    # a[10:13] = _angles
    # angle between [0-4] and xy plane
    a[10] = angle_between_plane_and_vector(joint_pos[0] - joint_pos[4], x_dir, y_dir) * -1.5
    a[11] = angle_between_vectors(joint_pos[3] - joint_pos[2], joint_pos[4] - joint_pos[3]) * 1.2
    a[12] = angle_between_vectors(joint_pos[3] - joint_pos[2], joint_pos[4] - joint_pos[3]) * 1.2
    a[13] = angle_between_vectors(joint_pos[3] - joint_pos[2], joint_pos[4] - joint_pos[3]) * 1.2

    # index
    # a[18] = angle_between_plane_and_vector(
    #     joint_pos[7] - joint_pos[6], joint_pos[8] - joint_pos[7], y_dir
    # )
    a[18] = angle_between_vectors(joint_pos[5] - joint_pos[8], joint_pos[9] - joint_pos[12]) * 2 - 0.3
    a[21] = angle_between_vectors(joint_pos[7] - joint_pos[6], joint_pos[8] - joint_pos[7]) - 0.1
    a[19] = angle_between_vectors(joint_pos[6] - joint_pos[5], joint_pos[7] - joint_pos[6]) - 0.1
    a[20] = 0.5 * (a[19] + a[21])

    # middle
    a[14] = angle_between_plane_and_vector(
        joint_pos[11] - joint_pos[10], joint_pos[12] - joint_pos[11], y_dir
    ) * 2 - 0.4
    a[17] = angle_between_vectors(joint_pos[11] - joint_pos[10], joint_pos[12] - joint_pos[11]) - 0.1
    a[16] = angle_between_vectors(joint_pos[10] - joint_pos[9], joint_pos[11] - joint_pos[10]) - 0.1
    a[15] = 0.5 * (a[14] + a[16])

    # ring
    # a[6] = angle_between_plane_and_vector(
    #     joint_pos[15] - joint_pos[14], joint_pos[16] - joint_pos[15], y_dir
    # )
    a[6] = angle_between_vectors(joint_pos[13] - joint_pos[16], joint_pos[9] - joint_pos[12]) * -2 + 0.3
    a[9] = angle_between_vectors(joint_pos[15] - joint_pos[14], joint_pos[16] - joint_pos[15]) - 0.1
    a[8] = angle_between_vectors(joint_pos[14] - joint_pos[13], joint_pos[15] - joint_pos[14]) - 0.1
    a[7] = 0.5 * (a[6] + a[8])

    a = a.astype(np.float32)
    # ---- End inlined copy ----

    result = np.zeros(24, dtype=np.float32)
    result[0:6] = a[0:6]
    result[6:8] = 0.0
    result[8:12] = a[18:22]  # index
    result[12:16] = a[14:18]  # middle
    result[16:20] = a[6:10]  # ring
    result[20:24] = a[10:14]  # thumb
    return result


class JointOptimizer(Optimizer):
    """Closed-form / analytic ``joint_pos → qpos`` (no NLopt).

    Unlike vector/position/DexPilot optimizers, this reads full ``joint_pos`` from the
    detector and maps it with hand-crafted geometry inside ``_compute_qpos_from_joint_pos``.
    """
    retargeting_type = "JOINT"

    # Supported URDF stems for direct mapping
    _SUPPORTED_ROBOTS = frozenset(
        {
            "allegro_hand_left",
            "hmf_hand_proto5_release_left",
            "hmf_hand_proto5_release_right_ur7e",
        }
    )

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_human_indices: Optional[np.ndarray] = None,
        robot_name: Optional[str] = None,
    ):
        """
        Args:
            robot: Robot wrapper (used for limits / metadata).
            target_joint_names: Output joint ordering.
            target_link_human_indices: Unused here; kept for API parity with ``Optimizer``.
            robot_name: URDF stem key (e.g. ``allegro_hand_left``) to select the mapping.
        """
        if target_link_human_indices is None:
            target_link_human_indices = np.zeros((2, len(target_joint_names)), dtype=np.int64)
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.robot_name = robot_name

    def _compute_qpos_from_joint_pos(self, joint_pos: np.ndarray) -> np.ndarray:
        """
        Args:
            joint_pos: Detector landmarks, shape (N, 3).

        Returns:
            ``qpos`` slice for ``idx_pin2target``, length ``opt_dof``.
        """
        if self.robot_name not in self._SUPPORTED_ROBOTS:
            raise ValueError(
                f"JointOptimizer: unsupported robot {self.robot_name!r}; "
                f"supported: {sorted(self._SUPPORTED_ROBOTS)}."
            )
        if self.robot_name == "allegro_hand_left":
            raw = allegro_left_dummy_qpos_from_leap_joint_pos(joint_pos)
            if raw.shape[0] != self.opt_dof:
                raise ValueError(
                    f"JointOptimizer: allegro expects opt_dof=22, got {self.opt_dof}"
                )
            return raw
        if self.robot_name in {
            "hmf_hand_proto5_release_left",
            "hmf_hand_proto5_release_right_ur7e",
        }:
            # Canonical Proto5 layout from detector mapping:
            # [dummy6, wrist2, finger16] = 24 DOF
            proto5_24 = hmf_proto5_left_dummy_qpos_from_leap_joint_pos(joint_pos)
            if self.opt_dof == 24:
                return proto5_24
            if self.opt_dof == 22:
                # Finger-only control layout:
                # [dummy6, finger16] (drop WRZ/WRY)
                return np.concatenate([proto5_24[:6], proto5_24[8:24]], axis=0).astype(
                    np.float32
                )
            if self.opt_dof == 30:
                # UR7e+hand full layout:
                # [dummy6, arm6, wrist2, finger16]
                arm6 = np.zeros(6, dtype=np.float32)
                return np.concatenate([proto5_24[:6], arm6, proto5_24[6:24]], axis=0)
            raise ValueError(
                f"JointOptimizer: Proto5 expects opt_dof in {{22, 24, 30}}, got {self.opt_dof}"
            )
        raise ValueError(f"JointOptimizer: unsupported robot name: {self.robot_name}")

    def retarget(self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        """Return ``_compute_qpos_from_joint_pos(ref_value)`` (``ref_value`` is ``joint_pos``)."""
        # JointOptimizer computes qpos analytically from landmarks and can safely
        # ignore fixed_qpos in streaming teleop (often passed as empty).
        if len(fixed_qpos) not in (0, len(self.idx_pin2fixed)):
            raise ValueError(
                f"JointOptimizer: expected {len(self.idx_pin2fixed)} fixed joints, got {len(fixed_qpos)}"
            )
        return self._compute_qpos_from_joint_pos(ref_value)

    def get_objective_function(
        self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        """Unused stub for NLopt interface."""
        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            return 0.0
        return objective
