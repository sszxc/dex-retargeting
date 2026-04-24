import time
from typing import Optional

import numpy as np
from pytransform3d import rotations

from dex_retargeting.constants import OPERATOR2MANO, HandType
from dex_retargeting.optimizer import Optimizer
from dex_retargeting.optimizer_utils import LPFilter


class SeqRetargeting:
    """
    Sequence retargeting: map a stream of human hand poses to robot hand joints.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        has_joint_limits=True,
        lp_filter: Optional[LPFilter] = None,
    ):
        """
        Args:
            optimizer: Optimizer used for each retargeting step.
            has_joint_limits: If True, apply robot joint limits to the optimization variables.
            lp_filter: Optional low-pass filter on full ``robot_qpos``.
        """
        self.optimizer = optimizer
        robot = self.optimizer.robot

        joint_limits = np.ones_like(robot.joint_limits)
        joint_limits[:, 0] = -1e4
        joint_limits[:, 1] = 1e4
        if has_joint_limits:
            joint_limits[:] = robot.joint_limits[:]
            self.optimizer.set_joint_limit(joint_limits[self.optimizer.idx_pin2target])
        self.joint_limits = joint_limits[self.optimizer.idx_pin2target]

        self.last_qpos = joint_limits.mean(1)[self.optimizer.idx_pin2target].astype(
            np.float32
        )
        self.accumulated_time = 0
        self.num_retargeting = 0

        self.filter = lp_filter

        self.is_warm_started = False

    def warm_start(
        self,
        wrist_pos: np.ndarray,
        wrist_quat: np.ndarray,
        hand_type: HandType = HandType.right,
        is_mano_convention: bool = False,
    ):
        """
        Initialize the floating-base wrist with an analytic solve instead of optimization.

        Intended for **position** retargeting with a 6-DoF dummy base. Not for vector /
        teleop-style retargeting.

        Args:
            wrist_pos: Wrist position (3,) in the same frame as the operator definition.
            wrist_quat: Wrist quaternion (wxyz or library convention per ``is_mano_convention``).
            hand_type: Selects ``OPERATOR2MANO`` when ``is_mano_convention`` is True.
            is_mano_convention: If True, apply MANO/operator alignment via ``OPERATOR2MANO``.
        """
        if len(wrist_pos) != 3:
            raise ValueError(f"Wrist pos: {wrist_pos} is not a 3-dim vector.")
        if len(wrist_quat) != 4:
            raise ValueError(f"Wrist quat: {wrist_quat} is not a 4-dim vector.")

        operator2mano = OPERATOR2MANO[hand_type] if is_mano_convention else np.eye(3)
        robot = self.optimizer.robot
        target_wrist_pose = np.eye(4)
        target_wrist_pose[:3, :3] = (
            rotations.matrix_from_quaternion(wrist_quat) @ operator2mano.T
        )
        target_wrist_pose[:3, 3] = wrist_pos

        name_list = [
            "dummy_x_translation_joint",
            "dummy_y_translation_joint",
            "dummy_z_translation_joint",
            "dummy_x_rotation_joint",
            "dummy_y_rotation_joint",
            "dummy_z_rotation_joint",
        ]
        wrist_link_id = robot.get_joint_parent_child_frames(name_list[5])[1]

        old_qpos = robot.q0
        new_qpos = old_qpos.copy()
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                new_qpos[num] = 0

        robot.compute_forward_kinematics(new_qpos)
        root2wrist = robot.get_link_pose_inv(wrist_link_id)
        target_root_pose = target_wrist_pose @ root2wrist

        euler = rotations.euler_from_matrix(
            target_root_pose[:3, :3], 0, 1, 2, extrinsic=False
        )
        pose_vec = np.concatenate([target_root_pose[:3, 3], euler])
        print("Called warm_start with pose_vec: ", pose_vec)
        raise Exception("When do you call this function?")
        breakpoint()

        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                index = name_list.index(joint_name)
                self.last_qpos[num] = pose_vec[index]

        self.is_warm_started = True

    def retarget(self, ref_value, fixed_qpos=np.array([])):
        """
        Run one retargeting step.

        Args:
            ref_value: Human reference (positions or vectors, depending on optimizer).
            fixed_qpos: Fixed (non-optimized) joint values aligned with ``idx_pin2fixed``.

        Returns:
            Full ``robot_qpos`` including fixed, target, adaptor, and optional filtering.
        """
        tic = time.perf_counter()

        qpos = self.optimizer.retarget(
            ref_value=ref_value.astype(np.float32),
            fixed_qpos=fixed_qpos.astype(np.float32),
            last_qpos=np.clip(
                self.last_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]
            ),
        )
        self.accumulated_time += time.perf_counter() - tic
        self.num_retargeting += 1
        self.last_qpos = qpos

        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos
        robot_qpos[self.optimizer.idx_pin2target] = qpos

        if self.optimizer.adaptor is not None:
            robot_qpos = self.optimizer.adaptor.forward_qpos(robot_qpos)

        if self.filter is not None:
            robot_qpos = self.filter.next(robot_qpos)
        return robot_qpos

    def set_qpos(self, robot_qpos: np.ndarray):
        """Seed ``last_qpos`` from a full configuration."""
        target_qpos = robot_qpos[self.optimizer.idx_pin2target]
        self.last_qpos = target_qpos

    def get_qpos(self, fixed_qpos: Optional[np.ndarray] = None):
        """Reconstruct full ``robot_qpos`` from ``last_qpos`` and optional ``fixed_qpos``."""
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.idx_pin2target] = self.last_qpos
        if fixed_qpos is not None:
            robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos
        return robot_qpos

    def verbose(self):
        """Print timing and last objective value."""
        min_value = self.optimizer.opt.last_optimum_value()
        print(
            f"Retargeting {self.num_retargeting} times takes: {self.accumulated_time}s"
        )
        print(f"Last distance: {min_value}")

    def reset(self):
        """Reset state to mid-range limits and zero counters."""
        self.last_qpos = self.joint_limits.mean(1).astype(np.float32)
        self.num_retargeting = 0
        self.accumulated_time = 0

    @property
    def joint_names(self):
        return self.optimizer.robot.dof_joint_names
