from typing import List, Tuple, Optional

import numpy as np
import numpy.typing as npt
import pinocchio as pin


class RobotWrapper:
    """
    This class does not take mimic joint into consideration
    """

    def __init__(self, urdf_path: str, use_collision=False, use_visual=False):
        # Create robot model and data
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()

        if use_visual or use_collision:
            raise NotImplementedError

        self.q0 = pin.neutral(self.model)
        if self.model.nv != self.model.nq:
            raise NotImplementedError("Can not handle robot with special joint.")
        
        # Cache for link connections (computed once, reused many times)
        self._cached_link_connections: Optional[List[Tuple[int, int]]] = None
        
        # Cache for non-dummy joint indices (computed once, reused many times)
        self._non_dummy_joint_indices: Optional[List[int]] = None

    # -------------------------------------------------------------------------- #
    # Robot property
    # -------------------------------------------------------------------------- #
    @property
    def joint_names(self) -> List[str]:
        return list(self.model.names)

    @property
    def dof_joint_names(self) -> List[str]:
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def link_names(self) -> List[str]:
        link_names = []
        for i, frame in enumerate(self.model.frames):
            link_names.append(frame.name)
        return link_names

    @property
    def joint_limits(self):
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    # -------------------------------------------------------------------------- #
    # Query function
    # -------------------------------------------------------------------------- #
    def get_joint_index(self, name: str):
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str):
        if name not in self.link_names:
            raise ValueError(
                f"{name} is not a link name. Valid link names: \n{self.link_names}"
            )
        return self.model.getFrameId(name, pin.BODY)

    def get_joint_parent_child_frames(self, joint_name: str):
        joint_id = self.model.getFrameId(joint_name)
        parent_id = self.model.frames[joint_id].parent
        child_id = -1
        for idx, frame in enumerate(self.model.frames):
            if frame.previousFrame == joint_id:
                child_id = idx
        if child_id == -1:
            raise ValueError(f"Can not find child link of {joint_name}")

        return parent_id, child_id

    # -------------------------------------------------------------------------- #
    # Kinematics function
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics(self, qpos: npt.NDArray):
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.homogeneous

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.inverse().homogeneous

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> npt.NDArray:
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
        return J

    def _get_non_dummy_joint_indices(self) -> List[int]:
        """
        Indices of joints to visualize, keeping the subtree from the last ``dummy*`` joint
        so the kinematic tree stays connected.
        """
        if self._non_dummy_joint_indices is not None:
            return self._non_dummy_joint_indices

        dummy_indices = [
            i for i, name in enumerate(self.joint_names)
            if "dummy" in name.lower()
        ]

        last_dummy_idx = dummy_indices[-1] if dummy_indices else None

        # Legacy note: filter dummies but keep the last dummy chain
        # self._non_dummy_joint_indices = [
        #     i for i, name in enumerate(self.joint_names)
        #     if "dummy" not in name.lower() or i == last_dummy_idx
        # ]
        if last_dummy_idx is not None:
            self._non_dummy_joint_indices = list(range(last_dummy_idx, len(self.joint_names)))
        else:
            self._non_dummy_joint_indices = list(range(len(self.joint_names)))
        return self._non_dummy_joint_indices

    def get_all_joint_positions(self, qpos: npt.NDArray) -> npt.NDArray:
        """
        3D origins for the filtered joint list (see ``_get_non_dummy_joint_indices``).

        Args:
            qpos: Configuration (dof,).

        Returns:
            (N, 3) positions in world frame, same order as filtered joint indices.
        """
        self.compute_forward_kinematics(qpos)

        non_dummy_indices = self._get_non_dummy_joint_indices()

        joint_positions = []
        for joint_idx in non_dummy_indices:
            joint_name = self.joint_names[joint_idx]
            try:
                joint_frame_id = self.model.getFrameId(joint_name)
                joint_pose = pin.updateFramePlacement(self.model, self.data, joint_frame_id)
                position = joint_pose.homogeneous[:3, 3]
                joint_positions.append(position)
            except (ValueError, RuntimeError):
                continue
        
        return np.array(joint_positions)

    def get_joint_connections(self) -> List[Tuple[int, int]]:
        """
        Parent/child pairs in the **filtered** joint index space (MediaPipe-style connectivity).

        Cached after first call.

        Returns:
            List of ``(parent_idx, child_idx)`` in filtered order.
        """
        if self._cached_link_connections is not None:
            return self._cached_link_connections

        non_dummy_indices = self._get_non_dummy_joint_indices()
        original_to_filtered = {orig_idx: filtered_idx for filtered_idx, orig_idx in enumerate(non_dummy_indices)}

        connections = []

        for joint_id in range(1, self.model.njoints):
            parent_joint_id = self.model.parents[joint_id]

            if parent_joint_id >= 0:
                if parent_joint_id in original_to_filtered and joint_id in original_to_filtered:
                    filtered_parent_idx = original_to_filtered[parent_joint_id]
                    filtered_child_idx = original_to_filtered[joint_id]
                    connections.append((filtered_parent_idx, filtered_child_idx))

        self._cached_link_connections = connections
        return connections

if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    from loguru import logger
    import rerun as rr
    
    # Add project root for ``utils`` imports
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from dex_retargeting.constants import (
        RobotName,
        RetargetingType,
        HandType,
        get_default_config_path,
    )
    from dex_retargeting.retargeting_config import RetargetingConfig
    from utils.rerun_board import RerunBoard
    
    # Default config
    robot_name = RobotName.allegro
    retargeting_type = RetargetingType.position
    hand_type = HandType.left
    
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    if config_path is None:
        raise ValueError(f"Config not found: {robot_name}, {retargeting_type}, {hand_type}")

    # Default URDF directory
    robot_dir = Path(__file__).parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    
    logger.info(f"Loading config: {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    robot = retargeting.optimizer.robot
    
    total_dim = robot.dof
    logger.info(f"Robot DoF: {total_dim}")

    # Rerun board
    board = RerunBoard(
        f"RobotWrapperTest_{time.strftime('%m_%d_%H_%M', time.localtime())}",
        template="dex_retargeting"
    )
    
    DIM_TEST_DURATION = 3.0
    CYCLE_COUNT = 2
    UPDATE_RATE = 30.0
    UPDATE_INTERVAL = 1.0 / UPDATE_RATE
    
    joint_connections = robot.get_joint_connections()
    logger.info(f"Joint connections: {len(joint_connections)}")

    while True:
        for dim_idx in range(total_dim):
            dim_start_time = time.time()
            logger.info(f"Testing DoF {dim_idx}/{total_dim - 1}")

            while True:
                current_time = time.time()
                elapsed = current_time - dim_start_time

                if elapsed >= DIM_TEST_DURATION:
                    break

                progress = elapsed / DIM_TEST_DURATION

                angle = 4 * np.pi * progress
                sin_value = np.sin(angle)

                qpos = np.zeros(total_dim, dtype=np.float32)
                qpos[dim_idx] = sin_value

                logger.info(
                    f"DoF {dim_idx}, value: {sin_value:.4f}, progress: {progress * 100:.1f}%"
                )

                joint_positions = robot.get_all_joint_positions(qpos)

                # Log joint spheres
                for i in range(joint_positions.shape[0]):
                    board.log(
                        f"world/robot/joint/joint_{i}",
                        rr.Points3D(
                            positions=[joint_positions[i]], 
                            colors=[[0, 0, 255]], 
                            radii=0.005, 
                            # labels=f"{i}"
                        ),
                    )
                
                for pair in joint_connections:
                    assert pair[0] < joint_positions.shape[0] and pair[1] < joint_positions.shape[0], f"pair: {pair} is out of range"
                    board.log(
                        f"world/robot/link/{pair}",
                        rr.Arrows3D(
                            origins=[joint_positions[pair[0]]], 
                            vectors=[joint_positions[pair[1]] - joint_positions[pair[0]]], 
                            colors=[[255, 255, 0]]
                        ),
                    )
                
                origin = np.array([0.0, 0.0, 0.0])
                identity_rotation = np.eye(3)
                board.log_axes(
                    translation=origin,
                    rotation=identity_rotation,
                    root="world",
                    name="origin_axes",
                    axis_size=0.5,
                )

                time.sleep(UPDATE_INTERVAL)

            logger.info(f"Finished DoF {dim_idx}/{total_dim - 1}")

        logger.info("Completed full-DoF sweep; repeating")
