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

    def get_all_joint_positions(self, qpos: npt.NDArray) -> npt.NDArray:
        """
        计算所有关节的3D空间位置
        
        Args:
            qpos: 关节角度数组，形状为 (dof,)
            
        Returns:
            joint_positions: 所有关节的位置数组，形状为 (num_joints, 3)
            顺序与 self.joint_names 一致
        """
        # 计算正向运动学
        self.compute_forward_kinematics(qpos)
        
        # 获取所有关节的位置
        joint_positions = []
        for joint_name in self.joint_names:
            try:
                # 获取关节的frame ID
                joint_frame_id = self.model.getFrameId(joint_name)
                # 获取关节frame的位姿
                joint_pose = pin.updateFramePlacement(self.model, self.data, joint_frame_id)
                # 从 4x4 齐次变换矩阵中提取位置（第4列的前3个元素）
                position = joint_pose.homogeneous[:3, 3]
                joint_positions.append(position)
            except (ValueError, RuntimeError):
                # 如果关节不存在或无法获取，跳过
                continue
        
        return np.array(joint_positions)

    def get_joint_connections(self) -> List[Tuple[int, int]]:
        """
        获取关节之间的连接关系，类似 mp_hands.HAND_CONNECTIONS
        
        该方法会缓存结果，因为机器人的结构不会改变。
        每个连接表示一个link，连接两个关节（父关节和子关节）。
        
        Returns:
            connections: 关节连接列表，每个元素为 (parent_joint_index, child_joint_index)
            索引对应 self.joint_names 的顺序
        """
        # 如果已经缓存，直接返回
        if self._cached_link_connections is not None:
            return self._cached_link_connections
        
        connections = []
        
        # 使用 Pinocchio 的 model.parents 来获取关节的父子关系
        # model.parents[joint_id] 返回父关节的ID
        for joint_id in range(1, self.model.njoints):  # 从1开始，跳过根关节
            parent_joint_id = self.model.parents[joint_id]
            
            # 如果存在父关节，添加连接
            if parent_joint_id >= 0:
                connections.append((parent_joint_id, joint_id))
        
        # 缓存结果
        self._cached_link_connections = connections
        return connections
