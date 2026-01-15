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
        获取非 dummy 关节的索引列表（保留最后一个 dummy 节点以保持树结构）
        
        Returns:
            non_dummy_indices: 非 dummy 关节的索引列表（包含最后一个 dummy 关节）
        """
        if self._non_dummy_joint_indices is not None:
            return self._non_dummy_joint_indices
        
        # 找到所有 dummy 关节的索引
        dummy_indices = [
            i for i, name in enumerate(self.joint_names)
            if "dummy" in name.lower()
        ]
        
        # 找到最后一个 dummy 关节的索引（通常是 dummy_z_rotation_joint）
        last_dummy_idx = dummy_indices[-1] if dummy_indices else None
        
        # 过滤掉包含 "dummy" 的关节，但保留最后一个
        self._non_dummy_joint_indices = [
            i for i, name in enumerate(self.joint_names)
            if "dummy" not in name.lower() or i == last_dummy_idx
        ]
        return self._non_dummy_joint_indices

    def get_all_joint_positions(self, qpos: npt.NDArray) -> npt.NDArray:
        """
        计算所有关节的3D空间位置（排除大部分 dummy joints，但保留最后一个以保持树结构）
        
        Args:
            qpos: 关节角度数组，形状为 (dof,)
            
        Returns:
            joint_positions: 关节位置数组，形状为 (num_joints, 3)
            包含所有非 dummy 关节和最后一个 dummy 关节，顺序与过滤后的关节名称一致
        """
        # 计算正向运动学
        self.compute_forward_kinematics(qpos)
        
        # 获取非 dummy 关节的索引
        non_dummy_indices = self._get_non_dummy_joint_indices()
        
        # 获取所有关节的位置（只包含非 dummy 关节）
        joint_positions = []
        for joint_idx in non_dummy_indices:
            joint_name = self.joint_names[joint_idx]
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
        获取关节之间的连接关系，类似 mp_hands.HAND_CONNECTIONS（排除大部分 dummy joints，但保留最后一个）
        
        该方法会缓存结果，因为机器人的结构不会改变。
        每个连接表示一个link，连接两个关节（父关节和子关节）。
        只返回过滤后关节之间的连接，索引对应过滤后的关节顺序（包含最后一个 dummy 关节）。
        
        Returns:
            connections: 关节连接列表，每个元素为 (parent_joint_index, child_joint_index)
            索引对应过滤后的关节顺序（包含最后一个 dummy 关节）
        """
        # 如果已经缓存，直接返回
        if self._cached_link_connections is not None:
            return self._cached_link_connections
        
        # 获取非 dummy 关节的索引
        non_dummy_indices = self._get_non_dummy_joint_indices()
        # 创建从原始索引到过滤后索引的映射
        original_to_filtered = {orig_idx: filtered_idx for filtered_idx, orig_idx in enumerate(non_dummy_indices)}
        
        connections = []
        
        # 使用 Pinocchio 的 model.parents 来获取关节的父子关系
        # model.parents[joint_id] 返回父关节的ID
        for joint_id in range(1, self.model.njoints):  # 从1开始，跳过根关节
            parent_joint_id = self.model.parents[joint_id]
            
            # 如果存在父关节，且父子关节都不是 dummy 关节，添加连接
            if parent_joint_id >= 0:
                # 检查父关节和子关节是否都是非 dummy 关节
                if parent_joint_id in original_to_filtered and joint_id in original_to_filtered:
                    # 使用过滤后的索引
                    filtered_parent_idx = original_to_filtered[parent_joint_id]
                    filtered_child_idx = original_to_filtered[joint_id]
                    connections.append((filtered_parent_idx, filtered_child_idx))
        
        # 缓存结果
        self._cached_link_connections = connections
        return connections
