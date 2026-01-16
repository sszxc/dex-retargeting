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

if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    from loguru import logger
    import rerun as rr
    
    # 添加项目根目录到路径，以便导入 utils
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
    
    # 设置默认配置路径
    robot_name = RobotName.allegro
    retargeting_type = RetargetingType.position
    hand_type = HandType.left
    
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    if config_path is None:
        raise ValueError(f"无法找到配置文件: {robot_name}, {retargeting_type}, {hand_type}")
    
    # 设置默认 URDF 目录
    robot_dir = Path(__file__).parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    
    # 加载配置并构建 retargeting（用于获取 robot）
    logger.info(f"加载配置文件: {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    robot = retargeting.optimizer.robot
    
    # 获取机器人自由度
    total_dim = robot.dof
    logger.info(f"机器人自由度: {total_dim}")
    
    # 初始化 rerun board
    board = RerunBoard(
        f"RobotWrapperTest_{time.strftime('%m_%d_%H_%M', time.localtime())}",
        template="dex_retargeting"
    )
    
    # 测试参数
    DIM_TEST_DURATION = 3.0  # 每个维度测试3秒
    CYCLE_COUNT = 2  # 每个维度波动2个来回（4π）
    UPDATE_RATE = 30.0  # 更新频率 30Hz
    UPDATE_INTERVAL = 1.0 / UPDATE_RATE
    
    # 获取关节连接关系（方法内部有缓存，第一次调用后会自动缓存）
    joint_connections = robot.get_joint_connections()
    logger.info(f"关节连接数量: {len(joint_connections)}")
    
    # 测试循环
    while True:
        # 循环测试所有维度
        for dim_idx in range(total_dim):
            dim_start_time = time.time()
            logger.info(f"开始测试维度 {dim_idx}/{total_dim-1}")
            
            # 在当前维度的3秒测试时间内循环
            while True:
                current_time = time.time()
                elapsed = current_time - dim_start_time
                
                # 如果超过3秒，切换到下一个维度
                if elapsed >= DIM_TEST_DURATION:
                    break
                
                # 计算当前维度在3秒内的进度 [0, 1]
                progress = elapsed / DIM_TEST_DURATION
                
                # 计算正弦值：2个来回 = 4π，所以角度是 4π * progress
                angle = 4 * np.pi * progress
                sin_value = np.sin(angle)
                
                # 创建 qpos 数组：当前测试维度使用正弦值，其他维度为0
                qpos = np.zeros(total_dim, dtype=np.float32)
                # qpos = np.ones(total_dim, dtype=np.float32)  * 0.1
                qpos[dim_idx] = sin_value
                
                # 输出日志
                logger.info(f"测试维度 {dim_idx}, 值: {sin_value:.4f}, 进度: {progress*100:.1f}%")
                
                # 计算所有关节的3D位置
                joint_positions = robot.get_all_joint_positions(qpos)
                
                # 可视化关节位置到 rerun
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
                
                # 可视化连接线到 rerun
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
                
                # 在原点绘制单位坐标轴
                origin = np.array([0.0, 0.0, 0.0])
                identity_rotation = np.eye(3)  # 单位旋转矩阵
                board.log_axes(
                    translation=origin,
                    rotation=identity_rotation,
                    root="world",
                    name="origin_axes",
                    axis_size=0.5,  # 单位长度
                )
                
                # 控制更新频率
                time.sleep(UPDATE_INTERVAL)
            
            logger.info(f"完成测试维度 {dim_idx}/{total_dim-1}")
        
        logger.info("完成一轮所有维度测试，开始下一轮循环")
