import time
from typing import Optional

import numpy as np
from pytransform3d import rotations

from dex_retargeting.constants import OPERATOR2MANO, HandType
from dex_retargeting.optimizer import Optimizer
from dex_retargeting.optimizer_utils import LPFilter


class SeqRetargeting:
    """
    序列重定向类，用于将人类手部姿态序列重定向到机器人手部
    """
    def __init__(
        self,
        optimizer: Optimizer,
        has_joint_limits=True,
        lp_filter: Optional[LPFilter] = None,
    ):
        """
        初始化序列重定向器
        
        Args:
            optimizer: 优化器对象，用于执行重定向优化
            has_joint_limits: 是否启用关节限制
            lp_filter: 低通滤波器，用于平滑关节位置（可选）
        """
        self.optimizer = optimizer
        robot = self.optimizer.robot

        # Joint limit
        # 关节限制设置
        self.has_joint_limits = has_joint_limits
        joint_limits = np.ones_like(robot.joint_limits)
        joint_limits[:, 0] = -1e4  # a large value is equivalent to no limit
        # 大数值等价于无限制
        joint_limits[:, 1] = 1e4
        if has_joint_limits:
            joint_limits[:] = robot.joint_limits[:]
            self.optimizer.set_joint_limit(joint_limits[self.optimizer.idx_pin2target])
        self.joint_limits = joint_limits[self.optimizer.idx_pin2target]

        # Temporal information
        # 时间信息：记录上一次的关节位置、累计时间和重定向次数
        self.last_qpos = joint_limits.mean(1)[self.optimizer.idx_pin2target].astype(
            np.float32
        )
        self.accumulated_time = 0  # 累计重定向耗时
        self.num_retargeting = 0  # 重定向次数

        # Filter
        # 滤波器：用于平滑关节位置
        self.filter = lp_filter

        # Warm started
        # 是否已预热启动
        self.is_warm_started = False

    def warm_start(
        self,
        wrist_pos: np.ndarray,
        wrist_quat: np.ndarray,
        hand_type: HandType = HandType.right,
        is_mano_convention: bool = False,
    ):
        """
        Initialize the wrist joint pose using analytical computation instead of retargeting optimization.
        This function is specifically for position retargeting with the flying robot hand, i.e. has 6D free joint
        You are not expected to use this function for vector retargeting, e.g. when you are working on teleoperation

        使用解析计算初始化腕关节姿态，而不是通过重定向优化。
        此函数专门用于具有6自由度自由关节的飞行机器人手的位置重定向。
        不建议在向量重定向中使用此函数，例如在遥操作场景中。

        Args:
            wrist_pos: position of the hand wrist, typically from human hand pose
                      手腕位置，通常来自人类手部姿态
            wrist_quat: quaternion of the hand wrist, the same convention as the operator frame definition if not is_mano_convention
                       手腕四元数，如果不是mano约定，则与操作者坐标系定义相同
            hand_type: hand type, used to determine the operator2mano matrix
                       手部类型，用于确定operator2mano矩阵
            is_mano_convention: whether the wrist_quat is in mano convention
                               手腕四元数是否采用mano约定
        """
        # This function can only be used when the first joints of robot are free joints
        # 此函数只能在机器人的前几个关节是自由关节时使用

        # 验证输入维度
        if len(wrist_pos) != 3:
            raise ValueError(f"Wrist pos: {wrist_pos} is not a 3-dim vector.")
        if len(wrist_quat) != 4:
            raise ValueError(f"Wrist quat: {wrist_quat} is not a 4-dim vector.")

        # 计算从操作者坐标系到MANO坐标系的转换矩阵
        operator2mano = OPERATOR2MANO[hand_type] if is_mano_convention else np.eye(3)
        robot = self.optimizer.robot
        # 构建目标手腕姿态矩阵（4x4齐次变换矩阵）
        target_wrist_pose = np.eye(4)
        target_wrist_pose[:3, :3] = (
            rotations.matrix_from_quaternion(wrist_quat) @ operator2mano.T
        )
        target_wrist_pose[:3, 3] = wrist_pos

        # 定义虚拟关节名称列表（6自由度：3个平移 + 3个旋转）
        name_list = [
            "dummy_x_translation_joint",
            "dummy_y_translation_joint",
            "dummy_z_translation_joint",
            "dummy_x_rotation_joint",
            "dummy_y_rotation_joint",
            "dummy_z_rotation_joint",
        ]
        # 获取手腕链接ID
        wrist_link_id = robot.get_joint_parent_child_frames(name_list[5])[1]

        # Set the dummy joints angles to zero
        # 将虚拟关节角度设置为零
        old_qpos = robot.q0
        new_qpos = old_qpos.copy()
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                new_qpos[num] = 0

        # 计算正向运动学，获取根到手腕的变换
        robot.compute_forward_kinematics(new_qpos)
        root2wrist = robot.get_link_pose_inv(wrist_link_id)
        # 计算目标根姿态
        target_root_pose = target_wrist_pose @ root2wrist

        # 将旋转矩阵转换为欧拉角
        euler = rotations.euler_from_matrix(
            target_root_pose[:3, :3], 0, 1, 2, extrinsic=False
        )
        # 组合位置和姿态为姿态向量 [x, y, z, roll, pitch, yaw]
        pose_vec = np.concatenate([target_root_pose[:3, 3], euler])

        # Find the dummy joints
        # 找到虚拟关节并设置其位置
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                index = name_list.index(joint_name)
                self.last_qpos[num] = pose_vec[index]

        self.is_warm_started = True

    def retarget(self, ref_value, fixed_qpos=np.array([])):
        """
        执行重定向，将参考值转换为机器人关节位置
        Perform retargeting to convert reference values to robot joint positions
        
        Args:
            ref_value: 参考值，通常是人类手部的关键点位置或向量
            fixed_qpos: 固定的关节位置（可选），用于指定某些关节的固定值
            
        Returns:
            robot_qpos: 完整的机器人关节位置数组
        """
        # 记录开始时间
        tic = time.perf_counter()

        # 执行优化重定向，将参考值转换为关节位置
        # 使用上一次的关节位置作为初始值，并限制在关节范围内
        qpos = self.optimizer.retarget(
            ref_value=ref_value.astype(np.float32),
            fixed_qpos=fixed_qpos.astype(np.float32),
            last_qpos=np.clip(
                self.last_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]
            ),
        )
        # 累计耗时和重定向次数
        self.accumulated_time += time.perf_counter() - tic
        self.num_retargeting += 1
        # 更新上一次的关节位置
        self.last_qpos = qpos
        
        # 构建完整的机器人关节位置数组
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos  # 固定关节
        robot_qpos[self.optimizer.idx_pin2target] = qpos  # 目标关节

        # 如果存在适配器，应用适配器变换
        if self.optimizer.adaptor is not None:
            robot_qpos = self.optimizer.adaptor.forward_qpos(robot_qpos)

        # 如果存在滤波器，应用低通滤波平滑关节位置
        if self.filter is not None:
            robot_qpos = self.filter.next(robot_qpos)
        return robot_qpos

    def set_qpos(self, robot_qpos: np.ndarray):
        """
        设置关节位置，更新上一次的关节位置
        Set joint positions and update the last joint positions
        
        Args:
            robot_qpos: 完整的机器人关节位置数组
        """
        target_qpos = robot_qpos[self.optimizer.idx_pin2target]
        self.last_qpos = target_qpos

    def get_qpos(self, fixed_qpos: Optional[np.ndarray] = None):
        """
        获取当前的关节位置
        Get current joint positions
        
        Args:
            fixed_qpos: 固定的关节位置（可选）
            
        Returns:
            robot_qpos: 完整的机器人关节位置数组
        """
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.idx_pin2target] = self.last_qpos
        if fixed_qpos is not None:
            robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos
        return robot_qpos

    def verbose(self):
        """
        打印重定向的详细信息（耗时和距离）
        Print detailed information about retargeting (time and distance)
        """
        min_value = self.optimizer.opt.last_optimum_value()
        print(
            f"Retargeting {self.num_retargeting} times takes: {self.accumulated_time}s"
        )
        print(f"Last distance: {min_value}")

    def reset(self):
        """
        重置重定向器状态
        Reset the retargeting state
        """
        # 将关节位置重置为关节限制的中间值
        self.last_qpos = self.joint_limits.mean(1).astype(np.float32)
        self.num_retargeting = 0
        self.accumulated_time = 0

    @property
    def joint_names(self):
        """
        获取所有关节的名称
        Get names of all joints
        
        Returns:
            关节名称列表
        """
        return self.optimizer.robot.dof_joint_names
