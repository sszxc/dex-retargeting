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
    """优化器基类 - 用于机器人重定向的非线性优化
    
    该类提供了机器人关节重定向的基础框架，使用非线性优化方法
    将人类手部动作映射到机器人关节空间。
    """
    retargeting_type = "BASE"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_human_indices: np.ndarray,
    ):
        """
        初始化优化器
        
        Args:
            robot: 机器人包装器，包含机器人的运动学信息
            target_joint_names: 需要优化的目标关节名称列表
            target_link_human_indices: 目标链接对应的人手索引数组
        """
        self.robot = robot
        self.num_joints = robot.dof

        # 获取目标关节在机器人关节列表中的索引
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

        # 获取固定关节（不需要优化的关节）的索引
        self.idx_pin2fixed = np.array(
            [i for i in range(robot.dof) if i not in idx_pin2target], dtype=int
        )
        # 初始化NLopt优化器，使用序列二次规划算法
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_pin2target))
        self.opt_dof = len(idx_pin2target)  # This dof includes the mimic joints

        # Target - 目标链接的人手索引
        self.target_link_human_indices = target_link_human_indices

        # Free joint - 检查是否存在自由关节（通常用于浮动基座）
        link_names = robot.link_names
        self.has_free_joint = len([name for name in link_names if "dummy" in name]) >= 6

        # Kinematics adaptor - 运动学适配器，用于处理模拟关节等特殊情况
        self.adaptor: Optional[KinematicAdaptor] = None

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        """设置关节限制
        
        Args:
            joint_limits: 关节限制数组，形状为 (opt_dof, 2)，每行包含 [下界, 上界]
            epsilon: 边界容差，用于扩展关节限制范围
        """
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(
                f"Expect joint limits have shape: {(self.opt_dof, 2)}, but get {joint_limits.shape}"
            )
        # 设置优化器的下界和上界，并添加容差
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def get_link_indices(self, target_link_names):
        """获取目标链接的索引
        
        Args:
            target_link_names: 目标链接名称列表
            
        Returns:
            链接索引列表
        """
        return [self.robot.get_link_index(link_name) for link_name in target_link_names]

    def set_kinematic_adaptor(self, adaptor: KinematicAdaptor):
        """设置运动学适配器
        
        Args:
            adaptor: 运动学适配器，用于处理模拟关节等特殊情况
        """
        self.adaptor = adaptor

        # Remove mimic joints from fixed joint list
        # 从固定关节列表中移除模拟关节（因为模拟关节会由适配器自动处理）
        if isinstance(adaptor, MimicJointKinematicAdaptor):
            fixed_idx = self.idx_pin2fixed
            mimic_idx = adaptor.idx_pin2mimic
            new_fixed_id = np.array(
                [x for x in fixed_idx if x not in mimic_idx], dtype=int
            )
            self.idx_pin2fixed = new_fixed_id

    def retarget(self, ref_value, fixed_qpos, last_qpos):
        """
        Compute the retargeting results using non-linear optimization
        使用非线性优化计算重定向结果
        
        Args:
            ref_value: the reference value in cartesian space as input, different optimizer has different reference
                      笛卡尔空间中的参考值，不同优化器有不同的参考类型
            fixed_qpos: the fixed value (not optimized) in retargeting, consistent with self.fixed_joint_names
                        重定向中固定的关节位置（不进行优化），与 self.fixed_joint_names 一致
            last_qpos: the last retargeting results or initial value, consistent with function return
                       上一次重定向结果或初始值，与函数返回值一致

        Returns: joint position of robot, the joint order and dim is consistent with self.target_joint_names
                 机器人的关节位置，关节顺序和维度与 self.target_joint_names 一致

        """
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but non_target_qpos {fixed_qpos} is given"
            )
        # 获取目标函数
        objective_fn = self.get_objective_function(
            ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32)
        )

        # 设置优化目标并执行优化
        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            return np.array(qpos, dtype=np.float32)
        except RuntimeError as e:
            # 如果优化失败，返回上一次的关节位置
            print(e)
            return np.array(last_qpos, dtype=np.float32)

    @abstractmethod
    def get_objective_function(
        self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        """获取目标函数（抽象方法，由子类实现）
        
        Args:
            ref_value: 参考值
            fixed_qpos: 固定关节位置
            last_qpos: 上一次的关节位置
            
        Returns:
            目标函数，用于NLopt优化
        """
        pass

    @property
    def fixed_joint_names(self):
        """获取固定关节名称列表（属性）"""
        joint_names = self.robot.dof_joint_names
        return [joint_names[i] for i in self.idx_pin2fixed]


class PositionOptimizer(Optimizer):
    """位置优化器 - 基于3D位置误差的重定向优化器
    
    该优化器通过最小化目标链接的3D位置误差来实现重定向，
    使用Huber损失函数来处理异常值。
    """
    retargeting_type = "POSITION"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
    ):
        """
        初始化位置优化器
        
        Args:
            robot: 机器人包装器
            target_joint_names: 目标关节名称列表
            target_link_names: 目标链接名称列表
            target_link_human_indices: 目标链接对应的人手索引
            huber_delta: Huber损失的阈值参数
            norm_delta: 正则化项的权重
        """
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        # 使用Huber损失（平滑L1损失）来处理位置误差
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta

        # Sanity check and cache link indices
        # 验证并缓存链接索引，避免重复查找
        self.target_link_indices = self.get_link_indices(target_link_names)

        # 设置优化器的函数容差
        self.opt.set_ftol_abs(1e-5)

    def get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        """获取位置优化的目标函数
        
        Args:
            target_pos: 目标位置数组，形状为 (n, 3)
            fixed_qpos: 固定关节位置
            last_qpos: 上一次的关节位置
            
        Returns:
            目标函数，用于NLopt优化
        """
        # 初始化关节位置，设置固定关节的值
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            """目标函数：计算位置误差和梯度
            
            Args:
                x: 当前优化变量（目标关节位置）
                grad: 梯度输出数组
                
            Returns:
                损失值（标量）
            """
            # 设置目标关节位置
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            # 运动学前向传播，处理模拟关节等特殊情况
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            # 计算正向运动学，获取目标链接的位姿
            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.target_link_indices
            ]
            # 提取位置信息（位姿矩阵的第4列前3个元素）
            body_pos = np.stack(
                [pose[:3, 3] for pose in target_link_poses], axis=0
            )  # (n ,3)

            # Torch computation for accurate loss and grad
            # 使用PyTorch进行精确的损失和梯度计算
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            # 基于3D位置误差的运动学重定向损失项
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                # 计算雅可比矩阵（用于梯度计算）
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    # 获取链接在局部坐标系下的雅可比（仅位置部分，前3行）
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]  # 旋转矩阵
                    # 将局部雅可比转换到全局坐标系
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                # 注意：此雅可比中的关节顺序与pinocchio一致
                jacobians = np.stack(jacobians, axis=0)
                # 反向传播计算位置梯度
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                # 将雅可比从pinocchio顺序转换为目标关节顺序
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                # Compute the gradient to the qpos
                # 计算对关节位置的梯度
                grad_qpos = np.matmul(grad_pos, jacobians)
                grad_qpos = grad_qpos.mean(1).sum(0)
                # 添加正则化项（平滑项，使关节位置接近上一次的值）
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class VectorOptimizer(Optimizer):
    """向量优化器 - 基于向量误差的重定向优化器
    
    该优化器通过最小化链接之间的向量误差来实现重定向，
    适用于需要保持相对位置关系的场景（如手指之间的相对位置）。
    """
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
        初始化向量优化器
        
        Args:
            robot: 机器人包装器
            target_joint_names: 目标关节名称列表
            target_origin_link_names: 向量起点链接名称列表
            target_task_link_names: 向量终点链接名称列表
            target_link_human_indices: 目标链接对应的人手索引
            huber_delta: Huber损失的阈值参数
            norm_delta: 正则化项的权重
            scaling: 向量缩放因子
        """
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta
        self.scaling = scaling

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        # 计算缓存以提高性能：对于在多个向量中使用的链接（如手掌），避免重复计算
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        # 计算链接在缓存列表中的索引
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Cache link indices that will involve in kinematics computation
        # 缓存参与运动学计算的链接索引
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        """获取向量优化的目标函数
        
        Args:
            target_vector: 目标向量数组，形状为 (n, 3)
            fixed_qpos: 固定关节位置
            last_qpos: 上一次的关节位置
            
        Returns:
            目标函数，用于NLopt优化
        """
        # 初始化关节位置，设置固定关节的值
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        # 应用缩放因子并转换为PyTorch张量
        torch_target_vec = torch.as_tensor(target_vector) * self.scaling
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            """目标函数：计算向量误差和梯度
            
            Args:
                x: 当前优化变量（目标关节位置）
                grad: 梯度输出数组
                
            Returns:
                损失值（标量）
            """
            # 设置目标关节位置
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            # 运动学前向传播，处理模拟关节等特殊情况
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            # 计算正向运动学，获取链接位姿
            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            # 提取位置信息
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            # 使用PyTorch进行精确的损失和梯度计算
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            # 索引链接用于计算向量
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            # 计算机器人实际向量（从起点到终点） (4,3)
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            # 基于3D位置误差的运动学重定向损失项
            # 计算向量距离（L2范数）
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            # 使用Huber损失计算距离误差
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                # 计算雅可比矩阵（用于梯度计算）
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):  # 考虑的是四个手指指尖 + 手腕 的 link
                    # 获取链接在局部坐标系下的雅可比（仅位置部分）  原始值是 (6, 16)
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]  # 旋转矩阵
                    # 将局部雅可比转换到全局坐标系
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                # 注意：此雅可比中的关节顺序与pinocchio一致
                jacobians = np.stack(jacobians, axis=0)  # 5*(3,16) -> (5,3,16)
                # 反向传播计算位置梯度
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                # 将雅可比从pinocchio顺序转换为目标关节顺序
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                # 计算对关节位置的梯度
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))  # qpos->link * link->loss
                grad_qpos = grad_qpos.mean(1).sum(0)
                # 添加正则化项（平滑项） 单纯是为了让当前关节位置接近上一次的关节位置
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class DexPilotOptimizer(Optimizer):
    """使用DexPilot方法的重定向优化器
    
    这是对DexPilot论文中原始优化器的更广泛适配。
    虽然最初的DexPilot研究仅专注于四指Allegro手，但此版本的优化器
    将相同的原理应用于四指和五指手。它投影拇指和其他手指之间的距离
    以促进更稳定的抓取。
    参考: https://arxiv.org/abs/1910.03135

    Retargeting optimizer using the method proposed in DexPilot

    This is a broader adaptation of the original optimizer delineated in the DexPilot paper.
    While the initial DexPilot study focused solely on the four-fingered Allegro Hand, this version of the optimizer
    embraces the same principles for both four-fingered and five-fingered hands. It projects the distance between the
    thumb and the other fingers to facilitate more stable grasping.
    Reference: https://arxiv.org/abs/1910.03135

    Args:
        robot: 机器人包装器
        target_joint_names: 目标关节名称列表
        finger_tip_link_names: 指尖链接名称列表
        wrist_link_name: 手腕链接名称
        gamma: 正则化权重（已注释，未使用）
        project_dist: 投影距离阈值
        escape_dist: 退出投影的距离阈值
        eta1: 第一层投影距离参数
        eta2: 第二层投影距离参数
        scaling: 向量缩放因子
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
        # 验证手指数量（DexPilot仅支持2-5指）
        if len(finger_tip_link_names) < 2 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"DexPilot optimizer can only be applied to hands with 2 to 5 fingers, but got "
                f"{len(finger_tip_link_names)} fingers."
            )
        self.num_fingers = len(finger_tip_link_names)

        # 生成链接索引（用于定义向量连接关系）
        origin_link_index, task_link_index = self.generate_link_indices(
            self.num_fingers
        )

        # 如果没有提供目标链接的人手索引，则自动生成
        if target_link_human_indices is None:
            target_link_human_indices = (
                np.stack([origin_link_index, task_link_index], axis=0) * 4
            ).astype(int)
        # 构建链接名称列表（手腕 + 指尖）
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]

        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.scaling = scaling
        # 使用Huber损失，reduction="none"以便后续加权
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta

        # DexPilot parameters - DexPilot算法参数
        self.project_dist = project_dist  # 投影距离阈值
        self.escape_dist = escape_dist    # 退出投影的距离阈值
        self.eta1 = eta1                  # 第一层投影距离参数
        self.eta2 = eta2                  # 第二层投影距离参数

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        # 计算缓存以提高性能：对于在多个向量中使用的链接（如手掌），避免重复计算
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        # 计算链接在缓存列表中的索引
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Sanity check and cache link indices
        # 验证并缓存链接索引
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

        # DexPilot cache - 初始化DexPilot算法的缓存数据
        (
            self.projected,                    # 投影指示器数组
            self.s2_project_index_origin,      # 第二层投影的起点索引
            self.s2_project_index_task,        # 第二层投影的终点索引
            self.projected_dist,               # 投影距离数组
        ) = self.set_dexpilot_cache(self.num_fingers, eta1, eta2)

    @staticmethod
    def generate_link_indices(num_fingers):
        """生成链接索引，定义向量连接关系
        
        生成两个列表：起点链接索引和终点链接索引。
        包括：1) 手指之间的连接；2) 手腕到手指的连接。
        
        Example:
        >>> generate_link_indices(4)
        ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
        
        Args:
            num_fingers: 手指数量
            
        Returns:
            (origin_link_index, task_link_index): 起点和终点链接索引列表
        """
        origin_link_index = []
        task_link_index = []

        # Add indices for connections between fingers
        # 添加手指之间的连接索引（手指1到其他手指）
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)

        # Add indices for connections to the base (0)
        # 添加与基座（手腕，索引0）的连接索引
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)

        return origin_link_index, task_link_index

    @staticmethod
    def set_dexpilot_cache(num_fingers, eta1, eta2):
        """设置DexPilot算法的缓存数据
        
        初始化投影指示器、第二层投影索引和投影距离数组。
        
        Example:
        >>> set_dexpilot_cache(4, 0.1, 0.2)
        (array([False, False, False, False, False, False]),
        [1, 2, 2],
        [0, 0, 1],
        array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        
        Args:
            num_fingers: 手指数量
            eta1: 第一层投影距离参数（用于手指之间的连接）
            eta2: 第二层投影距离参数（用于第二层投影）
            
        Returns:
            (projected, s2_project_index_origin, s2_project_index_task, projected_dist):
            - projected: 投影指示器数组（布尔型）
            - s2_project_index_origin: 第二层投影的起点索引
            - s2_project_index_task: 第二层投影的终点索引
            - projected_dist: 投影距离数组
        """
        # 初始化投影指示器（手指之间的连接数量）
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

        # 生成第二层投影的索引（用于更复杂的投影关系）
        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)

        # 构建投影距离数组：前(num_fingers-1)个使用eta1，其余使用eta2
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

        # 计算各种长度
        len_proj = len(self.projected)  # 投影向量的总长度
        len_s2 = len(self.s2_project_index_task)  # 第二层投影的长度
        len_s1 = len_proj - len_s2  # 第一层投影的长度

        # Update projection indicator
        # 更新投影指示器：根据目标向量距离决定是否启用投影
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        # 第一层：当距离小于project_dist时启用投影，大于escape_dist时禁用
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        # 第二层：需要对应的第一层投影都启用，且距离小于0.03
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        # Update weight vector
        # 更新权重向量：投影时使用高权重，否则使用正常权重
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)

        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        # 对于从手腕到指尖的向量，我们将权重改为(len_proj + num_fingers)而不是1
        # 这确保了由于错误姿态检测而导致的更好的直观映射
        weight = torch.from_numpy(
            np.concatenate(
                [
                    weight,
                    np.ones(self.num_fingers, dtype=np.float32) * len_proj
                    + self.num_fingers,
                ]
            )
        )

        # Compute reference distance vector
        # 计算参考距离向量
        normal_vec = target_vector * self.scaling  # (10, 3) 正常向量（应用缩放）
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3) 方向向量
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3) 投影向量

        # Compute final reference vector
        # 计算最终参考向量：根据投影指示器选择投影向量或正常向量
        reference_vec = np.where(
            self.projected[:, None], projected_vec, normal_vec[:len_proj]
        )  # (6, 3)
        # 拼接投影向量和手腕到指尖的向量
        reference_vec = np.concatenate(
            [reference_vec, normal_vec[len_proj:]], axis=0
        )  # (10, 3)
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
