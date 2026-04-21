"""
Leap Motion 手部检测器
将 Leap Motion API 的手部数据转换为与 MediaPipe 兼容的格式（21个关键点）
"""
import numpy as np
import cv2
import leap
from leap import datatypes as ldt
from typing import Optional, Tuple
from timeit import default_timer as timer
import time
from typing import Callable

# MediaPipe 到 MANO 的坐标转换矩阵（与 single_hand_detector.py 保持一致）
OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


DEFAULT_CAMERA2TABLE = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

def wait_until(condition: Callable[[], bool], timeout: float = 5, poll_delay: float = 0.01):
    """等待条件满足"""
    start_time = timer()
    while timer() - start_time < timeout:
        if condition():
            return True
        time.sleep(poll_delay)
    return False


class LeapMotionHandDetector:
    """
    Leap Motion 手部检测器类
    使用 Leap Motion API 进行手部关键点检测
    支持将检测结果转换为与 MediaPipe 兼容的格式（21个关键点）
    """

    def __init__(
        self,
        hand_type: str = "Right",
        tracking_mode: leap.TrackingMode = leap.TrackingMode.Desktop,
        camera2table: Optional[np.ndarray] = None,
    ):
        """
        初始化 Leap Motion 检测器
        
        Args:
            hand_type: 手部类型，"Right" 或 "Left"
            tracking_mode: 跟踪模式，Desktop/HMD/ScreenTop
        """
        self.hand_type = hand_type
        self.tracking_mode = tracking_mode
        self.camera2table = (
            np.asarray(camera2table, dtype=np.float64)
            if camera2table is not None
            else DEFAULT_CAMERA2TABLE
        )

        # 根据手部类型选择对应的坐标转换矩阵
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )

        # Leap Motion 期望的手部类型
        self.leap_hand_type = (
            leap.HandType.Right if hand_type == "Right" else leap.HandType.Left
        )

        # 初始化连接和监听器
        self.connection = None
        self.listener = None
        self.latest_event = None
        self.connected = False
        self.connection_context = None

        # 创建一个简单的监听器来获取跟踪事件
        class TrackingListener(leap.Listener):
            def __init__(self, detector):
                super().__init__()
                self.detector = detector

            def on_connection_event(self, event):
                self.detector.connected = True
                print("Leap Motion: Connected")

            def on_device_event(self, event):
                try:
                    with event.device.open():
                        info = event.device.get_info()
                except leap.LeapCannotOpenDeviceError:
                    info = event.device.get_info()
                print(f"Leap Motion: Found device {info.serial}")

            def on_tracking_event(self, event):
                self.detector.latest_event = event

        self.listener = TrackingListener(self)
        self.connection = leap.Connection()
        self.connection.add_listener(self.listener)

        # 打开连接（使用 context manager 确保连接保持打开）
        # 注意：connection.open() 返回一个 context manager，我们需要保持它打开
        self.connection_context = self.connection.open()
        self.connection_context.__enter__()  # 手动进入 context
        self.connection.set_tracking_mode(tracking_mode)

        # 等待连接建立
        wait_until(lambda: self.connected, timeout=5)
        print(f"Leap Motion: 初始化完成，等待 {hand_type} 手部数据...")

    def extract_keypoints_from_hand(self, hand: ldt.Hand) -> Optional[np.ndarray]:
        """
        从 Leap Motion 的手部数据中提取21个关键点（与 MediaPipe 格式兼容）
        
        MediaPipe 关键点顺序：
        0: 手腕 (Wrist)
        1-4: 拇指 (Thumb: CMC, MCP, IP, Tip)
        5-8: 食指 (Index: MCP, PIP, DIP, Tip)
        9-12: 中指 (Middle: MCP, PIP, DIP, Tip)
        13-16: 无名指 (Ring: MCP, PIP, DIP, Tip)
        17-20: 小指 (Pinky: MCP, PIP, DIP, Tip)
        
        Leap Motion 结构：
        - hand.arm.next_joint: 手腕
        - hand.digits[0-4]: 5个手指（拇指、食指、中指、无名指、小指）
        - digit.bones[0-3]: 每个手指4个骨骼
        - bone.prev_joint, bone.next_joint: 骨骼的起点和终点
        
        Args:
            hand: Leap Motion 的手部对象
            
        Returns:
            形状为 (21, 3) 的 numpy 数组，包含21个关键点的 x, y, z 坐标（单位：毫米）
        """
        keypoints = np.zeros((21, 3))

        try:
            # 0: 手腕位置
            if hand.arm and hand.arm.next_joint:
                wrist = hand.arm.next_joint
                keypoints[0] = [wrist.x, wrist.y, wrist.z]
            else:
                # 如果没有手臂数据，使用手掌中心
                palm = hand.palm.position
                keypoints[0] = [palm.x, palm.y, palm.z]

            # 遍历5个手指
            for digit_idx in range(5):
                digit = hand.digits[digit_idx]

                # 每个手指有4个骨骼，对应 MediaPipe 的4个关键点
                # 骨骼0: 从手掌到 MCP (Metacarpophalangeal joint)
                # 骨骼1: MCP 到 PIP (Proximal Interphalangeal joint)
                # 骨骼2: PIP 到 DIP (Distal Interphalangeal joint)
                # 骨骼3: DIP 到 Tip

                if len(digit.bones) >= 4:
                    # MCP: 第一个骨骼的起点
                    if digit.bones[0].prev_joint:
                        mcp = digit.bones[0].prev_joint
                        keypoint_idx = 1 + digit_idx * 4  # 拇指:1, 食指:5, 中指:9, 无名指:13, 小指:17
                        keypoints[keypoint_idx] = [mcp.x, mcp.y, mcp.z]

                    # PIP: 第二个骨骼的起点
                    if digit.bones[1].prev_joint:
                        pip = digit.bones[1].prev_joint
                        keypoint_idx = 2 + digit_idx * 4
                        keypoints[keypoint_idx] = [pip.x, pip.y, pip.z]

                    # DIP: 第三个骨骼的起点
                    if digit.bones[2].prev_joint:
                        dip = digit.bones[2].prev_joint
                        keypoint_idx = 3 + digit_idx * 4
                        keypoints[keypoint_idx] = [dip.x, dip.y, dip.z]

                    # Tip: 最后一个骨骼的终点
                    if digit.bones[3].next_joint:
                        tip = digit.bones[3].next_joint
                        keypoint_idx = 4 + digit_idx * 4
                        keypoints[keypoint_idx] = [tip.x, tip.y, tip.z]
                else:
                    # 如果骨骼数据不完整，尝试使用 distal
                    if hasattr(digit, 'distal') and digit.distal and digit.distal.next_joint:
                        tip = digit.distal.next_joint
                        keypoint_idx = 4 + digit_idx * 4
                        keypoints[keypoint_idx] = [tip.x, tip.y, tip.z]

            # 将单位从毫米转换为米（Leap Motion 使用毫米，MediaPipe 使用米）
            keypoints = keypoints / 1000.0

            return keypoints

        except Exception as e:
            print(f"提取关键点时出错: {e}")
            return None

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        从手部关键点估计手腕坐标系（旋转矩阵）
        与 single_hand_detector.py 中的方法保持一致
        
        Args:
            keypoint_3d_array: 形状为 (21, 3) 的关键点数组
            
        Returns:
            形状为 (3, 3) 的旋转矩阵
        """
        assert keypoint_3d_array.shape == (21, 3)
        # 使用手腕(0)、食指MCP(5)、中指MCP(9)
        points = keypoint_3d_array[[0, 5, 9], :]

        # 计算从手掌到中指MCP的向量
        x_vector = points[0] - points[2]

        # 使用SVD进行法向量拟合
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram-Schmidt 正交化
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # 假设从无名指到食指的向量与 MANO 坐标系中的 z 轴相似
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

    def detect(self, rgb: Optional[np.ndarray] = None) -> Tuple[int, Optional[np.ndarray], Optional[object], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        检测手部关键点（与 SingleHandDetector 接口兼容）
        
        Args:
            rgb: RGB图像（Leap Motion 不需要，但保留接口兼容性）
            
        Returns:
            (检测到的手数量, 关节位置(21x3), 2D关键点(占位符), 手腕旋转矩阵, 3D关键点数组)
        """
        if self.latest_event is None:
            return 0, None, None, None, None

        # 查找指定类型的手
        target_hand = None
        for hand in self.latest_event.hands:
            if hand.type == self.leap_hand_type:
                target_hand = hand
                break

        if target_hand is None:
            return 0, None, None, None, None

        # 提取关键点（全局位置）
        keypoint_3d_global = self.extract_keypoints_from_hand(target_hand)
        if keypoint_3d_global is None:
            return 0, None, None, None, None

        keypoint_3d_global = keypoint_3d_global @ self.camera2table.T

        # 保存全局位置用于可视化
        keypoint_3d_for_vis = keypoint_3d_global.copy()

        # 将坐标原点移到手腕（索引0）- 用于重定向计算
        keypoint_3d_relative = keypoint_3d_global - keypoint_3d_global[0:1, :]

        # 估计手腕坐标系（旋转矩阵）- 使用相对位置
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_relative)

        # 将关键点从 Leap Motion 坐标系转换到 MANO 坐标系（消除了旋转，但是有绝对位置）
        joint_pos = keypoint_3d_global # @ mediapipe_wrist_rot @ self.operator2mano

        # 返回格式与 SingleHandDetector 兼容
        # keypoint_2d 设为 None（保持兼容性）
        # keypoint_3d_for_vis 是全局位置，用于可视化
        return 1, joint_pos, None, mediapipe_wrist_rot, keypoint_3d_for_vis

    def project_3d_to_2d(self, keypoint_3d: np.ndarray, image_size: tuple = (720, 1280)) -> np.ndarray:
        """
        将3D关键点投影到2D图像平面（参考 visualiser.py 的方法）
        使用 x 和 z 坐标进行投影，y 作为深度信息
        直接使用原始坐标，不进行自动缩放和居中
        
        Args:
            keypoint_3d: 形状为 (21, 3) 的3D关键点数组（单位：米）
            image_size: 图像尺寸 (height, width)
            
        Returns:
            形状为 (21, 2) 的2D关键点数组（像素坐标）
        """
        keypoint_3d = keypoint_3d @ self.camera2table
        # 将单位从米转换回毫米（用于投影计算）
        keypoints_mm = keypoint_3d * 1000.0

        # 计算中心偏移（参考 visualiser.py 的 get_joint_position）
        # visualiser.py: return int(bone.x + (self.screen_size[1] / 2)), int(bone.z + (self.screen_size[0] / 2))
        center_x = image_size[1] / 2
        center_y = image_size[0] / 2

        # 直接使用原始坐标投影，不进行缩放和居中
        keypoint_2d = np.zeros((21, 2))
        keypoint_2d[:, 0] = keypoints_mm[:, 0] + center_x  # x 坐标
        keypoint_2d[:, 1] = keypoints_mm[:, 2] + center_y  # z 坐标作为 y

        return keypoint_2d.astype(int)

    def draw_skeleton_on_image(
        self, image: np.ndarray, keypoint_3d: Optional[np.ndarray], style: str = "default"
    ) -> np.ndarray:
        """
        在图像上绘制手部骨架（基于3D关键点投影到2D）
        
        Args:
            image: 输入图像（如果为None，会创建一个新的画布）
            keypoint_3d: 3D关键点数组（形状为 (21, 3)）
            style: 绘制样式，"default" 或 "white"
            
        Returns:
            绘制了骨架的图像
        """
        if keypoint_3d is None:
            if image is None:
                # 创建一个默认大小的画布
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
            return image

        # 如果图像为None，创建一个画布
        if image is None:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 将3D关键点投影到2D
        keypoint_2d = self.project_3d_to_2d(keypoint_3d, image.shape[:2])

        # MediaPipe 手部连接关系
        # 定义手部连接（参考 MediaPipe 的 HAND_CONNECTIONS）
        connections = [
            # 手腕到手指根部
            (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
            # 拇指
            (1, 2), (2, 3), (3, 4),
            # 食指
            (5, 6), (6, 7), (7, 8),
            # 中指
            (9, 10), (10, 11), (11, 12),
            # 无名指
            (13, 14), (14, 15), (15, 16),
            # 小指
            (17, 18), (18, 19), (19, 20),
        ]

        # 绘制连接线
        if style == "default":
            line_color = (0, 255, 0)  # 绿色
            point_color = (0, 0, 255)  # 红色
            line_thickness = 2
            point_radius = 4
        else:  # white style
            line_color = (255, 255, 255)  # 白色
            point_color = (255, 48, 48)  # 红色
            line_thickness = 2
            point_radius = 4

        # 绘制连接线
        for start_idx, end_idx in connections:
            start = tuple(keypoint_2d[start_idx])
            end = tuple(keypoint_2d[end_idx])
            # 检查点是否在图像范围内
            if (0 <= start[0] < image.shape[1] and 0 <= start[1] < image.shape[0] and
                0 <= end[0] < image.shape[1] and 0 <= end[1] < image.shape[0]):
                cv2.line(image, start, end, line_color, line_thickness)

        # 绘制关键点
        for i, point in enumerate(keypoint_2d):
            point_tuple = tuple(point)
            if 0 <= point_tuple[0] < image.shape[1] and 0 <= point_tuple[1] < image.shape[0]:
                cv2.circle(image, point_tuple, point_radius, point_color, -1)
                # 可选：在关键点上绘制一个小圆表示关键点
                if i == 0:  # 手腕用更大的点
                    cv2.circle(image, point_tuple, point_radius + 2, point_color, -1)

        # 添加文本信息
        cv2.putText(
            image,
            f"Leap Motion - {self.hand_type} Hand",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # 添加帧ID信息（如果有）
        if hasattr(self, 'latest_event') and self.latest_event is not None:
            frame_id_text = f"Frame: {self.latest_event.tracking_frame_id}"
            cv2.putText(
                image,
                frame_id_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        return image

    def close(self):
        """关闭 Leap Motion 连接"""
        if self.connection_context:
            try:
                self.connection_context.__exit__(None, None, None)
            except Exception as e:
                print(f"关闭 Leap Motion 连接时出错: {e}")
            self.connection_context = None
        if self.connection:
            self.connection = None
