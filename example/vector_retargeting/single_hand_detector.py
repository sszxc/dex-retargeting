import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark

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


class SingleHandDetector:
    """
    单只手检测器类，使用MediaPipe进行手部关键点检测
    支持将检测结果转换为MANO坐标系
    """
    def __init__(
        self,
        hand_type="Right",
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
    ):
        # 初始化MediaPipe手部检测器
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,  # 视频流模式
            max_num_hands=1,  # 最多检测1只手
            min_detection_confidence=min_detection_confidence,  # 检测置信度阈值
            min_tracking_confidence=min_tracking_confidence,  # 跟踪置信度阈值
        )
        self.selfie = selfie  # 是否为自拍模式（镜像翻转）
        # 根据手部类型选择对应的坐标转换矩阵
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        # 由于摄像头镜像效果，需要反转手部类型
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    @staticmethod
    def draw_skeleton_on_image(
        image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white"
    ):
        """
        在图像上绘制手部骨架
        :param image: 输入图像
        :param keypoint_2d: 2D关键点（归一化坐标）
        :param style: 绘制样式，"default"或"white"
        :return: 绘制了骨架的图像
        """
        if style == "default":
            # 使用默认样式绘制
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
        elif style == "white":
            # 自定义白色样式：红色关键点，白色连接线
            landmark_style = {}
            for landmark in HandLandmark:
                landmark_style[landmark] = DrawingSpec(
                    color=(255, 48, 48), circle_radius=4, thickness=-1  # 红色关键点
                )

            connections = hands_connections.HAND_CONNECTIONS
            connection_style = {}
            for pair in connections:
                connection_style[pair] = DrawingSpec(thickness=2)  # 白色连接线

            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style,
            )

        return image

    def detect(self, rgb):
        """
        检测RGB图像中的手部关键点
        :param rgb: RGB图像数组
        :return: (检测到的手数量, 关节位置(21x3), 2D关键点, 手腕旋转矩阵)
        """
        # 使用MediaPipe处理图像
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return 0, None, None, None, None

        # 查找指定类型的手（左手或右手）获取 index
        desired_hand_num = -1
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            return 0, None, None, None, None

        # 获取检测到的手的关键点
        keypoint_3d = results.multi_hand_world_landmarks[desired_hand_num]  # 3D世界坐标
        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]  # 2D图像坐标（归一化）
        num_box = len(results.multi_hand_landmarks)  # 检测到的手数量

        # 解析3D关键点并转换为numpy数组
        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        # 将坐标原点移到手腕（索引0）注意如果不进行平移，原点在中指指根附近
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        # 估计手腕坐标系（旋转矩阵）
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        # 将关键点从MediaPipe坐标系转换到MANO坐标系（旋转 + 坐标匹配）
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano

        return num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot, keypoint_3d_array

    @staticmethod
    def parse_keypoint_3d(
        keypoint_3d: framework.formats.landmark_pb2.LandmarkList,
    ) -> np.ndarray:
        """
        将MediaPipe的3D关键点protobuf格式转换为numpy数组
        :param keypoint_3d: MediaPipe的3D关键点列表
        :return: 形状为(21, 3)的numpy数组，包含21个关键点的x, y, z坐标
        """
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(
        keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size
    ) -> np.ndarray:
        """
        将MediaPipe的2D关键点（归一化坐标）转换为像素坐标
        :param keypoint_2d: MediaPipe的2D关键点列表（归一化坐标，范围0-1）
        :param img_size: 图像尺寸 (height, width)
        :return: 形状为(21, 2)的numpy数组，包含21个关键点的像素坐标
        """
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        # 将归一化坐标转换为像素坐标
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        x 从 中指指根 指向 手腕
        y 从 手掌心 指向 手背 (normal)
        z 从 中指 指向 食指/大拇指一侧
        注意这里没有用小指，和注释不同
        此外 SVD 是用于 >3 个点的情况，这里只有 3 个可以直接叉乘
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
