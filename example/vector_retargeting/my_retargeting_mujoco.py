import multiprocessing
import sys
import time
import threading
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import h5py
import numpy as np
import tyro
from loguru import logger
import mujoco
import mujoco.viewer
from pynput import keyboard

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector
from leap_motion_detector import LeapMotionHandDetector

from utils.timer import Timer
from utils.opencv_cam import find_camera_with_resolution
from utils.rerun_board import RerunBoard
from utils.misc_utils import DummyClass
import rerun as rr
import mediapipe as mp
import leap


CAMERA2TABLE = np.array(
    [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]
)


def process_detection_and_retargeting(
    qpos_queue: multiprocessing.Queue,
    robot_dir: str,
    config_path: str,
    camera_path: Optional[str] = None,
    input_source: str = "webcam",
):
    """
    进程一：从相机获取图像、处理帧、检测手部、执行重定向、存储关节角到队列
    
    Args:
        input_source: 输入源类型，"webcam" 或 "leap_motion"
    """
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"进程一：开始重定向计算，配置文件 {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"

    # 根据输入源选择检测器
    if input_source == "leap_motion":
        detector = LeapMotionHandDetector(hand_type=hand_type, tracking_mode=leap.TrackingMode.Desktop)
        cap = None  # Leap Motion 不需要 OpenCV 相机
        logger.info("使用 Leap Motion 作为输入源")
    else:
        detector = SingleHandDetector(hand_type=hand_type, selfie=False)
        # 打开相机
        if camera_path is None:
            camera_id = find_camera_with_resolution(target_width=1280, target_height=720)
            cap = cv2.VideoCapture(camera_id)
        else:
            cap = cv2.VideoCapture(camera_path)

        if not cap.isOpened():
            logger.error("无法打开相机")
            return
        logger.info("使用 Webcam 作为输入源")

    # rerun board
    board = RerunBoard(f"DexRetargeting_{time.strftime('%m_%d_%H_%M', time.localtime())}",
                       template="dex_retargeting")
    # board = DummyClass()

    # 计时器：用于统计关键步骤的耗时（仅用于 debug 分析）
    timer = Timer(enabled=True)

    while True:
        # 以每帧为单位重新开始计时
        timer.start()

        # 根据输入源获取数据
        if input_source == "leap_motion":
            # Leap Motion: 直接检测，不需要读取图像
            rgb = None
            bgr = None
        else:
            # Webcam: 读取图像
            success, bgr = cap.read()
            if not success:
                time.sleep(1 / 30.0)
                continue
            # 处理帧：BGR转RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        timer.check("preprocess")

        # 检测手部
        _, joint_pos, keypoint_2d, mediapipe_wrist_rot, keypoint_3d = detector.detect(rgb)
        timer.check("detect")

        # # 显示检测结果
        # if input_source == "webcam" and bgr is not None:
        #     # Webcam 模式：在原始图像上绘制
        #     bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        #     cv2.imshow("realtime_retargeting_demo", bgr)
        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break
        # elif input_source == "leap_motion":
        #     # Leap Motion 模式：创建虚拟画布并绘制3D关键点
        #     if keypoint_3d is not None:
        #         # 创建一个画布用于可视化
        #         vis_image = detector.draw_skeleton_on_image(None, keypoint_3d, style="default")
        #         cv2.imshow("realtime_retargeting_demo", vis_image)
        #     else:
        #         # 如果没有检测到手部，显示空白画布
        #         vis_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        #         cv2.putText(
        #             vis_image,
        #             f"Leap Motion - Waiting for {hand_type} hand...",
        #             (10, 360),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1.0,
        #             (255, 255, 255),
        #             2,
        #         )
        #         cv2.imshow("realtime_retargeting_demo", vis_image)

        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break

        # 记录结果到 rerun
        if joint_pos is not None:
            # 关键点
            for i in range(keypoint_3d.shape[0]):
                board.log(
                    f"world/human/keypoint/{i}",
                    rr.Points3D(positions=[keypoint_3d[i]],
                                colors=[[255, 0, 0]], radii=0.005,
                                labels=f"{i}",
                    ),
                )  # , static=True
            # 连接线
            mp_hands = mp.solutions.hands
            hand_connections = mp_hands.HAND_CONNECTIONS
            for pair in hand_connections:
                board.log(
                    f"world/human/connection/{pair}",
                    rr.Arrows3D(origins=[keypoint_3d[pair[0]]],
                                vectors=[keypoint_3d[pair[1]] - keypoint_3d[pair[0]]],
                                colors=[[0, 255, 0]],
                                # labels=f"{pair}",
                    ),
                )
            # 手腕坐标系
            board.log_axes(
                translation=keypoint_3d[0],
                rotation=mediapipe_wrist_rot,
                root="world/human",
                name="wrist_rot",
                axis_size=0.05,
            )

        # 执行重定向
        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
            # 即使没有检测到手部，也发送None到队列，让渲染进程知道
            try:
                qpos_queue.put_nowait(None)
            except:
                pass
        else:
            # 从人手上拿位置信息
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices

            if retargeting_type == "POSITION":
                # Position retargeting: 使用绝对位置
                ref_value = joint_pos[indices, :]
            elif retargeting_type == "JOINT":
                # Joint retargeting: 用完整 joint_pos 直接计算 robot_qpos（由 JointOptimizer 完成）
                ref_value = joint_pos  # (N, 3) 人手关键点 3D 位置
            else:
                # Vector retargeting: 使用相对位置
                joint_pos_relative = joint_pos - joint_pos[0, :]
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos_relative[task_indices, :] - joint_pos_relative[origin_indices, :]  # 第二组 index 减去 第一组 index 的相对位置
            # for allegro & vector: array([[ 0,  0,  0,  0], [ 4,  8, 12, 16]]), ref_value 计算了 Thumb Tip, Index Tip, Middle Tip, Ring Tip 的相对根部的位置
            # for allegro & dexpilot: array([[ 8, 12, 16, 12, 16, 16,  0,  0,  0,  0], [ 4,  4,  4,  8,  8, 12,  4,  8, 12, 16]]), ref_value 计算四个指尖和手腕、四个指尖之间的相对位置

            # 执行重定向（返回完整的 robot_qpos，包括固定关节，已应用适配器）
            robot_qpos = retargeting.retarget(ref_value)
            timer.check("retarget")
            for i in range(robot_qpos.shape[0]):
                # if i not in [10, 11, 12, 13]:  # select some of the joint to show
                #     continue
                board.log(
                    f"joint_angles/joint_{i:02d}",
                    rr.Scalars(robot_qpos[i]),
                )

            # 获取机器人关节的3D位置并可视化
            robot = retargeting.optimizer.robot

            # 计算所有关节的3D位置
            joint_positions = robot.get_all_joint_positions(robot_qpos)

            # 获取关节连接关系（方法内部有缓存，第一次调用后会自动缓存）
            joint_connections = robot.get_joint_connections()

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

            # logger.debug(f"Joint angles: {robot_qpos.round(2)}")

            # 记录本帧关键步骤的耗时（debug 级别）
            try:
                preprocess_time = timer.times["preprocess"][-1]
                detect_time = timer.times["detect"][-1]
                retarget_time = timer.times["retarget"][-1]
                # logger.debug(
                #     "Timing (s) - preprocess: {:.4f}, detect: {:.4f}, retarget: {:.4f}",
                #     preprocess_time,
                #     detect_time,
                #     retarget_time,
                # )
            except Exception:
                # 计时信息仅用于调试分析，任何异常都不应影响主流程
                pass

            try:
                qpos_queue.put_nowait(
                    {
                        "qpos": robot_qpos,
                    }
                )
            except:
                pass  # 队列满了，跳过这一帧，保持实时性

        time.sleep(1 / 30.0)
        # time.sleep(1 / 10.0)

    # 清理资源
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
    if hasattr(detector, 'close'):
        detector.close()
    logger.info("进程一：结束")


def generate_test_qpos(dim_idx: int, progress: float, total_dim: int = 22) -> np.ndarray:
    """
    生成测试用的 qpos 向量，指定维度按照正弦值波动两个来回（4π周期）
    
    Args:
        dim_idx: 要测试的维度索引
        progress: 当前维度测试的进度 [0, 1]，0表示开始，1表示结束
        total_dim: 总维度数，默认22（6+16）
    
    Returns:
        qpos: 测试用的关节角度向量，形状为 (total_dim,)
              只有指定维度有正弦值，其他维度为0
    """
    # 计算正弦值：2个来回 = 4π，所以角度是 4π * progress
    angle = 4 * np.pi * progress
    sin_value = np.sin(angle)
    
    # 创建 qpos 数组：当前测试维度使用正弦值，其他维度为0
    qpos = np.zeros(total_dim, dtype=np.float32)
    qpos[dim_idx] = sin_value
    
    return qpos


def process_test_qpos(qpos_queue: multiprocessing.Queue):
    """
    测试子进程：生成测试用的 qpos 数据，用于测试主进程的 mujoco
    
    每三秒测试一个维度（共22个维度：6+16），每个维度按照正弦值波动两个来回（4π周期），
    共测试 3*22=66 秒，然后循环。
    
    Args:
        qpos_queue: 用于传递 qpos 数据的队列
    """
    logger.info("测试子进程：开始生成测试 qpos 数据")
    
    TOTAL_DIM = 22  # 6 + 16 个维度
    DIM_TEST_DURATION = 3.0  # 每个维度测试3秒
    CYCLE_COUNT = 2  # 每个维度波动2个来回（4π）
    UPDATE_RATE = 30.0  # 更新频率 30Hz
    UPDATE_INTERVAL = 1.0 / UPDATE_RATE
    
    while True:
        # 循环测试所有维度
        for dim_idx in range(TOTAL_DIM):
            dim_start_time = time.time()
            logger.info(f"开始测试维度 {dim_idx}/{TOTAL_DIM-1}")
            
            # 在当前维度的3秒测试时间内循环
            while True:
                current_time = time.time()
                elapsed = current_time - dim_start_time
                
                # 如果超过3秒，切换到下一个维度
                if elapsed >= DIM_TEST_DURATION:
                    break
                
                # 计算当前维度在3秒内的进度 [0, 1]
                progress = elapsed / DIM_TEST_DURATION
                
                # 使用核心函数生成 qpos
                qpos = generate_test_qpos(dim_idx, progress, TOTAL_DIM)
                
                # 计算正弦值用于日志输出
                sin_value = qpos[dim_idx]
                
                # 输出日志
                logger.info(f"测试维度 {dim_idx}, 值: {sin_value:.4f}, 进度: {progress*100:.1f}%")
                
                # 将 qpos 放入队列
                try:
                    qpos_queue.put_nowait(
                        {
                            "qpos": qpos,
                        }
                    )
                except:
                    pass  # 队列满了，跳过这一帧，保持实时性
                
                # 控制更新频率
                time.sleep(UPDATE_INTERVAL)
            
            logger.info(f"完成测试维度 {dim_idx}/{TOTAL_DIM-1}")
        
        logger.info("完成一轮所有维度测试，开始下一轮循环")


# 无 XML 相机时使用的默认相机参数（与 data/frame_render.py 对齐）
DEFAULT_LOOKAT = [0.0, 0.0, 0.2]
DEFAULT_CAMERA_PARAMS = {"distance": 0.8, "elevation": -30, "azimuth": 0}


def _make_dynamic_camera(par: dict) -> mujoco.MjvCamera:
    """根据参数字典创建并配置一个 MjvCamera，look at DEFAULT_LOOKAT。"""
    camera = mujoco.MjvCamera()
    try:
        mujoco.mjv_defaultCamera(camera)
    except AttributeError:
        pass  # 部分版本无此函数，下面手动设置的参数已足够
    camera.lookat[:] = DEFAULT_LOOKAT
    camera.distance = par["distance"]
    camera.elevation = par["elevation"]
    camera.azimuth = par["azimuth"]
    return camera


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
    input_source: str = "webcam",
    config_path_override: Optional[str] = None,
    dataset_dir: str = "data/hdf5",
    start_key: str = "s",
    stop_key: str = "e",
    camera_names: list[str] = [],  # 为空时使用 XML 中定义的所有相机；非空时仅使用列出的相机
    joint_indices: Optional[list[int]] = list(range(22)),
    mj_xml_path: str = "/mnt/1tb1/xuechao/MuJoCo-Asset-Pipeline/asset/scene/freejoint/teleop_scene_left_077_rubiks_cube",
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
        input_source: Input source type, "webcam" (default) or "leap_motion".
        config_path_override: Optional custom config path. If provided, will override the default config path.
        mj_xml_path: MuJoCo 场景 XML 文件路径或包含 .xml 的目录；为目录时自动取该目录下第一个 .xml。
    """
    if config_path_override is not None:
        config_path = Path(config_path_override)
    else:
        config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    print(f"Using config_path: {config_path}")
    input("Press Enter to continue...")
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    qpos_queue = multiprocessing.Queue(maxsize=2)  # 只保留最新2个关节角数据

    # process_detection_and_retargeting(
    #     qpos_queue, str(robot_dir), str(config_path), camera_path, input_source
    # )
    # quit()

    # # 进程一：检测和重定向
    # detection_process = multiprocessing.Process(
    #     target=process_detection_and_retargeting,
    #     args=(qpos_queue, str(robot_dir), str(config_path), camera_path, input_source),
    # )

    # # 进程二：可视化
    # visualization_process = multiprocessing.Process(
    #     target=process_visualization,
    #     args=(qpos_queue, str(robot_dir), str(config_path)),
    # )

    # detection_process.start()
    # visualization_process.start()

    # detection_process.join()
    # visualization_process.join()

    # ---------------- Mujoco 可视化（必须在主线程中运行） ----------------
    # 加载 Mujoco 场景
    mj_xml_path = Path(mj_xml_path)
    if mj_xml_path.is_dir():
        first_xml = next(mj_xml_path.glob("*.xml"), None)
        if first_xml is None:
            raise FileNotFoundError(f"目录下未找到 .xml 文件: {mj_xml_path}")
        mj_xml_path = first_xml
    if not mj_xml_path.exists():
        raise FileNotFoundError(f"Mujoco 场景文件不存在: {mj_xml_path}")

    model = mujoco.MjModel.from_xml_path(str(mj_xml_path))
    data = mujoco.MjData(model)

    # 用于录制的相机列表：(相机名, 相机规格)；规格为 XML 相机名(str) 或 MjvCamera
    if model.ncam > 0:
        # XML 中所有相机名称
        xml_camera_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)
        ]
        if not camera_names:
            # 默认为空：使用 XML 中定义的所有相机
            names_to_use = xml_camera_names
        else:
            # 非空：仅使用指定的相机（与 XML 中的名称匹配）
            names_to_use = [n for n in camera_names if n in xml_camera_names]
            if len(names_to_use) < len(camera_names):
                unknown = [n for n in camera_names if n not in xml_camera_names]
                logger.warning(f"以下相机名在 XML 中不存在，已忽略: {unknown}")
        recording_camera_specs: list[tuple[str, str | mujoco.MjvCamera]] = [
            (name, name) for name in names_to_use
        ]
        recording_camera_names = list(names_to_use)
    else:
        # XML 中无相机：使用内置默认相机
        default_camera = _make_dynamic_camera(DEFAULT_CAMERA_PARAMS)
        recording_camera_specs = [("default", default_camera)]
        recording_camera_names = ["default"]
        if camera_names and camera_names != ["default"]:
            logger.info(
                "场景 XML 中未定义相机，使用内置默认相机（相机名为 default）；"
                "录制时请使用 --camera-names default"
            )

    # 离屏渲染器，用于采集图像
    renderer = mujoco.Renderer(model, width=640, height=480)

    # 控制向量：前 6 个为 hand_root 的 6DOF（tx, ty, tz, rx, ry, rz），后 16 个为 Allegro 手指关节
    ROOT_CTRL_DIM = 6
    FINGER_CTRL_DIM = 16
    assert (
        model.nu == ROOT_CTRL_DIM + FINGER_CTRL_DIM
    ), "Mujoco 模型中的 actuator 数量不等于 6DOF 根关节 + 16 个手指关节"

    root_slice = slice(0, ROOT_CTRL_DIM)
    finger_slice = slice(ROOT_CTRL_DIM, ROOT_CTRL_DIM + FINGER_CTRL_DIM)

    # 用于保存最新一帧的目标
    latest_qpos = None
    latest_wrist_pos = None
    latest_wrist_rot = None

    # 简单的旋转矩阵 -> ZYX 欧拉角 (rz, ry, rx)，用于驱动 hand_root 的转动关节
    def mat_to_euler_zyx(R: np.ndarray):
        """将 3x3 旋转矩阵转换为 ZYX 欧拉角 (rz, ry, rx)。"""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0.0
        return rz, ry, rx

    # 归一化关节索引配置（如果提供的话）
    if joint_indices is not None:
        joint_indices = list(joint_indices)

    # 子进程：检测和重定向（子进程，避免阻塞主线程的渲染）
    detection_process = multiprocessing.Process(
        target=process_detection_and_retargeting,
        args=(qpos_queue, str(robot_dir), str(config_path), camera_path, input_source),
    )
    # detection_process = multiprocessing.Process(
    #     target=process_test_qpos,
    #     args=(qpos_queue,),
    # )
    detection_process.start()
    logger.info("检测与重定向子进程已启动")

    # 键盘监听相关变量
    keyboard_lock = threading.Lock()
    should_exit = False  # 退出标志

    record_start_requested = False
    record_stop_requested = False
    is_recording = False
    episode_idx = 0
    episode_buffers: Optional[dict[str, list]] = None

    # 数据保存目录（HDF5）：在 dataset_dir 下再按当前日期时间创建子文件夹
    dataset_root = Path(dataset_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dataset_dir_path = dataset_root / timestamp
    dataset_dir_path.mkdir(parents=True, exist_ok=True)

    # 将启动的 Python 命令行保存到该子文件夹，便于后续查看参数配置
    try:
        cmd_str = " ".join(sys.argv)
        cmd_file = dataset_dir_path / "command.txt"
        cmd_file.write_text(cmd_str + "\n", encoding="utf-8")
        logger.info(f"启动命令已保存到: {cmd_file}")
    except Exception as e:
        logger.warning(f"无法写入启动命令到 {dataset_dir_path}: {e}")
    # xml 文件名（不含扩展名），用于保存到元数据
    mj_xml_stem = Path(mj_xml_path).stem if mj_xml_path else "scene"

    def init_episode_buffers() -> dict[str, list]:
        buffers: dict[str, list] = {
            "/observations/qpos": [],
            "/observations/qvel": [],
            "/action": [],
        }
        for cam in recording_camera_names:
            buffers[f"/observations/images/{cam}"] = []
        return buffers

    def save_episode(buffers: dict[str, list], idx: int) -> None:
        if not buffers["/action"]:
            logger.warning(f"Episode {idx} 没有任何 step，跳过保存。")
            return

        t0 = time.time()

        # 关节与动作
        qpos_array = np.stack(buffers["/observations/qpos"], axis=0)
        qvel_array = np.stack(buffers["/observations/qvel"], axis=0)
        action_array = np.stack(buffers["/action"], axis=0)

        max_timesteps = min(
            qpos_array.shape[0],
            qvel_array.shape[0],
            action_array.shape[0],
        )
        qpos_array = qpos_array[:max_timesteps]
        qvel_array = qvel_array[:max_timesteps]
        action_array = action_array[:max_timesteps]

        q_dim = qpos_array.shape[1]
        action_dim = action_array.shape[1]

        # 图像
        images_arrays: dict[str, np.ndarray] = {}
        for cam in recording_camera_names:
            key = f"/observations/images/{cam}"
            cam_list = buffers.get(key, [])
            if not cam_list:
                continue
            img_array = np.stack(cam_list, axis=0)
            img_array = img_array[:max_timesteps]
            images_arrays[cam] = img_array

        dataset_path = dataset_dir_path / f"episode_{idx}.hdf5"
        with h5py.File(str(dataset_path), "w", rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs["sim"] = True
            root.attrs["mj_xml_path"] = str(Path(mj_xml_path).resolve())
            root.attrs["scene"] = mj_xml_stem

            obs = root.create_group("observations")
            image_group = obs.create_group("images")

            for cam, img_array in images_arrays.items():
                H, W = img_array.shape[1:3]
                dset = image_group.create_dataset(
                    cam,
                    data=img_array,
                    dtype="uint8",
                    chunks=(1, H, W, 3),
                )
                dset.attrs["CLASS"] = np.bytes_("IMAGE")

            obs.create_dataset("qpos", data=qpos_array)
            obs.create_dataset("qvel", data=qvel_array)
            root.create_dataset("action", data=action_array)

        logger.info(
            f"Episode {idx} 已保存到 {dataset_path}，"
            f"步数={max_timesteps}, q_dim={q_dim}, action_dim={action_dim}, "
            f"耗时 {time.time() - t0:.2f} 秒"
        )

    def on_press(key):
        """键盘按下事件处理。"""
        nonlocal record_start_requested, record_stop_requested, should_exit
        try:
            if hasattr(key, "char") and key.char:
                with keyboard_lock:
                    if key.char == start_key and not is_recording:
                        record_start_requested = True
                    elif key.char == stop_key and is_recording:
                        record_stop_requested = True
                    elif key.char == "q":
                        should_exit = True
        except AttributeError:
            pass

    # 启动键盘监听器（在后台线程中运行）
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()
    logger.info(
        f"键盘监听已启动：按 '{start_key}' 开始记录，按 '{stop_key}' 结束并保存，按 'q' 退出"
    )

    # 启动被动 viewer，在主线程中进行物理仿真与渲染
    with mujoco.viewer.launch_passive(model, data) as viewer:
        logger.info("Mujoco viewer 已启动")
        # 可视化相机所在的位置和名字
        options = viewer.opt
        mujoco.mjv_defaultOption(options)
        options.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        options.label = mujoco.mjtLabel.mjLABEL_CAMERA
        sim_start = time.time()
        control_rate_hz = 60.0
        control_interval = 1.0 / control_rate_hz
        last_control_time = time.time()

        while viewer.is_running() and not should_exit:
            now = time.time()

            # 从队列中取出最新的目标（关节角 + 手腕姿态）
            msg = None
            while True:
                try:
                    msg = qpos_queue.get_nowait()
                except Empty:
                    break

            if msg is not None:
                if isinstance(msg, dict):
                    latest_qpos = msg.get("qpos", None)
                    latest_wrist_pos = msg.get("wrist_pos", None)
                    latest_wrist_rot = msg.get("wrist_rot", None)

            # 以固定频率更新控制目标（角度目标），Mujoco 继续做物理仿真
            if now - last_control_time >= control_interval:
                # 手指关节目标：使用检测得到的 robot_qpos 作为期望位置
                if latest_qpos is not None:
                    q = np.asarray(latest_qpos).reshape(-1)
                    data.ctrl[0] = q[0] + 0.2
                    data.ctrl[1] = q[1]
                    data.ctrl[2] = q[2] - 0.6

                    data.ctrl[3] = q[3]  # + np.pi/2
                    data.ctrl[4] = q[4]  # - np.pi
                    data.ctrl[5] = q[5]  # - np.pi

                    data.ctrl[14:18] = q[6:10]  # 小指
                    data.ctrl[18:22] = q[10:14]  # 拇指
                    data.ctrl[10:14] = q[14:18]  # 中指
                    data.ctrl[6:10] = q[18:22]  # 食指

                    # 记录当前步的数据到缓存（若正在录制）
                    if is_recording and episode_buffers is not None:
                        if joint_indices is None:
                            qpos_sample = np.asarray(data.qpos).copy()
                            qvel_sample = np.asarray(data.qvel).copy()
                        else:
                            qpos_sample = np.asarray(data.qpos)[joint_indices].copy()
                            qvel_sample = np.asarray(data.qvel)[joint_indices].copy()

                        episode_buffers["/observations/qpos"].append(qpos_sample)
                        episode_buffers["/observations/qvel"].append(qvel_sample)

                        action_sample = np.asarray(latest_qpos).reshape(-1)
                        episode_buffers["/action"].append(action_sample)

                        for cam_name, cam_spec in recording_camera_specs:
                            try:
                                renderer.update_scene(data, camera=cam_spec)
                                img = renderer.render()
                                if img.dtype != np.uint8:
                                    img = (np.clip(img, 0.0, 1.0) * 255).astype(
                                        np.uint8
                                    )
                                episode_buffers[
                                    f"/observations/images/{cam_name}"
                                ].append(img)
                            except Exception as e:
                                logger.warning(f"渲染相机 {cam_name} 失败: {e}")

                last_control_time = now

            # 按真实时间步长推进物理仿真
            while data.time < now - sim_start:
                mujoco.mj_step(model, data)

            # 检查退出标志
            pending_buffers = None
            pending_episode_idx = None
            with keyboard_lock:
                if should_exit:
                    logger.info("检测到退出按键 'q'，准备退出...")
                    break
                
                if record_start_requested:
                    episode_buffers = init_episode_buffers()
                    is_recording = True
                    record_start_requested = False
                    logger.info(f"开始记录 episode_{episode_idx}")

                if record_stop_requested:
                    if episode_buffers is not None:
                        pending_buffers = episode_buffers
                        pending_episode_idx = episode_idx
                        episode_idx += 1
                        episode_buffers = None
                    is_recording = False
                    record_stop_requested = False

            if pending_buffers is not None and pending_episode_idx is not None:
                save_episode(pending_buffers, pending_episode_idx)

            # 同步 viewer
            viewer.sync()

        logger.info("Mujoco viewer 已关闭，准备结束子进程")

    # 停止键盘监听器
    keyboard_listener.stop()
    keyboard_listener.join()
    logger.info("键盘监听已停止")
    
    # 关闭检测进程
    if detection_process.is_alive():
        detection_process.terminate()
        detection_process.join()
    logger.info("检测与重定向子进程已结束")


if __name__ == "__main__":
    tyro.cli(main)
