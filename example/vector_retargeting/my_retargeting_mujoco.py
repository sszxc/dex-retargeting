import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import tyro
from loguru import logger
import mujoco
import mujoco.viewer

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

        # 显示检测结果
        if input_source == "webcam" and bgr is not None:
            # Webcam 模式：在原始图像上绘制
            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
            cv2.imshow("realtime_retargeting_demo", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif input_source == "leap_motion":
            # Leap Motion 模式：创建虚拟画布并绘制3D关键点
            if keypoint_3d is not None:
                # 创建一个画布用于可视化
                vis_image = detector.draw_skeleton_on_image(None, keypoint_3d, style="default")
                cv2.imshow("realtime_retargeting_demo", vis_image)
            else:
                # 如果没有检测到手部，显示空白画布
                vis_image = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    vis_image,
                    f"Leap Motion - Waiting for {hand_type} hand...",
                    (10, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("realtime_retargeting_demo", vis_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 记录结果到 rerun
        if joint_pos is not None:
            # 关键点
            for i in range(keypoint_3d.shape[0]):
                board.log(
                    f"world/human/keypoint/{i}",
                    rr.Points3D(positions=[keypoint_3d[i]],
                                colors=[[255, 0, 0]], radii=0.005,
                                # labels=f"{i}",
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
            else:
                # Vector retargeting: 使用相对位置
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]  # 第二组 index 减去 第一组 index 的相对位置
            # for allegro & vector: array([[ 0,  0,  0,  0], [ 4,  8, 12, 16]]), ref_value 计算了 Thumb Tip, Index Tip, Middle Tip, Ring Tip 的相对根部的位置
            # for allegro & dexpilot: array([[ 8, 12, 16, 12, 16, 16,  0,  0,  0,  0], [ 4,  4,  4,  8,  8, 12,  4,  8, 12, 16]]), ref_value 计算四个指尖和手腕、四个指尖之间的相对位置

            # 执行重定向（返回完整的 robot_qpos，包括固定关节，已应用适配器）
            robot_qpos = retargeting.retarget(ref_value)
            timer.check("retarget")

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

            logger.debug(f"Joint angles: {robot_qpos.round(2)}")

            # 记录本帧关键步骤的耗时（debug 级别）
            try:
                preprocess_time = timer.times["preprocess"][-1]
                detect_time = timer.times["detect"][-1]
                retarget_time = timer.times["retarget"][-1]
                logger.debug(
                    "Timing (s) - preprocess: {:.4f}, detect: {:.4f}, retarget: {:.4f}",
                    preprocess_time,
                    detect_time,
                    retarget_time,
                )
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

        # time.sleep(1 / 30.0)
        time.sleep(1 / 10.0)

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


def process_visualization_SAPIEN(qpos_queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    """
    进程二：初始化仿真器以及渲染，读取关节角，更新状态，渲染
    """
    try:
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        logger.info(f"进程二：开始初始化仿真器，配置文件 {config_path}")

        config = RetargetingConfig.load_from_file(config_path)
        retargeting = config.build()

        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")

        # 初始化场景
        scene = sapien.Scene()
        render_mat = sapien.render.RenderMaterial()
        render_mat.base_color = [0.06, 0.08, 0.12, 1]
        render_mat.metallic = 0.0
        render_mat.roughness = 0.9
        render_mat.specular = 0.8
        scene.add_ground(
            -0.2, render_material=render_mat, render_half_size=[1000, 1000]
        )

        # 光照设置
        scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
        scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
        scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
        scene.set_environment_map(
            create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
        )
        scene.add_area_light_for_ray_tracing(
            sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
        )

        # 相机设置
        cam = scene.add_camera(
            name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
        )
        cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

        # 初始化Viewer
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())

        # 加载机器人
        loader = scene.create_urdf_loader()
        filepath = Path(config.urdf_path)
        robot_name = filepath.stem
        loader.load_multiple_collisions_from_file = True

        # 根据机器人类型设置缩放
        if "ability" in robot_name:
            loader.scale = 1.5
        elif "dclaw" in robot_name:
            loader.scale = 1.25
        elif "allegro" in robot_name:
            loader.scale = 1.4
        elif "shadow" in robot_name:
            loader.scale = 0.9
        elif "bhand" in robot_name:
            loader.scale = 1.5
        elif "leap" in robot_name:
            loader.scale = 1.4
        elif "svh" in robot_name:
            loader.scale = 1.5

        if "glb" not in robot_name:
            filepath = str(filepath).replace(".urdf", "_glb.urdf")
        else:
            filepath = str(filepath)

        robot = loader.load(filepath)

        # 根据机器人类型设置初始姿态
        if "ability" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.15]))
        elif "shadow" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.2]))
        elif "dclaw" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.15]))
        elif "allegro" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.05]))
        elif "bhand" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.2]))
        elif "leap" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.15]))
        elif "svh" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.13]))

        # 建立关节名称映射（retargeting的关节顺序 -> sapien的关节顺序）
        sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        retargeting_joint_names = retargeting.joint_names
        retargeting_to_sapien = np.array(
            [retargeting_joint_names.index(name) for name in sapien_joint_names]
        ).astype(int)

        logger.info("进程二：开始渲染循环")

        # 初始化完成后立即渲染一次，确保窗口显示出来
        viewer.render()
        logger.info("进程二：窗口已初始化")

        last_render_time = time.time()
        render_interval = 1.0 / 30.0  # 限制渲染频率到30fps

        while True:
            # 从队列读取关节角（清空旧数据，只保留最新的）
            qpos = None
            while True:
                try:
                    qpos = qpos_queue.get_nowait()
                except Empty:
                    break

            # 更新机器人状态
            if qpos is not None:
                robot.set_qpos(qpos[retargeting_to_sapien])

            # 限制渲染频率，但即使没有新数据也要持续渲染以保持窗口响应
            current_time = time.time()
            if current_time - last_render_time >= render_interval:
                viewer.render()
                last_render_time = current_time

            # 如果没有新数据，稍微等待一下
            if qpos is None:
                time.sleep(0.01)
    except Exception as e:
        logger.error(f"进程二发生错误: {e}")
        import traceback

        traceback.print_exc()
        raise


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
    input_source: str = "webcam",
    config_path_override: Optional[str] = None,
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
    """
    if config_path_override is not None:
        config_path = Path(config_path_override)
    else:
        config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
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
    # 加载 Mujoco 场景：基于 teleop_scene_left.xml
    project_root = Path(__file__).absolute().parent.parent.parent
    mj_xml_path = (
        project_root / "src" / "mujoco" / "wonik_allegro" / "teleop_scene_left.xml"
    )
    if not mj_xml_path.exists():
        raise FileNotFoundError(f"Mujoco 场景文件不存在: {mj_xml_path}")

    model = mujoco.MjModel.from_xml_path(str(mj_xml_path))
    data = mujoco.MjData(model)

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

    # 启动被动 viewer，在主线程中进行物理仿真与渲染
    with mujoco.viewer.launch_passive(model, data) as viewer:
        logger.info("Mujoco viewer 已启动")
        sim_start = time.time()
        control_rate_hz = 60.0
        control_interval = 1.0 / control_rate_hz
        last_control_time = time.time()

        while viewer.is_running():
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
                    data.ctrl[0] = q[0]
                    data.ctrl[1] = q[1]
                    data.ctrl[2] = q[2] - 0.6

                    data.ctrl[3] = q[3]  # - np.pi
                    data.ctrl[4] = q[4] - np.pi
                    data.ctrl[5] = q[5]  # - np.pi

                    data.ctrl[14:18] = q[6:10]  # 小指
                    data.ctrl[18:22] = q[10:14]  # 拇指
                    data.ctrl[10:14] = q[14:18]  # 中指
                    data.ctrl[6:10] = q[18:22]  # 食指

                last_control_time = now

            # 按真实时间步长推进物理仿真
            while data.time < now - sim_start:
                mujoco.mj_step(model, data)

            # 同步 viewer
            viewer.sync()

        logger.info("Mujoco viewer 已关闭，准备结束子进程")

    # 关闭检测进程
    if detection_process.is_alive():
        detection_process.terminate()
        detection_process.join()
    logger.info("检测与重定向子进程已结束")


if __name__ == "__main__":
    tyro.cli(main)
