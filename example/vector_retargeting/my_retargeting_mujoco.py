import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector

from utils.timer import Timer
from utils.opencv_cam import find_camera_with_resolution
from utils.rerun_board import RerunBoard
import rerun as rr
import mediapipe as mp


def process_detection_and_retargeting(
    qpos_queue: multiprocessing.Queue,
    robot_dir: str,
    config_path: str,
    camera_path: Optional[str] = None,
):
    """
    进程一：从相机获取图像、处理帧、检测手部、执行重定向、存储关节角到队列
    """
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"进程一：开始重定向计算，配置文件 {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    # 计时器：用于统计关键步骤的耗时（仅用于 debug 分析）
    timer = Timer(enabled=True)

    # 打开相机
    if camera_path is None:
        camera_id = find_camera_with_resolution(target_width=1280, target_height=720)
        cap = cv2.VideoCapture(camera_id)
    else:
        cap = cv2.VideoCapture(camera_path)

    if not cap.isOpened():
        logger.error("无法打开相机")
        return

    logger.info("进程一：开始处理图像和重定向")


    board = RerunBoard(f"RerunTest_{time.strftime('%m_%d_%H_%M', time.localtime())}")


    while cap.isOpened():
        # 以每帧为单位重新开始计时
        timer.start()

        success, bgr = cap.read()
        if not success:
            time.sleep(1 / 30.0)
            continue

        # 处理帧：BGR转RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        timer.check("preprocess")

        # 检测手部
        _, joint_pos, keypoint_2d, _, keypoint_3d = detector.detect(rgb)
        timer.check("detect")

        # 显示检测结果
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("realtime_retargeting_demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if joint_pos is not None:
            num_box, joint_pos, keypoint_2d, _, keypoint_3d = detector.detect(rgb)
            for i in range(keypoint_3d.shape[0]):
                board.log(
                    f"3d_points/keypoint_{i}",
                    rr.Points3D(positions=[keypoint_3d[i]], colors=[[255, 0, 0]], radii=0.005),
                )  # , static=True
            mp_hands = mp.solutions.hands
            hand_connections = mp_hands.HAND_CONNECTIONS
            for pair in hand_connections:
                board.log(
                    f"3d_points/connection_{pair}",
                    rr.Arrows3D(origins=[keypoint_3d[pair[0]]], vectors=[keypoint_3d[pair[1]] - keypoint_3d[pair[0]]], colors=[[0, 255, 0]]),
                )

            # board.log(
            #     "3d_points/force",
            #     rr.Arrows3D(origins=[0, 0, 0], vectors=[point_3d], colors=[[0, 255, 0]]),
            # )  # , static=True
            # board.log(
            #     "3d_points/pos_3d",
            #     rr.Transform3D(translation=point_3d, mat3x3=np.random.rand(3, 3)),
            # )

        # 执行重定向
        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
            # 即使没有检测到手部，也发送None到队列，让渲染进程知道
            try:
                qpos_queue.put_nowait(None)
            except:
                pass
        else:
            pass
            # retargeting_type = retargeting.optimizer.retargeting_type
            # indices = retargeting.optimizer.target_link_human_indices
            # if retargeting_type == "POSITION":
            #     ref_value = joint_pos[indices, :]
            # else:
            #     origin_indices = indices[0, :]
            #     task_indices = indices[1, :]
            #     ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            # # 执行重定向
            # qpos = retargeting.retarget(ref_value)
            # timer.check("retarget")

            # # 记录每一时刻的关节角（debug级别）
            # logger.debug(f"Joint angles: {qpos.round(2)}")

            # # 记录本帧关键步骤的耗时（debug 级别）
            # try:
            #     preprocess_time = timer.times["preprocess"][-1]
            #     detect_time = timer.times["detect"][-1]
            #     retarget_time = timer.times["retarget"][-1]
            #     logger.debug(
            #         "Timing (s) - preprocess: {:.4f}, detect: {:.4f}, retarget: {:.4f}",
            #         preprocess_time,
            #         detect_time,
            #         retarget_time,
            #     )
            # except Exception:
            #     # 计时信息仅用于调试分析，任何异常都不应影响主流程
            #     pass

            # # 将关节角放入队列，供进程二使用
            # try:
            #     qpos_queue.put_nowait(qpos)
            # except:
            #     pass  # 队列满了，跳过这一帧，保持实时性

        time.sleep(1 / 30.0)

    cap.release()
    cv2.destroyAllWindows()
    logger.info("进程一：结束")


def process_visualization(
    qpos_queue: multiprocessing.Queue, robot_dir: str, config_path: str
):
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
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    # 创建队列用于传递关节角（从进程一到进程二）
    qpos_queue = multiprocessing.Queue(maxsize=2)  # 只保留最新2个关节角数据

    process_detection_and_retargeting(qpos_queue, str(robot_dir), str(config_path), camera_path)

    # # 进程一：检测和重定向
    # detection_process = multiprocessing.Process(
    #     target=process_detection_and_retargeting,
    #     args=(qpos_queue, str(robot_dir), str(config_path), camera_path),
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

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
