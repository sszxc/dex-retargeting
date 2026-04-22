from __future__ import annotations

import multiprocessing
import sys
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import h5py
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from loguru import logger
from pynput import keyboard

from mujoco_control import MujocoHandController
from retarget_worker import run_retarget_worker
from runtime_config import RuntimeConfig, load_runtime_config


DEFAULT_LOOKAT = [0.0, 0.0, 0.2]
DEFAULT_CAMERA_PARAMS = {"distance": 0.8, "elevation": -30, "azimuth": 0}

def _repo_root() -> Path:
    return Path(__file__).absolute().parent.parent.parent


def _ensure_src_on_syspath() -> None:
    src = _repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _hand_connections_21():
    # MediaPipe Hands connections, but kept as a fallback constant to avoid
    # hard dependency on mediapipe in the mujoco demo.
    try:
        import mediapipe as mp  # type: ignore

        return list(mp.solutions.hands.HAND_CONNECTIONS)
    except Exception:
        return [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (5, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (9, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (13, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (0, 17),
        ]


def _make_rerun_board(enabled: bool):
    if not enabled:
        return None, None

    _ensure_src_on_syspath()
    try:
        from utils.misc_utils import DummyClass  # type: ignore
    except Exception:
        DummyClass = None  # type: ignore

    try:
        import rerun as rr  # type: ignore
        from utils.rerun_board import RerunBoard  # type: ignore

        board = RerunBoard(
            f"DexRetargeting_{time.strftime('%m_%d_%H_%M', time.localtime())}",
            template="dex_retargeting",
        )
        logger.info("RerunBoard 已启用")
        return board, rr
    except Exception as err:
        logger.warning(f"RerunBoard 启用失败（将继续运行）：{err}")
        return (DummyClass() if DummyClass is not None else None), None


def _make_dynamic_camera(par: dict) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    try:
        mujoco.mjv_defaultCamera(camera)
    except AttributeError:
        pass
    camera.lookat[:] = DEFAULT_LOOKAT
    camera.distance = par["distance"]
    camera.elevation = par["elevation"]
    camera.azimuth = par["azimuth"]
    return camera


def _resolve_xml_path(mj_xml_path: str | Path) -> Path:
    p = Path(mj_xml_path).expanduser().resolve()
    if p.is_dir():
        first_xml = next(p.glob("*.xml"), None)
        if first_xml is None:
            raise FileNotFoundError(f"目录下未找到 .xml 文件: {p}")
        return first_xml
    if not p.exists():
        raise FileNotFoundError(f"Mujoco 场景文件不存在: {p}")
    return p


def _setup_recording_cameras(
    model: mujoco.MjModel, camera_names: list[str]
) -> tuple[list[tuple[str, str | mujoco.MjvCamera]], list[str]]:
    if model.ncam > 0:
        xml_camera_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)
        ]
        if not camera_names:
            names_to_use = xml_camera_names
        else:
            names_to_use = [n for n in camera_names if n in xml_camera_names]
            if len(names_to_use) < len(camera_names):
                unknown = [n for n in camera_names if n not in xml_camera_names]
                logger.warning(f"以下相机名在 XML 中不存在，已忽略: {unknown}")
        specs = [(name, name) for name in names_to_use]
        return specs, list(names_to_use)

    default_camera = _make_dynamic_camera(DEFAULT_CAMERA_PARAMS)
    logger.info("场景 XML 中未定义相机，使用内置默认相机 default")
    return [("default", default_camera)], ["default"]


def main(
    runtime_config_path: str,
    dataset_dir: str = "data/hdf5",
    start_key: str = "s",
    stop_key: str = "e",
):
    runtime_cfg: RuntimeConfig = load_runtime_config(runtime_config_path)
    logger.info(
        f"加载配置成功: input_source={runtime_cfg.sensor.input_source}, mode={runtime_cfg.retargeting.mode}"
    )
    board, rr = _make_rerun_board(runtime_cfg.sensor.rerun_enabled)
    hand_connections = _hand_connections_21()

    robot_dir = (
        _repo_root() / "assets" / "robots" / "hands"
    )
    qpos_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=4)

    mj_xml_path = _resolve_xml_path(runtime_cfg.simulation.mj_xml_path)
    model = mujoco.MjModel.from_xml_path(str(mj_xml_path))
    data = mujoco.MjData(model)

    # 可选：启动后立即加载指定 keyframe（MJCF <keyframe name="...">）
    if runtime_cfg.simulation.startup_keyframe:
        kf_name = runtime_cfg.simulation.startup_keyframe
        kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, kf_name)
        if kf_id < 0:
            logger.warning(
                f"startup_keyframe='{kf_name}' 未在模型中找到（请检查 XML 里的 <keyframe> name），将忽略。"
            )
        else:
            mujoco.mj_resetDataKeyframe(model, data, kf_id)
            mujoco.mj_forward(model, data)
            logger.info(f"已加载 startup_keyframe='{kf_name}' (id={kf_id})")

    controller = MujocoHandController(simulation=runtime_cfg.simulation, model=model)

    recording_camera_specs, recording_camera_names = _setup_recording_cameras(
        model, runtime_cfg.simulation.camera_names
    )
    renderer = mujoco.Renderer(model, width=640, height=480)

    robots_for_vis = {}
    joint_connections_for_vis = {}
    if rr is not None and board is not None:
        try:
            from dex_retargeting.retargeting_config import RetargetingConfig

            RetargetingConfig.set_default_urdf_dir(str(robot_dir))
            for hand in runtime_cfg.retargeting.active_hands():
                hand_cfg = runtime_cfg.retargeting.build_hand_config(hand)
                retargeting = RetargetingConfig.from_dict(
                    hand_cfg.to_retargeting_dict()
                ).build()
                robot = retargeting.optimizer.robot
                robots_for_vis[hand] = robot
                joint_connections_for_vis[hand] = robot.get_joint_connections()
        except Exception as err:
            logger.warning(f"初始化 robot 可视化失败（将跳过 joint 3D 可视化）：{err}")

    worker = multiprocessing.Process(
        target=run_retarget_worker,
        args=(qpos_queue, runtime_cfg, str(robot_dir)),
    )
    worker.start()
    logger.info("检测与重定向子进程已启动")

    keyboard_lock = threading.Lock()
    should_exit = False
    record_start_requested = False
    record_stop_requested = False
    is_recording = False
    episode_idx = 0
    episode_buffers: Optional[dict[str, list]] = None

    dataset_root = Path(dataset_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dataset_dir_path = dataset_root / timestamp
    dataset_dir_path.mkdir(parents=True, exist_ok=True)
    (dataset_dir_path / "command.txt").write_text(
        " ".join(sys.argv) + "\n", encoding="utf-8"
    )

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

        qpos_array = np.stack(buffers["/observations/qpos"], axis=0)
        qvel_array = np.stack(buffers["/observations/qvel"], axis=0)
        action_array = np.stack(buffers["/action"], axis=0)

        max_timesteps = min(
            qpos_array.shape[0], qvel_array.shape[0], action_array.shape[0]
        )
        qpos_array = qpos_array[:max_timesteps]
        qvel_array = qvel_array[:max_timesteps]
        action_array = action_array[:max_timesteps]

        images_arrays: dict[str, np.ndarray] = {}
        for cam in recording_camera_names:
            key = f"/observations/images/{cam}"
            cam_list = buffers.get(key, [])
            if cam_list:
                images_arrays[cam] = np.stack(cam_list, axis=0)[:max_timesteps]

        dataset_path = dataset_dir_path / f"episode_{idx}.hdf5"
        with h5py.File(str(dataset_path), "w", rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs["sim"] = True
            root.attrs["mj_xml_path"] = str(mj_xml_path)
            root.attrs["scene"] = mj_xml_path.stem

            obs = root.create_group("observations")
            image_group = obs.create_group("images")
            for cam, img_array in images_arrays.items():
                h, w = img_array.shape[1:3]
                dset = image_group.create_dataset(
                    cam, data=img_array, dtype="uint8", chunks=(1, h, w, 3)
                )
                dset.attrs["CLASS"] = np.bytes_("IMAGE")

            obs.create_dataset("qpos", data=qpos_array)
            obs.create_dataset("qvel", data=qvel_array)
            root.create_dataset("action", data=action_array)
        logger.info(f"Episode {idx} 已保存到 {dataset_path}, steps={max_timesteps}")

    def on_press(key):
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

    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()
    logger.info(
        f"键盘监听已启动：按 '{start_key}' 开始记录，按 '{stop_key}' 结束保存，按 'q' 退出"
    )

    latest_msg = None
    control_interval = 1.0 / runtime_cfg.simulation.control_rate_hz
    last_control_time = time.time()

    viewer = mujoco.viewer.launch_passive(model, data)
    try:
        logger.info("Mujoco viewer 已启动")
        options = viewer.opt
        mujoco.mjv_defaultOption(options)
        options.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        options.label = mujoco.mjtLabel.mjLABEL_CAMERA
        sim_start = time.time()

        while viewer.is_running() and not should_exit:
            now = time.time()

            while True:
                try:
                    msg = qpos_queue.get_nowait()
                except Empty:
                    break
                latest_msg = msg

            if latest_msg is not None and now - last_control_time >= control_interval:
                controller.apply(data, latest_msg)

                if rr is not None and board is not None:
                    try:
                        for hand in runtime_cfg.retargeting.active_hands():
                            keypoint_3d = latest_msg.get(f"keypoint_{hand}_3d")
                            wrist_rot = latest_msg.get(f"wrist_{hand}_rot")
                            if keypoint_3d is not None:
                                keypoint_3d = np.asarray(keypoint_3d, dtype=np.float64)
                                for i in range(keypoint_3d.shape[0]):
                                    board.log(
                                        f"world/human/{hand}/keypoint/{i}",
                                        rr.Points3D(
                                            positions=[keypoint_3d[i]],
                                            colors=[[255, 0, 0]],
                                            radii=0.005,
                                            labels=f"{i}",
                                        ),
                                    )
                                for pair in hand_connections:
                                    board.log(
                                        f"world/human/{hand}/connection/{pair}",
                                        rr.Arrows3D(
                                            origins=[keypoint_3d[pair[0]]],
                                            vectors=[
                                                keypoint_3d[pair[1]] - keypoint_3d[pair[0]]
                                            ],
                                            colors=[[0, 255, 0]],
                                        ),
                                    )
                            if keypoint_3d is not None and wrist_rot is not None:
                                board.log_axes(
                                    translation=keypoint_3d[0],
                                    rotation=np.asarray(wrist_rot, dtype=np.float64),
                                    root=f"world/human/{hand}",
                                    name="wrist_rot",
                                    axis_size=0.05,
                                )

                            robot_qpos = latest_msg.get(f"hand_{hand}_qpos")
                            if robot_qpos is not None:
                                robot_qpos = np.asarray(robot_qpos)
                                for i in range(robot_qpos.shape[0]):
                                    board.log(
                                        f"joint_angles/{hand}/joint_{i:02d}",
                                        rr.Scalars(robot_qpos[i]),
                                    )

                                robot = robots_for_vis.get(hand)
                                if robot is not None:
                                    joint_positions = robot.get_all_joint_positions(robot_qpos)
                                    for i in range(joint_positions.shape[0]):
                                        board.log(
                                            f"world/robot/{hand}/joint/joint_{i}",
                                            rr.Points3D(
                                                positions=[joint_positions[i]],
                                                colors=[[0, 0, 255]],
                                                radii=0.005,
                                            ),
                                        )
                                    for pair in joint_connections_for_vis.get(hand, []):
                                        if (
                                            pair[0] < joint_positions.shape[0]
                                            and pair[1] < joint_positions.shape[0]
                                        ):
                                            board.log(
                                                f"world/robot/{hand}/link/{pair}",
                                                rr.Arrows3D(
                                                    origins=[joint_positions[pair[0]]],
                                                    vectors=[
                                                        joint_positions[pair[1]]
                                                        - joint_positions[pair[0]]
                                                    ],
                                                    colors=[[255, 255, 0]],
                                                ),
                                            )
                    except Exception as err:
                        logger.debug(f"Rerun log 失败（已忽略）：{err}")

                if is_recording and episode_buffers is not None:
                    joint_indices = runtime_cfg.simulation.joint_indices
                    if joint_indices is None:
                        qpos_sample = np.asarray(data.qpos).copy()
                        qvel_sample = np.asarray(data.qvel).copy()
                    else:
                        qpos_sample = np.asarray(data.qpos)[joint_indices].copy()
                        qvel_sample = np.asarray(data.qvel)[joint_indices].copy()

                    episode_buffers["/observations/qpos"].append(qpos_sample)
                    episode_buffers["/observations/qvel"].append(qvel_sample)
                    episode_buffers["/action"].append(np.asarray(data.ctrl).copy())

                    for cam_name, cam_spec in recording_camera_specs:
                        try:
                            renderer.update_scene(data, camera=cam_spec)
                            img = renderer.render()
                            if img.dtype != np.uint8:
                                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                            episode_buffers[f"/observations/images/{cam_name}"].append(img)
                        except Exception as err:
                            logger.warning(f"渲染相机 {cam_name} 失败: {err}")
                last_control_time = now

            while data.time < now - sim_start:
                mujoco.mj_step(model, data)

            pending_buffers = None
            pending_episode_idx = None
            with keyboard_lock:
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

            viewer.sync()
    except KeyboardInterrupt:
        logger.info("收到 Ctrl-C，准备退出并清理资源")
    finally:
        # 保证无论如何退出，都能正确清理
        try:
            keyboard_listener.stop()
            keyboard_listener.join()
        except Exception:
            pass

        try:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        except Exception:
            pass

        try:
            viewer.close()
        except Exception:
            pass

        # 某些环境下 Ctrl-C 会导致 GLFW 清理顺序异常，额外 terminate 一次可避免警告反复出现
        try:
            import glfw  # type: ignore

            glfw.terminate()
        except Exception:
            pass

        logger.info("已退出")


if __name__ == "__main__":
    tyro.cli(main)
