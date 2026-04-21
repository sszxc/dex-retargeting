from __future__ import annotations

import multiprocessing
import time
from queue import Full
from typing import Dict, Optional

import numpy as np
from loguru import logger

from dex_retargeting.retargeting_config import RetargetingConfig
from input_sources import (
    HandObservation,
    LeapInputSource,
    WebcamInputSource,
    generate_sine_test_qpos,
)
from runtime_config import RuntimeConfig


def _empty_message() -> Dict[str, Optional[np.ndarray]]:
    return {
        "hand_left_qpos": None,
        "wrist_left_pos": None,
        "wrist_left_quat": None,
        "wrist_left_rot": None,
        "keypoint_left_3d": None,
        "hand_right_qpos": None,
        "wrist_right_pos": None,
        "wrist_right_quat": None,
        "wrist_right_rot": None,
        "keypoint_right_3d": None,
    }


def _rotation_matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _build_ref_value(retargeting, joint_pos: np.ndarray) -> np.ndarray:
    optimizer = retargeting.optimizer
    retargeting_type = str(optimizer.retargeting_type).upper()
    indices = optimizer.target_link_human_indices
    if retargeting_type == "POSITION":
        return joint_pos[indices, :]
    if retargeting_type == "JOINT":
        return joint_pos
    joint_pos_relative = joint_pos - joint_pos[0, :]
    origin_indices = indices[0, :]
    task_indices = indices[1, :]
    return (
        joint_pos_relative[task_indices, :]
        - joint_pos_relative[origin_indices, :]
    )


def _build_source(cfg: RuntimeConfig, active_hands):
    src = cfg.sensor.input_source
    camera2table = np.asarray(cfg.sensor.camera2table, dtype=np.float64)
    if src == "webcam":
        return WebcamInputSource(
            active_hands=active_hands,
            webcam_index=cfg.sensor.webcam_index,
            camera2table=camera2table,
        )
    if src == "leap_motion":
        return LeapInputSource(active_hands=active_hands, camera2table=camera2table)
    return None


def run_retarget_worker(
    qpos_queue: multiprocessing.Queue,
    runtime_cfg: RuntimeConfig,
    robot_urdf_dir: str,
) -> None:
    active_hands = runtime_cfg.retargeting.active_hands()
    RetargetingConfig.set_default_urdf_dir(robot_urdf_dir)
    retargeters = {}
    for hand in active_hands:
        hand_cfg = runtime_cfg.retargeting.build_hand_config(hand)
        retargeters[hand] = RetargetingConfig.from_dict(
            hand_cfg.to_retargeting_dict()
        ).build()
        logger.info(
            f"初始化 retargeting hand={hand}, optimizer={hand_cfg.optimizer_type}, urdf={hand_cfg.urdf_path}"
        )

    if runtime_cfg.sensor.input_source == "test_sine":
        logger.info("输入源: test_sine")
        total_dim = 22
        dim_duration = 3.0
        dim_idx = 0
        dim_start = time.time()
        while True:
            msg = _empty_message()
            elapsed = time.time() - dim_start
            progress = min(max(elapsed / dim_duration, 0.0), 1.0)
            qpos = generate_sine_test_qpos(dim_idx=dim_idx, progress=progress, total_dim=total_dim)
            for hand in active_hands:
                msg[f"hand_{hand}_qpos"] = qpos
            try:
                qpos_queue.put_nowait(msg)
            except Full:
                pass
            if elapsed >= dim_duration:
                dim_idx = (dim_idx + 1) % total_dim
                dim_start = time.time()
            time.sleep(1.0 / 30.0)

    source = _build_source(runtime_cfg, active_hands)
    if source is None:
        raise ValueError(f"不支持的输入源: {runtime_cfg.sensor.input_source}")

    try:
        while True:
            obs_map: Dict[str, HandObservation] = source.poll()
            msg = _empty_message()
            for hand in active_hands:
                obs = obs_map.get(hand, HandObservation(None, None, None))
                if obs.joint_pos is None:
                    continue
                ref_value = _build_ref_value(retargeters[hand], obs.joint_pos)
                robot_qpos = retargeters[hand].retarget(ref_value)
                msg[f"hand_{hand}_qpos"] = robot_qpos
                if obs.keypoint_3d is not None:
                    msg[f"wrist_{hand}_pos"] = np.asarray(obs.keypoint_3d[0], dtype=np.float64)
                    msg[f"keypoint_{hand}_3d"] = np.asarray(obs.keypoint_3d, dtype=np.float64)
                if obs.wrist_rot is not None:
                    msg[f"wrist_{hand}_quat"] = _rotation_matrix_to_quat_wxyz(obs.wrist_rot)
                    msg[f"wrist_{hand}_rot"] = np.asarray(obs.wrist_rot, dtype=np.float64)
            try:
                qpos_queue.put_nowait(msg)
            except Full:
                pass
            time.sleep(1.0 / 30.0)
    finally:
        source.close()
