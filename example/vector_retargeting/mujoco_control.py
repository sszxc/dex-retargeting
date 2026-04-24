from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as SciRotation

from runtime_config import SimulationConfig


def _quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quat, dtype=np.float64).reshape(4)
    return SciRotation.from_quat([qx, qy, qz, qw]).as_matrix()


def _rotmat_to_quat_wxyz(rotmat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = SciRotation.from_matrix(rotmat).as_quat()
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def _quat_wxyz_to_euler_zyx(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    rx = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        ry = np.sign(sinp) * (np.pi / 2)
    else:
        ry = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    rz = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([rz, ry, rx], dtype=np.float64)


@dataclass
class MujocoHandController:
    simulation: SimulationConfig
    model: mujoco.MjModel
    _resolved_mocap_id: Optional[int] = None

    def _resolve_mocap_id(self) -> Optional[int]:
        if not self.simulation.mocap.wrist_mocap:
            return None
        if self._resolved_mocap_id is not None:
            return self._resolved_mocap_id
        if self.simulation.mocap.mocap_id is not None:
            self._resolved_mocap_id = int(self.simulation.mocap.mocap_id)
            return self._resolved_mocap_id
        body_name = self.simulation.mocap.mocap_body_name
        if not body_name:
            return None
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id < 0:
            raise ValueError(f"未找到 mocap body: {body_name}")
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"body {body_name} 不是 mocap body")
        self._resolved_mocap_id = mocap_id
        return mocap_id

    def apply(self, data: mujoco.MjData, msg: dict) -> None:
        hand = self.simulation.control_hand
        qpos = msg.get(f"hand_{hand}_qpos")
        wrist_pos = msg.get(f"wrist_{hand}_pos")
        wrist_quat = msg.get(f"wrist_{hand}_quat")
        if qpos is None:
            return

        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        n_finger = len(self.simulation.finger_ctrl_indices)
        need = 6 + n_finger
        if q.shape[0] < need:
            logger.warning(
                f"hand qpos 长度不足: 需要 {need}（dummy 6 + 手指 {n_finger}），实际 {q.shape[0]}，跳过本帧控制"
            )
            return
        finger_values = q[6:need]

        for ctrl_idx, value in zip(self.simulation.finger_ctrl_indices, finger_values):
            data.ctrl[int(ctrl_idx)] = value

        if self.simulation.mocap.wrist_mocap:
            mocap_id = self._resolve_mocap_id()
            if mocap_id is None:
                logger.warning("wrist_mocap=True 但未解析到 mocap id，跳过 wrist 输出")
                return
            pos_off = np.asarray(self.simulation.root_position_offset, dtype=np.float64)
            r_cal = np.asarray(
                self.simulation.wrist_rotation_calib_matrix, dtype=np.float64
            ).reshape(3, 3)
            if wrist_pos is not None:
                data.mocap_pos[mocap_id] = (
                    np.asarray(wrist_pos, dtype=np.float64) + pos_off
                )
            if wrist_quat is not None:
                quat = np.asarray(wrist_quat, dtype=np.float64).reshape(4)
                r_w = _quat_wxyz_to_rotmat(quat)
                data.mocap_quat[mocap_id] = _rotmat_to_quat_wxyz(r_w @ r_cal)
            return

        if q.shape[0] >= 6:
            root = q[:6].copy()
            root[:3] += np.asarray(self.simulation.root_position_offset, dtype=np.float64)
            for local_i, ctrl_idx in enumerate(self.simulation.root_ctrl_indices[:3]):
                data.ctrl[int(ctrl_idx)] = root[local_i]
            for local_i, ctrl_idx in enumerate(self.simulation.root_ctrl_indices[3:6], start=3):
                data.ctrl[int(ctrl_idx)] = root[local_i]
            return

        # 兜底：若 q 不包含 root，但有 wrist quaternion，则用其驱动 root 旋转
        if wrist_quat is not None:
            r_cal = np.asarray(
                self.simulation.wrist_rotation_calib_matrix, dtype=np.float64
            ).reshape(3, 3)
            r_out = r_cal @ _quat_wxyz_to_rotmat(
                np.asarray(wrist_quat, dtype=np.float64).reshape(4)
            )
            euler = _quat_wxyz_to_euler_zyx(_rotmat_to_quat_wxyz(r_out))
            for i, ctrl_idx in enumerate(self.simulation.root_ctrl_indices[3:6]):
                data.ctrl[int(ctrl_idx)] = euler[i]
