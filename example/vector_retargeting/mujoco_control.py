from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

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
    _last_diag_log_time: float = 0.0

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
            raise ValueError(f"mocap body not found: {body_name}")
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"body {body_name} is not a mocap body")
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
                f"hand qpos too short: need {need} (dummy 6 + fingers {n_finger}), got {q.shape[0]}; skipping frame"
            )
            return
        # Robustly take the trailing finger block.
        # This supports:
        # - dummy + fingers
        # - dummy + wrist + fingers
        # - dummy + arm + wrist + fingers
        # as long as finger joints are the last n_finger entries.
        finger_values = q[-n_finger:]

        for ctrl_idx, value in zip(self.simulation.finger_ctrl_indices, finger_values):
            data.ctrl[int(ctrl_idx)] = value

        now = time.time()
        if now - self._last_diag_log_time > 1.0:
            self._last_diag_log_time = now
            finger_ctrl = np.asarray(
                [data.ctrl[int(i)] for i in self.simulation.finger_ctrl_indices],
                dtype=np.float64,
            )
            finger_qpos = []
            for act_id in self.simulation.finger_ctrl_indices:
                jnt_id = int(self.model.actuator_trnid[int(act_id), 0])
                if jnt_id < 0:
                    continue
                qadr = int(self.model.jnt_qposadr[jnt_id])
                finger_qpos.append(float(data.qpos[qadr]))
            finger_qpos_arr = np.asarray(finger_qpos, dtype=np.float64)
            if finger_qpos_arr.size > 0:
                track_err = float(np.mean(np.abs(finger_ctrl[: finger_qpos_arr.size] - finger_qpos_arr)))
                qpos_part = (
                    f" finger_joint_qpos[min,max]=[{float(np.min(finger_qpos_arr)):.4f},{float(np.max(finger_qpos_arr)):.4f}]"
                    f" mean|ctrl-qpos|={track_err:.4f}"
                )
            else:
                qpos_part = " finger_joint_qpos=n/a"
            # logger.info(
            #     f"[control_diag] hand={hand} qdim={q.shape[0]} "
            #     f"finger_q[min,max]=[{float(np.min(finger_values)):.4f},{float(np.max(finger_values)):.4f}] "
            #     f"ctrl[min,max]=[{float(np.min(finger_ctrl)):.4f},{float(np.max(finger_ctrl)):.4f}]"
            #     f"{qpos_part}"
            # )

        if self.simulation.mocap.wrist_mocap:
            mocap_id = self._resolve_mocap_id()
            if mocap_id is None:
                logger.warning("wrist_mocap=True but mocap id unresolved; skipping wrist output")
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

        # Fallback: if q has no root but wrist quaternion exists, drive root rotation from it
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
