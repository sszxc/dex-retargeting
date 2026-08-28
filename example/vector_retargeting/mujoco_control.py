from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
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


def _euler_xyz_intrinsic_to_rotmat(euler: np.ndarray) -> np.ndarray:
    """Inverse of optimizer.py's `Rotation.from_matrix(...).as_euler("XYZ")` — the
    convention the retargeting solver uses for the dummy free joint's root[3:6]
    (matches the serial x->y->z revolute chain built by add_dummy_free_joints)."""
    return SciRotation.from_euler("XYZ", np.asarray(euler, dtype=np.float64).reshape(3)).as_matrix()


def _rotmat_to_euler_xyz_intrinsic(rotmat: np.ndarray) -> np.ndarray:
    return SciRotation.from_matrix(rotmat).as_euler("XYZ")


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


def read_finger_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, ctrl_indices: List[int]
) -> np.ndarray:
    """Read the actual, physically-simulated joint angle (data.qpos) driven by each
    actuator index, in ctrl_indices order. Same actuator->joint lookup pattern as the
    control-tracking diagnostic below, factored out for reuse (e.g. state publishing)."""
    values = []
    for act_idx in ctrl_indices:
        jnt_id = int(model.actuator_trnid[int(act_idx), 0])
        qadr = int(model.jnt_qposadr[jnt_id])
        values.append(float(data.qpos[qadr]))
    return np.asarray(values, dtype=np.float64)


def _project_fingertip_translation(
    requested_pos: np.ndarray,
    fingertip_offsets: np.ndarray,
    fingertips_pos_min: np.ndarray,
    fingertips_pos_max: np.ndarray,
    wrist_pos_min: Optional[np.ndarray] = None,
    wrist_pos_max: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Project a wrist target onto the translations that keep every tip in bounds."""
    requested = np.asarray(requested_pos, dtype=np.float64).reshape(3)
    offsets = np.asarray(fingertip_offsets, dtype=np.float64).reshape(-1, 3)
    tip_lo = np.asarray(fingertips_pos_min, dtype=np.float64).reshape(3)
    tip_hi = np.asarray(fingertips_pos_max, dtype=np.float64).reshape(3)
    if (
        offsets.shape[0] == 0
        or not np.all(np.isfinite(requested))
        or not np.all(np.isfinite(offsets))
    ):
        return None

    # For every tip i, tip_lo <= wrist + offset_i <= tip_hi. Intersect all
    # per-tip intervals, then intersect the result with the optional wrist AABB.
    allowed_lo = np.max(tip_lo[None, :] - offsets, axis=0)
    allowed_hi = np.min(tip_hi[None, :] - offsets, axis=0)
    if wrist_pos_min is not None:
        allowed_lo = np.maximum(
            allowed_lo, np.asarray(wrist_pos_min, dtype=np.float64).reshape(3)
        )
    if wrist_pos_max is not None:
        allowed_hi = np.minimum(
            allowed_hi, np.asarray(wrist_pos_max, dtype=np.float64).reshape(3)
        )
    if not np.all(np.isfinite(allowed_lo)) or not np.all(np.isfinite(allowed_hi)):
        return None
    if np.any(allowed_lo > allowed_hi):
        return None
    return np.clip(requested, allowed_lo, allowed_hi)


@dataclass
class MujocoHandController:
    simulation: SimulationConfig
    model: mujoco.MjModel
    _resolved_mocap_id: Optional[int] = None
    _last_diag_log_time: float = 0.0
    _last_safety_warning_time: float = 0.0
    _fingertip_body_ids: Optional[np.ndarray] = field(
        default=None, init=False, repr=False
    )
    _weld_partner_body_id: Optional[int] = field(
        default=None, init=False, repr=False
    )
    _finger_qpos_addresses: Optional[np.ndarray] = field(
        default=None, init=False, repr=False
    )
    _prediction_data: Optional[mujoco.MjData] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.simulation.fingertip_body_names is None:
            return
        if (
            self.simulation.fingertips_pos_min is None
            or self.simulation.fingertips_pos_max is None
        ):
            raise ValueError(
                "fingertip_body_names requires fingertips_pos_min and "
                "fingertips_pos_max"
            )
        if not self.simulation.mocap.wrist_mocap:
            raise ValueError("fingertip bounds require wrist_mocap=True")

        mocap_id = self._resolve_mocap_id()
        if mocap_id is None:
            raise ValueError("fingertip bounds require a resolvable mocap body")
        mocap_body_matches = np.flatnonzero(
            np.asarray(self.model.body_mocapid, dtype=np.int64) == mocap_id
        )
        if mocap_body_matches.size != 1:
            raise ValueError(
                f"could not uniquely resolve body for mocap id {mocap_id}"
            )
        mocap_body_id = int(mocap_body_matches[0])

        for eq_id in range(self.model.neq):
            if self.model.eq_type[eq_id] != mujoco.mjtEq.mjEQ_WELD:
                continue
            obj1 = int(self.model.eq_obj1id[eq_id])
            obj2 = int(self.model.eq_obj2id[eq_id])
            if obj1 == mocap_body_id:
                self._weld_partner_body_id = obj2
                break
            if obj2 == mocap_body_id:
                self._weld_partner_body_id = obj1
                break
        if self._weld_partner_body_id is None:
            mocap_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_id
            )
            raise ValueError(
                f"fingertip bounds require mocap body '{mocap_name}' to have an "
                "equality weld partner"
            )

        body_ids = []
        missing_names = []
        for body_name in self.simulation.fingertip_body_names:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                missing_names.append(body_name)
            else:
                body_ids.append(body_id)
        if missing_names:
            raise ValueError(
                "configured fingertip bodies not found in MuJoCo model: "
                + ", ".join(missing_names)
            )
        self._fingertip_body_ids = np.asarray(body_ids, dtype=np.int64)

        qpos_addresses = []
        for actuator_id in self.simulation.finger_ctrl_indices:
            if actuator_id < 0 or actuator_id >= self.model.nu:
                raise ValueError(
                    f"finger actuator index {actuator_id} is outside model "
                    "actuator range"
                )
            joint_id = int(self.model.actuator_trnid[int(actuator_id), 0])
            if joint_id < 0:
                raise ValueError(
                    f"finger actuator {actuator_id} is not attached to a joint"
                )
            joint_type = self.model.jnt_type[joint_id]
            if joint_type not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError(
                    f"finger actuator {actuator_id} must drive a scalar hinge or "
                    "slide joint"
                )
            qpos_addresses.append(int(self.model.jnt_qposadr[joint_id]))
        self._finger_qpos_addresses = np.asarray(qpos_addresses, dtype=np.int64)
        self._prediction_data = mujoco.MjData(self.model)

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

    def _clip_wrist_pos(self, pos: np.ndarray) -> np.ndarray:
        lo = self.simulation.wrist_pos_min
        hi = self.simulation.wrist_pos_max
        if lo is None or hi is None:
            return pos
        return np.clip(
            np.asarray(pos, dtype=np.float64).reshape(3),
            np.asarray(lo, dtype=np.float64).reshape(3),
            np.asarray(hi, dtype=np.float64).reshape(3),
        )

    def _predict_fingertip_offsets(
        self,
        data: mujoco.MjData,
        finger_values: np.ndarray,
        desired_wrist_rotation: np.ndarray,
    ) -> np.ndarray:
        """Return commanded fingertip origins relative to the desired wrist target."""
        if (
            self._prediction_data is None
            or self._finger_qpos_addresses is None
            or self._fingertip_body_ids is None
            or self._weld_partner_body_id is None
        ):
            raise RuntimeError("fingertip bounds were not initialized")

        prediction = self._prediction_data
        prediction.qpos[:] = data.qpos
        if self.model.nmocap:
            prediction.mocap_pos[:] = data.mocap_pos
            prediction.mocap_quat[:] = data.mocap_quat
        prediction.qpos[self._finger_qpos_addresses] = finger_values
        mujoco.mj_forward(self.model, prediction)

        wrist_pos = np.asarray(
            prediction.xpos[self._weld_partner_body_id], dtype=np.float64
        ).reshape(3)
        wrist_rotation = np.asarray(
            prediction.xmat[self._weld_partner_body_id], dtype=np.float64
        ).reshape(3, 3)
        fingertip_pos = np.asarray(
            prediction.xpos[self._fingertip_body_ids], dtype=np.float64
        ).reshape(-1, 3)
        local_offsets = (fingertip_pos - wrist_pos[None, :]) @ wrist_rotation
        desired_rotation = np.asarray(
            desired_wrist_rotation, dtype=np.float64
        ).reshape(3, 3)
        return local_offsets @ desired_rotation.T

    def _constrain_mocap_translation(
        self,
        data: mujoco.MjData,
        requested_pos: np.ndarray,
        finger_values: np.ndarray,
        desired_wrist_rotation: np.ndarray,
    ) -> Optional[np.ndarray]:
        offsets = self._predict_fingertip_offsets(
            data, finger_values, desired_wrist_rotation
        )
        wrist_lo = self.simulation.wrist_pos_min
        wrist_hi = self.simulation.wrist_pos_max
        return _project_fingertip_translation(
            requested_pos=requested_pos,
            fingertip_offsets=offsets,
            fingertips_pos_min=np.asarray(
                self.simulation.fingertips_pos_min, dtype=np.float64
            ),
            fingertips_pos_max=np.asarray(
                self.simulation.fingertips_pos_max, dtype=np.float64
            ),
            wrist_pos_min=(
                None if wrist_lo is None else np.asarray(wrist_lo, dtype=np.float64)
            ),
            wrist_pos_max=(
                None if wrist_hi is None else np.asarray(wrist_hi, dtype=np.float64)
            ),
        )

    def _warn_infeasible_fingertip_command(self) -> None:
        now = time.time()
        if now - self._last_safety_warning_time < 1.0:
            return
        self._last_safety_warning_time = now
        logger.warning(
            "Rejected hand command: no wrist translation can satisfy both the "
            "configured fingertip bounds and wrist bounds for this finger pose."
        )

    def _commit_finger_controls(self, data: mujoco.MjData, values: np.ndarray) -> None:
        for ctrl_idx, value in zip(self.simulation.finger_ctrl_indices, values):
            data.ctrl[int(ctrl_idx)] = value

    def _run_finger_diagnostics(self, data: mujoco.MjData) -> None:
        now = time.time()
        if now - self._last_diag_log_time <= 1.0:
            return
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
            # Kept as a cheap periodic calculation for the existing commented-out
            # control diagnostic below.
            _ = float(
                np.mean(
                    np.abs(
                        finger_ctrl[: finger_qpos_arr.size] - finger_qpos_arr
                    )
                )
            )

    def _engage_translation_tracking_mocap(
        self,
        data: mujoco.MjData,
        mocap_id: int,
        raw_pos: Optional[np.ndarray],
    ) -> None:
        """Mocap-path equivalent of _engage_translation_tracking: recompute root_position_offset
        (translation only) from data.mocap_pos (the current commanded target) vs. this
        frame's raw camera wrist position, so translation tracking resumes with zero
        jump. Rotation is never engaged here — it always follows the camera hand via the
        static wrist_rotation_calib_matrix (parsed from simulation.wrist_rotation_offset_rpy),
        independent of the track/engage state."""
        if raw_pos is None:
            return
        current_pos = np.asarray(data.mocap_pos[mocap_id], dtype=np.float64).reshape(3)
        new_offset = current_pos - np.asarray(raw_pos, dtype=np.float64).reshape(3)
        for i in range(3):
            self.simulation.root_position_offset[i] = float(new_offset[i])

    def _engage_translation_tracking(self, data: mujoco.MjData, raw_root: np.ndarray) -> None:
        """Recompute root_position_offset (translation only) so translation tracking
        resumes from the wrist's current commanded position (data.ctrl) with zero jump.
        Rotation is never engaged here — it always follows the camera hand via the static
        wrist_rotation_calib_matrix, independent of the track/engage state. Called once on
        the rising edge of a wrist-tracking toggle-on (track key press)."""
        idx = self.simulation.root_ctrl_indices
        current_pos = np.array([data.ctrl[int(i)] for i in idx[:3]], dtype=np.float64)
        raw_pos = raw_root[:3]
        new_offset = current_pos - raw_pos
        for i in range(3):
            self.simulation.root_position_offset[i] = float(new_offset[i])

    def apply(
        self,
        data: mujoco.MjData,
        msg: dict,
        track_translation: bool = True,
        engage_translation: bool = False,
    ) -> None:
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

        if self.simulation.mocap.wrist_mocap:
            mocap_id = self._resolve_mocap_id()
            if mocap_id is None:
                self._commit_finger_controls(data, finger_values)
                self._run_finger_diagnostics(data)
                logger.warning("wrist_mocap=True but mocap id unresolved; skipping wrist output")
                return

            # Rotation always follows the camera hand via the static calibration from
            # config, regardless of track_translation — the 't' key only gates translation.
            desired_quat = np.asarray(
                data.mocap_quat[mocap_id], dtype=np.float64
            ).reshape(4).copy()
            if wrist_quat is not None:
                r_cal = np.asarray(
                    self.simulation.wrist_rotation_calib_matrix, dtype=np.float64
                ).reshape(3, 3)
                quat = np.asarray(wrist_quat, dtype=np.float64).reshape(4)
                r_w = _quat_wxyz_to_rotmat(quat)
                desired_quat = _rotmat_to_quat_wxyz(r_w @ r_cal)

            current_pos = np.asarray(
                data.mocap_pos[mocap_id], dtype=np.float64
            ).reshape(3).copy()
            requested_pos = current_pos
            if track_translation and engage_translation:
                self._engage_translation_tracking_mocap(data, mocap_id, wrist_pos)
            if track_translation and wrist_pos is not None:
                pos_off = np.asarray(
                    self.simulation.root_position_offset, dtype=np.float64
                )
                requested_pos = np.asarray(
                    wrist_pos, dtype=np.float64
                ).reshape(3) + pos_off

            if self._fingertip_body_ids is not None:
                constrained_pos = self._constrain_mocap_translation(
                    data=data,
                    requested_pos=requested_pos,
                    finger_values=finger_values,
                    desired_wrist_rotation=_quat_wxyz_to_rotmat(desired_quat),
                )
                if constrained_pos is None:
                    self._warn_infeasible_fingertip_command()
                    return
                requested_pos = constrained_pos
            elif track_translation and wrist_pos is not None:
                requested_pos = self._clip_wrist_pos(requested_pos)

            # Safety feasibility is established before any live command is changed.
            self._commit_finger_controls(data, finger_values)
            data.mocap_quat[mocap_id] = desired_quat
            data.mocap_pos[mocap_id] = requested_pos
            self._run_finger_diagnostics(data)
            return

        self._commit_finger_controls(data, finger_values)
        self._run_finger_diagnostics(data)
        if q.shape[0] >= 6:
            root = q[:6].copy()

            # Rotation always follows the camera hand via the static calibration from
            # config, regardless of track_translation — the 't' key only gates translation.
            r_cal = np.asarray(
                self.simulation.wrist_rotation_calib_matrix, dtype=np.float64
            ).reshape(3, 3)
            r_out = _euler_xyz_intrinsic_to_rotmat(root[3:6]) @ r_cal
            euler_out = _rotmat_to_euler_xyz_intrinsic(r_out)
            for local_i, ctrl_idx in enumerate(self.simulation.root_ctrl_indices[3:6]):
                data.ctrl[int(ctrl_idx)] = euler_out[local_i]

            if not track_translation:
                # Translation tracking toggled off: leave root position ctrl untouched
                # so the wrist freezes in place. Fingers above still update every frame.
                return
            if engage_translation:
                self._engage_translation_tracking(data, root)

            pos = self._clip_wrist_pos(
                root[:3] + np.asarray(self.simulation.root_position_offset, dtype=np.float64)
            )
            for local_i, ctrl_idx in enumerate(self.simulation.root_ctrl_indices[:3]):
                data.ctrl[int(ctrl_idx)] = pos[local_i]
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
