from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


VALID_INPUT_SOURCES = {"webcam", "leap_motion", "test_sine"}
VALID_MODES = {"single_left", "single_right", "bimanual"}
VALID_OPTIMIZERS = {"vector", "position", "dexpilot", "joint", "dex"}


def _identity_3x3() -> List[List[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _rotation_matrix_fixed_zyx(rz: float, ry: float, rx: float) -> np.ndarray:
    """与 mujoco_control 一致：固定轴 R = Rz(rz) @ Ry(ry) @ Rx(rx)。"""
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz_m @ ry_m @ rx_m


def _default_camera2table() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]


@dataclass
class SensorConfig:
    input_source: str = "webcam"
    webcam_index: int = 0
    camera2table: List[List[float]] = field(default_factory=_default_camera2table)
    rerun_enabled: bool = False

    def validate(self) -> None:
        if self.input_source not in VALID_INPUT_SOURCES:
            raise ValueError(
                f"sensor.input_source 必须为 {sorted(VALID_INPUT_SOURCES)}"
            )
        mat = np.asarray(self.camera2table, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError("sensor.camera2table 必须是 3x3 矩阵")


@dataclass
class HandRetargetingConfig:
    hand: str
    urdf_path: str
    add_dummy_free_joint: bool
    optimizer_type: str
    optimizer_params: Dict[str, Any]
    passthrough: Dict[str, Any]

    def to_retargeting_dict(self) -> Dict[str, Any]:
        cfg = {
            "type": self.optimizer_type,
            "urdf_path": self.urdf_path,
            "add_dummy_free_joint": self.add_dummy_free_joint,
        }
        cfg.update(self.optimizer_params)
        cfg.update(self.passthrough)
        return cfg


@dataclass
class RetargetingConfigRuntime:
    mode: str
    left: Dict[str, Any]
    right: Dict[str, Any]

    def active_hands(self) -> List[str]:
        if self.mode == "single_left":
            return ["left"]
        if self.mode == "single_right":
            return ["right"]
        return ["left", "right"]

    def build_hand_config(self, hand: str) -> HandRetargetingConfig:
        if hand not in ("left", "right"):
            raise ValueError(f"未知 hand: {hand}")
        merged = dict(self.left if hand == "left" else self.right)

        if "urdf_path" not in merged:
            raise ValueError(f"retargeting.{hand}.urdf_path 缺失")
        if "optimizer" not in merged:
            raise ValueError(f"retargeting.{hand}.optimizer 缺失")

        optimizer = merged.pop("optimizer")
        if isinstance(optimizer, str):
            optimizer_type = optimizer.lower()
            optimizer_params: Dict[str, Any] = {}
        elif isinstance(optimizer, dict):
            optimizer_type = str(optimizer.get("type", "")).lower()
            optimizer_params = dict(optimizer.get("params", {}))
            for key, value in optimizer.items():
                if key not in {"type", "params"}:
                    optimizer_params[key] = value
        else:
            raise ValueError(f"retargeting.{hand}.optimizer 必须是 string 或 dict")

        if optimizer_type not in VALID_OPTIMIZERS:
            raise ValueError(
                f"retargeting.{hand}.optimizer.type 必须为 {sorted(VALID_OPTIMIZERS)}"
            )
        if optimizer_type == "dex":
            optimizer_type = "dexpilot"

        urdf_path = str(merged.pop("urdf_path"))
        add_dummy_free_joint = bool(merged.pop("add_dummy_free_joint", False))
        return HandRetargetingConfig(
            hand=hand,
            urdf_path=urdf_path,
            add_dummy_free_joint=add_dummy_free_joint,
            optimizer_type=optimizer_type,
            optimizer_params=optimizer_params,
            passthrough=merged,
        )

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"retargeting.mode 必须为 {sorted(VALID_MODES)}")
        for hand in self.active_hands():
            self.build_hand_config(hand)


@dataclass
class MocapConfig:
    wrist_mocap: bool = False
    mocap_body_name: Optional[str] = None
    mocap_id: Optional[int] = None


@dataclass
class SimulationConfig:
    mj_xml_path: str
    # 若提供，则在 Mujoco 启动后立即加载该 keyframe（MJCF 中 <keyframe> 的 name）
    startup_keyframe: Optional[str] = None
    control_hand: str = "left"
    root_ctrl_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    finger_ctrl_indices: List[int] = field(
        default_factory=lambda: [14, 15, 16, 17, 18, 19, 20, 21, 10, 11, 12, 13, 6, 7, 8, 9]
    )
    root_position_offset: List[float] = field(default_factory=lambda: [0.2, 0.0, -0.6])
    # 左手腕旋转标定：R_out = wrist_rotation_calib_matrix @ R_wrist（与 detector 输出的旋转矩阵同约定）
    wrist_rotation_calib_matrix: List[List[float]] = field(default_factory=_identity_3x3)
    joint_indices: Optional[List[int]] = field(default_factory=lambda: list(range(22)))
    camera_names: List[str] = field(default_factory=list)
    control_rate_hz: float = 60.0
    mocap: MocapConfig = field(default_factory=MocapConfig)

    def validate(self) -> None:
        if self.control_hand not in {"left", "right"}:
            raise ValueError("simulation.control_hand 必须为 left/right")
        if len(self.root_ctrl_indices) != 6:
            raise ValueError("simulation.root_ctrl_indices 必须长度为6")
        if len(self.finger_ctrl_indices) < 1:
            raise ValueError("simulation.finger_ctrl_indices 不能为空")
        if len(self.root_position_offset) != 3:
            raise ValueError("simulation.root_position_offset 必须长度为3")
        mat = np.asarray(self.wrist_rotation_calib_matrix, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError("simulation.wrist_rotation_calib_matrix 必须是 3x3")
        det = float(np.linalg.det(mat))
        if not np.isfinite(det) or abs(det) < 0.01 or abs(det) > 100.0:
            raise ValueError(
                f"simulation.wrist_rotation_calib_matrix 行列式异常: {det}，请检查是否为有效旋转"
            )
        if self.control_rate_hz <= 0:
            raise ValueError("simulation.control_rate_hz 必须大于0")


@dataclass
class RuntimeConfig:
    sensor: SensorConfig
    retargeting: RetargetingConfigRuntime
    simulation: SimulationConfig

    def validate(self) -> None:
        self.sensor.validate()
        self.retargeting.validate()
        self.simulation.validate()


def _parse_sensor(raw: Dict[str, Any]) -> SensorConfig:
    webcam = raw.get("webcam", {})
    return SensorConfig(
        input_source=str(raw.get("input_source", "webcam")),
        webcam_index=int(webcam.get("index", 0)),
        camera2table=raw.get("camera2table", _default_camera2table()),
        rerun_enabled=bool(raw.get("rerun_enabled", False)),
    )


def _parse_retargeting(raw: Dict[str, Any]) -> RetargetingConfigRuntime:
    mode = str(raw.get("mode", "single_left"))

    return RetargetingConfigRuntime(
        mode=mode,
        left=dict(raw.get("left", {})),
        right=dict(raw.get("right", {})),
    )


def _parse_wrist_rotation_calib_matrix(raw: Dict[str, Any]) -> List[List[float]]:
    """解析 wrist_rotation_calib_matrix；若仅有已弃用的 root_rotation_offset_euler_zyx 则按固定轴 ZYX 合成矩阵。"""
    if "wrist_rotation_calib_matrix" in raw:
        mat = np.asarray(raw["wrist_rotation_calib_matrix"], dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError("simulation.wrist_rotation_calib_matrix 必须是 3x3")
        return mat.tolist()
    legacy = raw.get("root_rotation_offset_euler_zyx")
    if legacy is not None:
        euler = np.asarray(legacy, dtype=np.float64).reshape(-1)
        if euler.shape[0] != 3:
            raise ValueError("simulation.root_rotation_offset_euler_zyx（已弃用）必须长度为3")
        rz, ry, rx = float(euler[0]), float(euler[1]), float(euler[2])
        if abs(rz) + abs(ry) + abs(rx) < 1e-12:
            return _identity_3x3()
        return _rotation_matrix_fixed_zyx(rz, ry, rx).tolist()
    return _identity_3x3()


def _parse_simulation(raw: Dict[str, Any]) -> SimulationConfig:
    mocap_raw = dict(raw.get("mocap", {}))
    mocap = MocapConfig(
        wrist_mocap=bool(mocap_raw.get("wrist_mocap", False)),
        mocap_body_name=mocap_raw.get("mocap_body_name"),
        mocap_id=mocap_raw.get("mocap_id"),
    )
    return SimulationConfig(
        mj_xml_path=str(raw["mj_xml_path"]),
        startup_keyframe=(
            str(raw["startup_keyframe"]).strip()
            if raw.get("startup_keyframe") is not None
            else None
        ),
        control_hand=str(raw.get("control_hand", "left")),
        root_ctrl_indices=list(raw.get("root_ctrl_indices", [0, 1, 2, 3, 4, 5])),
        finger_ctrl_indices=list(
            raw.get(
                "finger_ctrl_indices",
                [14, 15, 16, 17, 18, 19, 20, 21, 10, 11, 12, 13, 6, 7, 8, 9],
            )
        ),
        root_position_offset=list(raw.get("root_position_offset", [0.2, 0.0, -0.6])),
        wrist_rotation_calib_matrix=_parse_wrist_rotation_calib_matrix(raw),
        joint_indices=raw.get("joint_indices", list(range(22))),
        camera_names=list(raw.get("camera_names", [])),
        control_rate_hz=float(raw.get("control_rate_hz", 60.0)),
        mocap=mocap,
    )


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)

    if not isinstance(raw, dict):
        raise ValueError("配置文件必须是字典结构")

    sensor = _parse_sensor(dict(raw.get("sensor", {})))
    retargeting = _parse_retargeting(dict(raw.get("retargeting", {})))
    if "simulation" not in raw:
        raise ValueError("配置缺少 simulation 字段")
    simulation = _parse_simulation(dict(raw["simulation"]))
    cfg = RuntimeConfig(sensor=sensor, retargeting=retargeting, simulation=simulation)
    cfg.validate()
    return cfg
