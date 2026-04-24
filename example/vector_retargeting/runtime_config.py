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
    """Same as mujoco_control: fixed-axis R = Rz(rz) @ Ry(ry) @ Rx(rx)."""
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


def _default_obj_position_ranges() -> List[List[float]]:
    return [[-0.2, 0.2], [0.5, 0.8], [0.03, 0.03]]


def _default_goal_position_ranges() -> List[List[float]]:
    return [[-0.3, 0.3], [0.5, 0.8], [0.1, 0.3]]


@dataclass
class SensorConfig:
    input_source: str = "webcam"
    webcam_index: int = 0
    camera2table: List[List[float]] = field(default_factory=_default_camera2table)
    rerun_enabled: bool = False

    def validate(self) -> None:
        if self.input_source not in VALID_INPUT_SOURCES:
            raise ValueError(
                f"sensor.input_source must be one of {sorted(VALID_INPUT_SOURCES)}"
            )
        mat = np.asarray(self.camera2table, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError("sensor.camera2table must be a 3x3 matrix")


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
            raise ValueError(f"Unknown hand: {hand}")
        merged = dict(self.left if hand == "left" else self.right)

        if "urdf_path" not in merged:
            raise ValueError(f"retargeting.{hand}.urdf_path is missing")
        if "optimizer" not in merged:
            raise ValueError(f"retargeting.{hand}.optimizer is missing")

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
            raise ValueError(f"retargeting.{hand}.optimizer must be a string or dict")

        if optimizer_type not in VALID_OPTIMIZERS:
            raise ValueError(
                f"retargeting.{hand}.optimizer.type must be one of {sorted(VALID_OPTIMIZERS)}"
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
            raise ValueError(f"retargeting.mode must be one of {sorted(VALID_MODES)}")
        for hand in self.active_hands():
            self.build_hand_config(hand)


@dataclass
class MocapConfig:
    wrist_mocap: bool = False
    mocap_body_name: Optional[str] = None
    mocap_id: Optional[int] = None


@dataclass
class RandomObjGoalConfig:
    enabled: bool = False
    obj_body_name: str = "obj"
    goal_site_name: str = "site"
    obj_position_ranges: List[List[float]] = field(default_factory=_default_obj_position_ranges)
    goal_position_ranges: List[List[float]] = field(default_factory=_default_goal_position_ranges)

    @staticmethod
    def _validate_ranges(name: str, ranges: List[List[float]]) -> None:
        arr = np.asarray(ranges, dtype=np.float64)
        if arr.shape != (3, 2):
            raise ValueError(
                f"{name} must be 3x2, e.g. [[xmin,xmax],[ymin,ymax],[zmin,zmax]]"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        if np.any(arr[:, 0] > arr[:, 1]):
            raise ValueError(f"{name} has min > max for some axis")

    def validate(self) -> None:
        if not self.obj_body_name.strip():
            raise ValueError("simulation.random_obj_goal.obj_body_name cannot be empty")
        if not self.goal_site_name.strip():
            raise ValueError("simulation.random_obj_goal.goal_site_name cannot be empty")
        self._validate_ranges(
            "simulation.random_obj_goal.obj_position_ranges", self.obj_position_ranges
        )
        self._validate_ranges(
            "simulation.random_obj_goal.goal_position_ranges", self.goal_position_ranges
        )


@dataclass
class AssistNearObjectConfig:
    """Space key: nudge root_position_offset along (obj - palm) to help teleop reach the object."""

    gain: float = 0.25
    max_step_m: float = 0.12
    palm_body_name: str = "LHand_PALM_LINK"
    obj_body_name: str = "obj"
    # Target point = obj_world_pos + preset_offset_xyz (world-frame translation offset, meters)
    preset_offset_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def validate(self) -> None:
        if not np.isfinite(self.gain) or self.gain < 0.0:
            raise ValueError("simulation.assist_near_object.gain must be a non-negative finite number")
        if not np.isfinite(self.max_step_m) or self.max_step_m <= 0.0:
            raise ValueError("simulation.assist_near_object.max_step_m must be a positive finite number")
        if not str(self.palm_body_name).strip():
            raise ValueError("simulation.assist_near_object.palm_body_name cannot be empty")
        if not str(self.obj_body_name).strip():
            raise ValueError("simulation.assist_near_object.obj_body_name cannot be empty")
        off = np.asarray(self.preset_offset_xyz, dtype=np.float64).reshape(-1)
        if off.shape[0] != 3 or not np.all(np.isfinite(off)):
            raise ValueError(
                "simulation.assist_near_object.preset_offset_xyz must be length-3 finite values"
            )


@dataclass
class SimulationConfig:
    mj_xml_path: str
    # If set, load this keyframe immediately after MuJoCo starts (<keyframe> name in MJCF)
    startup_keyframe: Optional[str] = None
    control_hand: str = "left"
    root_ctrl_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    finger_ctrl_indices: List[int] = field(
        default_factory=lambda: [14, 15, 16, 17, 18, 19, 20, 21, 10, 11, 12, 13, 6, 7, 8, 9]
    )
    root_position_offset: List[float] = field(default_factory=lambda: [0.2, 0.0, -0.6])
    # Left wrist rotation calibration: R_out = wrist_rotation_calib_matrix @ R_wrist (same convention as detector)
    wrist_rotation_calib_matrix: List[List[float]] = field(default_factory=_identity_3x3)
    joint_indices: Optional[List[int]] = field(default_factory=lambda: list(range(22)))
    camera_names: List[str] = field(default_factory=list)
    control_rate_hz: float = 60.0
    mocap: MocapConfig = field(default_factory=MocapConfig)
    random_obj_goal: RandomObjGoalConfig = field(default_factory=RandomObjGoalConfig)
    assist_near_object: AssistNearObjectConfig = field(default_factory=AssistNearObjectConfig)

    def validate(self) -> None:
        if self.control_hand not in {"left", "right"}:
            raise ValueError("simulation.control_hand must be left or right")
        if len(self.root_ctrl_indices) != 6:
            raise ValueError("simulation.root_ctrl_indices must have length 6")
        if len(self.finger_ctrl_indices) < 1:
            raise ValueError("simulation.finger_ctrl_indices cannot be empty")
        if len(self.root_position_offset) != 3:
            raise ValueError("simulation.root_position_offset must have length 3")
        mat = np.asarray(self.wrist_rotation_calib_matrix, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError("simulation.wrist_rotation_calib_matrix must be 3x3")
        det = float(np.linalg.det(mat))
        if not np.isfinite(det) or abs(det) < 0.01 or abs(det) > 100.0:
            raise ValueError(
                f"simulation.wrist_rotation_calib_matrix has invalid determinant {det}; expected a proper rotation"
            )
        if self.control_rate_hz <= 0:
            raise ValueError("simulation.control_rate_hz must be > 0")
        self.random_obj_goal.validate()
        self.assist_near_object.validate()


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
    """Parse wrist_rotation_calib_matrix; if only deprecated root_rotation_offset_euler_zyx is set, build ZYX matrix."""
    if "wrist_rotation_calib_matrix" in raw:
        mat = np.asarray(raw["wrist_rotation_calib_matrix"], dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError("simulation.wrist_rotation_calib_matrix must be 3x3")
        return mat.tolist()
    legacy = raw.get("root_rotation_offset_euler_zyx")
    if legacy is not None:
        euler = np.asarray(legacy, dtype=np.float64).reshape(-1)
        if euler.shape[0] != 3:
            raise ValueError("simulation.root_rotation_offset_euler_zyx (deprecated) must have length 3")
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
    random_obj_goal_raw = dict(raw.get("random_obj_goal", {}))
    random_obj_goal = RandomObjGoalConfig(
        enabled=bool(random_obj_goal_raw.get("enabled", False)),
        obj_body_name=str(random_obj_goal_raw.get("obj_body_name", "obj")),
        goal_site_name=str(random_obj_goal_raw.get("goal_site_name", "site")),
        obj_position_ranges=list(
            random_obj_goal_raw.get("obj_position_ranges", _default_obj_position_ranges())
        ),
        goal_position_ranges=list(
            random_obj_goal_raw.get("goal_position_ranges", _default_goal_position_ranges())
        ),
    )
    assist_raw = dict(raw.get("assist_near_object", {}))
    default_obj_for_assist = str(random_obj_goal_raw.get("obj_body_name", "obj"))
    assist_near_object = AssistNearObjectConfig(
        gain=float(assist_raw.get("gain", 0.25)),
        max_step_m=float(assist_raw.get("max_step_m", 0.12)),
        palm_body_name=str(assist_raw.get("palm_body_name", "LHand_PALM_LINK")),
        obj_body_name=str(assist_raw.get("obj_body_name", default_obj_for_assist)),
        preset_offset_xyz=list(assist_raw.get("preset_offset_xyz", [0.0, 0.0, 0.0])),
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
        random_obj_goal=random_obj_goal,
        assist_near_object=assist_near_object,
    )


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)

    if not isinstance(raw, dict):
        raise ValueError("Config file must parse to a dict")

    sensor = _parse_sensor(dict(raw.get("sensor", {})))
    retargeting = _parse_retargeting(dict(raw.get("retargeting", {})))
    if "simulation" not in raw:
        raise ValueError("Config is missing 'simulation' section")
    simulation = _parse_simulation(dict(raw["simulation"]))
    cfg = RuntimeConfig(sensor=sensor, retargeting=retargeting, simulation=simulation)
    cfg.validate()
    return cfg
