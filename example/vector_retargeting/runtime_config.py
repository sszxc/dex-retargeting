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
class RandomObjGoalTargetConfig:
    name: str
    type: str
    position_ranges: List[List[float]]

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
        if not str(self.name).strip():
            raise ValueError("simulation.random_obj_goal[].name cannot be empty")
        if self.type not in {"body", "site"}:
            raise ValueError(
                f"simulation.random_obj_goal[{self.name}].type must be 'body' or 'site'"
            )
        self._validate_ranges(
            f"simulation.random_obj_goal[{self.name}].position_ranges",
            self.position_ranges,
        )


@dataclass
class RandomObjGoalConfig:
    targets: List[RandomObjGoalTargetConfig] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return len(self.targets) > 0

    def validate(self) -> None:
        for target in self.targets:
            target.validate()


@dataclass
class TaskResetJointConfig:
    enabled: bool = False
    name: Optional[str] = None
    value: Any = 0.0

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.name is None or not str(self.name).strip():
            raise ValueError("simulation.task_reset_joint.name cannot be empty when enabled")
        arr = np.asarray(self.value, dtype=np.float64).reshape(-1)
        if arr.size < 1 or not np.all(np.isfinite(arr)):
            raise ValueError("simulation.task_reset_joint.value must contain finite value(s)")


def _optional_float_in_dict(d: Dict[str, Any], key: str) -> Optional[float]:
    if key not in d:
        return None
    v = d[key]
    if v is None:
        return None
    return float(v)


@dataclass
class PassiveViewerCameraConfig:
    """Optional MuJoCo passive viewer (mjvCamera) overrides; unset fields keep viewer defaults."""

    lookat: Optional[List[float]] = None
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    distance: Optional[float] = None

    def validate(self) -> None:
        if self.lookat is not None:
            arr = np.asarray(self.lookat, dtype=np.float64).reshape(-1)
            if arr.shape[0] != 3 or not np.all(np.isfinite(arr)):
                raise ValueError(
                    "simulation.viewer_camera.lookat must be length-3 finite values [x, y, z]"
                )
        for name, val in (
            ("azimuth", self.azimuth),
            ("elevation", self.elevation),
            ("distance", self.distance),
        ):
            if val is not None and not np.isfinite(val):
                raise ValueError(f"simulation.viewer_camera.{name} must be finite or omitted")


@dataclass
class SocketPublishConfig:
    """UDP JSON broadcast of the control_hand's wrist pose + finger joint angles.

    One JSON datagram per control tick (see my_retargeting_mujoco.py); subscribers
    only need stdlib `socket` + `json`, so they can run in a different environment.
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 6001

    def validate(self) -> None:
        if not self.enabled:
            return
        if not str(self.host).strip():
            raise ValueError("simulation.socket_publish.host cannot be empty")
        if not (0 < self.port < 65536):
            raise ValueError("simulation.socket_publish.port must be in (0, 65536)")


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
    # World-frame AABB for wrist target after root_position_offset. Both None = no clip.
    wrist_pos_min: Optional[List[float]] = None
    wrist_pos_max: Optional[List[float]] = None
    # Left wrist rotation calibration: R_out = wrist_rotation_calib_matrix @ R_wrist (same convention as detector)
    wrist_rotation_calib_matrix: List[List[float]] = field(default_factory=_identity_3x3)
    joint_indices: Optional[List[int]] = field(default_factory=lambda: list(range(22)))
    camera_names: List[str] = field(default_factory=list)
    control_rate_hz: float = 60.0
    mocap: MocapConfig = field(default_factory=MocapConfig)
    random_obj_goal: RandomObjGoalConfig = field(default_factory=RandomObjGoalConfig)
    task_reset_joint: TaskResetJointConfig = field(default_factory=TaskResetJointConfig)
    assist_near_object: AssistNearObjectConfig = field(default_factory=AssistNearObjectConfig)
    viewer_camera: Optional[PassiveViewerCameraConfig] = None
    socket_publish: SocketPublishConfig = field(default_factory=SocketPublishConfig)

    def validate(self) -> None:
        if self.control_hand not in {"left", "right"}:
            raise ValueError("simulation.control_hand must be left or right")
        if len(self.root_ctrl_indices) != 6:
            raise ValueError("simulation.root_ctrl_indices must have length 6")
        if len(self.finger_ctrl_indices) < 1:
            raise ValueError("simulation.finger_ctrl_indices cannot be empty")
        if len(self.root_position_offset) != 3:
            raise ValueError("simulation.root_position_offset must have length 3")
        if (self.wrist_pos_min is None) != (self.wrist_pos_max is None):
            raise ValueError(
                "simulation.wrist_pos_min and simulation.wrist_pos_max must be set together"
            )
        if self.wrist_pos_min is not None:
            lo = np.asarray(self.wrist_pos_min, dtype=np.float64).reshape(-1)
            hi = np.asarray(self.wrist_pos_max, dtype=np.float64).reshape(-1)
            if lo.size != 3 or hi.size != 3:
                raise ValueError(
                    "simulation.wrist_pos_min and wrist_pos_max must have length 3"
                )
            if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
                raise ValueError(
                    "simulation.wrist_pos_min and wrist_pos_max must be finite"
                )
            if np.any(lo > hi):
                raise ValueError(
                    "simulation.wrist_pos_min must be <= wrist_pos_max on every axis"
                )
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
        self.task_reset_joint.validate()
        self.assist_near_object.validate()
        if self.viewer_camera is not None:
            self.viewer_camera.validate()
        self.socket_publish.validate()


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
    """Parse wrist rotation offset from [roll, pitch, yaw] (radians)."""
    if "wrist_rotation_calib_matrix" in raw or "root_rotation_offset_euler_zyx" in raw:
        raise ValueError(
            "simulation.wrist_rotation_calib_matrix and simulation.root_rotation_offset_euler_zyx are no longer supported. "
            "Use simulation.wrist_rotation_offset_rpy: [roll, pitch, yaw] in radians."
        )

    rpy = raw.get("wrist_rotation_offset_rpy")
    if rpy is None:
        return _identity_3x3()

    euler = np.asarray(rpy, dtype=np.float64).reshape(-1)
    if euler.shape[0] != 3:
        raise ValueError("simulation.wrist_rotation_offset_rpy must have length 3: [roll, pitch, yaw]")
    roll, pitch, yaw = float(euler[0]), float(euler[1]), float(euler[2])
    if abs(roll) + abs(pitch) + abs(yaw) < 1e-12:
        return _identity_3x3()
    # Same fixed-axis convention as mujoco_control: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return _rotation_matrix_fixed_zyx(yaw, pitch, roll).tolist()


def _parse_passive_viewer_camera(raw: Dict[str, Any]) -> Optional[PassiveViewerCameraConfig]:
    block = raw.get("viewer_camera")
    if block is None or block is False:
        return None
    if not isinstance(block, dict):
        raise ValueError("simulation.viewer_camera must be a mapping or null")
    lookat: Optional[List[float]] = None
    if "lookat" in block and block["lookat"] is not None:
        lookat = [float(x) for x in block["lookat"]]
    azimuth = _optional_float_in_dict(block, "azimuth")
    elevation = _optional_float_in_dict(block, "elevation")
    distance = _optional_float_in_dict(block, "distance")
    if lookat is None and azimuth is None and elevation is None and distance is None:
        return None
    return PassiveViewerCameraConfig(
        lookat=lookat,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
    )


def _parse_task_reset_joint(raw: Dict[str, Any]) -> TaskResetJointConfig:
    block = raw.get("task_reset_joint", {})
    if block is None or block is False:
        return TaskResetJointConfig(enabled=False)
    if block is True:
        raise ValueError(
            "simulation.task_reset_joint=true requires a mapping with name/value, "
            "e.g. task_reset_joint: {enabled: true, name: goal_slidey, value: 0.0}"
        )
    if not isinstance(block, dict):
        raise ValueError("simulation.task_reset_joint must be a mapping, false, or omitted")
    return TaskResetJointConfig(
        enabled=bool(block.get("enabled", False)),
        name=(str(block["name"]).strip() if block.get("name") is not None else None),
        value=block.get("value", 0.0),
    )


def _parse_random_obj_goal(raw: Dict[str, Any]) -> RandomObjGoalConfig:
    block = raw.get("random_obj_goal")
    if block is None or block is False:
        return RandomObjGoalConfig()
    if not isinstance(block, list):
        raise ValueError(
            "simulation.random_obj_goal must be a list of targets, false, or omitted. "
            "Example: [{name: obj, type: body, position_ranges: [[-0.2,0.2],[0.5,0.8],[0.03,0.03]]}]"
        )

    targets: List[RandomObjGoalTargetConfig] = []
    for i, item in enumerate(block):
        if not isinstance(item, dict):
            raise ValueError(f"simulation.random_obj_goal[{i}] must be a mapping")
        if "name" not in item:
            raise ValueError(f"simulation.random_obj_goal[{i}].name is required")
        if "type" not in item:
            raise ValueError(f"simulation.random_obj_goal[{i}].type is required")
        if "position_ranges" not in item:
            raise ValueError(f"simulation.random_obj_goal[{i}].position_ranges is required")
        targets.append(
            RandomObjGoalTargetConfig(
                name=str(item["name"]).strip(),
                type=str(item["type"]).strip(),
                position_ranges=list(item["position_ranges"]),
            )
        )
    return RandomObjGoalConfig(targets=targets)


def _parse_optional_xyz(raw: Dict[str, Any], key: str) -> Optional[List[float]]:
    value = raw.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        raise ValueError(f"simulation.{key} must be length-3 finite values")
    return [float(x) for x in arr]


def _parse_socket_publish(raw: Dict[str, Any]) -> SocketPublishConfig:
    block = dict(raw.get("socket_publish", {}))
    return SocketPublishConfig(
        enabled=bool(block.get("enabled", False)),
        host=str(block.get("host", "127.0.0.1")),
        port=int(block.get("port", 6001)),
    )


def _parse_simulation(raw: Dict[str, Any]) -> SimulationConfig:
    mocap_raw = dict(raw.get("mocap", {}))
    mocap = MocapConfig(
        wrist_mocap=bool(mocap_raw.get("wrist_mocap", False)),
        mocap_body_name=mocap_raw.get("mocap_body_name"),
        mocap_id=mocap_raw.get("mocap_id"),
    )
    random_obj_goal = _parse_random_obj_goal(raw)
    assist_raw = dict(raw.get("assist_near_object", {}))
    default_obj_for_assist = next(
        (target.name for target in random_obj_goal.targets if target.type == "body"),
        "obj",
    )
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
        wrist_pos_min=_parse_optional_xyz(raw, "wrist_pos_min"),
        wrist_pos_max=_parse_optional_xyz(raw, "wrist_pos_max"),
        wrist_rotation_calib_matrix=_parse_wrist_rotation_calib_matrix(raw),
        joint_indices=raw.get("joint_indices", list(range(22))),
        camera_names=list(raw.get("camera_names", [])),
        control_rate_hz=float(raw.get("control_rate_hz", 60.0)),
        mocap=mocap,
        random_obj_goal=random_obj_goal,
        task_reset_joint=_parse_task_reset_joint(raw),
        assist_near_object=assist_near_object,
        viewer_camera=_parse_passive_viewer_camera(raw),
        socket_publish=_parse_socket_publish(raw),
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
