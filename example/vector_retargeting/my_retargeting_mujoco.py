from __future__ import annotations

import multiprocessing
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Full
from typing import Optional

import h5py
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from loguru import logger
from pynput import keyboard

from mujoco_control import MujocoHandController, read_finger_qpos
from retarget_worker import run_retarget_worker
from runtime_config import RuntimeConfig, SHAPE_SIZE_LEN, SimulationConfig, load_runtime_config
from state_publisher import HandStatePublisher


DEFAULT_LOOKAT = [0.0, 0.0, 0.2]
DEFAULT_CAMERA_PARAMS = {"distance": 0.8, "elevation": -30, "azimuth": 0}
DEFAULT_VIEWER_COUNT = 2

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
        logger.info("RerunBoard enabled")
        return board, rr
    except Exception as err:
        logger.warning(f"RerunBoard failed to start (continuing without it): {err}")
        return (DummyClass() if DummyClass is not None else None), None


def _apply_passive_viewer_camera(viewer, sim: SimulationConfig) -> None:
    """Apply optional mjvCamera fields from runtime YAML; no-op if simulation.viewer_camera is absent."""
    cfg = sim.viewer_camera
    if cfg is None:
        return
    if cfg.lookat is not None:
        viewer.cam.lookat[:] = np.asarray(cfg.lookat, dtype=np.float64).reshape(3)
    if cfg.azimuth is not None:
        viewer.cam.azimuth = float(cfg.azimuth)
    if cfg.elevation is not None:
        viewer.cam.elevation = float(cfg.elevation)
    if cfg.distance is not None:
        viewer.cam.distance = float(cfg.distance)


def _configure_passive_viewer(viewer, sim: SimulationConfig) -> None:
    options = viewer.opt
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
    options.label = mujoco.mjtLabel.mjLABEL_CAMERA
    _apply_passive_viewer_camera(viewer, sim)


def _launch_passive_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sim: SimulationConfig,
    viewer_index: int,
    viewer_count: int,
):
    viewer = mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    )
    _configure_passive_viewer(viewer, sim)
    logger.info(f"MuJoCo viewer {viewer_index}/{viewer_count} started")
    return viewer


def _make_viewer_state_snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    return {
        "time": float(data.time),
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "ctrl": np.asarray(data.ctrl).copy(),
        "mocap_pos": np.asarray(data.mocap_pos).copy(),
        "mocap_quat": np.asarray(data.mocap_quat).copy(),
        "body_pos": np.asarray(model.body_pos).copy(),
        "site_pos": np.asarray(model.site_pos).copy(),
        # Mutated in-place by `_apply_random_object_shape` (shape/size/color randomization
        # of the pick object) -- must be forwarded too, or secondary viewers keep showing
        # the old shape/size/color even though the object's pose (qpos, above) does update.
        "geom_type": np.asarray(model.geom_type).copy(),
        "geom_size": np.asarray(model.geom_size).copy(),
        "geom_rgba": np.asarray(model.geom_rgba).copy(),
        "body_mass": np.asarray(model.body_mass).copy(),
        "body_inertia": np.asarray(model.body_inertia).copy(),
        "body_ipos": np.asarray(model.body_ipos).copy(),
        "body_iquat": np.asarray(model.body_iquat).copy(),
    }


def _apply_viewer_state_snapshot(
    model: mujoco.MjModel, data: mujoco.MjData, snapshot: dict
) -> None:
    data.time = float(snapshot["time"])
    data.qpos[:] = snapshot["qpos"]
    data.qvel[:] = snapshot["qvel"]
    data.ctrl[:] = snapshot["ctrl"]
    if model.nmocap > 0:
        data.mocap_pos[:] = snapshot["mocap_pos"]
        data.mocap_quat[:] = snapshot["mocap_quat"]
    model.body_pos[:] = snapshot["body_pos"]
    model.site_pos[:] = snapshot["site_pos"]
    model.geom_type[:] = snapshot["geom_type"]
    model.geom_size[:] = snapshot["geom_size"]
    model.geom_rgba[:] = snapshot["geom_rgba"]
    model.body_mass[:] = snapshot["body_mass"]
    model.body_inertia[:] = snapshot["body_inertia"]
    model.body_ipos[:] = snapshot["body_ipos"]
    model.body_iquat[:] = snapshot["body_iquat"]
    mujoco.mj_forward(model, data)


def _publish_viewer_state(queues: list[multiprocessing.Queue], snapshot: dict) -> None:
    for queue in queues:
        try:
            queue.put_nowait(snapshot)
        except Full:
            try:
                queue.get_nowait()
            except Empty:
                pass
            try:
                queue.put_nowait(snapshot)
            except Full:
                pass


def _run_secondary_viewer(
    mj_xml_path: str,
    sim: SimulationConfig,
    state_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    viewer_index: int,
    viewer_count: int,
) -> None:
    model = mujoco.MjModel.from_xml_path(mj_xml_path)
    data = mujoco.MjData(model)
    viewer = _launch_passive_viewer(model, data, sim, viewer_index, viewer_count)
    try:
        while viewer.is_running() and not stop_event.is_set():
            latest_snapshot = None
            try:
                latest_snapshot = state_queue.get(timeout=0.01)
            except Empty:
                pass
            while True:
                try:
                    latest_snapshot = state_queue.get_nowait()
                except Empty:
                    break

            if latest_snapshot is not None:
                _apply_viewer_state_snapshot(model, data, latest_snapshot)
            viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            viewer.close()
        except Exception:
            pass


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
            raise FileNotFoundError(f"No .xml file found in directory: {p}")
        return first_xml
    if not p.exists():
        raise FileNotFoundError(f"MuJoCo scene file does not exist: {p}")
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
                logger.warning(f"Camera name(s) not found in XML, ignored: {unknown}")
        specs = [(name, name) for name in names_to_use]
        return specs, list(names_to_use)

    default_camera = _make_dynamic_camera(DEFAULT_CAMERA_PARAMS)
    logger.info("No cameras in scene XML; using built-in default camera 'default'")
    return [("default", default_camera)], ["default"]


def _sample_xyz(ranges: list[list[float]]) -> np.ndarray:
    arr = np.asarray(ranges, dtype=np.float64)
    return np.array(
        [
            np.random.uniform(arr[0, 0], arr[0, 1]),
            np.random.uniform(arr[1, 0], arr[1, 1]),
            np.random.uniform(arr[2, 0], arr[2, 1]),
        ],
        dtype=np.float64,
    )


def _apply_assist_root_offset_from_palm_obj(
    model: mujoco.MjModel, data: mujoco.MjData, sim: SimulationConfig
) -> tuple[bool, str]:
    """Nudge root_position_offset along palm→obj in simulation (pull mocap/root target toward the object)."""
    cfg = sim.assist_near_object
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.palm_body_name)
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.obj_body_name)
    if palm_id < 0:
        return False, f"assist: palm body not found '{cfg.palm_body_name}'"
    if obj_id < 0:
        return False, f"assist: obj body not found '{cfg.obj_body_name}'"
    palm = np.asarray(data.xpos[palm_id], dtype=np.float64).reshape(3)
    objp = np.asarray(data.xpos[obj_id], dtype=np.float64).reshape(3)
    preset = np.asarray(cfg.preset_offset_xyz, dtype=np.float64).reshape(3)
    rel = (objp + preset) - palm
    step = float(cfg.gain) * rel
    norm = float(np.linalg.norm(step))
    if norm > float(cfg.max_step_m) and norm > 1e-12:
        step = step * (float(cfg.max_step_m) / norm)
    off = np.asarray(sim.root_position_offset, dtype=np.float64).reshape(3) + step
    for i in range(3):
        sim.root_position_offset[i] = float(off[i])
    return True, (
        f"assist_near_object: Δoffset={np.round(step, 4).tolist()}, "
        f"preset={np.round(preset, 4).tolist()}, "
        f"root_position_offset={np.round(off, 4).tolist()}"
    )


def _randomize_obj_goal_pose(
    model: mujoco.MjModel, data: mujoco.MjData, runtime_cfg: RuntimeConfig
) -> bool:
    cfg = runtime_cfg.simulation.random_obj_goal
    if not cfg.enabled:
        return False

    updated = False
    randomized: list[str] = []

    for target in cfg.targets:
        pos = _sample_xyz(target.position_ranges)
        if target.type == "body":
            target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target.name)
            if target_id < 0:
                logger.warning(f"Randomize scene failed: body not found '{target.name}'")
                continue

            body_joint_num = int(model.body_jntnum[target_id])
            body_joint_adr = int(model.body_jntadr[target_id])
            if body_joint_num > 0 and body_joint_adr >= 0:
                joint_id = body_joint_adr
                joint_type = model.jnt_type[joint_id]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    qadr = int(model.jnt_qposadr[joint_id])
                    dadr = int(model.jnt_dofadr[joint_id])
                    data.qpos[qadr : qadr + 3] = pos
                    data.qvel[dadr : dadr + 6] = 0.0
                else:
                    logger.warning(
                        f"body '{target.name}' is not a free joint; writing model.body_pos instead"
                    )
                    model.body_pos[target_id] = pos
            else:
                model.body_pos[target_id] = pos
        elif target.type == "site":
            target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, target.name)
            if target_id < 0:
                logger.warning(f"Randomize scene failed: site not found '{target.name}'")
                continue
            model.site_pos[target_id] = pos
        else:
            logger.warning(f"Randomize scene skipped unsupported type '{target.type}'")
            continue

        updated = True
        randomized.append(f"{target.name}({target.type})={pos.round(4).tolist()}")

    if updated:
        mujoco.mj_forward(model, data)
        logger.info(f"Randomized targets: {', '.join(randomized)}")
    return updated


_SHAPE_NAME_TO_MJTGEOM = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
}


def _shape_z_half_extent(shape_name: str, size: np.ndarray) -> float:
    """Local +z half-extent of a resting primitive, i.e. how far its center must sit
    above the table surface so its bottom just touches it."""
    if shape_name == "box":
        return float(size[2])
    if shape_name == "sphere":
        return float(size[0])
    if shape_name == "cylinder":
        return float(size[1])
    if shape_name == "capsule":
        return float(size[1]) + float(size[0])
    raise ValueError(f"Unsupported shape '{shape_name}'")


def _shape_mass_and_diag_inertia(
    shape_name: str, size: np.ndarray, density: float
) -> tuple[float, np.ndarray]:
    """Exact analytic mass + principal (diagonal) inertia for a uniform-density primitive,
    centered at the geom's own origin. Verified numerically to match MuJoCo's own compiler
    output bit-for-bit (`MjModel.from_xml_string` with `density=...`) for all four shapes.

    This is needed because mutating `geom_type`/`geom_size` on an already-compiled model
    does NOT update `body_mass`/`body_inertia` for us -- confirmed empirically that even
    `mj_setConst` leaves them untouched (it only recomputes qpos0-dependent constants,
    e.g. tendon lengths), so we compute and assign them ourselves every time the shape
    changes.
    """
    if shape_name == "box":
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        a, b, c = 2 * sx, 2 * sy, 2 * sz
        m = density * a * b * c
        return m, np.array(
            [m * (b**2 + c**2) / 12.0, m * (a**2 + c**2) / 12.0, m * (a**2 + b**2) / 12.0]
        )
    if shape_name == "sphere":
        r = float(size[0])
        m = density * (4.0 / 3.0) * np.pi * r**3
        i = (2.0 / 5.0) * m * r**2
        return m, np.array([i, i, i])
    if shape_name == "cylinder":
        r, h = float(size[0]), float(size[1])
        m = density * np.pi * r**2 * (2 * h)
        izz = 0.5 * m * r**2
        ixx = m * (3 * r**2 + (2 * h) ** 2) / 12.0
        return m, np.array([ixx, ixx, izz])
    if shape_name == "capsule":
        r, h = float(size[0]), float(size[1])
        m_cyl = density * np.pi * r**2 * (2 * h)
        m_hemi = density * (2.0 / 3.0) * np.pi * r**3  # one hemispherical cap
        m = m_cyl + 2 * m_hemi
        izz = 0.5 * m_cyl * r**2 + 2 * (2.0 / 5.0) * m_hemi * r**2
        ixx_cyl = m_cyl * (3 * r**2 + (2 * h) ** 2) / 12.0
        ixx_hemi_own = (83.0 / 320.0) * m_hemi * r**2  # about the hemisphere's own centroid
        d = h + 3 * r / 8.0  # centroid offset of each cap from the capsule's own center
        ixx = ixx_cyl + 2 * (ixx_hemi_own + m_hemi * d**2)
        return m, np.array([ixx, ixx, izz])
    raise ValueError(f"Unsupported shape '{shape_name}'")


def _apply_random_object_shape(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime_cfg: RuntimeConfig,
    force_shape_name: Optional[str] = None,
) -> Optional[dict]:
    """Randomize the pick object's shape/size/color/pose in place (mutating the already-
    compiled model -- no XML recompile, so it fits the same 'r' key / startup reset path
    as `_randomize_obj_goal_pose`).

    Draws only from `train_shapes` by default, so `held_out_shape` never appears during
    normal data collection. Pass force_shape_name (e.g. the held-out shape's name) to
    deliberately collect/preview an eval set with one specific shape instead.

    Returns metadata about what was sampled (for episode bookkeeping), or None if this
    feature is disabled for the currently loaded task/xml (i.e. any task other than pick,
    where `simulation.random_object_shape` is simply absent from the yml).
    """
    cfg = runtime_cfg.simulation.random_object_shape
    if not cfg.enabled:
        return None

    shapes_by_name = cfg.all_shapes_by_name
    if force_shape_name is not None:
        if force_shape_name not in shapes_by_name:
            raise ValueError(
                f"--force-shape '{force_shape_name}' is not defined in random_object_shape "
                f"(known: {sorted(shapes_by_name)})"
            )
        shape = shapes_by_name[force_shape_name]
    else:
        shape = cfg.train_shapes[int(np.random.randint(len(cfg.train_shapes)))]

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.body_name)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.geom_name)
    if body_id < 0 or geom_id < 0:
        logger.warning(
            f"random_object_shape: body '{cfg.body_name}' or geom '{cfg.geom_name}' not found; skipping"
        )
        return None

    size = np.array([np.random.uniform(lo, hi) for lo, hi in shape.size_ranges], dtype=np.float64)
    assert size.shape[0] == SHAPE_SIZE_LEN[shape.name]
    full_size = np.zeros(3, dtype=np.float64)
    full_size[: size.shape[0]] = size

    model.geom_type[geom_id] = int(_SHAPE_NAME_TO_MJTGEOM[shape.name])
    model.geom_size[geom_id] = full_size
    if cfg.randomize_color:
        model.geom_rgba[geom_id] = [
            np.random.uniform(0.15, 0.95),
            np.random.uniform(0.15, 0.95),
            np.random.uniform(0.15, 0.95),
            1.0,
        ]

    mass, diag_inertia = _shape_mass_and_diag_inertia(shape.name, size, cfg.density)
    model.body_mass[body_id] = mass
    model.body_inertia[body_id] = diag_inertia
    # The geom sits at the body's own local origin in the pick scene XML, so the inertial
    # frame stays centered there regardless of which shape/size is currently active.
    model.body_ipos[body_id] = [0.0, 0.0, 0.0]
    model.body_iquat[body_id] = [1.0, 0.0, 0.0, 0.0]

    xy = np.array(
        [np.random.uniform(*cfg.position_ranges[0]), np.random.uniform(*cfg.position_ranges[1])],
        dtype=np.float64,
    )
    z = cfg.table_z + _shape_z_half_extent(shape.name, size)
    yaw = np.random.uniform(cfg.yaw_range[0], cfg.yaw_range[1])
    quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)

    body_joint_num = int(model.body_jntnum[body_id])
    body_joint_adr = int(model.body_jntadr[body_id])
    if body_joint_num < 1 or model.jnt_type[body_joint_adr] != mujoco.mjtJoint.mjJNT_FREE:
        logger.warning(
            f"random_object_shape: body '{cfg.body_name}' has no free joint; skipping pose randomization"
        )
    else:
        qadr = int(model.jnt_qposadr[body_joint_adr])
        dadr = int(model.jnt_dofadr[body_joint_adr])
        data.qpos[qadr : qadr + 3] = [xy[0], xy[1], z]
        data.qpos[qadr + 3 : qadr + 7] = quat
        data.qvel[dadr : dadr + 6] = 0.0

    mujoco.mj_forward(model, data)

    info = {
        "shape": shape.name,
        "size": size.tolist(),
        "position": [float(xy[0]), float(xy[1]), float(z)],
        "yaw": float(yaw),
    }
    logger.info(f"Randomized pick object: {info}")
    return info


def _apply_task_reset_joint(
    model: mujoco.MjModel, data: mujoco.MjData, sim: SimulationConfig
) -> bool:
    cfg = sim.task_reset_joint
    if not cfg.enabled:
        return False

    joint_name = str(cfg.name)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        logger.warning(f"Task reset joint failed: joint not found '{joint_name}'")
        return False

    qadr = int(model.jnt_qposadr[joint_id])
    dadr = int(model.jnt_dofadr[joint_id])
    joint_type = model.jnt_type[joint_id]
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        qwidth, dwidth = 7, 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        qwidth, dwidth = 4, 3
    else:
        qwidth, dwidth = 1, 1

    value = np.asarray(cfg.value, dtype=np.float64).reshape(-1)
    if value.size == 1 and qwidth == 1:
        value = np.full(qwidth, float(value[0]), dtype=np.float64)
    if value.size != qwidth:
        logger.warning(
            f"Task reset joint failed: joint '{joint_name}' expects {qwidth} qpos value(s), got {value.size}"
        )
        return False

    data.qpos[qadr : qadr + qwidth] = value
    data.qvel[dadr : dadr + dwidth] = 0.0
    mujoco.mj_forward(model, data)
    logger.info(f"Task reset joint: {joint_name}={value.round(4).tolist()}")
    return True


def _apply_startup_joint_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, sim: SimulationConfig
) -> bool:
    """Force-set named joints' qpos right after startup (after startup_keyframe, so it
    wins). Looked up by joint name directly, so it works whether or not the joint has an
    actuator (e.g. a passive, mocap/weld-driven arm joint) — unlike driving through a
    ctrl index, which silently breaks if that ctrl index doesn't exist or maps to a
    different joint than intended. Hinge/slide (1 qpos) joints only."""
    mapping = sim.startup_joint_qpos
    if not mapping:
        return False

    applied: dict[str, float] = {}
    for joint_name, value in mapping.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            logger.warning(f"startup_joint_qpos: joint not found '{joint_name}'; skipping")
            continue
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            logger.warning(
                f"startup_joint_qpos: joint '{joint_name}' is not hinge/slide (1 qpos); skipping"
            )
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        data.qpos[qadr] = float(value)
        data.qvel[dadr] = 0.0
        applied[joint_name] = float(value)

    if applied:
        mujoco.mj_forward(model, data)
        logger.info(f"Loaded startup_joint_qpos={applied}")
    return bool(applied)


def _find_weld_partner_body(model: mujoco.MjModel, body_id: int) -> Optional[int]:
    """Find the other body in an <equality type="weld"> constraint involving body_id,
    e.g. the arm end-effector welded to a teleop mocap target."""
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_WELD:
            continue
        obj1, obj2 = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
        if obj1 == body_id:
            return obj2
        if obj2 == body_id:
            return obj1
    return None


def _sync_mocap_to_startup_pose(
    model: mujoco.MjModel, data: mujoco.MjData, controller: MujocoHandController
) -> None:
    """After a startup_joint_qpos override, snap the mocap target to its <equality weld>
    partner body's current FK pose (e.g. the arm's end-effector), so the weld constraint
    starts out satisfied instead of immediately pulling the arm away from the override
    on the very first mj_step. Note: for an underactuated/redundant chain (more passive
    joints than the weld's 6 constraint DOF), this only pins the end-effector pose —
    the individual joint angles you set may still drift over the next few seconds toward
    whichever configuration the rest of the passive dynamics (springs/gravity) settle on."""
    if not controller.simulation.mocap.wrist_mocap:
        return
    try:
        mocap_id = controller._resolve_mocap_id()  # type: ignore[attr-defined]
    except Exception as err:
        logger.warning(f"startup_joint_qpos: mocap id unresolved, skipping mocap sync: {err}")
        return
    if mocap_id is None:
        return

    body_mocapid = np.asarray(model.body_mocapid)
    match = np.where(body_mocapid == mocap_id)[0]
    if match.size == 0:
        logger.warning("startup_joint_qpos: could not resolve mocap body id; skipping mocap sync")
        return
    mocap_body_id = int(match[0])

    partner_body_id = _find_weld_partner_body(model, mocap_body_id)
    if partner_body_id is None:
        logger.warning(
            "startup_joint_qpos: mocap enabled but no <equality weld> found for the mocap "
            "body; mocap target left unchanged (it may pull the arm away from the override)."
        )
        return

    data.mocap_pos[mocap_id] = np.asarray(data.xpos[partner_body_id], dtype=np.float64).copy()
    data.mocap_quat[mocap_id] = np.asarray(data.xquat[partner_body_id], dtype=np.float64).copy()
    mujoco.mj_forward(model, data)
    logger.info(
        f"Synced mocap target to weld partner body id={partner_body_id} after startup_joint_qpos"
    )


def _reset_randomized_env(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime_cfg: RuntimeConfig,
    force_shape_name: Optional[str] = None,
) -> tuple[bool, Optional[dict]]:
    randomized = _randomize_obj_goal_pose(model, data, runtime_cfg)
    joint_reset = _apply_task_reset_joint(model, data, runtime_cfg.simulation)
    shape_info = _apply_random_object_shape(
        model, data, runtime_cfg, force_shape_name=force_shape_name
    )
    return (randomized or joint_reset or shape_info is not None), shape_info


def main(
    runtime_config_path: str,
    dataset_dir: str = "data/hdf5",
    start_key: str = "s",
    stop_key: str = "e",
    track_key: str = "t",
    viewer_count: int = DEFAULT_VIEWER_COUNT,
    force_shape: Optional[str] = None,
):
    """
    Args:
        force_shape: Only meaningful when the loaded task's `simulation.random_object_shape`
            is enabled (e.g. the pick task). If set, every reset uses exactly this shape
            instead of sampling from `train_shapes` -- pass the `held_out_shape` name
            (e.g. "capsule") to deliberately collect/preview an unseen-shape eval set,
            or a train shape's name to spot-check just that one. Leave unset for normal
            training-data collection, which draws only from `train_shapes`.
    """
    runtime_cfg: RuntimeConfig = load_runtime_config(runtime_config_path)
    logger.info(
        f"Loaded config: input_source={runtime_cfg.sensor.input_source}, mode={runtime_cfg.retargeting.mode}"
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

    # Optional: load a named keyframe right after startup (MJCF <keyframe name="...">)
    if runtime_cfg.simulation.startup_keyframe:
        kf_name = runtime_cfg.simulation.startup_keyframe
        kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, kf_name)
        if kf_id < 0:
            logger.warning(
                f"startup_keyframe='{kf_name}' not found in model (check <keyframe> name in XML); ignoring."
            )
        else:
            mujoco.mj_resetDataKeyframe(model, data, kf_id)
            mujoco.mj_forward(model, data)
            logger.info(f"Loaded startup_keyframe='{kf_name}' (id={kf_id})")

    controller = MujocoHandController(simulation=runtime_cfg.simulation, model=model)

    # Optional: override named joints' qpos with a manually-supplied value, e.g. copied
    # from a real arm's current joint reading. Applied after startup_keyframe so it wins.
    if _apply_startup_joint_qpos(model, data, runtime_cfg.simulation):
        _sync_mocap_to_startup_pose(model, data, controller)

    _, latest_object_info = _reset_randomized_env(
        model, data, runtime_cfg, force_shape_name=force_shape
    )

    initial_root_position_offset = np.asarray(
        runtime_cfg.simulation.root_position_offset, dtype=np.float64
    ).reshape(3).copy()
    # Mocap id for recording action in wrist_mocap mode only (does not affect control)
    mocap_id_for_action: Optional[int] = None
    if runtime_cfg.simulation.mocap.wrist_mocap:
        try:
            mocap_id_for_action = controller._resolve_mocap_id()  # type: ignore[attr-defined]
        except Exception as err:
            logger.warning(f"Failed to parse mocap_id (action will omit mocap pose): {err}")

    state_publisher: Optional[HandStatePublisher] = None
    if runtime_cfg.simulation.socket_publish.enabled:
        state_publisher = HandStatePublisher(
            host=runtime_cfg.simulation.socket_publish.host,
            port=runtime_cfg.simulation.socket_publish.port,
        )
        logger.info(
            f"State publisher enabled: udp://{runtime_cfg.simulation.socket_publish.host}"
            f":{runtime_cfg.simulation.socket_publish.port}"
        )

    recording_camera_specs, recording_camera_names = _setup_recording_cameras(
        model, runtime_cfg.simulation.camera_names
    )

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
            logger.warning(f"Robot visualization init failed (skipping joint 3D viz): {err}")

    worker = multiprocessing.Process(
        target=run_retarget_worker,
        args=(qpos_queue, runtime_cfg, str(robot_dir)),
    )
    worker.start()
    logger.info("Detection/retargeting worker process started")

    keyboard_lock = threading.Lock()
    should_exit = False
    record_start_requested = False
    record_stop_requested = False
    randomize_requested = False
    assist_near_object_pending = False
    translation_tracking_active = False
    translation_engage_pending = False
    translation_track_key_down = False
    is_recording = False
    episode_idx = 0
    episode_buffers: Optional[dict[str, list]] = None
    # Object shape/size/pose sampled at the moment recording started for the in-progress
    # episode (snapshotted here rather than read at save time, in case a stray 'r' press
    # re-randomizes the object again before the episode is stopped and saved).
    episode_object_info: Optional[dict] = None

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

    def save_episode(
        buffers: dict[str, list], idx: int, object_info: Optional[dict] = None
    ) -> None:
        if not buffers["/action"]:
            logger.warning(f"Episode {idx} has no steps; skipping save.")
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
            if runtime_cfg.simulation.task_name:
                root.attrs["task_name"] = runtime_cfg.simulation.task_name
            if object_info is not None:
                root.attrs["object_shape"] = object_info["shape"]
                root.attrs["object_size"] = np.asarray(object_info["size"], dtype=np.float64)
                root.attrs["object_position"] = np.asarray(
                    object_info["position"], dtype=np.float64
                )
                root.attrs["object_yaw"] = float(object_info["yaw"])

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
        logger.info(f"Episode {idx} saved to {dataset_path}, steps={max_timesteps}")

    def on_press(key):
        nonlocal record_start_requested, record_stop_requested, randomize_requested, should_exit
        nonlocal assist_near_object_pending, translation_tracking_active, translation_engage_pending
        nonlocal translation_track_key_down
        print(f"[DEBUG] on_press fired: key={key!r}", flush=True)
        try:
            if hasattr(key, "char") and key.char:
                with keyboard_lock:
                    if key.char == track_key:
                        # OS key-repeat re-fires on_press while the key is held, with no
                        # on_release in between; only the first press of a given physical
                        # hold (key_down False -> True) should flip the toggle.
                        if not translation_track_key_down:
                            translation_track_key_down = True
                            translation_tracking_active = not translation_tracking_active
                            if translation_tracking_active:
                                translation_engage_pending = True
                            print(
                                f"[DEBUG] TRACK key toggled -> translation_tracking_active={translation_tracking_active}",
                                flush=True,
                            )
                    elif key.char == start_key and not is_recording:
                        record_start_requested = True
                    elif key.char == stop_key and is_recording:
                        record_stop_requested = True
                    elif key.char == "r":
                        randomize_requested = True
                    elif key.char == "a":
                        assist_near_object_pending = True
                    elif key.char == "q":
                        should_exit = True
        except AttributeError:
            pass

    def on_release(key):
        nonlocal translation_track_key_down
        print(f"[DEBUG] on_release fired: key={key!r}", flush=True)
        if hasattr(key, "char") and key.char == track_key:
            with keyboard_lock:
                translation_track_key_down = False

    total_viewer_count = max(1, int(viewer_count))
    secondary_viewer_queues: list[multiprocessing.Queue] = []
    secondary_viewer_stop_events: list[multiprocessing.Event] = []
    secondary_viewer_processes: list[multiprocessing.Process] = []
    for secondary_i in range(1, total_viewer_count):
        state_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=2)
        stop_event = multiprocessing.Event()
        viewer_process = multiprocessing.Process(
            target=_run_secondary_viewer,
            args=(
                str(mj_xml_path),
                runtime_cfg.simulation,
                state_queue,
                stop_event,
                secondary_i + 1,
                total_viewer_count,
            ),
        )
        viewer_process.start()
        secondary_viewer_queues.append(state_queue)
        secondary_viewer_stop_events.append(stop_event)
        secondary_viewer_processes.append(viewer_process)

    renderer = mujoco.Renderer(model, width=640, height=480)

    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()
    logger.info(
        f"Keyboard listener started: '{start_key}' start recording, '{stop_key}' stop and save, "
        f"'r' randomize obj/goal (and shape/size/color when random_object_shape is enabled), "
        f"'{track_key}' toggle wrist TRANSLATION tracking to camera "
        f"hand position (offset recomputed each time it turns on, so it resumes without a "
        f"jump; rotation always tracks via the config's static wrist_rotation_offset_rpy), "
        f"'a' (when hand detected) nudge root_position_offset along palm→obj, 'q' quit"
    )

    latest_msg = None
    control_interval = 1.0 / runtime_cfg.simulation.control_rate_hz
    last_control_time = time.time()

    viewer = _launch_passive_viewer(
        model=model,
        data=data,
        sim=runtime_cfg.simulation,
        viewer_index=1,
        viewer_count=total_viewer_count,
    )
    _publish_viewer_state(
        secondary_viewer_queues, _make_viewer_state_snapshot(model, data)
    )
    try:
        sim_start = time.time()
        sim_time_start = float(data.time)

        while viewer.is_running() and not should_exit:
            now = time.time()

            while True:
                try:
                    msg = qpos_queue.get_nowait()
                except Empty:
                    break
                latest_msg = msg

            if latest_msg is not None and now - last_control_time >= control_interval:
                with keyboard_lock:
                    assist_do = assist_near_object_pending
                    track_translation = translation_tracking_active
                    engage_translation = translation_engage_pending
                    translation_engage_pending = False
                if assist_do:
                    ch = runtime_cfg.simulation.control_hand
                    if latest_msg.get(f"hand_{ch}_qpos") is not None:
                        ok, assist_msg = _apply_assist_root_offset_from_palm_obj(
                            model, data, runtime_cfg.simulation
                        )
                        if ok:
                            logger.info(assist_msg)
                        else:
                            logger.warning(assist_msg)
                        with keyboard_lock:
                            assist_near_object_pending = False
                controller.apply(
                    data,
                    latest_msg,
                    track_translation=track_translation,
                    engage_translation=engage_translation,
                )

                if state_publisher is not None and mocap_id_for_action is not None:
                    ch = runtime_cfg.simulation.control_hand
                    joint_names = getattr(runtime_cfg.retargeting, ch).get("target_joint_names") or []
                    if joint_names:
                        state_publisher.publish(
                            hand=ch,
                            sim_time=data.time,
                            wrist_pos=np.asarray(data.mocap_pos[mocap_id_for_action]).reshape(3),
                            wrist_quat_wxyz=np.asarray(data.mocap_quat[mocap_id_for_action]).reshape(4),
                            joint_names=joint_names,
                            joint_angles=read_finger_qpos(
                                model, data, runtime_cfg.simulation.finger_ctrl_indices
                            ),
                        )

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
                        logger.debug(f"Rerun log failed (ignored): {err}")

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
                    ctrl_sample = np.asarray(data.ctrl).copy()
                    if runtime_cfg.simulation.mocap.wrist_mocap and mocap_id_for_action is not None:
                        mocap_pos = np.asarray(data.mocap_pos[mocap_id_for_action]).reshape(3).copy()
                        mocap_quat = np.asarray(data.mocap_quat[mocap_id_for_action]).reshape(4).copy()
                        action_sample = np.concatenate([mocap_pos, mocap_quat, ctrl_sample], axis=0)
                    else:
                        action_sample = ctrl_sample
                    episode_buffers["/action"].append(action_sample)

                    for cam_name, cam_spec in recording_camera_specs:
                        try:
                            renderer.update_scene(data, camera=cam_spec)
                            img = renderer.render()
                            if img.dtype != np.uint8:
                                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                            episode_buffers[f"/observations/images/{cam_name}"].append(img)
                        except Exception as err:
                            logger.warning(f"Render camera {cam_name} failed: {err}")
                last_control_time = now

            # Keep simulation wall-clock synchronized even when startup keyframe
            # initializes data.time to a non-zero value.
            target_sim_time = sim_time_start + (now - sim_start)
            while data.time < target_sim_time:
                mujoco.mj_step(model, data)

            pending_buffers = None
            pending_episode_idx = None
            pending_object_info = None
            do_randomize = False
            with keyboard_lock:
                if randomize_requested:
                    if episode_buffers is not None:
                        logger.info(
                            f"Reset requested; discarded unsaved episode_{episode_idx}"
                        )
                    episode_buffers = None
                    is_recording = False
                    record_start_requested = False
                    record_stop_requested = False
                    assist_near_object_pending = False
                    translation_tracking_active = False
                    translation_engage_pending = False
                    translation_track_key_down = False
                    runtime_cfg.simulation.root_position_offset[0] = float(initial_root_position_offset[0])
                    runtime_cfg.simulation.root_position_offset[1] = float(initial_root_position_offset[1])
                    runtime_cfg.simulation.root_position_offset[2] = float(initial_root_position_offset[2])
                    do_randomize = True
                    randomize_requested = False
                elif record_start_requested:
                    episode_buffers = init_episode_buffers()
                    episode_object_info = latest_object_info
                    is_recording = True
                    record_start_requested = False
                    logger.info(f"Started recording episode_{episode_idx}")
                if record_stop_requested:
                    if episode_buffers is not None:
                        pending_buffers = episode_buffers
                        pending_episode_idx = episode_idx
                        pending_object_info = episode_object_info
                        episode_idx += 1
                        episode_buffers = None
                    is_recording = False
                    record_stop_requested = False
            if pending_buffers is not None and pending_episode_idx is not None:
                save_episode(pending_buffers, pending_episode_idx, pending_object_info)
            if do_randomize:
                _, shape_info = _reset_randomized_env(
                    model, data, runtime_cfg, force_shape_name=force_shape
                )
                if shape_info is not None:
                    latest_object_info = shape_info

            _publish_viewer_state(
                secondary_viewer_queues, _make_viewer_state_snapshot(model, data)
            )
            viewer.sync()
    except KeyboardInterrupt:
        logger.info("Ctrl-C received; shutting down and cleaning up")
    finally:
        # Ensure cleanup on any exit path
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
            if state_publisher is not None:
                state_publisher.close()
        except Exception:
            pass

        try:
            viewer.close()
        except Exception:
            pass

        for stop_event in secondary_viewer_stop_events:
            stop_event.set()
        for viewer_process in secondary_viewer_processes:
            try:
                viewer_process.join(timeout=1.0)
                if viewer_process.is_alive():
                    viewer_process.terminate()
                    viewer_process.join()
            except Exception:
                pass

        # Extra glfw.terminate() avoids repeated GLFW cleanup warnings after Ctrl-C on some setups
        try:
            import glfw  # type: ignore

            glfw.terminate()
        except Exception:
            pass

        logger.info("Exited")


if __name__ == "__main__":
    tyro.cli(main)
