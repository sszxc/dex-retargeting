import sys
from pathlib import Path

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "vector_retargeting"
sys.path.insert(0, str(EXAMPLE_DIR))

from mujoco_control import (  # noqa: E402
    MujocoHandController,
    _project_fingertip_translation,
)
from runtime_config import MocapConfig, SimulationConfig  # noqa: E402


MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="mocap" mocap="true"/>
    <body name="palm">
      <freejoint/>
      <geom type="sphere" size="0.01" mass="0.1"/>
      <body name="finger_a">
        <joint name="finger_a_joint" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.01" mass="0.01"/>
        <body name="tip_a" pos="0.2 0 0"/>
      </body>
      <body name="finger_b">
        <joint name="finger_b_joint" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.01" mass="0.01"/>
        <body name="tip_b" pos="-0.2 0 0"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <weld body1="mocap" body2="palm"/>
  </equality>
  <actuator>
    <position joint="finger_a_joint" kp="10"/>
    <position joint="finger_b_joint" kp="10"/>
  </actuator>
</mujoco>
"""


def _model_and_data(xml: str = MODEL_XML):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _simulation(**overrides) -> SimulationConfig:
    values = {
        "mj_xml_path": "unused.xml",
        "control_hand": "right",
        "finger_ctrl_indices": [0, 1],
        "root_position_offset": [0.0, 0.0, 0.0],
        "mocap": MocapConfig(wrist_mocap=True, mocap_body_name="mocap"),
        "fingertip_body_names": ["tip_a", "tip_b"],
        "fingertips_pos_min": [0.0, -1.0, -1.0],
        "fingertips_pos_max": [1.0, 1.0, 1.0],
    }
    values.update(overrides)
    simulation = SimulationConfig(**values)
    simulation.validate()
    return simulation


def _message(wrist_pos=(-0.5, 0.0, 0.0), fingers=(0.0, 0.0)) -> dict:
    return {
        "hand_right_qpos": np.concatenate([np.zeros(6), np.asarray(fingers)]),
        "wrist_right_pos": np.asarray(wrist_pos, dtype=np.float64),
        "wrist_right_quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }


def test_projection_applies_minimum_multi_tip_and_wrist_correction() -> None:
    projected = _project_fingertip_translation(
        requested_pos=np.array([-2.0, 0.0, 0.0]),
        fingertip_offsets=np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]),
        fingertips_pos_min=np.array([0.0, -1.0, -1.0]),
        fingertips_pos_max=np.array([1.0, 1.0, 1.0]),
        wrist_pos_min=np.array([0.3, -2.0, -2.0]),
        wrist_pos_max=np.array([0.8, 2.0, 2.0]),
    )

    np.testing.assert_allclose(projected, [0.3, 0.0, 0.0])


def test_projection_returns_none_when_tip_spread_is_infeasible() -> None:
    projected = _project_fingertip_translation(
        requested_pos=np.zeros(3),
        fingertip_offsets=np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]),
        fingertips_pos_min=np.array([0.0, -1.0, -1.0]),
        fingertips_pos_max=np.array([0.1, 1.0, 1.0]),
    )

    assert projected is None


def test_projection_corrects_upper_bounds_on_each_axis() -> None:
    projected = _project_fingertip_translation(
        requested_pos=np.array([2.0, 3.0, 4.0]),
        fingertip_offsets=np.array([[0.2, 0.3, 0.4], [-0.2, -0.3, -0.4]]),
        fingertips_pos_min=np.array([-1.0, -1.0, -1.0]),
        fingertips_pos_max=np.array([1.0, 1.0, 1.0]),
    )

    np.testing.assert_allclose(projected, [0.8, 0.7, 0.6])


def test_prediction_uses_requested_finger_pose_and_wrist_rotation() -> None:
    model, data = _model_and_data()
    controller = MujocoHandController(_simulation(), model)
    rotate_z_90 = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    offsets = controller._predict_fingertip_offsets(
        data,
        finger_values=np.array([np.pi / 2.0, 0.0]),
        desired_wrist_rotation=rotate_z_90,
    )

    np.testing.assert_allclose(offsets[0], [0.0, 0.0, -0.2], atol=1e-8)
    np.testing.assert_allclose(offsets[1], [0.0, -0.2, 0.0], atol=1e-8)


def test_controller_shifts_mocap_target_and_commits_fingers() -> None:
    model, data = _model_and_data()
    controller = MujocoHandController(_simulation(), model)

    controller.apply(data, _message())

    np.testing.assert_allclose(data.mocap_pos[0], [0.2, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(data.ctrl, [0.0, 0.0])


def test_bounds_override_translation_tracking_freeze() -> None:
    model, data = _model_and_data()
    controller = MujocoHandController(_simulation(), model)
    data.mocap_pos[0] = [-0.5, 0.0, 0.0]

    controller.apply(data, _message(wrist_pos=(0.8, 0.0, 0.0)), track_translation=False)

    np.testing.assert_allclose(data.mocap_pos[0], [0.2, 0.0, 0.0], atol=1e-8)


def test_infeasible_frame_preserves_all_live_commands() -> None:
    model, data = _model_and_data()
    simulation = _simulation(fingertips_pos_max=[0.1, 1.0, 1.0])
    controller = MujocoHandController(simulation, model)
    data.ctrl[:] = [0.3, 0.4]
    data.mocap_pos[0] = [0.7, 0.1, 0.2]
    old_quat = data.mocap_quat[0].copy()

    controller.apply(data, _message(fingers=(0.5, -0.5)))

    np.testing.assert_allclose(data.ctrl, [0.3, 0.4])
    np.testing.assert_allclose(data.mocap_pos[0], [0.7, 0.1, 0.2])
    np.testing.assert_allclose(data.mocap_quat[0], old_quat)


def test_missing_fingertip_body_fails_at_controller_initialization() -> None:
    model, _ = _model_and_data()
    simulation = _simulation(fingertip_body_names=["tip_a", "missing_tip"])

    with pytest.raises(ValueError, match="missing_tip"):
        MujocoHandController(simulation, model)


def test_missing_weld_partner_fails_at_controller_initialization() -> None:
    xml_without_weld = MODEL_XML.replace(
        '<weld body1="mocap" body2="palm"/>', ""
    )
    model, _ = _model_and_data(xml_without_weld)

    with pytest.raises(ValueError, match="equality weld partner"):
        MujocoHandController(_simulation(), model)


def test_disabled_fingertip_bounds_keep_existing_wrist_clip() -> None:
    model, data = _model_and_data()
    simulation = _simulation(
        fingertip_body_names=None,
        fingertips_pos_min=None,
        fingertips_pos_max=None,
        wrist_pos_min=[-0.25, -1.0, -1.0],
        wrist_pos_max=[0.25, 1.0, 1.0],
    )
    controller = MujocoHandController(simulation, model)

    controller.apply(data, _message(wrist_pos=(-0.5, 0.0, 0.0), fingers=(0.2, 0.3)))

    np.testing.assert_allclose(data.mocap_pos[0], [-0.25, 0.0, 0.0])
    np.testing.assert_allclose(data.ctrl, [0.2, 0.3])
