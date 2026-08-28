import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "vector_retargeting"
sys.path.insert(0, str(EXAMPLE_DIR))

from runtime_config import (  # noqa: E402
    MocapConfig,
    SimulationConfig,
    _parse_simulation,
)


def _simulation(**overrides) -> SimulationConfig:
    values = {
        "mj_xml_path": "unused.xml",
        "mocap": MocapConfig(wrist_mocap=True, mocap_body_name="mocap"),
        "fingertip_body_names": ["tip_a", "tip_b"],
        "fingertips_pos_min": [-1.0, -2.0, -3.0],
        "fingertips_pos_max": [1.0, 2.0, 3.0],
    }
    values.update(overrides)
    return SimulationConfig(**values)


def test_fingertip_bounds_parse_and_validate() -> None:
    simulation = _parse_simulation(
        {
            "mj_xml_path": "unused.xml",
            "fingertip_body_names": [" tip_a ", "tip_b"],
            "fingertips_pos_min": [-1, -2, -3],
            "fingertips_pos_max": [1, 2, 3],
            "mocap": {"wrist_mocap": True, "mocap_body_name": "mocap"},
        }
    )

    simulation.validate()
    assert simulation.fingertip_body_names == ["tip_a", "tip_b"]
    assert simulation.fingertips_pos_min == [-1.0, -2.0, -3.0]
    assert simulation.fingertips_pos_max == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"fingertips_pos_max": None},
            "must be set together",
        ),
        (
            {"fingertip_body_names": []},
            "non-empty body names",
        ),
        (
            {"fingertip_body_names": ["tip_a", ""]},
            "non-empty body names",
        ),
        (
            {"fingertip_body_names": ["tip_a", "tip_a"]},
            "must not contain duplicates",
        ),
        (
            {"fingertip_body_names": ["tip_a", 2]},
            "must contain only strings",
        ),
        (
            {"fingertips_pos_min": [2.0, -2.0, -3.0]},
            "must be <= fingertips_pos_max",
        ),
        (
            {"fingertips_pos_min": [float("nan"), -2.0, -3.0]},
            "must be finite",
        ),
        (
            {"mocap": MocapConfig(wrist_mocap=False)},
            "wrist_mocap=true",
        ),
    ],
)
def test_fingertip_bounds_validation_errors(overrides, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _simulation(**overrides).validate()


def test_fingertip_body_names_must_parse_from_a_list() -> None:
    with pytest.raises(ValueError, match="must be a list of body names"):
        _parse_simulation(
            {
                "mj_xml_path": "unused.xml",
                "fingertip_body_names": "tip_a",
                "fingertips_pos_min": [-1, -1, -1],
                "fingertips_pos_max": [1, 1, 1],
                "mocap": {"wrist_mocap": True},
            }
        )


def test_omitted_fingertip_bounds_remain_disabled() -> None:
    simulation = SimulationConfig(mj_xml_path="unused.xml")

    simulation.validate()
    assert simulation.fingertip_body_names is None
    assert simulation.fingertips_pos_min is None
    assert simulation.fingertips_pos_max is None
