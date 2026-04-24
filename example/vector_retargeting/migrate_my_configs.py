from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a dict YAML root")
    return data


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def _infer_hand_mode(stem: str) -> str:
    s = stem.lower()
    if "left" in s and "right" not in s:
        return "single_left"
    if "right" in s and "left" not in s:
        return "single_right"
    return "single_left"


def _normalize_optimizer_type(t: Any) -> str:
    if t is None:
        raise ValueError("retargeting.type is missing")
    s = str(t).strip().lower()
    if s == "dex":
        return "dexpilot"
    return s


def convert_one(
    old_cfg: Dict[str, Any],
    *,
    input_source: str,
    webcam_index: int,
    mj_xml_path: str,
    camera2table: list,
) -> Dict[str, Any]:
    if "retargeting" not in old_cfg or not isinstance(old_cfg["retargeting"], dict):
        raise ValueError("Legacy config missing retargeting section")
    r = dict(old_cfg["retargeting"])

    opt_type = _normalize_optimizer_type(r.pop("type", None))
    urdf_path = r.pop("urdf_path", None)
    if urdf_path is None:
        raise ValueError("Legacy config missing retargeting.urdf_path")
    add_dummy_free_joint = bool(r.pop("add_dummy_free_joint", False))

    # Remaining keys become optimizer params (unchanged)
    optimizer_params = r

    left_cfg: Dict[str, Any] = {
        "urdf_path": urdf_path,
        "add_dummy_free_joint": add_dummy_free_joint,
        "optimizer": {
            "type": opt_type,
            "params": optimizer_params,
        },
    }
    right_cfg: Dict[str, Any] = dict(left_cfg)

    new_cfg: Dict[str, Any] = {
        "sensor": {
            "input_source": input_source,
            "webcam": {"index": webcam_index},
            "camera2table": camera2table,
            "rerun_enabled": False,
        },
        "retargeting": {
            "mode": "single_left",  # Overridden from filename by caller
            "left": left_cfg,
            "right": right_cfg,
        },
        "simulation": {
            "mj_xml_path": mj_xml_path,
            "control_hand": "left",
            "root_ctrl_indices": [0, 1, 2, 3, 4, 5],
            "finger_ctrl_indices": [
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                10,
                11,
                12,
                13,
                6,
                7,
                8,
                9,
            ],
            "root_position_offset": [0.2, 0.0, -0.6],
            "wrist_rotation_calib_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "joint_indices": list(range(22)),
            "camera_names": [],
            "control_rate_hz": 60.0,
            "mocap": {"wrist_mocap": False, "mocap_body_name": None, "mocap_id": None},
        },
    }
    return new_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=str,
        default="src/dex_retargeting/configs/my",
        help="Legacy config directory (*.yml only)",
    )
    ap.add_argument(
        "--output-suffix",
        type=str,
        default="_runtime.yml",
        help="Output filename suffix (default *_runtime.yml)",
    )
    ap.add_argument(
        "--input-source",
        type=str,
        default="leap_motion",
        choices=["webcam", "leap_motion", "test_sine"],
    )
    ap.add_argument("--webcam-index", type=int, default=0)
    ap.add_argument(
        "--mj-xml-path",
        type=str,
        default="/home/lab/Documents/teleop_scene_left_077_rubiks_cube",
    )
    args = ap.parse_args()

    camera2table = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]

    in_dir = Path(args.input_dir).expanduser().resolve()
    if not in_dir.exists():
        raise FileNotFoundError(in_dir)

    ymls = sorted(in_dir.glob("*.yml"))
    if not ymls:
        print(f"No .yml files in: {in_dir}")
        return

    count = 0
    for yml in ymls:
        if yml.name.endswith(args.output_suffix):
            continue
        old_cfg = _load_yaml(yml)
        new_cfg = convert_one(
            old_cfg,
            input_source=args.input_source,
            webcam_index=args.webcam_index,
            mj_xml_path=args.mj_xml_path,
            camera2table=camera2table,
        )
        new_cfg["retargeting"]["mode"] = _infer_hand_mode(yml.stem)
        if new_cfg["retargeting"]["mode"] == "single_right":
            new_cfg["simulation"]["control_hand"] = "right"

        out_path = yml.with_name(f"{yml.stem}{args.output_suffix}")
        _dump_yaml(out_path, new_cfg)
        count += 1
        print(f"Wrote {out_path}")

    print(f"Done. generated={count}")


if __name__ == "__main__":
    main()

