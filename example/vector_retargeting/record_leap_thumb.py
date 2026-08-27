"""
Record raw Leap Motion joint_pos (21 MediaPipe-style keypoints) over time, so a
new per-joint angle formula (e.g. for a[10], the thumb's first/CMC joint) can be
designed from real motion data instead of guessing.

Saves ONLY the raw (T, 21, 3) keypoints + per-frame timestamps to an .npz file.
That's enough to recompute x_dir/y_dir/z_dir and try out any candidate formula
offline afterwards (see optimizer.py: hmf_proto5_left_dummy_qpos_from_leap_joint_pos),
since those are all derived purely from joint_pos.

Usage:
    python example/vector_retargeting/record_leap_thumb.py --hand Right --duration 20

Suggested motion pattern while it runs (helps tell segments apart later just by
looking at the timestamps/keypoints, no extra markers needed):
    1. ~2s: hold hand still (rest baseline)
    2. ~2s: move ONLY the thumb's first/CMC joint (sweep it across the palm),
       try to keep MCP/IP/tip relatively straight
    3. ~2s: hold still again
    4. ~2s: for contrast, move ONLY the thumb's MCP/IP joints (curl the tip),
       keep the CMC still
    5. ~2s: hold still to finish

Ctrl-C stops early and still saves whatever was captured.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

from leap_motion_detector import LeapMotionHandDetector


# Same default used by LeapInputSource in input_sources.py / the runtime configs.
DEFAULT_CAMERA2TABLE = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)


def main(
    hand: str = "Right",
    duration: float = 20.0,
    hz: float = 30.0,
    out: str = "",
):
    """Record raw Leap joint_pos keypoints to an .npz file.

    Args:
        hand: "Right" or "Left".
        duration: seconds to record (Ctrl-C stops early and still saves).
        hz: sampling rate.
        out: output .npz path; default: data/leap_records/leap_<hand>_<timestamp>.npz
    """
    detector = LeapMotionHandDetector(hand_type=hand, camera2table=DEFAULT_CAMERA2TABLE)

    if not out:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path("data/leap_records") / f"leap_{hand.lower()}_{ts}.npz"
    else:
        out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Recording {hand} hand for up to {duration:.1f}s @ {hz:.0f}Hz -> {out_path}")
    logger.info("Follow the motion pattern in the file docstring; Ctrl-C to stop early.")

    timestamps: list[float] = []
    all_joint_pos: list[np.ndarray] = []

    period = 1.0 / hz
    t_start = time.time()
    try:
        while time.time() - t_start < duration:
            loop_t0 = time.time()
            num_hands, joint_pos, _, _, _ = detector.detect()
            if num_hands and joint_pos is not None:
                t = loop_t0 - t_start
                timestamps.append(t)
                all_joint_pos.append(joint_pos.copy())
                # Live sanity check: thumb keypoints are indices 1(CMC),2(MCP),3(IP),4(Tip).
                cmc, tip = joint_pos[1], joint_pos[4]
                print(
                    f"\rt={t:5.1f}s  n={len(timestamps):4d}  "
                    f"thumb_cmc=({cmc[0]:+.3f},{cmc[1]:+.3f},{cmc[2]:+.3f})  "
                    f"thumb_tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f})",
                    end="",
                    flush=True,
                )
            else:
                print(f"\rt={loop_t0 - t_start:5.1f}s  (no hand detected)", end="", flush=True)
            elapsed = time.time() - loop_t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        logger.info("Stopped early by Ctrl-C")
    finally:
        detector.close()

    print()
    if not all_joint_pos:
        logger.warning("No frames captured (hand not detected?) -- nothing saved.")
        return

    joint_pos_arr = np.stack(all_joint_pos, axis=0)  # (T, 21, 3)
    ts_arr = np.asarray(timestamps)
    np.savez(out_path, timestamps=ts_arr, joint_pos=joint_pos_arr)
    logger.info(f"Saved {joint_pos_arr.shape[0]} frames to {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
