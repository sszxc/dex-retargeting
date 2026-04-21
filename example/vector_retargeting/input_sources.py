from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import leap
import numpy as np
from loguru import logger

from leap_motion_detector import LeapMotionHandDetector
from single_hand_detector import SingleHandDetector


@dataclass
class HandObservation:
    joint_pos: Optional[np.ndarray]
    wrist_rot: Optional[np.ndarray]
    keypoint_3d: Optional[np.ndarray]


def _normalize_hand_name(hand: str) -> str:
    if hand.lower() not in {"left", "right"}:
        raise ValueError(f"不支持的手类型: {hand}")
    return hand.lower()


def _to_detector_hand(hand: str) -> str:
    return "Left" if hand == "left" else "Right"


class WebcamInputSource:
    def __init__(
        self,
        active_hands: List[str],
        webcam_index: int,
        camera2table: np.ndarray,
    ):
        self.active_hands = [_normalize_hand_name(h) for h in active_hands]
        self.camera2table = np.asarray(camera2table, dtype=np.float64)
        self.cap = cv2.VideoCapture(webcam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开 webcam index={webcam_index}")
        self.detectors = {
            hand: SingleHandDetector(hand_type=_to_detector_hand(hand), selfie=False)
            for hand in self.active_hands
        }
        logger.info(f"输入源: webcam(index={webcam_index}), hands={self.active_hands}")

    def poll(self) -> Dict[str, HandObservation]:
        ok, bgr = self.cap.read()
        if not ok:
            time.sleep(1 / 60.0)
            return {hand: HandObservation(None, None, None) for hand in self.active_hands}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        out: Dict[str, HandObservation] = {}
        for hand, detector in self.detectors.items():
            _, joint_pos, _, wrist_rot, keypoint_3d = detector.detect(rgb)
            if joint_pos is not None:
                joint_pos = np.asarray(joint_pos) @ self.camera2table.T
            if keypoint_3d is not None:
                keypoint_3d = np.asarray(keypoint_3d) @ self.camera2table.T
            out[hand] = HandObservation(
                joint_pos=joint_pos,
                wrist_rot=wrist_rot,
                keypoint_3d=keypoint_3d,
            )
        return out

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        for detector in self.detectors.values():
            if hasattr(detector, "hand_detector"):
                try:
                    detector.hand_detector.close()
                except Exception:
                    pass


class LeapInputSource:
    def __init__(self, active_hands: List[str], camera2table: np.ndarray):
        self.active_hands = [_normalize_hand_name(h) for h in active_hands]
        self.detectors = {
            hand: LeapMotionHandDetector(
                hand_type=_to_detector_hand(hand),
                tracking_mode=leap.TrackingMode.Desktop,
                camera2table=np.asarray(camera2table, dtype=np.float64),
            )
            for hand in self.active_hands
        }
        logger.info(f"输入源: leap_motion, hands={self.active_hands}")

    def poll(self) -> Dict[str, HandObservation]:
        out: Dict[str, HandObservation] = {}
        for hand, detector in self.detectors.items():
            _, joint_pos, _, wrist_rot, keypoint_3d = detector.detect(None)
            out[hand] = HandObservation(
                joint_pos=joint_pos,
                wrist_rot=wrist_rot,
                keypoint_3d=keypoint_3d,
            )
        return out

    def close(self) -> None:
        for detector in self.detectors.values():
            detector.close()


def generate_sine_test_qpos(
    dim_idx: int,
    progress: float,
    total_dim: int = 22,
    cycle_count: int = 2,
) -> np.ndarray:
    angle = (2 * cycle_count) * np.pi * progress
    out = np.zeros(total_dim, dtype=np.float32)
    out[dim_idx] = np.sin(angle)
    return out
