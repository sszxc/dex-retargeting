"""
Leap Motion hand detector.

Converts Leap Motion API hand data to a MediaPipe-compatible layout (21 keypoints).
"""
import numpy as np
import cv2
import leap
from leap import datatypes as ldt
from typing import Optional, Tuple
from timeit import default_timer as timer
import time
from typing import Callable

# MediaPipe-to-MANO coordinate transform (same convention as single_hand_detector.py)
OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


DEFAULT_CAMERA2TABLE = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def wait_until(condition: Callable[[], bool], timeout: float = 5, poll_delay: float = 0.01):
    """Block until ``condition()`` is true or ``timeout`` seconds elapse."""
    start_time = timer()
    while timer() - start_time < timeout:
        if condition():
            return True
        time.sleep(poll_delay)
    return False


class LeapMotionHandDetector:
    """
    Leap Motion hand detector.

    Uses the Leap API for 21 keypoints and can expose them in a MediaPipe-compatible order.
    """

    def __init__(
        self,
        hand_type: str = "Right",
        tracking_mode: leap.TrackingMode = leap.TrackingMode.Desktop,
        camera2table: Optional[np.ndarray] = None,
    ):
        """
        Args:
            hand_type: "Right" or "Left".
            tracking_mode: Desktop / HMD / ScreenTop.
        """
        self.hand_type = hand_type
        self.tracking_mode = tracking_mode
        self.camera2table = (
            np.asarray(camera2table, dtype=np.float64)
            if camera2table is not None
            else DEFAULT_CAMERA2TABLE
        )

        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )

        self.leap_hand_type = (
            leap.HandType.Right if hand_type == "Right" else leap.HandType.Left
        )

        self.connection = None
        self.listener = None
        self.latest_event = None
        self.connected = False
        self.connection_context = None

        class TrackingListener(leap.Listener):
            def __init__(self, detector):
                super().__init__()
                self.detector = detector

            def on_connection_event(self, event):
                self.detector.connected = True
                print("Leap Motion: Connected")

            def on_device_event(self, event):
                try:
                    with event.device.open():
                        info = event.device.get_info()
                except leap.LeapCannotOpenDeviceError:
                    info = event.device.get_info()
                print(f"Leap Motion: Found device {info.serial}")

            def on_tracking_event(self, event):
                self.detector.latest_event = event

        self.listener = TrackingListener(self)
        self.connection = leap.Connection()
        self.connection.add_listener(self.listener)

        # Keep connection open: connection.open() returns a context manager
        self.connection_context = self.connection.open()
        self.connection_context.__enter__()
        self.connection.set_tracking_mode(tracking_mode)

        wait_until(lambda: self.connected, timeout=5)
        print(f"Leap Motion: initialized, waiting for {hand_type} hand data...")

    def extract_keypoints_from_hand(self, hand: ldt.Hand) -> Optional[np.ndarray]:
        """
        Extract 21 keypoints from a Leap ``Hand`` (MediaPipe ordering).

        MediaPipe order:
        0: Wrist
        1–4: Thumb (CMC, MCP, IP, Tip)
        5–8: Index (MCP, PIP, DIP, Tip)
        9–12: Middle (MCP, PIP, DIP, Tip)
        13–16: Ring (MCP, PIP, DIP, Tip)
        17–20: Pinky (MCP, PIP, DIP, Tip)

        Leap layout:
        - hand.arm.next_joint: wrist
        - hand.digits[0–4]: five digits
        - digit.bones[0–3]: four bones per finger
        - bone.prev_joint / next_joint: bone endpoints

        Args:
            hand: Leap hand object.

        Returns:
            Array of shape (21, 3) with x,y,z in meters, or None on failure.
        """
        keypoints = np.zeros((21, 3))

        try:
            if hand.arm and hand.arm.next_joint:
                wrist = hand.arm.next_joint
                keypoints[0] = [wrist.x, wrist.y, wrist.z]
            else:
                palm = hand.palm.position
                keypoints[0] = [palm.x, palm.y, palm.z]

            for digit_idx in range(5):
                digit = hand.digits[digit_idx]

                # Four bones per finger map to four MediaPipe joints
                # Bone 0: palm to MCP; 1: MCP–PIP; 2: PIP–DIP; 3: DIP–tip

                if len(digit.bones) >= 4:
                    if digit.bones[0].prev_joint:
                        mcp = digit.bones[0].prev_joint
                        keypoint_idx = 1 + digit_idx * 4
                        keypoints[keypoint_idx] = [mcp.x, mcp.y, mcp.z]

                    if digit.bones[1].prev_joint:
                        pip = digit.bones[1].prev_joint
                        keypoint_idx = 2 + digit_idx * 4
                        keypoints[keypoint_idx] = [pip.x, pip.y, pip.z]

                    if digit.bones[2].prev_joint:
                        dip = digit.bones[2].prev_joint
                        keypoint_idx = 3 + digit_idx * 4
                        keypoints[keypoint_idx] = [dip.x, dip.y, dip.z]

                    if digit.bones[3].next_joint:
                        tip = digit.bones[3].next_joint
                        keypoint_idx = 4 + digit_idx * 4
                        keypoints[keypoint_idx] = [tip.x, tip.y, tip.z]
                else:
                    if hasattr(digit, "distal") and digit.distal and digit.distal.next_joint:
                        tip = digit.distal.next_joint
                        keypoint_idx = 4 + digit_idx * 4
                        keypoints[keypoint_idx] = [tip.x, tip.y, tip.z]

            keypoints = keypoints / 1000.0

            return keypoints

        except Exception as e:
            print(f"Error extracting keypoints: {e}")
            return None

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Estimate wrist orientation (3×3 rotation) from keypoints; same idea as single_hand_detector.

        Args:
            keypoint_3d_array: (21, 3) keypoints.

        Returns:
            Rotation matrix (3, 3).
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        x_vector = points[0] - points[2]

        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

    def detect(self, rgb: Optional[np.ndarray] = None) -> Tuple[int, Optional[np.ndarray], Optional[object], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect hand keypoints (API-compatible with SingleHandDetector).

        Args:
            rgb: Unused for Leap; kept for a uniform interface.

        Returns:
            (num_hands, joint_pos 21×3, keypoint_2d placeholder, wrist_rot, keypoint_3d_for_vis).
        """
        if self.latest_event is None:
            return 0, None, None, None, None

        target_hand = None
        for hand in self.latest_event.hands:
            if hand.type == self.leap_hand_type:
                target_hand = hand
                break

        if target_hand is None:
            return 0, None, None, None, None

        keypoint_3d_global = self.extract_keypoints_from_hand(target_hand)
        if keypoint_3d_global is None:
            return 0, None, None, None, None

        keypoint_3d_global = keypoint_3d_global @ self.camera2table.T

        keypoint_3d_for_vis = keypoint_3d_global.copy()

        keypoint_3d_relative = keypoint_3d_global - keypoint_3d_global[0:1, :]

        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_relative)

        joint_pos = keypoint_3d_global

        return 1, joint_pos, None, mediapipe_wrist_rot, keypoint_3d_for_vis

    def project_3d_to_2d(self, keypoint_3d: np.ndarray, image_size: tuple = (720, 1280)) -> np.ndarray:
        """
        Project 3D keypoints to 2D (visualiser-style: x and z → image plane, y as depth).

        Args:
            keypoint_3d: (21, 3) in meters.
            image_size: (height, width).

        Returns:
            (21, 2) integer pixel coordinates.
        """
        keypoint_3d = keypoint_3d @ self.camera2table
        keypoints_mm = keypoint_3d * 1000.0

        center_x = image_size[1] / 2
        center_y = image_size[0] / 2

        keypoint_2d = np.zeros((21, 2))
        keypoint_2d[:, 0] = keypoints_mm[:, 0] + center_x
        keypoint_2d[:, 1] = keypoints_mm[:, 2] + center_y

        return keypoint_2d.astype(int)

    def draw_skeleton_on_image(
        self, image: np.ndarray, keypoint_3d: Optional[np.ndarray], style: str = "default"
    ) -> np.ndarray:
        """
        Draw hand skeleton from 3D keypoints projected to 2D.

        Args:
            image: BGR image (or None to allocate a blank canvas).
            keypoint_3d: (21, 3) or None.
            style: "default" or "white".

        Returns:
            Image with overlay.
        """
        if keypoint_3d is None:
            if image is None:
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
            return image

        if image is None:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)

        keypoint_2d = self.project_3d_to_2d(keypoint_3d, image.shape[:2])

        connections = [
            (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
            (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
        ]

        if style == "default":
            line_color = (0, 255, 0)
            point_color = (0, 0, 255)
            line_thickness = 2
            point_radius = 4
        else:
            line_color = (255, 255, 255)
            point_color = (255, 48, 48)
            line_thickness = 2
            point_radius = 4

        for start_idx, end_idx in connections:
            start = tuple(keypoint_2d[start_idx])
            end = tuple(keypoint_2d[end_idx])
            if (0 <= start[0] < image.shape[1] and 0 <= start[1] < image.shape[0] and
                0 <= end[0] < image.shape[1] and 0 <= end[1] < image.shape[0]):
                cv2.line(image, start, end, line_color, line_thickness)

        for i, point in enumerate(keypoint_2d):
            point_tuple = tuple(point)
            if 0 <= point_tuple[0] < image.shape[1] and 0 <= point_tuple[1] < image.shape[0]:
                cv2.circle(image, point_tuple, point_radius, point_color, -1)
                if i == 0:
                    cv2.circle(image, point_tuple, point_radius + 2, point_color, -1)

        cv2.putText(
            image,
            f"Leap Motion - {self.hand_type} Hand",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if hasattr(self, "latest_event") and self.latest_event is not None:
            frame_id_text = f"Frame: {self.latest_event.tracking_frame_id}"
            cv2.putText(
                image,
                frame_id_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        return image

    def close(self):
        """Close the Leap Motion connection."""
        if self.connection_context:
            try:
                self.connection_context.__exit__(None, None, None)
            except Exception as e:
                print(f"Error closing Leap Motion connection: {e}")
            self.connection_context = None
        if self.connection:
            self.connection = None
