import cv2
from typing import Optional


def get_camera_info(camera_id):
    """Return detailed info for a camera index."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return None

    info = {
        "id": camera_id,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "backend": cap.getBackendName(),
    }
    cap.release()
    return info


def find_camera_with_resolution(
    target_width: int = 1280,
    target_height: int = 720,
    max_cameras: int = 10,
    default_camera_id: int = 0,
) -> Optional[int]:
    """
    Find a camera that reports the requested resolution and return its ID.

    Scans camera indices, checks reported width/height, and returns the first match.

    Args:
        target_width: Desired width (default 1280).
        target_height: Desired height (default 720).
        max_cameras: Max camera indices to probe (default 10).

    Returns:
        Camera ID (int) if a match is found; otherwise default_camera_id.
    """
    for i in range(max_cameras):
        info = get_camera_info(i)
        if info:
            print(info)
            if info["width"] == target_width and info["height"] == target_height:
                return info["id"]
        else:
            break
    
    return default_camera_id

if __name__ == "__main__":
    camera_id = find_camera_with_resolution(target_width=1280, target_height=720)
    print(camera_id)
