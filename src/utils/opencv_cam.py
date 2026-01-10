import cv2
from typing import Optional


def get_camera_info(camera_id):
    """获取相机的详细信息"""
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
) -> Optional[int]:
    """
    查找支持指定分辨率的相机并返回其 ID。
    
    该函数会遍历可用的相机，尝试设置目标分辨率，并验证是否成功。
    如果找到支持该分辨率的相机，返回第一个匹配的相机 ID。
    
    Args:
        target_width: 目标宽度，默认为 1280
        target_height: 目标高度，默认为 720
        max_cameras: 最大检测相机数量，默认为 10
    
    Returns:
        如果找到支持该分辨率的相机，返回相机 ID (int)
        如果未找到，返回 None
    """
    for i in range(max_cameras):
        info = get_camera_info(i)
        if info:
            print(info)
            if info["width"] == target_width and info["height"] == target_height:
                return info["id"]
        else:
            break
    
    return None

if __name__ == "__main__":
    camera_id = find_camera_with_resolution(target_width=1280, target_height=720)
    print(camera_id)
