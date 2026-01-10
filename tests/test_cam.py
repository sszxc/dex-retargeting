import cv2
import sys


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


def find_available_cameras(max_cameras=10):
    """查找所有可用的相机"""
    available_cameras = []
    print("正在检测可用相机...")
    print("-" * 60)

    for i in range(max_cameras):
        info = get_camera_info(i)
        if info:
            available_cameras.append(info)
            print(f"相机 ID: {info['id']}")
            print(f"  分辨率: {info['width']} x {info['height']}")
            print(f"  帧率: {info['fps']:.2f} FPS")
            print(f"  后端: {info['backend']}")
            print("-" * 60)
        else:
            print(f"相机 ID: {i} 不可用")
            break

    return available_cameras


def main():
    # 查找所有可用相机
    cameras = find_available_cameras()

    if not cameras:
        print("未找到任何可用相机！")
        sys.exit(1)

    print(f"\n找到 {len(cameras)} 个可用相机")
    
    # 让用户选择要打开的相机
    print("\n请输入要打开的相机 ID (直接回车将使用第一个相机): ", end="")
    try:
        user_input = input().strip()
        if user_input == "":
            selected_id = cameras[0]["id"]
            print(f"使用默认相机 (ID: {selected_id})...\n")
        else:
            selected_id = int(user_input)
            # 验证输入的相机ID是否可用
            if not any(cam["id"] == selected_id for cam in cameras):
                print(f"错误：相机 ID {selected_id} 不可用！")
                print(f"可用的相机 ID: {[cam['id'] for cam in cameras]}")
                sys.exit(1)
            print(f"正在打开相机 (ID: {selected_id})...\n")
    except ValueError:
        print("错误：请输入有效的数字！")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(0)

    # 打开用户选择的相机
    cap = cv2.VideoCapture(selected_id)

    if not cap.isOpened():
        print(f"无法打开相机 ID: {selected_id}")
        sys.exit(1)

    # 获取选中相机的信息
    selected_camera = next(cam for cam in cameras if cam["id"] == selected_id)

    print("按 'q' 键退出")
    print("实时画面显示中...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("无法读取画面")
            break

        # 在画面上显示信息
        info_text = (
            f"Camera {selected_camera['id']} | {selected_camera['width']}x{selected_camera['height']}"
        )
        cv2.putText(
            frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.imshow("Camera View", frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()
