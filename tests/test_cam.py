import cv2
import sys


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


def find_available_cameras(max_cameras=10):
    """Probe indices 0..max_cameras-1 and list working cameras."""
    available_cameras = []
    print("Scanning for cameras...")
    print("-" * 60)

    for i in range(max_cameras):
        info = get_camera_info(i)
        if info:
            available_cameras.append(info)
            print(f"Camera ID: {info['id']}")
            print(f"  Resolution: {info['width']} x {info['height']}")
            print(f"  FPS: {info['fps']:.2f}")
            print(f"  Backend: {info['backend']}")
            print("-" * 60)
        else:
            print(f"Camera ID: {i} unavailable")
            break

    return available_cameras


def main():
    cameras = find_available_cameras()

    if not cameras:
        print("No cameras found.")
        sys.exit(1)

    print(f"\nFound {len(cameras)} camera(s)")

    print("\nEnter camera ID to open (Enter for first camera): ", end="")
    try:
        user_input = input().strip()
        if user_input == "":
            selected_id = cameras[0]["id"]
            print(f"Using default camera (ID: {selected_id})...\n")
        else:
            selected_id = int(user_input)
            if not any(cam["id"] == selected_id for cam in cameras):
                print(f"Error: camera ID {selected_id} is not available.")
                print(f"Available IDs: {[cam['id'] for cam in cameras]}")
                sys.exit(1)
            print(f"Opening camera (ID: {selected_id})...\n")
    except ValueError:
        print("Error: enter a valid integer.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled")
        sys.exit(0)

    cap = cv2.VideoCapture(selected_id)

    if not cap.isOpened():
        print(f"Failed to open camera ID: {selected_id}")
        sys.exit(1)

    selected_camera = next(cam for cam in cameras if cam["id"] == selected_id)

    print("Press 'q' to quit")
    print("Live preview...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            break

        info_text = (
            f"Camera {selected_camera['id']} | {selected_camera['width']}x{selected_camera['height']}"
        )
        cv2.putText(
            frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.imshow("Camera View", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exited")


if __name__ == "__main__":
    main()
