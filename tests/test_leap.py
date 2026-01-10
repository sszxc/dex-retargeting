import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import struct

# note that 384x384 resolution will only work on Python 3.6 and opencv-python==4.1.1.26


cam = cv2.VideoCapture(0)
w = 1024
h = 1024
cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cam.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # NB: this needs to work!
once = True
frame_count_bright = 0
frame_count_dark = 0
while True:

    r, frame = cam.read()
    frame = np.reshape(frame, (h, int(frame.size / h)))
    embedded_line = frame[-1, :68]
    embedded_params = struct.unpack("IIIIHHHHIQIIIIIII", embedded_line.tobytes())
    frame_label = embedded_params[1]
    if once:
        once = False
        print(frame.shape)
    frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame_label == 0:
        frame_count_bright = frame_count_bright + 1
        cv2.putText(
            frame,
            f"bright frame count: {frame_count_bright}",
            (5, frame.shape[0] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
        )
        cv2.waitKey(1)
        cv2.imshow("bright frames", frame)
    else:
        frame_count_dark = frame_count_dark + 1
        cv2.putText(
            frame,
            f"dark frame count: {frame_count_dark}",
            (5, frame.shape[0] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
        )
        cv2.waitKey(1)
        cv2.imshow("dark frames", frame)
