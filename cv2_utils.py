import cv2
import numpy as np

from openvino_utils.fps import FPS


class CV2Utils:
    def __init__(self):
        self.w = 1280
        self.h = 720
        self.cap = cv2.VideoCapture(0 , cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

        self.fps = FPS(mean_nb_frames=20)

    def read(self):
        return self.cap.read()

    def unpad(self, frame):
        frame_size = max(self.h, self.w)
        pad_h = int((frame_size - self.h) / 2)
        pad_w = int((frame_size - self.w) / 2)

        return frame[pad_h: pad_h + self.h, pad_w: pad_w + self.w]

    def annotated_frame(self, frame):
        h, w = frame.shape[:2]

        # Padding on the small side to get a square shape
        frame_size = max(h, w)
        pad_h = int((frame_size - h) / 2)
        pad_w = int((frame_size - w) / 2)
        frame = cv2.copyMakeBorder(
            frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT
        )
        return frame.copy()

    def print_landmark(self, frame, landmark, color=(0, 255, 0)):
        list_connections = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [5, 9, 10, 11, 12],
            [9, 13, 14, 15, 16],
            [13, 17],
            [0, 17, 18, 19, 20],
        ]
        lines = [
            np.array([landmark[point] for point in line])
            for line in list_connections
        ]
        cv2.polylines(frame, lines, False, (255, 0, 0), 2, cv2.LINE_AA)
        for x, y in landmark:
            cv2.circle(frame, (x, y), 6, color, -1)

        return frame
