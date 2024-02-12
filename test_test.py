import cv2
import mediapipe as mp
import time
import torch
import torch.nn as nn
import json
import subprocess
from HandTracker import HandTracker
import test_utils as utils
from FPS import FPS, now
import numpy as np
import mediapipe_utils as mpu
from run import run


class Model(nn.Module):
    def __init__(self, input_size, hidden_dim, tagset_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, landmarks):
        x = self.fc1(landmarks)
        x = self.fc2(x)
        res = self.softmax(x)
        return res


def initialize_model():
    INPUT_SIZE = 42
    HIDDEN_DIM = 32
    TARGET_SIZE = 6

    model = Model(INPUT_SIZE, HIDDEN_DIM, TARGET_SIZE)
    model.load_state_dict(torch.load("model2.pt"))

    return model


def initialize_mediapipe_hands():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Initialize MediaPipe Hands model
    hands = mp_hands.Hands(max_num_hands=5)
    ht = HandTracker(
        input_src="1",
        pd_xml="mediapipe_models/palm_detection_FP32.xml",
        pd_device="CPU",
        pd_score_thresh=0.5,
        pd_nms_thresh=0.3,
        use_lm=True,
        lm_xml="mediapipe_models/hand_landmark_FP32.xml",
        lm_device="CPU",
        lm_score_threshold=0.5,
        use_gesture=False,
        crop=False,
        is_getdata=False,
    )

    return hands, mp_drawing


def load_parameters(parameters_dir):
    res = input("want to use saved parameters? [ y / n ]\n>>> ")
    if res == "y":
        parameter = json.load(open(parameters_dir))
    else:
        parameter = {
            "time": 10,
            "same_hand": 10,
            "skip_frame": 1,
            "start_time": 1,
            "stop_time": 1,
            "multi_time": 1,
            "multi_cooltime": 2,
        }

    return parameter


def create_trackbars(parameter):
    cv2.namedWindow("gesture recognition")
    cv2.createTrackbar("time", "gesture recognition", parameter["time"], 100, utils.nothing)
    cv2.createTrackbar("same_hand", "gesture recognition", parameter["same_hand"], 100, utils.nothing)
    cv2.createTrackbar("skip_frame", "gesture recognition", parameter["skip_frame"], 50, utils.nothing)
    cv2.createTrackbar("start_time", "gesture recognition", parameter["start_time"], 10, utils.nothing)
    cv2.createTrackbar("stop_time", "gesture recognition", parameter["stop_time"], 10, utils.nothing)
    cv2.createTrackbar("multi_time", "gesture recognition", parameter["multi_time"], 10, utils.nothing)
    cv2.createTrackbar("multi_cooltime", "gesture recognition", parameter["multi_cooltime"], 10, utils.nothing)


def save_current_parameters(parameters_dir):
    res = input("want to save current parameters? [ y / n ]\n>>> ")
    if res == "y":
        with open(parameters_dir, "w") as f:
            json.dump(parameter, f)


if __name__ == "__main__":
    ht = HandTracker(
        input_src="1",
        pd_xml="mediapipe_models/palm_detection_FP32.xml",
        pd_device="CPU",
        pd_score_thresh=0.5,
        pd_nms_thresh=0.3,
        use_lm=True,
        lm_xml="mediapipe_models/hand_landmark_FP32.xml",
        lm_device="CPU",
        lm_score_threshold=0.5,
        use_gesture=False,
        crop=False,
        is_getdata=False,
    )

    model = initialize_model()

    run(ht, model)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Initialize MediaPipe Hands model
    # hands, mp_drawing = initialize_mediapipe_hands()

    # Open a webcam
    # w = 1280
    # h = 720
    # cap = cv2.VideoCapture(1)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    parameters_dir = "./parameters.json"
    # parameter = load_parameters(parameters_dir)
    # create_trackbars(parameter)

    # save paramters
    parameter = {
        "time": cv2.getTrackbarPos("time", "gesture recognition"),
        "same_hand": cv2.getTrackbarPos("same_hand", "gesture recognition"),
        "skip_frame": cv2.getTrackbarPos("skip_frame", "gesture recognition"),
        "start_time": cv2.getTrackbarPos("start_time", "gesture recognition"),
        "stop_time": cv2.getTrackbarPos("stop_time", "gesture recognition"),
        "multi_time": cv2.getTrackbarPos("multi_time", "gesture recognition"),
        "multi_cooltime": cv2.getTrackbarPos("multi_cooltime", "gesture recognition"),
    }

    # Release the webcam and close all windows
    ht.cap.release()
    cv2.destroyAllWindows()

    # print average inference time
    # print("landmark:", landmark_time / landmark_num)
    # print("gesture:", gesture_time / gesture_num)

    # save_current_parameters(parameters_dir)
