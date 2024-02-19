import cv2
import torch
import torch.nn as nn
import json

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from MediaPipe.HandTracker import HandTracker
import test_utils as utils
from run import run


class Model(nn.Module):
    def __init__(self, input_size, hidden_dim1, hidden_dim2, target_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, target_size)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim = 0)
        self.relu = nn.ReLU()
    def forward(self, landmarks):
        x = self.dropout1(self.relu(self.fc1(landmarks)))
        x = self.dropout2(self.relu(self.fc2(x)))
        res = self.softmax(self.fc3(x))
        return res


def initialize_model():
    INPUT_SIZE = 42
    HIDDEN_DIM1 = 32
    HIDDEN_DIM2 = 32
    TARGET_SIZE = 8

    model = Model(INPUT_SIZE, HIDDEN_DIM1, HIDDEN_DIM2, TARGET_SIZE)
    model.load_state_dict(torch.load("../model.pt"))

    return model


def load_parameters(parameters_dir):
    res = "y" # input("want to use saved parameters? [ y / n ]\n>>> ")
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
    # Get the directory of test.py
    current_dir = os.path.dirname(os.path.relpath(__file__))

    # Construct the path to palm_detection.xml
    pd_model_path = os.path.join(current_dir, "..", "MediaPipe", "mediapipe_models", "palm_detection_FP16.xml")
    lm_model_path = os.path.join(current_dir, "..", "MediaPipe", "mediapipe_models", "hand_landmark_FP16.xml")

    ht = HandTracker(
        input_src="0",
        pd_xml=pd_model_path,
        pd_device="CPU",
        pd_score_thresh=0.6,
        pd_nms_thresh=0.3,
        use_lm=True,
        lm_xml=lm_model_path,
        lm_device="CPU",
        lm_score_threshold=0.6,
        crop=False,
        is_getdata=False,
    )

    model = initialize_model()

    parameters_dir = "./parameters.json"
    parameter = load_parameters(parameters_dir)
    create_trackbars(parameter)

    run(ht, model)

    # Release the webcam and close all windows
    ht.cap.release()
    cv2.destroyAllWindows()

    save_current_parameters(parameters_dir)
