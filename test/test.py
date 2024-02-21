import json
import os
import subprocess
import sys
import time
from multiprocessing import Process

import cv2
import torch
import yaml

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 

import utils
from cv2_utils import CV2Utils
from models.model import Model
from openvino_utils.hand_tracker import HandTracker

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def run(hand_tracker, model, cv2_util):
    gestures = config["gestures"]
    gesture_num = 0

    state = {
        "gesture": 5,
        "start_time": time.time(),
        "prev_gesture": 5,
        "multi_action_start_time": -1,
        "multi_action_cnt": 0,
        "prev_action": ["", 0],
    }

    frame_num = 0

    subprocess.run(
        "adb connect 192.168.1.103:5555; adb root; adb connect 192.168.1.103:5555",
        shell=True,
        check=False,
    )

    landmark_num = 0
    gesture_time = 0
    gesture_num = 0

    recognized_hands = []
    recognized_hand = []
    text_a = ""

    recognizing = False
    recognized_hand_prev_pos = [-999, -999]

    last_hand_time = time.time()

    wake_up_state = []

    while True:
        cv2_util.fps.update()
        # require more time than time_threshold to recognize it as an gesture
        time_threshold = (
            cv2.getTrackbarPos("time", "gesture recognition") / 100
        )
        # distance between this frame's hand and last frame's recognized hand should be smaller than same_hand_threshold to regard them as same hand
        same_hand_threshold = (
            cv2.getTrackbarPos("same_hand", "gesture recognition") * 100
        )
        landmark_skip_frame = max(
            cv2.getTrackbarPos("skip_frame", "gesture recognition"), 1
        )
        start_recognizing_time_threshold = cv2.getTrackbarPos(
            "start_time", "gesture recognition"
        )
        stop_recognizing_time_threshold = cv2.getTrackbarPos(
            "stop_time", "gesture recognition"
        )
        multi_action_time_threshold = cv2.getTrackbarPos(
            "multi_time", "gesture recognition"
        )
        multi_action_cooltime = (
            cv2.getTrackbarPos("multi_cooltime", "gesture recognition") / 10
        )

        ok, frame = cv2_util.read()
        if not ok:
            break

        frame_num += 1
        if frame_num % landmark_skip_frame == 0:
            # Process the frame with MediaPipe Hands
            results = hand_tracker.inference(frame)

            landmark_num += 1

            right_hands = []
            recognized_hands = []
            if results:
                for result in results:
                    if result["handedness"] > 0.5:  # Right Hand
                        # Convert right hand coordinations for rendering
                        right_hands.append(result["landmark"])
                        recognized_hands.append(result["landmark"])

                if recognizing:
                    # find closest hand
                    hand_idx, recognized_hand_prev_pos = utils.same_hand_tracking(
                        right_hands, recognized_hand_prev_pos, same_hand_threshold
                    )

                    if hand_idx != -1:
                        last_hand_time = time.time()

                        recognized_hand = recognized_hands[hand_idx]
                        recognized_hand_prev_pos = utils.get_center(recognized_hand)

                        lst, _ = utils.normalize_points(recognized_hand)

                        start = time.time_ns() // 1000000
                        res = list(
                            model.result_with_softmax(
                                torch.tensor(
                                    [element for row in lst for element in row],
                                    dtype=torch.float,
                                )
                            )
                        )
                        end = time.time_ns() // 1000000
                        gesture_time += end - start
                        gesture_num += 1

                        probability = max(res)
                        gesture_idx = (
                            res.index(probability) if probability >= config["gesture_prob_threshold"] else len(gestures) - 1
                        )
                        text_a = f"{gestures[gesture_idx]} {int(probability * 100)}%"

                        if state["gesture"] == gesture_idx:
                            # start multi action when user hold one gesture enough time
                            if (
                                time.time() - state["start_time"]
                                > multi_action_time_threshold
                            ):
                                if state["multi_action_start_time"] == -1:
                                    state["multi_action_start_time"] = time.time()
                                if (
                                    time.time() - state["multi_action_start_time"]
                                    > multi_action_cooltime * state["multi_action_cnt"]
                                ):
                                    state["multi_action_cnt"] += 1
                                    state["prev_action"] = utils.perform_action(
                                        state["prev_action"][0], infinite=True
                                    )

                            elif time.time() - state["start_time"] > time_threshold:
                                if gestures[state["prev_gesture"]] == "default":
                                    state["prev_action"] = utils.perform_action(
                                        gestures[state["gesture"]]
                                    )
                                state["prev_gesture"] = gesture_idx
                        else:
                            state = {
                                "gesture": gesture_idx,
                                "start_time": time.time(),
                                "prev_gesture": state["prev_gesture"],
                                "multi_action_start_time": -1,
                                "multi_action_cnt": 0,
                                "prev_action": ["", 0],
                            }
                    else:
                        # stop recognizing
                        recognized_hand = []
                        text_a = ""
                        if (
                            recognizing and time.time() - last_hand_time > stop_recognizing_time_threshold
                        ):
                            print("stop recognizing")
                            utils.play_audio_file("Stop")
                            recognizing = False
                            state = {
                                "gesture": 5,
                                "start_time": time.time(),
                                "prev_gesture": 5,
                                "multi_action_start_time": -1,
                                "multi_action_cnt": 0,
                                "prev_action": ["", 0],
                            }
                else:
                    # when not recognizing, get hands with 'default' gesture and measure elapsed time
                    delete_list = []
                    wake_up_hands = []
                    for right_hand in right_hands:
                        lst, _ = utils.normalize_points(right_hand)
                        res = list(
                            model.result_with_softmax(
                                torch.tensor(
                                    [element for row in lst for element in row],
                                    dtype=torch.float,
                                )
                            )
                        )
                        probability = max(res)
                        gesture_idx = (
                            res.index(probability) if probability >= config["gesture_prob_threshold"] else len(gestures) - 1
                        )
                        if gestures[gesture_idx] == "default":
                            wake_up_hands.append(right_hand)
                    checked = [0 for _ in range(len(wake_up_hands))]
                    for i, [prev_pos, start_time] in enumerate(wake_up_state):
                        hand_idx, prev_pos = utils.same_hand_tracking(
                            wake_up_hands, prev_pos, same_hand_threshold
                        )
                        if hand_idx == -1:
                            delete_list = [i] + delete_list
                        elif (
                            time.time() - start_time > start_recognizing_time_threshold
                        ):
                            # when there are default gestured hand for enough time, start recognizing and track the hand
                            print("start recognizing")
                            recognized_hand_prev_pos = utils.get_center(
                                wake_up_hands[hand_idx]
                            )
                            utils.play_audio_file("Start")
                            recognizing = True
                            wake_up_state = []
                            break
                        else:
                            checked[hand_idx] = 1

                    # wake_up_state refreshing
                    if not recognizing:
                        for i in delete_list:
                            wake_up_state.pop(i)

                        for idx, _ in enumerate(checked):
                            if checked[idx] == 0:
                                wake_up_state.append(
                                    [utils.get_center(wake_up_hands[idx]), time.time()]
                                )
            else:
                # stop recognizing
                recognized_hands = []
                recognized_hand = []
                text_a = ""
                if (
                    recognizing and time.time() - last_hand_time > stop_recognizing_time_threshold
                ):
                    print("stop recognizing")
                    utils.play_audio_file("Stop")
                    recognizing = False
                    state = {
                        "gesture": 5,
                        "start_time": time.time(),
                        "prev_gesture": 5,
                        "multi_action_start_time": -1,
                        "multi_action_cnt": 0,
                        "prev_action": ["", 0],
                    }

        annotated_frame = cv2_util.annotated_frame(frame)

        for rh in recognized_hands:
            annotated_frame = cv2_util.print_landmark(annotated_frame, rh)
        if len(recognized_hand) > 0:
            annotated_frame = cv2_util.print_landmark(annotated_frame, recognized_hand, (255, 0, 0))

        # Print Current Hand's Gesture
        cv2.putText(
            annotated_frame,
            text_a,
            (annotated_frame.shape[1] // 2 + 230, annotated_frame.shape[0] // 2 - 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
        )

        # print recognized gesture
        if time.time() - state["prev_action"][1] < time_threshold * 2:
            cv2.putText(
                annotated_frame,
                state["prev_action"][0],
                (
                    annotated_frame.shape[1] // 2 + 250,
                    annotated_frame.shape[0] // 2 - 100,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                3,
            )

        annotated_frame = cv2_util.unpad(annotated_frame)

        cv2_util.fps.display(annotated_frame, orig=(50, 50), color=(240, 180, 100))
        cv2.imshow("gesture recognition", annotated_frame)

        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break
        elif key == 32:
            # Pause on space bar
            cv2.waitKey(0)


def initialize_model():
    model = Model()
    if os.path.exists("../models/model.pt"):
        model.load_state_dict(torch.load("../models/model.pt"))
    else:
        model.load_state_dict(torch.load("../models/base_model.pt"))
    return model


def load_parameters(parameters_dir):
    res = "y"  # input("want to use saved parameters? [ y / n ]\n>>> ")
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
    cv2.createTrackbar(
        "time", "gesture recognition", parameter["time"], 100, utils.nothing
    )
    cv2.createTrackbar(
        "same_hand", "gesture recognition", parameter["same_hand"], 100, utils.nothing
    )
    cv2.createTrackbar(
        "skip_frame", "gesture recognition", parameter["skip_frame"], 50, utils.nothing
    )
    cv2.createTrackbar(
        "start_time", "gesture recognition", parameter["start_time"], 10, utils.nothing
    )
    cv2.createTrackbar(
        "stop_time", "gesture recognition", parameter["stop_time"], 10, utils.nothing
    )
    cv2.createTrackbar(
        "multi_time", "gesture recognition", parameter["multi_time"], 10, utils.nothing
    )
    cv2.createTrackbar(
        "multi_cooltime",
        "gesture recognition",
        parameter["multi_cooltime"],
        10,
        utils.nothing,
    )


def save_current_parameters(parameters_dir, parameter):
    res = input("want to save current parameters? [ y / n ]\n>>> ")
    if res == "y":
        with open(parameters_dir, "w") as f:
            json.dump(parameter, f)


if __name__ == "__main__":    
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # This could cause error
    pid = os.getpid() 
    cpu_process = Process(target=utils.monitor, args=(config["device"], pid))
    cpu_process.start()

    # Get the directory of test.py
    current_dir = os.path.dirname(os.path.relpath(__file__))

    # Construct the path to palm_detection.xml
    pd_model_path = os.path.join(
        current_dir,
        "..",
        "openvino_utils",
        "mediapipe_models",
        "palm_detection_FP16.xml"
    )
    lm_model_path = os.path.join(
        current_dir,
        "..",
        "openvino_utils",
        "mediapipe_models",
        "hand_landmark_FP16.xml"
    )

    ht = HandTracker(
        input_src="0",
        pd_xml=pd_model_path,
        pd_device=config["device"],
        pd_score_thresh=0.6,
        pd_nms_thresh=0.3,
        lm_xml=lm_model_path,
        lm_device=config["device"],
        lm_score_threshold=0.6,
    )

    model_obj = initialize_model()

    PARAMETERS_DIR = "./parameters.json"
    my_parameters = load_parameters(PARAMETERS_DIR)
    create_trackbars(my_parameters)

    cv2_util_obj = CV2Utils()

    run(ht, model_obj, cv2_util_obj)

    # Release the webcam and close all windows
    ht.cap.release()
    cv2.destroyAllWindows()

    save_current_parameters(PARAMETERS_DIR, my_parameters)
