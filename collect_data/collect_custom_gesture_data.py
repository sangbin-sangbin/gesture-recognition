import json
import os
import sys
import time

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

MAX_COUNT = 5


def initialize_model():
    model = Model()
    model.load_state_dict(torch.load("../models/base_model.pt"))
    return model


def run(hand_tracker, model, cv2_util):
    gestures = config['gestures']

    stop_recognizing_time_threshold = 1
    start_recognizing_time_threshold = 1
    same_hand_threshold = 1000

    waked = False
    recognizing = False
    recognized_hand_prev_pos = [-999, -999]
    recognized_hand = []
    recognized_hands = []

    wake_up_state = []

    dataset = []

    countdown = MAX_COUNT
    wake_up_time = float('inf')

    while True:
        ok, frame = cv2_util.read()
        if not ok:
            break

        results = hand_tracker.inference(frame)
        annotated_frame = cv2_util.annotated_frame(frame)

        right_hands = []
        recognized_hands = []
        if results:
            for result in results:
                if result["handedness"] > 0.5:  # Right Hand
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
                else:
                    # stop recognizing
                    recognized_hand = []
                    if (recognizing and time.time() - last_hand_time > stop_recognizing_time_threshold):
                        print("stop recognizing")
                        utils.play_audio_file("Stop")
                        recognizing = False
                        if waked:
                            break
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
                    print(gestures[res.index(probability)], probability)
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
                        recognized_hand_prev_pos = utils.get_center(
                            wake_up_hands[hand_idx]
                        )
                        utils.play_audio_file("Action")
                        recognizing = True
                        wake_up_state = []
                        waked = True
                        wake_up_time = time.time()
                        break
                    else:
                        checked[hand_idx] = 1

                # wake_up_state refreshing
                if not recognizing:
                    for i in delete_list:
                        wake_up_state.pop(i)

                    for i, chk in enumerate(checked):
                        if chk == 0:
                            wake_up_state.append(
                                [utils.get_center(wake_up_hands[i]), time.time()]
                            )
        else:
            # stop recognizing
            recognized_hands = []
            recognized_hand = []
            if (
                recognizing and time.time() - last_hand_time > stop_recognizing_time_threshold
            ):
                print("stop recognizing")
                utils.play_audio_file("Stop")
                recognizing = False
            if waked:
                break

        if countdown > 0 and MAX_COUNT - countdown + 1 <= time.time() - wake_up_time:
            countdown -= 1
            utils.play_audio_file("Action")
        elif countdown == 0:
            if len(recognized_hand) > 0:
                dataset.append({
                    "landmarks": utils.normalize_points(recognized_hand)[0],
                    "gesture": config['custom_gesture_index'],
                })

        if recognizing and countdown > 0:
            cv2.putText(
                annotated_frame,
                str(countdown),
                (annotated_frame.shape[1] // 2 + 230, annotated_frame.shape[0] // 2 - 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3,
            )

        for rh in recognized_hands:
            annotated_frame = cv2_util.print_landmark(annotated_frame, rh)
        if len(recognized_hand) > 0:
            annotated_frame = cv2_util.print_landmark(annotated_frame, recognized_hand, (255, 0, 0))
        annotated_frame = cv2_util.unpad(annotated_frame)

        cv2.imshow("gesture recognition", annotated_frame)

        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break

    return dataset


def collect_data(hand_tracker, cv2_util):
    model = initialize_model()

    # Construct the path to dataset/tmp
    dataset_dir = os.path.join("..", "dataset")

    min_dataset_len = 500

    while True:
        dataset = run(hand_tracker, model, cv2_util)
        dataset_len = len(dataset)
        print(len(dataset), "data generated")
        if dataset_len >= min_dataset_len:
            break
        print(f"You should collect data more than {min_dataset_len}. Please collect your custom data again.")

    save = input("want to save? Previous custom gesture will be removed. [ y / n ]\n>>> ")
    if save == "y":
        output_file_path = os.path.join(dataset_dir, "custom_gesture.json")
        with open(output_file_path, "w") as f:
            json.dump(dataset, f, indent=4)
        print(f"Data saved to {output_file_path}")


if __name__ == "__main__":
    # Get the directory of getdata.py
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
        pd_xml=pd_model_path,
        pd_device=config["device"],
        pd_score_thresh=0.6,
        pd_nms_thresh=0.3,
        lm_xml=lm_model_path,
        lm_device=config["device"],
        lm_score_threshold=0.6,
    )

    cv2_util_obj = CV2Utils()

    collect_data(ht, cv2_util_obj)
