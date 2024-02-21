import datetime as dt
import json
import os
import sys
import time

import cv2
import yaml

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))   

import utils
from cv2_utils import CV2Utils
from openvino_utils.hand_tracker import HandTracker

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def run(hand_tracker, cv2_util):
    gestures = config['gestures']

    state = "break"
    start_time = time.time()
    gesture_num = 0
    
    dataset = []

    while True:
        ok, frame = cv2_util.read()
        if not ok:
            break

        results = hand_tracker.inference(frame)
        annotated_frame = cv2_util.annotated_frame(frame)
        
        if results:
            for result in results:
                if result["handedness"] > 0.5:  # Right Hand
                    annotated_frame = cv2_util.print_landmark(annotated_frame, result["landmark"])
                    if state == "recording":
                        dataset.append({
                            "landmarks": utils.normalize_points(result["landmark"])[0],
                            "gesture": gesture_num,
                        })

        if (time.time() - start_time) > 30 and state == "recording":
            start_time = time.time()
            state = "break"
            # collect gesture except custom, none
            gesture_num = (gesture_num + 1) % (len(gestures) - 2)
            utils.play_audio_file("Action")
        elif (time.time() - start_time) > 10 and state == "break":
            start_time = time.time()
            state = "recording"
            utils.play_audio_file("Action")

        if state == "break":
            cv2.putText(
                annotated_frame,
                'Break.. Next gesture is "' + gestures[gesture_num] + '"',
                (annotated_frame.shape[1] // 2 - 430, annotated_frame.shape[0] // 2 - 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                3,
            )
        else:
            cv2.putText(
                annotated_frame,
                'Recorging "' + gestures[gesture_num] + '"',
                (annotated_frame.shape[1] // 2 - 430, annotated_frame.shape[0] // 2 - 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                3,
            )
        
        annotated_frame = cv2_util.unpad(annotated_frame)

        cv2.imshow("gesture recognition", annotated_frame)

        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break

    return dataset


def get_data(hand_tracker, cv2_util):
    dataset = run(hand_tracker, cv2_util)

    # Construct the path to dataset/tmp
    dataset_dir = os.path.join("..", "dataset", "tmp")
    print(dataset_dir)
    # Check if the directory exists, if not, create it
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    print(len(dataset), "data generated")

    save = input("want to save? [ y / n ]\n>>> ")
    if save == "y":
        name = input("what is your name?\n>>> ")
        datetime = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Construct the output file path using os.path.join()
        output_file_path = os.path.join(dataset_dir, f"{name}_{datetime}.json")
        with open(output_file_path, "w") as f:
            json.dump(dataset, f, indent=4)
        print(f"Data saved to {output_file_path}")


if __name__ == "__main__":
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

    cv2_util_obj = CV2Utils()

    get_data(ht, cv2_util_obj)
