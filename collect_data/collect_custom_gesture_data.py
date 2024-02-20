import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from openvino_utils import mediapipe_utils as mpu
from openvino_utils.fps import FPS, now
from openvino_utils.hand_tracker import HandTracker
import utils
from models.model import Model
import yaml

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

MAX_COUNT = 5

def initialize_model():
    model = Model()
    model.load_state_dict(torch.load("../models/base_model.pt"))
    return model

def run(hand_tracker, model):
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
        ok, vid_frame = hand_tracker.cap.read()
        if not ok:
            break

        h, w = vid_frame.shape[:2]

        if hand_tracker.crop:
            # Cropping the long side to get a square shape
            hand_tracker.frame_size = min(h, w)
            dx = (w - hand_tracker.frame_size) // 2
            dy = (h - hand_tracker.frame_size) // 2
            video_frame = vid_frame[
                dy: dy + hand_tracker.frame_size,
                dx: dx + hand_tracker.frame_size
            ]
        else:
            # Padding on the small side to get a square shape
            hand_tracker.frame_size = max(h, w)
            pad_h = int((hand_tracker.frame_size - h) / 2)
            pad_w = int((hand_tracker.frame_size - w) / 2)
            video_frame = cv2.copyMakeBorder(
                vid_frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT
            )

        # Resize image to NN square input shape
        frame_nn = cv2.resize(
            video_frame,
            (hand_tracker.pd_w, hand_tracker.pd_h),
            interpolation=cv2.INTER_AREA,
        )

        # Transpose hxwx3 -> 1x3xhxw
        frame_nn = np.transpose(frame_nn, (2, 0, 1))[None,]

        annotated_frame = video_frame.copy()

        # Get palm detection
        infer_request = hand_tracker.pd_exec_model.create_infer_request()
        inference = infer_request.infer(
            inputs={hand_tracker.pd_input_blob: frame_nn}
        )
        hand_tracker.pd_postprocess(inference)
        hand_tracker.pd_render(annotated_frame)

        # Hand landmarks
        if hand_tracker.use_lm:
            for i, r in enumerate(hand_tracker.regions):
                frame_nn = mpu.warp_rect_img(
                    r.rect_points, video_frame, hand_tracker.lm_w, hand_tracker.lm_h
                )
                # Transpose hxwx3 -> 1x3xhxw
                frame_nn = np.transpose(frame_nn, (2, 0, 1))[None,]

                # Get hand landmarks
                lm_infer_request = hand_tracker.lm_exec_model.create_infer_request()
                inference = lm_infer_request.infer(
                    inputs={hand_tracker.lm_input_blob: frame_nn}
                )
                hand_tracker.lm_postprocess(r, inference)
                hand_tracker.lm_render(annotated_frame, r)

        # Process the frame with MediaPipe Hands
        results = hand_tracker.regions

        right_hands = []
        recognized_hands = []
        if results:
            for result in results:
                if result.handedness > 0.5:  # Right Hand
                    # Convert right hand coordinations for rendering
                    src = np.array(
                        [(0, 0), (1, 0), (1, 1)],
                        dtype=np.float32
                    )
                    dst = np.array(
                        [(x, y) for x, y in result.rect_points[1:]],
                        dtype=np.float32,
                    )  # region.rect_points[0] is left bottom point !
                    mat = cv2.getAffineTransform(src, dst)
                    lm_xy = np.expand_dims(
                        np.array([(l[0], l[1]) for l in result.landmarks]),
                        axis=0
                    )
                    lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int32)
                    right_hands.append(lm_xy)
                    recognized_hands.append(lm_xy)

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
                        model(
                            torch.tensor(
                                [element for row in lst for element in row],
                                dtype=torch.float,
                            )
                        )
                    )

                    probability = max(res)
                    gesture_idx = (
                        res.index(probability) if probability >= 0.9 else 5
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
                        model(
                            torch.tensor(
                                [element for row in lst for element in row],
                                dtype=torch.float,
                            )
                        )
                    )
                    probability = max(res)
                    gesture_idx = (
                        res.index(probability) if probability >= 0.9 else 5
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
                        waked = False
                        wake_up_time = time.time()
                        break
                    else:
                        checked[hand_idx] = 1

                # wake_up_state refreshing
                if not recognizing:
                    for i in delete_list:
                        wake_up_state.pop(i)

                    for chk in checked:
                        if chk == 0:
                            wake_up_state.append(
                                [utils.get_center(wake_up_hands[i]), time.time()]
                            )
        else:
            # stop recognizing
            recognized_hands = []
            recognized_hand = []
            if (
                recognizing
                and time.time() - last_hand_time > stop_recognizing_time_threshold
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
                dataset.append(
                    {
                        "landmarks": utils.normalize_points(recognized_hand),
                        "gesture": config['custom_gesture_index'],
                    }
                )

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
            for x, y in rh:
                # Draw a circle at the fingertip position
                cv2.circle(annotated_frame, (x, y), 6, (0, 255, 0), -1)
        for x, y in recognized_hand:
            # Draw a circle at the fingertip position
            cv2.circle(annotated_frame, (x, y), 6, (255, 0, 0), -1)

        if not hand_tracker.crop:
            annotated_frame = annotated_frame[pad_h: pad_h + h, pad_w: pad_w + w]

        cv2.imshow("gesture recognition", annotated_frame)

        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break

    return dataset

def get_data(hand_tracker):
    model = initialize_model()

    dataset = run(hand_tracker, model)

    # Get the directory of getdata.py
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one level to the parent directory of dataset
    parent_dir = os.path.dirname(current_dir)

    # Construct the path to dataset/tmp
    dataset_dir = os.path.join(parent_dir, "dataset")

    # Check if the directory exists, if not, create it
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    print(len(dataset), "data generated")

    save = input("want to save? Previous custom gesture will be removed. [ y / n ]\n>>> ")
    if save == "y":
        output_file_path = os.path.join(dataset_dir, "custom_gesture.json")
        try:
            with open(output_file_path, "w") as f:
                json.dump(dataset, f, indent=4)
            print(f"Data saved to {output_file_path}")
        except Exception as e:
            print(f"Error occurred while saving data: {e}")


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
        input_src="0",
        pd_xml=pd_model_path,
        pd_device=config["device"],
        pd_score_thresh=0.6,
        pd_nms_thresh=0.3,
        use_lm=True,
        lm_xml=lm_model_path,
        lm_device=config["device"],
        lm_score_threshold=0.6,
        crop=False,
        is_getdata=False,
    )

    get_data(ht)
