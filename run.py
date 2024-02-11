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


def run(hand_tracker, model):
    gestures = ["default", "left", "right", "select", "exit", "none"]
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
    )

    landmark_time = 0
    landmark_num = 0
    gesture_time = 0
    gesture_num = 0

    recognizing_hands = []
    recognizing_hand = []
    text_a = ""
    cur_gesture = gestures[5]
    elapsed_time = "0"
    prev_gesture = gestures[5]

    recognizing = False
    last_hand_time = time.time()

    wake_up_state = []

    hand_tracker.fps = FPS(mean_nb_frames=20)

    nb_pd_inferences = 0
    nb_lm_inferences = 0
    glob_pd_rtrip_time = 0
    glob_lm_rtrip_time = 0
    while True:
        hand_tracker.fps.update()
        if hand_tracker.image_mode:
            vid_frame = hand_tracker.img
        else:
            # require more time than time_threshold to recognize it as an gesture
            time_threshold = cv2.getTrackbarPos("time", "gesture recognition") / 100
            # distance between this frame's hand and last frame's recognized hand should be smaller than same_hand_threshold to regard them as same hand
            same_hand_threshold = cv2.getTrackbarPos("same_hand", "gesture recognition") / 1000
            landmark_skip_frame = max(cv2.getTrackbarPos("skip_frame", "gesture recognition"), 1)
            start_recognizing_time_threshold = cv2.getTrackbarPos("start_time", "gesture recognition")
            stop_recognizing_time_threshold = cv2.getTrackbarPos("stop_time", "gesture recognition")
            multi_action_time_threshold = cv2.getTrackbarPos("multi_time", "gesture recognition")
            multi_action_cooltime = cv2.getTrackbarPos("multi_cooltime", "gesture recognition") / 10

            ok, vid_frame = hand_tracker.cap.read()
            if not ok:
                break

        h, w = vid_frame.shape[:2]
        if hand_tracker.crop:
            # Cropping the long side to get a square shape
            hand_tracker.frame_size = min(h, w)
            dx = (w - hand_tracker.frame_size) // 2
            dy = (h - hand_tracker.frame_size) // 2
            video_frame = vid_frame[dy : dy + hand_tracker.frame_size, dx : dx + hand_tracker.frame_size]
        else:
            # Padding on the small side to get a square shape
            hand_tracker.frame_size = max(h, w)
            pad_h = int((hand_tracker.frame_size - h) / 2)
            pad_w = int((hand_tracker.frame_size - w) / 2)
            video_frame = cv2.copyMakeBorder(vid_frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)

        # Resize image to NN square input shape
        frame_nn = cv2.resize(video_frame, (hand_tracker.pd_w, hand_tracker.pd_h), interpolation=cv2.INTER_AREA)

        # Transpose hxwx3 -> 1x3xhxw
        frame_nn = np.transpose(frame_nn, (2, 0, 1))[None,]

        annotated_frame = video_frame.copy()

        # Get palm detection
        pd_rtrip_time = now()
        inference = hand_tracker.pd_exec_net.infer(inputs={hand_tracker.pd_input_blob: frame_nn})
        glob_pd_rtrip_time += now() - pd_rtrip_time
        hand_tracker.pd_postprocess(inference)
        hand_tracker.pd_render(annotated_frame)
        nb_pd_inferences += 1

        # Hand landmarks
        if hand_tracker.use_lm:
            for i, r in enumerate(hand_tracker.regions):
                frame_nn = mpu.warp_rect_img(r.rect_points, video_frame, hand_tracker.lm_w, hand_tracker.lm_h)
                # Transpose hxwx3 -> 1x3xhxw
                frame_nn = np.transpose(frame_nn, (2, 0, 1))[None,]

                # Get hand landmarks
                lm_rtrip_time = now()
                inference = hand_tracker.lm_exec_net.infer(inputs={hand_tracker.lm_input_blob: frame_nn})
                glob_lm_rtrip_time += now() - lm_rtrip_time
                nb_lm_inferences += 1
                hand_tracker.lm_postprocess(r, inference)
                hand_tracker.lm_render(annotated_frame, r)

        frame_num += 1
        if frame_num % landmark_skip_frame == 0:
            # Process the frame with MediaPipe Hands
            start = time.time_ns() // 1000000
            # results = hands.process(rgb_frame)
            results2 = hand_tracker.regions
            end = time.time_ns() // 1000000
            landmark_time += end - start
            landmark_num += 1

            right_hands = []
            if results2:
                for result in results2:
                    if result.handedness > 0.5:  # Right Hand
                        right_hands.append(hand_tracker.coords)

                recognizing_hands = right_hands

                if recognizing:
                    # find closest hand
                    hand_idx, recognized_hand_prev_pos = utils.same_hand_tracking(
                        right_hands, recognized_hand_prev_pos, same_hand_threshold
                    )

                    if hand_idx != -1:
                        last_hand_time = time.time()

                        landmark = right_hands[hand_idx]
                        recognizing_hand = landmark
                        lst, scale = utils.normalize_points(landmark)

                        start = time.time_ns() // 1000000
                        res = list(
                            model(
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
                        gesture_idx = res.index(probability) if probability >= 0.9 else 5
                        text_a = f"{gestures[gesture_idx]} {int(probability*100)}%"

                        cur_gesture = gestures[state["gesture"]]
                        elapsed_time = str(round(time.time() - state["start_time"], 2))
                        prev_gesture = gestures[state["prev_gesture"]]

                        if state["gesture"] == gesture_idx:
                            # start multi action when user hold one gesture enough time
                            if time.time() - state["start_time"] > multi_action_time_threshold:
                                if state["multi_action_start_time"] == -1:
                                    state["multi_action_start_time"] = time.time()
                                if (
                                    time.time() - state["multi_action_start_time"]
                                    > multi_action_cooltime * state["multi_action_cnt"]
                                ):
                                    state["multi_action_cnt"] += 1
                                    state["prev_action"] = utils.perform_action(state["prev_action"][0])

                            elif time.time() - state["start_time"] > time_threshold:
                                if gestures[state["prev_gesture"]] == "default":
                                    state["prev_action"] = utils.perform_action(gestures[state["gesture"]])
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
                        recognizing_hand = []
                        text_a = ""
                        if recognizing and time.time() - last_hand_time > stop_recognizing_time_threshold:
                            print("stop recognizing")
                            utils.play_wav_file("stop")
                            recognizing = False
                            state = {
                                "gesture": 5,
                                "start_time": time.time(),
                                "prev_gesture": 5,
                                "multi_action_start_time": -1,
                                "multi_action_cnt": 0,
                                "prev_action": ["", 0],
                            }

                            cur_gesture = "none"
                            elapsed_time = "0"
                            prev_gesture = "none"
                else:
                    # when not recognizing, get hands with 'default' gesture and measure elapsed time
                    delete_list = []
                    wake_up_hands = []
                    print(right_hands)
                    if len(right_hands) > 0:
                        for right_hand in right_hands:
                            lst, scale = utils.normalize_points(right_hand)

                            res = list(
                                model(
                                    torch.tensor(
                                        [element for row in lst for element in row],
                                        dtype=torch.float,
                                    )
                                )
                            )
                #         probability = max(res)
                #         gesture_idx = res.index(probability) if probability >= 0.9 else 5
                #         if gestures[gesture_idx] == "default":
                #             wake_up_hands.append(right_hand)

                #     checked = [0 for _ in range(len(wake_up_hands))]
                #     for i, [prev_pos, start_time] in enumerate(wake_up_state):
                #         hand_idx, prev_pos = utils.same_hand_tracking(wake_up_hands, prev_pos, same_hand_threshold)
                #         if hand_idx == -1:
                #             delete_list = [i] + delete_list
                #         elif time.time() - start_time > start_recognizing_time_threshold:
                #             # when there are default gestured hand for enough time, start recognizing and track the hand
                #             print("start recognizing")
                #             recognized_hand_prev_pos = utils.get_center(wake_up_hands[hand_idx])
                #             utils.play_wav_file("start")
                #             recognizing = True
                #             wake_up_state = []
                #             break
                #         else:
                #             checked[hand_idx] = 1

                #     # wake_up_state refreshing
                #     if not recognizing:
                #         for i in delete_list:
                #             wake_up_state.pop(i)

                #         for i in range(len(checked)):
                #             if checked[i] == 0:
                #                 wake_up_state.append([utils.get_center(wake_up_hands[i]), time.time()])
            else:
                # stop recognizing
                recognizing_hands = []
                recognizing_hand = []
                text_a = ""
                if recognizing and time.time() - last_hand_time > stop_recognizing_time_threshold:
                    print("stop recognizing")
                    utils.play_wav_file("stop")
                    recognizing = False
                    state = {
                        "gesture": 5,
                        "start_time": time.time(),
                        "prev_gesture": 5,
                        "multi_action_start_time": -1,
                        "multi_action_cnt": 0,
                        "prev_action": ["", 0],
                    }

                    cur_gesture = "none"
                    elapsed_time = "0"
                    prev_gesture = "none"

        # for rh in recognizing_hands:
        #     for x, y in rh:
        #         # Draw a circle at the fingertip position
        #         cv2.circle(
        #             annotated_frame,
        #             (int(x * annotated_frame.shape[1]), int(y * annotated_frame.shape[0])),
        #             5,
        #             (0, 255, 0),
        #             -1,
        #         )
        for x, y in recognizing_hand:
            # Draw a circle at the fingertip position
            cv2.circle(
                annotated_frame,
                (int(x * annotated_frame.shape[1]), int(y * annotated_frame.shape[0])),
                5,
                (255, 0, 0),
                -1,
            )

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
                (annotated_frame.shape[1] // 2 + 250, annotated_frame.shape[0] // 2 + 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                3,
            )

        # Print Table
        header_data = ["curr_gesture", "elapsed_time", "prev_gesture"]
        table_data = [cur_gesture, elapsed_time, prev_gesture]
        cell_height = 50
        cell_width = 250
        text_position = (50, 50)

        for i, data in enumerate([header_data] + [table_data]):
            for j, cell in enumerate(data):
                x = 50 + cell_width * j + 30
                y = text_position[1] + 20 + i * cell_height + cell_height // 2 + 10
                cv2.putText(annotated_frame, str(cell), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if i == 0:
                    cv2.rectangle(
                        annotated_frame,
                        (50 + cell_width * j, text_position[1] + 20),
                        (50 + cell_width * (j + 1), text_position[1] + 20 + cell_height),
                        (255, 0, 0),
                        2,
                    )
                else:
                    cv2.rectangle(
                        annotated_frame,
                        (50 + cell_width * j, text_position[1] + 20 + i * cell_height),
                        (
                            50 + cell_width * (j + 1),
                            text_position[1] + 20 + (i + 1) * cell_height,
                        ),
                        (255, 0, 0),
                        2,
                    )

        if not hand_tracker.crop:
            annotated_frame = annotated_frame[pad_h : pad_h + h, pad_w : pad_w + w]

        hand_tracker.fps.display(annotated_frame, orig=(50, 50), color=(240, 180, 100))
        cv2.imshow("video", annotated_frame)

        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break
        elif key == 32:
            # Pause on space bar
            cv2.waitKey(0)
        elif key == ord("1"):
            hand_tracker.show_pd_box = not hand_tracker.show_pd_box
        elif key == ord("2"):
            hand_tracker.show_pd_kps = not hand_tracker.show_pd_kps
        elif key == ord("3"):
            hand_tracker.show_rot_rect = not hand_tracker.show_rot_rect
        elif key == ord("4"):
            hand_tracker.show_landmarks = not hand_tracker.show_landmarks
        elif key == ord("5"):
            hand_tracker.show_handedness = not hand_tracker.show_handedness
        elif key == ord("6"):
            hand_tracker.show_scores = not hand_tracker.show_scores
        elif key == ord("7"):
            hand_tracker.show_gesture = not hand_tracker.show_gesture

    # Print some stats
    print(f"# palm detection inferences : {nb_pd_inferences}")
    print(f"# hand landmark inferences  : {nb_lm_inferences}")
    print(f"Palm detection round trip   : {glob_pd_rtrip_time/nb_pd_inferences*1000:.1f} ms")
    print(f"Hand landmark round trip    : {glob_lm_rtrip_time/nb_lm_inferences*1000:.1f} ms")