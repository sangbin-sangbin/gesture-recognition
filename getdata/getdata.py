import datetime as dt
import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from MediaPipe import mediapipe_utils as mpu
from MediaPipe.FPS import FPS, now
from MediaPipe.HandTracker import HandTracker


def run(hand_tracker):

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
                dy : dy + hand_tracker.frame_size, dx : dx + hand_tracker.frame_size
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
        pd_rtrip_time = now()
        infer_request = hand_tracker.pd_exec_model.create_infer_request()
        inference = infer_request.infer(inputs={hand_tracker.pd_input_blob: frame_nn})
        glob_pd_rtrip_time += now() - pd_rtrip_time
        hand_tracker.pd_postprocess(inference)
        hand_tracker.pd_render(annotated_frame)
        nb_pd_inferences += 1

        # Hand landmarks
        if hand_tracker.use_lm:
            for i, r in enumerate(hand_tracker.regions):
                frame_nn = mpu.warp_rect_img(
                    r.rect_points, video_frame, hand_tracker.lm_w, hand_tracker.lm_h
                )
                # Transpose hxwx3 -> 1x3xhxw
                frame_nn = np.transpose(frame_nn, (2, 0, 1))[None,]

                # Get hand landmarks
                lm_rtrip_time = now()
                lm_infer_request = hand_tracker.lm_exec_model.create_infer_request()
                inference = lm_infer_request.infer(
                    inputs={hand_tracker.lm_input_blob: frame_nn}
                )
                glob_lm_rtrip_time += now() - lm_rtrip_time
                nb_lm_inferences += 1
                hand_tracker.lm_postprocess(r, inference)
                hand_tracker.lm_render(annotated_frame, r)

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
    print(
        f"Palm detection round trip   : {glob_pd_rtrip_time/nb_pd_inferences*1000:.1f} ms"
    )
    print(
        f"Hand landmark round trip    : {glob_lm_rtrip_time/nb_lm_inferences*1000:.1f} ms"
    )


def get_data(hand_tracker):
    run(hand_tracker)

    # Get the directory of getdata.py
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one level to the parent directory of dataset
    parent_dir = os.path.dirname(current_dir)

    # Construct the path to dataset/tmp
    dataset_dir = os.path.join(parent_dir, "dataset", "tmp")

    # Check if the directory exists, if not, create it
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    print(len(hand_tracker.dataset), "data generated")

    save = input("want to save? [ y / n ]\n>>> ")
    if save == "y":
        name = input("what is your name?\n>>> ")
        datetime = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Construct the output file path using os.path.join()
        output_file_path = os.path.join(dataset_dir, f"{name}_{datetime}.json")
        try:
            with open(output_file_path, "w") as f:
                json.dump(hand_tracker.dataset, f, indent=4)
            print(f"Data saved to {output_file_path}")
        except Exception as e:
            print(f"Error occurred while saving data: {e}")


if __name__ == "__main__":
    # Get the directory of getdata.py
    current_dir = os.path.dirname(os.path.relpath(__file__))

    # Construct the path to palm_detection.xml
    pd_model_path = os.path.join(
        current_dir, "..", "MediaPipe", "mediapipe_models", "palm_detection_FP16.xml"
    )
    lm_model_path = os.path.join(
        current_dir, "..", "MediaPipe", "mediapipe_models", "hand_landmark_FP16.xml"
    )

    ht = HandTracker(
        input_src="0",
        pd_xml=pd_model_path,
        pd_device="GPU",
        pd_score_thresh=0.5,
        pd_nms_thresh=0.3,
        use_lm=True,
        lm_xml=lm_model_path,
        lm_device="GPU",
        lm_score_threshold=0.5,
        crop=False,
        is_getdata=True,
    )

    get_data(ht)
