import cv2
import argparse
import json
import time
from HandTracker import HandTracker  # Assuming HandTracker is defined in HandTracker.py


def get_data(hand_tracker):
    dataset_dir = "./dataset2.json"
    save = input("want to add previous dataset? [ y / n ]\n>>> ")
    if save == "y":
        hand_tracker.dataset = json.load(open(dataset_dir))

    hand_tracker.run()

    print(len(hand_tracker.dataset), "data generated")
    save = input("want to save? [ y / n ]\n>>> ")
    if save == "y":
        with open(dataset_dir, "w") as f:
            json.dump(hand_tracker.dataset, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="0",
        help="Path to video or image file to use as input (default=%(default)s)",
    )
    parser.add_argument(
        "--pd_m",
        default="models/palm_detection_FP32.xml",
        type=str,
        help="Path to an .xml file for palm detection model (default=%(default)s)",
    )
    parser.add_argument(
        "--pd_device",
        default="CPU",
        type=str,
        help="Target device for the palm detection model (default=%(default)s)",
    )
    parser.add_argument(
        "--no_lm",
        action="store_true",
        help="only the palm detection model is run, not the hand landmark model",
    )
    parser.add_argument(
        "--lm_m",
        default="models/hand_landmark_FP32.xml",
        type=str,
        help="Path to an .xml file for landmark model (default=%(default)s)",
    )
    parser.add_argument(
        "--lm_device",
        default="CPU",
        type=str,
        help="Target device for the landmark regression model (default=%(default)s)",
    )
    parser.add_argument(
        "-c",
        "--crop",
        action="store_true",
        help="center crop frames to a square shape before feeding palm detection model",
    )

    args = parser.parse_args()

    ht = HandTracker(
        input_src=args.input,
        pd_device=args.pd_device,
        use_lm=not args.no_lm,
        lm_device=args.lm_device,
        crop=args.crop,
    )

    get_data(ht)
