import os
import subprocess
import time

import pygame


def get_center(landmark):
    sum_x = 0
    sum_y = 0
    for x, y in landmark:
        sum_x += x
        sum_y += y
    return [sum_x / len(landmark), sum_y / len(landmark)]


def same_hand_tracking(hands, prev_pos, same_hand_threshold):
    # -1 index means there is no same hand
    if len(hands) == 0:
        return -1, prev_pos

    positions = []
    for landmark in hands:
        center = get_center(landmark)
        positions.append([center[0], center[1]])

    min_idx = -1
    min_val = float("inf")
    for i, [x, y] in enumerate(positions):
        distance = (x - prev_pos[0]) ** 2 + (y - prev_pos[1]) ** 2
        if min_val > distance:
            min_idx = i
            min_val = distance

    if min_val > same_hand_threshold:
        return -1, prev_pos
    return min_idx, [positions[min_idx][0], positions[min_idx][1]]


def play_audio_file(file_name):
    pygame.mixer.init()

    # Get the directory of test.py
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one level to the parent directory of test
    parent_dir = os.path.dirname(current_dir)

    # Construct the path to sound directory
    sound_dir = os.path.join(parent_dir, "sound")

    # Construct the file paths for both .wav and .mp3 files
    wav_file_path = os.path.join(sound_dir, file_name + ".wav")
    mp3_file_path = os.path.join(sound_dir, file_name + ".mp3")

    # Check if the .wav file exists, if not, check for .mp3 file
    if os.path.isfile(wav_file_path):
        audio_file_path = wav_file_path
    elif os.path.isfile(mp3_file_path):
        audio_file_path = mp3_file_path
    else:
        print("No such audio file!")
        return

    # Load and play the audio file
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()


def nothing(_):
    pass


def normalize_points(points):
    """
    Normalize points so that every coordinate value falls between 0 and 1
    """
    min_x, min_y = points[0]
    max_x, max_y = points[0]

    for x, y in points:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    scale = max(max_x - min_x, max_y - min_y)

    normalized_points = []
    for x, y in points:
        normalized_x = (x - min_x) / scale
        normalized_y = (y - min_y) / scale
        normalized_points.append((normalized_x, normalized_y))

    return normalized_points, scale


def perform_action(action, infinite=False):
    if action == "right":
        command = "adb shell input keyevent KEYCODE_DPAD_RIGHT"
        subprocess.run(command, shell=True, check=False)
        print("right")
        if infinite:
            play_audio_file("Swipe1")
        else:
            play_audio_file("Right")
        return ["right", time.time()]
    elif action == "left":
        command = "adb shell input keyevent KEYCODE_DPAD_LEFT"
        subprocess.run(command, shell=True, check=False)
        print("left")
        if infinite:
            play_audio_file("Swipe2")
        else:
            play_audio_file("Left")
        return ["left", time.time()]
    elif action == "select":
        command = "adb shell input keyevent KEYCODE_BUTTON_SELECT"
        subprocess.run(command, shell=True, check=False)
        print("select")
        play_audio_file("Select")
        return ["select", time.time()]
    elif action == "exit":
        command = "adb shell input keyevent KEYCODE_BACK"
        subprocess.run(command, shell=True, check=False)
        print("exit")
        play_audio_file("Exit")
        return ["exit", time.time()]
    elif action == "shortcut1":
        command = "adb shell input keyevent SHORTCUT1"
        subprocess.run(command, shell=True, check=False)
        print("shortcut 1")
        play_audio_file("First")
        return ["shortcut1", time.time()]
    elif action == "shortcut2":
        command = "adb shell input keyevent SHORTCUT2"
        subprocess.run(command, shell=True, check=False)
        print("shortcut 2")
        play_audio_file("Second")
        return ["shortcut2", time.time()]
    return ["", 0]
