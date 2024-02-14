import subprocess
import pygame
import time


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


def play_wav_file(file_name):
    pygame.mixer.init()
    pygame.mixer.music.load("./sound/" + file_name + ".wav")
    pygame.mixer.music.play()


def nothing(x):
    pass


def normalize_points(points):
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


def perform_action(action):
    if action == "right":
        subprocess.run("adb shell input keyevent KEYCODE_DPAD_RIGHT", shell=True)
        print("right")
        play_wav_file("action")
        return ["right", time.time()]
    elif action == "left":
        subprocess.run("adb shell input keyevent KEYCODE_DPAD_LEFT", shell=True)
        print("left")
        play_wav_file("action")
        return ["left", time.time()]
    elif action == "select":
        subprocess.run("adb shell input keyevent KEYCODE_BUTTON_SELECT", shell=True)
        print("select")
        play_wav_file("action")
        return ["select", time.time()]
    elif action == "exit":
        subprocess.run("adb shell input keyevent KEYCODE_BACK", shell=True)
        print("exit")
        play_wav_file("action")
        return ["exit", time.time()]
    return ["", 0]
