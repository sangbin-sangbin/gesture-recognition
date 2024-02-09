import cv2
import mediapipe as mp
import math
import time
import torch
import torch.nn as nn
import json


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

    return normalized_points

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands model
hands = mp_hands.Hands()

# Open a webcam
w = 1280
h = 720
cap = cv2.VideoCapture(0)#1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

start_time = time.time()
state = 'break'

dataset_dir = "./dataset.json"
dataset = []
save = input("want to add previous dataset? [ y / n ]\n>>> ")
if save == 'y':
    dataset = json.load(open(dataset_dir))
data = []

gestures = [ 'default', 'left', 'right', 'select', 'exit', 'none' ]
gesture_num = 0

while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()
    if frame is None:
        continue

    # Check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    # Extract hand landmarks if available
    if results.multi_hand_landmarks:
        # Get the coordinates of the index fingertip (landmark index 8)
        for i in range(len(results.multi_hand_landmarks[0].landmark)):
            tip_x = results.multi_hand_landmarks[0].landmark[i].x * frame.shape[1]
            tip_y = results.multi_hand_landmarks[0].landmark[i].y * frame.shape[0]

            # Draw a circle at the fingertip position
            cv2.circle(frame, (int(tip_x), int(tip_y)), 5, (0, 255, 0), -1)

        lst = list(map(lambda x : [x.x, x.y], results.multi_hand_landmarks[0].landmark))

        if state == 'recording':
            dataset.append({'landmarks':normalize_points(lst), 'gesture':gesture_num})

    if (time.time() - start_time) > 30 and state == 'recording':
        start_time = time.time()
        state = 'break'
        gesture_num = (gesture_num + 1) % len(gestures)
    elif (time.time() - start_time) > 10 and state == 'break':
        start_time = time.time()
        state = 'recording'
        
    if state == 'break':
        cv2.putText(frame, "Break.. Next gesture is \"" + gestures[gesture_num] + "\"", (frame.shape[1] // 2 - 430, frame.shape[0] // 2 - 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    else:
        cv2.putText(frame, "Recorging \"" + gestures[gesture_num] + "\"", (frame.shape[1] // 2 - 430, frame.shape[0] // 2 - 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    # Display the output
    cv2.imshow('1', frame)
 
    # Check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

print(len(dataset), 'data generated')
save = input("want to save? [ y / n ]\n>>> ")
if save == 'y':
    with open(dataset_dir, "w") as f:
        json.dump(dataset, f, indent=4)
