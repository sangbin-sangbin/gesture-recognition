import cv2
import mediapipe as mp
import math
import time
import torch
import torch.nn as nn
import json
from math import sqrt
import subprocess


class Model(nn.Module):
    def __init__(self, input_size, hidden_dim, tagset_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, landmarks):
        x = self.fc1(landmarks)
        x = self.fc2(x)
        res = self.softmax(x)
        return res

INPUT_SIZE = 42
HIDDEN_DIM = 32
TARGET_SIZE = 6

model = Model(INPUT_SIZE, HIDDEN_DIM, TARGET_SIZE)
model.load_state_dict(torch.load('model.pt'))

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

def nothing(x):
    pass
    
cv2.namedWindow('gesture recognition')
cv2.createTrackbar('time','gesture recognition',10,100,nothing)
cv2.createTrackbar('skip_frame','gesture recognition',1,50,nothing)

gestures = [ 'default', 'left', 'right', 'select', 'exit', 'none' ]
gesture_num = 0

time_threshold = cv2.getTrackbarPos('time','gesture recognition')/100 #0.1
state = {'gesture':5, 'start_time':time.time(), 'prev_gesture':4}

landmark_skip_frame = max(cv2.getTrackbarPos('skip_frame','gesture recognition'), 1) #3
frame_num = 0

def direction(a, b):
    x = b[0] - a[0]
    y = b[1] - a[1]
    if x >= abs(y):
        return 0
    elif -x >= abs(y):
        return 1
    elif abs(x) < -y:
        return 2
    elif abs(x) < y:
        return 3
    else:
        return 4

def distance(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

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

subprocess.run('adb connect 192.168.1.103:5555; adb root; adb connect 192.168.1.103:5555', shell=True)

landmark_time = 0
landmark_num = 0
gesture_time = 0
gesture_num = 0

landmark = []
text_a = ''
cur_gesture = gestures[5]
elapsed_time = '0'
prev_gesture = gestures[5]

while cap.isOpened():    
    time_threshold = cv2.getTrackbarPos('time','gesture recognition')/100

    # Read a frame from the webcam
    ret, frame = cap.read()
    if frame is None:
        continue
 
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_num += 1
    if frame_num % landmark_skip_frame == 0:
        # Process the frame with MediaPipe Hands
        start = time.time_ns()//1000000
        results = hands.process(rgb_frame)
        end = time.time_ns()//1000000
        landmark_time += end - start
        landmark_num += 1
        # Extract hand landmarks if available
        if results.multi_hand_landmarks:
            # Get the coordinates of the index fingertip (landmark index 8)
            hand_idx = -1
            for idx, hand in enumerate(results.multi_handedness):
                if hand.classification[0].label == 'Left':
                    hand_idx = idx
                    break
            if hand_idx > -1: 
                landmark = results.multi_hand_landmarks[hand_idx].landmark
                lst, scale = normalize_points(list(map(lambda x : [x.x, x.y], landmark)))

                start = time.time_ns()//1000000
                res = list(model(torch.tensor([element for row in lst for element in row], dtype=torch.float)))
                end = time.time_ns()//1000000
                gesture_time += end - start
                gesture_num += 1

                p = max(res)
                gesture_idx = res.index(p) if p >= 0.9 else 5
                text_a = f"{gestures[gesture_idx]} {int(p*100)}%"

                cur_gesture = gestures[state['gesture']]
                elapsed_time = str(round(time.time() - state['start_time'], 2))
                prev_gesture = gestures[state['prev_gesture']]

                if state['gesture'] == gesture_idx:
                    if time.time()-state['start_time'] > time_threshold:
                        if gestures[state['gesture']] == 'right' and gestures[state['prev_gesture']] == 'default':
                            print('right')
                        elif gestures[state['gesture']] == 'left' and gestures[state['prev_gesture']] == 'default':
                            print('left')
                        elif gestures[state['gesture']] == 'select' and gestures[state['prev_gesture']] == 'default':
                            print('select')
                        elif gestures[state['gesture']] == 'exit' and gestures[state['prev_gesture']] == 'default':
                            print('exit')
                        state['prev_gesture'] = gesture_idx
                else:
                    state = {'gesture':gesture_idx, 'start_time':time.time(), 'prev_gesture':state['prev_gesture']}
        else:
            landmark = []
            text_a = ''

    for i in range(len(landmark)):
        x = landmark[i].x * frame.shape[1]
        y = landmark[i].y * frame.shape[0]

        # Draw a circle at the fingertip position
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Print Gesture    
    cv2.putText(frame, text_a, (frame.shape[1] // 2 + 230, frame.shape[0] // 2 - 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

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
            cv2.putText(frame, str(cell), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if i == 0:
                cv2.rectangle(frame, (50 + cell_width * j, text_position[1] + 20),
                                (50 + cell_width * (j + 1), text_position[1] + 20 + cell_height), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (50 + cell_width * j, text_position[1] + 20 + i * cell_height),
                                (50 + cell_width * (j + 1), text_position[1] + 20 + (i + 1) * cell_height), (0, 255, 0), 2)

    # Display the output
    cv2.imshow('gesture recognition', frame)

    # Check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
print('landmark:', landmark_time/landmark_num)
print('gesture:', gesture_time / gesture_num)
