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
TARGET_SIZE = 5

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
cv2.createTrackbar('speed','gesture recognition',20,500,nothing)
cv2.createTrackbar('time','gesture recognition',10,100,nothing)
cv2.createTrackbar('distance','gesture recognition',50,500,nothing)
cv2.createTrackbar('tolerance','gesture recognition',5,50,nothing)
cv2.createTrackbar('skip_frame','gesture recognition',3,50,nothing)

gestures = [ 'left', 'right', 'select', 'exit', 'none' ]
gesture_num = 0

directions = [ 'right', 'left', 'down', 'up', 'stop' ]
speed_threshold = cv2.getTrackbarPos('speed','gesture recognition') #20
time_threshold = cv2.getTrackbarPos('time','gesture recognition')/100 #0.1
distance_threshold = cv2.getTrackbarPos('distance','gesture recognition') #50
default_tolerance = cv2.getTrackbarPos('tolerance','gesture recognition') #5
state = {'gesture':4, 'start_time':time.time(), 'direction':0, 'prev_pos':[0,0], 'first_pos':[0,0], 'tolerance':default_tolerance}

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
text_b = ''

while cap.isOpened():    
    speed_threshold = cv2.getTrackbarPos('speed','gesture recognition')
    time_threshold = cv2.getTrackbarPos('time','gesture recognition')/100
    distance_threshold = cv2.getTrackbarPos('distance','gesture recognition')
    default_tolerance = cv2.getTrackbarPos('tolerance','gesture recognition')
    landmark_skip_frame = max(cv2.getTrackbarPos('skip_frame','gesture recognition'), 1)

    # Read a frame from the webcam
    ret, frame = cap.read()
    if frame is None:
        continue
 
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_num += 1
    if frame_num % landmark_skip_frame == 0:
        # Process the frame with MediaPipe Hands
        s = time.time_ns()//1000000
        results = hands.process(rgb_frame)
        e = time.time_ns()//1000000
        landmark_time += e - s
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

                s = time.time_ns()//1000000
                res = list(model(torch.tensor([element for row in lst for element in row], dtype=torch.float)))
                e = time.time_ns()//1000000
                gesture_time += e - s
                gesture_num += 1

                p = max(res)
                gesture_idx = res.index(p) if p >= 0.9 else 4
                text_a = gestures[gesture_idx]+' '+str(int(p*100))

                pos_x = results.multi_hand_landmarks[hand_idx].landmark[9].x * frame.shape[1]
                pos_y = results.multi_hand_landmarks[hand_idx].landmark[9].y * frame.shape[0]

                d = direction([pos_x, pos_y], state['prev_pos'])
                spd = distance([pos_x, pos_y], state['prev_pos']) / scale
                text_b = directions[d]+' '+str(int(spd))+' '+str(int(distance(state['first_pos'], [pos_x, pos_y]) / scale))

                if state['gesture'] == gesture_idx and spd > speed_threshold and d == state['direction']:
                    state['prev_pos'] = [pos_x, pos_y]
                elif state['tolerance'] > 0:
                    state['prev_pos'] = [pos_x, pos_y]
                    state['tolerance'] -= 1
                else:
                    if time.time()-state['start_time'] > time_threshold and distance(state['first_pos'], [pos_x, pos_y]) / scale >= distance_threshold:
                        if gestures[gesture_idx] == 'right' and directions[state['direction']] == 'right':
                            print('right')
                            subprocess.run('adb shell input tap 80 600', shell=True)
                        elif gestures[gesture_idx] == 'left' and directions[state['direction']] == 'left':
                            print('left')
                            subprocess.run('adb shell input tap 80 500', shell=True)
                        if gestures[gesture_idx] == 'select' and directions[state['direction']] == 'down':
                            print('select')
                            subprocess.run('adb shell input tap 80 720', shell=True)
                        elif gestures[gesture_idx] == 'exit' and directions[state['direction']] == 'right':
                            print('exit')
                            subprocess.run('adb shell input tap 80 820', shell=True)
                    elif time.time()-state['start_time'] > time_threshold and distance(state['first_pos'], [pos_x, pos_y]) / scale >= distance_threshold * 0.5:
                        if gestures[gesture_idx] == 'select' and directions[state['direction']] == 'down':
                            print('select')
                            subprocess.run('adb shell input tap 80 720', shell=True)
 
                    state = {'gesture':gesture_idx, 'start_time':time.time(), 'direction':d, 'prev_pos':[pos_x, pos_y], 'first_pos':[pos_x, pos_y], 'tolerance':default_tolerance}
        else:
            landmark = []
            text_a = ''
            text_b = ''          

    for i in range(len(landmark)):
        x = landmark[i].x * frame.shape[1]
        y = landmark[i].y * frame.shape[0]

        # Draw a circle at the fingertip position
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
    cv2.putText(frame, text_a, (frame.shape[1] // 2, frame.shape[0] // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(frame, text_b, (frame.shape[1] // 2 - 100, frame.shape[0] // 2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

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
