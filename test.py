import cv2
import mediapipe as mp
import math
import time
import torch
import torch.nn as nn
import json
from math import sqrt
import subprocess
import pygame
import asyncio


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
        positions.append([ center[0], center[1] ])

    min_idx = -1
    min_val = float('inf')
    for i, [x, y] in enumerate(positions):
        distance = (x-prev_pos[0])**2 + (y-prev_pos[1])**2
        if min_val > distance:
            min_idx = i
            min_val = distance

    if min_val > same_hand_threshold:
        return -1, prev_pos

    return min_idx, [positions[min_idx][0], positions[min_idx][1]]

def play_wav_file(file_name):
    pygame.mixer.init()
    pygame.mixer.music.load('./sound/'+file_name+'.wav')
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

INPUT_SIZE = 42
HIDDEN_DIM = 32
TARGET_SIZE = 6

model = Model(INPUT_SIZE, HIDDEN_DIM, TARGET_SIZE)
model.load_state_dict(torch.load('model.pt'))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands model
hands = mp_hands.Hands(max_num_hands=5)

# Open a webcam
w = 1280
h = 720
cap = cv2.VideoCapture(0)#1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    
cv2.namedWindow('gesture recognition')
cv2.createTrackbar('time','gesture recognition',10,100,nothing)
cv2.createTrackbar('same_hand','gesture recognition',5,100,nothing)
cv2.createTrackbar('skip_frame','gesture recognition',1,50,nothing)
cv2.createTrackbar('start_time','gesture recognition',2,10,nothing)
cv2.createTrackbar('stop_time','gesture recognition',2,10,nothing)

gestures = [ 'default', 'left', 'right', 'select', 'exit', 'none' ]
gesture_num = 0

state = {'gesture':5, 'start_time':time.time(), 'prev_gesture':5}

frame_num = 0

subprocess.run('adb connect 192.168.1.103:5555; adb root; adb connect 192.168.1.103:5555', shell=True)

landmark_time = 0
landmark_num = 0
gesture_time = 0
gesture_num = 0

recognizing_hands = []
recognizing_hand = []
text_a = ''
cur_gesture = gestures[5]
elapsed_time = '0'
prev_gesture = gestures[5]

recognizing = False
last_hand_time = time.time()

wake_up_state = []

visual_notification = ['', 0]

while cap.isOpened():    
    # require more time than time_threshold to recognize it as an gesture
    time_threshold = cv2.getTrackbarPos('time','gesture recognition')/100
    # distance between this frame's hand and last frame's recognized hand should be smaller than same_hand_threshold to regard them as same hand
    same_hand_threshold = cv2.getTrackbarPos('same_hand','gesture recognition')/1000
    landmark_skip_frame = max(cv2.getTrackbarPos('skip_frame','gesture recognition'), 1)
    start_recognizing_time_threshold = cv2.getTrackbarPos('start_time','gesture recognition')
    stop_recognizing_time_threshold = cv2.getTrackbarPos('stop_time','gesture recognition')

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
    
        if results.multi_hand_landmarks:
            right_hands = []
            for idx, hand in enumerate(results.multi_handedness):
                # label 'Left' means it is right hand because of left/right inversion
                if hand.classification[0].label == 'Left':
                    right_hands.append(list(map(lambda x : [x.x, x.y], results.multi_hand_landmarks[idx].landmark)))
            recognizing_hands = right_hands

            if recognizing:
                # find closest hand
                hand_idx, recognized_hand_prev_pos = same_hand_tracking(right_hands, recognized_hand_prev_pos, same_hand_threshold)

                if hand_idx != -1:
                    last_hand_time = time.time()

                    landmark = results.multi_hand_landmarks[hand_idx].landmark
                    landmark_lst = list(map(lambda x : [x.x, x.y], landmark))
                    recognizing_hand = landmark_lst
                    lst, scale = normalize_points(landmark_lst)

                    start = time.time_ns()//1000000
                    res = list(model(torch.tensor([element for row in lst for element in row], dtype=torch.float)))
                    end = time.time_ns()//1000000
                    gesture_time += end - start
                    gesture_num += 1

                    probability = max(res)
                    gesture_idx = res.index(probability) if probability >= 0.9 else 5
                    text_a = f"{gestures[gesture_idx]} {int(probability*100)}%"

                    cur_gesture = gestures[state['gesture']]
                    elapsed_time = str(round(time.time() - state['start_time'], 2))
                    prev_gesture = gestures[state['prev_gesture']]

                    if state['gesture'] == gesture_idx:
                        if time.time()-state['start_time'] > time_threshold:
                            if gestures[state['gesture']] == 'right' and gestures[state['prev_gesture']] == 'default':
                                subprocess.run('adb shell input keyevent KEYCODE_DPAD_RIGHT', shell=True)
                                print('right')
                                play_wav_file('action')
                                visual_notification = ['right', time.time()]
                            elif gestures[state['gesture']] == 'left' and gestures[state['prev_gesture']] == 'default':
                                subprocess.run('adb shell input keyevent KEYCODE_DPAD_LEFT', shell=True)
                                print('left')
                                play_wav_file('action')
                                visual_notification = ['left', time.time()]
                            elif gestures[state['gesture']] == 'select' and gestures[state['prev_gesture']] == 'default':
                                subprocess.run('adb shell input keyevent KEYCODE_BUTTON_SELECT', shell=True)
                                print('select')
                                play_wav_file('action')
                                visual_notification = ['select', time.time()]
                            elif gestures[state['gesture']] == 'exit' and (gestures[state['prev_gesture']] == 'default' or gestures[state['prev_gesture']] == 'left'):
                                subprocess.run('adb shell input keyevent KEYCODE_BACK', shell=True)
                                print('exit')
                                play_wav_file('action')
                                visual_notification = ['exit', time.time()]
                            state['prev_gesture'] = gesture_idx
                    else:
                        state = {'gesture':gesture_idx, 'start_time':time.time(), 'prev_gesture':state['prev_gesture']}
                else:
                    # stop recognizing
                    recognizing_hand = []
                    text_a = ''
                    if recognizing and time.time() - last_hand_time > stop_recognizing_time_threshold:
                        print('stop recognizing')
                        play_wav_file('stop')
                        recognizing = False
                        state = {'gesture':5, 'start_time':time.time(), 'prev_gesture':5}

                        cur_gesture = 'none'
                        elapsed_time = '0'
                        prev_gesture = 'none'
            else:
                # when not recognizing, get hands with 'default' gesture and measure elapsed time
                delete_list = []
                wake_up_hands = []
                for right_hand in right_hands:
                    lst, scale = normalize_points(right_hand)

                    res = list(model(torch.tensor([element for row in lst for element in row], dtype=torch.float)))
                    probability = max(res)
                    gesture_idx = res.index(probability) if probability >= 0.9 else 5
                    if gestures[gesture_idx] == 'default':
                        wake_up_hands.append(right_hand)

                checked = [0 for _ in range(len(wake_up_hands))]
                for i, [prev_pos, start_time] in enumerate(wake_up_state):
                    hand_idx, prev_pos = same_hand_tracking(wake_up_hands, prev_pos, same_hand_threshold)
                    if hand_idx == -1:
                        delete_list = [i] + delete_list
                    elif time.time()-start_time > start_recognizing_time_threshold:
                        # when there are default gestured hand for enough time, start recognizing and track the hand
                        print('start recognizing') 
                        recognized_hand_prev_pos = get_center(wake_up_hands[hand_idx])
                        play_wav_file('start')
                        recognizing = True
                        wake_up_state = []
                        break
                    else:
                        checked[hand_idx] = 1

                # wake_up_state refreshing
                if not recognizing:
                    for i in delete_list:
                        wake_up_state.pop(i)

                    for i in range(len(checked)):
                        if checked[i] == 0:
                            wake_up_state.append([get_center(wake_up_hands[i]), time.time()])
        else:
            # stop recognizing
            recognizing_hands = []
            recognizing_hand = []
            text_a = ''
            if recognizing and time.time() - last_hand_time > stop_recognizing_time_threshold:
                print('stop recognizing')
                play_wav_file('stop')
                recognizing = False  
                state = {'gesture':5, 'start_time':time.time(), 'prev_gesture':5}
                
                cur_gesture = 'none'
                elapsed_time = '0'
                prev_gesture = 'none'
                
    for rh in recognizing_hands:
        for x, y in rh:
            # Draw a circle at the fingertip position
            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)
    for x, y in recognizing_hand:
        # Draw a circle at the fingertip position
        cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (255, 0, 0), -1)

    # Print Current Hand's Gesture    
    cv2.putText(frame, text_a, (frame.shape[1] // 2 + 230, frame.shape[0] // 2 - 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # print recognized gesture
    if time.time() - visual_notification[1] < time_threshold * 2:
        cv2.putText(frame, visual_notification[0], (frame.shape[1] // 2 + 250, frame.shape[0] // 2 + 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

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
            cv2.putText(frame, str(cell), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if i == 0:
                cv2.rectangle(frame, (50 + cell_width * j, text_position[1] + 20),
                                (50 + cell_width * (j + 1), text_position[1] + 20 + cell_height), (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (50 + cell_width * j, text_position[1] + 20 + i * cell_height),
                                (50 + cell_width * (j + 1), text_position[1] + 20 + (i + 1) * cell_height), (255, 0, 0), 2)

    # Display the output
    cv2.imshow('gesture recognition', frame)

    # Check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# average inference time
print('landmark:', landmark_time / landmark_num)
print('gesture:', gesture_time / gesture_num)