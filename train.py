import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import random
from tqdm import tqdm
import time
from datetime import timedelta
 
 
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
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
 
dataset = json.load(open('dataset2.json'))
random.shuffle(dataset)
train_dataset = dataset[:int(len(dataset)*0.8)]
val_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
test_dataset = dataset[int(len(dataset)*0.9):]
 
def print_progress_bar(iteration, total, time_per_step, prefix='', suffix='', length=30, fill='='):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    time_str = str(timedelta(seconds=int(time_per_step * total)))
    print(f'\r{prefix} [{bar}] - {percent}% - {suffix} - {time_str}/step', end='', flush=True)
 
total_epochs = 20
total_steps = 100
time_per_step = 0.01

prev_loss = float('inf')
tolerance = 3
 
for epoch in range(total_epochs): 
    for data in train_dataset:
        model.zero_grad()
 
        landmarks = torch.tensor([element for row in data['landmarks'] for element in row], dtype=torch.float)
 
        ans = [0 for _ in range(TARGET_SIZE)]
        ans[data['gesture']] = 1
        ans = torch.tensor(ans, dtype=torch.float)
 
        res = model(landmarks)
 
        loss = loss_function(res, ans)
 
        optimizer.zero_grad()
 
        loss.backward()
 
        optimizer.step()
  
    # Loop through steps with a progress bar
    for step in range(1, total_steps + 1):
        # Simulate some processing time (replace this with your actual training code)
        time.sleep(time_per_step)
 
        # Display the progress bar
        print_progress_bar(step, total_steps, time_per_step, prefix=f'Epoch {epoch}/{total_epochs}', length=30, fill='=')

    val_loss = 0.0
    for data in val_dataset:
        landmarks = torch.tensor([element for row in data['landmarks'] for element in row], dtype=torch.float)
        ans = [0 for _ in range(TARGET_SIZE)]
        ans[data['gesture']] = 1
        ans = torch.tensor(ans, dtype=torch.float)
        res = model(landmarks)
        loss = loss_function(res, ans)
        val_loss += loss.item()  # Accumulate loss value

    average_loss = val_loss / len(val_dataset)  # Calculate average loss for the epoch
    print(f'\nValidation Loss: {average_loss:.4f}\n')

    if prev_loss < average_loss:
        tolerance -= 1
        if tolerance == 0:
            break

    prev_loss = average_loss
    
print("Training completed.")
 
good = 0
bad = 0
for data in test_dataset:
    res = list(model(torch.tensor([element for row in data['landmarks'] for element in row], dtype=torch.float)))
    if data['gesture'] == res.index(max(res)):
        good += 1
    else:
        bad += 1
print(f"Accuracy: { good / (good + bad) * 100}")

torch.save(model.state_dict(), './model.pt')