import glob
import json
import os
import random
import sys
import time
from datetime import timedelta
import torch
from torch import nn
from torch import optim

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.model import Model
import utils

def initialize_model():
    model = Model()
    model.load_state_dict(torch.load("../models/base_model.pt"))
    return model

if __name__ == "__main__":
    custom_gesture_dir = "../dataset/custom_gesture.json"

    if not os.path.exists(custom_gesture_dir):
        print("no custom gesture data error")
        sys.exit()

    custom_gesture_dataset = json.load(open(custom_gesture_dir))

    model = initialize_model()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Get the directory of train.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up one level to the parent directory of dataset
    parent_dir = os.path.dirname(current_dir)
    # Construct the path to dataset/tmp
    data_dir = os.path.join(parent_dir, "dataset", "used/*")
    file_list = glob.glob(data_dir)
    file_list_json = [file for file in file_list if file.endswith(".json")]

    dataset = []
    for dataset_dir in file_list_json:
        tmp = json.load(open(dataset_dir))
        dataset += tmp
    dataset += custom_gesture_dataset

    print(len(dataset), " data")
    
    random.shuffle(dataset)
    train_dataset = dataset[: int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8): int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]

    pretrain_total_epochs = 10
    pretrain_total_steps = len(custom_gesture_dataset)

    for epoch in range(pretrain_total_epochs):
        start_time = time.time()
        for step, data in enumerate(custom_gesture_dataset):
            optimizer.zero_grad()

            landmarks = torch.tensor(
                [element for row in data["landmarks"] for element in row],
                dtype=torch.float
            )

            ans = [0 for _ in range(model.target_size)]
            ans[data["gesture"]] = 1
            ans = torch.tensor(ans, dtype=torch.float)

            res = model(landmarks)

            loss = loss_function(res, ans)

            loss.backward()

            optimizer.step()

            # Display the progress bar
            utils.print_progress_bar(
                step + 1,
                pretrain_total_steps,
                time.time() - start_time,
                prefix=f"Epoch {epoch + 1}/{pretrain_total_epochs}",
                suffix=f"Loss: {loss.item(): .4f}",
            )
    print("\npretrain finished")

    total_epochs = 3
    total_steps = len(train_dataset)

    for epoch in range(total_epochs):
        start_time = time.time()
        for step, data in enumerate(train_dataset):
            optimizer.zero_grad()

            landmarks = torch.tensor(
                [element for row in data["landmarks"] for element in row],
                dtype=torch.float
            )

            ans = [0 for _ in range(model.target_size)]
            ans[data["gesture"]] = 1
            ans = torch.tensor(ans, dtype=torch.float)

            res = model(landmarks)

            loss = loss_function(res, ans)

            loss.backward()

            optimizer.step()

            # Display the progress bar
            utils.print_progress_bar(
                step + 1,
                total_steps,
                time.time() - start_time,
                prefix=f"Epoch {epoch + 1}/{total_epochs}",
                suffix=f"Loss: {loss.item(): .4f}",
            )

        # Prevent from overfitting
        val_loss = 0.0
        for data in val_dataset:
            landmarks = torch.tensor(
                [element for row in data["landmarks"] for element in row],
                dtype=torch.float
            )
            ans = [0 for _ in range(model.target_size)]
            ans[data["gesture"]] = 1
            ans = torch.tensor(ans, dtype=torch.float)
            res = model(landmarks)
            loss = loss_function(res, ans)
            val_loss += loss.item()  # Accumulate loss value

        average_loss = val_loss / len(val_dataset)  # Calculate average loss
        print(f"\nValidation Loss: {average_loss:.4f}\n")

    print("Training completed.")

    good = 0
    bad = 0
    for data in test_dataset:
        res = list(
            model(
                torch.tensor(
                    [element for row in data["landmarks"] for element in row],
                    dtype=torch.float,
                )
            )
        )
        if data["gesture"] == res.index(max(res)):
            good += 1
        else:
            bad += 1

    accuracy = good / (good + bad) * 100
    print(f"Accuracy: {accuracy}")
    
    if accuracy > 90:
        torch.save(model.state_dict(), "../models/model.pt")
    else:
        print("[ERROR] Your custom gesture is hard to identify. Please provide another gesture.")