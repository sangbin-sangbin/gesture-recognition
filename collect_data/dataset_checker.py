import glob
import json
import os
import random
import shutil
import sys
import time

import torch
from torch import nn, optim

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import utils
from models.model import Model


TEST_DATA_DIR = os.path.join("..", "dataset", "test_data.json")
TMP_DATA_DIR = os.path.join("..", "dataset", "tmp/*")
tmp_file_list = glob.glob(TMP_DATA_DIR)
tmp_file_list_json = [file for file in tmp_file_list if file.endswith(".json")]

BASE_DATA_DIR = os.path.join("..", "dataset", "used/*")
file_list = glob.glob(BASE_DATA_DIR)
file_list_json = [file for file in file_list if file.endswith(".json")]

base_dataset = []
for dataset_dir in file_list_json:
    tmp = json.load(open(dataset_dir))
    base_dataset += tmp
print(len(base_dataset), " data")
random.shuffle(base_dataset)

for tmp_data_dir in tmp_file_list_json:
    tmp_dataset = json.load(open(tmp_data_dir))
    dataset = base_dataset + tmp_dataset
    random.shuffle(dataset)
    train_dataset = dataset[: int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8): int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]

    model = Model()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    TOTAL_EPOCHS = 10
    total_steps = len(train_dataset)
    average_train_loss = 0

    for epoch in range(TOTAL_EPOCHS):
        start_time = time.time()
        for step, data in enumerate(train_dataset):
            model.zero_grad()

            landmarks = torch.tensor(
                [element for row in data["landmarks"] for element in row],
                dtype=torch.float,
            )

            ans = [0 for _ in range(model.target_size)]
            ans[data["gesture"]] = 1
            ans = torch.tensor(ans, dtype=torch.float)

            res = model(landmarks)

            loss = loss_function(res, ans)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            average_train_loss = (average_train_loss * step + loss.item()) / (step + 1)
            # Display the progress bar
            utils.print_progress_bar(
                step + 1,
                total_steps,
                time.time() - start_time,
                prefix=f"Epoch {epoch + 1}/{TOTAL_EPOCHS}",
                suffix=f"Loss: {average_train_loss: .4f}",
            )

        # Prevent from overfitting
        val_loss = 0.0
        for data in val_dataset:
            landmarks = torch.tensor(
                [element for row in data["landmarks"] for element in row],
                dtype=torch.float,
            )
            ans = [0 for _ in range(model.target_size)]
            ans[data["gesture"]] = 1
            ans = torch.tensor(ans, dtype=torch.float)
            res = model(landmarks)
            loss = loss_function(res, ans)
            val_loss += loss.item()  # Accumulate loss value

        average_loss = val_loss / len(
            val_dataset
        )  # Calculate average loss for the epoch
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
        print(tmp_data_dir, "is good dataset")
        dataset_dir = os.path.join("..", "dataset", "used")
        shutil.move(tmp_data_dir, dataset_dir)
    else:
        print(tmp_data_dir, "is bad dataset")
        dataset_dir = os.path.join("..", "dataset", "unused")
        shutil.move(tmp_data_dir, dataset_dir)
