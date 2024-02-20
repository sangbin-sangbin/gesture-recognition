import glob
import json
import random
import shutil
import time
from datetime import timedelta
import sys
import os

import torch
from torch import nn
from torch import optim

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.model import Model

test_data_dir = "../dataset/test_data.json"
data_dir = "../dataset/tmp/*"
file_list = glob.glob(data_dir)
file_list_json = [file for file in file_list if file.endswith(".json")]

test_dataset = json.load(open(test_data_dir))


def print_progress_bar(
    iteration, total, time_per_step, prefix="", suffix="", length=30, fill="="
):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    time_str = str(timedelta(seconds=int(time_per_step * total)))
    print(
        f"\r{prefix} [{bar}] - {percent}% - {suffix} - {time_str}/epoch",
        end="",
        flush=True,
    )


for data_dir in file_list_json:
    dataset = json.load(open(data_dir))
    random.shuffle(dataset)
    train_dataset = dataset[: int(len(dataset) * 0.9)]
    val_dataset = dataset[int(len(dataset) * 0.9):]

    model = Model()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    total_epochs = 10
    total_steps = len(train_dataset)

    prev_loss = float("inf")
    tolerance = 20

    for epoch in range(total_epochs):
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

            # Display the progress bar
            print_progress_bar(
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

        if prev_loss < average_loss:
            tolerance -= 1
            if tolerance == 0:
                break

        prev_loss = average_loss

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
        print(data_dir, "is good dataset")
        shutil.move(file_list_json[0], "../dataset/used")
    else:
        print(data_dir, "is bad dataset")
        shutil.move(file_list_json[0], "../dataset/unused")