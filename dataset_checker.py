import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import time
from datetime import timedelta
import glob
import shutil


class Model(nn.Module):
    def __init__(self, input_size, hidden_dim1, hidden_dim2, target_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, target_size)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim = 0)
        self.relu = nn.ReLU()

    def forward(self, landmarks):
        x = self.dropout1(self.relu(self.fc1(landmarks)))
        x = self.dropout2(self.relu(self.fc2(x)))
        res = self.softmax(self.fc3(x))
        return res


INPUT_SIZE = 42
HIDDEN_DIM1 = 32
HIDDEN_DIM2 = 32
TARGET_SIZE = 8


test_data_dir = "./dataset/test_data.json"
data_dir = "./dataset/tmp/*"
file_list = glob.glob(data_dir)
file_list_json = [file for file in file_list if file.endswith(".json")]

test_dataset = json.load(open(test_data_dir))


def print_progress_bar(iteration, total, time_per_step, prefix="", suffix="", length=30, fill="="):
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
    train_dataset = dataset[ : int(len(dataset) * 0.9) ]
    val_dataset = dataset[ int(len(dataset) * 0.9) : ]

    model = Model(INPUT_SIZE, HIDDEN_DIM1, HIDDEN_DIM2, TARGET_SIZE)
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

            landmarks = torch.tensor([element for row in data["landmarks"] for element in row], dtype=torch.float)

            ans = [0 for _ in range(TARGET_SIZE)]
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
            landmarks = torch.tensor([element for row in data["landmarks"] for element in row], dtype=torch.float)
            ans = [0 for _ in range(TARGET_SIZE)]
            ans[data["gesture"]] = 1
            ans = torch.tensor(ans, dtype=torch.float)
            res = model(landmarks)
            loss = loss_function(res, ans)
            val_loss += loss.item()  # Accumulate loss value

        average_loss = val_loss / len(val_dataset)  # Calculate average loss for the epoch
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
            if data["gesture"] != 7:
                good += 1
        else:
            bad += 1
    accuracy = good / (good + bad) * 100
    print(f"Accuracy: {accuracy}")

    if accuracy > 90:
        print(data_dir, "is good dataset")
        shutil.move(file_list_json[0], "./dataset/used") 
    else:
        print(data_dir, "is bad dataset")
        shutil.move(file_list_json[0], "./dataset/unused") 