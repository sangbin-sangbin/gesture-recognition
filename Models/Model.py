from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        input_size=42,
        hidden_dim1=32,
        hidden_dim2=32,
        target_size=8
    ):
        super(Model, self).__init__()
        self.target_size = target_size

        self.fc1 = nn.Linear(input_size, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, target_size)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, landmarks):
        x = self.dropout1(self.relu(self.fc1(landmarks)))
        x = self.dropout2(self.relu(self.fc2(x)))
        res = self.softmax(self.fc3(x))
        return res
