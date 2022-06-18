import torch
import torch.nn as nn


class ToyModel1(nn.Module):
    def __init__(self, input_shape=(1, 3, 15), num_classes=3):
        super(ToyModel1, self).__init__()
        self.input_shape = input_shape
        self.conv0 = nn.Conv1d(3, 32, (3,), padding='same')
        self.conv1 = nn.Conv1d(3, 32, (3,), padding='same')
        self.conv2 = nn.Conv1d(3, 64, (3,), padding='same')
        self.conv3 = nn.Conv1d(3, 50, (3,), padding='same')
        self.conv4 = nn.Conv1d(50, 64, (3,), padding='same')
        self.conv5 = nn.Conv1d(64, 64, (3,), padding='same')
        self.fc = nn.Linear(15, 2)
        self.reluadd = nn.ReLU()

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        z = torch.cat((a, b), dim=1)
        z = self.conv5(z)
        y = self.conv2(x)
        w = self.conv3(x)
        w = self.conv4(w)
        x = self.reluadd(z + y + w)
        x = self.fc(x)
        return x


class ToyModel2(nn.Module):
    def __init__(self, input_shape=(1, 3, 60), num_classes=3):
        super(ToyModel2, self).__init__()
        self.input_shape = input_shape
        self.conv0 = nn.Conv1d(3, 40, (3,), padding='same')
        self.conv1 = nn.Conv1d(3, 40, (3,), padding='same')
        self.bn0 = nn.BatchNorm1d(40, track_running_stats=False)
        self.pool0 = nn.AvgPool1d(2)
        self.conv2 = nn.Conv1d(3, 20, (3,), padding='same')
        self.conv3 = nn.Conv1d(3, 20, (3,), padding='same')
        self.bn1 = nn.BatchNorm1d(20, track_running_stats=False)
        self.pool1 = nn.AvgPool1d(2)
        self.fc1 = nn.Linear(30, 40)
        self.fc2 = nn.Linear(40, 30)
        self.relu = nn.ReLU()
        self.reluadd = nn.ReLU()
        self.conv4 = nn.Conv1d(40, 15, (3,), padding='same')

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        z = torch.add(a, b)
        z = self.pool0(self.bn0(z))
        y = self.pool1(self.bn1(self.conv2(x)))
        w = self.pool1(self.bn1(self.conv3(x)))
        y = torch.cat((y, w), dim=1)
        y = self.relu(y)
        y = self.fc2(self.fc1(y))
        x = self.reluadd(y + z)
        x = self.conv4(x)
        return x


class ToyModel3(nn.Module):
    def __init__(self, input_shape=(1, 3, 15), num_classes=3):
        super(ToyModel3, self).__init__()
        self.input_shape = input_shape
        self.conv0 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv1 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv2 = nn.Conv1d(3, 20, (3,), padding='same')
        self.conv3 = nn.Conv1d(3, 8, (3,), padding='same')
        self.conv4 = nn.Conv1d(8, 20, (3,), padding='same')
        self.conv5 = nn.Conv1d(20, 20, (3,), padding='same')
        self.conv6 = nn.Conv1d(20, 20, (3,), padding='same')
        self.fc = nn.Linear(15, 2)
        self.reluadd = nn.ReLU()

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        z = torch.cat((a, b), dim=1)
        z = self.conv5(z)
        y = self.conv2(x)
        w = self.conv3(x)
        w = self.conv4(w)
        x = self.reluadd(z + y + w)
        x = self.fc(self.conv6(x))
        return x


class ToyModel4(nn.Module):
    def __init__(self, input_shape=(1, 3, 15), num_classes=3):
        super(ToyModel4, self).__init__()
        self.input_shape = input_shape
        self.conv0 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv1 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv2 = nn.Conv1d(10, 4, (3,), padding='same')
        self.reluadd = nn.ReLU()

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        x = self.reluadd(a + b)
        x = self.conv2(x)
        return x


class ToyModel5(nn.Module):
    def __init__(self, input_shape=(1, 3, 15), num_classes=3):
        super(ToyModel5, self).__init__()
        self.input_shape = input_shape
        self.conv0 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv1 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv2 = nn.Conv1d(20, 4, (3,), padding='same')

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        x = torch.cat((a, b), dim=1)
        x = self.conv2(x)
        return x


class ToyModel6(nn.Module):
    def __init__(self, input_shape=(1, 3, 15), num_classes=3):
        super(ToyModel6, self).__init__()
        self.input_shape = input_shape
        self.conv0 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv1 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv2 = nn.Conv1d(20, 4, (9,), padding='same')

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        x = torch.cat((a, b), dim=1)
        x = self.conv2(x)
        return x


class ToyModel7(nn.Module):
    def __init__(self, input_shape=(1, 3, 15), num_classes=3):
        super(ToyModel7, self).__init__()
        self.input_shape = input_shape
        self.conv0 = nn.Conv1d(3, 10, (7,), padding='same')
        self.conv1 = nn.Conv1d(3, 10, (7,), padding='same')
        self.conv2 = nn.Conv1d(20, 4, (9,), padding='same')

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        x = torch.cat((a, b), dim=1)
        x = self.conv2(x)
        return x


class ToyModel8(nn.Module):
    def __init__(self, input_shape=(2, 3, 8), num_classes=3):
        super(ToyModel8, self).__init__()
        self.input_shape = input_shape
        self.conv0 = nn.Conv1d(3, 4, (3,), padding='same')
        self.conv1 = nn.Conv1d(3, 4, (3,), padding='same')
        self.conv2 = nn.Conv1d(8, 2, (3,), padding='same')
        self.conv3 = nn.Conv1d(2, 1, (2,), padding='same')
        self.pool0 = nn.AvgPool1d(2)
        self.pool1 = nn.AvgPool1d(2)
        self.bn0 = nn.BatchNorm1d(2, track_running_stats=False)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        x = torch.cat((a, b), dim=1)
        x = self.pool0(self.bn0(self.conv2(x)))
        x = self.conv3(x)
        x = torch.squeeze(x)
        # x = x.flatten(1)
        x = self.fc(x)

        return x
