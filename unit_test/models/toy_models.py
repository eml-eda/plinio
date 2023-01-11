import torch
import torch.nn as nn
import torch.nn.functional as F


class ToySequentialConv1d(nn.Module):
    def __init__(self):
        super(ToySequentialConv1d, self).__init__()
        self.input_shape = (3, 12)
        self.conv0 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv1 = nn.Conv1d(10, 20, (9,), padding='same')

    def forward(self, x):
        return self.conv1(self.conv0(x))


class ToySequentialSeparated(nn.Module):
    def __init__(self):
        super(ToySequentialSeparated, self).__init__()
        self.input_shape = (3, 10)
        self.conv0 = nn.Conv1d(3, 10, (3,), padding='same')
        self.bn0 = nn.BatchNorm1d(10, track_running_stats=True)
        self.pool0 = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(10, 20, (9,), padding='same')

    def forward(self, x):
        return self.conv1(F.relu6(self.pool0(self.bn0(self.conv0(x)))))


class ToyAdd(nn.Module):
    def __init__(self):
        super(ToyAdd, self).__init__()
        self.input_shape = (3, 15)
        self.conv0 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv1 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv2 = nn.Conv1d(10, 20, (5,), padding='same')
        self.reluadd = nn.ReLU()
        self.pool = nn.MaxPool1d((2,))
        self.fc = nn.Linear(20 * 7, 2)

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        x = self.reluadd(a + b)
        y = self.conv2(self.pool(x))
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return y


class ToyTimeCat(nn.Module):
    def __init__(self):
        super(ToyTimeCat, self).__init__()
        self.input_shape = (3, 20)
        self.conv0 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv1 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv2 = nn.Conv1d(10, 32, (3,), padding='same')

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        x = torch.cat((a, b), dim=2)
        y = self.conv2(x)
        return y


class ToyChannelsCat(nn.Module):
    def __init__(self):
        super(ToyChannelsCat, self).__init__()
        self.input_shape = (3, 24)  # fully-convolutional
        self.conv0 = nn.Conv1d(3, 10, (3,), padding='same')
        self.conv1 = nn.Conv1d(3, 15, (3,), padding='same')
        self.conv2 = nn.Conv1d(25, 32, (3,), padding='same')

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        x = torch.cat((a, b), dim=1)
        y = self.conv2(x)
        return y


class ToyFlatten(nn.Module):
    def __init__(self):
        super(ToyFlatten, self).__init__()
        self.input_shape = (3, 8)
        self.conv0 = nn.Conv1d(3, 4, (3,), padding='same')
        self.conv1 = nn.Conv1d(4, 2, (3,), padding='same')
        self.conv2 = nn.Conv1d(2, 1, (2,), padding='same')
        self.pool0 = nn.AvgPool1d(2)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.pool0(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class ToyMultiPath1(nn.Module):
    def __init__(self):
        super(ToyMultiPath1, self).__init__()
        self.input_shape = (3, 10)
        self.conv0 = nn.Conv1d(3, 32, (3,), padding='same')
        self.conv1 = nn.Conv1d(3, 32, (3,), padding='same')
        self.conv2 = nn.Conv1d(3, 64, (3,), padding='same')
        self.conv3 = nn.Conv1d(3, 50, (3,), padding='same')
        self.conv4 = nn.Conv1d(50, 64, (3,), padding='same')
        self.conv5 = nn.Conv1d(64, 64, (3,), padding='same')
        self.fc = nn.Linear(640, 2)
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
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class ToyMultiPath1_2D(nn.Module):
    def __init__(self):
        super(ToyMultiPath1_2D, self).__init__()
        self.input_shape = (3, 10, 10)
        self.conv0 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.conv2 = nn.Conv2d(3, 64, (3, 3), padding='same')
        self.conv3 = nn.Conv2d(3, 50, (3, 3), padding='same')
        self.conv4 = nn.Conv2d(50, 64, (3, 3), padding='same')
        self.conv5 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.fc = nn.Linear(6400, 2)
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
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class ToyMultiPath2(nn.Module):
    def __init__(self):
        super(ToyMultiPath2, self).__init__()
        self.input_shape = (3, 60)
        self.conv0 = nn.Conv1d(3, 40, (3,), padding='same')
        self.conv1 = nn.Conv1d(3, 40, (3,), padding='same')
        self.bn0 = nn.BatchNorm1d(40, track_running_stats=True)
        self.pool0 = nn.AvgPool1d(2)
        self.conv2 = nn.Conv1d(3, 20, (3,), padding='same')
        self.conv3 = nn.Conv1d(3, 20, (3,), padding='same')
        self.bn1 = nn.BatchNorm1d(20, track_running_stats=True)
        self.pool1 = nn.AvgPool1d(2)
        self.relu = nn.ReLU()
        self.reluadd = nn.ReLU()
        self.conv4 = nn.Conv1d(40, 15, (3,), padding='same')
        self.fc = nn.Linear(450, 10)

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        z = torch.add(a, b)
        z = self.pool0(self.bn0(z))
        y = self.pool1(self.bn1(self.conv2(x)))
        w = self.pool1(self.bn1(self.conv3(x)))
        y = torch.cat((y, w), dim=1)
        y = self.relu(y)
        x = self.reluadd(y + z)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ToyMultiPath2_2D(nn.Module):
    def __init__(self):
        super(ToyMultiPath2_2D, self).__init__()
        self.input_shape = (3, 10, 10)
        self.conv0 = nn.Conv2d(3, 40, (3, 3), padding='same')
        self.conv1 = nn.Conv2d(3, 40, (3, 3), padding='same')
        self.bn0 = nn.BatchNorm2d(40, track_running_stats=True)
        self.pool0 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(3, 20, (3, 3), padding='same')
        self.conv3 = nn.Conv2d(3, 20, (3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(20, track_running_stats=True)
        self.pool1 = nn.AvgPool2d(2)
        self.relu = nn.ReLU()
        self.reluadd = nn.ReLU()
        self.conv4 = nn.Conv2d(40, 15, (3, 3), padding='same')
        self.fc = nn.Linear(375, 10)

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        z = torch.add(a, b)
        z = self.pool0(self.bn0(z))
        y = self.pool1(self.bn1(self.conv2(x)))
        w = self.pool1(self.bn1(self.conv3(x)))
        y = torch.cat((y, w), dim=1)
        y = self.relu(y)
        x = self.reluadd(y + z)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ToyRegression(nn.Module):
    def __init__(self):
        super(ToyRegression, self).__init__()
        self.input_shape = (1, 10)
        self.conv0 = nn.Conv1d(1, 10, (3,), padding='same')
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
