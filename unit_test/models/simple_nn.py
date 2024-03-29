import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(3, 40), num_classes=3):
        super(SimpleNN, self).__init__()
        self.input_shape = input_shape
        self.conv0 = nn.Conv1d(3, 32, (3,), padding='same')
        self.bn0 = nn.BatchNorm1d(32, track_running_stats=True)
        self.pool0 = nn.AvgPool1d(2)
        self.conv1 = nn.Conv1d(32, 57, (5,), padding='same')
        self.bn1 = nn.BatchNorm1d(57, track_running_stats=True)
        self.pool1 = nn.AvgPool1d(2)
        self.flatten = nn.Flatten(start_dim=1)
        self.dpout = nn.Dropout(0.5)
        self.fc = nn.Linear(57 * (input_shape[-1] // 2 // 2), num_classes)
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = F.relu6(self.pool0(self.bn0(self.conv0(x))))
        x = torch.relu(self.pool1(self.bn1(self.conv1(x))))
        # x = self.dpout(x.flatten(1))
        x = self.dpout(self.flatten(x))
        res = self.fc(x)
        return res


class SimpleNN2D(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(3, 40, 40), num_classes=3):
        super(SimpleNN2D, self).__init__()
        self.input_shape = input_shape
        self.conv0 = nn.Conv2d(3, 32, (3, 3), padding='same', bias=False)
        self.bn0 = nn.BatchNorm2d(32, track_running_stats=True)
        self.pool0 = nn.AvgPool2d(2)
        self.conv1 = nn.Conv2d(32, 57, (5, 5), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(57, track_running_stats=True)
        self.pool1 = nn.AvgPool2d(2)
        self.dpout = nn.Dropout(0.5)
        self.fc = nn.Linear(57 * (input_shape[-1] // 2 // 2)**2,
                            num_classes)
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = F.relu6(self.pool0(self.bn0(self.conv0(x))))
        x = F.relu6(self.pool1(self.bn1(self.conv1(x))))
        x = self.dpout(x.flatten(1))
        res = self.fc(x)
        return res


class SimpleNN2D_NoBN(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(3, 40, 40), num_classes=3):
        super(SimpleNN2D_NoBN, self).__init__()
        self.input_shape = input_shape
        self.conv0 = nn.Conv2d(3, 32, (3, 3), padding='same', bias=False)
        self.pool0 = nn.AvgPool2d(2)
        self.conv1 = nn.Conv2d(32, 57, (5, 5), padding='same', bias=False)
        self.pool1 = nn.AvgPool2d(2)
        self.dpout = nn.Dropout(0.5)
        self.fc = nn.Linear(57 * (input_shape[-1] // 2 // 2)**2,
                            num_classes)
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = F.relu6(self.pool0(self.conv0(x)))
        x = F.relu6(self.pool1(self.conv1(x)))
        x = self.dpout(x.flatten(1))
        res = self.fc(x)
        return res
