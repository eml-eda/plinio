from typing import cast
import torch
import torch.nn as nn
from flexnas.methods import SuperNetModule


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        #                         padding=padding, bias=False)

        self.conv1 = SuperNetModule([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=padding, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 5, padding=2 * padding, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, groups=in_channels,
                          padding=padding, stride=stride),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            # nn.Identity()
        ])
        # nn.init.kaiming_normal_(self.conv1.weight)
        it = self.conv1[0].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        it = self.conv1[1].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        children = list(self.conv1[2].children())
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[3]).weight)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        # return self.relu(self.bn(x))
        return x


class ResNet8SN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Resnet v1 parameters
        self.input_shape = [3, 32, 32]  # default size for cifar10
        self.num_classes = 10  # default class number for cifar10
        self.num_filters = 16  # this should be 64 for an official resnet model

        # Resnet v1 layers

        # First stack
        self.inputblock = ConvBlock(in_channels=3, out_channels=16,
                                    kernel_size=3, stride=1, padding=1)
        self.convblock1 = ConvBlock(in_channels=16, out_channels=16,
                                    kernel_size=3, stride=1, padding=1)

        # self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv1 = SuperNetModule([
            nn.Sequential(
                nn.Conv2d(16, 16, 3, padding='same'),
                nn.BatchNorm2d(16),
            ),
            nn.Sequential(
                nn.Conv2d(16, 16, 5, padding='same'),
                nn.BatchNorm2d(16),
            ),
            nn.Sequential(
                nn.Conv2d(16, 16, 3, groups=16, padding='same'),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, 1),
                nn.BatchNorm2d(16),
            ),
            nn.Identity()
        ])
        # nn.init.kaiming_normal_(self.conv1.weight)
        it = self.conv1[0].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        it = self.conv1[1].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        children = list(self.conv1[2].children())
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[3]).weight)
        # self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # Second stack
        self.convblock2 = ConvBlock(in_channels=16, out_channels=32,
                                    kernel_size=3, stride=2, padding=1)

        # self.conv2y = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2y = SuperNetModule([
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding='same'),
                nn.BatchNorm2d(32)
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 5, padding='same'),
                nn.BatchNorm2d(32)
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, groups=32, padding='same'),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 1),
                nn.BatchNorm2d(32)
            ),
            nn.Identity()
        ])
        # nn.init.kaiming_normal_(self.conv2y.weight)
        it = self.conv2y[0].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        it = self.conv2y[1].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        children = list(self.conv2y[2].children())
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[3]).weight)
        # self.bn2 = nn.BatchNorm2d(32)

        self.conv2x = nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0)
        self.bn2x = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv2x.weight)

        # Third stack
        self.convblock3 = ConvBlock(in_channels=32, out_channels=64,
                                    kernel_size=3, stride=2, padding=1)

        # self.conv3y = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3y = SuperNetModule([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding='same'),
                nn.BatchNorm2d(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 5, padding='same'),
                nn.BatchNorm2d(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, groups=64, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 1),
                nn.BatchNorm2d(64)
            ),
            nn.Identity()
        ])
        # nn.init.kaiming_normal_(self.conv3y.weight)
        it = self.conv3y[0].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        it = self.conv3y[1].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        children = list(self.conv3y[2].children())
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[3]).weight)
        # self.bn3 = nn.BatchNorm2d(64)

        self.conv3x = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)
        self.bn3x = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.conv3x.weight)

        self.avgpool = torch.nn.AvgPool2d(8)

        self.out = nn.Linear(64, 10)
        nn.init.kaiming_normal_(self.out.weight)

    def forward(self, input):
        # Input layer
        x = self.inputblock(input)  # [32, 32, 16]

        # First stack
        y = self.convblock1(x)      # [32, 32, 16]
        y = self.conv1(y)
        # y = self.bn1(y)
        x = torch.add(x, y)         # [32, 32, 16]
        x = self.relu(x)

        # Second stack
        y = self.convblock2(x)      # [16, 16, 32]
        y = self.conv2y(y)
        # y = self.bn2(y)
        x = self.conv2x(x)          # [16, 16, 32]
        x = self.bn2x(x)
        x = torch.add(x, y)         # [16, 16, 32]
        x = self.relu(x)

        # Third stack
        y = self.convblock3(x)      # [8, 8, 64]
        y = self.conv3y(y)
        # y = self.bn3(y)
        x = self.conv3x(x)          # [8, 8, 64]
        x = self.bn3x(x)
        x = torch.add(x, y)         # [8, 8, 64]
        x = self.relu(x)

        x = self.avgpool(x)         # [1, 1, 64]
        # x = torch.squeeze(x)        # [64]
        x = torch.flatten(x, 1)
        x = self.out(x)             # [10]

        return x
