import torch
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, input_shape=(1, 49, 12), num_classes=12):
        super(ResNet18, self).__init__()
        self.input_shape = input_shape
        self.layer0 = nn.Sequential(
            nn.Conv1d(49, 64, kernel_size=3, stride=2, padding=3),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, downsample=False),
            ResBlock(64, 64, downsample=False)
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128, downsample=False)
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256, downsample=False)
        )
        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(256, 12)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = self.relu1(self.bn1(self.conv1(input)))
        input = self.relu2(self.bn2(self.conv2(input)))
        input = input + shortcut
        return self.relu3(input)
