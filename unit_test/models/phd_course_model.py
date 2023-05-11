import torch
import torch.nn as nn


class Conv3x3Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, input):
        x = self.conv1(input)
        return x


class Conv5x5Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, padding=2, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, input):
        x = self.conv1(input)
        return x


class DWSBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3,
                      groups=in_channels, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, input):
        x = self.conv1(input)
        return x


class TutorialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (3, 32, 32)
        # ch = [16, 32, 64, 128, 256]  # Original channels
        ch = [16, 29, 64, 122, 191]  # Channels after PIT
        self.block1 = Conv3x3Block(3, ch[0])
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.block2 = Conv3x3Block(ch[0], ch[1])
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.block3 = DWSBlock(ch[1], ch[2])
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.block4 = DWSBlock(ch[2], ch[3])
        self.maxpool4 = nn.MaxPool2d(2, stride=2)
        self.block5 = DWSBlock(ch[3], ch[4])
        self.maxpool5 = nn.MaxPool2d(2, stride=2)
        self.out = nn.Linear(ch[4], 10)

    def forward(self, input):
        x = self.block1(input)  # [16, 16, 16]
        x = self.maxpool1(x)  # [8, 8, 32]
        x = self.block2(x)  # [8, 8, 32]
        x = self.maxpool2(x)  # [8, 8, 32]
        x = self.block3(x)  # [4, 4, 64]
        x = self.maxpool3(x)  # [8, 8, 32]
        x = self.block4(x)  # [2, 2, 128]
        x = self.maxpool4(x)  # [8, 8, 32]
        x = self.block5(x)  # [1, 1, 256]
        x = self.maxpool5(x)  # [8, 8, 32]
        x = torch.flatten(x, 1)
        x = self.out(x)  # [10]
        return x
