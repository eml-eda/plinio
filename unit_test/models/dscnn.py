import torch
import torch.nn as nn


class DSCNN(torch.nn.Module):
    def __init__(self, input_shape=(1, 49, 10)):
        self.input_shape = input_shape
        super().__init__()

        # Model layers

        # Input pure conv2d
        self.inputlayer = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(10, 4), stride=(2, 2), padding=(5, 1))
        self.bn = nn.BatchNorm2d(64, momentum=0.99)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)

        # First layer of separable depthwise conv2d
        # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
        self.depthwise1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn11 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu11 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn12 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu12 = nn.ReLU()

        # Second layer of separable depthwise conv2d
        self.depthwise2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn21 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu21 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn22 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu22 = nn.ReLU()

        # Third layer of separable depthwise conv2d
        self.depthwise3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn31 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu31 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn32 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu32 = nn.ReLU()

        # Fourth layer of separable depthwise conv2d
        self.depthwise4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn41 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu41 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn42 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu42 = nn.ReLU()

        self.dropout2 = nn.Dropout(p=0.4)
        self.avgpool = torch.nn.AvgPool2d((25, 5))
        self.out = nn.Linear(64, 12)

    def forward(self, input):

        # Input pure conv2d
        x = self.inputlayer(input)
        x = self.dropout1(self.relu(self.bn(x)))

        # First layer of separable depthwise conv2d
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.relu11(self.bn11(x))
        x = self.conv1(x)
        x = self.relu12(self.bn12(x))

        # Second layer of separable depthwise conv2d
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.relu21(self.bn21(x))
        x = self.conv2(x)
        x = self.relu22(self.bn22(x))

        # Third layer of separable depthwise conv2d
        x = self.depthwise3(x)
        x = self.pointwise3(x)
        x = self.relu31(self.bn31(x))
        x = self.conv3(x)
        x = self.relu32(self.bn32(x))

        # Fourth layer of separable depthwise conv2d
        x = self.depthwise4(x)
        x = self.pointwise4(x)
        x = self.relu41(self.bn41(x))
        x = self.conv4(x)
        x = self.relu42(self.bn42(x))

        x = self.dropout2(x)
        x = self.avgpool(x)
        # x = torch.squeeze(x)
        x = x.flatten(1)
        x = self.out(x)

        return x
