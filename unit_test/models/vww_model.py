import torch
import torch.nn as nn


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False,
                               groups=groups)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class MobileNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # MobileNet v1 parameters
        self.input_shape = [3, 96, 96]  # default size for coco dataset
        self.num_classes = 2  # binary classification: person or non person
        self.num_filters = 8

        # MobileNet v1 layers

        # 1st layer
        self.inputblock = ConvBlock(in_channels=3, out_channels=8,
                                    kernel_size=3, stride=2, padding=1)

        # 2nd layer
        self.depthwise2 = ConvBlock(in_channels=8, out_channels=8,
                                    kernel_size=3, stride=1, padding=1, groups=8)
        self.pointwise2 = ConvBlock(in_channels=8, out_channels=16,
                                    kernel_size=1, stride=1, padding=0)

        # 3d layer
        self.depthwise3 = ConvBlock(in_channels=16, out_channels=16,
                                    kernel_size=3, stride=2, padding=1, groups=16)
        self.pointwise3 = ConvBlock(in_channels=16, out_channels=32,
                                    kernel_size=1, stride=1, padding=0)

        # 4th layer
        self.depthwise4 = ConvBlock(in_channels=32, out_channels=32,
                                    kernel_size=3, stride=1, padding=1, groups=32)
        self.pointwise4 = ConvBlock(in_channels=32, out_channels=32,
                                    kernel_size=1, stride=1, padding=0)

        # 5h layer
        self.depthwise5 = ConvBlock(in_channels=32, out_channels=32,
                                    kernel_size=3, stride=2, padding=1, groups=32)
        self.pointwise5 = ConvBlock(in_channels=32, out_channels=64,
                                    kernel_size=1, stride=1, padding=0)

        # 6th layer
        self.depthwise6 = ConvBlock(in_channels=64, out_channels=64,
                                    kernel_size=3, stride=1, padding=1, groups=64)
        self.pointwise6 = ConvBlock(in_channels=64, out_channels=64,
                                    kernel_size=1, stride=1, padding=0)

        # 7th layer
        self.depthwise7 = ConvBlock(in_channels=64, out_channels=64,
                                    kernel_size=3, stride=2, padding=1, groups=64)
        self.pointwise7 = ConvBlock(in_channels=64, out_channels=128,
                                    kernel_size=1, stride=1, padding=0)

        # 8th layer
        self.depthwise8 = ConvBlock(in_channels=128, out_channels=128,
                                    kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise8 = ConvBlock(in_channels=128, out_channels=128,
                                    kernel_size=1, stride=1, padding=0)

        # 9th layer
        self.depthwise9 = ConvBlock(in_channels=128, out_channels=128,
                                    kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise9 = ConvBlock(in_channels=128, out_channels=128,
                                    kernel_size=1, stride=1, padding=0)

        # 10th layer
        self.depthwise10 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise10 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=1, stride=1, padding=0)

        # 11th layer
        self.depthwise11 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise11 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=1, stride=1, padding=0)

        # 12th layer
        self.depthwise12 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise12 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=1, stride=1, padding=0)

        # 13th layer
        self.depthwise13 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=3, stride=2, padding=1, groups=128)
        self.pointwise13 = ConvBlock(in_channels=128, out_channels=256,
                                     kernel_size=1, stride=1, padding=0)

        # 14th layer
        self.depthwise14 = ConvBlock(in_channels=256, out_channels=256,
                                     kernel_size=3, stride=1, padding=1, groups=256)
        self.pointwise14 = ConvBlock(in_channels=256, out_channels=256,
                                     kernel_size=1, stride=1, padding=0)

        self.avgpool = torch.nn.AvgPool2d(3)

        self.out = nn.Linear(256, 2)
        nn.init.kaiming_normal_(self.out.weight)

    def forward(self, input):

        # Input tensor shape        # [96, 96,  3]

        # 1st layer
        x = self.inputblock(input)  # [48, 48,  8]

        # 2nd layer
        x = self.depthwise2(x)      # [48, 48,  8]
        x = self.pointwise2(x)      # [48, 48, 16]

        # 3rd layer
        x = self.depthwise3(x)      # [24, 24, 16]
        x = self.pointwise3(x)      # [24, 24, 32]

        # 4th layer
        x = self.depthwise4(x)      # [24, 24, 32]
        x = self.pointwise4(x)      # [24, 24, 32]

        # 5th layer
        x = self.depthwise5(x)      # [12, 12, 32]
        x = self.pointwise5(x)      # [12, 12, 64]

        # 6th layer
        x = self.depthwise6(x)      # [12, 12, 64]
        x = self.pointwise6(x)      # [12, 12, 64]

        # 7th layer
        x = self.depthwise7(x)      # [ 6,  6, 64]
        x = self.pointwise7(x)      # [ 6,  6, 128]

        # 8th layer
        x = self.depthwise8(x)      # [ 6,  6, 128]
        x = self.pointwise8(x)      # [ 6,  6, 128]

        # 9th layer
        x = self.depthwise9(x)      # [ 6,  6, 128]
        x = self.pointwise9(x)      # [ 6,  6, 128]

        # 10th layer
        x = self.depthwise10(x)     # [ 6,  6, 128]
        x = self.pointwise10(x)     # [ 6,  6, 128]

        # 11th layer
        x = self.depthwise11(x)     # [ 6,  6, 128]
        x = self.pointwise11(x)     # [ 6,  6, 128]

        # 12th layer
        x = self.depthwise12(x)     # [ 6,  6, 128]
        x = self.pointwise12(x)     # [ 6,  6, 128]

        # 13th layer
        x = self.depthwise13(x)     # [ 3,  3, 128]
        x = self.pointwise13(x)     # [ 3,  3, 256]

        # 14th layer
        x = self.depthwise14(x)     # [ 3,  3, 256]
        x = self.pointwise14(x)     # [ 3,  3, 256]

        x = self.avgpool(x)         # [ 1,  1, 256]
        x = torch.flatten(x, 1)     # [256]
        x = self.out(x)             # [2]

        return x
