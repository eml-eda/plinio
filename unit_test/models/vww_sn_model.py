from typing import cast, Dict, Any, Optional
import torch
import torch.nn as nn
from flexnas.methods import SuperNetModule


def get_reference_model(model_name: str, model_config: Optional[Dict[str, Any]] = None
                        ) -> nn.Module:
    if model_name == 'mobilenet_sn':
        return MobileNetSN()
    else:
        raise ValueError(f"Unsupported model name {model_name}")


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


class MobileNetSN(torch.nn.Module):
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
        '''
        self.depthwise2 = ConvBlock(in_channels=8, out_channels=8,
                                    kernel_size=3, stride=1, padding=1, groups=8)
        self.pointwise2 = ConvBlock(in_channels=8, out_channels=16,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint2 = SuperNetModule([
            nn.Conv2d(8, 16, 3, padding='same'),
            nn.Conv2d(8, 16, 5, padding='same'),
            nn.Sequential(
                ConvBlock(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1,
                          groups=8),
                ConvBlock(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0)
            ),
            # nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint2[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint2[1]).weight)

        # 3d layer
        '''
        self.depthwise3 = ConvBlock(in_channels=16, out_channels=16,
                                    kernel_size=3, stride=2, padding=1, groups=16)
        self.pointwise3 = ConvBlock(in_channels=16, out_channels=32,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint3 = SuperNetModule([
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.Conv2d(16, 32, 5, padding=2, stride=2),
            nn.Sequential(
                ConvBlock(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1,
                          groups=16),
                ConvBlock(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0)
            ),
            # nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint3[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint3[1]).weight)

        # 4th layer
        '''
        self.depthwise4 = ConvBlock(in_channels=32, out_channels=32,
                                    kernel_size=3, stride=1, padding=1, groups=32)
        self.pointwise4 = ConvBlock(in_channels=32, out_channels=32,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint4 = SuperNetModule([
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.Conv2d(32, 32, 5, padding='same'),
            nn.Sequential(
                ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
                          groups=32),
                ConvBlock(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint4[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint4[1]).weight)

        # 5h layer
        '''
        self.depthwise5 = ConvBlock(in_channels=32, out_channels=32,
                                    kernel_size=3, stride=2, padding=1, groups=32)
        self.pointwise5 = ConvBlock(in_channels=32, out_channels=64,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint5 = SuperNetModule([
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.Conv2d(32, 64, 5, padding=2, stride=2),
            nn.Sequential(
                ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1,
                          groups=32),
                ConvBlock(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
            ),
            # nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint5[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint5[1]).weight)

        # 6th layer
        '''
        self.depthwise6 = ConvBlock(in_channels=64, out_channels=64,
                                    kernel_size=3, stride=1, padding=1, groups=64)
        self.pointwise6 = ConvBlock(in_channels=64, out_channels=64,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint6 = SuperNetModule([
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.Conv2d(64, 64, 5, padding='same'),
            nn.Sequential(
                ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
                          groups=64),
                ConvBlock(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint6[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint6[1]).weight)

        # 7th layer
        '''
        self.depthwise7 = ConvBlock(in_channels=64, out_channels=64,
                                    kernel_size=3, stride=2, padding=1, groups=64)
        self.pointwise7 = ConvBlock(in_channels=64, out_channels=128,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint7 = SuperNetModule([
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.Conv2d(64, 128, 5, padding=2, stride=2),
            nn.Sequential(
                ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,
                          groups=64),
                ConvBlock(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
            ),
            # nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint7[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint7[1]).weight)

        # 8th layer
        '''
        self.depthwise8 = ConvBlock(in_channels=128, out_channels=128,
                                    kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise8 = ConvBlock(in_channels=128, out_channels=128,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint8 = SuperNetModule([
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.Conv2d(128, 128, 5, padding='same'),
            nn.Sequential(
                ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                          groups=128),
                ConvBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint8[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint8[1]).weight)

        # 9th layer
        '''
        self.depthwise9 = ConvBlock(in_channels=128, out_channels=128,
                                    kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise9 = ConvBlock(in_channels=128, out_channels=128,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint9 = SuperNetModule([
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.Conv2d(128, 128, 5, padding='same'),
            nn.Sequential(
                ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                          groups=128),
                ConvBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint9[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint9[1]).weight)

        # 10th layer
        '''
        self.depthwise10 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise10 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint10 = SuperNetModule([
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.Conv2d(128, 128, 5, padding='same'),
            nn.Sequential(
                ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                          groups=128),
                ConvBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint10[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint10[1]).weight)

        # 11th layer
        '''
        self.depthwise11 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise11 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint11 = SuperNetModule([
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.Conv2d(128, 128, 5, padding='same'),
            nn.Sequential(
                ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                          groups=128),
                ConvBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint11[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint11[1]).weight)

        # 12th layer
        '''
        self.depthwise12 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise12 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint12 = SuperNetModule([
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.Conv2d(128, 128, 5, padding='same'),
            nn.Sequential(
                ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                          groups=128),
                ConvBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint12[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint12[1]).weight)

        # 13th layer
        '''
        self.depthwise13 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=3, stride=2, padding=1, groups=128)
        self.pointwise13 = ConvBlock(in_channels=128, out_channels=256,
                                     kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint13 = SuperNetModule([
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.Conv2d(128, 256, 5, padding=2, stride=2),
            nn.Sequential(
                ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1,
                          groups=128),
                ConvBlock(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)
            ),
            # nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint13[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint13[1]).weight)

        # 14th layer
        '''
        self.depthwise14 = ConvBlock(in_channels=256, out_channels=256,
                                     kernel_size=3, stride=1, padding=1, groups=256)
        self.pointwise14 = ConvBlock(in_channels=256, out_channels=256,
                                     kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint14 = SuperNetModule([
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.Conv2d(256, 256, 5, padding='same'),
            nn.Sequential(
                ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1,
                          groups=256),
                ConvBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ])
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint14[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, self.depthpoint14[1]).weight)

        self.avgpool = torch.nn.AvgPool2d(3)

        self.out = nn.Linear(256, 2)
        nn.init.kaiming_normal_(self.out.weight)

    def forward(self, input):

        # Input tensor shape        # [96, 96,  3]

        # 1st layer
        x = self.inputblock(input)  # [48, 48,  8]

        # 2nd layer
        # x = self.depthwise2(x)      # [48, 48,  8]
        # x = self.pointwise2(x)      # [48, 48, 16]
        x = self.depthpoint2(x)

        # 3rd layer
        # x = self.depthwise3(x)      # [24, 24, 16]
        # x = self.pointwise3(x)      # [24, 24, 32]
        x = self.depthpoint3(x)

        # 4th layer
        # x = self.depthwise4(x)      # [24, 24, 32]
        # x = self.pointwise4(x)      # [24, 24, 32]
        x = self.depthpoint4(x)

        # 5th layer
        # x = self.depthwise5(x)      # [12, 12, 32]
        # x = self.pointwise5(x)      # [12, 12, 64]
        x = self.depthpoint5(x)

        # 6th layer
        # x = self.depthwise6(x)      # [12, 12, 64]
        # x = self.pointwise6(x)      # [12, 12, 64]
        x = self.depthpoint6(x)

        # 7th layer
        # x = self.depthwise7(x)      # [ 6,  6, 64]
        # x = self.pointwise7(x)      # [ 6,  6, 128]
        x = self.depthpoint7(x)

        # 8th layer
        # x = self.depthwise8(x)      # [ 6,  6, 128]
        # x = self.pointwise8(x)      # [ 6,  6, 128]
        x = self.depthpoint8(x)

        # 9th layer
        # x = self.depthwise9(x)      # [ 6,  6, 128]
        # x = self.pointwise9(x)      # [ 6,  6, 128]
        x = self.depthpoint9(x)

        # 10th layer
        # x = self.depthwise10(x)     # [ 6,  6, 128]
        # x = self.pointwise10(x)     # [ 6,  6, 128]
        x = self.depthpoint10(x)

        # 11th layer
        # x = self.depthwise11(x)     # [ 6,  6, 128]
        # x = self.pointwise11(x)     # [ 6,  6, 128]
        x = self.depthpoint11(x)

        # 12th layer
        # x = self.depthwise12(x)     # [ 6,  6, 128]
        # x = self.pointwise12(x)     # [ 6,  6, 128]
        x = self.depthpoint12(x)

        # 13th layer
        # x = self.depthwise13(x)     # [ 3,  3, 128]
        # x = self.pointwise13(x)     # [ 3,  3, 256]
        x = self.depthpoint13(x)

        # 14th layer
        # x = self.depthwise14(x)     # [ 3,  3, 256]
        # x = self.pointwise14(x)     # [ 3,  3, 256]
        x = self.depthpoint14(x)

        x = self.avgpool(x)         # [ 1,  1, 256]
        x = torch.squeeze(x)        # [256]
        x = self.out(x)             # [2]

        return x
