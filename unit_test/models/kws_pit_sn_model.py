
import torch
import torch.nn as nn
from flexnas.methods import PITSuperNetModule
from flexnas.methods.pit import PITConv2d
from flexnas.methods.pit.pit_features_masker import PITFeaturesMasker


class DSCnnPITSN(torch.nn.Module):
    def __init__(self):
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
        '''
        self.depthwise1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint1 = PITSuperNetModule([
            nn.Sequential(
                # PITConv2d(nn.Conv2d(64, 64, 3, padding='same'),
                          # 64, 64, out_features_masker=PITFeaturesMasker(64)),
                nn.Conv2d(64, 64, 3, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                # PITConv2d(nn.Conv2d(64, 64, 5, padding='same'),
                          # 64, 64, out_features_masker=PITFeaturesMasker(64)),
                nn.Conv2d(64, 64, 5, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Identity()
        ])
        # self.bn11 = nn.BatchNorm2d(64, momentum=0.99)
        # self.relu11 = nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn12 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu12 = nn.ReLU()

        # Second layer of separable depthwise conv2d
        '''
        self.depthwise2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint2 = PITSuperNetModule([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 5, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Identity()
        ])
        # self.bn21 = nn.BatchNorm2d(64, momentum=0.99)
        # self.relu21 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn22 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu22 = nn.ReLU()

        # Third layer of separable depthwise conv2d
        '''
        self.depthwise3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint3 = PITSuperNetModule([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 5, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Identity()
        ])
        # self.bn31 = nn.BatchNorm2d(64, momentum=0.99)
        # self.relu31 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn32 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu32 = nn.ReLU()

        # Fourth layer of separable depthwise conv2d
        '''
        self.depthwise4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint4 = PITSuperNetModule([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 5, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Identity()
        ])
        # self.bn41 = nn.BatchNorm2d(64, momentum=0.99)
        # self.relu41 = nn.ReLU()

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
        # x = self.depthwise1(x)
        # x = self.pointwise1(x)
        x = self.depthpoint1(x)
        # x = self.relu11(self.bn11(x))
        x = self.conv1(x)
        x = self.relu12(self.bn12(x))

        # Second layer of separable depthwise conv2d
        # x = self.depthwise2(x)
        # x = self.pointwise2(x)
        x = self.depthpoint2(x)
        # x = self.relu21(self.bn21(x))
        x = self.conv2(x)
        x = self.relu22(self.bn22(x))

        # Third layer of separable depthwise conv2d
        # x = self.depthwise3(x)
        # x = self.pointwise3(x)
        x = self.depthpoint3(x)
        # x = self.relu31(self.bn31(x))
        x = self.conv3(x)
        x = self.relu32(self.bn32(x))

        # Fourth layer of separable depthwise conv2d
        # x = self.depthwise4(x)
        # x = self.pointwise4(x)
        x = self.depthpoint4(x)
        # x = self.relu41(self.bn41(x))
        x = self.conv4(x)
        x = self.relu42(self.bn42(x))

        x = self.dropout2(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.out(x)

        return x
