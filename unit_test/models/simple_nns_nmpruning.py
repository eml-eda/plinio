from torch import nn
import torch

class SimpleMlpNMPruning(nn.Module):
    def __init__(self, input_shape = (128,), num_classes = 3):
        super(SimpleMlpNMPruning, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        # Always compatible
        self.fc0 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, 16)
        # Not compatible with M > 16
        self.fc2 = nn.Linear(16, 12)
        # Not compatible with M >= 8
        self.fc3 = nn.Linear(12, 2)
        # Not compatible with M >= 4
        self.fc4 = nn.Linear(2, num_classes)
        self.foo = "non-nn.Module attribute"
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc0(x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        return x


class SimpleCnnNMPruning(nn.Module):
    def __init__(self, input_shape=(3, 40, 40), num_classes=3):
        super(SimpleCnnNMPruning, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        # This should always be excluded
        self.conv0 = nn.Conv2d(3, 32, (3, 3), padding='same', bias=False)
        self.bn0 = nn.BatchNorm2d(32, track_running_stats=True)

        # Depthwise should be always excluded
        self.conv1 = nn.Conv2d(32, 32, (3, 3), padding='same', groups=32, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=True)

        # Pointwise should be always excluded
        self.conv2 = nn.Conv2d(32, 4, (1, 1), padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(4, track_running_stats=True)

        # This is included only with M <= 4
        self.conv3 = nn.Conv2d(4, 8, (3, 3), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(8, track_running_stats=True)

        # This is included only with M <= 8
        self.conv4 = nn.Conv2d(8, 16, (3, 3), padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(16, track_running_stats=True)

        # This is always included
        self.conv5 = nn.Conv2d(16, 16, (5, 5), padding='same', bias=False)
        self.bn5 = nn.BatchNorm2d(16, track_running_stats=True)

        self.pool0 = nn.AdaptiveAvgPool2d(2)

        # This is always included
        self.fc = nn.Linear(16 * 2 * 2, num_classes)
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = torch.nn.functional.relu6(self.bn0(self.conv0(x)))
        x = torch.nn.functional.relu6(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu6(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu6(self.bn3(self.conv3(x)))
        x = torch.nn.functional.relu6(self.bn4(self.conv4(x)))
        x = torch.nn.functional.relu6(self.bn5(self.conv5(x)))
        x = self.pool0(x)
        x = x.flatten(1)
        res = self.fc(x)
        return res

