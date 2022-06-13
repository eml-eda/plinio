import torch.nn as nn
import torch.nn.functional as F
from flexnas.methods.pit.pit_conv1d import PITConv1d
from flexnas.methods.pit.pit_channel_masker import PITChannelMasker
from flexnas.methods.pit.pit_timestep_masker import PITTimestepMasker
from flexnas.methods.pit.pit_dilation_masker import PITDilationMasker


class SimplePitNN(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(1, 3, 40), num_classes=3):
        super(SimplePitNN, self).__init__()
        self.input_shape = input_shape
        self.pitconv0 = PITConv1d(nn.Conv1d(3, 32, (3,), padding='same'),
                                  32,
                                  PITChannelMasker(32),
                                  PITTimestepMasker(3),
                                  PITDilationMasker(1))
        # track_running_stats=False is important for testability, otherwise the DNAS Net and normal
        # net won't produce the exact same outputs due to BatchNorm differences
        self.bn0 = nn.BatchNorm1d(32, track_running_stats=False)
        self.pool0 = nn.AvgPool1d(2)
        self.conv1 = nn.Conv1d(32, 57, (5,), padding='same')
        self.bn1 = nn.BatchNorm1d(57, track_running_stats=False)
        self.pool1 = nn.AvgPool1d(2)
        self.dpout = nn.Dropout(0.5)
        self.fc = nn.Linear(57 * (input_shape[-1] // 2 // 2), num_classes)
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = F.relu6(self.pool0(self.bn0(self.pitconv0(x))))
        x = F.relu6(self.pool1(self.bn1(self.conv1(x))))
        x = self.dpout(x.flatten(1))
        res = self.fc(x)
        return res