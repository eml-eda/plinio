from torch import nn
from flexnas.methods import PITSuperNetModule
from flexnas.methods.pit import PITConv2d
from flexnas.methods.pit.pit_features_masker import PITFeaturesMasker


class StandardPITSNModule(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(32, 64, 64)):
        super(StandardPITSNModule, self).__init__()
        self.input_shape = input_shape

        self.conv1 = PITSuperNetModule([
            PITConv2d(nn.Conv2d(32, 32, 3, padding='same'),
                      32, 32, out_features_masker=PITFeaturesMasker(32)),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding='same'),
                PITConv2d(nn.Conv2d(32, 32, 3, padding='same'),
                          32, 32, out_features_masker=PITFeaturesMasker(32)),
            ),
            nn.Conv2d(32, 32, 5, padding='same'),
            nn.Identity()
        ])

    def forward(self, x):
        res = self.conv1(x)
        return res
