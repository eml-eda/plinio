from torch import nn
from plinio.methods.pit_supernet import PITSuperNetModule


class StandardPITSNModule(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(32, 64, 64)):
        super(StandardPITSNModule, self).__init__()
        self.input_shape = input_shape

        self.conv1 = PITSuperNetModule([
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding='same'),
                nn.Conv2d(32, 32, 3, padding='same'),
            ),
            nn.Conv2d(32, 32, 5, padding='same'),
            nn.Identity()
        ])

    def forward(self, x):
        res = self.conv1(x)
        return res
