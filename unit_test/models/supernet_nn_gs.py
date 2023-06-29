from torch import nn
from plinio.methods.supernet import SuperNetModule


class GumbelSNModule(nn.Module):
    """Defines a simple sequential DNN with Gumbel sampling, used within unit tests"""
    def __init__(self, input_shape=(32, 64, 64), hard_softmax: bool = False):
        super(GumbelSNModule, self).__init__()
        self.input_shape = input_shape

        self.conv1 = SuperNetModule([
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding='same'),
                nn.Conv2d(32, 32, 3, padding='same'),
            ),
            nn.Conv2d(32, 32, 5, padding='same'),
            nn.Identity()
        ], gumbel_softmax=True, hard_softmax=hard_softmax)

    def forward(self, x):
        res = self.conv1(x)
        return res
