
from torch import nn
from flexnas.methods.supernet.supernet_module import SuperNetModule


class SingleModuleNet1(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(32, 4)):
        super(SingleModuleNet1, self).__init__()
        self.input_shape = input_shape

        self.conv1 = SuperNetModule([
            nn.Conv1d(32, 57, (3,), padding='same'),  # Cin!=Cout
            nn.Conv1d(32, 57, (5,), padding='same'),  # Cin!=Cout
            nn.Conv1d(32, 57, (7,), padding='same')  # Cin!=Cout
        ])

    def forward(self, x):
        res = self.conv1(x)
        return res


class SingleModuleNet2(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(32, 4)):
        super(SingleModuleNet2, self).__init__()
        self.input_shape = input_shape

        self.mix = SuperNetModule([
            nn.Identity(),
            nn.Conv1d(32, 32, 3, padding='same')
        ])

    def forward(self, x):
        res = self.mix(x)
        return res


class MultipleModuleNet1(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(32, 4)):
        super(MultipleModuleNet1, self).__init__()
        self.input_shape = input_shape

        self.mix = SuperNetModule([
            nn.Identity(),
            nn.Conv1d(32, 32, 3, padding='same')
        ])

        self.conv1 = SuperNetModule([
            nn.Conv1d(32, 57, (3,), padding='same'),  # Cin!=Cout
            nn.Conv1d(32, 57, (5,), padding='same'),  # Cin!=Cout
            nn.Conv1d(32, 57, (7,), padding='same')  # Cin!=Cout
        ])

    def forward(self, x):
        res = self.mix(x)
        res = self.conv1(res)
        return res


class StandardSNModule(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(32, 64, 64)):
        super(StandardSNModule, self).__init__()
        self.input_shape = input_shape

        self.conv1 = SuperNetModule([
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.Conv2d(32, 32, 5, padding='same'),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, groups=32, padding='same'),
                nn.Conv2d(32, 32, 1, padding='same')
            ),
            nn.Identity()
        ])

    def forward(self, x):
        res = self.conv1(x)
        return res
