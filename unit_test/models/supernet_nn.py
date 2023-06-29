from torch import nn
from plinio.methods.supernet import SuperNetModule


class StandardSNModule(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(32, 64, 64)):
        super(StandardSNModule, self).__init__()
        self.input_shape = input_shape

        self.conv1 = SuperNetModule([
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


class CustomSeq(nn.Module):
    def __init__(self):
        super(CustomSeq, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 7, padding='same')
        self.conv2 = nn.Conv2d(32, 32, 7, padding='same')

    def forward(self, x):
        res = self.conv2(nn.functional.relu(self.conv1(x)))
        return res


class CustomSeqL2(nn.Module):
    def __init__(self):
        super(CustomSeqL2, self).__init__()
        self.c1 = CustomSeq()
        self.c2 = CustomSeq()

    def forward(self, x):
        res = self.c1(self.c2(x))
        return res


class StandardSNModuleV2(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(32, 64, 64)):
        super(StandardSNModuleV2, self).__init__()
        self.input_shape = input_shape

        self.conv1 = SuperNetModule([
            nn.Conv2d(32, 32, 3, padding='same'),
            CustomSeqL2(),
            nn.Conv2d(32, 32, 5, padding='same'),
            nn.Identity()
        ])

    def forward(self, x):
        res = self.conv1(x)
        return res
