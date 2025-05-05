import torch
import torch.nn as nn
import torch.nn.functional as F


class ToySequentialConv1d(nn.Module):
    def __init__(self):
        super(ToySequentialConv1d, self).__init__()
        self.input_shape = (3, 12)
        self.conv0 = nn.Conv1d(3, 10, (3,), padding="same")
        self.conv1 = nn.Conv1d(10, 20, (9,), padding="same")

    def forward(self, x):
        return self.conv1(self.conv0(x))


class ToySequentialFullyConv2d(nn.Module):
    def __init__(self):
        super(ToySequentialFullyConv2d, self).__init__()
        self.input_shape = (3, 12, 12)
        self.conv0 = nn.Conv2d(3, 10, (3, 3), padding=(2, 2))
        self.bn0 = nn.BatchNorm2d(10)
        self.conv1 = nn.Conv2d(10, 2, (14, 14))

    def forward(self, x):
        return self.conv1(F.relu(self.bn0(self.conv0(x))))


class ToySequentialFullyConv2dDil(nn.Module):
    def __init__(self):
        super(ToySequentialFullyConv2dDil, self).__init__()
        self.input_shape = (3, 12, 1)
        self.conv0 = nn.Conv2d(3, 10, (3, 1), dilation=(2, 1), padding=(2, 0))
        self.bn0 = nn.BatchNorm2d(10)
        self.conv1 = nn.Conv2d(10, 2, (12, 1))

    def forward(self, x):
        return self.conv1(F.relu(self.bn0(self.conv0(x))))


class ToySequentialConv2d(nn.Module):
    def __init__(self):
        super(ToySequentialConv2d, self).__init__()
        self.input_shape = (3, 12, 12)
        self.conv = nn.Conv2d(3, 10, (3, 3), padding=(2, 2))
        self.bn = nn.BatchNorm2d(10)
        self.lin = nn.Linear(1960, 2)

    def forward(self, x):
        return self.lin(torch.flatten(F.relu(self.bn(self.conv(x))), 1))


class ToySequentialConv2d_v2(nn.Module):
    def __init__(self):
        super(ToySequentialConv2d_v2, self).__init__()
        self.input_shape = (3, 12, 12)
        self.conv = nn.Conv2d(3, 10, (3, 3), padding=(2, 2))
        # self.conv = nn.Conv2d(3, 10, (3, 3), padding='valid')
        self.bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 10, (3, 3), padding=(1, 1))
        # self.conv2 = nn.Conv2d(10, 10, (3, 3), padding='valid')
        self.bn2 = nn.BatchNorm2d(10)
        self.lin = nn.Linear(1960, 2)
        # self.lin = nn.Linear(640, 2)

    def forward(self, x):
        # return self.lin(torch.flatten(
        #     F.relu(self.bn(self.conv(x))), 1))
        return self.lin(
            torch.flatten(
                F.relu(self.bn2(self.conv2(F.relu(self.bn(self.conv(x)))))), 1
            )
        )


class ToySequentialSeparated(nn.Module):
    def __init__(self):
        super(ToySequentialSeparated, self).__init__()
        self.input_shape = (3, 10)
        self.conv0 = nn.Conv1d(3, 10, (3,), padding="same")
        self.bn0 = nn.BatchNorm1d(10, track_running_stats=True)
        self.pool0 = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(10, 20, (9,), padding="same")

    def forward(self, x):
        return self.conv1(F.relu6(self.pool0(self.bn0(self.conv0(x)))))


class ToyAdd(nn.Module):
    def __init__(self):
        super(ToyAdd, self).__init__()
        self.input_shape = (3, 15)
        self.conv0 = nn.Conv1d(3, 10, (3,), padding="same")
        self.conv1 = nn.Conv1d(3, 10, (3,), padding="same")
        self.conv2 = nn.Conv1d(10, 20, (5,), padding="same")
        self.reluadd = nn.ReLU()
        self.pool = nn.MaxPool1d((2,))
        self.fc = nn.Linear(20 * 7, 2)

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        x = self.reluadd(a + b)
        y = self.conv2(self.pool(x))
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return y


class ToyAdd_2D(nn.Module):
    def __init__(self):
        super(ToyAdd_2D, self).__init__()
        self.input_shape = (3, 15, 15)
        self.conv0 = nn.Conv2d(3, 10, (3, 3), padding="same")
        self.conv1 = nn.Conv2d(3, 10, (3, 3), padding="same")
        self.conv2 = nn.Conv2d(10, 20, (5, 5), padding="same")
        self.reluadd = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2))
        self.fc = nn.Linear(20 * 7 * 7, 2)

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        x = self.reluadd(a + b)
        y = self.conv2(self.pool(x))
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return y


class ToyTimeCat(nn.Module):
    def __init__(self):
        super(ToyTimeCat, self).__init__()
        self.input_shape = (3, 20)
        self.conv0 = nn.Conv1d(3, 10, (3,), padding="same")
        self.conv1 = nn.Conv1d(3, 10, (3,), padding="same")
        self.conv2 = nn.Conv1d(10, 32, (3,), padding="same")

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        x = torch.cat((a, b), dim=2)
        y = self.conv2(x)
        return y


class ToyChannelsCat(nn.Module):
    def __init__(self):
        super(ToyChannelsCat, self).__init__()
        self.input_shape = (3, 24)  # fully-convolutional
        self.conv0 = nn.Conv1d(3, 10, (3,), padding="same")
        self.conv1 = nn.Conv1d(3, 15, (3,), padding="same")
        self.conv2 = nn.Conv1d(25, 32, (3,), padding="same")

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        x = torch.cat((a, b), dim=1)
        y = self.conv2(x)
        return y


class ToyFlatten(nn.Module):
    def __init__(self):
        super(ToyFlatten, self).__init__()
        self.input_shape = (3, 8)
        self.conv0 = nn.Conv1d(3, 4, (3,), padding="same")
        self.conv1 = nn.Conv1d(4, 2, (3,), padding="same")
        self.conv2 = nn.Conv1d(2, 1, (2,), padding="same")
        self.pool0 = nn.AvgPool1d(2)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.pool0(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class ToyMultiPath1(nn.Module):
    def __init__(self):
        super(ToyMultiPath1, self).__init__()
        self.input_shape = (3, 10)
        self.conv0 = nn.Conv1d(3, 32, (3,), padding="same")
        self.conv1 = nn.Conv1d(3, 32, (3,), padding="same")
        self.conv2 = nn.Conv1d(3, 64, (3,), padding="same")
        self.conv3 = nn.Conv1d(3, 50, (3,), padding="same")
        self.conv4 = nn.Conv1d(50, 64, (3,), padding="same")
        self.conv5 = nn.Conv1d(64, 64, (3,), padding="same")
        self.fc = nn.Linear(640, 2)
        self.reluadd = nn.ReLU()

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        z = torch.cat((a, b), dim=1)
        z = self.conv5(z)
        y = torch.tanh(self.conv2(x))
        w = torch.sigmoid(self.conv3(x))
        w = self.conv4(w)
        x = self.reluadd(z + y + w)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class ToyMultiPath1_2D(nn.Module):
    def __init__(self):
        super(ToyMultiPath1_2D, self).__init__()
        self.input_shape = (3, 10, 10)
        self.conv0 = nn.Conv2d(3, 32, (3, 3), padding="same")
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding="same")
        self.conv2 = nn.Conv2d(3, 64, (3, 3), padding="same")
        self.conv3 = nn.Conv2d(3, 50, (3, 3), padding="same")
        self.conv4 = nn.Conv2d(50, 64, (3, 3), padding="same")
        # self.conv5 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.conv5 = nn.Conv2d(32, 64, (3, 3), padding="same")
        self.fc = nn.Linear(6400, 2)
        self.reluadd = nn.ReLU()

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        # z = torch.cat((a, b), dim=1)
        z = a + b
        z = self.conv5(z)
        y = self.conv2(x)
        w = self.conv3(x)
        w = self.conv4(w)
        x = self.reluadd(z + y + w)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class ToyMultiPath2(nn.Module):
    def __init__(self):
        super(ToyMultiPath2, self).__init__()
        self.input_shape = (3, 60)
        self.conv0 = nn.Conv1d(3, 40, (3,), padding="same")
        self.conv1 = nn.Conv1d(3, 40, (3,), padding="same")
        self.bn0 = nn.BatchNorm1d(40, track_running_stats=True)
        self.pool0 = nn.AvgPool1d(2)
        self.conv2 = nn.Conv1d(3, 20, (3,), padding="same")
        self.conv3 = nn.Conv1d(3, 20, (3,), padding="same")
        self.bn1 = nn.BatchNorm1d(20, track_running_stats=True)
        self.pool1 = nn.AvgPool1d(2)
        self.relu = nn.ReLU()
        self.reluadd = nn.ReLU()
        self.conv4 = nn.Conv1d(40, 15, (3,), padding="same")
        self.fc = nn.Linear(450, 10)

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        z = torch.add(a, b)
        z = self.pool0(self.bn0(z))
        y = self.pool1(self.bn1(self.conv2(x)))
        w = self.pool1(self.bn1(self.conv3(x)))
        y = torch.cat((y, w), dim=1)
        y = self.relu(y)
        x = self.reluadd(y + z)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ToyMultiPath2_2D(nn.Module):
    def __init__(self):
        super(ToyMultiPath2_2D, self).__init__()
        self.input_shape = (3, 10, 10)
        self.conv0 = nn.Conv2d(3, 40, (3, 3), padding="same")
        self.conv1 = nn.Conv2d(3, 40, (3, 3), padding="same")
        self.bn0 = nn.BatchNorm2d(40, track_running_stats=True)
        self.pool0 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(3, 20, (3, 3), padding="same")
        self.conv3 = nn.Conv2d(3, 20, (3, 3), padding="same")
        self.bn1 = nn.BatchNorm2d(20, track_running_stats=True)
        self.pool1 = nn.AvgPool2d(2)
        self.relu = nn.ReLU()
        self.reluadd = nn.ReLU()
        self.conv4 = nn.Conv2d(40, 15, (3, 3), padding="same")
        self.fc = nn.Linear(375, 10)

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        z = torch.add(a, b)
        z = self.pool0(self.bn0(z))
        y = self.pool1(self.bn1(self.conv2(x)))
        w = self.pool1(self.bn1(self.conv3(x)))
        y = torch.cat((y, w), dim=1)
        y = self.relu(y)
        x = self.reluadd(y + z)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ToyRegression(nn.Module):
    def __init__(self):
        super(ToyRegression, self).__init__()
        self.input_shape = (1, 10)
        self.conv0 = nn.Conv1d(1, 10, (3,), padding="same")
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ToyRegression_2D(nn.Module):
    def __init__(self):
        super(ToyRegression_2D, self).__init__()
        self.input_shape = (1, 5, 5)
        self.conv0 = nn.Conv2d(1, 10, (3, 3), padding="same")
        self.fc = nn.Linear(250, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ToyInputConnectedDW(nn.Module):
    def __init__(self):
        super(ToyInputConnectedDW, self).__init__()
        self.input_shape = (3, 28, 28)
        self.dw_conv = nn.Conv2d(3, 3, (3, 3), groups=3, padding="same")
        self.dw_bn = nn.BatchNorm2d(3)
        self.dw_relu = nn.ReLU()
        self.pw_conv = nn.Conv2d(3, 16, (1, 1), padding="same")
        self.pw_bn = nn.BatchNorm2d(16)
        self.pw_relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(28)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.dw_relu(self.dw_bn(self.dw_conv(x)))
        x = self.pw_relu(self.pw_bn(self.pw_conv(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ToyBatchNorm(nn.Module):
    def __init__(self):
        super(ToyBatchNorm, self).__init__()
        self.input_shape = (3, 28, 28)
        self.dw_conv = nn.Conv2d(3, 3, (3, 3), groups=3, padding="same")
        self.dw_bn = nn.BatchNorm2d(3)
        self.dw_relu = nn.ReLU()
        self.pw_conv = nn.Conv2d(3, 16, (1, 1), padding="same")
        self.pw_bn = nn.BatchNorm2d(16)
        self.pw_relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(28)
        self.fc1 = nn.Linear(16, 16)
        self.fc1_bn = nn.BatchNorm1d(16)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.dw_relu(self.dw_bn(self.dw_conv(x)))
        x = self.pw_relu(self.pw_bn(self.pw_conv(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1_relu(self.fc1_bn(self.fc1(x)))
        return self.fc2(x)


class ToyIllegalBN(nn.Module):
    def __init__(self):
        super(ToyIllegalBN, self).__init__()
        self.input_shape = (3, 28, 28)
        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding="same")
        self.conv2 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv output used elsewhere
        tmp = self.conv1(x)
        x1 = self.relu(self.bn(tmp))
        x2 = self.conv2(tmp)
        return x1 + x2


class ToyResNet(nn.Module):
    def __init__(self):
        super(ToyResNet, self).__init__()
        self.input_shape = (16, 2, 2)
        self.conv0 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn0 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn2 = nn.BatchNorm2d(16)
        self.classifier = nn.Linear(16 * 2 * 2, 10)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        x3 = F.relu(x + x2)
        x3 = torch.flatten(x3, 1)
        return self.classifier(x3)


class ToyResNet_inp_conn(nn.Module):
    def __init__(self):
        super(ToyResNet_inp_conn, self).__init__()
        self.input_shape = (16, 2, 2)
        self.conv1 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn2 = nn.BatchNorm2d(16)
        self.classifier = nn.Linear(16 * 2 * 2, 10)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        x3 = F.relu(x + x2)
        x3 = torch.flatten(x3, 1)
        return self.classifier(x3)


class ToyResNet_inp_conn_add_out(nn.Module):
    def __init__(self):
        super(ToyResNet_inp_conn_add_out, self).__init__()
        self.input_shape = (16, 2, 2)
        self.conv1 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn2 = nn.BatchNorm2d(16)
        self.classifier = nn.Linear(16 * 2 * 2, 10)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        x3 = x + x2
        return x3


class ToyResNet_1D(nn.Module):
    def __init__(self):
        super(ToyResNet_1D, self).__init__()
        self.input_shape = (16, 2)
        self.conv1 = nn.Conv1d(16, 16, 3, padding="same")
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 16, 3, padding="same")
        self.bn2 = nn.BatchNorm1d(16)
        self.classifier = nn.Linear(16 * 2, 10)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        x3 = F.relu(x + x2)
        x3 = torch.flatten(x3, 1)
        return self.classifier(x3)

class ToyGroupedConv_1D(nn.Module):
    def __init__(self, groups=2):
        super(ToyGroupedConv_1D, self).__init__()
        self.input_shape = (16, 2)
        #2 parallel conv replace a conv(groups=2)
        self.conv0 = nn.Conv1d(16, 16, 5, padding="same", groups=1)
        self.conv1 = nn.Conv1d(8, 4, 3, padding="same", groups=1)
        self.conv2 = nn.Conv1d(8, 4, 3, padding="same", groups=1)
        self.conv3 = nn.Conv1d(8, 16, 3, padding="same", groups=1)

    def forward(self, x):
        x0  = F.relu(self.conv0(x))
        x1 = x0[:,0:8]
        x2 = x0[:,8:16]
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(self.conv2(x2))
        x3 = torch.cat((x1, x2), dim=1)
        x4 = self.conv3(x3)
        return x4

class ToyIndexingConv_1D(nn.Module):
    def __init__(self, groups=2, slicing_mode='slice'):
        super(ToyIndexingConv_1D, self).__init__()
        self.input_shape = (16, 2)
        self.slicing_mode = slicing_mode
        self.tensor_idx1 = torch.tensor([0,2])
        self.tensor_idx2 = torch.tensor([1,] + list(range(3,16)))
        #2 parallel conv replace a conv(groups=2)
        self.conv0 = nn.Conv1d(16, 16, 5, padding="same", groups=1)
        self.conv1 = nn.Conv1d(2, 4, 3, padding="same", groups=1)
        self.conv2 = nn.Conv1d(14, 4, 3, padding="same", groups=1)
        self.conv3 = nn.Conv1d(8, 16, 3, padding="same", groups=1)

    def forward(self, x):
        x0  = F.relu(self.conv0(x))
        if self.slicing_mode == 'slice':
            x1 = x0[:,0:2]
            x2 = x0[:,2:16]
        elif self.slicing_mode == 'list':
            x1 = x0[:,[0,2],:]
            x2 = x0[:,[1,] + list(range(3,16)),:]
        elif self.slicing_mode == 'bool':
            mask1=[True] * 2 + [False] * 14
            mask2=[False] * 2 + [True] * 14
            x1 = x0[:,mask1,:]
            x2 = x0[:,mask2,:]
        elif self.slicing_mode == 'int':
            x10 = x0[:,[0],:]
            x11 = x0[:,[-15],:]
            x1 = torch.cat((x10,x11), dim=1)
            x2 = x0[:,list(range(2,16)),:]
        elif self.slicing_mode == 'tensor':
            x1 = x0[:,torch.tensor([0,2]),:] #self.tensor_idx1
            x2 = x0[:,torch.tensor([1,] + list(range(3,16))),:]
        else:
            raise ValueError("Invalid slicing mode")
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(self.conv2(x2))
        x3 = torch.cat((x1, x2), dim=1)
        x4 = self.conv3(x3)
        return x4

class ToyMultiGroupConv_1D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ToyMultiGroupConv_1D, self).__init__()
        #2 parallel conv replace a conv(groups=2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = (self.in_channels, 2)
        self.groups=groups

        self.conv0 = nn.Conv1d(self.in_channels, self.in_channels, 5, padding="same", groups=1)
        self.groupconvs = nn.Sequential()
        for _ in range(self.groups):
            self.groupconvs.append(nn.Conv1d(self.in_channels//groups, self.out_channels//groups, 3, padding="same", groups=1))
        self.conv_final = nn.Conv1d(self.out_channels, self.in_channels, 3, padding="same", groups=1)

    def forward(self, x):
        x0  = F.relu(self.conv0(x))
        x2=[]
        for i in range(self.groups):
            x1 = x0[:,i*self.in_channels//self.groups:(i+1)*self.in_channels//self.groups,:]
            x2.append(self.groupconvs[i](x1))
        x3 = torch.cat(x2, dim=1)
        x4 = self.conv_final(x3)
        x4 = F.relu(x4)
        return x4

class ToyGroupedConv_2D(nn.Module):
    def __init__(self, groups=2):
        super(ToyGroupedConv_2D, self).__init__()
        self.input_shape = (16, 8, 8)
        #2 parallel conv replace a conv(groups=2)
        self.conv0 = nn.Conv2d(16, 16, 5, padding="same", groups=1)
        self.conv1 = nn.Conv2d(8, 4, 3, padding="same", groups=1)
        self.conv2 = nn.Conv2d(8, 4, 3, padding="same", groups=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding="same", groups=1)

    def forward(self, x):
        x0  = F.relu(self.conv0(x))
        x1 = x0[:,0:8]
        x2 = x0[:,8:16]
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(self.conv2(x2))
        x3 = torch.cat((x1, x2), dim=1)
        x4 = self.conv3(x3)
        return x4