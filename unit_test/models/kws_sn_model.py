from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from flexnas.methods import SuperNetModule


def get_reference_model(model_name: str, model_config: Optional[Dict[str, Any]] = None
                        ) -> nn.Module:
    if model_name == 'tc_resnet_14':
        if model_config is None:
            model_config = {
                "input_channels": 10,
                "output_size": 12,
                "num_channels": [49, 36, 36, 48, 48, 72, 72],
                "kernel_size": 9,
                "dropout": 0.5,
                "grad_clip": -1,
                "use_bias": True,
                "use_dilation": True,
                "avg_pool": True,
            }
        return TCResNet14(model_config)
    elif model_name == 'ds_cnn_sn':
        return DSCnnSN()
    else:
        raise ValueError(f"Unsupported model name {model_name}")


class TCResNet14(nn.Module):
    """
    TCResNet14 architecture used for unit tests.
    Variable number of repeated instances of TemporalBlock1 and TemporalBlock2.

    Testing on this network verifies that our conversion methods can handle residual connections
    """

    def __init__(self, config):
        super(TCResNet14, self).__init__()
        self.config = config

        self.pad0 = nn.ConstantPad1d(
            padding=(3 - 1, 0),
            value=0
        )
        self.conv0 = nn.Conv1d(
            in_channels=self.config['input_channels'],
            out_channels=self.config['num_channels'][0],
            kernel_size=(3,),
            bias=self.config['use_bias']
        )

        self.tcn = TempNet(
            num_inputs=self.config['num_channels'][0],
            num_channels=self.config['num_channels'][1:],
            kernel_size=self.config['kernel_size'],
            use_bias=self.config['use_bias'],
            use_dilation=self.config['use_dilation'],
        )

        if self.config['avg_pool']:
            self.avgpool = nn.AvgPool1d(
                kernel_size=2
            )
            self.ft_in = 3
        else:
            self.ft_in = 6

        self.dpout = nn.Dropout(
            p=self.config['dropout']
        )

        self.out = nn.Linear(
            in_features=self.config['num_channels'][-1] * self.ft_in,
            out_features=self.config['output_size'],
            bias=self.config['use_bias'],
        )

    def forward(self, x):
        x = self.pad0(x)
        x = self.conv0(x)
        y1 = self.tcn(x)
        if self.config['avg_pool']:
            y1 = self.avgpool(y1)
        y1_dpout = self.dpout(y1)
        out = y1_dpout
        out = out.flatten(1)
        out = self.out(out)
        out = F.log_softmax(out, dim=1)
        return out


class TempNet(nn.Module):
    """
    Temporal Convolutional Net composed of a number of an alternate number of TempBlock1 and
    TempBlock2 defined with a specific parameter.
    """

    def __init__(self, num_inputs, num_channels, kernel_size, use_bias=True, use_dilation=True):
        super(TempNet, self).__init__()
        layers = list()
        num_levels = len(num_channels)
        original_rf = [(kernel_size - 1) * (2 ** i) + 1 for i in range(num_levels)]
        for i in range(num_levels):
            dilation_size = list()
            k = list()
            if use_dilation:
                dilation_size.append(2 ** i)
                dilation_size.append(2 ** i)
                k.append(ceil(original_rf[i] / dilation_size[0]))
                k.append(ceil(original_rf[i] / dilation_size[1]))
            else:
                dilation_size.append(1)
                dilation_size.append(1)
                k.append(original_rf[i])
                k.append(original_rf[i])

            in_channels = [
                num_inputs, num_channels[0]] if i == 0 else [num_channels[i - 1], num_channels[i]]
            out_channels = [num_channels[i], num_channels[i]]

            if (i % 2) != 0:
                layers += [TempBlock1(
                    ch_in=in_channels,
                    ch_out=out_channels,
                    k_size=k,
                    dil=dilation_size,
                    use_bias=use_bias,
                )]
            else:
                layers += [TempBlock2(
                    ch_in=in_channels,
                    ch_out=out_channels,
                    k_size=k,
                    dil=dilation_size,
                    use_bias=use_bias,
                )]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TempBlock1(nn.Module):
    """
    Temporal Block composed of two temporal convolutional block.
    The temporal convolutional block is composed of :
    - Padding layer
    - Conv1d layer
    - BatchNorm layer
    - ReLU layer
    - Dropout layer (TODO: removed, re-insert)

    A residual connection between the input and the output of the block is present

    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param dil: Amount of dilation
    :param use_bias: Use bias if True
    """

    def __init__(self, ch_in, ch_out, k_size, dil, use_bias=True):
        super(TempBlock1, self).__init__()

        self.k0 = k_size[0]
        self.k1 = k_size[1]

        self.dil0 = dil[0]
        self.dil1 = dil[1]

        self.ch_in0 = ch_in[0]
        self.ch_in1 = ch_in[1]

        self.ch_out0 = ch_out[0]
        self.ch_out1 = ch_out[1]

        self.use_bias = use_bias

        self.pad0 = nn.ConstantPad1d(
            padding=((self.k0 - 1) * dil[0], 0),
            value=0
        )
        self.tcn0 = nn.Conv1d(
            in_channels=self.ch_in0,
            out_channels=self.ch_out0,
            kernel_size=self.k0,
            dilation=self.dil0,
            bias=self.use_bias
        )
        self.batchnorm0 = nn.BatchNorm1d(
            num_features=self.ch_out0
        )
        self.relu0 = nn.ReLU()

        self.pad1 = nn.ConstantPad1d(
            padding=((self.k1 - 1) * dil[1], 0),
            value=0
        )
        self.tcn1 = nn.Conv1d(
            in_channels=self.ch_in1,
            out_channels=self.ch_out1,
            kernel_size=self.k1,
            dilation=self.dil1,
            bias=self.use_bias
        )
        self.downsample = nn.Conv1d(
            in_channels=self.ch_in0,
            out_channels=self.ch_out1,
            kernel_size=(1,),
            bias=self.use_bias
        )
        self.downsamplerelu = nn.ReLU()
        self.downsamplebn = nn.BatchNorm1d(
            num_features=self.ch_out1
        )
        self.batchnorm1 = nn.BatchNorm1d(
            num_features=self.ch_out1
        )

        self.reluadd = nn.ReLU()

    def forward(self, x):
        x1 = self.relu0(self.batchnorm0(self.tcn0(self.pad0(x))))
        x2 = self.batchnorm1(self.tcn1(self.pad1(x1)))
        res = self.downsamplerelu(self.downsamplebn(self.downsample(x)))
        return self.reluadd(x2 + res)

    def init_weights(self):
        self.tcn0.weight.data.normal_(0, 0.01)
        self.tcn1.weight.data.normal_(0, 0.01)
        self.downsample.weight.data.normal_(0, 0.01)


class TempBlock2(nn.Module):
    """
    Temporal Block composed of two temporal convolutional block.
    The temporal convolutional block is composed of :
    - Padding layer
    - Conv1d layer
    - BatchNorm layer
    - ReLU layer
    - Dropout layer (TODO: removed, re-insert)

    A residual connection between the input and the output of the block is present

    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param dil: Amount of dilation
    :param use_bias: Use bias if True
    """

    def __init__(self, ch_in, ch_out, k_size, dil, use_bias=True):
        super(TempBlock2, self).__init__()

        self.k0 = k_size[0]
        self.k1 = k_size[1]

        self.dil0 = dil[0]
        self.dil1 = dil[1]

        self.ch_in0 = ch_in[0]
        self.ch_in1 = ch_in[1]

        self.ch_out0 = ch_out[0]
        self.ch_out1 = ch_out[1]

        self.use_bias = use_bias

        self.pad0 = nn.ConstantPad1d(
            padding=((self.k0 - 1) * 1, 0),
            value=0
        )
        self.tcn0 = nn.Conv1d(
            in_channels=self.ch_in0,
            out_channels=self.ch_out0,
            kernel_size=self.k0,
            stride=(2,),
            bias=self.use_bias
        )
        self.batchnorm0 = nn.BatchNorm1d(
            num_features=self.ch_out0
        )
        self.relu0 = nn.ReLU()

        self.pad1 = nn.ConstantPad1d(
            padding=((self.k1 - 1) * dil[1], 0),
            value=0
        )
        self.tcn1 = nn.Conv1d(
            in_channels=self.ch_in1,
            out_channels=self.ch_out1,
            kernel_size=self.k1,
            dilation=self.dil1,
            bias=self.use_bias
        )

        self.downsample = nn.Conv1d(
            in_channels=self.ch_in0,
            out_channels=self.ch_out1,
            stride=(2,),
            kernel_size=(1,),
            bias=self.use_bias
        )
        self.downsamplerelu = nn.ReLU()
        self.downsamplebn = nn.BatchNorm1d(
            num_features=self.ch_out1
        )

        self.batchnorm1 = nn.BatchNorm1d(
            num_features=self.ch_out1
        )

        self.reluadd = nn.ReLU()

    def forward(self, x):
        x1 = self.relu0(
            self.batchnorm0(
                self.tcn0(
                    self.pad0(x)
                )
            )
        )

        x2 = self.batchnorm1(
            self.tcn1(
                self.pad1(x1)
            )
        )
        res = self.downsample(x)
        res = self.downsamplebn(res)
        res = self.downsamplerelu(res)

        return self.reluadd(x2 + res)

    def init_weights(self):
        self.tcn0.weight.data.normal_(0, 0.01)
        self.tcn1.weight.data.normal_(0, 0.01)
        self.downsample.weight.data.normal_(0, 0.01)


class DSCnnSN(torch.nn.Module):
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
        self.depthpoint1 = SuperNetModule([
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.Conv2d(64, 64, 5, padding='same'),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ])
        self.bn11 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu11 = nn.ReLU()
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
        self.depthpoint2 = SuperNetModule([
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.Conv2d(64, 64, 5, padding='same'),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ])
        self.bn21 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu21 = nn.ReLU()
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
        self.depthpoint3 = SuperNetModule([
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.Conv2d(64, 64, 5, padding='same'),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ])
        self.bn31 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu31 = nn.ReLU()
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
        self.depthpoint4 = SuperNetModule([
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.Conv2d(64, 64, 5, padding='same'),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ])
        self.bn41 = nn.BatchNorm2d(64, momentum=0.99)
        self.relu41 = nn.ReLU()
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
        x = self.relu11(self.bn11(x))
        x = self.conv1(x)
        x = self.relu12(self.bn12(x))

        # Second layer of separable depthwise conv2d
        # x = self.depthwise2(x)
        # x = self.pointwise2(x)
        x = self.depthpoint2(x)
        x = self.relu21(self.bn21(x))
        x = self.conv2(x)
        x = self.relu22(self.bn22(x))

        # Third layer of separable depthwise conv2d
        # x = self.depthwise3(x)
        # x = self.pointwise3(x)
        x = self.depthpoint3(x)
        x = self.relu31(self.bn31(x))
        x = self.conv3(x)
        x = self.relu32(self.bn32(x))

        # Fourth layer of separable depthwise conv2d
        # x = self.depthwise4(x)
        # x = self.pointwise4(x)
        x = self.depthpoint4(x)
        x = self.relu41(self.bn41(x))
        x = self.conv4(x)
        x = self.relu42(self.bn42(x))

        x = self.dropout2(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.out(x)

        return x
