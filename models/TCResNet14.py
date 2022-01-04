#*----------------------------------------------------------------------------*
#* Copyright (C) 2022 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Matteo Risso <matteo.risso@polito.it>                             *
#*----------------------------------------------------------------------------*

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from math import ceil, floor

class TCResNet14(nn.Module):
    """
    TCResNet14 architecture:
    Variable number of repeated instances of TemporalBlock1 and TemporalBlock2.
    """
    def __init__(self, config):
        super(TCResNet14, self).__init__()
        self.config = config

        self.pad0 = nn.ConstantPad1d(
                padding = (3-1, 0),
                value = 0
                )
        self.conv0 = nn.Conv1d(
                in_channels = self.config['input_size'],
                out_channels = self.config['num_channels'][0],
                kernel_size = 3,
                bias = self.config['use_bias']
                )
        
        self.tcn = TempNet(
                num_inputs = self.config['num_channels'][0], 
                num_channels = self.config['num_channels'][1:], 
                kernel_size = self.config['kernel_size'], 
                dropout = self.config['dropout'], 
                use_bias = self.config['use_bias'],
                )

        if self.config['avg_pool']:
            self.avgpool = nn.AvgPool1d(
                    kernel_size = 2
                    )
            self.ft_in = floor(13 / 2)
        else:
            self.ft_in = 13

        self.dpout = nn.Dropout(
                p = self.config['dropout']
                )
         
        self.out = nn.Linear(
                in_features = self.config['num_channels'][-1]*self.ft_in,
                out_features = self.config['output_size'],
                bias = self.config['use_bias'],
                )
        
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
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
    Temporal Convolutional Net composed of a number of an alternate number of TempBlock1 and TempBlock2 defined with a specific parameter.
    """
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.5, use_bias=True):
        super(TempNet, self).__init__()
        layers = list()
        num_levels = len(num_channels)
        original_rf = [(kernel_size - 1) * (2**i) + 1 for i in range(num_levels)]
        j = 0
        l = 0
        for i in range(num_levels):
            dilation_size = list()
            k = list()
            dilation_size.append(2 ** i)
            dilation_size.append(2 ** i)

            k.append(ceil(original_rf[i]/dilation_size[0]))
            k.append(ceil(original_rf[i]/dilation_size[1]))

            in_channels = [num_inputs, num_channels[0]] if i == 0 else [num_channels[i-1], num_channels[i]]
            out_channels = [num_channels[i], num_channels[i]]

            if (i % 2) != 0:
                layers += [TempBlock1(
                    ch_in = in_channels, 
                    ch_out = out_channels, 
                    k_size = k, 
                    dil = dilation_size,
                    dropout = dropout,
                    use_bias = use_bias,
                    )]
            else:
                layers += [TempBlock2(
                    ch_in = in_channels, 
                    ch_out = out_channels, 
                    k_size = k, 
                    dil = dilation_size,
                    dropout = dropout,
                    use_bias = use_bias,
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
    - Dropout layer

    A residual connection between the input and the output of the block is present

    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param dil: Amount of dilation
    :param dropout: Dropout rate
    :param use_bias: Use bias if True
    """
    def __init__(self, ch_in, ch_out, k_size, dil, dropout=0.2, use_bias=True):
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
            padding = ((self.k0 - 1) * dil[0], 0),
            value = 0
            )
        self.tcn0 = nn.Conv1d(
                in_channels = self.ch_in0,
                out_channels = self.ch_out0,
                kernel_size = self.k0,
                dilation = self.dil0,
                bias = self.use_bias
                )
        self.batchnorm0 = nn.BatchNorm1d(
                num_features = self.ch_out0
                )
        self.relu0 = nn.ReLU()

        self.pad1 = nn.ConstantPad1d(
            padding = ((self.k1 - 1) * dil[1], 0),
            value = 0
            )
        self.tcn1 = nn.Conv1d(
                in_channels = self.ch_in1,
                out_channels = self.ch_out1,
                kernel_size = self.k1,
                dilation = self.dil1,
                bias = self.use_bias
                )
        self.downsample = nn.Conv1d(
            in_channels = self.ch_in0,
            out_channels = self.ch_out1,
            kernel_size = 1,
            bias = self.use_bias
            )
        self.downsamplerelu = nn.ReLU()
        self.downsamplebn = nn.BatchNorm1d(
                num_features = self.ch_out1
                )
        self.batchnorm1 = nn.BatchNorm1d(
                num_features = self.ch_out1
                )    
        self.relu1 = nn.ReLU()

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
    - Dropout layer

    A residual connection between the input and the output of the block is present

    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param dil: Amount of dilation
    :param dropout: Dropout rate
    :param use_bias: Use bias if True
    """
    def __init__(self, ch_in, ch_out, k_size, dil, dropout=0.2,  use_bias=True):
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
            padding = ((self.k0 - 1) * 1, 0),
            value = 0
            )
        self.tcn0 = nn.Conv1d(
                in_channels = self.ch_in0,
                out_channels = self.ch_out0,
                kernel_size = self.k0,
                stride = 2,
                bias = self.use_bias
                )
        self.batchnorm0 = nn.BatchNorm1d(
                num_features = self.ch_out0
                )
        self.relu0 = nn.ReLU()

        self.pad1 = nn.ConstantPad1d(
            padding = ((self.k1 - 1) * dil[1], 0),
            value = 0
            )
        self.tcn1 = nn.Conv1d(
                in_channels = self.ch_in1,
                out_channels = self.ch_out1,
                kernel_size = self.k1,
                dilation = self.dil1,
                bias = self.use_bias
                )

        self.downsample = nn.Conv1d(
            in_channels = self.ch_in0,
            out_channels = self.ch_out1,
            stride = 2,
            kernel_size = 1,
            bias = self.use_bias
            )
        self.downsamplerelu = nn.ReLU()
        self.downsamplebn = nn.BatchNorm1d(
                num_features = self.ch_out1
            )
            
        self.batchnorm1 = nn.BatchNorm1d(
                num_features = self.ch_out1
                )    
        self.relu1 = nn.ReLU()

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