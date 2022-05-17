# *----------------------------------------------------------------------------*
# * Copyright (C) 2021 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Daniele Jahier Pagliari <daniele.jahier@polito.it>                *
# *----------------------------------------------------------------------------*

from typing import Optional, Tuple
import torch
import torch.nn as nn
from flexnas.utils.features_calculator import FeaturesCalculator
from .pit_channel_masker import PITChannelMasker
from .pit_timestep_masker import PITTimestepMasker
from .pit_dilation_masker import PITDilationMasker


class PITConv1d(nn.Conv1d):

    regularizers = ('size', 'flops')

    def __init__(self,
                 conv: nn.Conv1d,
                 out_length: int,
                 regularizer: str,
                 out_channel_masker: PITChannelMasker,
                 timestep_masker: PITTimestepMasker,
                 dilation_masker: PITDilationMasker,
                 ):
        super(PITConv1d, self).__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode)
        self.weight = conv.weight
        self.bias = conv.bias
        self.out_length = out_length
        if regularizer not in PITConv1d.regularizers:
            raise ValueError("Unsupported regularizer {}".format(regularizer))
        self.regularizer = regularizer
        self._input_size_calculator = None
        self.out_channel_masker = out_channel_masker
        self.timestep_masker = timestep_masker
        self.dilation_masker = dilation_masker
        self._beta_norm, self._gamma_norm = self._generate_norm_constants()
        self.register_buffer('out_channels_eff', torch.tensor(0, dtype=torch.float32))
        self.register_buffer('k_eff', torch.tensor(0, dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # for now we keep the same order of the old version (ch --> dil --> rf)
        # but it's probably more natural to do ch --> rf --> dil
        bin_alpha = self.out_channel_masker()
        # TODO: check that the result is correct after removing the two transposes
        pruned_weight = torch.mul(self.weight, bin_alpha.unsqueeze(1).unsqueeze(1))
        theta_gamma = self.dilation_masker()
        pruned_weight = torch.mul(theta_gamma, pruned_weight)
        theta_beta = self.timestep_masker()
        pruned_weight = torch.mul(theta_beta, pruned_weight)

        # conv operation
        y = self._conv_forward(input, pruned_weight, self.bias)

        # save info for regularization
        norm_theta_beta = torch.mul(theta_beta, self._beta_norm)
        norm_theta_gamma = torch.mul(theta_gamma, self._gamma_norm)
        self.out_channels_eff.copy_(torch.sum(bin_alpha))
        self.k_eff.copy_(torch.sum(torch.mul(norm_theta_beta, norm_theta_gamma)))

        return y

    def get_regularization_loss(self):
        cin = self.input_size_calculator.input_features
        cost = cin * self.out_channels_eff * self.k_eff
        if self.regularizer == 'flops':
            cost = cost * self.out_length
        return cost

    def _generate_norm_constants(self) -> Tuple[torch.Tensor, torch.Tensor]:
        beta_norm = torch.tensor([1.0/(self.rf - i) for i in range(self.rf)], dtype=torch.float32)
        gamma_norm = []
        for i in range(self.rf):
            k_i = 0
            for p in range(self.dilation_masker._gamma_len):
                k_i += 0 if i % 2**p == 0 else 1
            gamma_norm.append(1.0/(self.dilation_masker._gamma_len - k_i))
        gamma_norm = torch.tensor(gamma_norm, dtype=torch.float32)
        return beta_norm, gamma_norm

    @property
    def input_size_calculator(self) -> Optional[FeaturesCalculator]:
        return self._input_size_calculator

    @input_size_calculator.setter
    def input_size_calculator(self, calc: FeaturesCalculator):
        self._input_size_calculator = calc

    @property
    def rf(self):
        return self.kernel_size[0]

    @property
    def train_channels(self):
        return self.out_channel_masker.trainable

    @train_channels.setter
    def train_channels(self, value: bool):
        self.out_channel_masker.trainable = value

    @property
    def train_rf(self):
        return self.timestep_masker.trainable

    @train_rf.setter
    def train_rf(self, value: bool):
        self.timestep_masker.trainable = value

    @property
    def train_dilation(self):
        return self.dilation_masker.trainable

    @train_dilation.setter
    def train_dilation(self, value: bool):
        self.dilation_masker.trainable = value
