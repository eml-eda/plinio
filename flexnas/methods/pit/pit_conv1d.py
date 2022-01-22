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

from typing import Tuple
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .pit_binarizer import PITBinarizer


# NOTE: made dummy for now, to be replaced with Matteo's implementation later

class PITConv1d(nn.Conv1d):

    regularizers = ('size', 'flops')

    def __init__(self,
                 conv: nn.Conv1d,
                 regularizer: str,
                 train_channels: bool = True,
                 train_rf: bool = True,
                 train_dilation: bool = True,
                 keep_alive_channels: int = 1,
                 binarization_threshold: float = 0.5):
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
        if regularizer not in PITConv1d.regularizers:
            raise ValueError("Unsupported regularizer {}".format(regularizer))
        self.regularizer = regularizer
        self.alpha = Parameter(
            torch.empty(self.out_channels, dtype=torch.float32).fill_(1.0), requires_grad=True)
        self.beta = Parameter(
            torch.empty(self.rf, dtype=torch.float32).fill_(1.0), requires_grad=True)
        self.gamma = Parameter(
            torch.empty(self._gamma_len, dtype=torch.float32).fill_(1.0), requires_grad=True)
        # this must be done after creating the masks
        self.train_channels = train_channels
        self.train_rf = train_rf
        self.train_dilation = train_dilation
        self._binarization_threshold = binarization_threshold
        self._keep_alpha, self._keep_beta, self._keep_gamma = self._generate_keep_alive_masks(keep_alive_channels)
        self._c_beta, self._c_gamma = self._generate_c_matrices()
        self._beta_norm, self._gamma_norm = self._generate_norm_constants()
        self.register_buffer('out_channels_eff', torch.tensor(0, dtype=torch.float32))
        self.register_buffer('k_eff', torch.tensor(0, dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # same order as in old version. probably more natural to do ch -> rf -> dil ?

        # keep alive
        # TODO: doing both the multiplication and the abs() is probably redundant
        keep_alive_alpha = torch.abs(self.alpha) * (1 - self._keep_alpha) + self._keep_alpha
        keep_alive_beta = torch.abs(self.beta) * (1 - self._keep_beta) + self._keep_beta
        keep_alive_gamma = torch.abs(self.gamma) * (1 - self._keep_gamma) + self._keep_gamma

        # channels mask
        bin_alpha = PITBinarizer.apply(keep_alive_alpha, self._binarization_threshold)
        # TODO: can we skip the two transposes? what's the overhead?
        masked_channels = torch.mul(self.weight.transpose(0, 2), bin_alpha)
        pruned_weight = masked_channels.transpose(0, 2)

        # dilation mask
        theta_gamma = torch.matmul(self._c_gamma, keep_alive_gamma)
        theta_gamma = PITBinarizer.apply(theta_gamma, self._binarization_threshold)
        pruned_weight = torch.mul(theta_gamma, pruned_weight)

        # filter mask
        theta_beta = torch.matmul(self._c_beta, keep_alive_beta)
        theta_beta = PITBinarizer.apply(theta_beta, self._binarization_threshold)
        pruned_weight = torch.mul(theta_beta, pruned_weight)

        # save info for regularization
        norm_theta_beta = torch.mul(theta_beta, self._beta_norm)
        norm_theta_gamma = torch.mul(theta_gamma, self._gamma_norm)
        print(theta_beta, norm_theta_beta, self.rf)
        self.k_eff.copy_(torch.sum(torch.mul(norm_theta_beta, norm_theta_gamma)))
        self.out_channels_eff.copy_(torch.sum(bin_alpha))

        return self._conv_forward(input, pruned_weight, self.bias)

    def _generate_keep_alive_masks(self, keep_alive_channels: int) -> Tuple[torch.Tensor, ...]:
        ka_alpha = torch.tensor([1.0] * keep_alive_channels + [0.0] * (self.out_channels - keep_alive_channels),
                                dtype=torch.float32)
        ka_beta = torch.tensor([1.0] + [0.0] * (self.rf - 1), dtype=torch.float32)
        ka_gamma = torch.tensor([1.0] + [0.0] * (self._gamma_len - 1), dtype=torch.float32)
        return ka_alpha, ka_beta, ka_gamma

    def _generate_c_matrices(self) -> Tuple[torch.Tensor, ...]:
        c_beta = torch.triu(torch.ones((self.rf, self.rf), dtype=torch.float32))
        c_gamma = []
        # generate:
        # 111111111
        # 101010101
        # 100010001
        # etc
        for i in range(self._gamma_len):
            c_gamma_i = [1.0 if j % (2**i) == 0 else 0.0 for j in range(self.rf)]
            c_gamma.append(c_gamma_i)
        c_gamma = torch.tensor(c_gamma, dtype=torch.float32)
        # transpose & flip
        c_gamma = torch.transpose(c_gamma, 0, 1)
        c_gamma = torch.fliplr(c_gamma)
        return c_beta, c_gamma

    def _generate_norm_constants(self) -> Tuple[torch.Tensor, ...]:
        beta_norm = torch.tensor([1.0/(self.rf - i) for i in range(self.rf)], dtype=torch.float32)
        gamma_norm = []
        for i in range(self.rf):
            k_i = 0
            for p in range(self._gamma_len):
                k_i += 0 if i % 2**p == 0 else 1
            gamma_norm.append(1.0/(self._gamma_len - k_i))
        gamma_norm = torch.tensor(gamma_norm, dtype=torch.float32)
        return beta_norm, gamma_norm

    @property
    def rf(self):
        return self.kernel_size[0]

    @property
    def _gamma_len(self):
        return max(math.ceil(math.log(self.rf, 2)), 1)

    @property
    def train_channels(self):
        return self.alpha.requires_grad

    @train_channels.setter
    def train_channels(self, value: bool):
        self.alpha.requires_grad = value

    @property
    def train_rf(self):
        return self.beta.requires_grad

    @train_rf.setter
    def train_rf(self, value: bool):
        self.beta.requires_grad = value

    @property
    def train_dilation(self):
        return self.gamma.requires_grad

    @train_dilation.setter
    def train_dilation(self, value: bool):
        self.gamma.requires_grad = value
