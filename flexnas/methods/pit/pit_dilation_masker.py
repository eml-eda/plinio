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

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .pit_binarizer import PITBinarizer


class PITDilationMasker(nn.Module):

    def __init__(self,
                 rf: int,
                 trainable: bool = True,
                 binarization_threshold: float = 0.5):
        super(PITDilationMasker, self).__init__()
        self.rf = rf
        self.gamma = Parameter(
            torch.empty(self._gamma_len, dtype=torch.float32).fill_(1.0), requires_grad=True)
        # this must be done after creating beta and gamma
        self.trainable = trainable
        self._binarization_threshold = binarization_threshold
        self._keep_alive = self._generate_keep_alive_mask()
        self._c_gamma = self._generate_c_matrix()

    def forward(self) -> torch.Tensor:
        keep_alive_gamma = torch.abs(self.gamma) * (1 - self._keep_alive) + self._keep_alive
        theta_gamma = torch.matmul(self._c_gamma, keep_alive_gamma)
        theta_gamma = PITBinarizer.apply(theta_gamma, self._binarization_threshold)
        return theta_gamma

    def _generate_keep_alive_mask(self) -> torch.Tensor:
        ka_gamma = torch.tensor([1.0] + [0.0] * (self._gamma_len - 1), dtype=torch.float32)
        return ka_gamma

    def _generate_c_matrix(self) -> torch.Tensor:
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
        return c_gamma

    @property
    def _gamma_len(self):
        return max(math.ceil(math.log(self.rf, 2)), 1)

    @property
    def trainable(self):
        return self.gamma.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        self.gamma.requires_grad = value
