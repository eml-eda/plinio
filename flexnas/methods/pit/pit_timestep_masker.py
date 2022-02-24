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


class PITTimestepMasker(nn.Module):

    def __init__(self,
                 rf: int,
                 trainable: bool = True,
                 binarization_threshold: float = 0.5):
        super(PITTimestepMasker, self).__init__()
        self.rf = rf
        self.beta = Parameter(
            torch.empty(self.rf, dtype=torch.float32).fill_(1.0), requires_grad=True)
        # this must be done after creating beta and gamma
        self.trainable = trainable
        self._binarization_threshold = binarization_threshold
        self._keep_alive = self._generate_keep_alive_mask()
        self._c_beta = self._generate_c_matrix()

    def forward(self) -> torch.Tensor:
        keep_alive_beta = torch.abs(self.beta) * (1 - self._keep_beta) + self._keep_beta
        theta_beta = torch.matmul(self._c_beta, keep_alive_beta)
        theta_beta = PITBinarizer.apply(theta_beta, self._binarization_threshold)
        return theta_beta

    def _generate_keep_alive_mask(self) -> torch.Tensor:
        ka_beta = torch.tensor([1.0] + [0.0] * (self.rf - 1), dtype=torch.float32)
        return ka_beta

    def _generate_c_matrix(self) -> torch.Tensor:
        c_beta = torch.triu(torch.ones((self.rf, self.rf), dtype=torch.float32))
        return c_beta

    @property
    def _gamma_len(self):
        return max(math.ceil(math.log(self.rf, 2)), 1)

    @property
    def trainable(self):
        return self.beta.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        self.beta.requires_grad = value
