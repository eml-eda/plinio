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

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .pit_binarizer import PITBinarizer


class PITChannelMasker(nn.Module):

    def __init__(self,
                 out_channels: int,
                 trainable: bool = True,
                 keep_alive_channels: int = 1,
                 binarization_threshold: float = 0.5):
        super(PITChannelMasker, self).__init__()
        self.out_channels = out_channels
        self.alpha = Parameter(
            torch.empty(self.out_channels, dtype=torch.float32).fill_(1.0), requires_grad=True)
        # this should be done after creating alpha
        self.trainable = trainable
        self._binarization_threshold = binarization_threshold
        self._keep_alive = self._generate_keep_alive_mask(keep_alive_channels)

    def forward(self) -> torch.Tensor:
        # TODO: doing both the multiplication and the abs() is probably redundant
        keep_alive_alpha = torch.abs(self.alpha) * (1 - self._keep_alive) + self._keep_alive
        bin_alpha = PITBinarizer.apply(keep_alive_alpha, self._binarization_threshold)
        return bin_alpha

    def _generate_keep_alive_mask(self, keep_alive_channels: int) -> torch.Tensor:
        return torch.tensor([1.0] * keep_alive_channels + [0.0] * (self.out_channels - keep_alive_channels),
                            dtype=torch.float32)

    @property
    def trainable(self):
        return self.alpha.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        self.alpha.requires_grad = value
