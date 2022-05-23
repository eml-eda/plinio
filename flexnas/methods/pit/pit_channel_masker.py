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
    """A nn.Module implementing the creation of output channels in the layer to be masked

    :param out_channels: the static (i.e., maximum) number of output channels in the mask
    :type out_channels: int
    :param trainable: should the masks be trained, defaults to True
    :type trainable: bool, optional
    :param keep_alive_channels: how many channels should always be kept alive (binarized at 1),
    defaults to 1
    :type keep_alive_channels: int, optional
    :param binarization_threshold: the binarization threshold, defaults to 0.5
    :type binarization_threshold: float, optional
    """
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
        """The forward function that generates the binary masks from the trainable floating point
        shadow copies.

        :return: the binary masks
        :rtype: torch.Tensor
        """
        # this makes sure that the first "keep_alive" channels are always binarized at 1, without
        # using ifs
        keep_alive_alpha = torch.abs(self.alpha) * (1 - self._keep_alive) + self._keep_alive
        bin_alpha = PITBinarizer.apply(keep_alive_alpha, self._binarization_threshold)
        return bin_alpha

    def _generate_keep_alive_mask(self, keep_alive_channels: int) -> torch.Tensor:
        """Method called at creation time, to generate a "keep-alive" mask vector.

        This is a vector with a number of leading 1s equal to the number of channels that should
        never be eliminated

        :return: a binary keep-alive mask vector, with 1s corresponding to elements that should
        never be masked
        :rtype: torch.Tensor
        """
        return torch.tensor(
            [1.0] * keep_alive_channels + [0.0] * (self.out_channels - keep_alive_channels),
            dtype=torch.float32)

    @property
    def trainable(self) -> bool:
        """Returns true if this mask is trainable

        :return: true if this mask is trainable
        :rtype: bool
        """
        return self.alpha.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        """Set to true to make the channel masker trainable

        :param value: true to make the channel masker trainable
        :type value: bool
        """
        self.alpha.requires_grad = value
