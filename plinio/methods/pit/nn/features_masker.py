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

from typing import cast
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class PITFeaturesMasker(nn.Module):
    """A nn.Module implementing the creation of output channels in the layer to be masked

    :param out_channels: the static (i.e., maximum) number of output channels in the mask
    :type out_channels: int
    :param trainable: should the masks be trained, defaults to True
    :type trainable: bool, optional
    :param keep_alive_channels: how many channels should always be kept alive (binarized at 1),
    defaults to 1
    :type keep_alive_channels: int, optional
    """
    def __init__(self,
                 out_channels: int,
                 trainable: bool = True,
                 keep_alive_channels: int = 1):
        super(PITFeaturesMasker, self).__init__()
        self.out_channels = out_channels
        self.alpha = Parameter(
            torch.empty(self.out_channels, dtype=torch.float32).fill_(1.0), requires_grad=True)
        # this should be done after creating alpha
        self.trainable = trainable
        self.register_buffer('_keep_alive', self._generate_keep_alive_mask(keep_alive_channels))

    @property
    def theta(self) -> torch.Tensor:
        """The forward function that generates the binary masks from the trainable floating point
        shadow copies.

        :return: the binary masks
        :rtype: torch.Tensor
        """
        # this makes sure that the first "keep_alive" channels are always binarized at 1, without
        # using ifs
        ka = cast(torch.Tensor, self._keep_alive)
        keep_alive_alpha = torch.abs(self.alpha) * (1 - ka) + ka
        return keep_alive_alpha

    def _generate_keep_alive_mask(self, keep_alive_channels: int) -> torch.Tensor:
        """Method called at creation time, to generate a "keep-alive" mask vector.

        This is a vector with a number of leading 1s equal to the number of channels that should
        never be eliminated

        :return: a binary keep-alive mask vector, with 1s corresponding to elements that should
        never be masked
        :rtype: torch.Tensor
        """
        # keep alive the last channel for consistency with rf and dilation
        return torch.tensor(
            [0.0] * (self.out_channels - keep_alive_channels) + [1.0] * keep_alive_channels,
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


class PITFrozenFeaturesMasker(PITFeaturesMasker):
    """A special case for the above masker used only for output nodes. Can never be trainable"""
    def __init__(self,
                 out_channels: int,
                 trainable: bool = True,
                 keep_alive_channels: int = 1):
        super(PITFrozenFeaturesMasker, self).__init__(
                out_channels, trainable=trainable, keep_alive_channels=keep_alive_channels)
        self.alpha.requires_grad = False
        self.register_buffer('_fixed_alpha', torch.ones(self.out_channels, dtype=torch.float32))

    @property
    def theta(self) -> torch.Tensor:
        return self._fixed_alpha

    @property
    def trainable(self) -> bool:
        return False

    @trainable.setter
    def trainable(self, value: bool):
        pass

