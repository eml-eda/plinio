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


class PITDilationMasker(nn.Module):
    """A nn.Module implementing the creation of dilation masks for PIT

    :param rf: the static (i.e., maximum) receptive field of the layer to be masked
    :type rf: int
    :param trainable: should the masks be trained, defaults to True
    :type trainable: bool, optional
    """
    def __init__(self,
                 rf: int,
                 trainable: bool = True):
        super(PITDilationMasker, self).__init__()
        self.rf = rf
        self.gamma = Parameter(
            torch.empty(self._gamma_len, dtype=torch.float32).fill_(1.0), requires_grad=True)
        # this must be done after creating beta and gamma
        self.trainable = trainable
        self._keep_alive = self._generate_keep_alive_mask()
        self._c_gamma = self._generate_c_matrix()

    def forward(self) -> torch.Tensor:
        """The forward function that generates the binary masks from the trainable floating point
        shadow copies

        Implemented as described in the journal paper.

        :return: the binary masks
        :rtype: torch.Tensor
        """
        # this makes sure that the first "keep_alive" timestep is always binarized at 1, without
        # using ifs
        keep_alive_gamma = torch.abs(self.gamma) * (1 - self._keep_alive) + self._keep_alive
        theta_gamma = torch.matmul(self._c_gamma, keep_alive_gamma)
        return theta_gamma

    def _generate_keep_alive_mask(self) -> torch.Tensor:
        """Method called at creation time, to generate a "keep-alive" mask vector.

        For dilation masking, the first mask element (gamma_0) should always be preserved.

        :return: a binary keep-alive mask vector, with 1s corresponding to elements that should
        never be masked
        :rtype: torch.Tensor
        """
        ka_gamma = torch.tensor([1.0] + [0.0] * (self._gamma_len - 1), dtype=torch.float32)
        return ka_gamma

    def _generate_c_matrix(self) -> torch.Tensor:
        """Method called at creation time, to generate the C_gamma matrix.

        The C_gamma matrix is used to combine different dilation mask elements (gamma_i), as
        described in the journal paper.

        :return: the C_gamma matrix as tensor
        :rtype: torch.Tensor
        """
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
    def _gamma_len(self) -> int:
        """Compute the length of the gamma mask based on the receptive field

        :return: the integer length
        :rtype: int
        """
        return max(math.ceil(math.log(self.rf, 2)), 1)

    @property
    def trainable(self) -> bool:
        """Returns true if this mask is trainable

        :return: true if this mask is trainable
        :rtype: bool
        """
        return self.gamma.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        """Set to true to make the channel masker trainable

        :param value: true to make the channel masker trainable
        :type value: bool
        """
        self.gamma.requires_grad = value
