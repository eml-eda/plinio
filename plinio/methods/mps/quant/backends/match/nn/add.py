# *----------------------------------------------------------------------------*
# * Copyright (C) 2022 Politecnico di Torino, Italy                            *
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
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *----------------------------------------------------------------------------*

from typing import Tuple
import torch
import torch.nn as nn
from plinio.methods.mps.quant.quantizers import Quantizer
from plinio.methods.mps.quant.backends.utils import binary_search
from .module import MATCHModule


class MATCHAdd(nn.Module, MATCHModule):
    """A nn.Module implementing an integer Add layer compatible with the MATCH
    backend.

    :param quantizer: output activations quantizer
    :type quantizer: Type[Quantizer]
    :param scale_bit: number of bits for the scale factor
    :type scale_bit: int
    :param shift_pos: number of bits for the shift position
    :type shift_pos: int
    """

    def __init__(
        self,
        quantizer: Quantizer,
        scale_bit: int = 24,
        shift_pos: int = 24,
    ):
        super(MATCHAdd, self).__init__()
        self.scale_bit = scale_bit
        self.shift_pos = shift_pos

        self.quantizer = quantizer
        self.s_x = quantizer.scale

        # self.scale, self.shift = self._integer_approximation(self.s_x)
        # NOTE: from graph construction we now that we will requantize in the same way
        # as the inputs, so we can just skip requantization on the add node.
        # The operation is kept only to generate the correct pattern in the ONNX
        self.scale, self.shift = (
            torch.tensor(1.0, device=self.device),
            torch.tensor(0.0, device=self.device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (self.scale * x) / (2**self.shift)
        out = torch.floor(out)
        out = torch.clip(out, self.clip_inf, self.clip_sup)
        return out

    @property
    def device(self):
        return next(self.quantizer.parameters()).device

    @property
    def clip_inf(self):
        # Define ReLU inferior extreme
        return torch.tensor(0.0, device=self.device)

    @property
    def clip_sup(self):
        return torch.tensor(2**self.quantizer.precision - 1, device=self.device)

    def _integer_approximation(
        self,
        s_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute an approximation of `s_x` as `scale` / 2**`shift`
        where scale is a vector of len(s_w) integer on 32bits and shift is
        a scalar between [0, 31].
        to `int_bias`.

        :param s_x: the floating-point input activations' quantizer scale-factor
        :type s_x: torch.Tensor
        :return: a tuple containing the computed `scale` and `shift` factors
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        # Value to be approximated as `scale` / 2**`shift`
        target = s_x
        device = target.device
        target = target.clone().detach().cpu()

        # Integer approximation #
        params = {}
        upper_bound = 2 ** (self.scale_bit - 1)
        # Create a dict indexed by possible shift amounts, each entry of the dict
        # contains a `scale` factor selected as the one minimizing abs(scale / 2**shift - target).
        for sh in range(self.shift_pos):
            params[sh] = binary_search(2**-sh, 1, upper_bound, target.item())
        # For each `shift` amount compute the average difference between the
        # integer approximation and targets
        diff = {}
        for sh in range(self.shift_pos):
            diff[sh] = abs(params[sh] / 2**sh - target.item())
        # Find the `shift` amount minimizing the difference between integer
        # approximation and targets
        min_diff = float("inf")
        min_scale, min_shift = None, None
        for key, val in diff.items():
            scale = params[key]
            shift = key
            if val < min_diff:
                min_diff = val
                min_scale = scale
                min_shift = shift

        # Build tensors with selected quantities, move to `device` and return
        scale_t = torch.tensor(min_scale, device=device)
        shift_t = torch.tensor(
            [
                min_shift,
            ],
            device=device,
        )
        return scale_t, shift_t
