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

from typing import Dict, Any, Optional, cast, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from plinio.methods.mps.quant.quantizers import Quantizer, DummyQuantizer
from plinio.methods.mps.quant.backends.utils import binary_search
from .module import MATCHModule


class MATCHConv3d(nn.Conv3d, MATCHModule):
    """A nn.Module implementing an integer quantized Conv3d layer compatible
    with the MATCH backend.

    :param conv: the inner `nn.Conv3d` layer
    :type conv: nn.Conv3d
    :param in_quantizer: input activation quantizer
    :type in_quantizer: Type[Quantizer]
    :param out_quantizer: output activation quantizer
    :type out_quantizer: Type[Quantizer]
    :param w_quantizer: weight quantizer
    :type w_quantizer: Type[Quantizer]
    :param b_quantizer: bias quantizer
    :type b_quantizer: Type[Quantizer]
    """

    def __init__(
        self,
        conv: nn.Conv3d,
        in_quantizer: Quantizer,
        out_quantizer: Quantizer,
        w_quantizer: Quantizer,
        b_quantizer: Optional[Quantizer],
        scale_bit: int = 24,
        shift_pos: int = 24,
        dequantize_output: bool = False,
    ):
        super(MATCHConv3d, self).__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
        )

        self.scale_bit = scale_bit
        self.shift_pos = shift_pos
        self.dequantize_output = dequantize_output

        # Store precisions and quantizers
        self.in_quantizer = in_quantizer
        self.out_quantizer = out_quantizer
        self.w_quantizer = w_quantizer
        if self.bias is not None:
            self.b_quantizer = cast(Quantizer, b_quantizer)
        else:
            self.b_quantizer = lambda *args: None  # Do Nothing

        # Copy and integerize pretrained weights
        # N.B., is mandatory to first integerize weight
        # compute self.scale and self.shift which depends upon
        # self.w_quantizer.s_w which is updated every time we quantize a tensor
        with torch.no_grad():
            self.w_quantizer.dequantize = False
            int_weight = self.w_quantizer(conv.weight)
            int_weight = cast(torch.Tensor, int_weight)
            self.weight.copy_(int_weight)

        # Compute self.scale_fact and self.shift
        self.s_w = self.w_quantizer.scale
        self.s_x = self.in_quantizer.scale
        if type(self.out_quantizer) != DummyQuantizer:
            self.s_y = self.out_quantizer.scale
            self.skip_requant = False
        else:
            self.s_y = torch.tensor(1.0, device=self.device)
            self.skip_requant = True

        # Copy and integerize pretrained biases
        with torch.no_grad():
            if conv.bias is not None:
                self.b_quantizer.dequantize = False
                int_bias = self.b_quantizer(conv.bias, self.s_x, self.s_w)
                int_bias = cast(torch.Tensor, int_bias)

        self.scale, self.shift = self._integer_approximation(
            self.s_w, self.s_x, self.s_y, int_bias
        )
        with torch.no_grad():
            if conv.bias is not None:
                if not self.skip_requant:
                    int_bias = int_bias * self.scale
                    self.add_bias = int_bias.view(1, self.out_channels, 1, 1, 1)
                else:
                    self.bias = cast(torch.Tensor, self.bias)
                    self.bias.copy_(int_bias)
            else:
                self.add_bias = None

        # Done here to avoid the reshape op in fwd
        self.scale = self.scale.view(1, self.out_channels, 1, 1, 1)

        # Raise error if dilation != 1
        assert all(d == 1 for d in self.dilation), "Dilation != 1 not supported for Conv3d"


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of integer conv3d layer.

        It performs:
        - Convolution of the input with the integerized `self.weight` tensor.
        - Requantization (if not self.skip_requant):
            - Multiplication of the `self.scale`
            - Sum the integerized `self.bias` vector.
            - Divide by 2 ** `self.shift` amount.
            - Computes floor operation.
            - Apply clipped relu between 0 and (2 ** `out_precision` - 1)

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """

        if not self.skip_requant:  # This should happen on the last layer
            # Convolution
            out = F.conv3d(
                input,
                self.weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            # Multiply scale factor, sum bias, shift
            out = (out * self.scale + self.add_bias) / (2**self.shift)
            # Compute floor
            out = torch.floor(out)
            # Compute relu
            out = torch.clip(out, self.clip_inf, self.clip_sup)
        else:
            # Convolution
            out = F.conv3d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            if self.dequantize_output:
                out = out * self.s_w.view(1, -1, 1, 1, 1) * self.s_x

        return out

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            "out_quantizer": self.out_quantizer.summary(),
            "w_quantizer": self.w_quantizer.summary(),
            "scale_factor": self.scale_fact,
            "shift": self.shift,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def clip_inf(self):
        # Define ReLU inferior extreme
        return torch.tensor(0.0, device=self.device)

    @property
    def clip_sup(self):
        # Define ReLU superior extreme
        return torch.tensor(
            2 ** cast(int, self.out_quantizer.precision) - 1, device=self.device
        )

    def _integer_approximation(
        self,
        s_w: torch.Tensor,
        s_x: torch.Tensor,
        s_y: torch.Tensor,
        int_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute an approximation of `s_w * s_x / s_y` as `scale` / 2**`shift`
        where scale is a vector of len(s_w) integer on 32bits and shift is
        a scalar between [0, 31].
        It also checks that the computed scale values does not overflow when multiplied
        to `int_bias`.

        :param s_w: the floating-point weights' quantizer scale-factor
        :type s_w: torch.Tensor
        :param s_x: the floating-point input activations' quantizer scale-factor
        :type s_x: torch.Tensor
        :param s_y: the floating-point output activations' quantizer scale-factor
        :type s_y: torch.Tensor
        :param int_bias: the integer bias
        :type int_bias: torch.Tensor
        :return: a tuple containing the computed `scale` and `shift` factors
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        # Value to be approximated as `scale` / 2**`shift`
        target = s_w * s_x / s_y
        device = target.device
        target = target.clone().detach().cpu()
        int_bias = int_bias.clone().detach().cpu()

        # Integer approximation #
        params = {}
        upper_bound = 2 ** (self.scale_bit - 1)
        # Create a dict indexed by possible shift amounts, each entry of the dict
        # contains a list where for each channel a `scale` factor is selected as
        # the one minimizing abs(scale / 2**shift - target).
        for idx in range(len(target)):
            for sh in range(self.shift_pos):
                if sh not in params.keys():
                    params[sh] = []
                params[sh].append(
                    binary_search(2**-sh, 1, upper_bound, target[idx].item())
                )
        # For each `shift` amount compute the average difference between the
        # integer approximation and targets
        avg_diff = {}
        for sh in range(self.shift_pos):
            diff = []
            for idx in range(len(target)):
                diff.append(abs(params[sh][idx] / 2**sh - target[idx].item()))
            avg_diff[sh] = sum(diff) / len(diff)
        # Find the `shift` amount minimizing the avg difference between integer
        # approximation and targets
        min_diff = float("inf")
        min_scale, min_shift = None, None
        for key, val in avg_diff.items():
            scale = params[key]
            shift = key
            scaled_bias = int_bias * torch.tensor(scale)
            overflow = any(
                torch.logical_or(scaled_bias > 2**31 - 1, scaled_bias < -(2**31))
            )
            if val < min_diff and (not overflow):
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

