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

from typing import Dict, Any, Optional, cast, Type, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from plinio.methods.mixprec.quant.quantizers import Quantizer
from .dory_module import DORYModule


class DORYConv2d(nn.Conv2d, DORYModule):
    """A nn.Module implementing an integer quantized Conv2d layer compatible
    with the DORY backend

    :param conv: the inner `nn.Conv2d` layer
    :type conv: nn.Conv2d
    :param a_precision: the input activation quantization precision
    :type a_precision: int
    :param w_precision: the weights' quantization precision
    :type w_precision: int
    :param in_a_quantizer: input activation quantizer
    :type in_a_quantizer: Type[Quantizer]
    :param out_a_quantizer: output activation quantizer
    :type out_a_quantizer: Type[Quantizer]
    :param w_quantizer: weight quantizer
    :type w_quantizer: Type[Quantizer]
    :param b_quantizer: bias quantizer
    :type b_quantizer: Type[Quantizer]
    """
    def __init__(self,
                 conv: nn.Conv2d,
                 a_precision: int,
                 w_precision: int,
                 in_a_quantizer: Type[Quantizer],
                 out_a_quantizer: Type[Quantizer],
                 w_quantizer: Type[Quantizer],
                 b_quantizer: Optional[Type[Quantizer]]):
        super(DORYConv2d, self).__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode)

        # Store precisions and quantizers
        self.a_precision = a_precision
        self.w_precision = w_precision
        self.in_a_quantizer = in_a_quantizer
        self.out_a_quantizer = out_a_quantizer
        self.w_quantizer = w_quantizer
        if self.bias is not None:
            b_quantizer = cast(Type[Quantizer], b_quantizer)
            self.b_quantizer = b_quantizer
        else:
            self.b_quantizer = lambda *args: None  # Do Nothing

        # Compute self.scale_fact and self.shift
        # TODO: to avoid to ignore linter warning refactor s_w and s_a
        # to be simply s (or similar name) and put it as a property
        # in the abstract Quantizer class
        self.s_w = self.w_quantizer.s_w  # type: ignore
        self.s_x = self.in_a_quantizer.s_a  # type: ignore
        self.s_y = self.out_a_quantizer.s_a  # type: ignore
        self.scale, self.shift = self._integer_approximation(self.s_w, self.s_x, self.s_y)

        # Copy and integerize pretrained weights and biases
        with torch.no_grad():
            self.w_quantizer.dequantize = False
            int_weight = self.w_quantizer(conv.weight)
            int_weight = cast(torch.Tensor, int_weight)
            self.weight.copy_(int_weight)
            if conv.bias is not None:
                self.b_quantizer.dequantize = False
                self.bias = cast(nn.parameter.Parameter, self.bias)
                int_bias = self.b_quantizer(conv.bias) * self.scale  # TODO: check this mul
                self.bias.copy_(int_bias)
            else:
                self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of integer conv2d layer.

        It performs:
        - Convolution of the input with the integerized `self.weight` tensor.
        - Multiplication of the `self.scale`
        - Sum the integerized `self.bias` vector.
        - Divide by 2 ** `self.shift` amount.
        - Computes floor operation.
        - Apply clipped relu between 0 and (2 ** `self.a_precision` - 1)

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        # Convolution
        conv_out = F.conv2d(input, self.weight, None, self.stride,
                            self.padding, self.dilation, self.groups)
        # Multiply scale factor, sum bias, shift
        scale_out = (conv_out * self.scale + self.bias) / (2 ** self.shift)
        # Compute floor
        floor_out = torch.floor(scale_out)
        # Compute relu
        relu_out = torch.min(torch.max(torch.tensor(0.), floor_out),
                             torch.tensor(2 ** self.a_precision - 1))

        return relu_out

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'a_precision': self.a_precision,
            'w_precision': self.w_precision,
            'scale_factor': self.scale_fact,
            'shift': self.shift,
        }

    def _integer_approximation(self,
                               s_w: torch.Tensor,
                               s_x: torch.Tensor,
                               s_y: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
