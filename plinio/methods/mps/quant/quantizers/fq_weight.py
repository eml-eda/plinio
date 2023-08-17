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

from typing import Dict, Any, Optional, Iterator, Tuple
import torch
import torch.fx as fx
import torch.nn as nn
from .quantizer import Quantizer


class FQWeight(Quantizer):
    """A nn.Module implementing the FQ quantization strategy for weights.
    More details can be found at: https://arxiv.org/abs/1912.09356

    :param precision: quantization precision
    :type precision: int
    :param cout: number of output channels, used only if ch_wise is True
    :type cout: int
    :param ch_wise: wether the quantization is channel-wise or not
    :type ch_wise: bool
    :param train_scale_param: wether the scale_param should be trained or not
    :type train_scale_param: bool
    :param dequantize: whether the output should be fake-quantized or not
    :type dequantize: bool
    """
    def __init__(self,
                 precision: int,
                 cout: int,
                 ch_wise: bool = True,
                 train_scale_param: bool = True,
                 dequantize: bool = True):
        super(FQWeight, self).__init__()
        self.precision = precision
        self.quant_bins = 2**(self.precision - 1) - 1
        self.cout = cout
        self.ch_wise = ch_wise
        self.train_scale_param
        self.dequantize = dequantize

        # Trainable scale param
        self.n_s = cout if ch_wise else 1
        self.scale_param = nn.Parameter(torch.Tensor(self.n_s),
                                        requires_grad=train_scale_param)
        # self.register_buffer('s_w', torch.Tensor(cout))
        self.s_w = torch.Tensor(cout)
        self.s_w.fill_(1.)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the FQ weight quantizer.

        Compute quantization using the learned scale-param and implements STE
        for the backward pass

        :param input: the input float weights tensor
        :type input: torch.Tensor
        :return: the output fake-quantized weights tensor
        :rtype: torch.Tensor
        """
        # Having a positive scale factor is preferable to avoid instabilities
        exp_scale_param = torch.exp(self.scale_param).view(
            (self.n_s,) + (1,) * len(input.shape[1:]))
        input_scaled = input / exp_scale_param
        # Quantize
        input_q = FQ_Quant_STE.apply(input_scaled,
                                     self.quant_bins)
        self.s_w = exp_scale_param / self.quant_bins
        if self.dequantize:
            return input_q * exp_scale_param
        else:
            return input_q * self.quant_bins

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule, backend: Optional[str]):
        """Replaces a fx.Node corresponding to a Quantizer, with a "backend-aware" layer
        within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        :param backend: an optional string specifying the target backend
        :type backend: Optional[str]
        """
        raise NotImplementedError("TODO")

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer quantization hyperparameters

        :return: a dictionary containing the optimized layer quantization hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'scale_factor': self.s_w,
        }

    def named_quant_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the quantization parameters of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API,
        but Quantizer never have sub-layers TODO: check if true
        :type recurse: bool
        :return: an iterator over the quantization parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        for name, param in self.named_parameters(
                prfx + "weight_quantizer", recurse):
            yield name, param

    def __repr__(self):
        msg = (
            f'{self.__class__.__name__}'
            f'(precision={self.precision}, '
            f'scale_factor={self.s_w})'
        )
        return msg


class FQ_Quant_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        # Hardtanh
        output = torch.clamp(x, -1, 1)
        # Multiply by number of levels
        output = output * n
        # Round
        output = torch.round(output)
        # Divide by number of levels
        output = output / n
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
