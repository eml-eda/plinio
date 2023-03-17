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


class MinMax_Weight(nn.Module, Quantizer):
    """A nn.Module implementing a min-max quantization strategy for weights.

    :param num_bits: quantization precision
    :type num_bits: int
    :param cout: number of output channels
    :type cout: int
    :param symmetric: wether the weight upper and lower bound should be the same
    :type init_clip_val: bool
    :param dequantize: whether the output should be fake-quantized or not
    :type dequantize: bool
    """
    def __init__(self,
                 num_bits: int,
                 cout: int,
                 symmetric: bool = True,
                 dequantize: bool = True):
        super(MinMax_Weight, self).__init__()
        self.num_bits = num_bits
        if symmetric:
            self.qtz_func = MinMax_Sym_STE if symmetric else MinMax_Asym_STE
            self.compute_min_max = self._compute_min_max_sym
        else:
            self.qtz_func = MinMax_Asym_STE
            self.compute_min_max = self._compute_min_max_asym
        self.dequantize = dequantize
        self.register_buffer('ch_max', torch.Tensor(cout))
        self.register_buffer('ch_min', torch.Tensor(cout))
        # self.s_w = torch.Tensor(cout)
        # self.s_w.fill_(1.)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the MinMax weight quantizer.

        Compute quantization using the whole weights' value span and implements STE
        for the backward pass

        :param input: the input float weights tensor
        :type input: torch.Tensor
        :return: the output fake-quantized weights tensor
        :rtype: torch.Tensor
        """
        self.ch_min, self.ch_max = self.compute_min_max(input)
        input_q = self.qtz_func.apply(input,
                                      self.ch_min,
                                      self.ch_max,
                                      self.num_bits,
                                      self.dequantize)
        return input_q

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

    @property
    def s_w(self) -> torch.Tensor:
        """Return the computed scale factor which depends upon self.num_bits and
        weights magnitude

        :return: the scale factor
        :rtype: torch.Tensor
        """
        ch_range = self.ch_max - self.ch_min
        if self.num_bits != 0:
            ch_range.masked_fill_(ch_range.eq(0), 1)
            n_steps = 2 ** self.num_bits - 1
            scale_factor = ch_range / n_steps
        else:
            scale_factor = torch.zeros(ch_range.shape, device=ch_range.device)
        return scale_factor

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

    def _compute_min_max_sym(self, input: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes symmetric ch_min and ch_max of the given input tensor.

        :param input: the input tensor to be analyzed
        :type input: torch.Tensor
        :return: a tuple containing respectively the min and the max.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        ch_max, _ = input.view(input.size(0), -1).abs().max(1)
        ch_min = -1 * self.ch_max
        return ch_min, ch_max

    def _compute_min_max_asym(self, input: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes asymmetric ch_min and ch_max of the given input tensor.

        :param input: the input tensor to be analyzed
        :type input: torch.Tensor
        :return: a tuple containing respectively the min and the max.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        ch_max, _ = input.view(input.size(0), -1).max(1)
        ch_min, _ = input.view(input.size(0), -1).min(1)
        return ch_min, ch_max

    def __repr__(self):
        msg = (
            f'{self.__class__.__name__}'
            f'(num_bits={self.num_bits}, '
            f'scale_factor={self.s_w})'
        )
        return msg


class MinMax_Asym_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ch_min, ch_max, num_bits, dequantize):
        return _min_max_quantize(x, ch_min, ch_max, num_bits, dequantize)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class MinMax_Sym_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ch_min, ch_max, num_bits, dequantize):
        return _min_max_quantize(x, ch_min, ch_max, num_bits, dequantize)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def _min_max_quantize(x, ch_min, ch_max, num_bits, dequantize):

    # if the precision is equal to 0 bit, return a zeros-filled tensor
    if num_bits != 0:
        # Compute scale factor
        ch_range = ch_max - ch_min
        ch_range.masked_fill_(ch_range.eq(0), 1)
        n_steps = 2 ** num_bits - 1
        scale_factor = ch_range / n_steps

        # Reshape
        shape = (x.shape[0],) + (1,) * len(x.shape[1:])
        scale_factor = scale_factor.view(shape)

        # Quantize
        y = torch.round(x / scale_factor)

        if dequantize:
            y = y * scale_factor

    else:  # 0-bit precision
        y = torch.zeros(x.shape, device=x.device)
    # return y, scale_factor
    return y
