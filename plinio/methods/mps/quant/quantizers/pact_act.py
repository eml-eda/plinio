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


class PACTAct(nn.Module, Quantizer):
    """A nn.Module implementing the PACT (PArametrized Clipping Activation)
    quantization strategy for activations.
    More details can be found at: https://openreview.net/forum?id=By5ugjyCb

    :param num_bits: quantization precision
    :type num_bits: int
    :param cout: dummy argument used to maintain same interface with other quantizers
    :type cout: None
    :param init_clip_val: input upper bound
    :type init_clip_val: float
    :param dequantize: whether the output should be fake-quantized or not
    :type dequantize: bool
    """
    def __init__(self,
                 num_bits: int,
                 cout: None = None,
                 init_clip_val: float = 6.,
                 dequantize: bool = True):
        super(PACTAct, self).__init__()
        self._num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([init_clip_val]), requires_grad=True)
        self._dequantize = dequantize
        # Buffer is probably wrong choice cause we might need gradients (?)
        # self.register_buffer('s_a', torch.Tensor(1))
        # self.s_a = torch.Tensor(1)
        # self.s_a.fill_(1.)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the PACT activartion quantizer.

        Compute quantization using the PACT strategy and implements STE
        for the backward pass

        :param input: the input float activations tensor
        :type input: torch.Tensor
        :return: the output fake-quantized activations tensor
        :rtype: torch.Tensor
        """
        # input_q, s_a = PACT_Act_STE.apply(input,
        input_q = PACT_Act_STE.apply(input,
                                     self.num_bits,
                                     self.clip_val,
                                     self.dequantize)
        # self.s_a = 1 / s_a
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
    def s_a(self) -> torch.Tensor:
        """Return the computed scale factor which depends upon self.num_bits and
        self.clip_val

        :return: the scale factor
        :rtype: torch.Tensor
        """
        return self.clip_val.data[0] / (2 ** self.num_bits - 1)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer quantization hyperparameters

        :return: a dictionary containing the optimized layer quantization hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'clip_val': self.clip_val,
            'scale_factor': self.s_a
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
                prfx + "act_quantizer", recurse):
            yield name, param

    def __repr__(self):
        msg = (
            f'{self.__class__.__name__}'
            f'(num_bits={self.num_bits}, '
            f'clip_val={self.clip_val}, '
            f'scale_factor={self.s_a})'
        )
        return msg

    @property
    def num_bits(self) -> int:
        return self._num_bits

    @num_bits.setter
    def num_bits(self, val: int):
        self._num_bits = val

    @property
    def dequantize(self) -> bool:
        return self._dequantize

    @dequantize.setter
    def dequantize(self, val: bool):
        self._dequantize = val


class PACT_Act_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, clip_val, dequantize):
        ctx.save_for_backward(input, clip_val)
        scale_factor = (2**num_bits - 1) / clip_val.data[0]
        output = torch.clamp(input, 0, clip_val.data[0])
        output = torch.floor(scale_factor * output)
        if dequantize:
            output = output / scale_factor
        # return output, scale_factor
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()

        # Gradient is zero for x <= 0 and x >= clip_val
        grad_input.masked_fill_(input.le(0), 0)
        grad_input.masked_fill_(input.ge(clip_val.data[0]), 0)

        grad_alpha = grad_output.clone()
        grad_alpha.masked_fill_(input.lt(clip_val.data[0]), 0)
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        # Straight-through estimator for the scale factor calculation
        return grad_input, None, grad_alpha, None


class PACT_Act_Signed(nn.Module, Quantizer):
    """A nn.Module implementing the PACT (PArametrized Clipping Activation)
    quantization strategy for activations.
    More details can be found at: https://openreview.net/forum?id=By5ugjyCb

    :param num_bits: quantization precision
    :type num_bits: int
    :param cout: dummy argument used to maintain same interface with other quantizers
    :type cout: None
    :param init_clip_val: input upper bound
    :type init_clip_val: float
    :param dequantize: whether the output should be fake-quantized or not
    :type dequantize: bool
    """
    def __init__(self,
                 num_bits: int,
                 cout: None = None,
                 init_clip_val_sup: float = 6.,
                 init_clip_val_inf: float = -6.,
                 dequantize: bool = True):
        super(PACT_Act_Signed, self).__init__()
        self._num_bits = num_bits
        self.clip_val_sup = nn.Parameter(torch.Tensor([init_clip_val_sup]), requires_grad=True)
        self.clip_val_inf = nn.Parameter(torch.Tensor([init_clip_val_inf]), requires_grad=True)
        self._dequantize = dequantize
        # Buffer is probably wrong choice cause we might need gradients (?)
        # self.register_buffer('s_a', torch.Tensor(1))
        # self.s_a = torch.Tensor(1)
        # self.s_a.fill_(1.)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the PACT activartion quantizer.

        Compute quantization using the PACT strategy and implements STE
        for the backward pass

        :param input: the input float activations tensor
        :type input: torch.Tensor
        :return: the output fake-quantized activations tensor
        :rtype: torch.Tensor
        """
        # input_q, s_a = PACT_Act_STE.apply(input,
        input_q = PACT_Act_Signed_STE.apply(input,
                                            self.num_bits,
                                            self.clip_val_sup,
                                            self.clip_val_inf,
                                            self.dequantize)
        # self.s_a = 1 / s_a
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
    def s_a(self) -> torch.Tensor:
        """Return the computed scale factor which depends upon self.num_bits and
        self.clip_val

        :return: the scale factor
        :rtype: torch.Tensor
        """
        return (self.clip_val_sup.data[0] - self.clip_val_inf.data[0]) / (2 ** self.num_bits - 1)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer quantization hyperparameters

        :return: a dictionary containing the optimized layer quantization hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'clip_val': self.clip_val,
            'scale_factor': self.s_a
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
                prfx + "act_quantizer", recurse):
            yield name, param

    def __repr__(self):
        msg = (
            f'{self.__class__.__name__}'
            f'(num_bits={self.num_bits}, '
            f'clip_val_sup={self.clip_val_sup}, '
            f'clip_val_inf={self.clip_val_inf}, '
            f'scale_factor={self.s_a})'
        )
        return msg

    @property
    def num_bits(self) -> int:
        return self._num_bits

    @num_bits.setter
    def num_bits(self, val: int):
        self._num_bits = val

    @property
    def dequantize(self) -> bool:
        return self._dequantize

    @dequantize.setter
    def dequantize(self, val: bool):
        self._dequantize = val


class PACT_Act_Signed_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, clip_val_sup, clip_val_inf, dequantize):
        ctx.save_for_backward(input, clip_val_sup, clip_val_inf)
        scale_factor = (2**num_bits - 1) / (clip_val_sup.data[0] - clip_val_inf.data[0])
        output = torch.clamp(input, clip_val_inf.data[0], clip_val_sup.data[0])
        output = torch.floor(scale_factor * output)
        if dequantize:
            output = output / scale_factor
        # return output, scale_factor
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val_sup, clip_val_inf = ctx.saved_tensors
        grad_input = grad_output.clone()

        # Gradient is zero for x <= clip_val_inf and x >= clip_val_sup
        grad_input.masked_fill_(input.le(clip_val_inf.data[0]), 0)
        grad_input.masked_fill_(input.ge(clip_val_sup.data[0]), 0)

        grad_alpha_sup = grad_output.clone()
        grad_alpha_sup.masked_fill_(input.lt(clip_val_sup.data[0]), 0)
        grad_alpha_sup = grad_alpha_sup.sum().expand_as(clip_val_sup)

        grad_alpha_inf = grad_output.clone()
        grad_alpha_inf.masked_fill_(input.gt(clip_val_inf.data[0]), 0)
        grad_alpha_inf = grad_alpha_inf.sum().expand_as(clip_val_inf)

        # Straight-through estimator for the scale factor calculation
        return grad_input, None, grad_alpha_sup, grad_alpha_inf, None
