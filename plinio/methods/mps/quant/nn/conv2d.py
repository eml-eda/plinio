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

from typing import Dict, Any, Optional, Iterator, Tuple, cast
import torch
import torch.fx as fx
import torch.nn as nn
from ..quantizers import Quantizer
from ..backends import Backend, backend_factory
from .module import QuantModule


class QuantConv2d(nn.Conv2d, QuantModule):
    """A nn.Module implementing a quantized Conv2d layer

    :param conv: the inner `nn.Conv2d` layer to be optimized
    :type conv: nn.Conv2d
    :param in_quantizer: input activation quantizer
    :type in_quantizer: Quantizer
    :param out_quantizer: output activation quantizer
    :type out_quantizer: Quantizer
    :param w_quantizer: weight quantizer
    :type w_quantizer: Quantizer
    :param b_quantizer: bias quantizer
    :type b_quantizer: Optional[Quantizer]
    """
    def __init__(self,
                 conv: nn.Conv2d,
                 in_quantizer: Quantizer,
                 out_quantizer: Quantizer,
                 w_quantizer: Quantizer,
                 b_quantizer: Optional[Quantizer]):
        super(QuantConv2d, self).__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode)
        with torch.no_grad():
            self.weight.copy_(conv.weight)
            if conv.bias is not None:
                self.bias = cast(torch.nn.parameter.Parameter, self.bias)
                self.bias.copy_(conv.bias)
            else:
                self.bias = None
        self.in_quantizer = in_quantizer
        self.out_quantizer = out_quantizer
        self.w_quantizer = w_quantizer
        if conv.bias is not None:
            self.b_quantizer = cast(Quantizer, b_quantizer)
        else:
            self.b_quantizer = lambda *args: None  # Do Nothing

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of linear conv2d layer.

        It performs:
        - Quantization of the `self.weight` tensor using `self.w_quantizer`.
        - Quantization of the `self.bias` vector using `self.b_quantizer` (if needed).
        - Computation of conv2d operation.
        - Quantization of the input tensor using `self.out_quantizer`.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        # Quantization of weight and bias
        q_w = self.w_quantizer(self.weight)
        q_b = self.b_quantizer(self.bias,
                               self.in_quantizer.scale, self.w_quantizer.scale)
        # Linear operation
        out = self._conv_forward(input, q_w, q_b)

        # Quantization of output
        q_out = self.out_quantizer(out)

        return q_out

    @staticmethod
    def autoimport():
        raise NotImplementedError

    @staticmethod
    def export(n: fx.Node,
               mod: fx.GraphModule,
               backend: Backend,
               backend_kwargs: Dict = {}):
        """Replaces a fx.Node corresponding to a Quant_Conv2d layer,
        with a backend-specific quantized Conv2d layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        :param backend: the specific backend to be used
        :type backend: Backend
        :param backend_kwargs: additional backend-specific arguments
        :type backend_kwargs: Dict
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != QuantConv2d:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")
        integer_conv = backend_factory(submodule, backend)
        new_submodule = integer_conv(
            submodule,
            submodule.in_quantizer,
            submodule.out_quantizer,
            submodule.w_quantizer,
            submodule.b_quantizer,
            **backend_kwargs)
        mod.add_submodule(str(n.target), new_submodule)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'in_quantizer': self.in_quantizer.summary(),
            'out_quantizer': self.out_quantizer.summary(),
            'w_quantizer': self.w_quantizer.summary(),
            'b_quantizer': self.b_quantizer.summary(),
        }

    def named_quant_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the quantization parameters of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: recurse to sub-modules
        :type recurse: bool
        :return: an iterator over the architectural parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        # Skips input quantizer, which is owned by another Module
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        for name, param in self.out_quantizer.named_quant_parameters(
                prfx + "out_quantizer", recurse):
            yield name, param
        for name, param in self.w_quantizer.named_quant_parameters(
                prfx + "w_quantizer", recurse):
            yield name, param
        if self.bias is not None:
            for name, param in self.b_quantizer.named_quant_parameters(
                    prfx + "b_quantizer", recurse):
                yield name, param
