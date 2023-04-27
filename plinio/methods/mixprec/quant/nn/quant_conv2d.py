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

from typing import Dict, Any, Optional, Iterator, Tuple, cast, Type
import torch
import torch.fx as fx
import torch.nn as nn
from ..quantizers import Quantizer
from ..backends import Backend, backend_solver
from .quant_module import QuantModule


class Quant_Conv2d(nn.Conv2d, QuantModule):
    """A nn.Module implementing a quantized Conv2d layer

    :param conv: the inner `nn.Conv2d` layer to be optimized
    :type conv: nn.Conv2d
    :param a_precision: the input activation quantization precision
    :type a_precision: int
    :param w_precision: the weights' quantization precision
    :type w_precision: int
    :param a_quantizer: activation quantizer
    :type a_quantizer: Type[Quantizer]
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
        super(Quant_Conv2d, self).__init__(
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of linear conv2d layer.

        It performs:
        - Quantization of the `self.weight` tensor using `self.w_quantizer`.
        - Quantization of the `self.bias` vector using `self.b_quantizer` (if needed).
        - Computation of conv2d operation.
        - Quantization of the input tensor using `self.a_quantizer`.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        # Quantization of weight and bias
        q_w = self.w_quantizer(self.weight)  # type: ignore
        q_w = cast(torch.Tensor, q_w)
        q_b = self.b_quantizer(self.bias,  # type: ignore
                               self.a_quantizer.s_a, self.w_quantizer.s_w)  # type: ignore
        q_b = cast(torch.Tensor, q_b)

        # Linear operation
        out = self._conv_forward(input, q_w, q_b)

        # Quantization of output
        q_out = self.out_a_quantizer(out)  # type: ignore
        q_out = cast(torch.Tensor, q_out)

        return q_out

    # TODO: this function needs to be checked, currently the usage of this class
    # is properly implemented only when converting from mixprec model
    @staticmethod
    def autoimport(n: fx.Node,
                   mod: fx.GraphModule,
                   a_precision: int,
                   w_precision: int,
                   a_quantizer: Quantizer,
                   w_quantizer: Quantizer,
                   b_quantizer: Quantizer,
                   a_sq: Optional[Quantizer],
                   w_sq: Optional[Quantizer],
                   b_sq: Optional[Quantizer],
                   a_quantizer_kwargs: Dict = {},
                   w_quantizer_kwargs: Dict = {},
                   b_quantizer_kwargs: Dict = {}
                   ) -> Optional[Quantizer]:
        """Create a new fx.Node relative to a Quant_Conv2d layer, starting from the fx.Node
        of a nn.Conv2d layer, and replace it into the parent fx.GraphModule

        Also returns a quantizer in case it needs to be shared with other layers

        :param n: a fx.Node corresponding to a nn.ReLU layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param a_precision: the input activation quantization precision
        :type a_precision: int
        :param w_precision: the weight quantization precision
        :type w_precision: int
        :param a_quantizer: the quantizer to be used for input activations
        :type a_quantizer: Type[Quantizer]
        :param w_quantizer: the quantizer to be used for weights
        :type w_quantizer: Type[Quantizer]
        :param b_quantizer: the quantizer to be used for biases
        :type b_quantizer: Type[Quantizer]
        :param a_sq: An optional shared quantizer derived from other layers
        for input activations
        :type a_sq: Optional[Quantizer]
        :param w_sq: An optional shared quantizer derived from other layers
        for weights
        :type w_sq: Optional[Quantizer]
        :param b_sq: An optional shared quantizer derived from other layers
        for biases
        :type b_sq: Optional[Quantizer]
        :param a_quantizer_kwargs: Activations' quantizer kwargs,
        if no kwargs are passed default is used
        :type a_quantizer_kwargs: Dict
        :param w_quantizer_kwargs: Weights' quantizer kwargs,
        if no kwargs are passed default is used
        :type w_quantizer_kwargs: Dict
        :param b_quantizer_kwargs: Biases' quantizer kwargs,
        if no kwargs are passed default is used
        :type b_quantizer_kwargs: Dict
        :raises TypeError: if the input fx.Node is not of the correct type
        :return: the updated shared quantizer
        :rtype: Optional[Quantizer]
        """
        raise NotImplementedError
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != nn.Conv2d:
            msg = f"Trying to generate Quant_Conv2d from layer of type {type(submodule)}"
            raise TypeError(msg)
        # here, this is guaranteed
        submodule = cast(nn.Conv2d, submodule)

        # Build activation quantizer
        if a_sq is not None:
            a_qtz_obj = a_sq
        else:
            a_qtz_obj = a_quantizer(a_precision,
                                    **a_quantizer_kwargs)  # type: ignore
        a_qtz_obj = cast(Type[Quantizer], a_qtz_obj)

        # Build weight quantizer
        if w_sq is not None:
            w_qtz_obj = w_sq
        else:
            w_qtz_obj = w_quantizer(w_precision,
                                    **w_quantizer_kwargs)  # type: ignore
        w_qtz_obj = cast(Type[Quantizer], w_qtz_obj)

        # Build bias quantizer
        if b_sq is not None:
            b_qtz_obj = b_sq
        else:
            b_qtz_obj = b_quantizer(w_precision,
                                    **b_quantizer_kwargs)  # type: ignore
        b_qtz_obj = cast(Type[Quantizer], b_qtz_obj)

        new_submodule = Quant_Conv2d(submodule,
                                     a_precision,
                                     w_precision,
                                     a_qtz_obj,
                                     w_qtz_obj,
                                     b_qtz_obj)
        mod.add_submodule(str(n.target), new_submodule)
        return None  # TODO: Understand if I should return something and when

    @staticmethod
    def export(n: fx.Node,
               mod: fx.GraphModule,
               backend: Backend):
        """Replaces a fx.Node corresponding to a Quant_Conv2d layer,
        with a backend-specific quantized Conv2d layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        :param backend: the specific backend to be used
        :type backend: Backend
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != Quant_Conv2d:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")
        integer_conv = backend_solver(submodule, backend)
        new_submodule = integer_conv(
            submodule.a_precision,
            submodule.w_precision,
            submodule.in_a_quantizer,
            submodule.out_a_quantizer,
            submodule.w_quantizer,
            submodule.b_quantizer)
        mod.add_submodule(str(n.target), new_submodule)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'a_precision': self.a_precision,
            'w_precision': self.w_precision,
            'in_a_quantizer': self.in_a_quantizer.summary(),  # type: ignore
            'out_a_quantizer': self.out_a_quantizer.summary(),  # type: ignore
            'w_quantizer': self.w_quantizer.summary(),  # type: ignore
            'b_quantizer': self.b_quantizer.summary(),  # type: ignore
        }

    def named_quant_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the quantization parameters of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API,
        but QuantModule never have sub-layers TODO: check if true
        :type recurse: bool
        :return: an iterator over the architectural parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        for name, param in self.out_a_quantizer.named_quant_parameters(
                prfx + "out_a_quantizer", recurse):  # type: ignore
            yield name, param
        for name, param in self.w_quantizer.named_quant_parameters(
                prfx + "w_quantizer", recurse):  # type: ignore
            yield name, param
        if self.bias is not None:
            for name, param in self.b_quantizer.named_quant_parameters(
                    prfx + "b_quantizer", recurse):  # type: ignore
                yield name, param
