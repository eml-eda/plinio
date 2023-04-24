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
from .dory_module import DORYModule


class DORYIdentity(nn.Identity, DORYModule):
    # TODO
    """A nn.Module implementing a quantized Identity layer

    :param precision: the quantization precision
    :type precisions: int
    :param quantizer: input tensor quantizer
    :type quantizer: Type[Quantizer]
    """
    def __init__(self,
                 precision: int,
                 quantizer: Quantizer):
        super(Quant_Identity, self).__init__()
        self.precision = precision
        self.quantizer = quantizer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of quantized layer.

        It simply quantize the input tensor using a specific `Quantizer`.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        out = self.quantizer(input)  # type: ignore
        out = cast(torch.Tensor, out)
        return out

    @staticmethod
    def autoimport(n: fx.Node,
                   mod: fx.GraphModule,
                   precision: int,
                   quantizer: Quantizer,
                   sq: Optional[Quantizer],
                   quantizer_kwargs: Dict = {}
                   ) -> Optional[Quantizer]:
        """Create a new fx.Node relative to a Quant_Identity layer, starting from the fx.Node
        of a nn.Identity layer, and replace it into the parent fx.GraphModule

        Also returns a quantizer in case it needs to be shared with other layers

        :param n: a fx.Node corresponding to a nn.Identity layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param precision: the quantization precision
        :type precision: int
        :param quantizer: the quantizer to be used
        :type quantizer: Type[Quantizer]
        :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
        :type quantizer_kwargs: Dict
        :param sq: An optional shared quantizer derived from other layers
        :type sq: Optional[Quantizer]
        :raises TypeError: if the input fx.Node is not of the correct type
        :return: the updated shared quantizer
        :rtype: Optional[Quantizer]
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != nn.Identity:
            msg = f"Trying to generate Quant_Identity from layer of type {type(submodule)}"
            raise TypeError(msg)
        if sq is not None:
            qtz_obj = sq
        else:
            qtz_obj = quantizer(precision,
                                **quantizer_kwargs)  # type: ignore

        new_submodule = Quant_Identity(precision, qtz_obj)
        mod.add_submodule(str(n.target), new_submodule)
        return None  # TODO: Understand if I should return something and when

    @staticmethod
    def export(n: fx.Node,
               mod: fx.GraphModule,
               backend: Backend):
        """Replaces a fx.Node corresponding to a Quant_Identity layer,
        with a backend-specific quantize Identity layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        :param backend: the specific backend to be used
        :type backend: Backend
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != Quant_Identity:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")
        integer_identity = backend_solver(type(submodule), backend)
        new_submodule = integer_identity(
            submodule.precision,
            submodule.quantizer
        )
        mod.add_submodule(str(n.target), new_submodule)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'precision': self.precision,
            'quantizer': self.quantizer.summary(),
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
        for name, param in self.quantizer.named_quant_parameters(
                prfx + "mixprec_quantizer", recurse):
            yield name, param
