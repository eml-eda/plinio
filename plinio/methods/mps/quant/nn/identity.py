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
from ..quantizers import Quantizer
from ..backends import Backend, backend_factory
from .module import QuantModule


class QuantIdentity(nn.Identity, QuantModule):
    """A nn.Module implementing a quantized Identity layer

    :param quantizer: output activations quantizer
    :type quantizer: Type[Quantizer]
    """
    def __init__(self,
                 quantizer: Quantizer):
        super(QuantIdentity, self).__init__()
        self.out_a_quantizer = quantizer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of quantized layer.

        It simply quantize the input tensor using a specific `Quantizer`.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        out = self.out_a_quantizer(input)
        return out

    # TODO: this function needs to be implemented, currently instances of this class
    # are only created when converting from a MPS model
    @staticmethod
    def autoimport() -> Optional[Quantizer]:
        """ TODO: implement """
        raise NotImplementedError

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
        if type(submodule) != QuantIdentity:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")
        integer_identity = backend_factory(submodule, backend)
        new_submodule = integer_identity(
            submodule.out_a_quantizer
        )
        mod.add_submodule(str(n.target), new_submodule)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'quantizer': self.out_a_quantizer.summary(),
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
                prfx + "out_a_quantizer", recurse):
            yield name, param
