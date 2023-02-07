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

from typing import Iterator, Tuple, List
import torch
import torch.fx as fx
import torch.nn as nn
from ..backends import Backend, backend_solver
from .quant_module import QuantModule


class Quant_List(nn.ModuleList, QuantModule):
    """A nn.Module implementing a collection of quantized layers sharing the input.
    The output is obtained as the concat of the outputs of each layer in the list.

    :param nn_list: the collection of quantized layers
    :type nn_list: nn.ModuleList
    """
    def __init__(self,
                 nn_list: List[nn.Module]):
        super(Quant_List, self).__init__(nn_list)
        # self.nn_list = nn_list

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the quantized layer collection. The layers share
        the input, while the output is obtained as the concat of the outputs
        of each layer in the list.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        out = []
        for layer in self:
            out.append(layer(input))
        # output = torch.stack(out, dim=0)
        output = torch.cat(out, dim=1)
        return output

    @staticmethod
    def export(n: fx.Node,
               mod: fx.GraphModule,
               backend: Backend):
        """Replaces a fx.Node corresponding to a Quant_List collection of layers,
        with a backend-specific quantized collection of layers within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        :param backend: the specific backend to be used
        :type backend: Backend
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != Quant_List:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")
        integer_list = backend_solver(type(submodule), backend)
        new_submodule = integer_list()  # TODO
        mod.add_submodule(str(n.target), new_submodule)

    # def summary(self) -> Dict[str, Any]:
    #     """Export a dictionary with the optimized layer hyperparameters

    #     :return: a dictionary containing the optimized layer hyperparameter values
    #     :rtype: Dict[str, Any]
    #     """
    #     return {
    #         'a_precision': self.a_precision,
    #         'w_precision': self.w_precision,
    #         'a_quantizer': self.a_quantizer.summary(),  # type: ignore
    #         'w_quantizer': self.w_quantizer.summary(),  # type: ignore
    #         'b_quantizer': self.b_quantizer.summary(),  # type: ignore
    #     }

    def named_quant_parameters(
            self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the quantization parameters of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API
        :type recurse: bool
        :return: an iterator over the architectural parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        # TODO: Check
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        for name, param in self.named_parameters(
                prfx + "nn_list", recurse):  # type: ignore
            yield name, param
