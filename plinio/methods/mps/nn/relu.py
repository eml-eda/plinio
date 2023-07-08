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

from typing import Dict, Any, Iterator, Tuple, Type, cast, Union
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from ..quant.quantizers import Quantizer
from ..quant.nn import Quant_ReLU
from .module import MPSModule
from .qtz import MPSQtzLayer, MPSQtzChannel, MPSQtzLayerBias, MPSQtzChannelBias


class MPSReLU(nn.ReLU, MPSModule):
    """A nn.Module implementing a ReLU layer with mixed-precision search support

    :param a_mps_quantizer: activation MPS quantizer
    :type a_mps_quantizer: MPSQtzLayer
    """
    def __init__(self,
                 a_mps_quantizer: MPSQtzLayer):
        super(MPSReLU, self).__init__()
        self.a_mps_quantizer = a_mps_quantizer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the mixed-precision NAS-able layer.

        In a nutshell, quantize and combine the input tensor at the different
        `precisions`.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        q_out = self.a_mps_quantizer(input)
        out = F.relu(q_out)
        return out

    @staticmethod
    def autoimport(n: fx.Node,
                   mod: fx.GraphModule,
                   a_mps_quantizer: MPSQtzLayer,
                   w_mps_quantizer: Union[MPSQtzLayer, MPSQtzChannel],
                   b_mps_quantizer: Union[MPSQtzLayerBias, MPSQtzChannelBias],
                   ):
        """Create a new fx.Node relative to a MPSReLU layer, starting from the fx.Node
        of a nn.ReLU layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to a nn.ReLU layer
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param a_mps_quantizer: The MPS quantizer to be used for activations
        :type a_mps_quantizer: MPSQtzLayer
        :param w_mps_quantizer: The MPS quantizer to be used for weights (ignored for ReLU)
        :type w_mps_quantizer: Union[MPSQtzLayer, MPSQtzChannel]
        :param b_mps_quantizer: The MPS quantizer to be used for biases (ignored for ReLU)
        :type b_mps_quantizer: Union[MPSQtzLayerBias, MPSQtzChannelBias]
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != nn.ReLU:
            msg = f"Trying to generate MPSReLU from layer of type {type(submodule)}"
            raise TypeError(msg)
        new_submodule = MPSReLU(a_mps_quantizer)
        mod.add_submodule(str(n.target), new_submodule)

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a MPSReLU layer,
        with the selected fake-quantized nn.ReLU layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != MPSReLU:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")
        selected_precision = submodule.selected_precision
        selected_precision = cast(int, selected_precision)
        selected_quantizer = submodule.selected_quantizer
        selected_quantizer = cast(Quantizer, selected_quantizer)
        new_submodule = Quant_ReLU(
            selected_precision,
            selected_quantizer
        )
        mod.add_submodule(str(n.target), new_submodule)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'precision': self.selected_precision,
        }

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API,
        but MPSModule never have sub-layers TODO: check if true
        :type recurse: bool
        :return: an iterator over the architectural parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        for name, param in self.a_mps_quantizer.named_parameters(
                prfx + "mixprec_quantizer", recurse):
            yield name, param

    @property
    def selected_precision(self) -> int:
        """Return the selected precision based on the magnitude of `alpha_prec`
        components

        :return: the selected precision
        :rtype: int
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.a_mps_quantizer.alpha_prec))
            return self.a_mps_quantizer.precisions[idx]

    @property
    def selected_quantizer(self) -> Type[Quantizer]:
        """Return the selected quantizer based on the magnitude of `alpha_prec`
        components

        :return: the selected precision
        :rtype: int
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.a_mps_quantizer.alpha_prec))
            qtz = self.a_mps_quantizer.mix_qtz[idx]
            qtz = cast(Type[Quantizer], qtz)
            return qtz

    @property
    def input_quantizer(self) -> MPSQtzLayer:
        """Returns the `MPSQtzLayer` for input activations calculation

        :return: the `MPSQtzLayer` instance that computes mixprec quantized
        versions of the input activations
        :rtype: MPSQtzLayer
        """
        return self._input_quantizer

    @input_quantizer.setter
    def input_quantizer(self, qtz: MPSQtzLayer):
        """Set the `MPSQtzLayer` for input activations calculation

        :param qtz: the `MPSQtzLayer` instance that computes mixprec quantized
        versions of the input activations
        :type qtz: MPSQtzLayer
        """
        self._input_quantizer = qtz
