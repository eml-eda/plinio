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

from typing import Dict, Any, Optional, Iterator, Tuple, Type, cast
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from ..quant.quantizers import Quantizer
from ..quant.nn import Quant_ReLU
from .mixprec_module import MixPrecModule
from .mixprec_qtz import MixPrec_Qtz_Layer


class MixPrec_ReLU(nn.ReLU, MixPrecModule):
    """A nn.Module implementing a ReLU layer with mixed-precision search support

    :param precisions: different bitwitdth alternatives among which perform search
    :type precisions: Tuple[int, ...]
    :param quantizer: input tensor quantizer
    :type quantizer: MixPrec_Qtz_Layer
    """
    def __init__(self,
                 precisions: Tuple[int, ...],
                 quantizer: MixPrec_Qtz_Layer):
        super(MixPrec_ReLU, self).__init__()
        self.precisions = precisions
        self.mixprec_quantizer = quantizer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the mixed-precision NAS-able layer.

        In a nutshell, quantize and combine the input tensor at the different
        `precisions`.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        q_out = self.mixprec_quantizer(input)
        out = F.relu(q_out)
        return out

    @staticmethod
    def autoimport(n: fx.Node,
                   mod: fx.GraphModule,
                   precisions: Tuple[int, ...],
                   quantizer: Type[Quantizer],
                   sq: Optional[Quantizer],
                   quantizer_kwargs: Dict = {}
                   ) -> Optional[Quantizer]:
        """Create a new fx.Node relative to a MixPrec_ReLU layer, starting from the fx.Node
        of a nn.ReLU layer, and replace it into the parent fx.GraphModule

        Also returns a quantizer in case it needs to be shared with other layers

        :param n: a fx.Node corresponding to a nn.ReLU layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param precisions: The precisions to be explored
        :type precisions: Tuple[int, ...]
        :param quantizer: The quantizer to be used
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
        if type(submodule) != nn.ReLU:
            msg = f"Trying to generate MixPrec_ReLU from layer of type {type(submodule)}"
            raise TypeError(msg)
        if sq is not None:
            mixprec_quantizer = sq
        else:
            mixprec_quantizer = MixPrec_Qtz_Layer(precisions,
                                                  quantizer,
                                                  quantizer_kwargs)

        mixprec_quantizer = cast(MixPrec_Qtz_Layer, mixprec_quantizer)
        new_submodule = MixPrec_ReLU(precisions, mixprec_quantizer)
        mod.add_submodule(str(n.target), new_submodule)
        return None  # TODO: Understand if I should return something and when

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a MixPrec_ReLU layer,
        with the selected fake-quantized nn.ReLU layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != MixPrec_ReLU:
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
        but MixPrecModule never have sub-layers TODO: check if true
        :type recurse: bool
        :return: an iterator over the architectural parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        for name, param in self.mixprec_quantizer.named_parameters(
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
            idx = int(torch.argmax(self.mixprec_quantizer.alpha_prec))
            return self.precisions[idx]

    @property
    def selected_quantizer(self) -> Type[Quantizer]:
        """Return the selected quantizer based on the magnitude of `alpha_prec`
        components

        :return: the selected precision
        :rtype: int
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.mixprec_quantizer.alpha_prec))
            qtz = self.mixprec_quantizer.mix_qtz[idx]
            qtz = cast(Type[Quantizer], qtz)
            return qtz
