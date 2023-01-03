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

from abc import abstractmethod
from typing import Dict, Any, Optional, Iterator, Tuple
import torch.fx as fx
import torch.nn as nn
from ..quantizers import Quantizer
from ..backends import Backend


class QuantModule:
    """An abstract class representing the interface that all quantized layers should implement
    """
    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Calling init on base abstract QuantModule class")

    @staticmethod
    @abstractmethod
    def autoimport(n: fx.Node,
                   mod: fx.GraphModule,
                   precision: int,
                   quantizer: Quantizer,
                   sq: Optional[Quantizer],
                   quantizer_kwargs: Dict = {}
                   ) -> Optional[Quantizer]:
        """Create a new fx.Node relative to a QuantModule layer, starting from the fx.Node
        of a nn.Module layer, and replace it into the parent fx.GraphModule

        Also returns a quantizer in case it needs to be shared with other layers

        :param n: a fx.Node corresponding to a standard nn.Module layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param precision: The precision to be enforced
        :type precision: int
        :param quantizer: The quantizer to be used
        :type quantizer: Quantizer
        :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
        :type quantizer_kwargs: Dict
        :param sq: An optional shared quantizer derived from other layers
        :type sq: Optional[Quantizer]
        :raises TypeError: if the input fx.Node is not of the correct type
        :return: the updated shared quantizer
        :rtype: Optional[Quantizer]
        """
        raise NotImplementedError("Trying to import layer using the base abstract class")

    @staticmethod
    @abstractmethod
    def export(n: fx.Node,
               mod: fx.GraphModule,
               backend: Backend):
        """Replaces a fx.Node corresponding to a QuantModule, with a backend-specific
        layer implementation within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        :param backend: the specific backend to be used
        :type backend: Backend
        """
        raise NotImplementedError("Trying to export layer using the base abstract class")

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        raise NotImplementedError("Calling summary on base abstract QuantModule class")

    @abstractmethod
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
        raise NotImplementedError("Calling arch_parameters on base abstract QuantModule class")

    def quant_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        """Returns an iterator over the quantization parameters of this layer

        :param recurse: kept for uniformity with pytorch API,
        but QuantModule never have sub-layers TODO: check if true
        :type recurse: bool
        :return: an iterator over the architectural parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        for name, param in self.named_quant_parameters(recurse=recurse):
            yield param
