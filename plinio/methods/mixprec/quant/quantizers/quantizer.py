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
import torch
import torch.fx as fx
import torch.nn as nn


class Quantizer:
    """An abstract class representing the interface that all Quantizer layers should implement
    """
    @abstractmethod
    def __init__(self, *args):
        raise NotImplementedError("Calling init on base abstract Quantizer class")

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Calling forward on base abstract Quantizer class")

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
        raise NotImplementedError("Trying to export layer using the base abstract class")

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer quantization hyperparameters

        :return: a dictionary containing the optimized layer quantization hyperparameter values
        :rtype: Dict[str, Any]
        """
        raise NotImplementedError("Calling summary on base abstract Quantizer class")

    @abstractmethod
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
        raise NotImplementedError("Calling arch_parameters on base abstract Quantizer class")

    def quant_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        """Returns an iterator over the quantization parameters of this layer

        :param recurse: kept for uniformity with pytorch API,
        but Quantizer never have sub-layers TODO: check if true
        :type recurse: bool
        :return: an iterator over the quantization parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        for name, param in self.named_quant_parameters(recurse=recurse):
            yield param

    @property
    @abstractmethod
    def dequantize(self) -> bool:
        raise NotImplementedError("Calling dequantize on base abstract Quantizer class")

    @dequantize.setter
    @abstractmethod
    def dequantize(self, val: bool):
        raise NotImplementedError("Calling dequantize on base abstract Quantizer class")
