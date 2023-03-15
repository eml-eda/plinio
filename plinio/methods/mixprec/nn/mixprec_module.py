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
from typing import Dict, Any, Optional, Iterator, Tuple, Type
import torch.fx as fx
import torch.nn as nn
from ..quant.quantizers import Quantizer
from .mixprec_qtz import MixPrecType
from plinio.graph.features_calculation import FeaturesCalculator


class MixPrecModule:
    """An abstract class representing the interface that all MixPrec layers should implement
    """
    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Calling init on base abstract MixPrecModule class")

    @property
    @abstractmethod
    def input_features_calculator(self) -> FeaturesCalculator:
        """Returns the `FeaturesCalculator` instance that computes the number of input features for
        this layer.

        :return: the `FeaturesCalculator` instance that computes the number of input features for
        this layer.
        :rtype: FeaturesCalculator
        """
        raise NotImplementedError("Trying to get input features on base abstract PITModule class")

    @input_features_calculator.setter
    @abstractmethod
    def input_features_calculator(self, calc: FeaturesCalculator):
        """Set the `FeaturesCalculator` instance that computes the number of input features for
        this layer.

        :param calc: the `FeaturesCalculator` instance that computes the number of input features
        for this layer
        :type calc: FeaturesCalculator
        """
        raise NotImplementedError("Trying to set input features on base abstract PITModule class")

    @staticmethod
    @abstractmethod
    def autoimport(n: fx.Node,
                   mod: fx.GraphModule,
                   w_mixprec_type: MixPrecType,
                   a_precisions: Tuple[int, ...],
                   w_precisions: Tuple[int, ...],
                   a_quantizer: Type[Quantizer],
                   w_quantizer: Type[Quantizer],
                   b_quantizer: Type[Quantizer],
                   a_sq: Optional[Quantizer],
                   a_quantizer_kwargs: Dict = {},
                   w_quantizer_kwargs: Dict = {},
                   b_quantizer_kwargs: Dict = {}
                   ) -> Optional[Quantizer]:
        """Create a new fx.Node relative to a MixPrecModule layer, starting from the fx.Node
        of a nn.Module layer, and replace it into the parent fx.GraphModule

        Also returns a quantizer in case it needs to be shared with other layers

        :param n: a fx.Node corresponding to a nn.ReLU layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param w_mixprec_type: the mixed precision strategy to be used for weigth
        i.e., `PER_CHANNEL` or `PER_LAYER`.
        :type w_mixprec_type: MixPrecType
        :param a_precisions: The precisions to be explored for activations
        :type a_precisions: Tuple[int, ...]
        :param w_precisions: The precisions to be explored for weights
        :type w_precisions: Tuple[int, ...]
        :param a_quantizer: The quantizer to be used for activations
        :type a_quantizer: Type[Quantizer]
        :param w_quantizer: The quantizer to be used for weights
        :type w_quantizer: Type[Quantizer]
        :param b_quantizer: The quantizer to be used for biases
        :type b_quantizer: Type[Quantizer]
        :param a_sq: An optional shared quantizer derived from other layers for activations
        :type a_sq: Optional[Quantizer]
        :param a_quantizer_kwargs: act quantizer kwargs, if no kwargs are passed default is used
        :type a_quantizer_kwargs: Dict
        :param w_quantizer_kwargs: weight quantizer kwargs, if no kwargs are passed default is used
        :type w_quantizer_kwargs: Dict
        :param b_quantizer_kwargs: bias quantizer kwargs, if no kwargs are passed default is used
        :type b_quantizer_kwargs: Dict
        :raises TypeError: if the input fx.Node is not of the correct type
        :return: the updated shared quantizer
        :rtype: Optional[Quantizer]
        """
        raise NotImplementedError("Trying to import layer using the base abstract class")

    @staticmethod
    @abstractmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a MixPrecModule, with a standard nn.Module layer
        within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        raise NotImplementedError("Trying to export layer using the base abstract class")

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        raise NotImplementedError("Calling summary on base abstract MixPrecModule class")

    @abstractmethod
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
        raise NotImplementedError("Calling arch_parameters on base abstract MixPrecModule class")

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        """Returns an iterator over the architectural parameters of this layer

        :param recurse: kept for uniformity with pytorch API,
        but MixPrecModule never have sub-layers TODO: check if true
        :type recurse: bool
        :return: an iterator over the architectural parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        for name, param in self.named_nas_parameters(recurse=recurse):
            yield param
