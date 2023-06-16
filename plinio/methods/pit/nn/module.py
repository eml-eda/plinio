# *----------------------------------------------------------------------------*
# * Copyright (C) 2021 Politecnico di Torino, Italy                            *
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
# * Author:  Daniele Jahier Pagliari <daniele.jahier@polito.it>                *
# *----------------------------------------------------------------------------*
from abc import abstractmethod
from typing import Dict, Any, Iterator, Tuple
import torch.fx as fx
import torch.nn as nn
from .features_masker import PITFeaturesMasker
from plinio.graph.features_calculation import FeaturesCalculator


class PITModule:
    """An abstract class representing the interface that all PIT layers should implement
    """
    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Calling init on base abstract PITModule class")

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
    def autoimport(n: fx.Node, mod: fx.GraphModule, fm: PITFeaturesMasker):
        """Create a new fx.Node relative to a PITModule layer, starting from the fx.Node
        of a nn.Module layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to a nn.Module layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param fm: the output features masker to use for this layer
        :type fm: PITFeaturesMasker
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        raise NotImplementedError("Trying to import layer using the base abstract class")

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a PITModule, with a standard nn.Module layer
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
        raise NotImplementedError("Calling summary on base abstract PITModule class")

    @abstractmethod
    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters (masks) of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API, but PITModules never have sub-layers
        :type recurse: bool
        :return: an iterator over the architectural parameters (masks) of this layer
        :rtype: Iterator[nn.Parameter]
        """
        raise NotImplementedError("Calling arch_parameters on base abstract PITModule class")

    @abstractmethod
    def get_modified_vars(self) -> Dict[str, Any]:
        """Method that returns the modified vars(self) dictionary for the instance, used for
        cost computation

        :return: the modified vars(self) data structure
        :rtype: Dict[str, Any]
        """
        raise NotImplementedError("Calling get_modified_vars on base abstract PITModule class")

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        """Returns an iterator over the architectural parameters (masks) of this layer

        :param recurse: kept for uniformity with pytorch API, but PITModules never have sub-layers
        :type recurse: bool
        :return: an iterator over the architectural parameters (masks) of this layer
        :rtype: Iterator[nn.Parameter]
        """
        for name, param in self.named_nas_parameters(recurse=recurse):
            yield param
