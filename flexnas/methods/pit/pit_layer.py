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
from typing import Dict, Any, Optional
import torch.fx as fx
from .pit_channel_masker import PITChannelMasker
from flexnas.utils.features_calculator import FeaturesCalculator


class PITLayer:
    """An abstract class representing the interface that all PIT layers should implement
    """
    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Calling init on base abstract PITLayer class")

    @property
    @abstractmethod
    def input_features_calculator(self) -> FeaturesCalculator:
        """Returns the `FeaturesCalculator` instance that computes the number of input features for
        this layer.

        :return: the `FeaturesCalculator` instance that computes the number of input features for
        this layer.
        :rtype: FeaturesCalculator
        """
        raise NotImplementedError("Trying to get input features on base abstract PITLayer class")

    @input_features_calculator.setter
    @abstractmethod
    def input_features_calculator(self, calc: FeaturesCalculator):
        """Set the `FeaturesCalculator` instance that computes the number of input features for
        this layer.

        :param calc: the `FeaturesCalculator` instance that computes the number of input features
        for this layer
        :type calc: FeaturesCalculator
        """
        raise NotImplementedError("Trying to set input features on base abstract PITLayer class")

    @staticmethod
    @abstractmethod
    def autoimport(n: fx.Node, mod: fx.GraphModule, sm: Optional[PITChannelMasker]):
        """Create a new fx.Node relative to a PITLayer layer, starting from the fx.Node
        of a nn.Module layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to a standard nn.Module layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param sm: An optional shared output channel masker derived from subsequent layers
        :type sm: Optional[PITChannelMasker]
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        raise NotImplementedError("Trying to import layer using the base abstract class")

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a PITLayer, with a standard nn.Module layer
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
        raise NotImplementedError("Calling summary on base abstract PITLayer class")
