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
from typing import Dict, Any, Iterator, Tuple, Union, Optional
import torch.fx as fx
import torch.nn as nn
from .qtz import MPSBaseQtz, MPSPerLayerQtz, MPSPerChannelQtz, MPSBiasQtz
from plinio.graph.features_calculation import FeaturesCalculator


class MPSModule:
    """An abstract class representing the interface that all MPS layers should implement
    """
    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Calling init on base abstract MPSModule class")

    def update_softmax_options(
            self,
            temperature: Optional[float] = None,
            hard: Optional[bool] = None,
            gumbel: Optional[bool] = None,
            disable_sampling: Optional[bool] = None):
        """Set the flags to choose between the softmax, the hard and soft Gumbel-softmax
        and the sampling disabling of the architectural coefficients in the quantizers

        :param temperature: SoftMax temperature
        :type temperature: Optional[float]
        :param hard: Hard vs Soft sampling
        :type hard: Optional[bool]
        :param gumbel: Gumbel-softmax vs standard softmax
        :type gumbel: Optional[bool]
        :param disable_sampling: disable the sampling of the architectural coefficients in the
        forward pass
        :type disable_sampling: Optional[bool]
        """
        for k, v in vars(self).items():
            # skips input quantiers which would affect another layer
            # skips bias quantizers which do not have softmax options
            # TODO: make this field-name-independent
            if k in ['out_a_mps_quantizer', 'w_mps_quantizer'] and isinstance(v, MPSBaseQtz):
                v.update_softmax_options(temperature, hard, gumbel, disable_sampling)

    @property
    @abstractmethod
    def input_features_calculator(self) -> FeaturesCalculator:
        """Returns the `FeaturesCalculator` instance that computes the number of input features for
        this layer.

        :return: the `FeaturesCalculator` instance that computes the number of input features for
        this layer.
        :rtype: FeaturesCalculator
        """
        raise NotImplementedError("Trying to get input features on abstract MPSModule class")

    @input_features_calculator.setter
    @abstractmethod
    def input_features_calculator(self, calc: FeaturesCalculator):
        """Set the `FeaturesCalculator` instance that computes the number of input features for
        this layer.

        :param calc: the `FeaturesCalculator` instance that computes the number of input features
        for this layer
        :type calc: FeaturesCalculator
        """
        raise NotImplementedError("Trying to set input features on abstract MPSModule class")

    @staticmethod
    @abstractmethod
    def autoimport(n: fx.Node,
                   mod: fx.GraphModule,
                   out_a_mps_quantizer: MPSPerLayerQtz,
                   w_mps_quantizer: Union[MPSPerLayerQtz, MPSPerChannelQtz],
                   b_mps_quantizer: MPSBiasQtz):
        """Create a new fx.Node relative to a MPSModule layer, starting from the fx.Node
        of a nn.Module layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to the module to be converted
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param out_a_mps_quantizer: The MPS quantizer to be used for activations
        :type out_a_mps_quantizer: MPSQtzLayer
        :param w_mps_quantizer: The MPS quantizer to be used for weights (if present)
        :type w_mps_quantizer: Union[MPSQtzLayer, MPSQtzChannel]
        :param b_mps_quantizer: The MPS quantizer to be used for biases (if present)
        :type b_mps_quantizer: MPSBiasQtz
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        raise NotImplementedError("Trying to import layer using the base abstract class")

    @staticmethod
    @abstractmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a MPSModule, with a standard nn.Module layer
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
        raise NotImplementedError("Calling summary on base abstract MPSModule class")

    @abstractmethod
    def get_modified_vars(self) -> Iterator[Dict[str, Any]]:
        """Method that returns the modified vars(self) dictionary for the instance, for each
        combination of supported precision, used for cost computation

        :return: an iterator over the modified vars(self) data structures
        :rtype: Iterator[Dict[str, Any]]
        """
        raise NotImplementedError("Calling get_modified_vars on base abstract MPSModule class")

    @abstractmethod
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
        raise NotImplementedError("Calling named_nas_parameters on base abstract MPSModule class")

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        """Returns an iterator over the architectural parameters of this layer

        :param recurse: kept for uniformity with pytorch API,
        but MPSModule never have sub-layers TODO: check if true
        :type recurse: bool
        :return: an iterator over the architectural parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        for name, param in self.named_nas_parameters(recurse=recurse):
            yield param

    @property
    @abstractmethod
    def in_a_mps_quantizer(self) -> MPSPerLayerQtz:
        """Returns the `MPSQtzLayer` for input activations calculation

        :return: the `MPSQtzLayer` instance that computes mixprec quantized
        versions of the input activations
        :rtype: MPSQtzLayer
        """
        raise NotImplementedError(
                "Trying to get input activations quantizer on base abstract MPSModule class")

    @in_a_mps_quantizer.setter
    def in_a_mps_quantizer(self, qtz: MPSPerLayerQtz):
        """Set the `MPSQtzLayer` for input activations calculation

        :param qtz: the `MPSQtzLayer` instance that computes mixprec quantized
        versions of the input activations
        :type qtz: MPSQtzLayer
        """
        raise NotImplementedError(
                "Trying to set input activations quantizer on base abstract MPSModule class")
