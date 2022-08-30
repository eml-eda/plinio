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
from typing import Dict, Any, Optional, cast
import torch
import torch.nn as nn
import torch.fx as fx
from .pit_features_masker import PITFeaturesMasker
from flexnas.utils.features_calculator import FeaturesCalculator
from .pit_layer import PITLayer


class PITBatchNorm1d(nn.BatchNorm1d, PITLayer):
    """A nn.Module implementing a BatchNorm1d layer optimizable with the PIT NAS tool

    Does not do much except memorizing the optimized number of features for correct export

    :param bn: the inner `torch.nn.BatchNorm1d` layer to be optimized
    :type bn: nn.BatchNorm1d
    """
    def __init__(self, bn: nn.BatchNorm1d):
        super(PITBatchNorm1d, self).__init__(
            bn.num_features,
            bn.eps,
            bn.momentum,
            bn.affine,
            bn.track_running_stats)
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var
        self.weight = bn.weight
        self.bias = bn.bias

    @staticmethod
    def autoimport(n: fx.Node, mod: fx.GraphModule, sm: Optional[PITFeaturesMasker]):
        """Create a new fx.Node relative to a PITBatchNorm1d layer, starting from the fx.Node
        of a nn.BatchNorm1d layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to a nn.BatchNorm1d layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param sm: An optional shared output channel masker derived from subsequent layers
        :type sm: Optional[PITChannelMasker]
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != nn.BatchNorm1d:
            raise TypeError(
                f"Trying to generate PITBatchNorm1d from layer of type{type(submodule)}")
        # here, this is guaranteed
        submodule = cast(nn.BatchNorm1d, submodule)
        new_submodule = PITBatchNorm1d(submodule)
        mod.add_submodule(str(n.target), new_submodule)
        return

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a PITBatchNorm1D layer, with a standard
        nn.BatchNorm1d layer within a fx.GraphModule

        :param n: the node to be rewritten, corresponds to a BatchNorm1d layer
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != PITBatchNorm1d:
            raise TypeError(f"Trying to export a layer of type{type(submodule)}")
        # here, this is guaranteed
        submodule = cast(PITBatchNorm1d, submodule)
        cout_mask = submodule.input_features_calculator.features_mask.bool()
        new_submodule = nn.BatchNorm1d(
            submodule.out_features_opt,
            submodule.eps,
            submodule.momentum,
            submodule.affine,
            submodule.track_running_stats)
        new_submodule.weight = nn.parameter.Parameter(submodule.weight[cout_mask])
        new_submodule.bias = nn.parameter.Parameter(submodule.bias[cout_mask])
        new_submodule.running_mean = submodule.running_mean
        new_submodule.running_var = submodule.running_var
        mod.add_submodule(str(n.target), new_submodule)
        return

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'in_features': self.in_features_opt,
            'out_features': self.out_features_opt,
        }

    @property
    def out_features_opt(self) -> int:
        """Get the number of output features found during the search

        :return: the number of output features found during the search
        :rtype: int
        """
        return self.in_features_opt

    @property
    def in_features_opt(self) -> int:
        """Get the number of input features found during the search

        :return: the number of input features found during the search
        :rtype: int
        """
        with torch.no_grad():
            return int(self.input_features_calculator.features)

    @property
    def features_mask(self) -> torch.Tensor:
        """Return the binarized mask that specifies which output features (channels) are kept by
        the NAS

        :return: the binarized mask over the features axis
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            return self.input_features_calculator.features_mask.bool()

    def get_size(self) -> torch.Tensor:
        """Method that computes the number of weights for the layer

        :return: the number of weights
        :rtype: torch.Tensor
        """
        return 2 * self.get_macs()

    def get_macs(self) -> torch.Tensor:
        """Method that computes the number of MAC operations for the layer

        :return: the number of MACs
        :rtype: torch.Tensor
        """
        return self.input_features_calculator.features

    @property
    def input_features_calculator(self) -> FeaturesCalculator:
        """Returns the `FeaturesCalculator` instance that computes the number of input features for
        this layer.

        :return: the `FeaturesCalculator` instance that computes the number of input features for
        this layer.
        :rtype: FeaturesCalculator
        """
        return self._input_features_calculator

    @input_features_calculator.setter
    def input_features_calculator(self, calc: FeaturesCalculator):
        """Set the `FeaturesCalculator` instance that computes the number of input features for
        this layer.

        :param calc: the `FeaturesCalculator` instance that computes the number of input features
        for this layer
        :type calc: FeaturesCalculator
        """
        self._input_features_calculator = calc
