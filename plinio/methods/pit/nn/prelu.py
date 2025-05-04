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
from typing import Dict, Any, Iterator, Tuple, cast
import torch
import torch.nn as nn
import torch.fx as fx
from .features_masker import PITFeaturesMasker
from plinio.graph.features_calculation import FeaturesCalculator
from .module import PITModule


class PITPReLU(nn.PReLU, PITModule):
    """A nn.Module implementing a BatchNorm1d layer optimizable with the PIT NAS tool

    Does not do much except memorizing the optimized number of features for correct export

    :param bn: the inner `torch.nn.PReLU` layer to be optimized
    :type bn: nn.PReLU
    """
    def __init__(self, pr: nn.PReLU):
        super(PITPReLU, self).__init__(
            pr.num_parameters,
            pr.init)
        with torch.no_grad():
            self.weight.copy_(pr.weight)
            #self.bias.copy_(instn.bias)

    @staticmethod
    def autoimport(n: fx.Node, mod: fx.GraphModule, fm: PITFeaturesMasker, fold_bn: bool):
        """Create a new fx.Node relative to a PITPReLU layer, starting from the fx.Node
        of a nn.InstanceNorm1d layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to a nn.PReLU layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param fm: The output features masker to use for this layer
        :type fm: PITFeaturesMasker
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != nn.PReLU:
            raise TypeError(
                f"Trying to generate PITPReLU from layer of type{type(submodule)}")
        # here, this is guaranteed
        submodule = cast(nn.PReLU, submodule)
        new_submodule = PITPReLU(submodule)
        mod.add_submodule(str(n.target), new_submodule)

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a PITPReLU layer, with a standard
        nn.PReLU layer within a fx.GraphModule

        :param n: the node to be rewritten, corresponds to a PReLU layer
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != PITPReLU:
            raise TypeError(f"Trying to export a layer of type{type(submodule)}")
        # here, this is guaranteed
        submodule = cast(PITPReLU, submodule)
        cout_mask = submodule.input_features_calculator.features_mask.bool()
        new_submodule = nn.PReLU(
            submodule.out_features_opt,
            submodule.init)
        with torch.no_grad():
            new_submodule.weight.copy_(submodule.weight[cout_mask])
        mod.add_submodule(str(n.target), new_submodule)
        return

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'num_features': self.out_features_opt,
        }

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters (masks) of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API, but PITLayers never have sub-layers
        :type recurse: bool
        :return: an iterator over the architectural parameters (masks) of this layer
        :rtype: Iterator[nn.Parameter]
        """
        yield ("", None)

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
            bin_alpha = self.input_features_calculator.features_mask
            return int(torch.sum(bin_alpha))

    @property
    def features_mask(self) -> torch.Tensor:
        """Return the binarized mask that specifies which output features (channels) are kept by
        the NAS

        :return: the binarized mask over the features axis
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            return self.input_features_calculator.features_mask.bool()

    def get_modified_vars(self) -> Dict[str, Any]:
        """Method that returns the modified vars(self) dictionary for the instance, used for
        cost computation

        :return: the modified vars(self) data structure
        :rtype: Dict[str, Any]
        """
        v = dict(vars(self))
        v['num_features'] = self.out_features_opt
        return v

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
        calc.register(self)
        self._input_features_calculator = calc
