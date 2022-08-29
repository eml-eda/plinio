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
from typing import Optional, cast, Dict, Any
import torch
import torch.nn as nn
import torch.fx as fx
import torch.nn.functional as F
from flexnas.utils.features_calculator import ConstFeaturesCalculator, FeaturesCalculator
from .pit_layer import PITLayer
from .pit_features_masker import PITFeaturesMasker
from .pit_binarizer import PITBinarizer


class PITLinear(nn.Linear, PITLayer):
    """A nn.Module implementing a Linear layer optimizable with the PIT NAS tool

    :param conv: the inner `torch.nn.Linear` layer to be optimized
    :type conv: nn.Linear
    :param out_features_masker: the `nn.Module` that generates the output featrues binary masks.
    :type out_features_masker: PITChannelMasker
    :raises ValueError: for unsupported regularizers
    :param binarization_threshold: the binarization threshold for PIT masks, defaults to 0.5
    :type binarization_threshold: float, optional
    """
    def __init__(self,
                 linear: nn.Linear,
                 out_features_masker: PITFeaturesMasker,
                 binarization_threshold: float = 0.5,
                 ):
        super(PITLinear, self).__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None)
        self.weight = linear.weight
        self.bias = linear.bias
        # this will be overwritten later when we process the model graph
        self._input_features_calculator = ConstFeaturesCalculator(linear.in_features)
        self.out_features_masker = out_features_masker
        self._binarization_threshold = binarization_threshold
        self.register_buffer('out_features_eff', torch.tensor(self.out_features,
                             dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the NAS-able layer.

        In a nutshell, uses the various Maskers to generate the binarized masks, then runs
        the linear layer with the masked weights tensor.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        alpha = self.out_features_masker()
        bin_alpha = PITBinarizer.apply(alpha, self._binarization_threshold)
        # TODO: check that the result is correct after removing the two transposes present in
        # Matteo's original version
        pruned_weight = torch.mul(self.weight, bin_alpha.unsqueeze(1))

        # linear operation
        y = F.linear(input, pruned_weight, self.bias)

        # save info for regularization
        self.out_features_eff = torch.sum(alpha)

        return y

    @staticmethod
    def autoimport(n: fx.Node, mod: fx.GraphModule, sm: Optional[PITFeaturesMasker]):
        """Create a new fx.Node relative to a PITLinear layer, starting from the fx.Node
        of a nn.Linear layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to a nn.Linear layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param sm: An optional shared output channel masker derived from subsequent layers
        :type sm: Optional[PITChannelMasker]
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != nn.Linear:
            raise TypeError(f"Trying to generate PITLinear from layer of type{type(submodule)}")
        # here, this is guaranteed
        submodule = cast(nn.Linear, submodule)
        chan_masker = sm if sm is not None else PITFeaturesMasker(submodule.out_features)
        new_submodule = PITLinear(submodule, out_features_masker=chan_masker)
        mod.add_submodule(str(n.target), new_submodule)
        return

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a PITConv2D layer, with a standard nn.Conv2D layer
        within a fx.GraphModule

        :param n: the node to be rewritten, corresponds to a Conv1D layer
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != PITLinear:
            raise TypeError(f"Trying to export a layer of type{type(submodule)}")
        # here, this is guaranteed
        submodule = cast(PITLinear, submodule)
        cout_mask = submodule.features_mask.bool()
        cin_mask = submodule.input_features_calculator.features_mask.bool()
        # note: kernel size and dilation are not optimized for conv2d
        new_submodule = nn.Linear(
            submodule.in_features_opt,
            submodule.out_features_opt,
            submodule.bias is not None)
        new_weights = submodule.weight[cout_mask, :]
        new_weights = new_weights[:, cin_mask]
        new_submodule.weight = nn.parameter.Parameter(new_weights)
        if submodule.bias is not None:
            new_submodule.bias = nn.parameter.Parameter(submodule.bias[cout_mask])
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
        with torch.no_grad():
            bin_alpha = self.features_mask
            return int(torch.sum(bin_alpha))

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
            alpha = self.out_features_masker()
            return PITBinarizer.apply(alpha, self._binarization_threshold)

    def get_size(self) -> torch.Tensor:
        """Method that computes the number of weights for the layer

        :return: the number of weights
        :rtype: torch.Tensor
        """
        cin = self.input_features_calculator.features
        cost = cin * self.out_features_eff
        return cost

    def get_macs(self) -> torch.Tensor:
        """Method that computes the number of MAC operations for the layer

        :return: the number of MACs
        :rtype: torch.Tensor
        """
        # size and MACs are roughly the same for a Linear layer
        return self.get_size()

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

    @property
    def train_features(self) -> bool:
        """True if the output features are being optimized by PIT for this layer.

        :return: True if the output features are being optimized by PIT for this layer
        :rtype: bool
        """
        return self.out_features_masker.trainable

    @train_features.setter
    def train_features(self, value: bool):
        """Set to True in order to let PIT optimize the output features for this layer

        :param value: set to True in order to let PIT optimize the output features for this layer
        :type value: bool
        """
        self.out_features_masker.trainable = value
