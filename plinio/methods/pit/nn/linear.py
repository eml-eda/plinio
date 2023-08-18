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
from typing import cast, Dict, Any, Iterator, Tuple, Optional
import torch
import torch.nn as nn
import torch.fx as fx
import torch.nn.functional as F
from plinio.graph.features_calculation import ConstFeaturesCalculator, FeaturesCalculator
from .module import PITModule
from .features_masker import PITFeaturesMasker
from .binarizer import PITBinarizer


class PITLinear(nn.Linear, PITModule):
    """A nn.Module implementing a Linear layer optimizable with the PIT NAS tool

    :param conv: the inner `torch.nn.Linear` layer to be optimized
    :type conv: nn.Linear
    :param out_features_masker: the `nn.Module` that generates the output featrues binary masks.
    :type out_features_masker: PITChannelMasker
    :raises ValueError: for unsupported regularizers
    :param binarization_threshold: the binarization threshold for PIT masks, defaults to 0.5
    :type binarization_threshold: float, optional
    :param discrete_cost: True if the layer cost should be computed on a discretized sample
    :type discrete_cost: bool, default False
    """
    def __init__(self,
                 linear: nn.Linear,
                 out_features_masker: PITFeaturesMasker,
                 binarization_threshold: float = 0.5,
                 discrete_cost: bool = False,
                 ):
        super(PITLinear, self).__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None)
        with torch.no_grad():
            self.weight.copy_(linear.weight)
            if linear.bias is not None:
                cast(torch.nn.parameter.Parameter, self.bias).copy_(linear.bias)
            else:
                self.bias = None
        self.following_bn_args: Optional[Dict[str, Any]] = None
        # this will be overwritten later when we process the model graph
        self._input_features_calculator = ConstFeaturesCalculator(linear.in_features)
        self.out_features_masker = out_features_masker
        self.binarization_threshold = binarization_threshold
        self.discrete_cost = discrete_cost

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the NAS-able layer.

        In a nutshell, uses the various Maskers to generate the binarized masks, then runs
        the linear layer with the masked weights tensor.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        cout_mask = self._features_mask(discrete=True)
        pruned_weight = torch.mul(self.weight, cout_mask.unsqueeze(1))
        return F.linear(input, pruned_weight, self.bias)

    @staticmethod
    def autoimport(n: fx.Node, mod: fx.GraphModule, fm: PITFeaturesMasker):
        """Create a new fx.Node relative to a PITLinear layer, starting from the fx.Node
        of a nn.Linear layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to a nn.Conv1d layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param fm: the output features masker to use for this layer
        :type fm: PITFeaturesMasker
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != nn.Linear:
            raise TypeError(f"Trying to generate PITLinear from layer of type{type(submodule)}")
        # here, this is guaranteed
        submodule = cast(nn.Linear, submodule)
        new_submodule = PITLinear(submodule, out_features_masker=fm)
        mod.add_submodule(str(n.target), new_submodule)

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
        # note: kernel size and dilation are not optimized for Linear
        new_submodule = nn.Linear(
            submodule.in_features_opt,
            submodule.out_features_opt,
            submodule.bias is not None)
        new_weights = submodule.weight[cout_mask, :]
        new_weights = new_weights[:, cin_mask]
        with torch.no_grad():
            new_submodule.weight.copy_(new_weights)
            if submodule.bias is not None:
                new_submodule.bias.copy_(submodule.bias[cout_mask])
        mod.add_submodule(str(n.target), new_submodule)

        # unfuse the BatchNorm
        if submodule.following_bn_args is not None:
            new_bn = nn.BatchNorm1d(
                submodule.out_features_opt,
                eps=submodule.following_bn_args['eps'],
                momentum=submodule.following_bn_args['momentum'],
                affine=submodule.following_bn_args['affine'],
                track_running_stats=submodule.following_bn_args['track_running_stats']
            )
            mod.add_submodule(str(n.target) + "_exported_bn", new_bn)
            # add the batchnorm just after the conv in the graph
            with mod.graph.inserting_after(n):
                new_node = mod.graph.call_module(
                    str(n.target) + "_exported_bn",
                    args=(n,)
                )
                n.replace_all_uses_with(new_node)
                # The previous line replaces also the input to the BN with the BN itself.
                # The following line fixes it. Not sure if there's a cleaner way to do this?
                new_node.replace_input_with(new_node, n)
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
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        for name, param in self.out_features_masker.named_parameters(
                prfx + "out_features_masker", recurse):
            yield name, param

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
    def out_features_eff(self) -> torch.Tensor:
        return torch.sum(self._features_mask(self.discrete_cost))

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
            return self._features_mask(discrete=True)

    def get_modified_vars(self) -> Dict[str, Any]:
        """Method that returns the modified vars(self) dictionary for the instance, used for
        cost computation

        :return: the modified vars(self) data structure
        :rtype: Dict[str, Any]
        """
        v = dict(vars(self))
        v['in_features'] = self.input_features_calculator.features
        v['out_features'] = self.out_features_eff
        return v

    def _features_mask(self, discrete: bool) -> torch.Tensor:
        theta_alpha = self.out_features_masker.theta
        if discrete:
            theta_alpha = PITBinarizer.apply(theta_alpha, self.binarization_threshold)
        return cast(torch.Tensor, theta_alpha)

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
