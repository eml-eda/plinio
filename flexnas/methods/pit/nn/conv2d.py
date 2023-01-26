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
from typing import Dict, Any, Optional, cast, Iterator, Tuple
import torch
import torch.nn as nn
import torch.fx as fx
from flexnas.graph.features_calculation import ConstFeaturesCalculator, FeaturesCalculator
from .features_masker import PITFeaturesMasker
from .binarizer import PITBinarizer
from .module import PITModule


class PITConv2d(nn.Conv2d, PITModule):
    """A nn.Module implementing a Conv2D layer optimizable with the PIT NAS tool

    :param conv: the inner `torch.nn.Conv2D` layer to be optimized
    :type conv: nn.Conv2d
    :param out_length: the height of the output feature map (needed to compute the MACs)
    :type out_length: int
    :param out_width: the width of the output feature map (needed to compute the MACs)
    :type out_length: int
    :param out_features_masker: the `nn.Module` that generates the output features binary masks
    :type out_features_masker: PITChannelMasker
    :raises ValueError: for unsupported regularizers
    :param binarization_threshold: the binarization threshold for PIT masks, defaults to 0.5
    :type binarization_threshold: float, optional
    """
    def __init__(self,
                 conv: nn.Conv2d,
                 out_height: int,
                 out_width: int,
                 out_features_masker: PITFeaturesMasker,
                 binarization_threshold: float = 0.5,
                 ):
        super(PITConv2d, self).__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode)
        is_depthwise = (conv.groups == conv.in_channels and conv.groups == conv.out_channels)
        if conv.groups != 1 and (not is_depthwise):
            raise AttributeError(
                "PIT currently supports only full or DepthWise Conv., not other groupwise versions")
        with torch.no_grad():
            self.weight.copy_(conv.weight)
            if conv.bias is not None:
                cast(torch.nn.parameter.Parameter, self.bias).copy_(conv.bias)
            else:
                self.bias = None
        self.out_height = out_height
        self.out_width = out_width
        self.following_bn_args: Optional[Dict[str, Any]] = None
        # this will be overwritten later when we process the model graph
        self._input_features_calculator = ConstFeaturesCalculator(conv.in_channels)
        self.out_features_masker = out_features_masker
        self._binarization_threshold = binarization_threshold
        self.register_buffer('out_features_eff', torch.tensor(self.out_channels,
                             dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the NAS-able layer.

        In a nutshell, uses the various Maskers to generate the binarized masks, then runs
        the convolution with the masked weights tensor.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        theta_alpha = self.out_features_masker.theta
        bin_theta_alpha = PITBinarizer.apply(theta_alpha, self._binarization_threshold)
        pruned_weight = torch.mul(self.weight, bin_theta_alpha.view(-1, 1, 1, 1))

        # conv operation
        y = self._conv_forward(input, pruned_weight, self.bias)

        # save info for regularization
        self.out_features_eff = torch.sum(theta_alpha)

        return y

    @staticmethod
    def autoimport(n: fx.Node, mod: fx.GraphModule, fm: PITFeaturesMasker):
        """Create a new fx.Node relative to a PITConv2d layer, starting from the fx.Node
        of a nn.Conv2d layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to a nn.Conv1d layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param fm: the output features masker to use for this layer
        :type fm: PITFeaturesMasker
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != nn.Conv2d:
            raise TypeError(f"Trying to generate PITConv1d from layer of type{type(submodule)}")
        # here, this is guaranteed
        submodule = cast(nn.Conv2d, submodule)
        # note: kernel size and dilation are not optimized for conv2d
        new_submodule = PITConv2d(
            submodule,
            out_height=n.meta['tensor_meta'].shape[2],
            out_width=n.meta['tensor_meta'].shape[3],
            out_features_masker=fm,
        )
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
        if type(submodule) != PITConv2d:
            raise TypeError(f"Trying to export a layer of type{type(submodule)}")
        # here, this is guaranteed
        submodule = cast(PITConv2d, submodule)
        cout_mask = submodule.features_mask.bool()
        cin_mask = submodule.input_features_calculator.features_mask.bool()
        is_depthwise = (submodule.groups == submodule.in_channels) and (
            submodule.groups == submodule.out_channels)
        if is_depthwise:
            groups_opt = submodule.in_features_opt
        else:
            groups_opt = submodule.groups
        # note: kernel size and dilation are not optimized for conv2d
        new_submodule = nn.Conv2d(
            submodule.in_features_opt,
            submodule.out_features_opt,
            submodule.kernel_size,
            submodule.stride,
            submodule.padding,
            submodule.dilation,
            groups_opt,
            submodule.bias is not None,
            submodule.padding_mode)
        new_weights = submodule.weight[cout_mask, :, :, :]

        if not is_depthwise:
            # for DWConv we have dimension 1 in the cin axis
            # note: we don't handle other groupwise variants yet
            new_weights = new_weights[:, cin_mask, :, :]
        with torch.no_grad():
            new_submodule.weight.copy_(new_weights)
            if submodule.bias is not None:
                cast(nn.parameter.Parameter, new_submodule.bias).copy_(submodule.bias[cout_mask])
        mod.add_submodule(str(n.target), new_submodule)
        # unfuse the BatchNorm
        if submodule.following_bn_args is not None:
            new_bn = nn.BatchNorm2d(
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
                # TODO: previous row replaces also the input to the BN with the BN itself.
                # The following line fixes it. Is there a cleaner way to do this?
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
            theta_alpha = self.out_features_masker.theta
            return PITBinarizer.apply(theta_alpha, self._binarization_threshold)

    def get_size(self) -> torch.Tensor:
        """Method that computes the number of weights for the layer

        :return: the number of weights
        :rtype: torch.Tensor
        """
        cin = self.input_features_calculator.features
        cost = cin * self.out_features_eff * self.kernel_size[0] * self.kernel_size[1]
        cost = cost / self.groups
        return cost

    def get_size_binarized(self) -> torch.Tensor:
        """Method that computes the number of weights for the layer considering
        binarized masks

        :return: the number of weights
        :rtype: torch.Tensor
        """
        # Compute actual integer number of input channels
        cin_mask = self.input_features_calculator.features_mask
        cin = torch.sum(PITBinarizer.apply(cin_mask, self._binarization_threshold))
        # Compute actual integer number of output channels
        cout_mask = self.out_features_masker.theta
        cout = torch.sum(PITBinarizer.apply(cout_mask, self._binarization_threshold))
        # Finally compute cost
        cost = cin * cout * self.kernel_size[0] * self.kernel_size[1]
        cost = cost / self.groups
        return cost

    def get_macs(self) -> torch.Tensor:
        """Method that computes the number of MAC operations for the layer

        :return: the number of MACs
        :rtype: torch.Tensor
        """
        return self.get_size() * self.out_height * self.out_width

    def get_macs_binarized(self) -> torch.Tensor:
        """Method that computes the number of MAC operations for the layer
        considering binarized masks

        :return: the number of MACs
        :rtype: torch.Tensor
        """
        return self.get_size_binarized() * self.out_height * self.out_width

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
        """True if the output features are being optimized by PIT for this layer

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
