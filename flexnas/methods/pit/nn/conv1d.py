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

from typing import Tuple, Dict, Any, Optional, cast, Iterator
import torch
import torch.fx as fx
import torch.nn as nn
import itertools
from flexnas.graph.features_calculation import ConstFeaturesCalculator, FeaturesCalculator
from .features_masker import PITFeaturesMasker
from .timestep_masker import PITTimestepMasker, PITFrozenTimestepMasker
from .dilation_masker import PITDilationMasker, PITFrozenDilationMasker
from .binarizer import PITBinarizer
from .module import PITModule


class PITConv1d(nn.Conv1d, PITModule):
    """A nn.Module implementing a Conv1D layer optimizable with the PIT NAS tool

    :param conv: the inner `torch.nn.Conv1D` layer to be optimized
    :type conv: nn.Conv1d
    :param out_length: the output length on the time axis (needed for timestep and dilation masking
    and to compute the number of MACs)
    :type out_length: int
    :param out_features_masker: the `nn.Module` that generates the output features binary masks
    :type out_features_masker: PITChannelMasker
    :param timestep_masker: the `nn.Module` that generates the output timesteps binary masks
    :type timestep_masker: PITTimestepMasker
    :param dilation_masker: the `nn.Module` that generates the dilation binary masks
    :type dilation_masker: PITDilationMasker
    :raises ValueError: for unsupported regularizers
    :param binarization_threshold: the binarization threshold for PIT masks, defaults to 0.5
    :type binarization_threshold: float, optional
    """
    def __init__(self,
                 conv: nn.Conv1d,
                 out_length: int,
                 out_features_masker: PITFeaturesMasker,
                 timestep_masker: PITTimestepMasker,
                 dilation_masker: PITDilationMasker,
                 binarization_threshold: float = 0.5,
                 ):
        super(PITConv1d, self).__init__(
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
        self.out_length = out_length
        # this will be overwritten later when we process the model graph
        self._input_features_calculator = ConstFeaturesCalculator(conv.in_channels)
        self.out_features_masker = out_features_masker
        self.timestep_masker = timestep_masker
        self.dilation_masker = dilation_masker
        self._binarization_threshold = binarization_threshold
        self.following_bn_args: Optional[Dict[str, Any]] = None
        _beta_norm, _gamma_norm = self._generate_norm_constants()
        self.register_buffer('_beta_norm', _beta_norm)
        self.register_buffer('_gamma_norm', _gamma_norm)
        self.register_buffer('out_features_eff', torch.tensor(self.out_channels,
                             dtype=torch.float32))
        self.register_buffer('k_eff', torch.tensor(self.kernel_size[0], dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the NAS-able layer.

        In a nutshell, uses the various maskers to generate the binarized masks, then runs
        the convolution with the masked weights tensor.
        This function has side effects, since it also saves the effective output features and
        effective filter size in `out_features_eff` and `k_eff` respectively.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        # for now we keep the same order of the old version (ch --> dil --> rf)
        # but it's probably more natural to do ch --> rf --> dil
        theta_alpha = self.out_features_masker.theta
        bin_theta_alpha = PITBinarizer.apply(theta_alpha, self._binarization_threshold)
        pruned_weight = torch.mul(self.weight, bin_theta_alpha.unsqueeze(1).unsqueeze(1))
        theta_gamma = self.dilation_masker.theta
        bin_theta_gamma = PITBinarizer.apply(theta_gamma, self._binarization_threshold)
        pruned_weight = torch.mul(bin_theta_gamma, pruned_weight)
        theta_beta = self.timestep_masker.theta
        bin_theta_beta = PITBinarizer.apply(theta_beta, self._binarization_threshold)
        pruned_weight = torch.mul(bin_theta_beta, pruned_weight)

        # conv operation
        y = self._conv_forward(input, pruned_weight, self.bias)

        # save info for regularization
        norm_theta_beta = torch.mul(theta_beta, cast(torch.Tensor, self._beta_norm))
        norm_theta_gamma = torch.mul(theta_gamma, cast(torch.Tensor, self._gamma_norm))
        self.out_features_eff = torch.sum(theta_alpha)
        self.k_eff = torch.sum(torch.mul(norm_theta_beta, norm_theta_gamma))

        return y

    @staticmethod
    def autoimport(n: fx.Node, mod: fx.GraphModule, fm: PITFeaturesMasker):
        """Create a new fx.Node relative to a PITConv1d layer, starting from the fx.Node
        of a nn.Conv1d layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to a nn.Conv1d layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param fm: the output features masker to use for this layer
        :type fm: PITFeaturesMasker
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != nn.Conv1d:
            raise TypeError(f"Trying to generate PITConv1d from layer of type{type(submodule)}")
        # here, this is guaranteed
        submodule = cast(nn.Conv1d, submodule)
        rf = submodule.kernel_size[0]
        stride = submodule.stride if isinstance(submodule.stride, int) else submodule.stride[0]
        # PIT cannot optimize rf and dilation with stride != 1
        time_masker = PITFrozenTimestepMasker(rf) if stride != 1 else PITTimestepMasker(rf)
        dil_masker = PITFrozenDilationMasker(rf) if stride != 1 else PITDilationMasker(rf)
        new_submodule = PITConv1d(
            submodule,
            out_length=n.meta['tensor_meta'].shape[2],
            out_features_masker=fm,
            timestep_masker=time_masker,
            dilation_masker=dil_masker,
        )
        mod.add_submodule(str(n.target), new_submodule)

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a PITConv1D layer, with a standard nn.Conv1D layer
        within a fx.GraphModule

        :param n: the node to be rewritten, corresponds to a Conv1D layer
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != PITConv1d:
            raise TypeError(f"Trying to export a layer of type{type(submodule)}")
        # here, this is guaranteed
        submodule = cast(PITConv1d, submodule)
        cout_mask = submodule.features_mask.bool()
        cin_mask = submodule.input_features_calculator.features_mask.bool()
        time_mask = submodule.time_mask.bool()
        is_depthwise = (submodule.groups == submodule.in_channels) and (
            submodule.groups == submodule.out_channels)
        if is_depthwise:
            groups_opt = submodule.in_features_opt
        else:
            groups_opt = submodule.groups
        new_submodule = nn.Conv1d(
            submodule.in_features_opt,
            submodule.out_features_opt,
            submodule.kernel_size_opt,
            submodule.stride,
            submodule.padding,
            submodule.dilation_opt,
            groups_opt,
            submodule.bias is not None,
            submodule.padding_mode)
        new_weights = submodule.weight[cout_mask, :, :]
        if not is_depthwise:
            # for DWConv we have dimension 1 in the cin axis
            # note: we don't handle other groupwise variants yet
            new_weights = new_weights[:, cin_mask, :]
        new_weights = new_weights[:, :, time_mask]
        with torch.no_grad():
            new_submodule.weight.copy_(new_weights)
            if submodule.bias is not None:
                cast(nn.parameter.Parameter, new_submodule.bias).copy_(submodule.bias[cout_mask])
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
            'kernel_size': self.kernel_size_opt,
            'dilation': self.dilation_opt
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
        for name, param in self.timestep_masker.named_parameters(
                prfx + "timestep_masker", recurse):
            yield name, param
        for name, param in self.dilation_masker.named_parameters(
                prfx + "dilation_masker", recurse):
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
    def dilation_opt(self) -> Tuple[int]:
        """Get the dilation found during the search

        :return: the dilation found during the search
        :rtype: Tuple[int]
        """
        with torch.no_grad():
            theta_gamma = self.dilation_masker.theta
            bin_theta_gamma = PITBinarizer.apply(theta_gamma, self._binarization_threshold)
            # find the longest sequence of 0s in the bin_theta_gamma mask
            dil = max((sum(1 for _ in group) for value, group in itertools.groupby(bin_theta_gamma)
                      if value == 0), default=0) + 1
            # multiply times dilation[0] because the original layer could have had dilation > 1
            return (int(dil) * self.dilation[0],)

    @property
    def kernel_size_opt(self) -> Tuple[int]:
        """Get the kernel size found during the search

        :return: the kernel sizse found during the search
        :rtype: Tuple[int]
        """
        with torch.no_grad():
            bg_prod = self.time_mask
            return (int(torch.sum(bg_prod)),)

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

    @property
    def time_mask(self) -> torch.Tensor:
        """Return the binarized mask that specifies which input timesteps are kept by the NAS

        This includes both the effect of dilation and receptive field searches

        :return: the binarized mask over the time axis
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            theta_gamma = self.dilation_masker.theta
            bin_theta_gamma = PITBinarizer.apply(theta_gamma, self._binarization_threshold)
            theta_beta = self.timestep_masker.theta
            bin_theta_beta = PITBinarizer.apply(theta_beta, self._binarization_threshold)
            bg_prod = torch.mul(bin_theta_gamma, bin_theta_beta)
            return bg_prod

    def get_size(self) -> torch.Tensor:
        """Method that computes the number of weights for the layer

        :return: the number of weights
        :rtype: torch.Tensor
        """
        cin = self.input_features_calculator.features
        cost = cin * self.out_features_eff * self.k_eff
        cost = cost / self.groups
        return cost

    def get_macs(self) -> torch.Tensor:
        """Method that computes the number of MAC operations for the layer

        :return: the number of MACs
        :rtype: torch.Tensor
        """
        return self.get_size() * self.out_length

    def _generate_norm_constants(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method called at construction time to generate the normalization constants for the
        correct evaluation of the effective kernel size.

        The details of how these constants are computed are found in the journal paper.

        :return: A tuple of (beta, gamma) normalization constants.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        beta_norm = torch.tensor(
            [1.0 / (self.kernel_size[0] - i) for i in range(self.kernel_size[0])],
        )
        # everything on the time-axis is flipped with respect to the paper
        beta_norm = torch.flip(beta_norm, (0,))
        gamma_norm = []
        for i in range(self.kernel_size[0]):
            k_i = 0
            for p in range(self.dilation_masker._gamma_len):
                k_i += 0 if i % 2**p == 0 else 1
            gamma_norm.append(1.0 / (self.dilation_masker._gamma_len - k_i))
        # everything on the time-axis is flipped with respect to the paper
        gamma_norm = torch.flip(torch.tensor(gamma_norm, dtype=torch.float32), (0,))
        return beta_norm, gamma_norm

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
    def rf(self) -> int:
        """Returns the static (i.e., maximum) receptive field of this layer

        :return: the static (i.e., maximum) receptive field of this layer
        :rtype: int
        """
        return (self.kernel_size[0] - 1) * self.dilation[0] + 1

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

    @property
    def train_rf(self) -> bool:
        """True if the receptive field is being optimized by PIT for this layer

        :return: True if the receptive field is being optimized by PIT for this layer
        :rtype: bool
        """
        return self.timestep_masker.trainable

    @train_rf.setter
    def train_rf(self, value: bool):
        """Set to True in order to let PIT optimize the receptive field for this layer

        :param value: set to True in order to let PIT optimize the receptive field for this layer
        :type value: bool
        """
        self.timestep_masker.trainable = value

    @property
    def train_dilation(self) -> bool:
        """True if the dilation is being optimized by PIT for this layer

        :return: True if the dilation is being optimized by PIT for this layer
        :rtype: bool
        """
        return self.dilation_masker.trainable

    @train_dilation.setter
    def train_dilation(self, value: bool):
        """Set to True in order to let PIT optimize the dilation for this layer

        :param value: set to True in order to let PIT optimize the dilation for this layer
        :type value: bool
        """
        self.dilation_masker.trainable = value
