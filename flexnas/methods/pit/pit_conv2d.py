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

import torch
import torch.nn as nn
from flexnas.utils.features_calculator import ConstFeaturesCalculator, FeaturesCalculator
from .pit_channel_masker import PITChannelMasker
from .pit_binarizer import PITBinarizer


class PITConv2d(nn.Conv2d):
    """A nn.Module implementing a Conv2D layer optimizable with the PIT NAS tool

    :param conv: the inner `torch.nn.Conv2D` layer to be optimized
    :type conv: nn.Conv2d
    :param out_length: the height of the output feature map (needed to compute the MACs)
    :type out_length: int
    :param out_width: the width of the output feature map (needed to compute the MACs)
    :type out_length: int
    :param out_channel_masker: the `nn.Module` that generates the output channels binary masks
    :type out_channel_masker: PITChannelMasker
    :raises ValueError: for unsupported regularizers
    :param binarization_threshold: the binarization threshold for PIT masks, defaults to 0.5
    :type binarization_threshold: float, optional
    """
    def __init__(self,
                 conv: nn.Conv2d,
                 out_height: int,
                 out_width: int,
                 out_channel_masker: PITChannelMasker,
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
        self.weight = conv.weight
        self.bias = conv.bias
        self.out_height = out_height
        self.out_width = out_width
        # this will be overwritten later when we process the model graph
        self._input_features_calculator = ConstFeaturesCalculator(conv.in_channels)
        self.out_channel_masker = out_channel_masker
        self._binarization_threshold = binarization_threshold
        self.register_buffer('out_channels_eff', torch.tensor(self.out_channels,
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
        alpha = self.out_channel_masker()
        bin_alpha = PITBinarizer.apply(alpha, self._binarization_threshold)
        # TODO: check that the result is correct after removing the two transposes present in
        # Matteo's original version
        pruned_weight = torch.mul(self.weight, bin_alpha.unsqueeze(1).unsqueeze(1))

        # conv operation
        y = self._conv_forward(input, pruned_weight, self.bias)

        # save info for regularization
        self.out_channels_eff = torch.sum(alpha)

        return y

    @property
    def out_channels_opt(self) -> int:
        """Get the number of output channels found during the search

        :return: the number of output channels found during the search
        :rtype: int
        """
        with torch.no_grad():
            bin_alpha = self.features_mask
            return int(torch.sum(bin_alpha))

    @property
    def in_channels_opt(self) -> int:
        """Get the number of input channels found during the search

        :return: the number of input channels found during the search
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
            alpha = self.out_channel_masker()
            return PITBinarizer.apply(alpha, self._binarization_threshold)

    def get_size(self) -> torch.Tensor:
        """Method that computes the number of weights for the layer

        :return: the number of weights
        :rtype: torch.Tensor
        """
        cin = self.input_features_calculator.features
        cost = cin * self.out_channels_eff * self.kernel_size[0] * self.kernel_size[1]
        return cost

    def get_macs(self) -> torch.Tensor:
        """Method that computes the number of MAC operations for the layer

        :return: the number of MACs
        :rtype: torch.Tensor
        """
        return self.get_size() * self.out_height * self.out_width

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
    def train_channels(self) -> bool:
        """True if the output channels are being optimized by PIT for this layer

        :return: True if the output channels are being optimized by PIT for this layer
        :rtype: bool
        """
        return self.out_channel_masker.trainable

    @train_channels.setter
    def train_channels(self, value: bool):
        """Set to True in order to let PIT optimize the output channels for this layer

        :param value: set to True in order to let PIT optimize the output channels for this layer
        :type value: bool
        """
        self.out_channel_masker.trainable = value
