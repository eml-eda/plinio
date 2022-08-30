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
from typing import List
import torch
import torch.nn as nn


class FeaturesCalculator:
    """Abstract class computing the number of features (or channels) for a layer.

    This is needed because we cannot rely on (static) tensor shapes to compute the computational
    cost of a layer during the NAS optimization phase, given that part of the previous layer
    could be masked. Therefore, we need a quick way to compute the number of active (unmasked)
    input features for different types of layers (NAS-able and not NAS-able ones), which is
    what children of this class provide.
    """
    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def features(self) -> torch.Tensor:
        """Returns the number of effective features/channels.

        :raises NotImplementedError: on the base FeaturesCalculator class
        :return: the number of effective features/channels.
        :rtype: int
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def features_mask(self) -> torch.Tensor:
        """Returns the binarized input features mask

        :raises NotImplementedError: on the base FeaturesCalculator class
        :return: the binarized features mask
        :rtype: int
        """
        raise NotImplementedError


class ConstFeaturesCalculator(FeaturesCalculator):
    """A `FeaturesCalculator` that simply returns a constant.

    Used for layers' whose predecessors are not NAS-able.

    :param const: the constant number of features.
    :type const: int
    """
    def __init__(self, const: int):
        super(ConstFeaturesCalculator, self).__init__()
        self.const = torch.tensor(const)

    @property
    def features(self) -> torch.Tensor:
        return self.const

    @property
    def features_mask(self) -> torch.Tensor:
        return torch.ones((int(self.const),))


class ModAttrFeaturesCalculator(FeaturesCalculator):
    """A `FeaturesCalculator` that returns the number of features as an attribute of a `nn.Module`
    instance.

    Used for NAS-able layers such as PITConv1D, where the number of output features is stored in
    the `out_features_eff` attribute.

    :param mod: the `nn.Module` instance.
    :type mod: nn.Module
    :param attr_name: the attribute name that corresponds to the requested number of features
    :type attr_name: str
    :param mask_attr_name: the attribute name that corresponds to the binary features mask
    :type attr_name: str
    """
    def __init__(self, mod: nn.Module, attr_name: str, mask_attr_name: str):
        super(ModAttrFeaturesCalculator, self).__init__()
        self.mod = mod
        self.attr_name = attr_name
        self.mask_attr_name = mask_attr_name

    @property
    def features(self) -> torch.Tensor:
        return getattr(self.mod, self.attr_name)

    @property
    def features_mask(self) -> torch.Tensor:
        return getattr(self.mod, self.mask_attr_name)


class FlattenFeaturesCalculator(FeaturesCalculator):
    """A `FeaturesCalculator` that computes the number of features for a flatten operation

    :param prev: the `FeaturesCalculator` instance relative to the previous layer
    :type prev: FeaturesCalculator
    :param multiplier: a constant multiplication factor corresponding to the number of weights in
    each channel
    :type multiplier: int
    """
    def __init__(self, prev: FeaturesCalculator, multiplier: int):
        super(FlattenFeaturesCalculator, self).__init__()
        self.prev = prev
        self.multiplier = torch.tensor(multiplier)

    @property
    def features(self) -> torch.Tensor:
        return self.multiplier * self.prev.features

    @property
    def features_mask(self) -> torch.Tensor:
        prev_mask = self.prev.features_mask
        mask_list = []
        for elm in prev_mask:
            mask_list.append(elm * torch.ones((int(self.multiplier),)))
        mask = torch.cat(mask_list, dim=0)
        return mask


class ConcatFeaturesCalculator(FeaturesCalculator):
    """A `FeaturesCalculator` that computes the number of features for a concat operation

    For concat, the number of output features is the sum of all predecessors' output features.

    :param inputs: the list of `FeaturesCalculator` instances relative to the predecessors
    :type inputs: List[FeaturesCalculator]
    """
    def __init__(self, inputs: List[FeaturesCalculator]):
        super(ConcatFeaturesCalculator, self).__init__()
        self.inputs = inputs

    @property
    def features(self) -> torch.Tensor:
        fn_params = [_.features for _ in self.inputs]
        return torch.stack(fn_params, dim=0).sum()

    @property
    def features_mask(self) -> torch.Tensor:
        mask_list = []
        for prev in self.inputs:
            mask_list.append(prev.features_mask)
        mask = torch.cat(mask_list, dim=0)
        return mask
