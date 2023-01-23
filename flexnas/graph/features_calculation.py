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
from typing import List, cast
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

    @abstractmethod
    def register(self, mod: nn.Module, prefix: str = ""):
        """Register the (optional) buffers required by the features calculator within a nn.Module

        :param mod: the target module
        :type mod: nn.Module
        :param prefix: a string prefix for the buffer name
        :type prefix: str
        :raises NotImplementedError: on the base FeaturesCalculator class
        :return: the binarized features mask
        :rtype: int
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class ConstFeaturesCalculator(FeaturesCalculator):
    """A `FeaturesCalculator` that simply returns a constant.

    Used for layers' whose predecessors are not NAS-able.

    :param const: the constant number of features.
    :type const: int
    """
    def __init__(self, const: int):
        super(ConstFeaturesCalculator, self).__init__()
        self.const = torch.tensor(const)
        self.mask = torch.ones((const,))
        self.mod = None

    @property
    def features(self) -> torch.Tensor:
        return cast(torch.Tensor, cast(nn.Module, self.mod).feat_calc_const)

    @property
    def features_mask(self) -> torch.Tensor:
        return cast(torch.Tensor, cast(nn.Module, self.mod).feat_calc_mask)

    def register(self, mod: nn.Module, prefix: str = ""):
        if self.mod is None:
            self.mod = mod
            mod.register_buffer('feat_calc_const', self.const)
            mod.register_buffer('feat_calc_mask', self.mask)


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

    def register(self, mod: nn.Module, prefix: str = ""):
        # nothing to do here. we assume the mod attr is always a registered buffer
        pass


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
        self.mod = None
        self.multiplier = torch.tensor(multiplier)
        self.mask_expander = torch.ones((multiplier,))

    @property
    def features(self) -> torch.Tensor:
        mul = cast(nn.Module, self.mod).feat_calc_multiplier
        return mul * self.prev.features

    @property
    def features_mask(self) -> torch.Tensor:
        prev_mask = self.prev.features_mask
        mask_list = []
        for elm in prev_mask:
            mask_list.append(elm * cast(nn.Module, self.mod).feat_calc_mask_expander)
        mask = torch.cat(mask_list, dim=0)
        return mask

    def register(self, mod: nn.Module, prefix: str = ""):
        # recursively ensure that predecessors are registers
        prefix = "prev_" + prefix
        self.prev.register(mod, prefix)
        if self.mod is None:
            self.mod = mod
            mod.register_buffer('feat_calc_multiplier', self.multiplier)
            mod.register_buffer('feat_calc_mask_expander', self.mask_expander)


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

    def register(self, mod: nn.Module, prefix: str = ""):
        # recursively ensure that predecessors are registers
        for i, fc in enumerate(self.inputs):
            prefix = f"prev_{i}" + prefix
            fc.register(mod, prefix)


class SoftMaxFeaturesCalculator(FeaturesCalculator):

    def __init__(self, mod: nn.Module, attr_name: str, inputs: List[FeaturesCalculator]):
        super(SoftMaxFeaturesCalculator, self).__init__()
        self.inputs = inputs
        self.mod = mod
        self.attr_name = attr_name

    @property
    def features(self) -> torch.Tensor:
        # combinazione lineare con ogni coeff (alpha) che moltiplica il corrispondente input,
        # non so se si fa cosi in torch, consideriamolo pseudo-codice
        coeff = nn.functional.softmax(getattr(self.mod, self.attr_name), dim=0)
        prev = [_.features for _ in self.inputs]

        outfeat = torch.zeros()
        for i, input in enumerate(prev):
            other = coeff[i] * input
            torch.add(outfeat, other, out=outfeat)

        return outfeat

    @property
    def features_mask(self) -> torch.Tensor:
        # maschera dell'input corrispondente all'argmax, anche qui pseudo-codice
        amax = torch.argmax(getattr(self.mod, self.attr_name)).item()
        return self.inputs[int(amax)].features_mask

    def register(self, mod: nn.Module, prefix: str = ""):
        # uguale al caso del concat
        # recursively ensure that predecessors are registers
        for i, fc in enumerate(self.inputs):
            prefix = f"prev_{i}" + prefix
            fc.register(mod, prefix)
