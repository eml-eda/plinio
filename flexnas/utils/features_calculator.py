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
from typing import List, Callable
import torch.nn as nn


class FeaturesCalculator:

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def input_features(self) -> int:
        raise NotImplementedError


class ConstFeaturesCalculator(FeaturesCalculator):

    def __init__(self, const: int):
        super(ConstFeaturesCalculator, self).__init__()
        self.const = const

    @property
    def input_features(self) -> int:
        return self.const


class ModAttrFeaturesCalculator(FeaturesCalculator):

    def __init__(self, mod: nn.Module, attr_name: str):
        super(ModAttrFeaturesCalculator, self).__init__()
        self.mod = mod
        self.attr_name = attr_name

    @property
    def input_features(self) -> int:
        return getattr(self.mod, self.attr_name)


class LinearFeaturesCalculator(FeaturesCalculator):

    def __init__(self, prev: FeaturesCalculator, multiplier: int):
        super(LinearFeaturesCalculator, self).__init__()
        self.prev = prev
        self.multiplier = multiplier

    @property
    def input_features(self) -> int:
        return self.multiplier * self.prev.input_features


class ListReduceFeaturesCalculator(FeaturesCalculator):

    def __init__(self, inputs: List[FeaturesCalculator], fn: Callable):
        super(ListReduceFeaturesCalculator, self).__init__()
        self.inputs = inputs
        self.fn = fn

    @property
    def input_features(self) -> int:
        fn_params = [_.input_features for _ in self.inputs]
        return self.fn(fn_params)
