#*----------------------------------------------------------------------------*
#* Copyright (C) 2022 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Daniele Jahier Pagliari <daniele.jahier@polito.it>                *
#*----------------------------------------------------------------------------*

from typing import Iterable, Tuple
import torch
import torch.nn as nn
from abc import abstractmethod

class DNAS:

    @abstractmethod
    def __init__(self, config = None):
        self.config = config

    def prepare(self, net : nn.Module, exclude_names: Iterable[str] = [], exclude_types: Iterable[nn.Module] = []) -> nn.Module:
        rep_layers = self.optimizable_layers()
        exclude_types = tuple(exclude_types)
        for name, child in net.named_modules():
            if isinstance(child, rep_layers):
                if (name not in exclude_names) and (not isinstance(child, exclude_types)):
                    print("Found replaceable layer:", name, child)
                    new_layer = self._replacement_layer(name, child, net)
                    print("Replacement layer:", new_layer)
                    print("Custom parameter:", new_layer.custom_param)

    def optimizable_layers(self) -> Tuple[nn.Module]:
        return None

    @abstractmethod
    def _replacement_layer(self, name: str, layer: nn.Module, net: nn.Module) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_regularization_loss(self, net: nn.Module) -> torch.Tensor:
        raise NotImplementedError