# *----------------------------------------------------------------------------*
# * Copyright (C) 2022 Politecnico di Torino, Italy                            *
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
from typing import Any, Iterable, Tuple, Type, List
import copy
import torch
import torch.nn as nn


class DNASModel(nn.Module):

    @abstractmethod
    def __init__(
            self,
            model: nn.Module,
            config: Any = None,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = ()):
        super(DNASModel, self).__init__()
        self.config = config
        self.exclude_names = exclude_names
        self.exclude_types = tuple(exclude_types)
        self._inner_model = self._prepare(model)
        self._target_layers = self._annotate_target_layers(self._inner_model)

    def forward(self, *args: Any):
        return self._inner_model.forward(*args)

    @abstractmethod
    def optimizable_layers(self) -> Tuple[Type[nn.Module]]:
        raise NotImplementedError

    @abstractmethod
    def replacement_layer(self, name: str, layer: nn.Module, model: nn.Module) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_regularization_loss(self) -> torch.Tensor:
        raise NotImplementedError

    def _prepare(self, model: nn.Module) -> nn.Module:
        model = copy.deepcopy(model)
        self._convert_layers(model, model)
        return model

    def _convert_layers(self, mod: nn.Module, top_level: nn.Module):
        reassign = {}
        for name, child in mod.named_children():
            self._convert_layers(child, top_level)
            if isinstance(child, self.optimizable_layers()):
                if (name not in self.exclude_names) and (not isinstance(child, self.exclude_types)):
                    reassign[name] = self.replacement_layer(
                        name, child, top_level)
        for k, new_layer in reassign.items():
            mod._modules[k] = new_layer

    def _annotate_target_layers(self, mod: nn.Module) -> List[nn.Module]:
        # this could have been done within _convert_layers, but separating the two avoids some corner case errors, e.g.
        # if a method first replaces children layers and then an entire hierarchical layer (making the children no
        # longer part of the model)
        tgt = []
        for name, child in mod.named_children():
            tgt += self._annotate_target_layers(child)
            if isinstance(child, self.optimizable_layers()):
                tgt.append(child)
        return tgt
