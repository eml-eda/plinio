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

from typing import Tuple, Type, Iterable, Optional, List
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from flexnas.methods.dnas_base import DNASModel
from .pit_conv1d import PITConv1d


class PITModel(DNASModel):

    replacement_dict = {nn.Conv1d: PITConv1d}
    regularizers = ('size', 'flops')

    def __init__(
            self,
            model: nn.Module,
            regularizer: Optional[str] = 'size',
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            train_channels=True,
            train_rf=True,
            train_dilation=True):
        super(PITModel, self).__init__(symbolic_trace(model), regularizer, exclude_names, exclude_types)
        self._convert_layers(self._inner_model)
        self._target_layers = self._annotate_target_layers(self._inner_model)
        self.train_channels = train_channels
        self.train_rf = train_rf
        self.train_dilation = train_dilation

    def supported_regularizers(self) -> Tuple[str, ...]:
        return PITModel.regularizers

    def get_regularization_loss(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def train_channels(self):
        return self._train_channels

    @train_channels.setter
    def train_channels(self, value: bool):
        for layer in self._target_layers:
            layer.train_channels = value
        self._train_channels = value

    @property
    def train_rf(self):
        return self._train_rf

    @train_rf.setter
    def train_rf(self, value: bool):
        for layer in self._target_layers:
            layer.train_rf = value
        self._train_rf = value

    @property
    def train_dilation(self):
        return self._train_dilation

    @train_dilation.setter
    def train_dilation(self, value: bool):
        for layer in self._target_layers:
            layer.train_dilation = value
        self._train_dilation = value

    def _convert_layers(self, mod: nn.Module):
        reassign = {}
        for name, child in mod.named_children():
            self._convert_layers(child)
            if isinstance(child, self._optimizable_layers()):
                if (name not in self.exclude_names) and (not isinstance(child, self.exclude_types)):
                    reassign[name] = self._replacement_layer(name, child)
        for k, new_layer in reassign.items():
            mod._modules[k] = new_layer

    def _annotate_target_layers(self, mod: nn.Module) -> List[nn.Module]:
        # this could have been done within _convert_layers, but separating the two avoids some corner case errors, e.g.
        # if a method first replaces children layers and then an entire hierarchical layer (making the children no
        # longer part of the model)
        tgt = []
        for name, child in mod.named_children():
            tgt += self._annotate_target_layers(child)
            if isinstance(child, PITModel._optimizable_layers()):
                tgt.append(child)
        return tgt

    def _replacement_layer(self, name: str, layer: nn.Module) -> nn.Module:
        for OldClass, NewClass in PITModel.replacement_dict.items():
            if isinstance(layer, OldClass):
                return NewClass(layer, self.regularizer)
        raise ValueError("Replacement Layer not found")

    @staticmethod
    def _optimizable_layers() -> Tuple[Type[nn.Module], ...]:
        return tuple(PITModel.replacement_dict.keys())
