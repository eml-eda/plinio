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

from typing import Tuple, Type, Iterable, Optional

import networkx as nx
import torch
import torch.nn as nn
import torch.fx as fx
from flexnas.methods.dnas_base import DNASModel
from flexnas.utils.model_graph import fx_to_nx_graph
from .pit_conv1d import PITConv1d


class PITModel(DNASModel):

    # dictionary of patterns and corresponding replacements
    module_rules = {
        nn.Conv1d: PITConv1d,
        # set([operator.add, torch.add, "add"]): PITAdd,
    }
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
        super(PITModel, self).__init__(model, regularizer, exclude_names, exclude_types)
        self._target_layers = []
        self._convert()
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

    def _convert(self):
        mod = fx.symbolic_trace(self._inner_model)
        grf = fx_to_nx_graph(mod.graph).reverse()
        self._convert_layers(mod, grf)
        mod.recompile()
        self._inner_model = mod

    def _convert_layers(self, mod: fx.GraphModule, g: nx.DiGraph):
        for n in mod.graph.nodes:
            if n.name not in self.exclude_names:
                if n.op == 'call_module':
                    mod = self._replace_module(n, mod, g)
                if n.op == 'call_function':
                    pass  # TODO: add

    def _replace_module(self, n: fx.Node, mod: fx.GraphModule, g: nx.DiGraph):
        old_submodule = mod.get_submodule(n.target)
        if isinstance(old_submodule, PITModel._opt_modules()):
            if not isinstance(old_submodule, self.exclude_types):
                new_submodule = self._replacement_module(old_submodule)
                mod.add_submodule(n.target, new_submodule)
                self._target_layers.append(new_submodule)
        return mod

    @staticmethod
    def _opt_modules() -> Tuple[Type[nn.Module], ...]:
        return tuple(PITModel.module_rules.keys())

    def _replacement_module(self, layer: nn.Module) -> nn.Module:
        for OldClass, NewClass in PITModel.module_rules.items():
            if isinstance(layer, OldClass):
                return NewClass(layer, self.regularizer)
        raise ValueError("Replacement Layer not found")
