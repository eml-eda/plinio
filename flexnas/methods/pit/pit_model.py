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

from typing import Tuple, Type, Iterable, Optional, Callable
import torch
import torch.nn as nn
import torch.fx as fx
from flexnas.methods.dnas_base import DNASModel
from .pit_conv1d import PITConv1d


class PITModel(DNASModel):

    # dictionary of nn.module classes and corresponding replacements
    replacement_module_rules = {
        nn.Conv1d: PITConv1d,
    }

    # list of NN modules that determine the number of input channels of subsequent layers, and corresponding
    # way to get then. Importantly, subclasses should be before superclasses.
    c_in_setter_module_rules = {
        PITConv1d: lambda x: x.out_channels_eff,
        nn.Conv1d: lambda x: x.out_channels,
        nn.Linear: lambda x: x.out_features,
    }

    # supported regularizers
    regularizers = (
        'size',
        'flops'
    )

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
        reg_loss = torch.tensor(0)
        for layer, in self._target_layers:
            reg_loss += layer.get_regularization_loss()
        return reg_loss

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
        self._convert_layers(mod)
        mod.recompile()
        self._inner_model = mod

    def _convert_layers(self, mod: fx.GraphModule):
        for n in mod.graph.nodes:
            if n.name not in self.exclude_names:
                if n.op == 'call_module':
                    mod = self._replace_module(n, mod)
                if n.op == 'call_function':
                    pass  # TODO: add (if needed)

    def _replace_module(self, n: fx.Node, mod: fx.GraphModule) -> fx.GraphModule:
        old_submodule = mod.get_submodule(n.target)
        if isinstance(old_submodule, tuple(PITModel.replacement_module_rules.keys())):
            if not isinstance(old_submodule, self.exclude_types):
                c_in_setter, c_in_func = PITModel._find_c_in_setter(n, mod)
                new_submodule = self._replacement_module(old_submodule, c_in_setter, c_in_func)
                mod.add_submodule(n.target, new_submodule)
                self._target_layers.append(new_submodule)
        return mod

    @staticmethod
    def _find_c_in_setter(n: fx.Node, mod: fx.GraphModule) -> Tuple[nn.Module, Callable]:
        r = PITModel._find_c_in_setter_recursive(n, mod)
        # this means that there's nothing before this layer in the graph that determines the number of output channels
        # so we simply use the number of input channels of this layer
        if r[1] is None:
            sub_mod = mod.get_submodule(n.target)
            return sub_mod, lambda x: x.in_channels
        return r

    @staticmethod
    def _find_c_in_setter_recursive(n: fx.Node, mod: fx.GraphModule) -> Tuple[Optional[nn.Module], Optional[Callable]]:
        if len(n.all_input_nodes) > 1:
            raise ValueError("More than one input node determines the number of input channels for a PIT layer." +
                             " This is not currently supported!")
        n = n.all_input_nodes[0]
        # input tensor
        if n.op == 'placeholder':
            return None, None
        # nn.Module
        elif n.op == 'call_module':
            sub_mod = mod.get_submodule(n.target)
            if isinstance(sub_mod, tuple(PITModel.c_in_setter_module_rules.keys())):
                return sub_mod, PITModel._passthrough_function(sub_mod)
            else:
                return PITModel._find_c_in_setter_recursive(n, mod)
        # torch method or builtin function
        elif n.op == 'call_function' or n.op == 'call_method':
            # TODO: add (if needed)
            return PITModel._find_c_in_setter_recursive(n, mod)
        # all other cases
        return PITModel._find_c_in_setter_recursive(n, mod)

    @staticmethod
    def _passthrough_function(layer: nn.Module) -> Callable:
        for ModuleClass, funct in PITModel.c_in_setter_module_rules.items():
            if isinstance(layer, ModuleClass):
                return funct
        raise ValueError("Passthrough function not found")

    def _replacement_module(
            self, layer: nn.Module, c_in_setter: Optional[nn.Module], c_in_funct: Callable) -> nn.Module:
        for OldClass, NewClass in PITModel.replacement_module_rules.items():
            if isinstance(layer, OldClass):
                return NewClass(layer, self.regularizer, c_in_setter, c_in_funct)
        raise ValueError("Replacement Layer not found")
