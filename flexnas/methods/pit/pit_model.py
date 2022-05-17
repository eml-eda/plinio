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

from typing import Tuple, Type, Iterable, Optional, Dict
import math
import operator
import torch
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp
from flexnas.methods.dnas_base import DNASModel
from .pit_conv1d import PITConv1d
from .pit_channel_masker import PITChannelMasker
from .pit_timestep_masker import PITTimestepMasker
from .pit_dilation_masker import PITDilationMasker
from flexnas.utils.model_graph import *
from flexnas.utils.features_calculator import *


class PITModel(DNASModel):

    # supported regularizers
    regularizers = (
        'size',
        'flops'
    )

    def __init__(
            self,
            model: nn.Module,
            input_example: torch.Tensor,
            regularizer: Optional[str] = 'size',
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            train_channels=True,
            train_rf=True,
            train_dilation=True):
        super(PITModel, self).__init__(model, regularizer, exclude_names, exclude_types)
        self._input_example = input_example
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
        ShapeProp(mod).propagate(self._input_example)
        self._convert_layers(mod)
        self._set_input_sizes(mod)
        mod.recompile()
        self._inner_model = mod

    def _convert_layers(self, mod: fx.GraphModule):
        # reverse bfs on the graph
        g = mod.graph
        queue = get_output_nodes(g)
        shared_masker_queue = [None] * len(queue)
        visited = []
        while queue:
            n = queue.pop(0)
            shared_masker = shared_masker_queue.pop(0)
            visited.append(n)
            shared_masker = self._rewrite_node(n, mod, shared_masker)
            for pred in n.all_input_nodes:
                queue.append(pred)
                shared_masker_queue.append(shared_masker)

    def _rewrite_node(self, n: fx.Node, mod: fx.GraphModule, shared_masker: Optional[PITChannelMasker]
                      ) -> Optional[PITChannelMasker]:

        same_input_size_functions = (
            torch.add, torch.sub, operator.add
        )

        same_input_size_modules = (
            # TODO: fill
        )

        if n.op == 'call_module':
            submodule = mod.get_submodule(n.target)
            # add other NAS-able layers here as needed
            if type(submodule) is nn.Conv1d:
                if type(submodule) not in self.exclude_types and n.name not in self.exclude_names:
                    if shared_masker is not None:
                        chan_masker = shared_masker
                    else:
                        chan_masker = PITChannelMasker(submodule.out_channels)
                    new_submodule = PITConv1d(
                        submodule,
                        n.meta['tensor_meta'].shape[1],
                        self.regularizer,
                        chan_masker,
                        PITTimestepMasker(submodule.kernel_size[0]),
                        PITDilationMasker(submodule.kernel_size[0]),
                    )
                    mod.add_submodule(n.target, new_submodule)
                    self._target_layers.append(new_submodule)
                    return None
            # other single-input modules, nothing to do and no sharing
            elif len(n.all_input_nodes) == 1:
                return None
            # other modules that require multiple inputs all of the same size
            elif type(submodule) in same_input_size_modules:
                # create a new shared masker with the common n. of input channels, (possibly) used by predecessors
                input_size = n.all_input_nodes[0].meta['tensor_meta'].shape[1]
                shared_masker = PITChannelMasker(input_size)
                return shared_masker
            else:
                raise ValueError("Unsupported module node {}".format(n))

        elif n.op == 'call_function':
            # single input functions, nothing to do and no sharing
            if len(n.all_input_nodes) == 1:
                return None
            # functions that require multiple inputs all of the same size
            elif n.target in same_input_size_functions:
                # create a new shared masker with the common n. of input channels, (possibly) used by predecessors
                input_size = n.all_input_nodes[0].meta['tensor_meta'].shape[1]
                shared_masker = PITChannelMasker(input_size)
                return shared_masker
            else:
                raise ValueError("Unsupported function node {}".format(n))
        return None

    def _set_input_sizes(self, mod: fx.GraphModule):
        # forward bfs on the graph
        g = mod.graph
        # convert to networkx graph to have successors information
        nx_graph = fx_to_nx_graph(g)
        queue = get_input_nodes(g)
        calc_dict = {}
        visited = []
        while queue:
            n = queue.pop(0)
            visited.append(n)
            self._update_input_size_calculator(n, mod, calc_dict)
            for succ in nx_graph.successors(n):
                queue.append(succ)

    @staticmethod
    def _update_input_size_calculator(n: fx.Node, mod: fx.GraphModule,
                                      calc_dict: Dict[fx.Node, Optional[FeaturesCalculator]]):

        channel_defining_modules = (
            nn.Conv1d, nn.Conv2d, nn.Linear
        )

        channel_propagating_modules = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.AvgPool1d, nn.AvgPool2d, nn.MaxPool1d, nn.BatchNorm2d, nn.Dropout,
            nn.ReLU, nn.ReLU6, nn.ConstantPad1d, nn.ConstantPad2d
        )

        channel_propagating_functions = (
            F.relu, F.relu6, F.log_softmax
        )

        # skip nodes for which predecessors have not yet been processed completely, we'll come back to them later
        if len(n.all_input_nodes) > 0:
            for i in n.all_input_nodes:
                if i not in calc_dict:
                    return

        if n.op == 'placeholder':
            calc_dict[n] = None

        if n.op == 'call_module':
            sub_mod = mod.get_submodule(n.target)
            if type(sub_mod) in channel_defining_modules:
                calc_dict[n] = ConstFeaturesCalculator(n.meta['tensor_meta'].shape[1])
            elif type(sub_mod) in channel_propagating_modules:
                calc_dict[n] = calc_dict[n.all_input_nodes[0]]  # they all have a single input
            elif type(sub_mod) == PITConv1d:
                prev = n.all_input_nodes[0]
                if calc_dict[prev] is None:  # it means this is the first layer after an input node
                    ifc = ModAttrFeaturesCalculator(sub_mod, 'in_channels')
                else:
                    ifc = calc_dict[prev]
                # special case of PIT layers. here we also have to set the input_size_calculator attribute
                sub_mod.input_size_calculator = ifc
                calc_dict[n] = ModAttrFeaturesCalculator(sub_mod, 'out_channels_eff')
            else:
                raise ValueError("Unsupported module node {}".format(n))

        if (n.op == 'call_method' and n.target == 'flatten') or (n.op == 'call_function' and n.target == torch.flatten):
                ifc = calc_dict[n.all_input_nodes[0]]
                input_size = n.all_input_nodes[0].meta['tensor_meta'].shape
                # exclude batch size and channels
                spatial_size = math.prod(input_size[2:])
                calc_dict[n] = LinearFeaturesCalculator(ifc, spatial_size)

        if n.op == 'call_function':
            if n.target == torch.add or n.target == operator.add:
                # assumes all inputs to add have the same number of features (as expected)
                calc_dict[n] = calc_dict[n.all_input_nodes[0]]
                # alternative
                # ifc = ListReduceFeaturesCalculator([calc_dict[_] for _ in n.all_input_nodes], max)
                # calc_dict[n] = ifc
            elif n.target == torch.cat:
                # TODO: this assumes that concatenation is always on the features axis. Not always true. Fix.
                ifc = ListReduceFeaturesCalculator([calc_dict[_] for _ in n.all_input_nodes], sum)
                calc_dict[n] = ifc
            elif n.target in channel_propagating_functions:
                calc_dict[n] = calc_dict[n.all_input_nodes[0]]
            else:
                raise ValueError("Unsupported function node {}".format(n))

        return
