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

from typing import cast, Tuple, Type, Iterable, Optional, Dict
import math
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
            train_channels: bool = True,
            train_rf: bool = True,
            train_dilation: bool = True):
        """PITModel NAS constructor.

        :param model: the inner nn.Module instance optimized by the NAS
        :type model: nn.Module
        :param input_example: an example of input tensor, required for symbolic tracing
        :type input_example: torch.Tensor`
        :param regularizer: a string defining the type of cost regularizer, defaults to 'size'
        :type regularizer: Optional[str], optional
        :param exclude_names: the names of `model` submodules that should be ignored by the NAS, defaults to ()
        :type exclude_names: Iterable[str], optional
        :param exclude_types: the types of `model` submodules that shuould be ignored by the NAS, defaults to ()
        :type exclude_types: Iterable[Type[nn.Module]], optional
        :param train_channels: flag to control whether output channels are optimized by PIT or not, defaults to True
        :type train_channels: bool, optional
        :param train_rf: flag to control whether receptive field is optimized by PIT or not, defaults to True
        :type train_rf: bool, optional
        :param train_dilation: flag to control whether dilation is optimized by PIT or not, defaults to True
        :type train_dilation: bool, optional
        """
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
    def train_channels(self) -> bool:
        """Returns True if PIT is training the output channels masks

        :return: True if PIT is training the output channels masks
        :rtype: bool
        """
        return self._train_channels

    @train_channels.setter
    def train_channels(self, value: bool):
        """Set to True to let PIT train the output channels masks

        :param value: set to True to let PIT train the output channels masks
        :type value: bool
        """
        for layer in self._target_layers:
            layer.train_channels = value
        self._train_channels = value

    @property
    def train_rf(self) -> bool:
        """Returns True if PIT is training the filters receptive fields masks

        :return: True if PIT is training the filters receptive fields masks
        :rtype: bool
        """
        return self._train_rf

    @train_rf.setter
    def train_rf(self, value: bool):
        """Set to True to let PIT train the filters receptive fields masks

        :param value: set to True to let PIT train the filters receptive fields masks
        :type value: bool
        """
        for layer in self._target_layers:
            layer.train_rf = value
        self._train_rf = value

    @property
    def train_dilation(self):
        """Returns True if PIT is training the filters dilation masks

        :return: True if PIT is training the filters dilation masks
        :rtype: bool
        """
        return self._train_dilation

    @train_dilation.setter
    def train_dilation(self, value: bool):
        """Set to True to let PIT train the filters dilation masks

        :param value: set to True to let PIT train the filters dilation masks
        :type value: bool
        """
        for layer in self._target_layers:
            layer.train_dilation = value
        self._train_dilation = value

    def _convert(self):
        """Converts the inner model, making it "NAS-able" by PIT
        """
        mod = fx.symbolic_trace(self._inner_model)
        ShapeProp(mod).propagate(self._input_example)
        self._convert_layers(mod)
        self._set_input_sizes(mod)
        mod.recompile()
        self._inner_model = mod

    def _convert_layers(self, mod: fx.GraphModule):
        """Replaces target layers (currently, only Conv1D) with their NAS-able version, while also recording the list of NAS-able
        layers for speeding up later regularization loss computations.

        Layer conversion is implemented as a reverse BFS on the model graph (starting from the output and reversing all edges).

        :param mod: a torch.fx.GraphModule with tensor shapes annotations. Those are needed to determine the sizes of PIT masks.
        :type mod: fx.GraphModule
        """
        g = mod.graph
        queue = get_output_nodes(g)
        shared_masker_queue = [None] * len(queue)
        while queue:
            n = queue.pop(0)
            shared_masker = shared_masker_queue.pop(0)
            self._rewrite_node(n, mod, shared_masker)
            shared_masker = self._update_shared_masker(n, mod, shared_masker)
            for pred in n.all_input_nodes:
                queue.append(pred)
                shared_masker_queue.append(shared_masker)

    def _rewrite_node(self, n: fx.Node, mod: fx.GraphModule, shared_masker: Optional[PITChannelMasker]):
        """Optionally rewrites a fx.GraphModule node replacing a sub-module instance with its corresponding NAS-able version

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be otpionally inserted
        :type mod: fx.GraphModule
        :param shared_masker: an optional shared channels mask derived from subsequent layers
        :type shared_masker: Optional[PITChannelMasker]
        """
        # TODO: add other NAS-able layers here
        if is_layer(n, mod, nn.Conv1d) and not self._exclude_mod(n, mod):
            self._rewrite_conv1d(n, mod, shared_masker)
        # if is_layer(n, mod, nn.Conv2d) and not self._exclude_mod(n, mod):
        #     return _rewrite_Conv2d()

    def _rewrite_conv1d(self, n: fx.Node, mod: fx.GraphModule, shared_masker: Optional[PITChannelMasker]):
        """Rewrites a fx.GraphModule node corresponding to a Conv1D layer, replacing it with a PITConv1D layer

        :param n: the node to be rewritten, corresponds to a Conv1D layer
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        :param shared_masker: an optional shared channels mask derived from subsequent layers
        :type shared_masker: Optional[PITChannelMasker]
        """
        submodule = mod.get_submodule(n.target)
        if shared_masker is not None:
            chan_masker = shared_masker
        else:
            chan_masker = PITChannelMasker(submodule.out_channels)
        new_submodule = PITConv1d(
            cast(submodule, nn.Conv1d),
            n.meta['tensor_meta'].shape[1],
            self.regularizer,
            chan_masker,
            PITTimestepMasker(submodule.kernel_size[0]),
            PITDilationMasker(submodule.kernel_size[0]),
        )
        mod.add_submodule(n.target, new_submodule)
        self._target_layers.append(new_submodule)
        return

    def _exclude_mod(self, n: fx.Node, mod: fx.GraphModule) -> bool:
        """Returns True if a submodule should be excluded from the NAS optimization, based on the names and types blacklists.

        :param n: the target node
        :type n: fx.Node
        :param mod: the parent module
        :type mod: fx.GraphModule
        :return: True if the node should be excluded
        :rtype: bool
        """
        return (type(mod.get_submodule(n.target)) in self.exclude_types) or (n.name in self.exclude_names)

    def _update_shared_masker(self, n: fx.Node, mod: fx.GraphModule, shared_masker: Optional[PITChannelMasker]
                      ) -> Optional[PITChannelMasker]:
        """Determines if the currently processed node requires that its predecessor share a common channels mask.

        :param n: the target node
        :type n: fx.Node
        :param mod: the parent module
        :type mod: fx.GraphModule
        :param shared_masker: the current shared_masker
        :type shared_masker: Optional[PITChannelMasker]
        :raises ValueError: for unsupported nodes, to avoid unexpected behaviors
        :return: the updated shared_masker
        :rtype: Optional[PITChannelMasker]
        """
        if zero_or_one_input_op(n):
            # definitely no channel sharing
            return None
        elif shared_input_size_op(n, mod):
            # modules that require multiple inputs all of the same size
            # create a new shared masker with the common n. of input channels, to be used by predecessors
            input_size = n.all_input_nodes[0].meta['tensor_meta'].shape[1]
            shared_masker = PITChannelMasker(input_size)
            return shared_masker
        else:
            raise ValueError("Unsupported node {} (op: {}, target: {})".format(n, n.op, n.target))

    def _set_input_sizes(self, mod: fx.GraphModule):
        """Determines, for each layer in the network, which preceding layer dictates its input shape.

        This is needed to correctly evaluate the cost loss function during NAS optimization.
        This pass is implemented as a forward BFS on the network graph.

        :param mod: a torch.fx.GraphModule with tensor shapes annotations.
        :type mod: fx.GraphModule
        """
        g = mod.graph
        # convert to networkx graph to have successors information, fx only gives predecessors unfortunately
        nx_graph = fx_to_nx_graph(g)
        queue = get_input_nodes(g)
        calc_dict = {}
        while queue:
            n = queue.pop(0)
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
