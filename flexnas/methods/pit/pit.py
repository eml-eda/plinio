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

from typing import cast, Any, List, Tuple, Type, Iterable, Optional, Dict
import math
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from flexnas.methods.dnas_base import DNAS
from flexnas.methods.pit.pit_tracer import PITTracer
from .pit_conv1d import PITConv1d
from .pit_channel_masker import PITChannelMasker
from .pit_timestep_masker import PITTimestepMasker
from .pit_dilation_masker import PITDilationMasker
from flexnas.utils import model_graph
from flexnas.utils.features_calculator import ConstFeaturesCalculator, FeaturesCalculator, \
    LinearFeaturesCalculator, ListReduceFeaturesCalculator, ModAttrFeaturesCalculator


class PIT(DNAS):
    """A class that wraps a nn.Module with the functionality of the PIT NAS tool

    :param model: the inner nn.Module instance optimized by the NAS
    :type model: nn.Module
    :param input_example: an example of input tensor, required for symbolic tracing without batch
    size that is forced before the ShapePropagate
    :type input_example: torch.Tensor
    :param regularizer: a string defining the type of cost regularizer, defaults to 'size'
    :type regularizer: Optional[str], optional
    :param autoconvert_layers: should the constructor try to autoconvert NAS-able layers,
    defaults to True
    :type autoconvert_layers: bool, optional
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    when auto-converting defaults to ()
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :param train_channels: flag to control whether output channels are optimized by PIT or not,
    defaults to True
    :type train_channels: bool, optional
    :param train_rf: flag to control whether receptive field is optimized by PIT or not, defaults
    to True
    :type train_rf: bool, optional
    :param train_dilation: flag to control whether dilation is optimized by PIT or not, defaults
    to True
    :type train_dilation: bool, optional
    """
    def __init__(
            self,
            model: nn.Module,
            input_example: torch.Tensor,
            regularizer: str = 'size',
            autoconvert_layers: bool = True,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            train_channels: bool = True,
            train_rf: bool = True,
            train_dilation: bool = True):
        super(PIT, self).__init__(regularizer, exclude_names, exclude_types)
        self._input_example = input_example
        self._autoconvert_layers = autoconvert_layers
        self._target_layers = []
        self._inner_model = self._convert(model)
        # after conversion to make sure they are applied to all layers
        self.train_channels = train_channels
        self.train_rf = train_rf
        self.train_dilation = train_dilation
        if self.regularizer == 'size':
            self.get_regularization_loss = self.get_size
        elif self.regularizer == 'macs':
            self.get_regularization_loss = self.get_macs

    def forward(self, *args: Any) -> torch.Tensor:
        """Forward function for the DNAS model. Simply invokes the inner model's forward

        :return: the output tensor
        :rtype: torch.Tensor
        """
        return self._inner_model.forward(*args)

    def supported_regularizers(self) -> Tuple[str, ...]:
        return ('size', 'macs')

    def get_size(self) -> torch.Tensor:
        """Computes the total number of parameters of all NAS-able layers

        :return: the total number of parameters
        :rtype: torch.Tensor
        """
        size = torch.tensor(0, dtype=torch.float32)
        # size = torch.tensor(0)
        for layer in self._target_layers:
            # size += layer.get_size()
            size = size + layer.get_size()
        return size

    def get_macs(self) -> torch.Tensor:
        """Computes the total number of MACs in all NAS-able layers

        :return: the total number of MACs
        :rtype: torch.Tensor
        """
        macs = torch.tensor(0, dtype=torch.float32)
        # macs = torch.tensor(0)
        for layer in self._target_layers:
            # macs += layer.get_macs()
            macs = macs + layer.get_macs()
        return macs

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

    def _convert(self, inner_model: nn.Module) -> nn.Module:
        """Converts the inner model, making it "NAS-able" by PIT

        :param inner_model: the original inner_model
        :type inner_model: nn.Module
        :return: the converted inner_model
        :rtype: nn.Module
        """
        tracer = PITTracer()
        graph = tracer.trace(inner_model)
        name = inner_model.__class__.__name__
        mod = fx.GraphModule(tracer.root, graph, name)
        # create a "fake" minibatch of 32 inputs for shape prop
        batch_example = torch.stack([self._input_example] * 32, 0)
        # ShapeProp(mod).propagate(self._input_example)
        ShapeProp(mod).propagate(batch_example)
        self._convert_layers(mod)
        self._set_input_features(mod)
        mod.recompile()
        return mod

    def _convert_layers(self, mod: fx.GraphModule):
        """Replaces target layers (currently, only Conv1D) with their NAS-able version, while also
        recording the list of NAS-able layers for speeding up later regularization loss
        computations.

        Layer conversion is implemented as a reverse BFS on the model graph (starting from the
        output and reversing all edges).

        :param mod: a torch.fx.GraphModule with tensor shapes annotations. Those are needed to
        determine the sizes of PIT masks.
        :type mod: fx.GraphModule
        """
        g = mod.graph
        queue = model_graph.get_output_nodes(g)
        shared_masker_queue: List[Optional[PITChannelMasker]] = [None] * len(queue)
        visited = []
        while queue:
            n = queue.pop(0)
            shared_masker = shared_masker_queue.pop(0)
            if n not in visited:
                if self._autoconvert_layers:
                    self._rewrite_node(n, mod, shared_masker)
                    shared_masker = self._update_shared_masker(n, mod, shared_masker)
                else:
                    shared_masker = None
                self._add_to_targets(n, mod)
                for pred in n.all_input_nodes:
                    queue.append(pred)
                    shared_masker_queue.append(shared_masker)
                visited.append(n)

    def _rewrite_node(self, n: fx.Node, mod: fx.GraphModule, sm: Optional[PITChannelMasker]):
        """Optionally rewrites a fx.GraphModule node replacing a sub-module instance with its
        corresponding NAS-able version

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be optionally inserted
        :type mod: fx.GraphModule
        :param shared_masker: an optional shared channels mask derived from subsequent layers
        :type shared_masker: Optional[PITChannelMasker]
        """
        # TODO: add other NAS-able layers here
        if model_graph.is_layer(n, mod, nn.Conv1d) and not self._exclude_mod(n, mod):
            self._rewrite_conv1d(n, mod, sm)
        # if is_layer(n, mod, nn.Conv2d) and not self._exclude_mod(n, mod):
        #     return _rewrite_Conv2d()

    def _rewrite_conv1d(self, n: fx.Node, mod: fx.GraphModule, sm: Optional[PITChannelMasker]):
        """Rewrites a fx.GraphModule node corresponding to a Conv1D layer, replacing it with a
        PITConv1D layer

        :param n: the node to be rewritten, corresponds to a Conv1D layer
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        :param shared_masker: an optional shared channels mask derived from subsequent layers
        :type shared_masker: Optional[PITChannelMasker]
        """
        submodule = cast(nn.Conv1d, mod.get_submodule(str(n.target)))
        if sm is not None:
            chan_masker = sm
        else:
            chan_masker = PITChannelMasker(submodule.out_channels)
        new_submodule = PITConv1d(
            submodule,
            n.meta['tensor_meta'].shape[1],
            chan_masker,
            PITTimestepMasker(submodule.kernel_size[0]),
            PITDilationMasker(submodule.kernel_size[0]),
        )
        mod.add_submodule(str(n.target), new_submodule)
        return

    def _exclude_mod(self, n: fx.Node, mod: fx.GraphModule) -> bool:
        """Returns True if a submodule should be excluded from the NAS optimization, based on the
        names and types blacklists.

        :param n: the target node
        :type n: fx.Node
        :param mod: the parent module
        :type mod: fx.GraphModule
        :return: True if the node should be excluded
        :rtype: bool
        """
        exc_type = type(mod.get_submodule(str(n.target))) in self.exclude_types
        return exc_type or (n.name in self.exclude_names)

    def _add_to_targets(self, n: fx.Node, mod: fx.GraphModule):
        """Optionally adds a layer to the list of NAS target layers

        :param n: the node to be checked for monitoring
        :type n: fx.Node
        :param mod: the parent module
        :type mod: fx.GraphModule
        """
        # TODO: add other NAS-able layers here
        if model_graph.is_layer(n, mod, PITConv1d):
            self._target_layers.append(mod.get_submodule(str(n.target)))
        # etc
        # elif model_graph.is_layer(n, mod, PITConv2d):
            # self._target_layers.append(mod)

    def _update_shared_masker(self, n: fx.Node, mod: fx.GraphModule,
                              sm: Optional[PITChannelMasker]) -> Optional[PITChannelMasker]:
        """Determines if the currently processed node requires that its predecessor share a common
        channels mask.

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
        if model_graph.is_zero_or_one_input_op(n):
            # definitely no channel sharing
            return None
        elif model_graph.is_untouchable_op(n):
            return None
        elif model_graph.is_shared_input_features_op(n, mod):
            # modules that require multiple inputs all of the same size
            # create a new shared masker with the common n. of input channels, to be used by
            # predecessors
            input_size = n.all_input_nodes[0].meta['tensor_meta'].shape[1]
            shared_masker = PITChannelMasker(input_size)
            return shared_masker
        else:
            raise ValueError("Unsupported node {} (op: {}, target: {})".format(n, n.op, n.target))

    def _set_input_features(self, mod: fx.GraphModule):
        """Determines, for each layer in the network, which preceding layer dictates its input
        number of features.

        This is needed to correctly evaluate the regularization loss function during NAS
        optimization.
        This pass is implemented as a forward BFS on the network graph.

        :param mod: a torch.fx.GraphModule with tensor shapes annotations.
        :type mod: fx.GraphModule
        """
        g = mod.graph
        # convert to networkx graph to have successors information, fx only gives predecessors
        # unfortunately
        nx_graph = model_graph.fx_to_nx_graph(g)
        queue = model_graph.get_input_nodes(g)
        calc_dict = {}
        while queue:
            n = queue.pop(0)
            # skip nodes for which predecessors have not yet been processed completely, we'll come
            # back to them later
            if len(n.all_input_nodes) > 0:
                for i in n.all_input_nodes:
                    if i not in calc_dict:
                        return
            self._set_input_features_calculator(n, mod, calc_dict)
            self._update_output_features_calculator(n, mod, calc_dict)
            for succ in nx_graph.successors(n):
                queue.append(succ)

    @staticmethod
    def _set_input_features_calculator(n: fx.Node, mod: fx.GraphModule,
                                       calc_dict: Dict[fx.Node, FeaturesCalculator]):
        """Set the input features calculator attribute for NAS-able layers (currently just Conv1D)

        :param n: the target node
        :type n: fx.Node
        :param mod: the parent module
        :type mod: fx.GraphModule
        :param calc_dict: a dictionary containing output features calculators for all preceding
        nodes in the network
        :type calc_dict: Dict[fx.Node, FeaturesCalculator]
        """
        # TODO: add other NAS-able layers here
        if model_graph.is_inherited_layer(n, mod, nn.Conv1d):
            prev = n.all_input_nodes[0]  # a Conv layer always has a single input
            sub_mod = mod.get_submodule(str(n.target))
            sub_mod._input_features_calculator = calc_dict[prev]
            # features = sub_mod._input_features_calculator.features

    @staticmethod
    def _update_output_features_calculator(n: fx.Node, mod: fx.GraphModule,
                                           calc_dict: Dict[fx.Node, FeaturesCalculator]):
        """Update the dictionary containing output features calculators for all nodes in the network

        :param n: the target node
        :type n: fx.Node
        :param mod: the parent module
        :type mod: fx.GraphModule
        :param calc_dict: a partially filled dictionary of output features calculators for all
        nodes in the network
        :type calc_dict: Dict[fx.Node, FeaturesCalculator]
        :raises ValueError: when the target node op is not supported
        """
        # TODO: add other NAS-able layers here
        if model_graph.is_layer(n, mod, PITConv1d):
            # For PITConv1D layers, the "active" output features are stored in the out_channels_eff
            # attribute
            sub_mod = mod.get_submodule(str(n.target))
            calc_dict[n] = ModAttrFeaturesCalculator(sub_mod, 'out_channels_eff')
        elif model_graph.is_flatten(n, mod):
            # For flatten ops, the output features are computed as: input_features * spatial_size
            # note that this is NOT simply equal to the output shape if the preceding layer is a
            # NAS-able one, for which some features # could be de-activated
            ifc = calc_dict[n.all_input_nodes[0]]
            input_shape = n.all_input_nodes[0].meta['tensor_meta'].shape
            spatial_size = math.prod(input_shape[2:])
            calc_dict[n] = LinearFeaturesCalculator(ifc, spatial_size)
        elif model_graph.is_concatenate(n, mod):
            # for concatenation ops the number of output features is the sum of the output features
            # of preceding layers as for flatten, this is NOT equal to the input shape of this
            # layer, when one or more predecessors are NAS-able
            # TODO: this assumes that concatenation is always on the features axis.
            # Not always true. Fix.
            ifc = ListReduceFeaturesCalculator([calc_dict[_] for _ in n.all_input_nodes], sum)
            calc_dict[n] = ifc
        elif model_graph.is_shared_input_features_op(n, mod):
            # for nodes that require identical number of features in all their inputs (e.g., add)
            # we simply assume that we can take any of the output features calculators from
            # predecessors
            # this is enforced for NAS-able layers by the use of shared maskers (see above)
            calc_dict[n] = calc_dict[n.all_input_nodes[0]]
        elif model_graph.is_features_defining_op(n, mod):
            # these are "static" (i.e., non NAS-able) nodes that alter the number of output
            # features, and hence the number of input features of subsequent layers
            calc_dict[n] = ConstFeaturesCalculator(n.meta['tensor_meta'].shape[1])
        elif model_graph.is_features_propagating_op(n, mod):
            # these are nodes that have a single input and n. output features == n. input features
            # so, we just propagate forward the features calculator of the input
            calc_dict[n] = calc_dict[n.all_input_nodes[0]]  # they all have a single input
        else:
            raise ValueError("Unsupported node {} (op: {}, target: {})".format(n, n.op, n.target))
        return

    def arch_summary(self):
        for name, layer in self._inner_model.named_modules():
            if isinstance(layer, PITConv1d):
                print("{}: (Cout = {}, K = {}, D = {})".format(
                    name,
                    layer.out_channels_opt,
                    layer.kernel_size_opt,
                    layer.dilation_opt
                ))
