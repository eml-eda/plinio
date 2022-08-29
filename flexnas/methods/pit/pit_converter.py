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
from typing import cast, List, Iterable, Type, Tuple, Optional, Dict
import math
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from flexnas.methods.pit.pit_conv1d import PITConv1d
from flexnas.methods.pit.pit_conv2d import PITConv2d
from flexnas.methods.pit.pit_linear import PITLinear
from .pit_layer import PITLayer
from .pit_features_masker import PITFeaturesMasker, PITFrozenFeaturesMasker
from flexnas.utils import model_graph
from flexnas.utils.features_calculator import ConstFeaturesCalculator, FeaturesCalculator, \
    FlattenFeaturesCalculator, ConcatFeaturesCalculator, ModAttrFeaturesCalculator

# add new supported layers here:
# TODO: can we fill this automatically based on classes that inherit from PITLayer?
pit_layer_map: Dict[Type[nn.Module], Type[PITLayer]] = {
    nn.Conv1d: PITConv1d,
    nn.Conv2d: PITConv2d,
    nn.Linear: PITLinear,
}


class PITTracer(fx.Tracer):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, PITLayer):
            return True
        else:
            return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)


def convert(model: nn.Module, input_shape: Tuple[int, ...], conversion_type: str,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            ) -> Tuple[nn.Module, List]:
    """Converts a nn.Module, to/from "NAS-able" PIT format

    :param model: the input nn.Module
    :type model: nn.Module
    :param input_shape: the shape of an input tensor, without batch size, required for symbolic
    tracing
    :type input_shape: Tuple[int, ...]
    :param conversion_type: a string specifying the type of conversion. Supported types:
    ('import', 'autoimport', 'export')
    :type conversion_type: str
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :raises ValueError: for unsupported conversion types
    :return: the converted model, and the list of target layers for the NAS (only for imports)
    :rtype: Tuple[nn.Module, List]
    """

    if conversion_type not in ('import', 'autoimport', 'export'):
        raise ValueError("Unsupported conversion type {}".format(conversion_type))

    tracer = PITTracer()
    graph = tracer.trace(model)
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    # create a "fake" minibatch of 32 inputs for shape prop
    batch_example = torch.stack([torch.rand(input_shape)] * 32, 0)
    ShapeProp(mod).propagate(batch_example)
    target_layers = convert_layers(mod, conversion_type, exclude_names, exclude_types)
    if conversion_type in ('autoimport', 'import'):
        set_input_features(mod)
    mod.recompile()
    return mod, target_layers


def convert_layers(mod: fx.GraphModule,
                   conversion_type: str,
                   exclude_names: Iterable[str],
                   exclude_types: Iterable[Type[nn.Module]],
                   ) -> List[nn.Module]:
    """Replaces target layers with their NAS-able version, or vice versa, while also
    recording the list of NAS-able layers for speeding up later regularization loss
    computations. Layer conversion is implemented as a reverse BFS on the model graph.

    :param mod: a torch.fx.GraphModule with tensor shapes annotations. Those are needed to
    determine the sizes of PIT masks.
    :type mod: fx.GraphModule
    :param conversion_type: a string specifying the type of conversion
    :type conversion_type: str
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :return: the list of target layers that will be optimized by the NAS
    :rtype: List[nn.Module]
    """
    g = mod.graph
    queue = model_graph.get_output_nodes(g)
    # the shared_masker_queue is only used in 'autoimport' mode.
    # initialied with Frozen maskers to ensure output layers are never trainable
    shared_masker_queue: List[Optional[PITFeaturesMasker]] = [
        PITFrozenFeaturesMasker(n.meta['tensor_meta'].shape[1]) for n in queue]
    # the list of target layers is only used in 'import' and 'autoimport' modes. Empty for export
    target_layers = []
    visited = []
    while queue:
        n = queue.pop(0)
        shared_masker = shared_masker_queue.pop(0)
        if n not in visited:
            if conversion_type == 'autoimport':
                autoimport_node(n, mod, shared_masker, exclude_names, exclude_types)
            if conversion_type == 'export':
                export_node(n, mod, exclude_names, exclude_types)
            if conversion_type in ('import', 'autoimport'):
                add_to_targets(n, mod, target_layers, exclude_names, exclude_types)
            if conversion_type == 'autoimport':
                shared_masker = update_shared_masker(n, mod, shared_masker)
            else:
                shared_masker = None

            for pred in n.all_input_nodes:
                queue.append(pred)
                shared_masker_queue.append(shared_masker)
            visited.append(n)
    return target_layers


def exclude(n: fx.Node, mod: fx.GraphModule,
            exclude_names: Iterable[str],
            exclude_types: Iterable[Type[nn.Module]],
            ) -> bool:
    """Returns True if a submodule should be excluded from the NAS optimization, based on the
    names and types blacklists.

    :param n: the target node
    :type n: fx.Node
    :param mod: the parent module
    :type mod: fx.GraphModule
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :return: True if the node should be excluded
    :rtype: bool
    """
    exc_type = type(mod.get_submodule(str(n.target))) in exclude_types
    # note that we use n.target and not n.name because it is consistent with the names obtained
    # by named_modules()
    return exc_type or (str(n.target) in exclude_names)


def autoimport_node(n: fx.Node, mod: fx.GraphModule, sm: Optional[PITFeaturesMasker],
                    exclude_names: Iterable[str],
                    exclude_types: Iterable[Type[nn.Module]]):
    """Rewrites a fx.GraphModule node replacing a sub-module instance corresponding to a standard
    nn.Module with its corresponding NAS-able version

    :param n: the node to be rewritten
    :type n: fx.Node
    :param mod: the parent module, where the new node has to be optionally inserted
    :type mod: fx.GraphModule
    :param sm: an optional shared features mask derived from subsequent layers
    :type sm: Optional[PITFeaturesMasker]
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    """
    if model_graph.is_layer(n, mod, tuple(pit_layer_map.keys())):
        if exclude(n, mod, exclude_names, exclude_types):
            return
        conv_layer_type = pit_layer_map[type(mod.get_submodule(str(n.target)))]
        conv_layer_type.autoimport(n, mod, sm)


def export_node(n: fx.Node, mod: fx.GraphModule,
                exclude_names: Iterable[str],
                exclude_types: Iterable[Type[nn.Module]]):
    """Rewrites a fx.GraphModule node replacing a sub-module instance corresponding to a NAS-able
    layer with its original nn.Module counterpart

    :param n: the node to be rewritten
    :type n: fx.Node
    :param mod: the parent module, where the new node has to be optionally inserted
    :type mod: fx.GraphModule
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    """
    if model_graph.is_inherited_layer(n, mod, (PITLayer,)):
        if exclude(n, mod, exclude_names, exclude_types):
            return
        layer = cast(PITLayer, mod.get_submodule(str(n.target)))
        layer.export(n, mod)


def add_to_targets(n: fx.Node, mod: fx.GraphModule, target_layers: List[nn.Module],
                   exclude_names: Iterable[str],
                   exclude_types: Iterable[Type[nn.Module]]):
    """Optionally adds the layer corresponding to a torch.fx.Node to the list of NAS target
    layers

    :param n: the node to be added
    :type n: fx.Node
    :param mod: the parent module
    :type mod: fx.GraphModule
    :param conversion_type: a string specifying the type of conversion
    :type conversion_type: str
    :param target_layers: the list of current target layers
    :type target_layers: List[nn.Module]
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    """
    if model_graph.is_inherited_layer(n, mod, (PITLayer,)):
        if exclude(n, mod, exclude_names, exclude_types):
            return
        target_layers.append(mod.get_submodule(str(n.target)))


def update_shared_masker(n: fx.Node, mod: fx.GraphModule,
                         sm: Optional[PITFeaturesMasker]) -> Optional[PITFeaturesMasker]:
    """Determines if the currently processed node requires that its predecessor share a common
    features mask.

    :param n: the target node
    :type n: fx.Node
    :param mod: the parent module
    :type mod: fx.GraphModule
    :param sm: the optional input shared features masker to propagate from subsequent layers
    :type sm: Optional[PITChannelMasker]
    :raises ValueError: for unsupported nodes, to avoid unexpected behaviors
    :return: the updated shared_masker
    :rtype: Optional[PITChannelMasker]
    """
    if model_graph.is_zero_or_one_input_op(n):
        if model_graph.is_features_defining_op(n, mod):
            # this op defines its output features, no propagation
            return None
        else:
            # this op has cin = cout. Return the masker received as input
            return sm
    elif model_graph.is_untouchable_op(n):
        return None
    elif model_graph.is_features_concatenate(n, mod):
        # if we concatenate over features, we don't need to share the mask
        return None
    elif model_graph.is_shared_input_features_op(n, mod):
        # modules that require multiple inputs all of the same size
        # create a new shared masker with the common n. of input channels, to be used by
        # predecessors
        if sm is None or model_graph.is_features_defining_op(n, mod):
            input_size = n.all_input_nodes[0].meta['tensor_meta'].shape[1]
            shared_masker = PITFeaturesMasker(input_size)
        else:
            shared_masker = sm
        return shared_masker
    else:
        raise ValueError("Unsupported node {} (op: {}, target: {})".format(n, n.op, n.target))


def set_input_features(mod: fx.GraphModule):
    """Determines, for each layer in the network, which preceding layer dictates its input
    number of features.

    This is needed to correctly evaluate the regularization loss function during NAS
    optimization. This pass is implemented as a forward BFS on the network graph.

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
        skip_flag = False
        if len(n.all_input_nodes) > 0:
            for i in n.all_input_nodes:
                if i not in calc_dict:
                    skip_flag = True
        if skip_flag:
            continue
        set_input_features_calculator(n, mod, calc_dict)
        update_output_features_calculator(n, mod, calc_dict)
        for succ in nx_graph.successors(n):
            queue.append(succ)


def set_input_features_calculator(n: fx.Node, mod: fx.GraphModule,
                                  calc_dict: Dict[fx.Node, FeaturesCalculator]):
    """Set the input features calculator attribute for NAS-able layers

    :param n: the target node
    :type n: fx.Node
    :param mod: the parent module
    :type mod: fx.GraphModule
    :param calc_dict: a dictionary containing output features calculators for all preceding
    nodes in the network
    :type calc_dict: Dict[fx.Node, FeaturesCalculator]
    """
    if model_graph.is_inherited_layer(n, mod, (PITLayer,)):
        prev = n.all_input_nodes[0]  # our NAS-able layers always have a single input (for now)
        sub_mod = cast(PITLayer, mod.get_submodule(str(n.target)))
        sub_mod.input_features_calculator = calc_dict[prev]


def update_output_features_calculator(n: fx.Node, mod: fx.GraphModule,
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
    if model_graph.is_inherited_layer(n, mod, (PITLayer,)):
        # For PIT NAS-able layers, the "active" output features are stored in the out_features_eff
        # attribute, and the binary mask is in features_mask
        sub_mod = mod.get_submodule(str(n.target))
        calc_dict[n] = ModAttrFeaturesCalculator(sub_mod, 'out_features_eff', 'features_mask')
    elif model_graph.is_flatten(n, mod):
        # For flatten ops, the output features are computed as: input_features * spatial_size
        # note that this is NOT simply equal to the output shape if the preceding layer is a
        # NAS-able one, for which some features # could be de-activated
        ifc = calc_dict[n.all_input_nodes[0]]
        input_shape = n.all_input_nodes[0].meta['tensor_meta'].shape
        spatial_size = math.prod(input_shape[2:])
        calc_dict[n] = FlattenFeaturesCalculator(ifc, spatial_size)
    elif model_graph.is_concatenate(n, mod):
        if model_graph.is_features_concatenate(n, mod):
            # for concatenation over the features axis the number of output features is the sum
            # of the output features of preceding layers as for flatten, this is NOT equal to the
            # input shape of this layer, when one or more predecessors are NAS-able
            ifc = ConcatFeaturesCalculator([calc_dict[_] for _ in n.all_input_nodes])
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
