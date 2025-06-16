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
from typing import cast, Iterable, Type, Tuple, Optional, Dict, Any

import copy
import networkx as nx
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from .nn.conv1d import PITConv1d
from .nn.conv2d import PITConv2d
from .nn.linear import PITLinear
from .nn.batchnorm_1d import PITBatchNorm1d
from .nn.batchnorm_2d import PITBatchNorm2d
from .nn.instancenorm_1d import PITInstanceNorm1d
from .nn.prelu import PITPReLU

from .nn.module import PITModule
from .nn.features_masker import PITFeaturesMasker, PITFrozenFeaturesMasker, PITConcatFeaturesMasker
from plinio.graph.annotation import add_features_calculator, add_node_properties, \
    associate_input_features, clean_up_propagated_shapes
from plinio.graph.inspection import is_layer, get_graph_outputs, is_inherited_layer, \
    get_graph_inputs, named_leaf_modules, uniquify_leaf_modules
from plinio.graph.transformation import fuse_consecutive_layers
from plinio.graph.features_calculation import ModAttrFeaturesCalculator
from plinio.graph.utils import fx_to_nx_graph, NamedLeafModules
#following needed for correct_get_item
import numpy as np
from plinio.graph.utils import try_get_args

# add new supported layers here:
pit_layer_map: Dict[Type[nn.Module], Type[PITModule]] = {
    nn.Conv1d: PITConv1d,
    nn.Conv2d: PITConv2d,
    nn.Linear: PITLinear,
    nn.BatchNorm1d: PITBatchNorm1d,
    nn.BatchNorm2d: PITBatchNorm2d,
    nn.InstanceNorm1d: PITInstanceNorm1d,
    nn.PReLU: PITPReLU,
}


class PITTracer(fx.Tracer):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, PITModule):
            return True
        else:
            return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)


def convert(model: nn.Module, input_example: Any, conversion_type: str,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            fold_bn: bool = False,
            ) -> Tuple[nn.Module, NamedLeafModules, NamedLeafModules]:
    """Converts a nn.Module, to/from "NAS-able" PIT format

    :param model: the input nn.Module
    :type model: nn.Module
    :param input_example: an input with the same shape and type of the seed's input, used
    for symbolic tracing
    :type input_example: Any
    :param conversion_type: a string specifying the type of conversion. Supported types:
    ('import', 'autoimport', 'export')
    :type conversion_type: str
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :raises ValueError: for unsupported conversion types
    :return: the converted model, and two lists of all (or all unique) leaf modules for
    the NAS
    :rtype: Tuple[nn.Module, NamedLeafModule, NamedLeafModules]
    """

    if conversion_type not in ('import', 'autoimport', 'export'):
        raise ValueError("Unsupported conversion type {}".format(conversion_type))
    if conversion_type == 'export':
        mod = copy.deepcopy(model)
        convert_layers(mod, conversion_type, dict(), exclude_names, exclude_types, fold_bn)
    if conversion_type in ('autoimport', 'import'):
        tracer = PITTracer()
        graph = tracer.trace(model.eval())
        name = model.__class__.__name__
        mod = fx.GraphModule(tracer.root, graph, name)
        if len(get_graph_inputs(mod.graph)) > 1:
            ShapeProp(mod).propagate(*input_example)
        else:
            ShapeProp(mod).propagate(input_example)
        clean_up_propagated_shapes(mod)
        add_node_properties(mod)
        if conversion_type in ('autoimport'):
            # dictionary of shared feature maskers. Used only in 'autoimport' mode.
            sm_dict = build_shared_features_map(mod)
            convert_layers(mod, conversion_type, sm_dict, exclude_names, exclude_types, fold_bn)
        fuse_pit_modules(mod, fold_bn)
        add_features_calculator(mod, [pit_features_calc])
        associate_input_features(mod)
        register_input_features(mod)
    mod.graph.lint()
    mod.recompile()
    nlf = named_leaf_modules(mod)
    ulf = uniquify_leaf_modules(nlf)
    return mod, nlf, ulf


def convert_layers(mod: fx.GraphModule,
                   conversion_type: str,
                   sm_dict: Dict,
                   exclude_names: Iterable[str],
                   exclude_types: Iterable[Type[nn.Module]],
                   fold_bn: bool
                   ):
    """Replaces target layers with their NAS-able version, or vice versa. Layer conversion
    is implemented as a reverse BFS on the model graph.

    :param mod: a torch.fx.GraphModule with tensor shapes annotations. Those are needed to
    determine the sizes of PIT masks.
    :type mod: fx.GraphModule
    :param conversion_type: a string specifying the type of conversion
    :type conversion_type: str
    :param sm_dict: dictionary associating each fx.Node to a shared feature masker
    :type sm_dict: Dict
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :param fold_bn: flag to fold the bn layer into the linear/conv layer
    :type fold_bn: bool
    """
    g = mod.graph
    queue = get_graph_outputs(g)
    visited = []
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        if conversion_type == 'autoimport':
            autoimport_node(n, mod, sm_dict, exclude_names, exclude_types, fold_bn)
        if conversion_type == 'export':
            export_node(n, mod, exclude_names, exclude_types)
        for pred in n.all_input_nodes:
            queue.append(pred)
        visited.append(n)
    return


def build_shared_features_map(mod: fx.GraphModule) -> Dict[fx.Node, PITFeaturesMasker]:
    """Create a map from fx.Node instances to instances of PITFeaturesMasker to be used by PIT
    to optimize the number of features of that node. Handles the sharing of masks among
    multiple nodes.

    :param mod: the fx-converted GraphModule
    :type mod: fx.GraphModule
    :return: a map (node -> feature masker)
    :rtype: Dict[fx.Node, PITFeaturesMasker]
    """
    # build a compatibility graph ("sharing graph") with paths between all nodes that must share
    # the same features masker. This is obtained by taking the original NN graph and removing
    # incoming edges to nodes whose output features are not dependent on the input features
    sharing_graph = fx_to_nx_graph(mod.graph)
    for n in sharing_graph.nodes:
        n = cast(fx.Node, n)
        if n.meta['untouchable'] or n.meta['features_concatenate'] or n.meta['features_defining']:
            # remove all incoming edges to this node from the "shared features graph"
            pred = list(sharing_graph.predecessors(n))
            if n.meta['features_concatenate']:
                n.meta['predecessors'] = pred
            for i in pred:
                sharing_graph.remove_edge(i, n)

    # handle the case of a forward function with multiple outputs (returned as a tuple or list) with
    # possibly independent shapes. In this case, the graph will contain a final output node that is
    # difficult to treat and we remove in this step, treating each single output independently.
    nodes_to_remove = []
    for n in sharing_graph.nodes:
        if n in get_graph_outputs(mod.graph) and len(n.meta['tensor_meta']) > 1:
            # tag each predecessor as output-connected
            pred = list(sharing_graph.predecessors(n))
            for i in pred:
                i.meta['output_connected'] = True
            # add the node to the removal list
            nodes_to_remove.append(n)
    for n in nodes_to_remove:
        sharing_graph.remove_node(n)

    # each weakly connected component of the sharing graph must share the same features masker
    sm_dict = {}
    for c in nx.weakly_connected_components(sharing_graph):
        sm = None
        for n in c:
            # identify a node which can give us the number of features with 100% certainty
            # such as a convolution. Nodes such as flatten/squeeze/view/etc make this necessary
            if n.meta['features_defining'] or n.meta['untouchable'] and sm is None:
                # distinguish the case in which the number of features must "frozen"
                # i.e. the case of input-connected or output-connected components,
                if (
                    any(n in get_graph_inputs(mod.graph) for n in c) or
                    any(n in get_graph_outputs(mod.graph) for n in c) or
                    any(n.meta.get('output_connected', False) for n in c)
                ):
                    sm = PITFrozenFeaturesMasker(n.meta['tensor_meta'].shape[1])
                else:
                    sm = PITFeaturesMasker(n.meta['tensor_meta'].shape[1])
                break
        for n in c:
            sm_dict[n] = sm
    for c in nx.weakly_connected_components(sharing_graph):
        for n in c:
            if n.meta['features_concatenate']:
                input_sm = [sm_dict[ni] for ni in n.meta['predecessors']]
                new_sm = PITConcatFeaturesMasker(input_sm)
                for n in c:
                    sm_dict[n] = new_sm
                break
    return sm_dict


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


def autoimport_node(n: fx.Node, mod: fx.GraphModule,
                    sm_dict: Dict[fx.Node, PITFeaturesMasker],
                    exclude_names: Iterable[str],
                    exclude_types: Iterable[Type[nn.Module]],
                    fold_bn: bool
                    ):
    """Possibly rewrites a fx.GraphModule node replacing a sub-module instance corresponding to a
    standard nn.Module with its corresponding NAS-able version.

    :param n: the node to be rewritten
    :type n: fx.Node
    :param mod: the parent module, where the new node has to be optionally inserted
    :type mod: fx.GraphModule
    :param sm_dict: the dictionary containing the shared feature maskers for all nodes
    :type sm_dict: Dict[fx.Node, PITFeaturesMasker]
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :param fold_bn: flag to fold the bn layer into the linear/conv layer
    :type fold_bn: bool
    """
    if is_layer(n, mod, tuple(pit_layer_map.keys())) and not exclude(
            n, mod, exclude_names, exclude_types):
        conv_layer_type = pit_layer_map[type(mod.get_submodule(str(n.target)))]
        conv_layer_type.autoimport(n, mod, sm_dict[n], fold_bn)


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
    #convert getitem before convolutional layers, modifying the slice
    if n.meta.get('features_slicing'):
        correct_get_item(n, mod)

    if is_inherited_layer(n, mod, (PITModule,)):
        if exclude(n, mod, exclude_names, exclude_types):
            return
        layer = cast(PITModule, mod.get_submodule(str(n.target)))
        layer.export(n, mod)


def remove_bn_inplace(lin: nn.Module, bn: nn.Module, fold: bool):
    """
    Removes BN layer followin linear layers. If fold is True, the BN layer is folded into the linear
    layer, otherwise, it is just added as a field of the linear layer.
    """
    assert (isinstance(lin, PITConv1d) or isinstance(lin, PITConv2d) or isinstance(lin, PITLinear))
    assert (isinstance(bn, nn.BatchNorm1d) or isinstance(bn, nn.BatchNorm2d) or isinstance(bn, nn.InstanceNorm1d))
    if not bn.track_running_stats:
        raise AttributeError("BatchNorm folding requires track_running_stats = True")
    with torch.no_grad():
        lin.bn = copy.deepcopy(bn)
        if fold:
            conv_w = lin.weight
            conv_b = lin.bias
            bn_rm = cast(torch.Tensor, bn.running_mean)
            bn_rv = cast(torch.Tensor, bn.running_var)
            bn_w = bn.weight
            bn_b = bn.bias
            if conv_b is None:
                conv_b = torch.zeros_like(bn_rm)
            if bn_w is None:
                bn_w = torch.ones_like(bn_rm)
            if bn_b is None:
                bn_b = torch.zeros_like(bn_rm)
            bn_var_rsqrt = torch.rsqrt(bn_rv + bn.eps)
            conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
            conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
            lin.weight.copy_(conv_w)
            if lin.bias is None:
                lin.bias = torch.nn.Parameter(conv_b)
            else:
                lin.bias.copy_(conv_b)


def fuse_pit_modules(mod: fx.GraphModule, fold_bn: bool) -> None:
    """Fuse sequences of layers as required by PIT. Namely: Conv-BN and Linear-BN
    :param mod: the parent module
    :type mod: fx.GraphModule
    :param fold_bn: flag to fold the bn layer into the linear/conv layer
    :type fold_bn: bool
    """
    fuse_consecutive_layers(mod, PITConv1d, nn.BatchNorm1d,
                            lambda x, y: remove_bn_inplace(x, y, fold_bn))
    fuse_consecutive_layers(mod, PITConv2d, nn.BatchNorm2d,
                            lambda x, y: remove_bn_inplace(x, y, fold_bn))
    fuse_consecutive_layers(mod, PITLinear, nn.BatchNorm1d,
                            lambda x, y: remove_bn_inplace(x, y, fold_bn))

    fuse_consecutive_layers(mod, PITLinear, nn.InstanceNorm1d,
                            lambda x, y: remove_bn_inplace(x, y, fold_bn))
    fuse_consecutive_layers(mod, PITConv1d, nn.InstanceNorm1d,
                            lambda x, y: remove_bn_inplace(x, y, fold_bn))
    fuse_consecutive_layers(mod, PITConv2d, nn.InstanceNorm2d,
                            lambda x, y: remove_bn_inplace(x, y, fold_bn))


def register_input_features(mod: fx.GraphModule):
    for n in mod.graph.nodes:
        if is_inherited_layer(n, mod, (PITModule,)):
            sub_mod = cast(PITModule, mod.get_submodule(str(n.target)))
            fc = n.meta['input_features_set_by'].meta['features_calculator']
            sub_mod.input_features_calculator = fc


def pit_features_calc(n: fx.Node, mod: fx.GraphModule) -> Optional[ModAttrFeaturesCalculator]:
    """Sets the feature calculator for a PIT node

    :param n: node
    :type n: fx.Node
    :param mod: the parent module
    :type mod: fx.GraphModule
    :return: optional feature calculator object for PIT node
    :rtype: ModAttrFeaturesCalculator
    """
    if is_inherited_layer(n, mod, (PITModule,)) and not (is_inherited_layer(n, mod, (PITBatchNorm1d, PITBatchNorm2d, PITInstanceNorm1d, PITPReLU))):
        # For PIT NAS-able layers, the "active" output features are stored in the
        # out_features_eff attribute, and the binary mask is in features_mask
        sub_mod = mod.get_submodule(str(n.target))
        return ModAttrFeaturesCalculator(sub_mod, 'out_features_eff', 'features_mask')
    else:
        return None

def correct_get_item(n: fx.Node, mod: fx.GraphModule):
    """
    fix indexes of an indexing operation, according to pruned channels of previous layer
    :param n: node
    :type n: fx.Node
    :param mod: the parent module
    :type mod: fx.GraphModule
    :return: optional feature calculator object for PIT node
    :rtype: ModAttrFeaturesCalculator
    """
    """
    retrieve the mask of the previous layers, subtract -1 so we separate pruned channels from
    substitute all 0s (the surviving channels) with a growing number, starting from 0,
    that will be their future index. Then apply previous slice and remove -1, obtaining
    indices of the surviving channels of this slice.
    The indices could be translated to a slice if the original step is 1 or -1 only.
    We replace the slice with the list of indices, because indexing with a list is supported for a torch tensor.
    """
    indices = try_get_args(n, mod, 1, 'dim', None) #n.args[1]
    if isinstance(indices[1], torch.fx.node.Node):
        indices = (indices[0], getattr(mod, indices[1].target), indices[2:])

    bin_alpha = n.all_input_nodes[0].meta['features_calculator'].features_mask #indexing has always one input only, no need to call associate_input_features
    #convert 0 to -1 and 1 to 0, to avoid confusion with indices
    bin_alpha=np.array(bin_alpha.tolist(),dtype=int)-1
    #replace non pruned channels with their future index, equal to the number of survived channels before them
    bin_alpha[np.where(bin_alpha!=-1)]=list(range(sum(bin_alpha!=-1))) #idx_alpha = (np.cumsum(bin_alpha)*bin_alpha)-1
    #select slice of the mask
    selected = bin_alpha[indices[1]]
    #retrieve non pruned channels indexes only
    selected = selected[selected!=-1]

    n.args=(n.args[0],(indices[0], selected.tolist()))