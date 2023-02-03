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
from .nn.module import PITModule
from .nn.features_masker import PITFeaturesMasker, PITFrozenFeaturesMasker
from flexnas.graph.annotation import add_features_calculator, add_node_properties, \
        associate_input_features
from flexnas.graph.inspection import is_layer, get_graph_outputs, is_inherited_layer, \
        get_graph_inputs, all_output_nodes
from flexnas.graph.features_calculation import ModAttrFeaturesCalculator
from flexnas.graph.utils import fx_to_nx_graph

# add new supported layers here:
pit_layer_map: Dict[Type[nn.Module], Type[PITModule]] = {
    nn.Conv1d: PITConv1d,
    nn.Conv2d: PITConv2d,
    nn.Linear: PITLinear,
    nn.BatchNorm1d: PITBatchNorm1d,
    nn.BatchNorm2d: PITBatchNorm2d,
}


class PITTracer(fx.Tracer):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, PITModule):
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
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    # create a "fake" minibatch of 1 input for shape prop
    batch_example = torch.stack([torch.rand(input_shape)] * 1, 0)
    device = next(model.parameters()).device
    ShapeProp(mod).propagate(batch_example.to(device))
    add_node_properties(mod)
    target_layers = convert_layers(mod, conversion_type, exclude_names, exclude_types)
    if conversion_type in ('autoimport', 'import'):
        fuse_conv_bn(mod)
        add_features_calculator(mod, [pit_features_calc])
        associate_input_features(mod)
        register_input_features(mod)
    mod.graph.lint()
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
    queue = get_graph_outputs(g)
    # dictionary of shared feature maskers. Used only in 'autoimport' mode.
    sm_dict = build_shared_features_map(mod)
    # the list of target layers is only used in 'import' and 'autoimport' modes.
    target_layers = []
    visited = []
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        if conversion_type == 'autoimport':
            autoimport_node(n, mod, sm_dict, exclude_names, exclude_types)
        if conversion_type in ('import', 'autoimport'):
            add_to_targets(n, mod, target_layers, exclude_names, exclude_types)
        if conversion_type == 'export':
            export_node(n, mod, exclude_names, exclude_types)
        for pred in n.all_input_nodes:
            queue.append(pred)
        visited.append(n)
    return target_layers


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
            for i in pred:
                sharing_graph.remove_edge(i, n)

    # each weakly connected component of the sharing graph must share the same features masker
    sm_dict = {}
    for c in nx.weakly_connected_components(sharing_graph):
        sm = None
        for n in c:
            # identify a node which can give us the number of features with 100% certainty
            # nodes such as flatten/squeeze etc make this necessary
            if n.meta['features_defining'] or n.meta['untouchable'] and sm is None:
                sm = PITFeaturesMasker(n.meta['tensor_meta'].shape[1])
            if n in get_graph_outputs(mod.graph):
                # distinguish the case in which the number of features must "frozen"
                # i.e. the case of output-connected components
                # this may overwrite a previously set "sm"
                sm = PITFrozenFeaturesMasker(n.meta['tensor_meta'].shape[1])
                break
        for n in c:
            sm_dict[n] = sm
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
                    exclude_types: Iterable[Type[nn.Module]]):
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
    """
    if is_layer(n, mod, tuple(pit_layer_map.keys())) and not exclude(
            n, mod, exclude_names, exclude_types):
        conv_layer_type = pit_layer_map[type(mod.get_submodule(str(n.target)))]
        conv_layer_type.autoimport(n, mod, sm_dict[n])


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
    if is_inherited_layer(n, mod, (PITModule,)):
        if exclude(n, mod, exclude_names, exclude_types):
            return
        layer = cast(PITModule, mod.get_submodule(str(n.target)))
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
    if is_inherited_layer(n, mod, (PITModule,)):
        if exclude(n, mod, exclude_names, exclude_types):
            return
        # only conv and FC, exclude BN
        if is_layer(n, mod, (PITBatchNorm1d, PITBatchNorm2d)):
            return
        target_layers.append(mod.get_submodule(str(n.target)))


def fuse_conv_bn_inplace(conv: PITModule, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, modifies A
    such that A(x) == B(A_old(x))
    """
    if not bn.track_running_stats:
        raise AttributeError("BatchNorm foldign requires track_running_stats = True")
    assert (isinstance(conv, PITConv1d) or isinstance(conv, PITConv2d))
    with torch.no_grad():
        conv.following_bn_args = {
            'eps': bn.eps,
            'momentum': bn.momentum,
            'affine': bn.affine,
            'track_running_stats': bn.track_running_stats,
        }
        conv_w = conv.weight
        conv_b = conv.bias
        bn_rm = bn.running_mean
        bn_rv = bn.running_var
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
        conv.weight.copy_(conv_w)
        if conv.bias is None:
            conv.bias = torch.nn.Parameter(conv_b)
        else:
            conv.bias.copy_(conv_b)


def fuse_conv_bn(mod: fx.GraphModule):
    """Fuses all BatchNorm layers occurring just after a PITLayer with it.
    This is needed because PIT channel masking does not work with BN just after conv.
    Also sets the "had_bn" field in PITLayers to then re-split the BN during export

    :param mod: a torch.fx.GraphModule with tensor shapes annotations.
    :type mod: fx.GraphModule
    """
    # partially taken from: https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html
    modules = dict(mod.named_modules())
    for node in mod.graph.nodes:
        if node.op != 'call_module':
            continue
        if not isinstance(node.args[0], fx.Node) or node.args[0].op != 'call_module':
            continue
        isbn1d = isinstance(modules[node.target], PITBatchNorm1d)
        prevconv1d = isinstance(modules[node.args[0].target], PITConv1d)
        isbn2d = isinstance(modules[node.target], PITBatchNorm2d)
        prevconv2d = isinstance(modules[node.args[0].target], PITConv2d)
        if (isbn1d and prevconv1d) or (isbn2d and prevconv2d):
            if len(node.args[0].users) > 1:
                raise ValueError("""Convolution followed by BN but used also by other layers.
                This layer cannot be converted by PIT""")
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            assert (isinstance(conv, PITConv1d) or isinstance(conv, PITConv2d))
            fuse_conv_bn_inplace(conv, bn)
            # next line removed because we modify inplace
            # replace_node_module(node.args[0], modules, fused_conv)
            node.replace_all_uses_with(node.args[0])
            # Now that all uses of the batch norm have been replaced, we can
            # safely remove the batch norm.
            mod.graph.erase_node(node)
    mod.delete_all_unused_submodules()


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
    if is_inherited_layer(n, mod, (PITModule,)):
        # For PIT NAS-able layers, the "active" output features are stored in the
        # out_features_eff attribute, and the binary mask is in features_mask
        sub_mod = mod.get_submodule(str(n.target))
        return ModAttrFeaturesCalculator(sub_mod, 'out_features_eff', 'features_mask')
    else:
        return None
