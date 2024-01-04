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
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *----------------------------------------------------------------------------*

import copy
import operator
from typing import cast, Iterable, Type, Tuple, Optional, Dict, Callable, Any, Union
import networkx as nx
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from plinio.methods.mps.nn import MPSLinear, MPSConv2d, MPSIdentity, \
    MPSModule, MPSAdd
from plinio.graph.annotation import add_features_calculator, add_node_properties, \
    associate_input_features, add_single_node_properties
from plinio.graph.inspection import is_layer, get_graph_outputs, is_inherited_layer, \
    get_graph_inputs, is_function, named_leaf_modules, uniquify_leaf_modules
from plinio.graph.transformation import fuse_consecutive_layers
from plinio.graph.features_calculation import ModAttrFeaturesCalculator
from plinio.graph.utils import fx_to_nx_graph, NamedLeafModules
from .nn.qtz import MPSType, MPSPerLayerQtz, MPSPerChannelQtz, MPSBiasQtz
from .quant.quantizers import DummyQuantizer

# add new supported layers here:
mps_layer_map: Dict[Type[nn.Module], Type[MPSModule]] = {
    nn.Conv2d: MPSConv2d,
    nn.Linear: MPSLinear,
}

# add new supported functions here:
mps_func_map: Dict[Callable, Type[MPSModule]] = {
    operator.add: MPSAdd,
    torch.add: MPSAdd,
}


class MPSTracer(fx.Tracer):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, MPSModule):
            return True
        else:
            return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)


def convert(model: nn.Module,
            input_example: Any,
            conversion_type: str,
            w_search_type: MPSType = MPSType.PER_LAYER,
            qinfo: Dict = {},
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            disable_shared_quantizers: bool = False
            ) -> Tuple[nn.Module, NamedLeafModules, NamedLeafModules]:
    """Converts a nn.Module, to/from "NAS-able" format for the mixed-precision search method

    :param model: the input nn.Module
    :type model: nn.Module
    :param input_example: an input with the same shape and type of the seed's input, used
    for symbolic tracing
    :type input_example: Any
    :param conversion_type: a string specifying the type of conversion. Supported types:
    ('import', 'autoimport', 'export')
    :type conversion_type: str
    :param w_search_type: the mixed precision strategy to be used for weigth
    i.e., `PER_CHANNEL` or `PER_LAYER`. Default is `PER_LAYER`
    :type w_search_type: MPSType
    :param qinfo: dict containing desired quantizers for act, weight and bias
    and their arguments excluding the precision precision
    :type qinfo: Dict
    :param qinfo_input_quantizer: desired quantizer for the input of the network. If set to None,
    the input is not quantized (default: None)
    :type qinfo_input_quantizer: Optional[dict]
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :param disable_shared_quantizers: a boolean to indicate whether to disable the quantizers
    sharing. It can be useful if precision '0' is in not in the search options.
    :type disable_shared_quantizers: bool
    :raises ValueError: for unsupported conversion types
    :return: the converted model, and two lists of all (or all unique) leaf modules for
    the NAS
    :rtype: Tuple[nn.Module, NamedLeafModule, NamedLeafModules]
    """
    if conversion_type not in ('import', 'autoimport', 'export'):
        raise ValueError("Unsupported conversion type {}".format(conversion_type))

    # Symbolic Tracing
    tracer = MPSTracer()
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    if len(get_graph_inputs(mod.graph)) > 1:
        ShapeProp(mod).propagate(*input_example)
    else:
        ShapeProp(mod).propagate(input_example)
    add_node_properties(mod)
    if conversion_type in ('autoimport', 'import'):
        fuse_mps_modules(mod)
    # Dictionary of shared quantizers. Used only in 'autoimport' mode.
    sq_dict = {} if conversion_type != 'autoimport' else build_shared_mps_qtz_map(
            mod, w_search_type, qinfo, disable_shared_quantizers)
    convert_layers(mod, conversion_type, qinfo, sq_dict, exclude_names, exclude_types)
    if conversion_type in ('autoimport', 'import'):
        add_input_quantizer(mod, qinfo)
        add_features_calculator(mod, [mps_features_calc])
        associate_input_features(mod)
        register_input_features(mod)
        register_in_mps_quantizers(mod)
    mod.graph.lint()
    mod.recompile()
    nlf = named_leaf_modules(mod)
    ulf = uniquify_leaf_modules(nlf)
    return mod, nlf, ulf


def convert_layers(mod: fx.GraphModule,
                   conversion_type: str,
                   qinfo: Dict,
                   sq_dict: Dict,
                   exclude_names: Iterable[str],
                   exclude_types: Iterable[Type[nn.Module]],
                   ):
    """Replaces target layers with their NAS-able version, or vice versa. Layer conversion
    is implemented as a reverse BFS on the model graph.

    :param mod: a torch.fx.GraphModule with tensor shapes annotations. Those are needed to
    determine the macs reg loss.
    :type mod: fx.GraphModule
    :param conversion_type: a string specifying the type of conversion
    :type conversion_type: str
    :param qinfo: dictionary containing desired quantizers for act, weight and bias
    and their arguments excluding the precision precision
    :type qinfo: Dict
    :param sq_dict: dictionary associating each fx.Node to a set of shared quantizers
    :type sq_dict: Dict
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    """
    g = mod.graph
    queue = get_graph_outputs(g)
    visited = []
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        if conversion_type == 'autoimport':
            autoimport_node(n, mod, qinfo, sq_dict, exclude_names, exclude_types)
        if conversion_type == 'export':
            export_node(n, mod, exclude_names, exclude_types)
        for pred in n.all_input_nodes:
            queue.append(pred)
        visited.append(n)
    return


def build_shared_mps_qtz_map(mod: fx.GraphModule,
                             w_search_type: MPSType,
                             qinfo: Dict,
                             disable_shared_quantizers: bool) -> Dict[
                                     fx.Node,
                                     Tuple[MPSPerLayerQtz, Union[MPSPerLayerQtz, MPSPerChannelQtz]]
                                     ]:
    """Create a map from fx.Node instances to instances of MPS Quantizers to be used by the NAS
    to optimize precision selection for both activations and weights of that node.
    Handles the sharing of quantizers among multiple nodes.

    :param mod: the fx-converted GraphModule
    :type mod: fx.GraphModule
    :param w_search_type: the mixed precision strategy to be used for weigth
    i.e., `PER_CHANNEL` or `PER_LAYER`. Default is `PER_LAYER`
    :type w_search_type: MPSType
    :param qinfo: dict containing desired quantizers for act, weight and bias
    and their arguments
    :type qinfo: Dict
    :param disable_shared_quantizers: a boolean to indicate whether to disable the quantizers
    sharing. It can be useful if precision '0' is in the search options.
    :type disable_shared_quantizers: bool
    :return: a map (node -> out_mps_quantizer, w_mps_quantizer)
    :rtype: Dict[fx.Node, Tuple[Quantizer, Quantizer]]
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

    # each weakly connected component of the sharing graph must share the same quantizers
    sq_dict = {}
    for c in nx.weakly_connected_components(sharing_graph):
        sq_a = None
        sq_w = None
        # This ensures to work at every iteration with a 'fresh' dict
        curr_qinfo = copy.deepcopy(qinfo)
        for n in c:
            # identify a node which can give us the number of features with 100% certainty
            # nodes such as flatten/squeeze etc make this necessary
            if (n.meta['features_defining'] or n.meta['untouchable']) and \
               (sq_a is None or sq_w is None):
                   if n.name in curr_qinfo.keys():
                       key = n.name
                   else:
                       key = 'layer_default'
                   a_quantizer = curr_qinfo[key]['output']['quantizer']
                   a_quantizer_kwargs = curr_qinfo[key]['output']['kwargs']
                   a_mps_precision = curr_qinfo[key]['output']['search_precision']
                   w_quantizer = curr_qinfo[key]['weight']['quantizer']
                   w_quantizer_kwargs = curr_qinfo[key]['weight']['kwargs']
                   w_mps_precision = curr_qinfo[key]['weight']['search_precision']
                   # Build activation shared quantizer
                   cout = n.meta['tensor_meta'].shape[1]
                   a_quantizer_kwargs['cout'] = cout
                   sq_a = MPSPerLayerQtz(a_mps_precision,
                                         a_quantizer,
                                         a_quantizer_kwargs)
                   # Build weight shared quantizer
                   w_quantizer_kwargs['cout'] = cout
                   if w_search_type == MPSType.PER_LAYER:
                       sq_w = MPSPerLayerQtz(w_mps_precision,
                                             w_quantizer,
                                             w_quantizer_kwargs)
                   elif w_search_type == MPSType.PER_CHANNEL:
                       sq_w = MPSPerChannelQtz(w_mps_precision,
                                               w_quantizer,
                                               w_quantizer_kwargs)
            if n in get_graph_outputs(mod.graph):
                # distinguish the case in which the number of features must "frozen"
                # i.e., the case of input-connected or output-connected components,
                # this may overwrite previously set "sq_w" and "sq_a"
                # In this case we simply remove the precision '0' from `weight_precisions`
                # if present and if mixprec search is PER_CHANNEL
                if n.name in curr_qinfo.keys():
                    key = n.name
                else:
                    key = 'layer_default'
                w_quantizer = curr_qinfo[key]['weight']['quantizer']
                w_quantizer_kwargs = curr_qinfo[key]['weight']['kwargs']
                w_mps_precision = curr_qinfo[key]['weight']['search_precision']
                cout = n.meta['tensor_meta'].shape[1]
                w_quantizer_kwargs['cout'] = cout
                if w_search_type == MPSType.PER_CHANNEL:
                    new_w_mps_precision = tuple(p for p in w_mps_precision if p != 0)
                    sq_w = MPSPerChannelQtz(new_w_mps_precision,
                                            w_quantizer,
                                            w_quantizer_kwargs)
                elif w_search_type == MPSType.PER_LAYER:
                    sq_w = MPSPerLayerQtz(w_mps_precision,
                                          w_quantizer,
                                          w_quantizer_kwargs)
                # If `c` contains an output node the output is not quantized
                # precision ignored
                sq_a = MPSPerLayerQtz((-1,), DummyQuantizer)
        for n in c:
            # if the flag 'disable_shared_quantizers' is set to True, then we can keep one quantizer
            # per connected component, which is the one defined in the previous for loop. Otherwise,
            # we are not forced to have the same weights quantizer between the various nodes.
            # Thus, we can instantiate a new weights quantizer per node, which will overwrite the
            # one previously set in "sq_w".
            if disable_shared_quantizers:
                if n.name in curr_qinfo.keys():
                    key = n.name
                else:
                    key = 'layer_default'
                w_quantizer = curr_qinfo[key]['weight']['quantizer']
                w_quantizer_kwargs = curr_qinfo[key]['weight']['kwargs']
                w_mps_precision = curr_qinfo[key]['weight']['search_precision']
                cout = n.meta['tensor_meta'].shape[1]
                w_quantizer_kwargs['cout'] = cout
                if w_search_type == MPSType.PER_LAYER:
                    sq_w = MPSPerLayerQtz(w_mps_precision,
                                          w_quantizer,
                                          w_quantizer_kwargs)
                elif w_search_type == MPSType.PER_CHANNEL:
                    sq_w = MPSPerChannelQtz(w_mps_precision,
                                            w_quantizer,
                                            w_quantizer_kwargs)

            sq_dict[n] = (sq_a, sq_w)
    return sq_dict


# N.B., same as PIT -> Remove duplicate
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


def autoimport_node(n: fx.Node,
                    mod: fx.GraphModule,
                    qinfo: Dict,
                    sq_dict: Dict[fx.Node, Tuple[MPSPerLayerQtz,
                                                 Union[MPSPerLayerQtz, MPSPerChannelQtz]]],
                    exclude_names: Iterable[str],
                    exclude_types: Iterable[Type[nn.Module]]
                    ):
    """Rewrites a fx.GraphModule node replacing a sub-module instance corresponding to a standard
    nn.Module with its corresponding NAS-able version.

    :param n: the node to be rewritten
    :type n: fx.Node
    :param mod: the parent module, where the new node has to be optionally inserted
    :type mod: fx.GraphModule
    :param qinfo: dict containing desired quantizers for act, weight and bias
    and their arguments
    :type qinfo: Dict
    :param sq_dict: a map (node -> out_mps_quantizer, w_mps_quantizer)
    :type sq_dict: Dict[fx.Node, Tuple[MPSPerLayerQtz, Union[MPSPerLayerQtz, MPSPerChannelQtz]]
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    """
    if is_layer(n, mod, tuple(mps_layer_map.keys())) and \
       not exclude(n, mod, exclude_names, exclude_types):
        module_type = mps_layer_map[type(mod.get_submodule(str(n.target)))]
    elif is_function(n, tuple(mps_func_map.keys())):
        module_type = mps_func_map[cast(Callable, n.target)]
    else:
        return

    out_mps_quantizer, w_mps_quantizer = sq_dict[n]

    # create bias mixprec quantizer (which is never shared)
    if n.name in qinfo.keys():
        b_quantizer = qinfo[n.name]['bias']['quantizer']
        b_quantizer_kwargs = qinfo[n.name]['bias']['kwargs']
    else:
        b_quantizer = qinfo['layer_default']['bias']['quantizer']
        b_quantizer_kwargs = qinfo['layer_default']['bias']['kwargs']
    cout = n.meta['tensor_meta'].shape[1]
    b_quantizer_kwargs['cout'] = cout
    b_mps_quantizer = MPSBiasQtz(b_quantizer,
                                 quantizer_kwargs=b_quantizer_kwargs)
    module_type.autoimport(n,
                           mod,
                           out_mps_quantizer,
                           w_mps_quantizer,
                           b_mps_quantizer)
    return


def export_node(n: fx.Node, mod: fx.GraphModule,
                exclude_names: Iterable[str],
                exclude_types: Iterable[Type[nn.Module]]):
    """Rewrites a fx.GraphModule node replacing a sub-module instance corresponding to a NAS-able
    layer with its corresponder quant.nn counterpart

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
    if is_inherited_layer(n, mod, (MPSModule,)):
        if exclude(n, mod, exclude_names, exclude_types):
            return
        layer = cast(MPSModule, mod.get_submodule(str(n.target)))
        layer.export(n, mod)


def add_input_quantizer(mod: fx.GraphModule,
                        qinfo: Dict):
    """Add input quantizer at the network input.

    :param mod: the parent module, where the new node has to be optionally inserted
    :type mod: fx.GraphModule
    :param activation_precisions: the possible activations' precisions assigment to be explored
    by the NAS
    :type activation_precisions: Tuple[int, ...]
    :param qinfo: dict containing desired quantizers for act and their arguments excluding
    the precision precision
    :type qinfo: Dict
    """
    g = mod.graph
    queue = get_graph_inputs(g)
    while queue:
        n = queue.pop(0)
        # Create quantizer
        if n.name in qinfo.keys():
            key = n.name
        elif 'input_default' not in qinfo.keys():
            return # no quantizer for input
        else:
            key = 'input_default'
        a_quantizer = qinfo[key]['quantizer']
        a_quantizer_kwargs = qinfo[key]['kwargs']
        a_mps_precision = qinfo[key]['search_precision']
        cout = n.meta['tensor_meta'].shape[1]
        a_quantizer_kwargs['cout'] = cout
        q_a = MPSPerLayerQtz(a_mps_precision,
                             a_quantizer,
                             a_quantizer_kwargs)
        inp_qtz = MPSIdentity(q_a)
        # Add quantizer to graph
        mod.add_submodule(n.name + '_input_quantizer', inp_qtz)
        with mod.graph.inserting_after(n):
            new_node = mod.graph.call_module(
                n.name + '_input_quantizer',
                args=(n,)
            )
            n.replace_all_uses_with(new_node)
            new_node.replace_input_with(new_node, n)
        # Add new node properties
        add_single_node_properties(new_node, mod)
        # Force the new node to be features_defining in order to be recognized
        # as predecessor when performing the `register_in_mps_quantizers` step
        new_node.meta['features_defining'] = True
        # Also copy the input shape information to the new node 'tensor_meta'
        new_node.meta['tensor_meta'] = n.meta['tensor_meta']


def fuse_bn_inplace(lin: nn.Module, bn: nn.Module):
    """
    Given a conv Module `A` and an batch_norm module `B`, modifies A
    such that A(x) == B(A_old(x))
    """
    # TODO: this is almost a duplicate of PIT. Resolve.
    assert (isinstance(lin, nn.Conv2d) or isinstance(lin, nn.Linear))
    assert (isinstance(bn, nn.BatchNorm1d) or isinstance(bn, nn.BatchNorm2d))
    if not bn.track_running_stats:
        raise AttributeError("BatchNorm folding requires track_running_stats = True")
    with torch.no_grad():
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


def fuse_mps_modules(mod: fx.GraphModule):
    """Fuse sequences of layers as required by MPS. Namely: Conv-BN and Linear-BN
    :param mod: the parent module
    :type mod: fx.GraphModule
    """
    # TODO: add Conv1d
    fuse_consecutive_layers(mod, nn.Conv2d, nn.BatchNorm2d, fuse_bn_inplace)
    fuse_consecutive_layers(mod, nn.Linear, nn.BatchNorm1d, fuse_bn_inplace)


def register_input_features(mod: fx.GraphModule):
    for n in mod.graph.nodes:
        if is_inherited_layer(n, mod, (MPSModule,)):
            # Set input features calculator
            sub_mod = cast(MPSModule, mod.get_submodule(str(n.target)))
            fc = n.meta['input_features_set_by'].meta['features_calculator']
            sub_mod.input_features_calculator = fc


def register_in_mps_quantizers(mod: fx.GraphModule):
    for n in mod.graph.nodes:
        if is_inherited_layer(n, mod, (MPSModule,)):
            sub_mod = cast(MPSModule, mod.get_submodule(str(n.target)))
            prev_n = n.meta['input_features_set_by']
            if prev_n.op == 'placeholder':
                continue
            while not is_inherited_layer(prev_n, mod, (MPSModule,)):
                prev_n = prev_n.meta['input_features_set_by']
            prev_submod = mod.get_submodule(str(prev_n.target))
            sub_mod.in_mps_quantizer = cast(MPSPerLayerQtz, prev_submod.out_mps_quantizer)


def mps_features_calc(n: fx.Node, mod: fx.GraphModule) -> Optional[ModAttrFeaturesCalculator]:
    """Sets the feature calculator for a MPS Module

    :param n: node
    :type n: fx.Node
    :param mod: the parent module
    :type mod: fx.GraphModule
    :return: optional feature calculator object for PIT node
    :rtype: ModAttrFeaturesCalculator
    """
    if is_inherited_layer(n, mod, (MPSModule,)):
        # For PIT NAS-able layers, the "active" output features are stored in the
        # out_features_eff attribute, and the binary mask is in features_mask
        sub_mod = mod.get_submodule(str(n.target))
        return ModAttrFeaturesCalculator(sub_mod, 'out_features_eff', 'features_mask')
    else:
        return None
