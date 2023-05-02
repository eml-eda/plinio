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

import operator
from typing import cast, List, Iterable, Type, Tuple, Optional, Dict, Callable
import networkx as nx
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from plinio.methods.mixprec.nn import MixPrec_Linear, MixPrec_Conv2d, MixPrec_Identity, \
    MixPrecModule, MixPrec_Add
from plinio.graph.annotation import add_features_calculator, add_node_properties, \
    associate_input_features, add_single_node_properties
from plinio.graph.inspection import is_layer, get_graph_outputs, is_inherited_layer, \
    get_graph_inputs, is_function
from plinio.graph.transformation import fuse_consecutive_layers
from plinio.methods.mixprec.nn.mixprec_qtz import MixPrecType, MixPrec_Qtz_Layer, \
    MixPrec_Qtz_Channel
from plinio.methods.mixprec.quant.quantizers import Quantizer
from plinio.graph.features_calculation import ModAttrFeaturesCalculator
from plinio.graph.utils import fx_to_nx_graph

# add new supported layers here:
mixprec_layer_map: Dict[Type[nn.Module], Type[MixPrecModule]] = {
    nn.Conv2d: MixPrec_Conv2d,
    nn.Linear: MixPrec_Linear,
}

# add new supported functions here:
mixprec_func_map: Dict[Callable, Type[MixPrecModule]] = {
    operator.add: MixPrec_Add,
    torch.add: MixPrec_Add,
}


class MixPrecTracer(fx.Tracer):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, MixPrecModule):
            return True
        else:
            return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)


def convert(model: nn.Module,
            input_shape: Tuple[int, ...],
            activation_precisions: Tuple[int, ...],
            weight_precisions: Tuple[int, ...],
            w_mixprec_type: MixPrecType,
            qinfo: Dict,
            conversion_type: str,
            input_quantization: bool = True,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            ) -> Tuple[nn.Module, List]:
    """Converts a nn.Module, to/from "NAS-able" MixPrec format

    :param model: the input nn.Module
    :type model: nn.Module
    :param input_shape: the shape of an input tensor, without batch size, required for symbolic
    tracing
    :type input_shape: Tuple[int, ...]
    :param activation_precisions: the possible activations' precisions assigment to be explored
    by the NAS
    :type activation_precisions: Iterable[int]
    :param weight_precisions: the possible weights' precisions assigment to be explored
    by the NAS
    :type weight_precisions: Iterable[int]
    :param w_mixprec_type: the mixed precision strategy to be used for weigth
    i.e., `PER_CHANNEL` or `PER_LAYER`. Default is `PER_LAYER`
    :type w_mixprec_type: MixPrecType
    :param qinfo: dict containing desired quantizers for act, weight and bias
    and their arguments excluding the num_bits precision
    :type qinfo: Dict
    :param conversion_type: a string specifying the type of conversion. Supported types:
    ('import', 'autoimport', 'export')
    :type conversion_type: str
    :param input_quantization: whether the input of the network needs
    to be quantized or not (default: True)
    :type input_quantization: bool
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

    # Symbolic Tracing
    tracer = MixPrecTracer()
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)

    # Shape Prop
    # create a "fake" minibatch of 1 input for shape prop
    batch_example = torch.stack([torch.rand(input_shape)] * 1, 0)
    device = next(model.parameters()).device
    ShapeProp(mod).propagate(batch_example.to(device))
    add_node_properties(mod)
    if conversion_type in ('autoimport', 'import'):
        fuse_mixprec_modules(mod)
    target_layers = convert_layers(mod,
                                   activation_precisions,
                                   weight_precisions,
                                   w_mixprec_type,
                                   qinfo,
                                   conversion_type,
                                   exclude_names,
                                   exclude_types)
    if conversion_type in ('autoimport', 'import'):
        if input_quantization:
            add_input_quantizer(mod, activation_precisions, qinfo)
        add_features_calculator(mod, [mixprec_features_calc])
        associate_input_features(mod)
        register_input_features(mod)
        register_input_quantizers(mod)
    mod.graph.lint()
    mod.recompile()
    return mod, target_layers


def convert_layers(mod: fx.GraphModule,
                   activation_precisions: Tuple[int, ...],
                   weight_precisions: Tuple[int, ...],
                   w_mixprec_type: MixPrecType,
                   qinfo: Dict,
                   conversion_type: str,
                   exclude_names: Iterable[str],
                   exclude_types: Iterable[Type[nn.Module]],
                   ) -> List[nn.Module]:
    """Replaces target layers with their NAS-able version, or vice versa, while also
    recording the list of NAS-able layers for speeding up later regularization loss
    computations. Layer conversion is implemented as a reverse BFS on the model graph.

    :param mod: a torch.fx.GraphModule with tensor shapes annotations. Those are needed to
    determine the macs reg loss.
    :type mod: fx.GraphModule
    :param activation_precisions: the possible activations' precisions assigment to be explored
    by the NAS
    :type activation_precisions: Iterable[int]
    :param weight_precisions: the possible weights' precisions assigment to be explored
    by the NAS
    :type weight_precisions: Iterable[int]
    :param w_mixprec_type: the mixed precision strategy to be used for weigth
    i.e., `PER_CHANNEL` or `PER_LAYER`. Default is `PER_LAYER`
    :type w_mixprec_type: MixPrecType
    :param qinfo: dict containing desired quantizers for act, weight and bias
    and their arguments excluding the num_bits precision
    :type qinfo: Dict
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
    # Dictionary of shared quantizers. Used only in 'autoimport' mode.
    sq_dict = build_shared_quantizers_map(mod,
                                          activation_precisions, weight_precisions,
                                          w_mixprec_type, qinfo)
    # the list of target layers is only used in 'import' and 'autoimport' modes. Empty for export
    target_layers = []
    visited = []
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        if conversion_type == 'autoimport':
            autoimport_node(n, mod, activation_precisions, weight_precisions,
                            w_mixprec_type, qinfo, sq_dict, exclude_names, exclude_types)
        if conversion_type in ('import', 'autoimport'):
            add_to_targets(n, mod, target_layers, exclude_names, exclude_types)
        if conversion_type == 'export':
            export_node(n, mod, exclude_names, exclude_types)
        for pred in n.all_input_nodes:
            queue.append(pred)
        visited.append(n)
    return target_layers


def build_shared_quantizers_map(mod: fx.GraphModule,
                                activation_precisions: Tuple[int, ...],
                                weight_precisions: Tuple[int, ...],
                                w_mixprec_type: MixPrecType,
                                qinfo: Dict) -> Dict[fx.Node, Tuple[Quantizer, Quantizer]]:
    """Create a map from fx.Node instances to instances of Quantizer to be used by MixPrec
    to optimize precision selection for both activations and weights of that node.
    Handles the sharing of quantizers among multiple nodes.

    :param mod: the fx-converted GraphModule
    :type mod: fx.GraphModule
    :param activation_precisions: the possible activations' precisions assigment to be explored
    by the NAS
    :type activation_precisions: Tuple[int]
    :param weight_precisions: the possible weights' precisions assigment to be explored
    by the NAS
    :type weight_precisions: Tuple[int]
    :param w_mixprec_type: the mixed precision strategy to be used for weigth
    i.e., `PER_CHANNEL` or `PER_LAYER`. Default is `PER_LAYER`
    :type w_mixprec_type: MixPrecType
    :param qinfo: dict containing desired quantizers for act, weight and bias
    and their arguments excluding the num_bits precision
    :type qinfo: Dict
    :return: a map (node -> quantizer_a, quantizer_w)
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
        for n in c:
            # identify a node which can give us the number of features with 100% certainty
            # nodes such as flatten/squeeze etc make this necessary
            if (n.meta['features_defining'] or n.meta['untouchable']) and \
               (sq_a is None or sq_w is None):
                # Build activation shared quantizer
                a_quantizer = qinfo['a_quantizer']['quantizer']
                a_quantizer_kwargs = qinfo['a_quantizer']['kwargs']
                cout = n.meta['tensor_meta'].shape[1]
                a_quantizer_kwargs['cout'] = cout
                sq_a = MixPrec_Qtz_Layer(activation_precisions,
                                         a_quantizer,
                                         a_quantizer_kwargs)
                # Build weight shared quantizer
                w_quantizer = qinfo['w_quantizer']['quantizer']
                w_quantizer_kwargs = qinfo['w_quantizer']['kwargs']
                cout = n.meta['tensor_meta'].shape[1]
                w_quantizer_kwargs['cout'] = cout
                if w_mixprec_type == MixPrecType.PER_LAYER:
                    sq_w = MixPrec_Qtz_Layer(weight_precisions,
                                             w_quantizer,
                                             w_quantizer_kwargs)
                elif w_mixprec_type == MixPrecType.PER_CHANNEL:
                    sq_w = MixPrec_Qtz_Channel(weight_precisions,
                                               cout,
                                               w_quantizer,
                                               w_quantizer_kwargs)
            if n in get_graph_outputs(mod.graph) or n in get_graph_inputs(mod.graph):
                # distinguish the case in which the number of features must "frozen"
                # i.e., the case of input-connected or output-connected components,
                # this may overwrite a previously set "sq_w"
                # In this case we simply remove the precision '0' from `weight_precisions`
                # if present and if mixprec search is PER_CHANNEL
                if w_mixprec_type == MixPrecType.PER_CHANNEL:
                    new_weight_precisions = tuple(p for p in weight_precisions if p != 0)
                    # new_weight_precisions = weight_precisions  # uncomment to have 0 in last fc
                    w_quantizer = qinfo['w_quantizer']['quantizer']
                    w_quantizer_kwargs = qinfo['w_quantizer']['kwargs']
                    cout = n.meta['tensor_meta'].shape[1]
                    w_quantizer_kwargs['cout'] = cout
                    sq_w = MixPrec_Qtz_Channel(new_weight_precisions,
                                               cout,
                                               w_quantizer,
                                               w_quantizer_kwargs)
                # continue
        for n in c:
            # If `c` contains an output node the output quantizer is forced to be
            # an identity op
            # if n in get_graph_outputs(mod.graph):
            if any([n in get_graph_outputs(mod.graph) for n in c]):
                sq_a = nn.Identity()  # Output is not quantized
            sq_dict[n] = (sq_a, sq_w)
    return sq_dict


# N.B., same of pit_converter -> No need to duplicate code
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
                    activation_precisions: Tuple[int, ...],
                    weight_precisions: Tuple[int, ...],
                    w_mixprec_type: MixPrecType,
                    qinfo: Dict,
                    sq_dict: Dict[fx.Node, Tuple[Quantizer, Quantizer]],
                    exclude_names: Iterable[str],
                    exclude_types: Iterable[Type[nn.Module]]
                    ) -> Optional[Quantizer]:
    """Rewrites a fx.GraphModule node replacing a sub-module instance corresponding to a standard
    nn.Module with its corresponding NAS-able version.

    Also determines if the currently processed node requires that its predecessor share a common
    features mask.

    :param n: the node to be rewritten
    :type n: fx.Node
    :param mod: the parent module, where the new node has to be optionally inserted
    :type mod: fx.GraphModule
    :param activation_precisions: the possible activations' precisions assigment to be explored
    by the NAS
    :type activation_precisions: Tuple[int, ...]
    :param weight_precisions: the possible weights' precisions assigment to be explored
    by the NAS
    :type weight_precisions: Tuple[int, ...]
    :param w_mixprec_type: the mixed precision strategy to be used for weigth
    i.e., `PER_CHANNEL` or `PER_LAYER`. Default is `PER_LAYER`
    :type w_mixprec_type: MixPrecType
    :param qinfo: dict containing desired quantizers for act, weight and bias
    and their arguments excluding the num_bits precision
    :type qinfo: Dict
    :param sm_dict: a map (node -> quantizer_a, quantizer_w)
    :type sm_dict: Dict[fx.Node, Tuple[Quantizer, Quantizer]]
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :return: the updated shared_quantizer
    :rtype: Optional[Quantizer]
    """
    if is_layer(n, mod, tuple(mixprec_layer_map.keys())) and \
       not exclude(n, mod, exclude_names, exclude_types):
        module_type = mixprec_layer_map[type(mod.get_submodule(str(n.target)))]
    elif is_function(n, tuple(mixprec_func_map.keys())):
        module_type = mixprec_func_map[cast(Callable, n.target)]
    else:
        return
    # TODO: Define some utility quantization function to do all this stuff
    # Unpack qinfo
    b_quantizer = qinfo['b_quantizer']['quantizer']
    b_quantizer_kwargs = qinfo['b_quantizer']['kwargs']
    # Add output channel info to wuantizer kwargs
    cout = n.meta['tensor_meta'].shape[1]
    b_quantizer_kwargs['cout'] = cout
    # Convert
    module_type.autoimport(n,
                           mod,
                           w_mixprec_type,
                           activation_precisions,
                           weight_precisions,
                           b_quantizer,
                           sq_dict[n],
                           b_quantizer_kwargs)


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
    if is_inherited_layer(n, mod, (MixPrecModule,)):
        if exclude(n, mod, exclude_names, exclude_types):
            return
        layer = cast(MixPrecModule, mod.get_submodule(str(n.target)))
        layer.export(n, mod)


def add_input_quantizer(mod: fx.GraphModule,
                        activation_precisions: Tuple[int, ...],
                        qinfo: Dict):
    """Add input quantizer at the network input.

    :param mod: the parent module, where the new node has to be optionally inserted
    :type mod: fx.GraphModule
    :param activation_precisions: the possible activations' precisions assigment to be explored
    by the NAS
    :type activation_precisions: Tuple[int, ...]
    :param qinfo: dict containing desired quantizers for act, weight and bias
    and their arguments excluding the num_bits precision
    :type qinfo: Dict
    """
    g = mod.graph
    queue = get_graph_inputs(g)
    while queue:
        n = queue.pop(0)
        # Create quantizer
        a_quantizer = qinfo['a_quantizer']['quantizer']
        a_quantizer_kwargs = qinfo['a_quantizer']['kwargs']
        cout = n.meta['tensor_meta'].shape[1]
        a_quantizer_kwargs['cout'] = cout
        a_quantizer_kwargs['init_clip_val'] = 1.
        # TODO: give more flexibility upon the choice of the input quantizer
        q_a = MixPrec_Qtz_Layer(activation_precisions,
                                a_quantizer,
                                a_quantizer_kwargs)
        inp_qtz = MixPrec_Identity(activation_precisions, q_a)
        # Add quantizer to graph
        mod.add_submodule('input_quantizer', inp_qtz)
        with mod.graph.inserting_after(n):
            new_node = mod.graph.call_module(
                'input_quantizer',
                args=(n,)
            )
            n.replace_all_uses_with(new_node)
            new_node.replace_input_with(new_node, n)
        # Add new node properties
        add_single_node_properties(new_node, mod)
        # Force the new node to be features_defining in order to be recognized
        # as predecessor when performin the `register_input_quantizers` step
        new_node.meta['features_defining'] = True


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
    if is_inherited_layer(n, mod, (MixPrecModule,)):
        if exclude(n, mod, exclude_names, exclude_types):
            return
        # only conv and FC, exclude BN
        if is_layer(n, mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            return
        target_layers.append(mod.get_submodule(str(n.target)))
    # TODO: is the following part of code needed?
    # TODO: Find better way to add modules that in original graph was functions (e.g., add)
    # if is_function(n, tuple(mixprec_func_map.keys())):
    #     name = str(n) + '_' + str(n.all_input_nodes) + '_quant'
    #     target_layers.append(mod.get_submodule(name))


def fuse_bn_inplace(lin: nn.Module, bn: nn.Module):
    """
    Given a conv Module `A` and an batch_norm module `B`, modifies A
    such that A(x) == B(A_old(x))
    """
    # TODO: this is almost a duplicate of PIT. Resolve.
    assert (isinstance(lin, nn.Conv2d) or isinstance(lin, nn.Linear))
    # or isinstance(lin, MixPrec_Conv1d))
    assert (isinstance(bn, nn.BatchNorm1d) or isinstance(bn, nn.BatchNorm2d))
    if not bn.track_running_stats:
        raise AttributeError("BatchNorm folding requires track_running_stats = True")
    with torch.no_grad():
        # mixprec_a_quantizerconv.following_bn_args = {
        #     'eps': bn.eps,
        #     'momentum': bn.momentum,
        #     'affine': bn.affine,
        #     'track_running_stats': bn.track_running_stats,
        # }
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


def fuse_mixprec_modules(mod: fx.GraphModule):
    """Fuse sequences of layers as required by MixPrec. Namely: Conv-BN and Linear-BN
    :param mod: the parent module
    :type mod: fx.GraphModule
    """
    # fuse_consecutive_layers(mod, MixPrec_Conv1d, nn.BatchNorm1d, fuse_bn_inplace)
    fuse_consecutive_layers(mod, nn.Conv2d, nn.BatchNorm2d, fuse_bn_inplace)
    fuse_consecutive_layers(mod, nn.Linear, nn.BatchNorm1d, fuse_bn_inplace)


def register_input_features(mod: fx.GraphModule):
    for n in mod.graph.nodes:
        if is_inherited_layer(n, mod, (MixPrecModule,)):
            # Set input features calculator
            sub_mod = cast(MixPrecModule, mod.get_submodule(str(n.target)))
            fc = n.meta['input_features_set_by'].meta['features_calculator']
            sub_mod.input_features_calculator = fc


def register_input_quantizers(mod: fx.GraphModule):
    for n in mod.graph.nodes:
        if is_inherited_layer(n, mod, (MixPrecModule,)):
            sub_mod = cast(MixPrecModule, mod.get_submodule(str(n.target)))
            # Set pointer to proper input act quantizer for bias quantizer
            # this should be done only for convolutional layers
            prev_n = n.meta['input_features_set_by']
            if prev_n.op == 'placeholder':
                continue
            while not is_inherited_layer(prev_n, mod, (MixPrecModule,)):
                prev_n = prev_n.meta['input_features_set_by']
            prev_submod = mod.get_submodule(str(prev_n.target))
            # sub_mod = cast(MixPrec_Conv2d, sub_mod)
            # cast(nn.Module,
            #      sub_mod.mixprec_b_quantizer
            #      ).mixprec_a_quantizer = prev_submod.mixprec_a_quantizer
            # sub_mod.input_quantizer = cast(MixPrec_Qtz_Layer, prev_submod.mixprec_a_quantizer)
            sub_mod.update_input_quantizer(
                cast(MixPrec_Qtz_Layer, prev_submod.mixprec_a_quantizer))


def mixprec_features_calc(n: fx.Node, mod: fx.GraphModule) -> Optional[ModAttrFeaturesCalculator]:
    """Sets the feature calculator for a MixPrec Module

    :param n: node
    :type n: fx.Node
    :param mod: the parent module
    :type mod: fx.GraphModule
    :return: optional feature calculator object for PIT node
    :rtype: ModAttrFeaturesCalculator
    """
    if is_inherited_layer(n, mod, (MixPrecModule,)):
        # For PIT NAS-able layers, the "active" output features are stored in the
        # out_features_eff attribute, and the binary mask is in features_mask
        sub_mod = mod.get_submodule(str(n.target))
        return ModAttrFeaturesCalculator(sub_mod, 'out_features_eff', 'features_mask')
    else:
        return None
