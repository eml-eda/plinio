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

from typing import cast, List, Iterable, Type, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from plinio.methods.mixprec.nn import MixPrec_Linear, MixPrec_Conv2d, \
    MixPrecModule
from plinio.graph.annotation import add_features_calculator, add_node_properties, \
    associate_input_features
from plinio.graph.inspection import is_inherited_layer
from plinio.graph.transformation import fuse_consecutive_layers
from plinio.methods.mixprec.nn.mixprec_qtz import MixPrecType, MixPrec_Qtz_Layer
from plinio.methods.mixprec.quant.quantizers import Quantizer
from plinio.graph import get_graph_outputs
from plinio.graph import inspection
from plinio.graph.features_calculation import ModAttrFeaturesCalculator

# add new supported layers here:
mixprec_layer_map: Dict[Type[nn.Module], Type[MixPrecModule]] = {
    nn.Conv2d: MixPrec_Conv2d,
    nn.Linear: MixPrec_Linear,
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
        add_features_calculator(mod, [mixprec_features_calc])
        associate_input_features(mod)
        register_input_features(mod)
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
    # the shared_quantizer_queue is only used in 'autoimport' mode.
    # We consider the case of shared quantizer for activations
    # TODO: Understand if weight/bias quantizer might be shared or not
    shared_quantizer_queue: List[Optional[Quantizer]] = [None]
    # the list of target layers is only used in 'import' and 'autoimport' modes. Empty for export
    target_layers = []
    visited = []
    while queue:
        n = queue.pop(0)
        shared_quantizer = shared_quantizer_queue.pop(0)
        if n not in visited:
            if conversion_type == 'autoimport':
                shared_quantizer = autoimport_node(n,
                                                   mod,
                                                   activation_precisions,
                                                   weight_precisions,
                                                   w_mixprec_type,
                                                   qinfo,
                                                   shared_quantizer,
                                                   exclude_names,
                                                   exclude_types)
                # shared_masker = update_shared_masker(n, mod, shared_masker)
            if conversion_type == 'export':
                export_node(n, mod, exclude_names, exclude_types)
            if conversion_type in ('import', 'autoimport'):
                add_to_targets(n, mod, target_layers, exclude_names, exclude_types)
            if conversion_type != 'autoimport':
                shared_quantizer = None

            for pred in n.all_input_nodes:
                queue.append(pred)
                shared_quantizer_queue.append(shared_quantizer)
            visited.append(n)
    return target_layers


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
                    sq: Optional[Quantizer],
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
    :param sq: an optional shared quantizer derived from subsequent layers
    :type sq: Optional[Quantizer]
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :return: the updated shared_quantizer
    :rtype: Optional[Quantizer]
    """
    if inspection.is_layer(n, mod, tuple(mixprec_layer_map.keys())) and \
       not exclude(n, mod, exclude_names, exclude_types):
        conv_layer_type = mixprec_layer_map[type(mod.get_submodule(str(n.target)))]
        # TODO: Define some utility quantization function to do all this stuff
        # Unpack qinfo
        a_quantizer = qinfo['a_quantizer']['quantizer']
        a_quantizer_kwargs = qinfo['a_quantizer']['kwargs']
        w_quantizer = qinfo['w_quantizer']['quantizer']
        w_quantizer_kwargs = qinfo['w_quantizer']['kwargs']
        b_quantizer = qinfo['b_quantizer']['quantizer']
        b_quantizer_kwargs = qinfo['b_quantizer']['kwargs']
        # Add output channel info to wuantizer kwargs
        cout = n.meta['tensor_meta'].shape[1]
        a_quantizer_kwargs['cout'] = cout
        w_quantizer_kwargs['cout'] = cout
        b_quantizer_kwargs['cout'] = cout
        # Convert
        sq = conv_layer_type.autoimport(n,
                                        mod,
                                        w_mixprec_type,
                                        activation_precisions,
                                        weight_precisions,
                                        a_quantizer,
                                        w_quantizer,
                                        b_quantizer,
                                        sq,
                                        a_quantizer_kwargs,
                                        w_quantizer_kwargs,
                                        b_quantizer_kwargs)
        return sq
    elif inspection.is_shared_input_features_op(n, mod):
        # modules that require same activation quantizer
        # creates a new shared quantizer to be used by predecessors
        if sq is None or inspection.is_features_defining_op(n, mod):
            a_quantizer = qinfo['a_quantizer']['quantizer']
            a_quantizer_kwargs = qinfo['a_quantizer']['kwargs']
            cout = n.all_input_nodes[-1].meta['tensor_meta'].shape[1]
            a_quantizer_kwargs['cout'] = cout
            shared_quantizer = MixPrec_Qtz_Layer(activation_precisions,
                                                 a_quantizer,
                                                 a_quantizer_kwargs)
            shared_quantizer = cast(Quantizer, shared_quantizer)
        else:
            shared_quantizer = sq
        return shared_quantizer
    elif inspection.is_flatten(n, mod):
        if sq is not None:
            raise ValueError("Shared channels masks not supported for flatten")
        return None
    elif inspection.is_untouchable_op(n):
        return None
    elif inspection.is_features_concatenate(n, mod):
        # if we concatenate over features, we need to share the mask
        # then we create a new shared quantizer to be used by predecessors
        if sq is None or inspection.is_features_defining_op(n, mod):
            a_quantizer = qinfo['a_quantizer']['quantizer']
            a_quantizer_kwargs = qinfo['a_quantizer']['kwargs']
            # Computes cout after cat operation
            cout = sum(list(map(
                lambda x: x.meta['tensor_meta'].shape[1],
                n.all_input_nodes)))
            a_quantizer_kwargs['cout'] = cout
            shared_quantizer = MixPrec_Qtz_Layer(activation_precisions,
                                                 a_quantizer,
                                                 a_quantizer_kwargs)
            shared_quantizer = cast(Quantizer, shared_quantizer)
        else:
            shared_quantizer = sq
        return shared_quantizer
    elif inspection.is_features_propagating_op(n, mod):
        # this op has cin = cout, so return what was received as input
        return sq
    elif inspection.is_features_defining_op(n, mod):
        # this op defines its output features, no propagation
        return None
    else:
        raise ValueError("Unsupported node {} (op: {}, target: {})".format(n, n.op, n.target))


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
    if inspection.is_inherited_layer(n, mod, (MixPrecModule,)):
        if exclude(n, mod, exclude_names, exclude_types):
            return
        layer = cast(MixPrecModule, mod.get_submodule(str(n.target)))
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
    if inspection.is_inherited_layer(n, mod, (MixPrecModule,)):
        if exclude(n, mod, exclude_names, exclude_types):
            return
        # only conv and FC, exclude BN
        if inspection.is_layer(n, mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            return
        target_layers.append(mod.get_submodule(str(n.target)))


def fuse_bn_inplace(lin: nn.Module, bn: nn.Module):
    """
    Given a conv Module `A` and an batch_norm module `B`, modifies A
    such that A(x) == B(A_old(x))
    """
    # TODO: this is almost a duplicate of PIT. Resolve.
    assert (isinstance(lin, MixPrec_Conv2d) or isinstance(lin, MixPrec_Linear))
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
    fuse_consecutive_layers(mod, MixPrec_Conv2d, nn.BatchNorm2d, fuse_bn_inplace)
    fuse_consecutive_layers(mod, MixPrec_Linear, nn.BatchNorm1d, fuse_bn_inplace)


def register_input_features(mod: fx.GraphModule):
    for n in mod.graph.nodes:
        if is_inherited_layer(n, mod, (MixPrecModule,)):
            sub_mod = cast(MixPrecModule, mod.get_submodule(str(n.target)))
            fc = n.meta['input_features_set_by'].meta['features_calculator']
            sub_mod.input_features_calculator = fc


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
