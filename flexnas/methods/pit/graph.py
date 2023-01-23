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
from flexnas.graph.inspection import is_layer, get_output_nodes, is_inherited_layer, \
        get_input_nodes
from flexnas.graph.utils import all_output_nodes
from flexnas.graph.features_calculation import ModAttrFeaturesCalculator

# add new supported layers here:
# TODO: can we fill this automatically based on classes that inherit from PITLayer?
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
    # create a "fake" minibatch of 32 inputs for shape prop
    batch_example = torch.stack([torch.rand(input_shape)] * 32, 0)
    # TODO: this is not very robust. Find a better way
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
    queue = get_output_nodes(g)
    # the shared_masker_queue is only used in 'autoimport' mode.
    # initialied with Frozen maskers to ensure output layers are never trainable
    shared_masker_dict: Dict[fx.Node, List[Optional[PITFeaturesMasker]]] = {
        n: [PITFrozenFeaturesMasker(n.meta['tensor_meta'].shape[1])] for n in queue
    }
    # the list of target layers is only used in 'import' and 'autoimport' modes. Empty for export
    target_layers = []
    visited = []
    while queue:
        n = queue.pop(0)
        if n not in visited and all(_ in visited for _ in list(n.users.keys())):
            shared_masker = merge_maskers(shared_masker_dict[n])
            if conversion_type == 'autoimport':
                shared_masker = autoimport_node(n, mod, shared_masker, exclude_names, exclude_types)
                # shared_masker = update_shared_masker(n, mod, shared_masker)
            if conversion_type == 'export':
                export_node(n, mod, exclude_names, exclude_types)
            if conversion_type in ('import', 'autoimport'):
                add_to_targets(n, mod, target_layers, exclude_names, exclude_types)
            if conversion_type != 'autoimport':
                shared_masker = None

            for pred in n.all_input_nodes:
                queue.append(pred)
                if pred in shared_masker_dict:
                    shared_masker_dict[pred].append(shared_masker)
                else:
                    shared_masker_dict[pred] = [shared_masker]
            visited.append(n)
    return target_layers


def merge_maskers(sm_list: List[Optional[PITFeaturesMasker]]) -> Optional[PITFeaturesMasker]:
    """Combine channel masks passed from successors to current node.
    Currently fails unless at most 1 mask is not None

    :param sm_list: the list of PITFeaturesMasker instances (or None) received from successor
    nodes
    :type sm_list: List[Optional[PITFeaturesMasker]]
    :return: the merged feature masker
    :rtype: Optional[PITFeaturesMasker]
    """
    merged_sm = None
    for sm in sm_list:
        if sm is not None:
            if merged_sm is not None:
                raise ValueError(
                   "PIT Conversion: receiving two incompatible feature masks from successors."
                )
            else:
                merged_sm = sm
    return merged_sm


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
                    exclude_types: Iterable[Type[nn.Module]]) -> Optional[PITFeaturesMasker]:
    """Rewrites a fx.GraphModule node replacing a sub-module instance corresponding to a standard
    nn.Module with its corresponding NAS-able version.

    Also determines if the currently processed node requires that its predecessor share a common
    features mask.

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
    :return: the updated shared_masker
    :rtype: Optional[PITChannelMasker]
    """
    if is_layer(n, mod, tuple(pit_layer_map.keys())) and not exclude(
            n, mod, exclude_names, exclude_types):
        conv_layer_type = pit_layer_map[type(mod.get_submodule(str(n.target)))]
        sm = conv_layer_type.autoimport(n, mod, sm)
        return sm
    elif n.meta['shared_input_features']:
        # modules that require multiple inputs all of the same size
        # create a new shared masker with the common n. of input channels, to be used by
        # predecessors
        if sm is None or n.meta['features_defining']:
            input_size = n.all_input_nodes[-1].meta['tensor_meta'].shape[1]
            shared_masker = PITFeaturesMasker(input_size)
        else:
            shared_masker = sm
        return shared_masker
    elif n.meta['flatten']:
        if sm is not None:
            raise ValueError("Shared channels masks not supported for flatten")
        return None
    elif n.meta['untouchable']:
        return None
    elif n.meta['features_concatenate']:
        # if we concatenate over features, we don't need to share the mask
        return None
    elif n.meta['features_propagating']:
        # this op has cin = cout, so return what was received as input
        return sm
    elif n.meta['features_defining']:
        # this op defines its output features, no propagation
        return None
    else:
        raise ValueError("Unsupported node {} (op: {}, target: {})".format(n, n.op, n.target))


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
    g = mod.graph
    queue = get_input_nodes(g)
    while queue:
        n = queue.pop(0)

        if is_inherited_layer(n, mod, (PITModule,)):
            sub_mod = cast(PITModule, mod.get_submodule(str(n.target)))
            fc = n.meta['input_features_set_by'].meta['features_calculator']
            sub_mod.input_features_calculator = fc

        for succ in all_output_nodes(n):
            queue.append(succ)


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
