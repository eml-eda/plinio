# *----------------------------------------------------------------------------*
# * Copyright (C) 2021 Politecnico di Torino, Italy                            *
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
from typing import List, Type, Tuple, Any, Dict, Optional, Callable
import math
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
import networkx as nx
from flexnas.utils.features_calculator import ConstFeaturesCalculator, \
    FlattenFeaturesCalculator, ConcatFeaturesCalculator, FeaturesCalculator


def fx_to_nx_graph(fx_graph: fx.Graph) -> nx.DiGraph:
    """Transforms a `torch.fx.Graph` into an equivalent `networkx.DiGraph` for easier visits.

    :param fx_graph: the `torch.fx.Graph` instance.
    :type fx_graph: fx.Graph
    :return: the corresponding `networkx.DiGraph`
    :rtype: nx.DiGraph
    """
    nx_graph = nx.DiGraph()
    for n in fx_graph.nodes:
        for i in n.all_input_nodes:
            nx_graph.add_edge(i, n)
    return nx_graph


def get_input_nodes(fx_graph: fx.Graph) -> List[fx.Node]:
    """From a `torch.fx.Graph`, return the list of nodes that correspond to network inputs.

    Basically finds all nodes of type 'placeholder'.

    :param fx_graph: the network graph
    :type fx_graph: fx.Graph
    :return: a list of `torch.fx.Node` instances corresponding to network inputs.
    :rtype: List[fx.Node]
    """
    ret = []
    for n in fx_graph.nodes:
        if n.op == 'placeholder':
            ret.append(n)
    return ret


def get_output_nodes(fx_graph: fx.Graph) -> List[fx.Node]:
    """From a `torch.fx.Graph`, return the list of nodes that correspond to network outputs.

    Basically finds all nodes of type 'output'.

    :param fx_graph: the network graph
    :type fx_graph: fx.Graph
    :return: a list of `torch.fx.Node` instances corresponding to network outputs.
    :rtype: List[fx.Node]
    """
    ret = []
    for n in fx_graph.nodes:
        if n.op == 'output':
            ret.append(n)
    return ret


def layer_type(n: fx.Node, parent: fx.GraphModule) -> Type:
    """Gets the layer type of a `torch.fx.Node` of type `call_module`

    :param n: the target node
    :type n: fx.Node
    :param parent: the parent `nn.Module`
    :type parent: fx.GraphModule
    :return: the type of n
    :rtype: Type
    """
    return type(parent.get_submodule(str(n.target)))


def is_layer(n: fx.Node, parent: fx.GraphModule,
             layers: Tuple[Type[Any], ...]) -> bool:
    """Checks if a `torch.fx.Node` corresponds to a specific layer type.

    :param n: the target node
    :type n: fx.Node
    :param parent: the parent `nn.Module`
    :type parent: fx.GraphModule
    :param layers: the layer types to be checked
    :type layers: Tuple[Type[Any], ...]]
    :return: `True` if `n` is of type `layer`
    :rtype: bool
    """
    if n.op != 'call_module':
        return False
    return layer_type(n, parent) in layers


def is_inherited_layer(n: fx.Node, parent: fx.GraphModule,
                       layers: Tuple[Type[Any], ...]) -> bool:
    """Checks if a `torch.fx.Node` corresponds to a specific layer type or to
       a layer that inherits the class of the specified layer
       (for instance PITConv1d inherits from nn.Conv1d).

    :param n: the target node
    :type n: fx.Node
    :param parent: the parent `nn.Module`
    :type parent: fx.GraphModule
    :param layers: the layer types to be checked
    :type layers: Tuple[Type[Any], ...]
    :return: `True` if `n` is of type `layer` or if 'n' inherits from 'layer'
    :rtype: bool
    """
    if n.op != 'call_module':
        return False
    return isinstance(parent.get_submodule(str(n.target)), layers)


def is_zero_or_one_input_op(n: fx.Node) -> bool:
    """Checks if a `torch.fx.Node` has no more than 1 input.

    :param n: the target node
    :type n: fx.Node
    :return: `True` if `n` has 0 or 1 inputs.
    :rtype: bool
    """
    return len(n.all_input_nodes) <= 1


def is_untouchable_op(n: fx.Node) -> bool:
    """Checks if a `torch.fx.Node` has a functional convolution.

    :param n: the target node
    :type n: fx.Node
    :return: `True` if `n` is a conv1d, conv2d or conv3d.
    :rtype: bool
    """
    # We assume that functional convolution should not be optimized
    if n.op == 'call_function':
        if n.target == torch.conv1d:
            return True
        if n.target == torch.conv2d:
            return True
        if n.target == torch.conv3d:
            return True
    return False


def is_shared_input_features_op(n: fx.Node, parent: fx.GraphModule) -> bool:
    """Checks if a `torch.fx.Node` corresponds to an operation that requires all its inputs to
    share the same number of features.

    Note that this is implemented as a simple pattern matching against a (non-exhaustive) list of
    `torch.fx` ops.

    :param n: the target node
    :type n: fx.Node
    :param parent: the parent sub-module
    :type parent: fx.GraphModule
    :return: `True` if `n` requires all its inputs to have the same number of features.
    :rtype: bool
    """
    if is_zero_or_one_input_op(n):
        return False
    if is_concatenate(n, parent) and not is_features_concatenate(n, parent):
        return True
    if n.op == 'call_function':
        if n.target == torch.add:
            return True
        if n.target == operator.add:
            return True
        if n.target == torch.sub:
            return True
        if n.target == operator.sub:
            return True
        if n.target == torch.squeeze:
            return True
        # TODO: add others
    # are there any modules that require same input size? if so, add them below. Same for methods
    # if n.op == 'call_module':
    # if n.op == 'call_method':
    return False


def is_features_defining_op(n: fx.Node, parent: fx.GraphModule) -> bool:
    """Checks if a `torch.fx.Node` corresponds to an operation that "defines" the number of
    features for successors.

    For example, convolutions and fully-connected layers have, in general,
    out_features != in_features, hence they are "features-defining". In contrast, ReLU has
    out_features == in_features, hence it is "features-propagating".

    Note that this is implemented as a simple pattern matching against a (non-exhaustive) list of
    `torch.fx` ops.

    :param n: the target node
    :type n: fx.Node
    :param parent: the parent sub-module
    :type parent: fx.GraphModule
    :return: `True` if `n` corresponds to a "features-defining" op.
    :rtype: bool
    """
    if n.op == 'placeholder' and len(n.all_input_nodes) == 0:  # input node
        return True
    if n.op == 'call_module':
        submodule = parent.get_submodule(str(n.target))
        if isinstance(submodule, nn.Conv1d) or isinstance(submodule, nn.Conv2d):
            if (submodule.groups == submodule.in_channels) and (
                    submodule.groups == submodule.out_channels):
                # this is the special case of a DepthWise Conv
                return False
            else:
                return True
        if isinstance(submodule, nn.Linear):
            return True
        if isinstance(submodule, nn.Sequential):
            return True
    return False


def is_features_propagating_op(n: fx.Node, parent: fx.GraphModule) -> bool:
    """Checks if a `torch.fx.Node` corresponds to an operation that "propagates" the number of
    input features to successors.

    For example, convolutions and fully-connected layers have, in general,
    out_features != in_features, hence they are "features-defining". In contrast, ReLU has
    out_features == in_features, hence it is "features-propagating".

    Note that this is implemented as a simple pattern matching against a (non-exhaustive) list of
    `torch.fx` ops.

    :param n: the target node
    :type n: fx.Node
    :param parent: the parent sub-module
    :type parent: fx.GraphModule
    :return: `True` if `n` corresponds to a "features-propagating" op.
    :rtype: bool
    """
    if n.op == 'output':
        return True
    if n.op == 'call_module':
        submodule = parent.get_submodule(str(n.target))
        if isinstance(submodule, nn.BatchNorm1d):
            return True
        if isinstance(submodule, nn.BatchNorm2d):
            return True
        if isinstance(submodule, nn.AvgPool1d):
            return True
        if isinstance(submodule, nn.AvgPool2d):
            return True
        if isinstance(submodule, nn.MaxPool1d):
            return True
        if isinstance(submodule, nn.BatchNorm2d):
            return True
        if isinstance(submodule, nn.Dropout):
            return True
        if isinstance(submodule, nn.ReLU):
            return True
        if isinstance(submodule, nn.ReLU6):
            return True
        if isinstance(submodule, nn.ConstantPad1d):
            return True
        if isinstance(submodule, nn.ConstantPad2d):
            return True
        if isinstance(submodule, nn.AdaptiveAvgPool1d):
            return True
        if isinstance(submodule, nn.Identity):
            return True
        if isinstance(submodule, nn.Conv1d) or isinstance(submodule, nn.Conv2d):
            if (submodule.groups == submodule.in_channels) and (
                    submodule.groups == submodule.out_channels):
                # this is the special case of a DepthWise Conv
                return True
            else:
                return False
        # TODO: add others
    if n.op == 'call_function':
        if n.target == F.relu:
            return True
        if n.target == F.relu6:
            return True
        if n.target == F.log_softmax:
            return True
        if n.target == torch.add:
            return True
        if n.target == operator.add:
            return True
        if n.target == torch.sub:
            return True
        if n.target == operator.sub:
            return True
        if n.target == torch.squeeze:
            return True
    return False


def is_flatten(n: fx.Node, parent: fx.GraphModule) -> bool:
    """Checks if a `torch.fx.Node` instance corresponds to a flatten operation.

    :param n: the target node
    :type n: fx.Node
    :param parent: the parent sub-module
    :type parent: fx.GraphModule
    :return: `True` if `n` corresponds to a flatten op.
    :rtype: bool
    """
    if n.op == 'call_method' and n.target == 'flatten':
        return True
    if n.op == 'call_function' and n.target == torch.flatten:
        return True
    return False


def is_squeeze(n: fx.Node, parent: fx.GraphModule) -> bool:
    """Checks if a `torch.fx.Node` instance corresponds to a squeeze operation.

    :param n: the target node
    :type n: fx.Node
    :param parent: the parent sub-module
    :type parent: fx.GraphModule
    :return: `True` if `n` corresponds to a squeeze op.
    :rtype: bool
    """
    if n.op == 'call_method' and n.target == 'squeeze':
        return True
    if n.op == 'call_function' and n.target == torch.squeeze:
        return True
    return False


def is_features_concatenate(n: fx.Node, parent: fx.GraphModule) -> bool:
    """Checks if a `torch.fx.Node` instance corresponds to a concat operation
    over the features axis.

    :param n: the target node
    :type n: fx.Node
    :param parent: the parent sub-module
    :type parent: fx.GraphModule
    :return: `True` if `n` corresponds to a concat op.
    :rtype: bool
    """
    dim = try_get_args(n, 1, 'dim', 0)
    if n.op == 'call_function' and n.target == torch.cat and dim == 1:
        return True
    return False


def is_concatenate(n: fx.Node, parent: fx.GraphModule) -> bool:
    """Checks if a `torch.fx.Node` instance corresponds to a concat operation.

    :param n: the target node
    :type n: fx.Node
    :param parent: the parent sub-module
    :type parent: fx.GraphModule
    :return: `True` if `n` corresponds to a concat op.
    :rtype: bool
    """
    if n.op == 'call_function' and n.target == torch.cat:
        return True
    return False


def try_get_args(n: fx.Node, args_idx: int, kwargs_str: str, default: Any) -> Any:
    """Look for an argument in a fx.Node. First looks within n.args, then n.kwargs.
    If not found, returns a default.
    """
    if len(n.args) > args_idx:
        return n.args[args_idx]
    arg = n.kwargs.get(kwargs_str)
    return arg if arg is not None else default


def parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a torch.fx qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    # copied from:
    # https://github.com/pytorch/pytorch/blob/master/torch/fx/experimental/optimization.py
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    """
    Replace the implementation of fx.Node pointed by `node` with `new_module` within the dictionary
    `modules`
    """
    assert isinstance(node.target, str)
    pn, name = parent_name(node.target)
    setattr(modules[pn], name, new_module)


def add_node_properties(mod: fx.GraphModule):
    g = mod.graph
    nx_graph = fx_to_nx_graph(g)
    queue = get_input_nodes(g)

    while queue:
        n = queue.pop(0)

        n.meta['features_propagating'] = is_features_propagating_op(n, mod)
        n.meta['features_defining'] = is_features_defining_op(n, mod)
        n.meta['shared_input_features'] = is_shared_input_features_op(n, mod)
        n.meta['flatten'] = is_flatten(n, mod)
        n.meta['squeeze'] = is_squeeze(n, mod)
        n.meta['features_concatenate'] = is_features_concatenate(n, mod)

        for succ in nx_graph.successors(n):
            queue.append(succ)


def add_features_calculator(mod: fx.GraphModule,
                            extra_rules: List[Callable]
                            ) -> Optional[FeaturesCalculator]:
    # List[((fx.node, fx.GraphModule) -> Optional[FeaturesCalculator])]
    # List[Callable[[fx.node, fx.GraphModule], Optional[FeaturesCalculator]]]
    # List[Callable]
    g = mod.graph
    nx_graph = fx_to_nx_graph(g)
    queue = get_input_nodes(g)

    print(type(extra_rules[0]))
    while queue:
        n = queue.pop(0)
        # skip nodes for which predecessors have not yet been processed completely, we'll come
        # back to them later
        skip_flag = False
        if len(n.all_input_nodes) > 0:
            for i in n.all_input_nodes:
                if 'features_calculator' not in i.meta:
                    skip_flag = True
        if skip_flag:
            continue

        fc = None
        if extra_rules:
            for rule in extra_rules:
                fc = rule(n, mod)
                if fc:
                    break
        if fc:
            n.meta['features_calculator'] = fc
        elif n.meta['flatten']:
            # For flatten ops, the output features are computed as: input_features * spatial_size
            # note that this is NOT simply equal to the output shape if the preceding layer is a
            # NAS-able one, for which some features # could be de-activated
            ifc = n.all_input_nodes[0].meta['features_calculator']
            input_shape = n.all_input_nodes[0].meta['tensor_meta'].shape
            start_dim = try_get_args(n, 1, 'start_dim', 0)
            end_dim = try_get_args(n, 2, 'end_dim', -1)
            assert start_dim != 0 and len(input_shape) - start_dim != 0, \
                "Flattening the batch not supported by PIT"
            # if flatten includes the channels
            if start_dim == 1 or len(input_shape) - start_dim == 1:
                n.meta['features_calculator'] = FlattenFeaturesCalculator
            if start_dim == 1 or len(input_shape) - start_dim == 1:
                flattened_size = math.prod(input_shape[2:end_dim if end_dim != -1 else None])
                n.meta['features_calculator'] = FlattenFeaturesCalculator(ifc, flattened_size)
            else:
                n.meta['features_calculator'] = ifc  # just propagate the features
        elif n.meta['squeeze']:
            # Squeeze is similar to flatten but the pytorch operation is slightly different
            ifc = n.all_input_nodes[0].meta['features_calculator']
            input_shape = n.all_input_nodes[0].meta['tensor_meta'].shape
            dim = try_get_args(n, 1, 'dim', None)
            '''
            if dim is None:
                raise ValueError("Squeeze without dim not supported by PIT")
            assert dim != 0 and len(input_shape) - dim != 0, \
                "Squeezing the batch is not supported by PIT"
            '''
            if dim is None:
                n.meta['features_calculator'] = ifc  # just propagate the features
            elif dim == 1 or len(input_shape) - dim == 1:
                flattened_size = input_shape[2]
                n.meta['features_calculator'] = FlattenFeaturesCalculator(ifc, flattened_size)
            else:
                n.meta['features_calculator'] = ifc  # just propagate the features
        elif n.meta['features_concatenate']:
            # for concatenation over the features axis the number of output features is the sum
            # of the output features of preceding layers as for flatten, this is NOT equal to the
            # input shape of this layer, when one or more predecessors are NAS-able
            # ifc = ConcatFeaturesCalculator([calc_dict[_] for _ in n.all_input_nodes])
            ifc = ConcatFeaturesCalculator(
                [prev.meta['features_calculator'] for prev in n.all_input_nodes]
            )
            n.meta['features_calculator'] = ifc
        elif n.meta['shared_input_features']:
            # for nodes that require identical number of features in all their inputs (e.g., add)
            # we simply assume that we can take any of the output features calculators from
            # predecessors
            # this is enforced for NAS-able layers by the use of shared maskers (see above)
            n.meta['features_calculator'] = n.all_input_nodes[0].meta['features_calculator']
            # just propagate the features
        elif n.meta['features_defining']:
            # these are "static" (i.e., non NAS-able) nodes that alter the number of output
            # features, and hence the number of input features of subsequent layers
            n.meta['features_calculator'] = ConstFeaturesCalculator(n.meta['tensor_meta'].shape[1])
        elif n.meta['features_propagating']:
            # these are nodes that have a single input and n. output features == n. input features
            # so, we just propagate forward the features calculator of the input
            # this also includes PITBatchNorm1d and PITBatchNorm2d
            n.meta['features_calculator'] = n.all_input_nodes[0].meta['features_calculator']
            # they all have a single input
        else:
            raise ValueError("Unsupported node {} (op: {}, target: {})".format(n, n.op, n.target))

        for succ in nx_graph.successors(n):
            queue.append(succ)


def associate_input_features(mod: fx.GraphModule):
    g = mod.graph
    nx_graph = fx_to_nx_graph(g)
    queue = get_input_nodes(g)

    while queue:
        n = queue.pop(0)

        input_nodes = n.all_input_nodes
        if len(input_nodes) > 0:
            prev = input_nodes[0]
            if 'input_features_set_by' in prev.meta:
                if prev.meta['features_defining']:
                    n.meta['input_features_set_by'] = prev
                elif prev.meta['features_propagating']:
                    n.meta['input_features_set_by'] = prev.meta['input_features_set_by']
                else:
                    n.meta['input_features_set_by'] = prev  # CHECK!!
        else:  # input node
            n.meta['input_features_set_by'] = n

        for succ in nx_graph.successors(n):
            queue.append(succ)
