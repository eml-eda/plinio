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
from typing import List, Type, Tuple, Any
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from .utils import try_get_args


def all_output_nodes(n: fx.Node) -> List[fx.Node]:
    """Return the list of successors for a fx.Node since
    torch.fx does not provide this functionality, but only gives input nodes

    :param n: the target node
    :type n:  fx.Node
    :return: the list of successors
    :rtype: List[fx.Node]
    """
    return list(n.users.keys())


def get_graph_inputs(fx_graph: fx.Graph) -> List[fx.Node]:
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


def get_graph_outputs(fx_graph: fx.Graph) -> List[fx.Node]:
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
        # if isinstance(submodule, nn.Sequential):
            # return True
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
        if isinstance(submodule, nn.MaxPool2d):
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


def parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a torch.fx qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    # copied from:
    # https://github.com/pytorch/pytorch/blob/master/torch/fx/experimental/optimization.py
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name
