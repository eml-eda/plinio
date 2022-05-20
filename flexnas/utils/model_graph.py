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
from typing import List, Type
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
import networkx as nx


def fx_to_nx_graph(fx_graph: fx.Graph) -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    for n in fx_graph.nodes:
        for i in n.all_input_nodes:
            nx_graph.add_edge(i, n)
    return nx_graph


def get_input_nodes(fx_graph: fx.Graph) -> List[fx.Node]:
    ret = []
    for n in fx_graph.nodes:
        if n.op == 'placeholder':
            ret.append(n)
    return ret


def get_output_nodes(fx_graph: fx.Graph) -> List[fx.Node]:
    ret = []
    for n in fx_graph.nodes:
        if n.op == 'output':
            ret.append(n)
    return ret


def is_layer(n: fx.Node, parent: fx.GraphModule, layer: Type[nn.Module]) -> bool:
    if n.op != 'call_module':
        return False
    return type(parent.get_submodule(str(n.target))) == layer


def is_zero_or_one_input_op(n: fx.Node) -> bool:
    return len(n.all_input_nodes) <= 1


def is_shared_input_features_op(n: fx.Node, parent: fx.GraphModule) -> bool:
    if is_zero_or_one_input_op(n):
        return False
    if n.op == 'call_function':
        if n.target == torch.add:
            return True
        if n.target == operator.add:
            return True
        if n.target == torch.sub:
            return True
        if n.target == operator.sub:
            return True
        # TODO: add others here
    # are there any modules that require same input size? if so, add them below. Same for methods
    # if n.op == 'call_module':
    # if n.op == 'call_method':
    return False


def is_features_defining_op(n: fx.Node, parent: fx.GraphModule) -> bool:
    if n.op == 'placeholder' and len(n.all_input_nodes) == 0:  # input node
        return True
    if n.op == 'call_module':
        submodule = parent.get_submodule(str(n.target))
        if type(submodule) == nn.Conv1d:
            return True
        if type(submodule) == nn.Conv2d:
            return True
        if type(submodule) == nn.Linear:
            return True
    return False


def is_features_propagating_op(n: fx.Node, parent: fx.GraphModule) -> bool:
    if n.op == 'output':
        return True
    if n.op == 'call_module':
        submodule = parent.get_submodule(str(n.target))
        if type(submodule) == nn.BatchNorm1d:
            return True
        if type(submodule) == nn.BatchNorm2d:
            return True
        if type(submodule) == nn.AvgPool1d:
            return True
        if type(submodule) == nn.AvgPool2d:
            return True
        if type(submodule) == nn.MaxPool1d:
            return True
        if type(submodule) == nn.BatchNorm2d:
            return True
        if type(submodule) == nn.Dropout:
            return True
        if type(submodule) == nn.ReLU:
            return True
        if type(submodule) == nn.ReLU6:
            return True
        if type(submodule) == nn.ConstantPad1d:
            return True
        if type(submodule) == nn.ConstantPad2d:
            return True
        # TODO: add others
    if n.op == 'call_function':
        if n.target == F.relu:
            return True
        if n.target == F.relu6:
            return True
        if n.target == F.log_softmax:
            return True
    return False


def is_flatten(n: fx.Node, parent: fx.GraphModule) -> bool:
    if n.op == 'call_method' and n.target == 'flatten':
        return True
    if n.op == 'call_function' and n.target == torch.flatten:
        return True
    return False


def is_concatenate(n: fx.Node, parent: fx.GraphModule) -> bool:
    if n.op == 'call_function' and n.target == torch.cat:
        return True
    return False
