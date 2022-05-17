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

def zero_or_one_input_op(n: fx.Node) -> bool:
    return len(n.all_input_nodes) <= 1

def is_layer(n: fx.Node, parent: fx.GraphModule, layer: Type[nn.Module]) -> bool:
    if n.op != 'call_module':
        return False
    return type(parent.get_submodule(n.target)) == layer


def shared_input_size_op(n: fx.Node, parent: fx.GraphModule) -> bool:
    if zero_or_one_input_op(n):
        return False
    if n.op == 'call_function':
        if n.target == torch.add: return True
        if n.target == operator.add: return True
        if n.target == torch.sub: return True
        if n.target == operator.sub: return True
        # TODO: add others here
    # are there any modules that require same input size? if so, add them below. Same for methods
    # if n.op == 'call_module':
    # if n.op == 'call_method':
    return False
