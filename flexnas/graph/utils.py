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
from typing import Any
import torch.fx as fx
import networkx as nx


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


def try_get_args(n: fx.Node, args_idx: int, kwargs_str: str, default: Any) -> Any:
    """Look for an argument in a fx.Node. First looks within n.args, then n.kwargs.
    If not found, returns a default.
    """
    if len(n.args) > args_idx:
        return n.args[args_idx]
    arg = n.kwargs.get(kwargs_str)
    return arg if arg is not None else default
