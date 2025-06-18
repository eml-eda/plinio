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
# * Author:  Francesco Daghero <francesco.daghero@polito.it>                             *
# *----------------------------------------------------------------------------*


import torch
from typing import cast, Iterable, Type, Tuple, Dict, Callable, Any

import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from .nn.linear import NMPruningLinear
from .nn.conv2d import NMPruningConv2d
from .nn.module import NMPruningModule

from plinio.graph.annotation import add_node_properties
from plinio.graph.inspection import is_layer, get_graph_outputs, is_inherited_layer, \
    get_graph_inputs, is_function, named_leaf_modules, uniquify_leaf_modules
from plinio.graph.utils import NamedLeafModules


nmpruning_layer_map: Dict[Type[nn.Module], Type[NMPruningModule]] = {
    nn.Linear: NMPruningLinear,
    nn.Conv2d: NMPruningConv2d,
}

nmpruning_function_map: Dict[Callable, Type[NMPruningModule]] = {}

class NMPruningTracer(fx.Tracer):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, NMPruningModule):
            return True
        else:
            return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)


def convert(model: nn.Module,
            input_example: Any,
            conversion_type: str,
            n: int,
            m: int,
            pruning_decay: float,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            ) -> Tuple[nn.Module, NamedLeafModules, NamedLeafModules]:

    """Converts a nn.Module, to/from "Prunable-able" format for the pruning search method

    :param model: the input nn.Module
    :type model: nn.Module
    :param input_example: an input with the same shape and type of the seed's input, used
    for symbolic tracing
    :type input_example: Any
    :param conversion_type: a string specifying the type of conversion. Supported types:
    ('import', 'autoimport', 'export')
    :type conversion_type: str
    :param n: the number of non-zero parameters every m weights
    :type n: int
    :param m: the group of weights considered for pruning
    :type m: int
    :param pruning_decay: the decay factor for the pruning mask
    :type pruning_decay: float
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :raises ValueError: for unsupported conversion types
    :return: the converted model, and two lists of all (or all unique) leaf modules for
    the NAS
    :rtype: Tuple[nn.Module, NamedLeafModule, NamedLeafModules]
    """

    if conversion_type not in ('import', 'autoimport', 'export'):
        raise ValueError("Unsupported conversion type {}".format(conversion_type))

    # Symbolic Tracing
    tracer = NMPruningTracer()
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    if len(get_graph_inputs(mod.graph)) > 1:
        ShapeProp(mod).propagate(*input_example)
    else:
        ShapeProp(mod).propagate(input_example)
    add_node_properties(mod)

    convert_layers(mod, conversion_type,  n, m, pruning_decay, exclude_names, exclude_types)

    mod.graph.lint()
    mod.recompile()
    nlf = named_leaf_modules(mod)
    ulf = uniquify_leaf_modules(nlf)

    return mod, nlf, ulf


def convert_layers(mod: fx.GraphModule,
                   conversion_type: str,
                   n: int,
                   m: int,
                   pruning_decay: float,
                   exclude_names: Iterable[str],
                   exclude_types: Iterable[Type[nn.Module]],
                   ):
    """Replaces target layers with their prunable version, or vice versa. Layer conversion
    is implemented as a reverse BFS on the model graph.

    :param mod: a torch.fx.GraphModule with tensor shapes annotations. Those are needed to
    determine the macs reg loss.
    :type mod: fx.GraphModule
    :param conversion_type: a string specifying the type of conversion
    :type conversion_type: str
    :param n: the number of non-zero parameters every m weights
    :type n: int
    :param m: the group of weights considered for pruning
    :type m: int
    :param pruning_decay: the decay factor for the pruning mask
    :type pruning_decay: float
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    :type exclude_types: Iterable[Type[nn.Module]], optional
    """

    g = mod.graph
    queue = get_graph_outputs(g)
    visited = []
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        if conversion_type == 'autoimport':
            autoimport_node(node, mod, n, m, pruning_decay, exclude_names, exclude_types)
        if conversion_type == 'export':
            export_node(node, mod, exclude_names, exclude_types)
        for pred in node.all_input_nodes:
            queue.append(pred)
        visited.append(node)
    return

# N.B., same as PIT -> Remove duplicate
def exclude(n: fx.Node, mod: fx.GraphModule,
            exclude_names: Iterable[str],
            exclude_types: Iterable[Type[nn.Module]],
            ) -> bool:
    """Returns True if a submodule should be excluded from the pruning, based on the
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


def autoimport_node(node: fx.Node,
                    mod: fx.GraphModule,
                    n: int,
                    m: int,
                    pruning_decay: float,
                    exclude_names: Iterable[str],
                    exclude_types: Iterable[Type[nn.Module]]
                    ):
    """Rewrites a fx.GraphModule node replacing a sub-module instance corresponding to a standard
    nn.Module with its corresponding prunable version.

    :param n: the node to be rewritten
    :type n: fx.Node
    :param mod: the parent module, where the new node has to be optionally inserted
    :type mod: fx.GraphModule
    :param exclude_names: the names of `model` submodules that should be ignored by the pruning
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the pruning
    :type exclude_types: Iterable[Type[nn.Module]], optional
    """
    if is_layer(node, mod, tuple(nmpruning_layer_map.keys())) and \
       not exclude(node, mod, exclude_names, exclude_types):
        module_type = nmpruning_layer_map[type(mod.get_submodule(str(node.target)))]
    elif is_function(node, tuple(nmpruning_function_map.keys())):
        raise NotImplementedError("Per-node autoimport for functions is not supported yet")
        # module_type = nmpruning_function_map[cast(Callable, node.target)]
    else:
        return

    module_type.autoimport(node,
                           mod,
                           n,
                           m,
                           pruning_decay)
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
    if is_inherited_layer(n, mod, (NMPruningModule,)):
        if exclude(n, mod, exclude_names, exclude_types):
            return
        layer = cast(NMPruningModule, mod.get_submodule(str(n.target)))
        layer.export(n, mod)