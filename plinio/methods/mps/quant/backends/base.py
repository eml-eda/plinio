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

from enum import Enum, auto
from typing import cast, Dict, Tuple, Type

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.experimental.optimization import replace_node_module

from plinio.graph.inspection import is_function, is_layer
from plinio.methods.mps.quant.quantizers import Quantizer
import plinio.methods.mps.quant.nn as qnn


class Backend(Enum):
    MATCH = auto()
    DIANA = auto()
    MAUPITI = auto()
    # Add new backends here
    ONNX = auto()

    @classmethod
    def has_entry(cls, value) -> bool:
        return value.name in cls.__members__


class IntegerizationTracer(fx.Tracer):
    """Consider layers contained in `target_layers` as leaf modules.

    :param target_layers: modules that should be considered as a leaf
    :type target_layers: Tuple[Type[nn.Module]]
    """

    def __init__(self, target_layers: Tuple[Type[nn.Module], ...]):
        super().__init__()
        self.target_layers = target_layers

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, self.target_layers):
            return True
        if isinstance(m, Quantizer):
            return True
        else:
            return m.__module__.startswith("torch.nn") and not isinstance(
                m, torch.nn.Sequential
            )


# N.B., ugly but is needed to avoid circular import
def get_map():
    from .match.base import match_layer_map
    from .maupiti.base import maupiti_layer_map
    from .onnx.base import onnx_layer_map

    # Add new supported backends here:
    maps = {
        "match": match_layer_map,
        "maupiti": maupiti_layer_map,
        "onnx": onnx_layer_map,
    }
    return maps


def backend_factory(layer: nn.Module, backend: Backend) -> nn.Module:
    """Depending on the specific `layer` and specified `backend` returns
    the appropriate backend-specific layer implementation.

    :param layer: the layer to be converted
    :type layer: nn.Module
    :param backend: the backend to be used
    :type backend: Backend
    :param backend: the specific backend to be used
    :type backend: Backend
    :return: the backend specific layer implementation
    :rtype: nn.Module
    """
    if Backend.has_entry(backend):
        backend_name = backend.name.lower()
        maps = get_map()
        layer_map = maps[backend_name]
        layer_map = cast(Dict, layer_map)
        if type(layer) in layer_map.keys():
            return layer_map[type(layer)]
        else:
            msg = f"Layer of type {type(layer)} is not supported by {backend_name} backend."
            raise ValueError(msg)
    else:
        msg = f"The {backend} is not supported."
        raise ValueError(msg)


def integerize_arch(
    model: nn.Module,
    backend: Backend,
    backend_kwargs: Dict = {},
    remove_input_quantizer: bool = False,
) -> nn.Module:
    """Convert a Fake Quantized model to a backend specific integer model

    :param model: the input Fake Quantized model
    :type model: nn.Module
    :param backend: the backend to be used
    :type backend: Backend
    :param backend_kwargs: additional backend-specific arguments
    :type backend_kwargs: Dict
    :param remove_input_quantizer: remove the input quantizer
    :type remove_input_quantizer: bool
    """
    assert (
        (backend == Backend.MAUPITI) or
        (backend == Backend.ONNX) or
        (not remove_input_quantizer and backend != Backend.MAUPITI)
    ), "Remove input quantizer is only supported for MAUPITI backend"
    if Backend.has_entry(backend):
        backend_name = backend.name.lower()
        maps = get_map()
        layer_map = maps[backend_name]
        layer_map = cast(Dict, layer_map)
    else:
        msg = f"The {backend} is not supported."
        raise ValueError(msg)
    target_layers = tuple(layer_map.keys())
    tracer = IntegerizationTracer(target_layers=target_layers)
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    modules = dict(mod.named_modules())
    for n in mod.graph.nodes:
        m = modules.get(n.target)
        # The input quantizer is kept and forced to return integer outputs
        if (
            "input_quantizer_out_quantizer" in n.name
        ):  # TODO: name-dependent, find better way
            m = cast(Quantizer, m)
            m.dequantize = False
        # Target layers are automagically converted with their backend-specific ver
        if isinstance(m, target_layers):
            m = cast(qnn.QuantModule, m)
            m.export(n, mod, backend, backend_kwargs)
    if backend == Backend.MAUPITI or backend == Backend.ONNX:
        # Remove relu
        mod = remove_relu(mod)
        # Remove input quantizer
        if remove_input_quantizer:
            mod = remove_inp_quantizer(mod)
    mod.delete_all_unused_submodules()
    mod.graph.lint()
    mod.recompile()
    return mod


def remove_inp_quantizer(mod: nn.Module) -> nn.Module:
    """MATCH does not expect an input quantizer."""
    if not isinstance(mod, fx.GraphModule):
        msg = f"Input is of type {type(mod)} instead of fx.GraphModule"
        raise ValueError(msg)
    mod = cast(fx.GraphModule, mod)
    modules = dict(mod.named_modules())
    for node in mod.graph.nodes:
        if (
            "input_quantizer_out_quantizer" in node.name
        ):  # TODO: name-dependent, find better way
            inp_qtz = nn.Identity()
            replace_node_module(node, modules, inp_qtz)
            node.replace_all_uses_with(node.args[0])
            mod.graph.erase_node(node)
    mod.delete_all_unused_submodules()
    mod.graph.lint()
    mod.recompile()
    return mod


def remove_relu(mod: nn.Module) -> nn.Module:
    """ReLU is already implemented as clip function in match.nn modules, then we
    can remove explicit calls to F.relu, torch.relu and nn.ReLU
    """
    if not isinstance(mod, fx.GraphModule):
        msg = f"Input is of type {type(mod)} instead of fx.GraphModule"
        raise ValueError(msg)
    mod = cast(fx.GraphModule, mod)
    for n in mod.graph.nodes:
        if is_function(
            n,
            (
                F.relu,
                torch.relu,
            ),
        ) or is_layer(n, mod, (nn.ReLU,)):
            assert len(n.all_input_nodes) == 1
            inp_node = n.all_input_nodes[0]
            new_submodule = nn.Identity()
            name = str(n) + "_" + str(n.all_input_nodes) + "_identity"
            mod.add_submodule(name, new_submodule)
            with mod.graph.inserting_after(n):
                new_node = mod.graph.call_module(name, args=(inp_node,))
                # Copy metadata
                new_node.meta = {}
                new_node.meta = n.meta
                # Insert node
                n.replace_all_uses_with(new_node)
                new_node.replace_input_with(new_node, inp_node)
            n.args = ()
            mod.graph.erase_node(n)
    mod.delete_all_unused_submodules()
    mod.graph.lint()
    mod.recompile()
    return mod
