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
from typing import Type, Callable, Optional
import torch.nn as nn
import torch.fx as fx
from torch.fx.experimental.optimization import replace_node_module


def fuse_consecutive_layers(mod: fx.GraphModule, first: Type[nn.Module], second: Type[nn.Module],
                            fusion_fn: Callable[[nn.Module, nn.Module], Optional[nn.Module]],
                            in_place: bool = True):
    """Fuses a sequence of two layers in a Torch.fx graph based on their type
    Note that this only works for cases where the second layer has the first one as its only input

    :param mod: a torch.fx.GraphModule with tensor shapes annotations.
    :type mod: fx.GraphModule
    :param first: the class of the first matched layer in the sequence
    :type first: Type[nn.Module]
    :param second: the class of the second matched layer in the sequence
    :type second: Type[nn.Module]
    :param fusion_fn: the function that does the fusion. if in_place is True, it is assumed that the
    function replaces the first node of the sequence
    :type second: Type[nn.Module]
    :param in_place: true if the fusion function works in-place
    :type in_place: bool
    """
    # partially taken from: https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html
    modules = dict(mod.named_modules())
    for node in mod.graph.nodes:
        if node.op != 'call_module':
            continue
        if not isinstance(node.args[0], fx.Node) or node.args[0].op != 'call_module':
            continue
        is_second = isinstance(modules[node.target], second)
        is_prev_first = isinstance(modules[node.args[0].target], first)
        if (is_second and is_prev_first):
            if len(node.args[0].users) > 1:
                raise ValueError("The first layer of the pair to be fused has multiple users")
            if in_place:
                fusion_fn(modules[node.args[0].target], modules[node.target])
            else:
                new_first = fusion_fn(modules[node.args[0].target], modules[node.target])
                assert isinstance(new_first, nn.Module)
                replace_node_module(node.args[0], modules, new_first)
            node.replace_all_uses_with(node.args[0])
            mod.graph.erase_node(node)
    mod.delete_all_unused_submodules()
