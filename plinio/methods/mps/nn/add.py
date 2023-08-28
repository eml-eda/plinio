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

import operator
from typing import Dict, Any, Iterator, Union, cast
import torch
import torch.fx as fx
from ..quant.nn import QuantIdentity
from .identity import MPSIdentity
from .qtz import MPSPerLayerQtz, MPSPerChannelQtz, MPSBiasQtz


class MPSAdd(MPSIdentity):
    """A nn.Module implementing mixed-precision search to the output of a sum layer
    Identical to MPSIdentity, but inserted after a torch.add (or similar) operation

    :param out_mps_quantizer: activation MPS quantizer
    :type out_mps_quantizer: MPSQtzLayer
    """
    def __init__(self,
                 out_mps_quantizer: MPSPerLayerQtz):
        super(MPSAdd, self).__init__(out_mps_quantizer)

    @staticmethod
    def autoimport(n: fx.Node,
                   mod: fx.GraphModule,
                   out_mps_quantizer: MPSPerLayerQtz,
                   w_mps_quantizer: Union[MPSPerLayerQtz, MPSPerChannelQtz],
                   b_mps_quantizer: MPSBiasQtz,
                   ):
        """Create a new fx.Node relative to a MPSAdd layer, starting from the fx.Node
        of a nn.Module layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to an add operation
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param out_mps_quantizer: The MPS quantizer to be used for activations
        :type out_mps_quantizer: MPSQtzLayer
        :param w_mps_quantizer: The MPS quantizer to be used for weights (ignored for this module)
        :type w_mps_quantizer: Union[MPSQtzLayer, MPSQtzChannel]
        :param b_mps_quantizer: The MPS quantizer to be used for biases (ignored for this module)
        :type b_mps_quantizer: MPSBiasQtz
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        if not isinstance(n.target, (type(operator.add), type(torch.add))):
            msg = f"Trying to generate MPSAdd from layer of type {type(n.target)}"
            raise TypeError(msg)
        new_submodule = MPSAdd(out_mps_quantizer)
        name = str('add_')
        name += str(n.all_input_nodes).replace('[', '').replace(']', '').replace(', ', '_')
        name += '_quant'
        mod.add_submodule(name, new_submodule)
        with mod.graph.inserting_after(n):
            new_node = mod.graph.call_module(
                name,
                args=(n,)
            )
            # Copy metadata
            new_node.meta = {}
            new_node.meta = n.meta
            # Insert node
            n.replace_all_uses_with(new_node)
            new_node.replace_input_with(new_node, n)

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a MPSAdd layer,
        with the selected fake-quantized addition layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != MPSAdd:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")
        new_submodule = QuantIdentity(
            submodule.selected_out_quantizer
        )
        mod.add_submodule(str(n.target), new_submodule)

    def get_modified_vars(self) -> Dict[str, Any]:
        """Method that returns the modified vars(self) dictionary for the instance, for each
        combination of supported precision, used for cost computation

        :return: an iterator over the modified vars(self) data structures
        :rtype: Dict[str, Any]
        """
        v = dict(vars(self))
        v['in_precision'] = self.in_mps_quantizer.precisions
        v['in_format'] = int
        # downscale the input_channels times the probability of using that
        # input precision
        # TODO: detach to be double-checked
        v['in_channels'] = (self.input_features_calculator.features.detach() *
                            self.in_mps_quantizer.theta_alpha)
        # TODO: verify that it's correct to use out_features_eff here, differently from
        # conv/linear layers.
        v['out_channels'] = self.out_features_eff
        return v
