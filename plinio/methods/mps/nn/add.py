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
from typing import Dict, Any, Iterator, Tuple, cast, Union, Optional
import torch
import torch.fx as fx
import torch.nn as nn
from ..quant.quantizers import Quantizer, DummyQuantizer
from ..quant.nn import QuantIdentity
from .module import MPSModule
from .qtz import MPSPerLayerQtz, MPSPerChannelQtz, MPSBiasQtz
from plinio.graph.features_calculation import ConstFeaturesCalculator, FeaturesCalculator


class MPSAdd(nn.Module, MPSModule):
    """A nn.Module implementing a sum layer with mixed-precision search support

    :param out_mps_quantizer: activation MPS quantizer
    :type out_mps_quantizer: MPSQtzLayer
    """
    def __init__(self,
                 out_mps_quantizer: MPSPerLayerQtz):
        super(MPSAdd, self).__init__()
        self.out_mps_quantizer = out_mps_quantizer
        # these two lines will be overwritten later when we process the model graph
        self._input_features_calculator = ConstFeaturesCalculator(1)
        self.in_mps_quantizer = MPSPerLayerQtz((32,), DummyQuantizer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the mixed-precision NAS-able layer.

        In a nutshell, sum together the input tensors and the quantize the
        result at the different `precisions`.

        :param input: the list of input activations tensor
        :type input: List[torch.Tensor}
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        q_out = self.out_mps_quantizer(input)
        return q_out

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
        :param w_mps_quantizer: The MPS quantizer to be used for weights (ignored for add)
        :type w_mps_quantizer: Union[MPSQtzLayer, MPSQtzChannel]
        :param b_mps_quantizer: The MPS quantizer to be used for biases (ignored for add)
        :type b_mps_quantizer: MPSBiasQtz
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        if not isinstance(n.target, (type(operator.add), type(torch.add))):
            msg = f"Trying to generate MPSAdd from layer of type {type(n.target)}"
            raise TypeError(msg)
        new_submodule = MPSAdd(out_mps_quantizer)
        name = str(n) + '_' + str(n.all_input_nodes) + '_quant'
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
        selected_quantizer = submodule.selected_out_quantizer
        # TODO: DP this is exported as QuantIdentity. Is it correct? Doesn't seem so...
        new_submodule = QuantIdentity(
            selected_quantizer
        )
        mod.add_submodule(str(n.target), new_submodule)

    def update_softmax_options(
            self,
            temperature: Optional[float] = None,
            hard: Optional[bool] = None,
            gumbel: Optional[bool] = None,
            disable_sampling: Optional[bool] = None):
        """Set the flags to choose between the softmax, the hard and soft Gumbel-softmax
        and the sampling disabling of the architectural coefficients in the quantizers

        :param temperature: SoftMax temperature
        :type temperature: Optional[float]
        :param hard: Hard vs Soft sampling
        :type hard: Optional[bool]
        :param gumbel: Gumbel-softmax vs standard softmax
        :type gumbel: Optional[bool]
        :param disable_sampling: disable the sampling of the architectural coefficients in the
        forward pass
        :type disable_sampling: Optional[bool]
        """
        self.out_mps_quantizer.update_softmax_options(
                temperature, hard, gumbel, disable_sampling)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'out_precision': self.selected_out_precision,
        }

    def nas_parameters_summary(self, post_sampling: bool = False) -> Dict[str, Any]:
        """Export a dictionary with the current NAS parameters of this layer

        :param post_sampling: true to get the post-softmax NAS parameters
        :type post_sofmatx: bool
        :return: a dictionary containing the current NAS parameters values
        :rtype: Dict[str, Any]
        """
        out_params = self.out_mps_quantizer.theta_alpha.detach() if post_sampling \
            else self.out_mps_quantizer.alpha.detach()
        return {
            'out_params': out_params
        }

    def get_modified_vars(self) -> Iterator[Dict[str, Any]]:
        """Method that returns the modified vars(self) dictionary for the instance, for each
        combination of supported precision, used for cost computation

        :return: an iterator over the modified vars(self) data structures
        :rtype: Iterator[Dict[str, Any]]
        """
        for i, a_prec in enumerate(self.in_mps_quantizer.precisions):
            v = dict(vars(self))
            v['in_precision'] = a_prec
            v['in_format'] = int
            # downscale the input_channels times the probability of using that
            # input precision
            # TODO: detach to be double-checked
            v['in_channels'] = (self.input_features_calculator.features.detach() *
                                self.in_mps_quantizer.theta_alpha[i])
            # TODO: verify that it's correct to use out_features_eff here, differently from
            # conv/linear layers.
            v['out_channels'] = self.out_features_eff
            yield v

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: recurse to sub-modules
        :type recurse: bool
        :return: an iterator over the architectural parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        for name, param in self.out_mps_quantizer.named_parameters(
                prfx + "out_mps_quantizer", recurse):
            yield name, param

    @property
    def selected_out_precision(self) -> int:
        """Return the selected precision based on the magnitude of `alpha`
        components

        :return: the selected precision
        :rtype: int
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.out_mps_quantizer.alpha))
            return int(self.out_mps_quantizer.precisions[idx])

    @property
    def selected_out_quantizer(self) -> Quantizer:
        """Return the selected quantizer based on the magnitude of `alpha`
        components

        :return: the selected precision
        :rtype: int
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.out_mps_quantizer.alpha))
            qtz = self.out_mps_quantizer.qtz_funcs[idx]
            qtz = cast(Quantizer, qtz)
            return qtz

    @property
    def out_features_eff(self) -> torch.Tensor:
        """Returns the number of channels for this layer (constant).

        :return: the number of channels for this layer.
        :rtype: torch.Tensor
        """
        return self.input_features_calculator.features

    @property
    def input_features_calculator(self) -> FeaturesCalculator:
        """Returns the `FeaturesCalculator` instance that computes the number of input features for
        this layer.

        :return: the `FeaturesCalculator` instance that computes the number of input features for
        this layer.
        :rtype: FeaturesCalculator
        """
        return self._input_features_calculator

    @input_features_calculator.setter
    def input_features_calculator(self, calc: FeaturesCalculator):
        """Set the `FeaturesCalculator` instance that computes the number of input features for
        this layer.

        :param calc: the `FeaturesCalculator` instance that computes the number of input features
        for this layer
        :type calc: FeaturesCalculator
        """
        calc.register(self)
        self._input_features_calculator = calc
