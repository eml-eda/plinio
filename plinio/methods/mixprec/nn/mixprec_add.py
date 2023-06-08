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
from typing import Dict, Any, Optional, Iterator, Tuple, Type, cast
import torch
import torch.fx as fx
import torch.nn as nn
from ..quant.quantizers import Quantizer
from ..quant.nn import Quant_Identity
from .mixprec_module import MixPrecModule, MixPrecType
from .mixprec_qtz import MixPrec_Qtz_Layer
from plinio.graph.features_calculation import ConstFeaturesCalculator, FeaturesCalculator


class MixPrec_Add(nn.Module, MixPrecModule):
    """A nn.Module implementing a sum layer with mixed-precision search support

    :param precisions: different bitwitdth alternatives among which perform search
    :type precisions: Tuple[int, ...]
    :param quantizer: input tensor quantizer
    :type quantizer: MixPrec_Qtz_Layer
    """
    def __init__(self,
                 precisions: Tuple[int, ...],
                 cout: int,
                 quantizer: MixPrec_Qtz_Layer):
        super(MixPrec_Add, self).__init__()
        self.in_features = cout
        self.precisions = precisions
        self.mixprec_a_quantizer = quantizer
        # this will be overwritten later when we process the model graph
        self._input_features_calculator = ConstFeaturesCalculator(self.in_features)
        # this will be overwritten later when we process the model graph
        self.input_quantizer = cast(MixPrec_Qtz_Layer, nn.Identity())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the mixed-precision NAS-able layer.

        In a nutshell, sum together the input tensors and the quantize the
        result at the different `precisions`.

        :param input: the list of input activations tensor
        :type input: List[torch.Tensor}
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        q_out = self.mixprec_a_quantizer(input)
        return q_out

    def set_temperatures(self, value):
        """Set the quantizers softmax temperature value

        :param value
        :type float
        """
        with torch.no_grad():
            self.mixprec_a_quantizer.temperature = torch.tensor(value, dtype=torch.float32)

    def update_softmax_options(self, gumbel_softmax, hard_softmax, disable_sampling):
        """Set the flags to choose between the softmax, the hard and soft Gumbel-softmax
        and the sampling disabling of the architectural coefficients in the quantizers

        :param gumbel_softmax: whether to use the Gumbel-softmax instead of the softmax
        :type gumbel_softmax: bool
        :param hard_softmax: whether to use the hard version of the Gumbel-softmax
        (param gumbel_softmax must be equal to True)
        :type gumbel_softmax: bool
        :param disable_sampling: whether to disable the sampling of the architectural
        coefficients in the forward pass
        :type disable_sampling: bool
        """
        self.mixprec_a_quantizer.update_softmax_options(gumbel_softmax,
                                                        hard_softmax,
                                                        disable_sampling)

    def update_input_quantizer(self, qtz: MixPrec_Qtz_Layer):
        """Set the `MixPrec_Qtz_Layer` for input activations calculation

        :param qtz: the `MixPrec_Qtz_Layer` instance that computes mixprec quantized
        versions of the input activations
        :type qtz: MixPrec_Qtz_Layer
        """
        self.input_quantizer = qtz

    @staticmethod
    def autoimport(n: fx.Node,
                   mod: fx.GraphModule,
                   w_mixprec_type: MixPrecType,
                   a_precisions: Tuple[int, ...],
                   w_precisions: Tuple[int, ...],
                   b_quantizer: Type[Quantizer],
                   aw_q: Tuple[Quantizer, Quantizer],
                   b_quantizer_kwargs: Dict = {}
                   ) -> Optional[Quantizer]:
        """Create a new fx.Node relative to a MixPrecModule layer, starting from the fx.Node
        of a nn.Module layer, and replace it into the parent fx.GraphModule

        Also returns a quantizer in case it needs to be shared with other layers

        :param n: a fx.Node corresponding to a nn.ReLU layer, with shape annotations
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param w_mixprec_type: the mixed precision strategy to be used for weigth
        i.e., `PER_CHANNEL` or `PER_LAYER`.
        :type w_mixprec_type: MixPrecType
        :param a_precisions: The precisions to be explored for activations
        :type a_precisions: Tuple[int, ...]
        :param w_precisions: The precisions to be explored for weights
        :type w_precisions: Tuple[int, ...]
        :param b_quantizer: The quantizer to be used for biases
        :type b_quantizer: Type[Quantizer]
        :param aw_q: A tuple containing respecitvely activation and weight quantizers
        :type aw_q: Tuple[Quantizer, Quantizer]
        :param b_quantizer_kwargs: bias quantizer kwargs, if no kwargs are passed default is used
        :type b_quantizer_kwargs: Dict
        :raises TypeError: if the input fx.Node is not of the correct type
        :return: the updated shared quantizer
        :rtype: Optional[Quantizer]
        """
        if not isinstance(n.target, (type(operator.add), type(torch.add))):
            msg = f"Trying to generate MixPrec_Add from layer of type {type(n.target)}"
            raise TypeError(msg)
        mixprec_quantizer = aw_q[0]

        mixprec_quantizer = cast(MixPrec_Qtz_Layer, mixprec_quantizer)
        cout = n.meta['tensor_meta'].shape[1]
        new_submodule = MixPrec_Add(a_precisions, cout, mixprec_quantizer)
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
        """Replaces a fx.Node corresponding to a MixPrec_Identity layer,
        with the selected fake-quantized nn.Identity layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != MixPrec_Add:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")
        selected_precision = submodule.selected_precision
        selected_precision = cast(int, selected_precision)
        selected_quantizer = submodule.selected_quantizer
        selected_quantizer = cast(Quantizer, selected_quantizer)
        new_submodule = Quant_Identity(
            selected_precision,
            selected_quantizer
        )
        mod.add_submodule(str(n.target), new_submodule)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'precision': self.selected_precision,
        }

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API,
        but MixPrecModule never have sub-layers TODO: check if true
        :type recurse: bool
        :return: an iterator over the architectural parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        for name, param in self.mixprec_a_quantizer.named_parameters(
                prfx + "mixprec_quantizer", recurse):
            yield name, param

    @property
    def selected_precision(self) -> int:
        """Return the selected precision based on the magnitude of `alpha_prec`
        components

        :return: the selected precision
        :rtype: int
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.mixprec_a_quantizer.alpha_prec))
            return self.precisions[idx]

    @property
    def selected_quantizer(self) -> Type[Quantizer]:
        """Return the selected quantizer based on the magnitude of `alpha_prec`
        components

        :return: the selected precision
        :rtype: int
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.mixprec_a_quantizer.alpha_prec))
            qtz = self.mixprec_a_quantizer.mix_qtz[idx]
            qtz = cast(Type[Quantizer], qtz)
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
