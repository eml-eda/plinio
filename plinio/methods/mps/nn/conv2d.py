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

from typing import Dict, Any, Iterator, Tuple, cast, Union, List, Optional
import torch
import torch.fx as fx
import torch.nn as nn
from ..quant.quantizers import Quantizer, DummyQuantizer
from ..quant.nn import QuantConv2d, QuantList
from .module import MPSModule
from .qtz import MPSType, MPSPerLayerQtz, MPSPerChannelQtz, MPSBiasQtz
from plinio.graph.features_calculation import ConstFeaturesCalculator, FeaturesCalculator
from plinio.cost import CostFn


class MPSConv2d(nn.Conv2d, MPSModule):
    """A nn.Module implementing a Conv2d layer with mixed-precision search support

    :param conv: the inner `nn.Conv2d` layer to be optimized
    :type conv: nn.Conv2d
    :param out_mps_quantizer: activation MPS quantizer
    :type out_mps_quantizer: MPSQtzLayer
    :param w_mps_quantizer: weight MPS quantizer
    :type w_mps_quantizer: Union[MPSQtzLayer, MPSQtzChannel]
    :param b_mps_quantizer: bias MPS quantizer
    :type b_mps_quantizer: MPSQtzBias
    """
    def __init__(self,
                 conv: nn.Conv2d,
                 out_mps_quantizer: MPSPerLayerQtz,
                 w_mps_quantizer: Union[MPSPerLayerQtz, MPSPerChannelQtz],
                 b_mps_quantizer: MPSBiasQtz):
        super(MPSConv2d, self).__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode)
        is_depthwise = (conv.groups == conv.in_channels and conv.groups == conv.out_channels)
        if conv.groups != 1 and (not is_depthwise):
            msg = ('MPS currently supports only full or DepthWise Conv.,'
                   'not other groupwise versions')
            raise AttributeError(msg)
        with torch.no_grad():
            self.weight.copy_(conv.weight)
            if conv.bias is not None:
                self.bias = cast(torch.nn.parameter.Parameter, self.bias)
                self.bias.copy_(conv.bias)
            else:
                self.bias = None
        self.out_mps_quantizer = out_mps_quantizer
        self.w_mps_quantizer = w_mps_quantizer
        if self.bias is not None:
            self.b_mps_quantizer = b_mps_quantizer
        else:
            self.b_mps_quantizer = lambda *args: None  # Do Nothing
        # these two lines will be overwritten later when we process the model graph
        self._input_features_calculator = ConstFeaturesCalculator(conv.in_channels)
        self.in_mps_quantizer = MPSPerLayerQtz((-1,), DummyQuantizer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the mixed-precision NAS-able layer.

        In a nutshell,:
        - Quantize and combine the weight tensor at the different `precision`.
        - Quantize and combine the bias tensor at fixed precision.
        - Compute Conv2d operation using the previous obtained quantized tensors.
        - Quantize and combine the output tensor at the different `precision`.

        :param input: the input activations tensor
        :type input: torch.Tensor
        :return: the output activations tensor
        :rtype: torch.Tensor
        """
        # Quantization of weight and bias
        q_w = self.w_mps_quantizer(self.weight)
        q_b = self.b_mps_quantizer(self.bias,
                                   self.in_mps_quantizer.effective_scale,
                                   self.w_mps_quantizer.effective_scale)
        # Linear operation
        out = self._conv_forward(input, q_w, q_b)
        # Quantization of output
        q_out = self.out_mps_quantizer(out)
        return q_out

    @staticmethod
    def autoimport(n: fx.Node,
                   mod: fx.GraphModule,
                   out_mps_quantizer: MPSPerLayerQtz,
                   w_mps_quantizer: Union[MPSPerLayerQtz, MPSPerChannelQtz],
                   b_mps_quantizer: MPSBiasQtz):
        """Create a new fx.Node relative to a MPSConv2d layer, starting from the fx.Node
        of a nn.Conv2d layer, and replace it into the parent fx.GraphModule

        :param n: a fx.Node corresponding to a nn.Conv2d layer
        :type n: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param out_mps_quantizer: The MPS quantizer to be used for activations
        :type out_mps_quantizer: MPSQtzLayer
        :param w_mps_quantizer: The MPS quantizer to be used for weights
        :type w_mps_quantizer: Union[MPSQtzLayer, MPSQtzChannel]
        :param b_mps_quantizer: The MPS quantizer to be used for biases (if present)
        :type b_mps_quantizer: MPSQtzBias
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != nn.Conv2d:
            msg = f"Trying to generate MPSConv2d from layer of type {type(submodule)}"
            raise TypeError(msg)
        submodule = cast(nn.Conv2d, submodule)
        new_submodule = MPSConv2d(submodule,
                                  out_mps_quantizer,
                                  w_mps_quantizer,
                                  b_mps_quantizer)
        mod.add_submodule(str(n.target), new_submodule)

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a MPSConv2d layer,
        with the selected fake-quantized nn.Conv2d layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != MPSConv2d:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")

        # per-layer search => single precision/quantizer
        if isinstance(submodule.w_mps_quantizer, MPSPerLayerQtz):
            if submodule.bias is not None:
                # TODO: DP not sure why bias quantizer was re-created here,
                # trying to use the existing one now...
                b_quantizer = cast(Quantizer, submodule.b_mps_quantizer.qtz_func)
            else:
                b_quantizer = None
            new_submodule = QuantConv2d(submodule,
                                        submodule.selected_in_quantizer,
                                        submodule.selected_out_quantizer,
                                        cast(Quantizer, submodule.selected_w_quantizer),
                                        b_quantizer)
        # per-channel search => multiple precision/quantizers
        elif isinstance(submodule.w_mps_quantizer, MPSPerChannelQtz):
            selected_w_precision = cast(List[int], submodule.selected_w_precision)
            selected_w_quantizer = cast(List[Quantizer], submodule.selected_w_quantizer)
            nn_list = []
            prec_and_quantz = dict(zip(selected_w_precision, selected_w_quantizer))
            # TODO: debug this. Isn't it doing multiple iterations on the same precision?
            for prec, w_quant in prec_and_quantz.items():
                mask = [c == prec for c in selected_w_precision]
                out_channels = sum(mask)
                if out_channels == 0:  # no out_channels for the current prec
                    continue
                new_conv = nn.Conv2d(submodule.in_channels,
                                     out_channels,
                                     submodule.kernel_size,
                                     submodule.stride,
                                     submodule.padding,
                                     submodule.dilation,
                                     submodule.groups,
                                     submodule.bias is not None,
                                     submodule.padding_mode)
                new_weights = submodule.weight[mask, :, :, :]
                with torch.no_grad():
                    new_conv.weight.copy_(new_weights)
                    if submodule.bias is not None:
                        cast(nn.parameter.Parameter, new_conv.bias).copy_(submodule.bias[mask])
                        # re-create bias quantizer using correct number of channels
                        # TODO: DP: shouldn't we also recreate the w_quantizer with fewer channels?
                        b_quantizer_class = submodule.b_mps_quantizer.quantizer
                        b_quantizer_kwargs = submodule.b_mps_quantizer.quantizer_kwargs
                        b_quantizer_kwargs['cout'] = out_channels
                        b_quantizer = b_quantizer_class(**b_quantizer_kwargs)
                    else:
                        b_quantizer = None
                quant_conv = QuantConv2d(new_conv,
                                         submodule.selected_in_quantizer,
                                         submodule.selected_out_quantizer,
                                         w_quant,
                                         b_quantizer)
                nn_list.append(quant_conv)
            new_submodule = QuantList(nn_list)
        else:
            msg = f'Supported mixed-precision types: {list(MPSType)}'
            raise ValueError(msg)
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
        self.w_mps_quantizer.update_softmax_options(
                temperature, hard, gumbel, disable_sampling)

    def compensate_weights_values(self):
        """Modify the initial weight values of MPSModules compensating the possible presence of
        0-bit among the weights precision
        """
        if 0 in cast(torch.Tensor, self.w_mps_quantizer.precision):
            theta_alpha_rescaling = 0.0
            for i, precision in enumerate(cast(torch.Tensor, self.w_mps_quantizer.precision)):
                if precision != 0:
                    if isinstance(self.w_mps_quantizer, MPSPerChannelQtz):
                        theta_alpha_rescaling += self.w_mps_quantizer.theta_alpha[i][0]
                    else:
                        theta_alpha_rescaling += self.w_mps_quantizer.theta_alpha[i]
            self.weight.data = self.weight.data / theta_alpha_rescaling
        return

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer hyperparameters

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'in_precision': self.selected_in_precision,
            'out_precision': self.selected_out_precision,
            'w_precision': self.selected_w_precision,
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
        w_params = self.w_mps_quantizer.theta_alpha.detach() if post_sampling \
            else self.w_mps_quantizer.alpha.detach()
        return {
            'out_params': out_params,
            'w_params': w_params
        }

    def get_modified_vars(self) -> Dict[str, Any]:
        """Method that returns the modified vars(self) dictionary for the instance, used for
        cost computation

        :return: the modified vars(self) data structure
        :rtype: Dict[str, Any]
        """
        v = dict(vars(self))
        # TODO: detach to be double-checked
        v['in_channels'] = self.input_features_calculator.features.detach()
        v['out_channels'] = self.out_features_eff
        return v

    def get_cost(self, cost_fn: CostFn, out_shape: Dict[str, Any]) -> torch.Tensor:
        """Method that returns the MPSModule cost, given a cost function and
        the layer's "fixed" hyperparameters

        Allows to flexibly handle multiple combinations of weights/act precision

        :param cost_fn: the scalar cost function for a single w/a prec combination
        :type cost_fn: CostFn
        :param out_shape: the output shape information
        :type out_shape: Dict[str, Any]
        :return: the layer cost for each combination of precision
        :rtype: torch.Tensor
        """

        if isinstance(self.w_mps_quantizer, MPSPerLayerQtz):
            w_theta_alpha_array = self.w_mps_quantizer.theta_alpha
        elif isinstance(self.w_mps_quantizer, MPSPerChannelQtz):
            w_theta_alpha_array = self.w_mps_quantizer.theta_alpha.mean(dim=1)
        else:
            msg = f'Supported mixed-precision types: {list(MPSType)}'
            raise ValueError(msg)

        cost = torch.zeros(size=(len(self.in_mps_quantizer.precision), len(self.w_mps_quantizer.precision)),
                           device=self.in_mps_quantizer.precision.device)
        for i, (in_prec, in_theta_alpha) in enumerate(zip(self.in_mps_quantizer.precision, self.in_mps_quantizer.theta_alpha)):
            for j, (w_prec, w_theta_alpha) in enumerate(zip(self.w_mps_quantizer.precision, w_theta_alpha_array)):
                v = self.get_modified_vars()
                v.update(out_shape)
                v['in_format'] = int
                v['w_format'] = int
                v['in_precision'] = in_prec
                v['w_precision'] = w_prec
                v['w_theta_alpha'] = w_theta_alpha
                cost[i][j] = in_theta_alpha * w_theta_alpha * cost_fn(v)
        return cost

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: recures to sub-modules
        :type recurse: bool
        :return: an iterator over the architectural parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        for name, param in self.out_mps_quantizer.named_parameters(
                prfx + "out_mps_quantizer", recurse):
            yield name, param
        for name, param in self.w_mps_quantizer.named_parameters(
                prfx + "w_mps_quantizer", recurse):
            yield name, param
        # we also return the input quantizer, since possible duplicates are removed at mps.py level
        for name, param in self.in_mps_quantizer.named_parameters(
                prfx + "in_mps_quantizer", recurse):
            yield name, param
        # no bias MPS quantizer since it is sharing the parameters of the act and weights

    @property
    def selected_in_precision(self) -> int:
        """Return the selected precision based on the magnitude of `alpha`
        components for input activations

        :return: the selected precision
        :rtype: int
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.in_mps_quantizer.alpha))
            return int(self.in_mps_quantizer.precision[idx])

    @property
    def selected_out_precision(self) -> Union[int, str]:
        """Return the selected precision based on the magnitude of `alpha`
        components for output activations.
        If output is not quantized returns the 'float' string.

        :return: the selected precision
        :rtype: Union[int, str]
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.out_mps_quantizer.alpha))
            return int(self.out_mps_quantizer.precision[idx])

    @property
    def selected_w_precision(self) -> Union[int, List[int]]:
        """Return the selected precision(s) based on the magnitude of `alpha`
        components for weights

        :return: the selected precision(s)
        :rtype: Union[int, List[int]]
        """
        with torch.no_grad():
            if isinstance(self.w_mps_quantizer, MPSPerLayerQtz):
                idx = int(torch.argmax(self.w_mps_quantizer.alpha))
                return int(self.w_mps_quantizer.precision[idx])
            elif isinstance(self.w_mps_quantizer, MPSPerChannelQtz):
                idx = torch.argmax(self.w_mps_quantizer.alpha, dim=0)
                return [int(self.w_mps_quantizer.precision[int(i)]) for i in idx]
            else:
                msg = f'Supported mixed-precision types: {list(MPSType)}'
                raise ValueError(msg)

    @property
    def selected_in_quantizer(self) -> Quantizer:
        """Return the selected quantizer based on the magnitude of `alpha`
        components for input activations

        :return: the selected quantizer
        :rtype: Quantizer
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.in_mps_quantizer.alpha))
            qtz = self.in_mps_quantizer.qtz_funcs[idx]
            qtz = cast(Quantizer, qtz)
            return qtz

    @property
    def selected_out_quantizer(self) -> Quantizer:
        """Return the selected quantizer based on the magnitude of `alpha`
        components for output activations

        :return: the selected quantizer
        :rtype: Quantizer
        """
        with torch.no_grad():
            idx = int(torch.argmax(self.out_mps_quantizer.alpha))
            qtz = self.out_mps_quantizer.qtz_funcs[idx]
            qtz = cast(Quantizer, qtz)
            return qtz

    @property
    def selected_w_quantizer(self) -> Union[Quantizer, List[Quantizer]]:
        """Return the selected quantizer(s) based on the magnitude of `alpha`
        components for weights

        :return: the selected quantizer(s)
        :rtype: Union[Quantizer, List[Quantizer]]
        """
        with torch.no_grad():
            if isinstance(self.w_mps_quantizer, MPSPerLayerQtz):
                idx = int(torch.argmax(self.w_mps_quantizer.alpha))
                qtz = self.w_mps_quantizer.qtz_funcs[idx]
                qtz = cast(Quantizer, qtz)
                return qtz
            elif isinstance(self.w_mps_quantizer, MPSPerChannelQtz):
                idx = torch.argmax(self.w_mps_quantizer.alpha, dim=0)
                qtz = [self.w_mps_quantizer.qtz_funcs[i] for i in idx]
                qtz = cast(List[Quantizer], qtz)
                return qtz
            else:
                msg = f'Supported mixed-precision types: {list(MPSType)}'
                raise ValueError(msg)

    @property
    def out_features_eff(self) -> torch.Tensor:
        """Returns the number of not pruned channels for this layer.

        :return: the number of not pruned channels for this layer.
        :rtype: torch.Tensor
        """
        if isinstance(self.w_mps_quantizer, MPSPerChannelQtz):
            return self.w_mps_quantizer.out_features_eff
        else:
            return torch.tensor(self.out_channels)

    @property
    def number_pruned_channels(self) -> torch.Tensor:
        """Returns the number of pruned channels for this layer.

        :return: the number of pruned channels for this layer.
        :rtype: torch.Tensor
        """
        return self.out_channels - self.out_features_eff

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
