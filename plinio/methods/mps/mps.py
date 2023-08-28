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

from typing import Any, Tuple, Type, Iterable, Dict, Iterator, Union, Optional
import torch
import torch.nn as nn

from plinio.methods.dnas_base import DNAS
from plinio.cost import CostSpec, CostFn, params_bit
from plinio.graph.inspection import shapes_dict
from .graph import convert, mps_layer_map
from .nn.module import MPSModule
from .nn.qtz import MPSType

from .quant.quantizers import PACTAct, MinMaxWeight, QuantizerBias

DEFAULT_QINFO = {
    'a_quantizer': {
        'quantizer': PACTAct,
        'kwargs': {},
    },
    'w_quantizer': {
        'quantizer': MinMaxWeight,
        'kwargs': {},
    },
    'b_quantizer': {
        'quantizer': QuantizerBias,
        'kwargs': {
            'precision': 32,
        },
    },
}


DEFAULT_QINFO_INPUT_QUANTIZER = {
    'a_quantizer': {
        'quantizer': PACTAct,
        'kwargs': {
            'init_clip_val': 1
        },
    }
}


class MPS(DNAS):
    """A class that wraps a nn.Module with DNAS-enabled Mixed Precision assigment

    :param model: the inner nn.Module instance optimized by the NAS
    :type model: nn.Module
    :param cost: the cost models(s) used by the NAS, defaults to the number of params.
    :type cost: Union[CostSpec, Dict[str, CostSpec]]
    :param input_shape: the shape of an input tensor, without batch size, required for symbolic
    tracing
    :type input_shape: Tuple[int, ...]
    :param a_precisions: the possible activations' precisions assigment to be explored
    by the NAS
    :type a_precisions: Iterable[int]
    :param w_precisions: the possible weights' precisions assigment to be explored
    by the NAS
    :type w_precisions: Iterable[int]
    :param w_search_type: the mixed precision strategy to be used for weigth
    i.e., `PER_CHANNEL` or `PER_LAYER`. Default is `PER_LAYER`
    :type w_search_type: MPSType
    :param qinfo: dict containing desired quantizers for act, weight and bias
    and their arguments excluding the precision precision
    :type qinfo: Dict
    :param qinfo_input_quanizer: optional dict containing desired quantizers for the network input
    and its arguments excluding the precision precision, defaults to None
    :type qinfo: Dict, optional
    :param autoconvert_layers: should the constructor try to autoconvert NAS-able layers,
    defaults to True
    :type autoconvert_layers: bool, optional
    :param full_cost: True is the cost model should be applied to the entire network, rather
    than just to the NAS-able layers, defaults to False
    :type full_cost: bool, optional
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS,
    defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS,
    defaults to ()
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :raises ValueError: when called with an unsupported regularizer
    :param temperature: the default sampling temperature (for SoftMax/Gumbel SoftMax)
    :type temperature: float, defaults to 1
    :param gumbel_softmax: use Gumbel SoftMax for sampling, instead of a normal SoftMax
    :type gumbel_softmax: bool, optional
    :param hard_softmax: use hard (discretized) SoftMax sampling
    :type hard_softmax: bool, optional
    :param disable_sampling: do not perform any update of the alpha coefficients,
    thus using the saved ones. Useful to perform fine-tuning of the saved model.
    :type disable_sampling: bool, optional
    """
    def __init__(
            self,
            model: nn.Module,
            cost: Union[CostSpec, Dict[str, CostSpec]] = params_bit,
            input_example: Optional[Any] = None,
            input_shape: Optional[Tuple[int, ...]] = None,
            a_precisions: Tuple[int, ...] = (2, 4, 8),
            w_precisions: Tuple[int, ...] = (2, 4, 8),
            w_search_type: MPSType = MPSType.PER_LAYER,
            qinfo: Dict = DEFAULT_QINFO,
            qinfo_input_quantizer: Optional[Dict] = DEFAULT_QINFO_INPUT_QUANTIZER,
            autoconvert_layers: bool = True,
            full_cost: bool = False,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            temperature: float = 1.,
            gumbel_softmax: bool = False,
            hard_softmax: bool = False,
            disable_sampling: bool = False,
            disable_shared_quantizers: bool = False):
        super(MPS, self).__init__(model, cost, input_example, input_shape)
        self.seed, self._leaf_modules, self._unique_leaf_modules = convert(
            model,
            self._input_example,
            'autoimport' if autoconvert_layers else 'import',
            a_precisions,
            w_precisions,
            w_search_type,
            qinfo,
            qinfo_input_quantizer,
            exclude_names,
            exclude_types,
            disable_shared_quantizers)
        self._cost_fn_map = self._create_cost_fn_map()
        self.update_softmax_options(temperature, hard_softmax, gumbel_softmax, disable_sampling)
        if not hard_softmax:
            self.compensate_weights_values()
        self.full_cost = full_cost

    def forward(self, *args: Any) -> torch.Tensor:
        """Forward function for the DNAS model.
        Simply invokes the inner model's forward

        :return: the output tensor
        :rtype: torch.Tensor
        """
        return self.seed.forward(*args)

    @property
    def cost_specification(self) -> Union[CostSpec, Dict[str, CostSpec]]:
        return self._cost_specification

    @cost_specification.setter
    def cost_specification(self, cs: Union[CostSpec, Dict[str, CostSpec]]):
        self._cost_specification = cs
        self._cost_fn_map = self._create_cost_fn_map()

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
        for _, _, layer in self._unique_leaf_modules:
            if isinstance(layer, MPSModule):
                layer.update_softmax_options(temperature, hard, gumbel, disable_sampling)

    def compensate_weights_values(self):
        """Modify the initial weight values of MPSModules compensating the possible presence of
        0-bit among the weights precisions"""
        for _, _, layer in self._unique_leaf_modules:
            if isinstance(layer, MPSModule):
                layer.compensate_weights_values()

    def export(self):
        """Export the architecture found by the NAS as a `quant.nn` module

        The returned model will have the trained weights found during the search filled in, but
        should be fine-tuned for optimal results.

        :return: the precision-assignement found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        mod, _, _ = convert(self.seed, self._input_example, 'export')
        return mod

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Generates a dictionary representation of the precision-assignment found by the NAS.
        Only optimized layers are reported

        :return: a dictionary representation of the precision-assignement found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        arch = {}
        for lname, _, layer in self._unique_leaf_modules:
            if isinstance(layer, MPSModule):
                arch[lname] = layer.summary()
                arch[lname]['type'] = layer.__class__.__name__
        return arch

    def nas_parameters_summary(self, post_sampling: bool = False) -> Dict[str, Dict[str, Any]]:
        """Generates a dictionary representation of the architectural parameters values found by
        the NAS.

        :return: a dictionary representation of the architectural parameters values found by the NAS
        :rtype: Dict[str, Dict[str, any]]"""
        arch = {}
        for lname, _, layer in self._unique_leaf_modules:
            if isinstance(layer, MPSModule):
                arch[lname] = layer.nas_parameters_summary(post_sampling=post_sampling)
                arch[lname]['type'] = layer.__class__.__name__
        return arch

    def alpha_summary(self) -> Dict[str, Dict[str, Any]]:
        """DEPRECATED: use nas_parameters_summary(post_sampling=False)
        :rtype: Dict[str, Dict[str, any]]"""
        return self.nas_parameters_summary(post_sampling=False)

    def theta_alpha_summary(self) -> Dict[str, Dict[str, Any]]:
        """DEPRECATED: use nas_parameters_summary(post_sampling=True)
        :rtype: Dict[str, Dict[str, any]]"""
        return self.nas_parameters_summary(post_sampling=True)

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of the NAS, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API, but PITLayers never have sub-layers
        :type recurse: bool
        :return: an iterator over the architectural parameters of the NAS
        :rtype: Iterator[nn.Parameter]
        """
        included = set()
        for lname, layer in self.named_modules():
            if isinstance(layer, MPSModule):
                prfx = prefix
                prfx += "." if len(prefix) > 0 else ""
                prfx += lname
                for name, param in layer.named_nas_parameters(prefix=prfx, recurse=recurse):
                    # avoid duplicates (e.g. shared params)
                    if param not in included:
                        included.add(param)
                        yield name, param

    def named_net_parameters(
            self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the inner network parameters, EXCEPT the NAS architectural
        parameters, yielding both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API, not actually used
        :type recurse: bool
        :return: an iterator over the inner network parameters
        :rtype: Iterator[nn.Parameter]
        """
        exclude = set(_[1] for _ in self.named_nas_parameters())
        for name, param in self.named_parameters(prefix=prefix, recurse=recurse):
            if param not in exclude:
                yield name, param

    def _get_single_cost(self, cost_spec: CostSpec,
                         cost_fn_map: Dict[str, CostFn]) -> torch.Tensor:
        """Private method to compute a single cost value"""
        cost = torch.tensor(0, dtype=torch.float32)
        target_list = self._unique_leaf_modules if cost_spec.shared else self._leaf_modules
        for lname, node, layer in target_list:
            if isinstance(layer, MPSModule):
                # a list of layer parameters for each combination of supported precision
                v = layer.get_modified_vars()
                v.update(shapes_dict(node))
                # TODO: replace torch.sum() with configurable reduce_fn to support ODiMO
                cost = cost + torch.sum(cost_fn_map[lname](v), dim=None)
            elif self.full_cost:
                # TODO: this is constant and can be pre-computed for efficiency
                # TODO: should we add default bitwidth and format for non-MPS layers or not?
                v = vars(layer)
                v.update(shapes_dict(node))
                cost = cost + cost_fn_map[lname](v)
        return cost

    def _single_cost_fn_map(self, c: CostSpec) -> Dict[str, CostFn]:
        """MPS-specific creator of {layertype, cost_fn} maps based on a CostSpec."""
        cost_fn_map = {}
        for lname, _, layer in self._unique_leaf_modules:
            if isinstance(layer, MPSModule):
                # get original layer type from MPSModule type
                # Note: not all MPSModules have a corresponding nn.Module. Notable cases
                # are MPSIdentity and MPSAdd. For now, we skip those because CostSpecs require
                # a nn.Module as "pattern". Will be fixed in a future release. This is why
                # we have a try/except here
                try:
                    # didnt' find a more readable way to implement this compactly
                    t = list(mps_layer_map.keys())[list(mps_layer_map.values()).index(type(layer))]
                    # equally unreadable alternative
                    # t = layer.__class__.__bases__[0]
                except ValueError:
                    # if we didn't find a corresponding nn.Module, just associate the cost of a
                    # generic nn.Module (which will likely be either 0 or trigger an exception
                    # depending on the CostSpec)
                    t = nn.Module
            else:
                t = type(layer)
            cost_fn_map[lname] = c[(t, vars(layer))]
        return cost_fn_map

    def __str__(self):
        """Prints the precision-assignent found by the NAS to screen

        :return: a str representation of the current architecture and
        its precision-assignement
        :rtype: str
        """
        arch = self.summary()
        return str(arch)
