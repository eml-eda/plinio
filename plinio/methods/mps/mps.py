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

from typing import Any, Tuple, Type, Iterable, Dict, cast, Iterator, Union, Optional
import torch
import torch.nn as nn

from plinio.methods.dnas_base import DNAS
from .graph import convert
from .nn.module import MPSModule
from .nn.qtz import MPSType, MPSQtzLayer, MPSQtzChannel
from plinio.cost import CostSpec, CostFn, bit_params

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
            'num_bits': 32,
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
    and their arguments excluding the num_bits precision
    :type qinfo: Dict
    :param regularizer: the name of the model cost regularizer used by the NAS, defaults to 'size'
    :type regularizer: Optional[str], optional
    :param autoconvert_layers: should the constructor try to autoconvert NAS-able layers,
    defaults to True
    :type autoconvert_layers: bool, optional
    :param input_quantization: whether the input of the network needs
    to be quantized or not (default: True)
    :type input_quantization: bool
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS,
    defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS,
    defaults to ()
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :raises ValueError: when called with an unsupported regularizer
    :param gumbel_softmax: use Gumbel SoftMax for sampling, instead of a normal SoftMax
    :type gumbel_softmax: bool, optional
    :param hard_softmax: use hard Gumbel SoftMax sampling (only applies when gumbel_softmax = True)
    :type hard_softmax: bool, optional
    :param disable_sampling: do not perform any update of the alpha coefficients,
    thus using the saved ones. Useful to perform fine-tuning of the saved model.
    :type disable_sampling: bool, optional
    """
    def __init__(
            self,
            model: nn.Module,
            cost: Union[CostSpec, Dict[str, CostSpec]] = bit_params,
            input_example: Optional[Any] = None,
            input_shape: Optional[Tuple[int, ...]] = None,
            a_precisions: Tuple[int, ...] = (2, 4, 8),
            w_precisions: Tuple[int, ...] = (2, 4, 8),
            w_search_type: MPSType = MPSType.PER_LAYER,
            qinfo: Dict = DEFAULT_QINFO,
            qinfo_input_quantizer: Optional[Dict] = DEFAULT_QINFO_INPUT_QUANTIZER,
            temperature: float = 1.,
            autoconvert_layers: bool = True,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            gumbel_softmax: bool = False,
            hard_softmax: bool = False,
            disable_sampling: bool = False,
            disable_shared_quantizers: bool = False):
        super(MPS, self).__init__(model, cost, input_example, input_shape)
        self.seed, self._target_layers = convert(
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
        self.activation_precisions = a_precisions
        self.weight_precisions = w_precisions
        self.w_search_type = w_search_type
        self.qinfo = qinfo
        self.qinfo_input_quantizer = qinfo_input_quantizer
        self.disable_shared_quantizers = disable_shared_quantizers
        self.update_softmax_options(temperature, hard_softmax, gumbel_softmax, disable_sampling)

    def forward(self, *args: Any) -> torch.Tensor:
        """Forward function for the DNAS model.
        Simply invokes the inner model's forward

        :return: the output tensor
        :rtype: torch.Tensor
        """
        return self.seed.forward(*args)

    def supported_regularizers(self) -> Tuple[str, ...]:
        """Returns a tuple of strings with the names of the supported cost regularizers

        :return: a tuple of strings with the names of the supported cost regularizers
        :rtype: Tuple[str, ...]
        """
        return ('size', 'macs')

    def get_size(self) -> torch.Tensor:
        """Computes the total number of effective parameters of all NAS-able layers

        :return: the effective memory occupation of weights
        :rtype: torch.Tensor
        """
        size = torch.tensor(0, dtype=torch.float32)
        for layer in self._target_layers:
            size = size + layer.get_size()
        return size

    def binarize_alpha(self):
        """Binarize the architecture coefficients by means of the argmax operator
        """
        for layer in self._target_layers:
            for quantizer in [layer.w_quantizer, layer.a_quantizer]:
                if isinstance(quantizer, (MPSQtzLayer, MPSQtzChannel)):
                    alpha_prec = quantizer.alpha_prec
                    max_index = alpha_prec.argmax(dim=0)
                    argmaxed_alpha = torch.zeros(alpha_prec.shape, device=alpha_prec.device)

                    if len(argmaxed_alpha.shape) > 1:
                        for channel_index in range(argmaxed_alpha.shape[-1]):
                            argmaxed_alpha[max_index[channel_index], channel_index] = 1
                    else:  # single precision
                        # TODO: check
                        argmaxed_alpha[max_index.reshape(-1)[0]] = 1
                    quantizer.theta_alpha = argmaxed_alpha
        return

    def freeze_alpha(self):
        """Freeze the alpha coefficients disabling the gradient computation.
        Useful for fine-tuning the model without changing its architecture
        """
        for layer in self._target_layers:
            for quantizer in [layer.w_quantizer, layer.a_quantizer]:
                if isinstance(quantizer, (MPSQtzLayer, MPSQtzChannel)):
                    quantizer.alpha_prec.requires_grad = False

    # N.B., this follows EdMIPS formulation -> layer_macs*mix_wbit*mix_abit
    # TODO: include formulation with cycles
    def get_macs(self) -> torch.Tensor:
        """Computes the total number of effective MACs of all NAS-able layers

        :return: the effective number of MACs
        :rtype: torch.Tensor
        """
        macs = torch.tensor(0, dtype=torch.float32)
        for layer in self._target_layers:
            macs = macs + layer.get_macs()
        return macs

    @property
    def regularizer(self) -> str:
        """Returns the regularizer type

        :raises ValueError: for unsupported conversion types
        :return: the string identifying the regularizer type
        :rtype: str
        """
        return self._regularizer

    @regularizer.setter
    def regularizer(self, value: str):
        if value == 'size':
            self.get_regularization_loss = self.get_size
        elif value == 'macs':
            self.get_regularization_loss = self.get_macs
        else:
            raise ValueError(f"Invalid regularizer {value}")
        self._regularizer = value

    def get_pruning_loss(self) -> torch.Tensor:
        """Returns the pruning loss, computed on top of the number of pruned channels of each layer

        :return: the pruning loss
        :rtype: torch.Tensor
        """
        ploss = torch.tensor(0, dtype=torch.float32)
        for layer in self._target_layers:
            ploss = ploss + layer.get_pruning_loss()
        return ploss

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
        :param disable_sampling: disable the sampling of the architectural coefficients in the forward pass
        :type disable_sampling: Optional[bool]
        """
        for _, _, layer in self._unique_leaf_modules:
            if isinstance(layer, MPSModule):
                if temperature is not None:
                    layer.softmax_temperature = temperature
                if hard is not None:
                    layer.hard_softmax = hard


    def arch_export(self):
        """Export the architecture found by the NAS as a `quant.nn` module

        The returned model will have the trained weights found during the search filled in, but
        should be fine-tuned for optimal results.

        :return: the precision-assignement found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        # TODO update
        mod, _ = convert(self.seed,
                         self._input_shape,
                         self.activation_precisions,
                         self.weight_precisions,
                         self.w_search_type,
                         self.qinfo,
                         self.qinfo_input_quantizer,
                         'export',
                         input_quantization=self.input_quantization,
                         disable_shared_quantizers=self.disable_shared_quantizers)

        return mod

    def arch_summary(self) -> Dict[str, Dict[str, Any]]:
        """Generates a dictionary representation of the precision-assignment found by the NAS.
        Only optimized layers are reported

        :return: a dictionary representation of the precision-assignement found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        arch = {}
        for name, layer in self.seed.named_modules():
            if layer in self._target_layers:
                layer = cast(MPSModule, layer)
                arch[name] = layer.summary()
                arch[name]['type'] = layer.__class__.__name__
        return arch

    def alpha_summary(self) -> Dict[str, Dict[str, Any]]:
        """Generates a dictionary representation of the architectural coefficients found by the NAS.
        Only optimized layers are reported

        :return: a dictionary representation of the architectural coefficients found by the NAS
        :rtype: Dict[str, Dict[str, any]]"""
        arch = {}
        for name, layer in self.seed.named_modules():
            if layer in self._target_layers:
                prec_dict = {}
                if isinstance(layer.a_quantizer, (MPSQtzLayer, MPSQtzChannel)):
                    prec_dict['a_precision'] = layer.a_quantizer.alpha_prec.detach()
                if isinstance(layer.w_quantizer, (MPSQtzLayer, MPSQtzChannel)):
                    prec_dict['w_precision'] = layer.w_quantizer.alpha_prec.detach()

                arch[name] = prec_dict
                arch[name]['type'] = layer.__class__.__name__
        return arch

    def theta_alpha_summary(self) -> Dict[str, Dict[str, Any]]:
        """Generates a dictionary representation of the architectural coefficients found by the NAS.
        Only optimized layers are reported

        :return: a dictionary representation of the architectural coefficients found by the NAS
        :rtype: Dict[str, Dict[str, any]]"""
        arch = {}
        for name, layer in self.seed.named_modules():
            if layer in self._target_layers:
                prec_dict = {}
                if isinstance(layer.a_quantizer, (MPSQtzLayer, MPSQtzChannel)):
                    prec_dict['a_precision'] = layer.a_quantizer.theta_alpha.detach()
                if isinstance(layer.w_quantizer, (MPSQtzLayer, MPSQtzChannel)):
                    prec_dict['w_precision'] = layer.w_quantizer.theta_alpha.detach()

                arch[name] = prec_dict
                arch[name]['type'] = layer.__class__.__name__
        return arch

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
            if layer in self._target_layers:
                layer = cast(MPSModule, layer)
                prfx = prefix
                prfx += "." if len(prefix) > 0 else ""
                prfx += lname
                for name, param in layer.named_nas_parameters(prefix=lname, recurse=recurse):
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
        exclude = set(_[0] for _ in self.named_nas_parameters())
        for name, param in self.named_parameters():
            if name not in exclude:
                yield name, param

    def get_cost(self, name: Optional[str] = None) -> torch.Tensor:
        """Returns the value of the model cost metric named "name".
        Only allowed alternative in case of multiple cost metrics.

        :raises NotImplementedError: on the base DNAS class
        :rtype: torch.Tensor
        """
        # TODO: Identical to PIT. Factor-out to DNAS?
        if name is None:
            assert isinstance(self._cost_specification, CostSpec), \
                "Multiple model cost metrics provided, you must use .get_cost('cost_name')"
            cost_spec = self._cost_specification
            cost_fn_map = self._cost_fn_map
        else:
            assert isinstance(self._cost_specification, dict), \
                "The provided cost specification is not a dictionary."
            cost_spec = self._cost_specification[name]
            cost_fn_map = self._cost_fn_map[name]
        cost_spec = cast(CostSpec, cost_spec)
        cost_fn_map = cast(Dict[str, CostFn], cost_fn_map)
        return self._get_single_cost(cost_spec, cost_fn_map)

    def _get_single_cost(self, cost_spec: CostSpec,
                         cost_fn_map: Dict[str, CostFn]) -> torch.Tensor:
        """Private method to compute a single cost value"""
        # TODO: relies on the attribute name sn_branches. Not nice, but didn't find
        # a better solution that remains flexible.
        cost = torch.tensor(0, dtype=torch.float32)
        target_list = self._unique_leaf_modules if cost_spec.shared else self._leaf_modules
        for lname, node, layer in target_list:
            if isinstance(layer, MPSModule):
                cost = cost + layer.get_cost(cost_spec, cost_fn_map)
            elif 'sn_branches' not in str(node.target) and self.full_cost:
                # TODO: this is constant and can be pre-computed for efficiency
                v = vars(layer)
                v.update(shapes_dict(node))
                cost = cost + cost_fn_map[lname](v)
        return cost

    def _create_cost_fn_map(self) -> Union[Dict[str, CostFn], Dict[str, Dict[str, CostFn]]]:
        """Private method to define a map from layers to cost value(s)"""
        # TODO: identical to PIT. Factor-out to DNAS?
        if isinstance(self._cost_specification, dict):
            cost_fn_maps = {}
            for n, c in self._cost_specification.items():
                cost_fn_maps[n] = self._single_cost_fn_map(c)
        else:
            cost_fn_maps = self._single_cost_fn_map(self._cost_specification)
        return cost_fn_maps

    def _single_cost_fn_map(self, c: CostSpec) -> Dict[str, CostFn]:
        """MPS-specific creator of {layertype, cost_fn} maps based on a CostSpec."""
        # simply computes cost of all layers that are not Combiners
        cost_fn_map = {}
        for lname, _, layer in self._unique_leaf_modules:
            if not isinstance(layer, MPSModule):
                t = type(layer)
                cost_fn_map[lname] = c[(t, vars(layer))]
        return cost_fn_map

    def __str__(self):
        """Prints the precision-assignent found by the NAS to screen

        :return: a str representation of the current architecture and
        its precision-assignement
        :rtype: str
        """
        arch = self.arch_summary()
        return str(arch)
