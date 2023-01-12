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

from typing import Any, Tuple, Type, Iterable, Dict, cast, Iterator
import torch
import torch.nn as nn

from flexnas.methods.dnas_base import DNAS
from .mixprec_converter import convert
from .nn.mixprec_module import MixPrecModule
from .nn.mixprec_qtz import MixPrecType, MixPrec_Qtz_Layer, MixPrec_Qtz_Channel

from flexnas.methods.mixprec.quant.quantizers import PACT_Act, MinMax_Weight, Quantizer_Bias

DEFAULT_QINFO = {
    'a_quantizer': {
        'quantizer': PACT_Act,
        'kwargs': {},
    },
    'w_quantizer': {
        'quantizer': MinMax_Weight,
        'kwargs': {},
    },
    'b_quantizer': {
        'quantizer': Quantizer_Bias,
        'kwargs': {
            'num_bits': 32,
        },
    },
}


class MixPrec(DNAS):
    """A class that wraps a nn.Module with DNAS-enabled Mixed Precision assigment

    :param model: the inner nn.Module instance optimized by the NAS
    :type model: nn.Module
    :param input_shape: the shape of an input tensor, without batch size, required for symbolic
    tracing
    :type input_shape: Tuple[int, ...]
    :param activation_precisions: the possible activations' precisions assigment to be explored
    by the NAS
    :type activation_precisions: Iterable[int]
    :param weight_precisions: the possible weights' precisions assigment to be explored
    by the NAS
    :type weight_precisions: Iterable[int]
    :param w_mixprec_type: the mixed precision strategy to be used for weigth
    i.e., `PER_CHANNEL` or `PER_LAYER`. Default is `PER_LAYER`
    :type w_mixprec_type: MixPrecType
    :param qinfo: dict containing desired quantizers for act, weight and bias
    and their arguments excluding the num_bits precision
    :type qinfo: Dict
    :param regularizer: the name of the model cost regularizer used by the NAS, defaults to 'size'
    :type regularizer: Optional[str], optional
    :param autoconvert_layers: should the constructor try to autoconvert NAS-able layers,
    defaults to True
    :type autoconvert_layers: bool, optional
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS,
    defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS,
    defaults to ()
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :raises ValueError: when called with an unsupported regularizer
    """
    def __init__(
            self,
            model: nn.Module,
            input_shape: Tuple[int, ...],
            activation_precisions: Tuple[int, ...] = (2, 4, 8),
            weight_precisions: Tuple[int, ...] = (2, 4, 8),
            w_mixprec_type: MixPrecType = MixPrecType.PER_LAYER,
            qinfo: Dict = DEFAULT_QINFO,
            temperature: float = 1.,
            regularizer: str = 'size',
            autoconvert_layers: bool = True,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = ()):
        super(MixPrec, self).__init__(regularizer, exclude_names, exclude_types)
        self._input_shape = input_shape
        self.seed, self._target_layers = convert(
            model,
            input_shape,
            activation_precisions,
            weight_precisions,
            w_mixprec_type,
            qinfo,
            'autoimport' if autoconvert_layers else 'import',
            exclude_names,
            exclude_types)
        self.activation_precisions = activation_precisions
        self.weight_precisions = weight_precisions
        self.w_mixprec_type = w_mixprec_type
        self.qinfo = qinfo
        self.initial_temperature = temperature
        self._regularizer = regularizer

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

    def arch_export(self):
        """Export the architecture found by the NAS as a `quant.nn` modules

        The returned model will have the trained weights found during the search filled in, but
        should be fine-tuned for optimal results.

        :return: the precision-assignement found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        mod, _ = convert(self.seed, self._input_shape, 'export')

        return mod

    def arch_summary(self) -> Dict[str, Dict[str, Any]]:
        """Generates a dictionary representation of the precision-assignement found by the NAS.
        Only optimized layers are reported

        :return: a dictionary representation of the precision-assignement found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        arch = {}
        for name, layer in self.seed.named_modules():
            if layer in self._target_layers:
                layer = cast(MixPrecModule, layer)
                arch[name] = layer.summary()
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
                layer = cast(MixPrecModule, layer)
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

    def _set_temperature(self, t_i):
        for _, module in self.seed.named_modules():
            if isinstance(module, (MixPrec_Qtz_Layer, MixPrec_Qtz_Channel)):
                module.temperature = t_i

    def __str__(self):
        """Prints the precision-assignent found by the NAS to screen

        :return: a str representation of the current architecture and
        its precision-assignement
        :rtype: str
        """
        arch = self.arch_summary()
        return str(arch)
