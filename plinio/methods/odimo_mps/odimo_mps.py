# *----------------------------------------------------------------------------*
# * Copyright (C) 2023 Politecnico di Torino, Italy                            *
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

from typing import Any, Tuple, Type, Iterable, Dict, Union, Optional, Callable
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from plinio.methods.mps import MPS, MPSType
from plinio.cost import CostSpec, diana_latency

from ..mps.quant.quantizers import PACTAct, MinMaxWeight, QuantizerBias

ODIMO_MPS_DEFAULT_QINFO = {
    'layer_default': {
        'output': {
            'quantizer': PACTAct,
            'search_precision': (2, 4, 8),
            'kwargs': {},
        },
        'weight': {
            'quantizer': MinMaxWeight,
            'search_precision': (2, 4, 8),
            'kwargs': {},
        },
        'bias': {
            'quantizer': QuantizerBias,
            'kwargs': {
                'precision': 32,
            },
        },
    },
    'input_default': {
        'quantizer': PACTAct,
        'search_precision': (2, 4, 8),
        'kwargs': {
            'init_clip_val': 1
        },
    }
}

def get_default_qinfo(
        w_precision: Tuple[int,...] = (2, 4, 8),
        a_precision: Tuple[int,...] = (2, 4, 8)) -> Dict[str, Dict[str, Any]]:
    """Function that returns the default quantization information for the NAS

    :param w_precision: the list of bitwidths to be considered for weights
    :type w_precision: Tuple[int,...]
    :param a_precision: the list of bitwidths to be considered for activations
    :type a_precision: Tuple[int,...]
    :return: the default quantization information for the NAS
    :rtype: Dict[str, Dict[str, Any]]
    """
    d = copy.deepcopy(ODIMO_MPS_DEFAULT_QINFO)
    d['input_default']['search_precision'] = a_precision
    d['layer_default']['output']['search_precision'] = a_precision
    d['layer_default']['weight']['search_precision'] = w_precision
    return d

def odimo_mps_latency_reduction(costs):
    """Function that computes the aggregated latency of a multi-precision convolution assuming that the
    convolutions at each precision are run in parallel on different accelerators"""
    s_c = F.softmax(costs, dim=0)
    return torch.dot(s_c, costs)

class ODiMO_MPS(MPS):
    """A class that wraps the ODiMO mixed-precision search DNAS

    :param model: the inner nn.Module instance optimized by the NAS
    :type model: nn.Module
    :param cost: the cost models(s) used by the NAS, defaults to the Diana latency cost model
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
    :param qinfo: dict containing desired quantizers for act, weight and bias
    and their arguments excluding the precision precision
    :type qinfo: Dict
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
    :param temperature: the default sampling temperature (for SoftMax/Gumbel SoftMax)
    :type temperature: float, defaults to 1
    :param gumbel_softmax: use Gumbel SoftMax for sampling, instead of a normal SoftMax,
    defaults to False
    :type gumbel_softmax: bool, optional
    :param disable_sampling: do not perform any update of the alpha coefficients,
    thus using the saved ones. Useful to perform fine-tuning of the saved model,
    defaults to False.
    :type disable_sampling: bool, optional
    :param disable_shared_quantizers: do not implement quantizer sharing. Useful
    to obtain upper bounds on the achievable performance,
    defaults to False.
    :type disable_shared_quantizers: bool, optional
    """
    def __init__(
            self,
            model: nn.Module,
            cost: Union[CostSpec, Dict[str, CostSpec]] = diana_latency,
            input_example: Optional[Any] = None,
            input_shape: Optional[Tuple[int, ...]] = None,
            a_precisions: Tuple[int, ...] = (2, 4, 8),
            w_precisions: Tuple[int, ...] = (2, 4, 8),
            qinfo: Dict = ODIMO_MPS_DEFAULT_QINFO,
            autoconvert_layers: bool = True,
            full_cost: bool = False,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            temperature: float = 1.,
            gumbel_softmax: bool = False,
            disable_sampling: bool = False,
            disable_shared_quantizers: bool = False,
            cost_reduction_fn: Callable = odimo_mps_latency_reduction):
        super(ODiMO_MPS, self).__init__(
                model=model,
                cost=cost,
                input_example=input_example,
                input_shape=input_shape,
                a_precisions=a_precisions,
                w_precisions=w_precisions,
                w_search_type=MPSType.PER_CHANNEL,  # ODiMO works with per-channel search
                qinfo=qinfo,
                autoconvert_layers=autoconvert_layers,
                full_cost=full_cost,
                exclude_names=exclude_names,
                exclude_types=exclude_types,
                temperature=temperature,
                gumbel_softmax=gumbel_softmax,
                hard_softmax=False,  # ODiMO does not support hard sampling
                disable_sampling=disable_sampling,
                disable_shared_quantizers=disable_shared_quantizers,
                cost_reduction_fn=cost_reduction_fn)

