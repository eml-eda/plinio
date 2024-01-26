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
# * Author:  Beatrice Alessandra Motetti <beatrice.motetti@polito.it>          *
# *----------------------------------------------------------------------------*
from . import CostSpec
from .ops import _ops_conv1d_generic, _ops_conv2d_generic, _ops_conv1d_dw, \
        _ops_conv2d_dw, _ops_linear_generic
from .pattern import Conv1dGeneric, Conv2dGeneric, LinearGeneric, \
        Conv1dDW, Conv2dDW


def _mpic_lut(a_bit, w_bit):
    """Retrieve the number of cycles/MAC given the activation and weight precision
    according to the MPIC LUT values.
    Reference: "A Mixed-Precision RISC-V Processor for Extreme-Edge DNN Inference",
    Ottavi et al. (https://arxiv.org/pdf/2010.04073.pdf)

    Parameters
    ----------
    - a_bit [`int`]: input activation precision
    - w_bit [`int`]: weight precision

    Output
    ------
    - `float`: number of cycles/MAC"""

    assert (a_bit in [2,4,8]) and (w_bit in [0,2,4,8]), \
        "MPIC model defined only for activation precisions {2,4,8} and weight precisions {0,2,4,8}"

    _MPIC_LUT = {
    2: {0: 0., 2: 1/6.5, 4: 1/4.0, 8: 1/2.2},
    4: {0: 0., 2: 1/3.9, 4: 1/3.5, 8: 1/2.1},
    8: {0: 0., 2: 1/2.5, 4: 1/2.3, 8: 1/2.1}}

    return _MPIC_LUT[a_bit][w_bit]


def _mpic_latency_conv1d_generic(spec):
    w_prec = spec['w_precision']
    in_prec = spec['in_precision']
    macs = _ops_conv1d_generic(spec)
    cost = macs * _mpic_lut(in_prec.item(), w_prec.item())
    return cost


def _mpic_latency_conv2d_generic(spec):
    w_prec = spec['w_precision']
    in_prec = spec['in_precision']
    macs = _ops_conv2d_generic(spec)
    cost = macs * _mpic_lut(in_prec.item(), w_prec.item())
    return cost


def _mpic_latency_conv1d_dw(spec):
    w_prec = spec['w_precision']
    in_prec = spec['in_precision']
    # There's a small caveat when using this model with the MPS NAS, between putting
    # cin (effective channels) or cout (actual channels) in the expression below.
    # The correct thing is using cout, but this is leaking information from the NAS
    # internals to the cost model. So this should be probably fixed (TODO)
    macs = _ops_conv1d_dw(spec)
    cost = macs * _mpic_lut(in_prec.item(), w_prec.item())
    return cost


def _mpic_latency_conv2d_dw(spec):
    w_prec = spec['w_precision']
    in_prec = spec['in_precision']
    # There's a small caveat when using this model with the MPS NAS, between putting
    # cin (effective channels) or cout (actual channels) in the expression below.
    # The correct thing is using cout, but this is leaking information from the NAS
    # internals to the cost model. So this should be probably fixed (TODO)
    macs = _ops_conv2d_dw(spec)
    cost = macs * _mpic_lut(in_prec.item(), w_prec.item())
    return cost


def _mpic_latency_linear(spec):
    w_prec = spec['w_precision']
    in_prec = spec['in_precision']
    # w_format = spec['w_format']
    # in_format = spec['in_format']
    # assert w_format == int and in_format == int, "Model only supports integer quantization"
    macs = _ops_linear_generic(spec)
    cost =  macs * _mpic_lut(in_prec.item(), w_prec.item())
    return cost


mpic_latency = CostSpec(shared=False, default_behavior='zero')
mpic_latency[Conv1dGeneric] = _mpic_latency_conv1d_generic
mpic_latency[Conv2dGeneric] = _mpic_latency_conv2d_generic
mpic_latency[Conv1dDW] = _mpic_latency_conv1d_dw
mpic_latency[Conv2dDW] = _mpic_latency_conv2d_dw
mpic_latency[LinearGeneric] = _mpic_latency_linear
