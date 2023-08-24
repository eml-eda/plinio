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
# * Author:  Daniele Jahier Pagliari <daniele.jahier@polito.it>                *
# *----------------------------------------------------------------------------*
from . import CostSpec
from .pattern import Conv1dGeneric, Conv2dGeneric, LinearGeneric, \
        Conv1dDW, Conv2dDW


def _ops_bit_conv1d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    w_precision = spec['w_precision']
    in_precision = spec['in_precision']
    # w_format = spec['w_format']
    # in_format = spec['in_format']
    # assert w_format == int and in_format == int, "Model only supports integer quantization"
    # cost = cout * (cin * k + 1) * w_precision * in_precision * out_shape[2]
    cost = cout * (cin * k) * w_precision * in_precision * out_shape[2]
    return cost


def _ops_bit_conv2d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    w_precision = spec['w_precision']
    in_precision = spec['in_precision']
    # w_format = spec['w_format']
    # in_format = spec['in_format']
    # assert w_format == int and in_format == int, "Model only supports integer quantization"
    # cost = cout * (cin * k[0] * k[1] + 1) * w_precision * in_precision * out_shape[2] * out_shape[3]
    cost = cout * (cin * k[0] * k[1]) * w_precision * in_precision * out_shape[2] * out_shape[3]
    return cost


def _ops_bit_conv1d_dw(spec):
    cin = spec['in_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    w_precision = spec['w_precision']
    in_precision = spec['in_precision']
    # w_format = spec['w_format']
    # in_format = spec['in_format']
    # assert w_format == int and in_format == int, "Model only supports integer quantization"
    # cost = cin * (k + 1) * w_precision * in_precision * out_shape[2]
    cost = cin * (k) * w_precision * in_precision * out_shape[2]
    return cost


def _ops_bit_conv2d_dw(spec):
    cin = spec['in_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    w_precision = spec['w_precision']
    in_precision = spec['in_precision']
    # w_format = spec['w_format']
    # in_format = spec['in_format']
    # assert w_format == int and in_format == int, "Model only supports integer quantization"
    # cost = cin * (k[0] * k[1] + 1) * w_precision * in_precision * out_shape[2] * out_shape[3]
    cost = cin * (k[0] * k[1]) * w_precision * in_precision * out_shape[2] * out_shape[3]
    return cost


def _ops_bit_linear(spec):
    cin = spec['in_features']
    cout = spec['out_features']
    w_precision = spec['w_precision']
    in_precision = spec['in_precision']
    # w_format = spec['w_format']
    # in_format = spec['in_format']
    # assert w_format == int and in_format == int, "Model only supports integer quantization"
    # cost = cout * (cin + 1) * w_precision * in_precision
    cost = cout * (cin) * w_precision * in_precision
    return cost


ops_bit = CostSpec(shared=True, default_behavior='zero')
ops_bit[Conv1dGeneric] = _ops_bit_conv1d_generic
ops_bit[Conv2dGeneric] = _ops_bit_conv2d_generic
ops_bit[Conv1dDW] = _ops_bit_conv1d_dw
ops_bit[Conv2dDW] = _ops_bit_conv2d_dw
ops_bit[LinearGeneric] = _ops_bit_linear
