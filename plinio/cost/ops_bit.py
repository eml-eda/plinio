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
from .pattern import Conv1dGeneric, Conv2dGeneric, Conv3dGeneric, LinearGeneric, \
        Conv1dDW, Conv2dDW


def _ops_bit_conv1d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    w_prec = spec['w_precision']
    in_prec = spec['in_precision']
    # w_format = spec['w_format']
    # in_format = spec['in_format']
    # assert w_format == int and in_format == int, "Model only supports integer quantization"
    cost = k[0] * cin * cout * w_prec * in_prec * out_shape[2]
    return cost


def _ops_bit_conv2d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    w_prec = spec['w_precision']
    in_prec = spec['in_precision']
    # w_format = spec['w_format']
    # in_format = spec['in_format']
    # assert w_format == int and in_format == int, "Model only supports integer quantization"
    cost = k[0] * k[1] * cin * cout * w_prec * in_prec * out_shape[2] * out_shape[3]
    return cost


def _ops_bit_conv3d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    w_prec = spec['w_precision']
    in_prec = spec['in_precision']
    # w_format = spec['w_format']
    # in_format = spec['in_format']
    # assert w_format == int and in_format == int, "Model only supports integer quantization"
    cost = k[0] * k[1] * k[2] * cin * cout * w_prec * in_prec * out_shape[2] * out_shape[3] * out_shape[4]
    return cost


def _ops_bit_conv1d_dw(spec):
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    w_prec = spec['w_precision']
    in_prec = spec['in_precision']
    # There's a small caveat when using this model with the MPS NAS, between putting
    # cin (effective channels) or cout (actual channels) in the expression below.
    # The correct thing is using cout, but this is leaking information from the NAS
    # internals to the cost model. So this should be probably fixed (TODO)
    cost = k[0] * cout * w_prec * in_prec * out_shape[2]
    return cost


def _ops_bit_conv2d_dw(spec):
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    w_prec = spec['w_precision']
    in_prec = spec['in_precision']
    # There's a small caveat when using this model with the MPS NAS, between putting
    # cin (effective channels) or cout (actual channels) in the expression below.
    # The correct thing is using cout, but this is leaking information from the NAS
    # internals to the cost model. So this should be probably fixed (TODO)
    cost = k[0] * k[1] * cout * w_prec * in_prec * out_shape[2] * out_shape[3]
    return cost


def _ops_bit_linear(spec):
    cin = spec['in_features']
    cout = spec['out_features']
    w_prec = spec['w_precision']
    in_prec = spec['in_precision']
    # w_format = spec['w_format']
    # in_format = spec['in_format']
    # assert w_format == int and in_format == int, "Model only supports integer quantization"
    cost = cin * cout * w_prec * in_prec
    return cost


ops_bit = CostSpec(shared=False, default_behavior='zero')
ops_bit[Conv1dGeneric] = _ops_bit_conv1d_generic
ops_bit[Conv2dGeneric] = _ops_bit_conv2d_generic
ops_bit[Conv3dGeneric] = _ops_bit_conv3d_generic
ops_bit[Conv1dDW] = _ops_bit_conv1d_dw
ops_bit[Conv2dDW] = _ops_bit_conv2d_dw
ops_bit[LinearGeneric] = _ops_bit_linear
