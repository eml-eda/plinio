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


def _params_bit_conv1d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    w_prec = spec['w_precision']
    # w_format = spec['w_format']
    # assert w_format == int, "Model only supports integer quantization"
    cost = k[0] * cin * cout * w_prec
    return cost


def _params_bit_conv2d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    w_prec = spec['w_precision']
    # w_format = spec['w_format']
    # assert w_format == int, "Model only supports integer quantization"
    cost = k[0] * k[1] * cin * cout * w_prec
    return cost


def _params_bit_conv3d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    w_prec = spec['w_precision']
    # w_format = spec['w_format']
    # assert w_format == int, "Model only supports integer quantization"
    cost = k[0] * k[1] * k[2] * cin * cout * w_prec
    return cost


def _params_bit_conv1d_dw(spec):
    # There's a small caveat when using this model with the MPS NAS, between putting
    # cin (effective channels) or cout (actual channels) in the expression below.
    # The correct thing is using cout, but this is leaking information from the NAS
    # internals to the cost model. So this should be probably fixed (TODO)
    cout = spec['out_channels']
    k = spec['kernel_size']
    w_prec = spec['w_precision']
    cost = k[0] * cout * w_prec
    return cost


def _params_bit_conv2d_dw(spec):
    # There's a small caveat when using this model with the MPS NAS, between putting
    # cin (effective channels) or cout (actual channels) in the expression below.
    # The correct thing is using cout, but this is leaking information from the NAS
    # internals to the cost model. So this should be probably fixed (TODO)
    cout = spec['out_channels']
    k = spec['kernel_size']
    w_prec = spec['w_precision']
    cost = k[0] * k[1] * cout * w_prec
    return cost


def _params_bit_linear(spec):
    cin = spec['in_features']
    cout = spec['out_features']
    w_prec = spec['w_precision']
    # w_format = spec['w_format']
    # assert w_format == int, "Model only supports integer quantization"
    cost = cout * cin * w_prec
    return cost


params_bit = CostSpec(shared=True, default_behavior='zero')
params_bit[Conv1dGeneric] = _params_bit_conv1d_generic
params_bit[Conv2dGeneric] = _params_bit_conv2d_generic
params_bit[Conv3dGeneric] = _params_bit_conv3d_generic
params_bit[Conv1dDW] = _params_bit_conv1d_dw
params_bit[Conv2dDW] = _params_bit_conv2d_dw
params_bit[LinearGeneric] = _params_bit_linear
