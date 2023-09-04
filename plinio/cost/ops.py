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


def _ops_conv1d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    cost = cin * cout * k
    cost = cost * out_shape[2]
    return cost


def _ops_conv2d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    cost = cin * cout * k[0] * k[1]
    cost = cost * out_shape[2] * out_shape[3]
    return cost


def _ops_conv1d_dw(spec):
    cin = spec['in_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    cost = cin * k
    cost = cost * out_shape[2]
    return cost


def _ops_conv2d_dw(spec):
    cin = spec['in_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    cost = cin * k[0] * k[1]
    cost = cost * out_shape[2] * out_shape[3]
    return cost


def _ops_linear_generic(spec):
    cin = spec['in_features']
    cout = spec['out_features']
    cost = cin * cout
    return cost


ops = CostSpec(shared=False, default_behavior='zero')
ops[Conv1dGeneric] = _ops_conv1d_generic
ops[Conv2dGeneric] = _ops_conv2d_generic
ops[Conv1dDW] = _ops_conv1d_dw
ops[Conv2dDW] = _ops_conv2d_dw
ops[LinearGeneric] = _ops_linear_generic
