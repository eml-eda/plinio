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

# Warning: this cost model does not take into account the bias contribution.
# It is kept mainly for reproducibility reasons of previous papers' experiments.
# Please use "params.py" instead.

from . import CostSpec
from .pattern import Conv1dGeneric, Conv2dGeneric, LinearGeneric, \
        Conv1dDW, Conv2dDW


def _params_conv1d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    cost = cin * cout * k
    return cost


def _params_conv2d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    cost = cin * cout * k[0] * k[1]
    return cost


def _params_conv1d_dw(spec):
    cin = spec['in_channels']
    k = spec['kernel_size']
    cost = cin * k
    return cost


def _params_conv2d_dw(spec):
    cin = spec['in_channels']
    k = spec['kernel_size']
    cost = cin * k[0] * k[1]
    return cost


def _params_linear(spec):
    cin = spec['in_features']
    cout = spec['out_features']
    cost = cin * cout
    return cost


params_no_bias = CostSpec(shared=True, default_behavior='zero')
params_no_bias[Conv1dGeneric] = _params_conv1d_generic
params_no_bias[Conv2dGeneric] = _params_conv2d_generic
params_no_bias[Conv1dDW] = _params_conv1d_dw
params_no_bias[Conv2dDW] = _params_conv2d_dw
params_no_bias[LinearGeneric] = _params_linear
