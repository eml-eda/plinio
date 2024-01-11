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
import math
import torch

from . import CostSpec
from .pattern import Conv2dGeneric, Conv2dDW, LinearGeneric

## TODO: document better the cycles functions below

class FloorSTE(torch.autograd.Function):
    """Torch autograd function that turns a number of channels ch into its next integer multiple of N"""
    @staticmethod
    def forward(ctx, ch, N):
        return torch.floor((ch + N - 1) / N)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def _floor(ch, N):
    """Same function without the autograd wrapper"""
    return math.floor((ch + N - 1) / N)

def _gap8_latency_conv2d_generic(spec):
    """conv2d layers latency model for GAP8. Parallelization is on output width."""
    ch_in = spec['in_channels']
    ch_out = spec['out_channels']
    k_x, k_y = spec['kernel_size']
    groups = spec['groups']
    out_x, out_y = spec['output_shape'][2:]
    iterations = _floor(out_x, 2) * _floor(out_y, 8)
    im2col = k_x * k_y * ch_in * 2
    matmul = FloorSTE.apply(ch_out, 4) * (5 + FloorSTE.apply(k_x * k_y * ch_in, 4) * (6 + 8) + 10)
    _latency = iterations * (im2col + matmul)
    return _latency

def _gap8_latency_conv2d_dw(spec):
    """conv2d depthwise layers latency model for GAP8. """
    ch_in = spec['in_channels']
    ch_out = spec['out_channels']
    k_x, k_y = spec['kernel_size']
    groups = spec['groups']
    out_x, out_y = spec['output_shape'][2:]
    _latency = 4 * FloorSTE.apply(ch_out, 4) * out_x * out_y * k_x * k_y
    return _latency

def _gap8_latency_linear(spec):
    """linear layers latency model for GAP8"""
    ch_in = spec['in_features']
    ch_out = spec['out_features']
    _latency = FloorSTE.apply(ch_in, 2) * FloorSTE.apply(ch_out, 4)
    return _latency

gap8_latency = CostSpec(shared=True, default_behavior='zero')
gap8_latency[Conv2dGeneric] = _gap8_latency_conv2d_generic
gap8_latency[Conv2dDW] = _gap8_latency_conv2d_dw
gap8_latency[LinearGeneric] = _gap8_latency_linear