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
from .pattern import Conv2dGeneric, LinearGeneric

## TODO: document better the cycles functions below

class ComputeOxUnrollSTE(torch.autograd.Function):
    """Torch autograd function that computes the optimal ox_unroll for the Diana analog accelerator"""
    @staticmethod
    def forward(ctx, ch_eff, ch_in, k_x, k_y):
        device = ch_eff.device
        ox_unroll_list = [1, 2, 4, 8]
        ox_unroll = torch.as_tensor(ox_unroll_list, device=device)
        ch_in_unroll = max(64, ch_in)
        mask_out = ox_unroll * ch_eff <= 512
        mask_in = (ox_unroll + k_x - 1) * ch_in_unroll * k_y <= 1152
        mask = torch.logical_and(mask_out, mask_in)
        mask[0] = True
        return ox_unroll[mask][-1]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


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


class GateSTE(torch.autograd.Function):
    """Torch autograd function that gates the number of channels of a layer based on a threshold"""
    @staticmethod
    def forward(ctx, ch, th):
        ctx.save_for_backward(ch, torch.tensor(th))
        return (ch >= th).float()

    @staticmethod
    def backward(ctx, grad_output):
        ch, th = ctx.saved_tensors
        grad = grad_output.clone()
        grad = 1 / (grad + 1)  # smooth step grad with log derivative
        grad.masked_fill_(ch.le(0), 0)
        grad.masked_fill_(ch.ge(th.data), 0)
        return grad, None


def _analog_cycles(spec, clock_freq=260e06):
    """function that computes the number of cycles for a convolution on the Diana analog accelerator"""
    ch_in = spec['in_channels']
    ch_out = spec['out_channels']
    k_x, k_y = spec['kernel_size']
    groups = spec['groups']
    if groups != 1:
        msg = f'groups={groups}. Analog accelerator supports only groups=1'
        raise ValueError(msg)
    out_x, out_y = spec['output_shape'][2:]
    ox_unroll_base = ComputeOxUnrollSTE.apply(ch_out, ch_in, k_x, k_y)
    cycles_comp = FloorSTE.apply(ch_out, 512) * _floor(ch_in, 128) * out_x * out_y / ox_unroll_base
    # cycles_weights = 4 * 2 * 1152
    cycles_weights = 4 * 2 * ch_in * k_x * k_y
    cycles_comp_norm = cycles_comp * 70 / (1000000000 / clock_freq)
    gate = GateSTE.apply(ch_out, 1.)
    return (gate * cycles_weights) + cycles_comp_norm


def _digital_cycles(spec):
    """function that computes the number of cycles for a convolution on the Diana digital accelerator"""
    ch_in = spec['in_channels']
    ch_out = spec['out_channels']
    k_x, k_y = spec['kernel_size']
    groups = spec['groups']
    out_x, out_y = spec['output_shape'][2:]
    # Original model (no depthwise):
    # cycles = FloorSTE.apply(ch_out, 16) * ch_in * _floor(out_x, 16) * out_y * k_x * k_y
    # Depthwise support:
    # min(ch_out, groups) * FloorSTE.apply(1, 16) * 1 * _floor(out_x, 16) * out_y * k_x * k_y
    # TODO: should we separate in two functions for depthwise and normal conv?
    # TODO: why here the first floor is STE-d and the second is not?
    cycles = FloorSTE.apply(ch_out / groups, 16) * ch_in * _floor(out_x, 16) * out_y * k_x * k_y
    # Works with both depthwise and normal conv:
    cycles_load_store = out_x * out_y * (ch_out + ch_in) / 8
    gate = GateSTE.apply(ch_out, 1.)
    return (gate * cycles_load_store) + cycles
    # return cycles_load_store + cycles


def _diana_latency_conv2d_generic(spec):
    # note that the Diana analog accelerator actually uses ternary precision, not really 2-bit
    # but in the DNAS, the ternary quantizer is treated as a 2-bit one
    # also, the activations are actually on 7-bit, but we use 8-bit during the search, as
    # explained in the paper
    if spec['w_precision'] == 2 and spec['a_precision'] == 8:
        return _analog_cycles(spec)
    elif spec['w_precision'] == 8 and spec['a_precision'] == 8:
        return _digital_cycles(spec)
    else:
        raise ValueError(f'Unsupported weights/activations precision: {spec["w_precision"]} / {spec["a_precision"]}')


def _diana_latency_linear(spec):
    """convert the linear layer in an equivalent convolution and then call the conv2d latency function"""
    new_spec = {}
    new_spec['in_channels'] = spec['in_features']
    new_spec['out_channels'] = spec['out_features']
    new_spec['kernel_size'] = (1, 1)
    new_spec['groups'] = 1
    new_spec['output_shape'] = spec['output_shape'] + (1, 1)
    new_spec['w_precision'] = spec['w_precision']
    new_spec['a_precision'] = spec['a_precision']
    return _diana_latency_conv2d_generic(new_spec)


diana_latency = CostSpec(shared=True, default_behavior='zero')
diana_latency[Conv2dGeneric] = _diana_latency_conv2d_generic
diana_latency[LinearGeneric] = _diana_latency_linear
