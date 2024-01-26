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
from numpy import prod
import torch

from . import CostSpec
from .pattern import Conv2dGeneric, LinearGeneric, Conv2dDW


class FloorDivideSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ch, N):
        return torch.floor_divide(ch, N)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class DivAndCeilSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return ((a - 1) // b) + 1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ModuloSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return a % b

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def Ne16PerfModel_generalized(name, ks, depthwise, weights_bitwidth, layer):
    """Compute the latency with the NE16 accelerator.

    Parameters
    ----------
    - name [`str`]: name of the layer
    - ks [`tuple`]: kernel size
    - depthwise [`bool`]: True in the case of a depthwise convolution
    - weights_bitwidth [`int`]: weight precision
    - layer [`tuple`]: layer shape

    Output
    ------
    - `float`: total latency
    - `float`: total number of operations
    - `float`: ratio between operations and latency"""

    n_3x3 = FloorDivideSTE.apply(ks[0], 3) * FloorDivideSTE.apply(ks[1], 3)
    n_1x1 = (ModuloSTE.apply(ks[0], 3) * ks[1] +
             ModuloSTE.apply(ks[1], 3) * ks[0] -
             ModuloSTE.apply(ks[0], 3) * ModuloSTE.apply(ks[1], 3))
    total_latency = 0
    total_ops = 0
    if n_3x3 > 0:
        ne16 = Ne16PerfModel(name, (3,3), depthwise=depthwise, weights_bitwidth=weights_bitwidth)
        ne16.set_layer(layer)
        total_ops += (ne16.ops * n_3x3)
        total_latency += (ne16.latency * n_3x3)
    if n_1x1 > 0:
        ne16 = Ne16PerfModel(name, (1,1), depthwise=depthwise, weights_bitwidth=weights_bitwidth)
        ne16.set_layer(layer)
        total_ops += (ne16.ops * n_1x1)
        total_latency += (ne16.latency  * n_1x1)
    return total_latency, total_ops, total_ops/total_latency


class Ne16PerfModel:
    """Model for the NE16 accelerator. The input activations are assumed to be 8-bit quantized."""
    INPUT_BUFFER_SHAPE = (5, 5, 16)
    OUTPUT_BUFFER_SHAPE = (3, 3, 32)
    FIFO_LATENCY = 6
    SHIFTER_COUNT = 4
    ADDER_COUNT = 8
    MULTIPLIER_COUNT = 4
    MEMORY_THROUGHPUT = 256  # bits per cycle

    def __init__(self,
                 operation,
                 kernel_shape,
                 depthwise=False,
                 nq_shift=False,
                 nq_bias=False,
                 nq_bits=32,
                 weights_bitwidth=8):
        self.operation = operation
        self.kernel_shape = kernel_shape
        self.depthwise = depthwise
        self.nq_shift = nq_shift
        self.nq_bias = nq_bias
        self.nq_bits = nq_bits
        self.weights_bitwidth = weights_bitwidth
        self.INPUT_BITWIDTH = 8
        self.OUTPUT_BITWIDTH = 8
        self.layer = (
                self.OUTPUT_BUFFER_SHAPE[0],
                self.OUTPUT_BUFFER_SHAPE[1],
                self.OUTPUT_BUFFER_SHAPE[2] if not depthwise else self.INPUT_BUFFER_SHAPE[2],
                self.INPUT_BUFFER_SHAPE[2])

    def set_layer(self, layer):
        self.layer = layer
        return self

    def set_subtile(self, h_out=None, w_out=None, k_out=None, k_in=None):
        h_out = h_out if h_out is not None else self.OUTPUT_BUFFER_SHAPE[0]
        w_out = w_out if w_out is not None else self.OUTPUT_BUFFER_SHAPE[1]
        k_out = k_out if k_out is not None else self.OUTPUT_BUFFER_SHAPE[2]
        k_in  = k_in  if k_in  is not None else self.INPUT_BUFFER_SHAPE[2]
        self.INPUT_BUFFER_SHAPE = (h_out + 2, w_out + 2, k_in)
        self.OUTPUT_BUFFER_SHAPE = (h_out, w_out, k_out)

    @property
    def is_3x3(self):
        return self.operation == 'conv' and self.kernel_shape == (3, 3) and not self.depthwise

    @property
    def is_1x1(self):
        return self.operation == 'conv' and self.kernel_shape == (1, 1) and not self.depthwise

    @property
    def is_dw(self):
        return self.operation == 'conv' and self.kernel_shape == (3, 3) and self.depthwise

    @property
    def load_latency(self):
        if self.is_1x1:
            return (10 + self.OUTPUT_BUFFER_SHAPE[0] *
                    self.OUTPUT_BUFFER_SHAPE[1] *
                    DivAndCeilSTE.apply(self.INPUT_BUFFER_SHAPE[2] * self.INPUT_BITWIDTH,
                                        self.MEMORY_THROUGHPUT))
        else:
            return (self.FIFO_LATENCY + self.INPUT_BUFFER_SHAPE[0] * self.INPUT_BUFFER_SHAPE[1] *
                    DivAndCeilSTE.apply(self.INPUT_BUFFER_SHAPE[2] * self.INPUT_BITWIDTH,
                                        self.MEMORY_THROUGHPUT))

    def weight_offset_latency(self, k):
        return (self.FIFO_LATENCY + k) if self.is_dw else self.FIFO_LATENCY

    def matrixvec_latency(self, k):
        return (self.FIFO_LATENCY + k) if self.is_1x1 else (self.FIFO_LATENCY +
                                                            k * self.weights_bitwidth)

    @property
    def update_idx_latency(self):
        return 2

    @property
    def nq_shift_latency(self):
        return 0 if not self.nq_shift else DivAndCeilSTE.apply(self.OUTPUT_BUFFER_SHAPE[2],
                                                               self.SHIFTER_COUNT)

    def nq_bias_latency(self, k):
        return 0 if not self.nq_bias else 8 + DivAndCeilSTE.apply(k, self.ADDER_COUNT)

    def nq_scale_latency(self, k):
        return 9 + DivAndCeilSTE.apply(k * FloorDivideSTE.apply(self.nq_bits, 8),
                                       self.MULTIPLIER_COUNT)

    def normquant_latency(self, k):
        return self.nq_shift_latency + self.nq_scale_latency(k) + self.nq_bias_latency(k)

    @property
    def streamout_latency(self):
        return (3 + self.OUTPUT_BUFFER_SHAPE[0] * self.OUTPUT_BUFFER_SHAPE[1] *
                DivAndCeilSTE.apply(self.OUTPUT_BUFFER_SHAPE[2] * self.OUTPUT_BITWIDTH,
                                    self.MEMORY_THROUGHPUT) + 1)  # + end

    @property
    def latency(self):
        k_out_body = self.INPUT_BUFFER_SHAPE[2] if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2]
        n_out_body = FloorDivideSTE.apply(self.layer[2], k_out_body)
        k_out_rem = ModuloSTE.apply(self.layer[2], k_out_body)

        # nothing depends on k_in so no need for remainder
        n_in = DivAndCeilSTE.apply(self.layer[3], self.INPUT_BUFFER_SHAPE[2])

        # depthwise doesn't care about spatial remainder, it just fetches the same
        n_spatial = (DivAndCeilSTE.apply(self.layer[0], self.OUTPUT_BUFFER_SHAPE[0]) *
                     DivAndCeilSTE.apply(self.layer[1], self.OUTPUT_BUFFER_SHAPE[1]))

        if self.is_dw:
            def iteration_latency(k):
                return (self.load_latency + self.weight_offset_latency(k) +
                        self.matrixvec_latency(k) + self.update_idx_latency +
                        self.normquant_latency(k) + self.streamout_latency)
        else:
            def iteration_latency(k):
                return (n_in * (self.load_latency + self.weight_offset_latency(None) +
                                self.matrixvec_latency(k) + self.update_idx_latency) +
                                self.normquant_latency(k) + self.streamout_latency)

        total_latency = n_spatial * (n_out_body * iteration_latency(k_out_body) +
                                     (iteration_latency(k_out_rem) if k_out_rem != 0 else 0))

        if self.is_dw:
            total_weight_offset_latency = (
                n_spatial * (n_out_body * self.weight_offset_latency(k_out_body) +
                             (self.weight_offset_latency(k_out_rem) if k_out_rem != 0 else 0)))
            total_matrixvec_latency = (
                n_spatial * (n_out_body * self.matrixvec_latency(k_out_body) +
                             (self.matrixvec_latency(k_out_rem) if k_out_rem != 0 else 0)))
            total_load_latency = (
                n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * self.load_latency)
            total_update_idx_latency = (
                n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * self.update_idx_latency)

            total_normquant_latency = (
                n_spatial * (n_out_body * self.normquant_latency(k_out_body) +
                             (self.normquant_latency(k_out_rem) if k_out_rem != 0 else 0)))
            total_streamout_latency = (
                n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * self.streamout_latency)
        else:
            total_weight_offset_latency = (
                n_spatial * (
                    n_out_body * n_in * self.weight_offset_latency(k_out_body) +
                    ((n_in * self.weight_offset_latency(k_out_rem))) if k_out_rem != 0 else 0))
            total_matrixvec_latency = (
                n_spatial * (
                    n_out_body * n_in * self.matrixvec_latency(k_out_body) +
                    ((n_in * self.matrixvec_latency(k_out_rem)) if k_out_rem != 0 else 0)))
            total_load_latency = (
                n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * n_in * self.load_latency)
            total_update_idx_latency = (n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) *
                                        n_in * self.update_idx_latency)

            total_normquant_latency = (
                n_spatial * (n_out_body * self.normquant_latency(k_out_body) +
                             (self.normquant_latency(k_out_rem) if k_out_rem != 0 else 0)))
            total_streamout_latency = (
                n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * self.streamout_latency)

        total_component_wise_latency = (total_weight_offset_latency +
                                        total_matrixvec_latency +
                                        total_load_latency +
                                        total_update_idx_latency +
                                        total_normquant_latency +
                                        total_streamout_latency)

        # assert total_latency == total_component_wise_latency, \
        #   f"total latencies don't match: {total_latency} vs. {total_component_wise_latency}"

        return total_latency

    @property
    def max_ops(self):
        Ho, Wo, Ko, Ki = self.layer
        Ho_max_util = (DivAndCeilSTE.apply(Ho, self.OUTPUT_BUFFER_SHAPE[0]) *
                       self.OUTPUT_BUFFER_SHAPE[0])
        Wo_max_util = (DivAndCeilSTE.apply(Wo, self.OUTPUT_BUFFER_SHAPE[1]) *
                       self.OUTPUT_BUFFER_SHAPE[1])
        Ki_max_util = (DivAndCeilSTE.apply(Ki, self.INPUT_BUFFER_SHAPE[2]) *
                       self.INPUT_BUFFER_SHAPE[2])
        if self.is_3x3 or self.is_1x1:
            return prod(self.kernel_shape) * Ki_max_util * Ho_max_util * Wo_max_util * Ko
        else:
            return prod(self.kernel_shape) * Ki * Ho * Wo

    @property
    def utilization(self):
        return self.ops / self.max_ops

    @property
    def ops(self):
        Ho, Wo, Ko, Ki = self.layer
        if self.is_3x3 or self.is_1x1:
            return prod(self.kernel_shape) * Ki * Ho * Wo * Ko
        else:
            return prod(self.kernel_shape) * Ki * Ho * Wo

    @property
    def perf(self):
        return self.ops / self.latency

    @property
    def max_perf(self):
        ops = (prod(self.kernel_shape) *
               self.INPUT_BUFFER_SHAPE[2] *
               self.OUTPUT_BUFFER_SHAPE[0] *
               self.OUTPUT_BUFFER_SHAPE[1] *
               (1 if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2]))
        k = self.INPUT_BUFFER_SHAPE[2] if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2]
        latency = (self.load_latency +
                   self.weight_offset_latency(k) +
                   self.matrixvec_latency(k) +
                   self.update_idx_latency +
                   (self.normquant_latency(k) + self.streamout_latency if self.is_dw else 0))
        return ops/latency

    def tiled_layer_latency(self, layer_shape_in, layer_shape_out, tile_shape_out):
        body_h_count = FloorDivideSTE.apply(layer_shape_out[0], tile_shape_out[0])
        body_w_count = FloorDivideSTE.apply(layer_shape_out[1], tile_shape_out[1])
        body_k_count = FloorDivideSTE.apply(layer_shape_out[2], tile_shape_out[2])
        rem_h = ModuloSTE.apply(layer_shape_out[0], tile_shape_out[0])
        rem_w = ModuloSTE.apply(layer_shape_out[1], tile_shape_out[1])
        rem_k = ModuloSTE.apply(layer_shape_out[2], tile_shape_out[2])
        layers = [
            (tile_shape_out[0], tile_shape_out[1], tile_shape_out[2], layer_shape_in[2]),
            (tile_shape_out[0], tile_shape_out[1], rem_k, layer_shape_in[2]),
            (tile_shape_out[0], rem_w, tile_shape_out[2], layer_shape_in[2]),
            (tile_shape_out[0], rem_w, rem_k, layer_shape_in[2]),
            (rem_h, tile_shape_out[1], tile_shape_out[2], layer_shape_in[2]),
            (rem_h, tile_shape_out[1], rem_k, layer_shape_in[2]),
            (rem_h, rem_w, tile_shape_out[2], layer_shape_in[2]),
            (rem_h, rem_w, rem_k, layer_shape_in[2])
        ]
        n_tiles = [
            body_h_count * body_w_count * body_k_count,
            body_h_count * body_w_count * (1 if rem_k > 0 else 0),
            body_h_count * (1 if rem_w > 0 else 0) * body_k_count,
            body_h_count * (1 if rem_w > 0 else 0) * (1 if rem_k > 0 else 0),
            (1 if rem_h > 0 else 0) * body_w_count * body_k_count,
            (1 if rem_h > 0 else 0) * body_w_count * (1 if rem_k > 0 else 0),
            (1 if rem_h > 0 else 0) * (1 if rem_w > 0 else 0) * body_k_count,
            (1 if rem_h > 0 else 0) * (1 if rem_w > 0 else 0) * (1 if rem_k > 0 else 0)
        ]

        latency = 0
        ops = 0
        max_ops = 0
        for layer, n in zip(layers, n_tiles):
            self.set_layer(layer)
            latency += n * self.latency
            ops += n * self.ops
            max_ops += n * self.max_ops

        return latency, ops, max_ops

    @property
    def layer_shape_in(self):
        return (self.layer[0] + self.kernel_shape[0] - 1,
                self.layer[1] + self.kernel_shape[1] - 1,
                self.layer[3])

    def dma_latency(self, dma_stall=8, bandwidth=4):
        h_out, w_out, k_out, _ = self.layer
        h_in, w_in, k_in = self.layer_shape_in
        mem = (h_in * w_in * k_in + h_out * w_out * k_out +
               self.kernel_shape[0] * self.kernel_shape[1] * k_out * k_in)
        return (mem / bandwidth) * dma_stall


def _ne16_latency_conv2d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    w_prec = spec['w_precision']
    w_theta_alpha_sum = spec['w_theta_alpha'] * cout
    is_depthwise = False

    assert spec['in_precision'] == 8, \
        "NE16 model only supports 8-bit quantization for the activations"
    assert ((k[0] == 3) and (k[1] == 3)) or ((k[0] == 1) and (k[1] == 1)), \
        "NE16 model only supports 3x3 or 1x1 convolutions"

    layer_params = (
        out_shape[2],
        out_shape[3],
        w_theta_alpha_sum,
        cin)

    latency, _, _ = Ne16PerfModel_generalized(
        name='conv',
        ks=k,
        depthwise=is_depthwise,
        weights_bitwidth=w_prec,
        layer=layer_params)
    cost = latency / spec['w_theta_alpha'] # division due to the product in the calling function
    return cost


def _ne16_latency_conv2d_dw(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    w_prec = spec['w_precision']
    w_theta_alpha_sum = spec['w_theta_alpha'] * cout
    is_depthwise = True

    assert spec['in_precision'] == 8, \
        "NE16 model only supports 8-bit quantization for the activations"
    assert ((k[0] == 3) and (k[1] == 3)), \
        "NE16 model only supports 3x3 depthwise convolutions"

    layer_params = (
        out_shape[2],
        out_shape[3],
        w_theta_alpha_sum,
        cin)

    latency, _, _ = Ne16PerfModel_generalized(
        name='conv',
        ks=k,
        depthwise=is_depthwise,
        weights_bitwidth=w_prec,
        layer=layer_params)
    cost = latency / spec['w_theta_alpha']
    return cost


def _ne16_latency_linear(spec):
    cin = spec['in_features']
    cout = spec['out_features']
    kernel_size = (1, 1)
    w_prec = spec['w_precision']
    w_theta_alpha_sum = spec['w_theta_alpha'] * cout
    is_depthwise = False

    assert spec['in_precision'] == 8, \
        "NE16 model only supports 8-bit quantization for the activations"

    layer_params = (
        1,
        1,
        w_theta_alpha_sum,
        cin)  # TODO: check detach

    latency, _, _ = Ne16PerfModel_generalized(
                name='conv',
                ks=kernel_size,
                depthwise=is_depthwise,
                weights_bitwidth=w_prec,
                layer=layer_params)
    cost = latency / spec['w_theta_alpha']
    return cost


ne16_latency = CostSpec(shared=False, default_behavior='zero')
ne16_latency[Conv2dGeneric] = _ne16_latency_conv2d_generic
ne16_latency[Conv2dDW] = _ne16_latency_conv2d_dw
ne16_latency[LinearGeneric] = _ne16_latency_linear
