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
# * Author: Beatrice Alessandra Motetti <beatrice.motetti@polito.it>           *
# *----------------------------------------------------------------------------*
import random
import unittest

import torch
import torch.nn as nn

from plinio.cost import mpic_latency


class TestMPICLatency(unittest.TestCase):
    """Verify correctness of the MPIC cost model.
    """

    # note: batch-size is fixed to 1 because plinio cares about single-input inference cost

    def test_lat_conv2d(self):
        # Layers specs
        spec = {}
        spec['_parameters'] = {}
        spec['_parameters']['bias'] = None
        spec['in_channels'] = random.randint(1, 20)
        spec['out_channels'] = torch.tensor(random.randint(32, 64))
        spec['groups'] = 1
        spec['kernel_size'] = (random.randint(1, 5), random.randint(1, 5))
        spec['output_shape'] = (1, spec['out_channels'],
                                random.randint(8, 64),
                                random.randint(8, 64))
        spec8        spec['w_precision'] = 8
        original_ch = spec['out_channels']

        est_cost_high = mpic_latency[nn.Conv2d, spec](spec)
        spec['out_channels'] = original_ch // 2
        est_cost_low = mpic_latency[nn.Conv2d, spec](spec)
        msg = f"""Wrong scaling of the cost with number of output channels.
                  estimated cost high: {est_cost_high},
                  estimated cost low: {est_cost_low}"""
        self.assertTrue(est_cost_high > est_cost_low, msg)

        # Depthwise convolution
        spec['out_channels'] = original_ch
        spec['groups'] = int(spec['out_channels'])
        est_cost_high = mpic_latency[nn.Conv2d, spec](spec)
        spec['out_channels'] = original_ch // 2
        est_cost_low = mpic_latency[nn.Conv2d, spec](spec)
        msg = f"""Wrong scaling of the cost with number of output channels.
                  estimated cost high: {est_cost_high},
                  estimated cost low: {est_cost_low}"""
        self.assertTrue(est_cost_high > est_cost_low, msg)

        # Vary the precision of the weights and the input activations
        spec['groups'] = 1
        spec['out_channels'] = original_ch
        original_w_prec = spec['w_precision']
        est_cost_high = mpic_latency[nn.Conv2d, spec](spec)
        spec['w_precision'] = spec['w_precision'] // 2
        est_cost_low = mpic_latency[nn.Conv2d, spec](spec)
        msg = f"""Wrong scaling of the cost with respect to
                  the weights precision.
                  estimated cost high: {est_cost_high},
                  estimated cost low: {est_cost_low}"""
        self.assertTrue(est_cost_high > est_cost_low, msg)

        spec['w_precision'] = original_w_prec
        est_cost_high = mpic_latency[nn.Conv2d, spec](spec)
        spec['in_precision'] = 2
        est_cost_low = mpic_latency[nn.Conv2d, spec](spec)
        msg = f"""Wrong scaling of the cost with respect to
                  the activations precision.
                  Estimated cost high: {est_cost_high},
                  Estimated cost low: {est_cost_low}"""
        self.assertTrue(est_cost_high > est_cost_low, msg)

        spec['groups'] = int(spec['out_channels'])
        spec['w_precision'] = original_w_prec
        spec['in_precision'] = 8
        est_cost_high = mpic_latency[nn.Conv2d, spec](spec)
        spec['w_precision'] = spec['w_precision'] // 2
        est_cost_low = mpic_latency[nn.Conv2d, spec](spec)
        msg = f"""Wrong scaling of the cost with respect to
                  the weights precision.
                  estimated cost high: {est_cost_high},
                  estimated cost low: {est_cost_low}"""
        self.assertTrue(est_cost_high > est_cost_low, msg)

        spec['w_precision'] = original_w_prec
        est_cost_high = mpic_latency[nn.Conv2d, spec](spec)
        spec['in_precision'] = 2
        est_cost_low = mpic_latency[nn.Conv2d, spec](spec)
        msg = f"""Wrong scaling of the cost with respect to
                  the activations precision.
                  Estimated cost high: {est_cost_high},
                  Estimated cost low: {est_cost_low}"""
        self.assertTrue(est_cost_high > est_cost_low, msg)


    def test_lat_linear(self):
        spec = {}
        spec['_parameters'] = {}
        spec['_parameters']['bias'] = None
        spec['in_features'] = random.randint(1, 20)
        spec['out_features'] = torch.tensor(random.randint(32, 64))
        spec['output_shape'] = (1, spec['out_features'])
        spec['in_precision'] = 8
        spec['w_precision'] = 8
        original_ch = spec['out_features']

        est_cost_high = mpic_latency[nn.Linear, spec](spec)
        spec['out_features'] = original_ch // 2
        est_cost_low = mpic_latency[nn.Linear, spec](spec)
        msg = """Wrong scaling of the cost with respect to
                 the number of output channels"""
        self.assertTrue(est_cost_high > est_cost_low, msg)

        # Vary the precision of the weights and the input activations
        spec['groups'] = 1
        spec['out_features'] = original_ch
        original_w_prec = spec['w_precision']
        est_cost_high = mpic_latency[nn.Linear, spec](spec)
        spec['w_precision'] = spec['w_precision'] // 2
        est_cost_low = mpic_latency[nn.Linear, spec](spec)
        msg = """Wrong scaling of the cost with respect to
                 the weights precision"""
        self.assertTrue(est_cost_high > est_cost_low, msg)

        spec['w_precision'] = original_w_prec
        est_cost_high = mpic_latency[nn.Linear, spec](spec)
        spec['in_precision'] = 2
        est_cost_low = mpic_latency[nn.Linear, spec](spec)
        msg = f"""Wrong scaling of the cost with respect to
                  the activations precision.
                  Estimated cost high: {est_cost_high},
                  Estimated cost low: {est_cost_low}"""
        self.assertTrue(est_cost_high > est_cost_low, msg)


    def test_gradient_flow(self):
        # Layers specs
        spec = {}
        spec['_parameters'] = {}
        spec['_parameters']['bias'] = None
        spec['in_channels'] = random.randint(1, 20)
        spec['groups'] = 1
        spec['kernel_size'] = (random.randint(1, 5), random.randint(1, 5))
        spec['in_precision'] = 8

        spec['out_channels'] = torch.tensor(512,
                                            requires_grad=True, dtype=torch.float)
        spec['output_shape'] = (1, spec['out_channels'],
                                random.randint(8, 64),
                                random.randint(8, 64))
        spec['w_precision'] = 8
        optimizer = torch.optim.SGD([spec['out_channels']], lr=1)
        est_cost = float('inf')
        for _ in range(10):
            new_cost = mpic_latency[nn.Conv2d, spec](spec)
            new_cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            msg = 'Cost is not decreasing, probably gradients are not flowing properly'
            self.assertTrue(new_cost <= est_cost, msg)
            est_cost = new_cost


if __name__ == '__main__':
    unittest.main(verbosity=2)
