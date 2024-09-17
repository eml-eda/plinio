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

from plinio.cost import ne16_latency


class TestNE16Latency(unittest.TestCase):
    """Verify correctness of the NE16 cost model.
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
        spec['kernel_size'] = (3, 3)
        spec['output_shape'] = (1, spec['out_channels'],
                                random.randint(8, 64),
                                random.randint(8, 64))
        spec['in_precision'] = 8
        spec['w_precision'] = 8
        spec['w_theta_alpha'] = 1
        original_ch = spec['out_channels']

        est_cost_high = ne16_latency[nn.Conv2d, spec](spec)
        spec['out_channels'] = original_ch // 2
        est_cost_low = ne16_latency[nn.Conv2d, spec](spec)
        msg = f"""Wrong scaling of the cost with number of output channels.
                  estimated cost high: {est_cost_high},
                  estimated cost low: {est_cost_low}"""
        self.assertTrue(est_cost_high >= est_cost_low, msg)

        # Depthwise convolution
        spec['out_channels'] = original_ch
        spec['groups'] = int(spec['out_channels'])
        est_cost_high = ne16_latency[nn.Conv2d, spec](spec)
        spec['out_channels'] = original_ch // 2
        est_cost_low = ne16_latency[nn.Conv2d, spec](spec)
        msg = f"""Wrong scaling of the cost with number of output channels.
                  estimated cost high: {est_cost_high},
                  estimated cost low: {est_cost_low}"""
        self.assertTrue(est_cost_high >= est_cost_low, msg)

        # Vary the precision of the weights
        spec['groups'] = 1
        spec['out_channels'] = original_ch
        original_w_prec = spec['w_precision']
        est_cost_high = ne16_latency[nn.Conv2d, spec](spec)
        spec['w_precision'] = spec['w_precision'] // 2
        est_cost_low = ne16_latency[nn.Conv2d, spec](spec)
        msg = f"""Wrong scaling of the cost with respect to
                  the weights precision.
                  estimated cost high: {est_cost_high},
                  estimated cost low: {est_cost_low}"""
        self.assertTrue(est_cost_high > est_cost_low, msg)

        # Test that the cost of a convolution with half channels at 4-bit and
        # half at 8-bit is the same as two subconvolutions with the half the number
        # of output channels each
        spec['w_precision'] = 4
        spec['out_channels'] = original_ch * 2  # Ensure even number
        spec['w_theta_alpha'] = 0.5
        est_cost = (ne16_latency[nn.Conv2d, spec](spec) * spec['w_theta_alpha'])
        spec['w_precision'] = 8
        est_cost += (ne16_latency[nn.Conv2d, spec](spec) * spec['w_theta_alpha'])

        spec['out_channels'] = spec['out_channels'] // 2
        spec['w_precision'] = 8
        spec['w_theta_alpha'] = 1
        est_cost_splitted = ne16_latency[nn.Conv2d, spec](spec)
        spec['w_precision'] = 4
        est_cost_splitted += ne16_latency[nn.Conv2d, spec](spec)

        self.assertTrue(abs(est_cost - est_cost_splitted) < 1e-6,
                        f"""Costs are different: {est_cost} vs splitted {est_cost_splitted}""")



    def test_lat_linear(self):
        spec = {}
        spec['_parameters'] = {}
        spec['_parameters']['bias'] = None
        spec['in_features'] = random.randint(1, 20)
        spec['out_features'] = torch.tensor(random.randint(32, 64))
        spec['output_shape'] = (1, spec['out_features'])
        spec['in_precision'] = 8
        spec['w_precision'] = 8
        spec['w_theta_alpha'] = 1
        original_ch = spec['out_features']

        est_cost_high = ne16_latency[nn.Linear, spec](spec)
        spec['out_features'] = original_ch // 2
        est_cost_low = ne16_latency[nn.Linear, spec](spec)
        msg = f"""Wrong scaling of the cost with respect to
                 the number of output channels.
                  estimated cost high: {est_cost_high},
                  estimated cost low: {est_cost_low}"""
        self.assertTrue(est_cost_high >= est_cost_low, msg)


    def test_gradient_flow(self):
        # Layers specs
        spec = {}
        spec['_parameters'] = {}
        spec['_parameters']['bias'] = None
        spec['in_channels'] = random.randint(1, 20)
        spec['groups'] = 1
        spec['kernel_size'] = (3, 3)
        spec['in_precision'] = 8

        spec['out_channels'] = torch.tensor(512,
                                            requires_grad=True, dtype=torch.float)
        spec['output_shape'] = (1, spec['out_channels'],
                                random.randint(8, 64),
                                random.randint(8, 64))
        spec['w_precision'] = 8
        spec['w_theta_alpha'] = 1
        optimizer = torch.optim.SGD([spec['out_channels']], lr=1)
        est_cost = float('inf')
        for _ in range(10):
            new_cost = ne16_latency[nn.Conv2d, spec](spec)
            new_cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            msg = 'Cost is not decreasing, probably gradients are not flowing properly'
            self.assertTrue(new_cost <= est_cost, msg)
            est_cost = new_cost


if __name__ == '__main__':
    unittest.main(verbosity=2)
