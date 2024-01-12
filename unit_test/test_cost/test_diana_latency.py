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
# * Author: Matteo Risso <matteo.risso@polito.it>                              *
# *----------------------------------------------------------------------------*
import unittest
import torch
import torch.nn as nn
import random
from plinio.cost import diana_latency


class TestDianaLat(unittest.TestCase):
    """
    Verify correctness of the diana latency cost model.

    Given that we don't have a "ground-truth" evaluaiton of the latency, we verify that
    the latency scales properly with the number of channels and the input size.
    Moreover, we test that the gradients correctly flow.
    """

    # note: batch-size is fixed to 1 because plinio cares about single-input inference cost

    def test_lat_conv2d(self):
        # Layers specs
        spec = {}
        spec['in_channels'] = random.randint(1, 20)
        spec['out_channels'] = torch.tensor(random.randint(32, 64))
        spec['groups'] = 1
        spec['kernel_size'] = (random.randint(1, 5), random.randint(1, 5))
        spec['output_shape'] = (1, spec['out_channels'],
                                random.randint(8, 64),
                                random.randint(8, 64))
        spec['a_precision'] = 8

        original_ch = spec['out_channels']

        # Digital
        spec['w_precision'] = 8
        est_cost_high = diana_latency[nn.Conv2d, spec](spec)
        spec['out_channels'] = original_ch // 2
        est_cost_low = diana_latency[nn.Conv2d, spec](spec)
        msg = 'Wrong scaling of digital cost with number of output channels'
        self.assertTrue(est_cost_high > est_cost_low, msg)
        spec['out_channels'] = torch.tensor(0)
        est_cost_zero = diana_latency[nn.Conv2d, spec](spec)
        msg = 'Wrong digital cost with zero output channels'
        self.assertEqual(est_cost_zero, 0, msg)
        spec['out_channels'] = original_ch

        # Digital - Depthwise
        spec['groups'] = int(spec['out_channels'])
        est_cost_high = diana_latency[nn.Conv2d, spec](spec)
        spec['out_channels'] = original_ch // 2
        est_cost_low = diana_latency[nn.Conv2d, spec](spec)
        msg = 'Wrong scaling of depthwise digital cost with number of output channels'
        self.assertTrue(est_cost_high > est_cost_low, msg)
        spec['out_channels'] = original_ch

        # Analog
        spec['groups'] = 1
        spec['w_precision'] = 2
        est_cost_high = diana_latency[nn.Conv2d, spec](spec)
        spec['out_channels'] = original_ch // 2
        est_cost_low = diana_latency[nn.Conv2d, spec](spec)
        msg = 'Wrong scaling of analog cost with number of output channels'
        # If we choose ch_out from 32 and 64 I expect same cost even if the number of channels is halved
        # because the analog array size is 512 for the output channels dimension
        self.assertTrue(est_cost_high == est_cost_low, msg)
        # if we choose ch_out from 512 and 1024 I expect a cost reduction
        spec['out_channels'] = torch.tensor(random.randint(512, 1024))
        est_cost_high = diana_latency[nn.Conv2d, spec](spec)
        spec['out_channels'] = original_ch // 2
        est_cost_low = diana_latency[nn.Conv2d, spec](spec)
        msg = 'Wrong scaling of analog cost with number of output channels'
        self.assertTrue(est_cost_high > est_cost_low, msg)
        spec['out_channels'] = torch.tensor(0)
        est_cost_zero = diana_latency[nn.Conv2d, spec](spec)
        msg = 'Wrong analog cost with zero output channels'
        self.assertEqual(est_cost_zero, 0, msg)

    def test_lat_linear(self):
        # Layers specs
        spec = {}
        spec['in_features'] = random.randint(1, 20)
        spec['out_features'] = torch.tensor(random.randint(32, 64))
        spec['output_shape'] = (1, spec['out_features'])
        spec['a_precision'] = 8

        original_ch = spec['out_features']

        # Digital
        spec['w_precision'] = 8
        est_cost_high = diana_latency[nn.Linear, spec](spec)
        spec['out_features'] = original_ch // 2
        est_cost_low = diana_latency[nn.Linear, spec](spec)
        msg = 'Wrong scaling of digital cost with number of output features'
        self.assertTrue(est_cost_high > est_cost_low, msg)
        spec['out_features'] = torch.tensor(0)
        est_cost_zero = diana_latency[nn.Linear, spec](spec)
        msg = 'Wrong digital cost with zero output features'
        self.assertEqual(est_cost_zero, 0, msg)
        spec['out_features'] = original_ch

        # Analog
        spec['w_precision'] = 2
        est_cost_high = diana_latency[nn.Linear, spec](spec)
        spec['out_features'] = original_ch // 2
        est_cost_low = diana_latency[nn.Linear, spec](spec)
        msg = 'Wrong scaling of analog cost with number of output features'
        # If we choose ch_out from 32 and 64 I expect same cost even if the number of channels is halved
        # because the analog array size is 512 for the output channels dimension
        self.assertTrue(est_cost_high == est_cost_low, msg)
        # if we choose ch_out from 512 and 1024 I expect a cost reduction
        spec['out_features'] = torch.tensor(random.randint(512, 1024))
        est_cost_high = diana_latency[nn.Linear, spec](spec)
        spec['out_features'] = original_ch // 2
        est_cost_low = diana_latency[nn.Linear, spec](spec)
        msg = 'Wrong scaling of analog cost with number of output features'
        self.assertTrue(est_cost_high > est_cost_low, msg)
        spec['out_features'] = torch.tensor(0)
        est_cost_zero = diana_latency[nn.Linear, spec](spec)
        msg = 'Wrong analog cost with zero output features'
        self.assertEqual(est_cost_zero, 0, msg)

    def test_gradient_flow(self):
        # Layers specs
        spec = {}
        spec['in_channels'] = random.randint(1, 20)
        spec['groups'] = 1
        spec['kernel_size'] = (random.randint(1, 5), random.randint(1, 5))
        spec['a_precision'] = 8

        # Digital
        # N.B., here we requires grad because we want to test the gradient flow
        spec['out_channels'] = torch.tensor(512,
                                            requires_grad=True, dtype=torch.float)
        spec['output_shape'] = (1, spec['out_channels'],
                                random.randint(8, 64),
                                random.randint(8, 64))
        spec['w_precision'] = 8
        optimizer = torch.optim.SGD([spec['out_channels']], lr=1)
        est_cost = float('inf')
        for _ in range(10):
            new_cost = diana_latency[nn.Conv2d, spec](spec)
            new_cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            msg = 'Digital cost is not decreasing, probably gradients are not flowing properly'
            self.assertTrue(new_cost <= est_cost, msg)
            est_cost = new_cost

        # Analog
        spec['w_precision'] = 2
        spec['out_channels'] = torch.tensor(1024,
                                            requires_grad=True, dtype=torch.float)
        spec['output_shape'] = (1, spec['out_channels'],
                                random.randint(8, 64),
                                random.randint(8, 64))
        optimizer = torch.optim.SGD([spec['out_channels']], lr=1)
        est_cost = float('inf')
        for _ in range(10):
            new_cost = diana_latency[nn.Conv2d, spec](spec)
            new_cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            msg = 'Analog cost is not decreasing, probably gradients are not flowing properly'
            self.assertTrue(new_cost <= est_cost, msg)
            est_cost = new_cost


if __name__ == '__main__':
    unittest.main(verbosity=2)
