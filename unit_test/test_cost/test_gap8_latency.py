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
# * Author: Daniele Jahier Pagliari <daniele.jahier@polito.it>                 *
# *----------------------------------------------------------------------------*
import unittest
import torchinfo
import torch
import torch.nn as nn
import random
from plinio.cost import gap8_latency
from plinio.methods.pit.nn import PITConv2d, PITLinear
from plinio.methods.pit.nn.features_masker import PITFeaturesMasker

estimated_MAC_cycles = {}
estimated_MAC_cycles[(nn.Conv2d, 0)] = 16
estimated_MAC_cycles[(nn.Conv2d, 1)] = 1.5
estimated_MAC_cycles[(nn.Linear, 0)] = 8

class TestGAP8Latency(unittest.TestCase):
    """Verify correctness of the ops cost model, using torchinfo as reference."""

    # note: batch-size is fixed to 1 because plinio cares about single-input inference cost

    def test_gap8latency_conv2d(self):
        print()
        cin = 16
        k = 3
        for cout in range(1,17):
            # regular
            conv = nn.Conv2d(cin, cout, (k,k))
            self._compute_and_assert(conv, cout,0, (1, cin, 16, 16),
                                    "Error in Conv2d regular")
            # depth-wise
            conv = nn.Conv2d(cout, cout, (k,k), groups=cout)
            self._compute_and_assert(conv, cout,1, (1, cout, 16, 16),
                                    "Error in Conv2d depth-wise")

    def test_ops_linear(self):
        print()
        fin = 16
        for fout in range(1,17):
            lin = nn.Linear(fin, fout, bias=False)
            self._compute_and_assert(lin, fout, 0, (1, fin), "Error in linear with bias")


    def _compute_and_assert(self, layer, cout,g, input_size, message):
        x = torch.randn(input_size)

        if isinstance(layer, nn.Conv2d):
            pit_ut = PITConv2d(
                layer,
                out_features_masker=PITFeaturesMasker(cout)
            )
        else:
            pit_ut = PITLinear(
                layer,
                out_features_masker=PITFeaturesMasker(cout)
            )
        y = pit_ut(x)
        spec = vars(pit_ut)
        spec['output_shape'] = y.shape
        if isinstance(layer, nn.Conv2d):
            spec['out_channels'] = pit_ut.out_features_eff
            spec['in_channels'] = pit_ut.out_features_eff/cout*spec['in_channels']
        else:
            spec['out_features'] = pit_ut.out_features_eff
            spec['in_features'] = pit_ut.out_features_eff/cout*spec['in_features']
        est_cost = gap8_latency[type(layer), spec](spec)
        model_summary = torchinfo.summary(layer, input_size, verbose=False)
        MACs = model_summary.total_mult_adds
        if g and isinstance(layer, nn.Conv2d):
            print("DW Conv MAC/cycles: " + str(MACs/est_cost))
        elif isinstance(layer, nn.Conv2d):
            print("RegularConv MAC/cycles: " + str(MACs/est_cost))
        else:
            print("Linear MAC/cycles: " + str(MACs/est_cost))
        self.assertTrue(bool((MACs/est_cost)<=estimated_MAC_cycles[(type(layer), g)]), message)


if __name__ == '__main__':
    unittest.main(verbosity=2)
