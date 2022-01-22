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
import unittest
# import matplotlib.pyplot as plt
from torch.fx import symbolic_trace
from models import MySimpleNN
from models import TCResNet14
from flexnas.utils.model_graph import *


class TestGraph(unittest.TestCase):

    def test_graph_from_simple_model(self):
        nn_ut = MySimpleNN()
        fx_graph = symbolic_trace(nn_ut).graph
        g = fx_to_nx_graph(fx_graph)
        self.assertEqual(len(g.nodes), 12)

    @unittest.skip("Missing ground truth number of nodes")
    def test_graph_from_tc_resnet_14(self):
        config = {
            "input_size": 40,
            "output_size": 12,
            "num_channels": [24, 36, 36, 48, 48, 72, 72],
            "kernel_size": 9,
            "dropout": 0.5,
            "grad_clip": -1,
            "use_bias": True,
            "avg_pool": True,
        }
        nn_ut = TCResNet14(config)
        fx_graph = symbolic_trace(nn_ut).graph
        g = fx_to_nx_graph(fx_graph)
        print(g)
        # f = plt.figure()
        # nx.draw(g, with_labels=True, ax=f.add_subplot(111))
        # f.savefig("tcresnet14.png")
        # self.assertEqual(len(g.nodes), 12)


if __name__ == '__main__':
    unittest.main()
