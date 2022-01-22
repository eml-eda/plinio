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
import networkx as nx
import matplotlib.pyplot as plt
from models import MySimpleNN
from models import TCResNet14
from flexnas.utils.model_graph import *


class TestGraph(unittest.TestCase):

    def test_graph_from_simple_model(self):
        nn_ut = MySimpleNN()
        g = model_to_nx_graph(nn_ut)
        self.assertEqual(len(g.nodes), 12)

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
        g = model_to_nx_graph(nn_ut)
        f = plt.figure()
        nx.draw(g, with_labels=True, ax=f.add_subplot(111))
        f.savefig("tcresnet14.png")
        # self.assertEqual(len(g.nodes), 12)

if __name__ == '__main__':
    unittest.main()
