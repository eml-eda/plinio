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
from torch.fx import symbolic_trace
from unit_test.models import SimpleNN
from flexnas.utils.model_graph import fx_to_nx_graph


class TestGraph(unittest.TestCase):
    """Class to test network graph utility functions"""

    def test_graph_from_simple_model(self):
        """checks that the graph generated from SimpleNN has the correct number of nodes"""
        nn_ut = SimpleNN()
        fx_graph = symbolic_trace(nn_ut).graph
        g = fx_to_nx_graph(fx_graph)
        self.assertEqual(len(g.nodes), 13)


if __name__ == '__main__':
    unittest.main()
