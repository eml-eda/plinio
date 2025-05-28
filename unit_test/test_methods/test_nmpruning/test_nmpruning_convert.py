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
# * Author: Francesco Daghero <francesco.daghero@polito.it>                              *
# *----------------------------------------------------------------------------*

from typing import cast
import unittest
import torch
import torch.nn as nn
from plinio.methods import NMPruning
from unit_test.models import (
    SimpleCnnNMPruning,
    SimpleMlpNMPruning,
    SimpleNN2D
)
from unit_test.test_methods.test_nmpruning.utils import (
    check_target_layers,
    check_layers_exclusion,
    compare_exported,
)


class TestNMPruningConvert(unittest.TestCase):
    """Test conversion operations to/from nn.Module from/to NNMPruning"""

    def test_autoimport_simple_layer_cnn(self):
        """Test the conversion of a simple sequential model with layer autoconversion
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = SimpleCnnNMPruning()
        new_nn = NMPruning(nn_ut, n = 1, m = 4, input_shape=nn_ut.input_shape)
        check_target_layers(self, new_nn, exp_tgt=4)
        new_nn = NMPruning(nn_ut, n = 1, m = 8, input_shape=nn_ut.input_shape)
        check_target_layers(self, new_nn, exp_tgt=3)
        new_nn = NMPruning(nn_ut, n = 1, m = 16, input_shape=nn_ut.input_shape)
        check_target_layers(self, new_nn, exp_tgt=2)

    def test_autoimport_simple_layer_mlp(self):
        nn_ut = SimpleMlpNMPruning()
        new_nn = NMPruning(nn_ut, n = 1, m = 4, input_shape=nn_ut.input_shape)
        check_target_layers(self, new_nn, exp_tgt=4)
        new_nn = NMPruning(nn_ut, n = 1, m = 8, input_shape=nn_ut.input_shape)
        check_target_layers(self, new_nn, exp_tgt=3)
        new_nn = NMPruning(nn_ut, n = 1, m = 16, input_shape=nn_ut.input_shape)
        check_target_layers(self, new_nn, exp_tgt=3)




if __name__ == "__main__":
    unittest.main(verbosity=2)
