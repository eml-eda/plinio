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


from unit_test.models import SimpleCnnNMPruning, SimpleMlpNMPruning
from plinio.methods import NMPruning
import torch
import unittest
from unit_test.test_methods.test_nmpruning.utils import (
    check_sparsity_conv2d,
    check_sparsity_linear,
)


class TestNMPruningTraining(unittest.TestCase):
    """Tests one epoch of training of a NMPruning model and its export"""

    def test_train_one_miniepoch_cnn(self):
        nn_ut = SimpleCnnNMPruning()
        new_nn = NMPruning(nn_ut, n=1, m=4, input_shape=nn_ut.input_shape)

        # Train the model for one mini-epoch (i.e. one batch), with fake data
        optimizer = torch.optim.Adam(new_nn.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()
        n_steps = 2
        for _ in range(n_steps):
            optimizer.zero_grad()
            output = new_nn(torch.randn(1, *nn_ut.input_shape))
            target = torch.randint(0, nn_ut.num_classes, (1,))
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        # Export the model
        exported = new_nn.export()
        check_sparsity_conv2d(self, exported, n=1, m=4)

    def test_train_one_miniepoch_cnn(self):
        nn_ut = SimpleMlpNMPruning()
        new_nn = NMPruning(nn_ut, n=1, m=4, input_shape=nn_ut.input_shape)

        # Train the model for one mini-epoch (i.e. one batch), with fake data
        optimizer = torch.optim.Adam(new_nn.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()
        n_steps = 2
        for _ in range(n_steps):
            optimizer.zero_grad()
            output = new_nn(torch.randn(1, *nn_ut.input_shape))
            target = torch.randint(0, nn_ut.num_classes, (1,))
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        # Export the model
        exported = new_nn.export()
        check_sparsity_linear(self, exported, n=1, m=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
