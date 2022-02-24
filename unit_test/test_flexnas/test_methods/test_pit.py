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
from typing import Iterable, Tuple, Type
import unittest
import torch
import torch.nn as nn
from flexnas.methods import DNASModel, PITModel
from models import MySimpleNN
from models import TCResNet14


class TestPIT(unittest.TestCase):

    def test_prepare_simple_model(self):
        nn_ut = MySimpleNN()
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 40)))
        self._compare_prepared(nn_ut, new_nn._inner_model, nn_ut, new_nn)
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 2
        self.assertEqual(exp_tgt, n_tgt, "SimpleNN has {} conv layers, but found {} target layers".format(
            exp_tgt, n_tgt))

    def test_prepare_tc_resnet_14(self):
        config = {
            "input_channels": 6,
            "output_size": 12,
            "num_channels": [24, 36, 36, 48, 48, 72, 72],
            "kernel_size": 9,
            "dropout": 0.5,
            "grad_clip": -1,
            "use_bias": True,
            "avg_pool": True,
        }
        nn_ut = TCResNet14(config)
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 6, 50)))
        self._compare_prepared(nn_ut, new_nn._inner_model, nn_ut, new_nn)
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 3 * len(config['num_channels'][1:]) + 1
        self.assertEqual(exp_tgt, n_tgt, "TCResNet14 has {} conv layers, but found {} target layers".format(
            exp_tgt, n_tgt))

    def test_keep_alive_masks_simple(self):
        # TODO: should generate more layers with random RF and Cout
        net = MySimpleNN()
        pit_net = PITModel(net, input_example=torch.rand((1, 3, 40)))
        # conv1 has a filter size of 5 and 57 output channels
        ka_alpha = pit_net._inner_model.conv1._keep_alpha
        exp_ka_alpha = torch.tensor([1.0] + [0.0] * 56, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_alpha, exp_ka_alpha), "Wrong keep-alive mask for channels")
        ka_beta = pit_net._inner_model.conv1._keep_beta
        exp_ka_beta = torch.tensor([1.0] + [0.0] * 4, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_beta, exp_ka_beta), "Wrong keep-alive mask for rf")
        ka_gamma = pit_net._inner_model.conv1._keep_alive
        exp_ka_gamma = torch.tensor([1.0] + [0.0] * 2, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_gamma, exp_ka_gamma), "Wrong keep-alive mask for dilation")

    def test_c_matrices_simple(self):
        # TODO: should generate more layers with random RF and Cout
        net = MySimpleNN()
        pit_net = PITModel(net, input_example=torch.rand((1, 3, 40)))
        # conv1 has a filter size of 5 and 57 output channels
        c_beta = pit_net._inner_model.conv1._c_beta
        exp_c_beta = torch.tensor([
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ], dtype=torch.float32)
        self.assertTrue(torch.equal(c_beta, exp_c_beta), "Wrong C beta matrix")
        c_gamma = pit_net._inner_model.conv1._c_gamma
        exp_c_gamma = torch.tensor([
            [1, 1, 1],
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ], dtype=torch.float32)
        self.assertTrue(torch.equal(c_gamma, exp_c_gamma), "Wrong C gamma matrix")

    def test_initial_inference(self):
        """ check that a PITModel just created returns the same output as its inner model"""
        net = MySimpleNN()
        x = torch.rand((32,) + tuple(net.input_shape))
        pit_net = PITModel(net, input_example=torch.rand((1, 3, 40)))
        net.eval()
        pit_net.eval()
        y = net(x)
        pit_y = pit_net(x)
        assert torch.all(torch.eq(y, pit_y))

    @staticmethod
    def _execute_prepare(
            nn_ut: nn.Module,
            input_example: torch.Tensor,
            regularizer: str = 'size',
            exclude_names: Iterable[str] = (),
            exclude_types: Tuple[Type[nn.Module], ...] = ()):
        new_nn = PITModel(nn_ut, input_example, regularizer, exclude_names, exclude_types)
        return new_nn

    def _compare_prepared(self,
                          old_mod: nn.Module, new_mod: nn.Module,
                          old_top: nn.Module, new_top: DNASModel,
                          exclude_names: Iterable[str] = (),
                          exclude_types: Tuple[Type[nn.Module]] = ()):
        for name, child in old_mod.named_children():
            new_child = new_mod._modules[name]
            self._compare_prepared(child, new_child, old_top, new_top, exclude_names, exclude_types)
            if isinstance(child, tuple(PITModel.replacement_module_rules.keys())):
                if (name not in exclude_names) and (not isinstance(child, exclude_types)):
                    repl = new_top._replacement_module(child, None, lambda x: x)
                    print(type(new_child))
                    print(type(repl))
                    assert isinstance(new_child, type(repl))


if __name__ == '__main__':
    unittest.main()
