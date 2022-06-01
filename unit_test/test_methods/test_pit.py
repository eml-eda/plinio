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
from flexnas.methods import DNAS, PIT
from flexnas.methods.pit import PITConv1d
from unit_test.models import SimpleNN
from unit_test.models import TCResNet14



class TestPIT(unittest.TestCase):
    """PIT NAS testing class.

    TODO: could be separated in more sub-classes, creating a test_pit folder with test_convert/
    test_extract/ etc subfolders.
    """

    def test_prepare_simple_model(self):
        """Test the conversion of a simple sequential model"""
        nn_ut = SimpleNN()
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 40)))
        self._compare_prepared(nn_ut, new_nn._inner_model, nn_ut, new_nn)
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 2
        self.assertEqual(exp_tgt, n_tgt,
                         "SimpleNN has {} conv layers, but found {} target layers".format(
                             exp_tgt, n_tgt))
        conv0_input = new_nn._inner_model.conv0.input_features_calculator.features  # type: ignore
        conv0_exp_input = 3
        self.assertEqual(conv0_exp_input, conv0_input,
                         "Conv0 has {} input features, but found {}".format(
                             conv0_exp_input, conv0_input))
        conv1_input = new_nn._inner_model.conv1.input_features_calculator.features.item()  # type: ignore
        conv1_exp_input = 32.0
        self.assertEqual(conv1_exp_input, conv1_input,
                         "Conv1 has {} input features, but found {}".format(
                             conv1_exp_input, conv1_input))


    def test_prepare_tc_resnet_14(self):
        """Test the conversion of a ResNet-like model"""
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
        self.assertEqual(exp_tgt, n_tgt,
                         "TCResNet14 has {} conv layers, but found {} target layers".format(
                             exp_tgt, n_tgt))

    def test_keep_alive_masks_simple(self):
        # TODO: should generate more layers with random RF and Cout
        net = SimpleNN()
        pit_net = PIT(net, input_example=torch.rand((1, 3, 40)))
        # conv1 has a filter size of 5 and 57 output channels
        # note: the type: ignore tells pylance to ignore type checks on the next line
        ka_alpha = pit_net._inner_model.conv1.out_channel_masker._keep_alive  # type: ignore
        exp_ka_alpha = torch.tensor([1.0] + [0.0] * 56, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_alpha, exp_ka_alpha), "Wrong keep-alive mask for channels")
        ka_beta = pit_net._inner_model.conv1.timestep_masker._keep_alive  # type: ignore
        exp_ka_beta = torch.tensor([1.0] + [0.0] * 4, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_beta, exp_ka_beta), "Wrong keep-alive mask for rf")
        ka_gamma = pit_net._inner_model.conv1.dilation_masker._keep_alive  # type: ignore
        exp_ka_gamma = torch.tensor([1.0] + [0.0] * 2, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_gamma, exp_ka_gamma), "Wrong keep-alive mask for dilation")

    def test_c_matrices_simple(self):
        # TODO: should generate more layers with random RF and Cout
        net = SimpleNN()
        pit_net = PIT(net, input_example=torch.rand((1, 3, 40)))
        # conv1 has a filter size of 5 and 57 output channels
        c_beta = pit_net._inner_model.conv1.timestep_masker._c_beta  # type: ignore
        exp_c_beta = torch.tensor([
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ], dtype=torch.float32)
        self.assertTrue(torch.equal(c_beta, exp_c_beta), "Wrong C beta matrix")
        c_gamma = pit_net._inner_model.conv1.dilation_masker._c_gamma  # type: ignore
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
        net = SimpleNN()
        x = torch.rand((32,) + tuple(net.input_shape[1:]))
        pit_net = PIT(net, input_example=x[0:1])
        net.eval()
        pit_net.eval()
        y = net(x)
        pit_y = pit_net(x)
        assert torch.all(torch.eq(y, pit_y))
        # TODO: after an initial inference, we can check also if the out_channels_eff and k_eff
        # fields are set correctly for all layers, and other stuff

    @staticmethod
    def _execute_prepare(
            nn_ut: nn.Module,
            input_example: torch.Tensor,
            regularizer: str = 'size',
            exclude_names: Iterable[str] = (),
            exclude_types: Tuple[Type[nn.Module], ...] = ()):
        new_nn = PIT(nn_ut, input_example, regularizer, exclude_names=exclude_names,
                     exclude_types=exclude_types)
        return new_nn

    def _compare_prepared(self,
                          old_mod: nn.Module, new_mod: nn.Module,
                          old_top: nn.Module, new_top: DNAS,
                          exclude_names: Iterable[str] = (),
                          exclude_types: Tuple[Type[nn.Module], ...] = ()):
        for name, child in old_mod.named_children():
            new_child = new_mod._modules[name]
            self._compare_prepared(child, new_child, old_top, new_top, exclude_names, exclude_types)
            if isinstance(child, nn.Conv1d):
                if (name not in exclude_names) and (not isinstance(child, exclude_types)):
                    assert isinstance(new_child, PITConv1d)
                    assert child.out_channels == new_child.out_channels
                    # TODO: add more checks


if __name__ == '__main__':
    unittest.main(verbosity=2)
