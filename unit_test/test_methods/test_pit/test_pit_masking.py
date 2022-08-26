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
# * Author:  Fabio Eterno <fabio.eterno@polito.it>                             *
# *----------------------------------------------------------------------------*
from typing import Tuple, Dict
import unittest
import math
import torch
import torch.nn as nn
from flexnas.methods import PIT
from flexnas.methods.pit import PITConv1d
from flexnas.methods.pit.pit_binarizer import PITBinarizer
from unit_test.models import SimpleNN
from unit_test.models import TCResNet14
from unit_test.models import MultiPath1
from unit_test.models import ToyAdd, ToyChannelsCat, ToyModel6, ToyModel7, ToyModel8
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np

from unit_test.models.toy_models import ToySequentialConv1d


class TestPIT(unittest.TestCase):
    """Test masking operations in PIT"""

    def setUp(self):
        self.tc_resnet_config = {
            "input_channels": 6,
            "output_size": 12,
            "num_channels": [24, 36, 36, 48, 48, 72, 72],
            "kernel_size": 9,
            "dropout": 0.5,
            "grad_clip": -1,
            "use_bias": True,
            "avg_pool": True,
        }

    def test_converted_output_simple(self):
        """Test that the output of a model converted to PIT is the same as the original nn.Module,
        before any training.
        """
        nn_ut = SimpleNN()
        pit_net = PIT(nn_ut, input_example=torch.rand(nn_ut.input_shape[1:]))
        self._check_output_equal(nn_ut, pit_net, nn_ut.input_shape[1:])

    def test_converted_output_advanced(self):
        """Test that the output of a model converted to PIT is the same as the original nn.Module,
        before any training.
        """
        input_shape = (6, 50)
        nn_ut = TCResNet14(self.tc_resnet_config)
        pit_net = PIT(nn_ut, input_example=torch.rand(input_shape))
        self._check_output_equal(nn_ut, pit_net, input_shape)

    def test_channel_mask_init(self):
        """Test initialization of channel masks"""
        nn_ut = ToySequentialConv1d()
        x = torch.rand((3, 15))
        pit_net = PIT(nn_ut, input_example=x)
        # check that the original channel mask is set with all 1
        self._check_channel_mask_init(pit_net, ('conv0', 'conv1'))

    def test_channel_mask_sharing(self):
        """Test that channel masks sharing works correctly"""
        nn_ut = ToyAdd()
        x = torch.rand((3, 15))
        pit_net = PIT(nn_ut, input_example=x)
        # check that the original channel mask is set with all 1
        self._check_channel_mask_init(pit_net, ('conv0', 'conv1'))
        mask0 = torch.Tensor([1, 1, 1, 1, 1, 1, 0, 0, 1, 1])
        self._write_channel_mask(pit_net, 'conv0', mask0)
        # since conv0 and conv1 share their maskers, we should see it also on conv1
        mask1 = self._read_channel_mask(pit_net, 'conv1')
        self.assertTrue(torch.all(mask0 == mask1), "Masks not correctly shared")
        # after a forward step, they should remain identical
        _ = pit_net(x)
        mask0 = self._read_channel_mask(pit_net, 'conv0')
        mask1 = self._read_channel_mask(pit_net, 'conv1')
        self.assertTrue(torch.all(mask0 == mask1), "Masks no longer equal after forward")

    def test_channel_mask_cat(self):
        """Test that layers fed into a cat operation over the channels axis have correct input
        features"""
        nn_ut = ToyChannelsCat()
        input_shape = (3, 15)
        pit_net = PIT(nn_ut, input_example=torch.rand(input_shape))
        mask0 = (torch.rand((10,)) > 0.5).float()
        self._write_channel_mask(pit_net, 'conv0', mask0)
        mask1 = (torch.rand((15,)) > 0.5).float()  # float but only 0 and 1
        self._write_channel_mask(pit_net, 'conv1', mask1)
        # execute model to propagate input features
        _ = pit_net(torch.stack([torch.rand(input_shape)] * 32, 0))
        exp_features = int(torch.sum(mask0)) + int(torch.sum(mask1))
        # first channels are always alive
        exp_features += 1 if mask0[0] == 0 else 0
        exp_features += 1 if mask1[0] == 0 else 0
        self._check_input_features(pit_net, {'conv2': exp_features})

    def test_rf_mask_init(self):
        """Test a pit layer receptive field masks at initialization"""
        nn_ut = ToySequentialConv1d()
        input_shape = (3, 15)
        x = torch.rand(input_shape)
        pit_net = PIT(nn_ut, input_example=x)
        self._check_rf_mask_init(pit_net, ('conv0', 'conv1'))

    def test_rf_mask_forced(self):
        """Test a pit layer receptive field masks forcing some beta values"""
        nn_ut = ToySequentialConv1d()
        input_shape = (3, 15)
        x = torch.rand(input_shape)
        pit_net = PIT(nn_ut, input_example=x)
        self._write_rf_mask(pit_net, 'conv0', torch.tensor([0.5, 0.5, 0.25]))
        pit_net(x)
        theta_beta = pit_net._inner_model.conv0.timestep_masker()  # type: ignore
        bin_theta_beta = PITBinarizer.apply(theta_beta, 0.5)
        # note: first element converted to 1 regardless of value (keep alive)
        theta_beta_exp = torch.tensor([1 + 0.5 + 0.25, 0.5 + 0.25, 0.25])
        bin_theta_beta_exp = torch.tensor([1, 1, 0])
        self.assertTrue(torch.all(theta_beta == theta_beta_exp))
        self.assertTrue(torch.all(bin_theta_beta == bin_theta_beta_exp))

        k_eff = pit_net._inner_model.conv0.k_eff  # type: ignore
        # obtained from the paper formulas based on normalization factors and beta values
        exp_norm_factors = torch.tensor([(1 / 3), (1 / 2), 1])
        k_eff_exp = torch.sum(torch.mul(theta_beta_exp, exp_norm_factors))
        self.assertAlmostEqual(float(k_eff), k_eff_exp)  # type: ignore

        self._write_rf_mask(pit_net, 'conv0', torch.tensor([0.4, 0.1, 0]))
        pit_net(x)
        theta_beta = pit_net._inner_model.conv0.timestep_masker()  # type: ignore
        bin_theta_beta = PITBinarizer.apply(theta_beta, 0.5)
        theta_beta_exp = torch.tensor([1 + 0.1, 0.1, 0])
        bin_theta_beta_exp = torch.tensor([1, 0, 0])
        self.assertTrue(torch.all(theta_beta == theta_beta_exp))
        self.assertTrue(torch.all(bin_theta_beta == bin_theta_beta_exp))

        k_eff = pit_net._inner_model.conv0.k_eff  # type: ignore
        k_eff_exp = torch.sum(torch.mul(theta_beta_exp, exp_norm_factors))
        self.assertAlmostEqual(float(k_eff), k_eff_exp)  # type: ignore

    def test_dilation_mask_init(self):
        """Test a pit layer dilation masks"""
        nn_ut = ToySequentialConv1d()
        input_shape = (3, 15)
        x = torch.rand(input_shape)
        pit_net = PIT(nn_ut, input_example=x)
        self._check_dilation_mask_init(pit_net, ('conv0', 'conv1'))

    # TODO: continue testing from here
    def test_keep_alive_masks_simple_SimpleNN(self):
        """Test keep alive mask on SimpleNN network"""
        net = SimpleNN()
        pit_net = PIT(net, input_example=torch.rand((3, 40)))
        # conv1 has a filter size of 5 and 57 output channels
        # note: the type: ignore tells pylance to ignore type checks on the next line
        ka_alpha = pit_net._inner_model.conv1.out_channel_masker._keep_alive  # type: ignore
        exp_ka_alpha = torch.tensor([1.0] + [0.0] * 56, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_alpha,  # type: ignore
                                    exp_ka_alpha), "Wrong keep-alive \
                                                    mask for channels")  # type: ignore
        ka_beta = pit_net._inner_model.conv1.timestep_masker._keep_alive  # type: ignore
        exp_ka_beta = torch.tensor([1.0] + [0.0] * 4, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_beta,   # type: ignore
                                    exp_ka_beta), "Wrong keep-alive \
                                                   mask for rf")  # type: ignore
        ka_gamma = pit_net._inner_model.conv1.dilation_masker._keep_alive  # type: ignore
        exp_ka_gamma = torch.tensor([1.0] + [0.0] * 2, dtype=torch.float32)  # type: ignore
        self.assertTrue(torch.equal(ka_gamma,  # type: ignore
                                    exp_ka_gamma), "Wrong keep-alive \
                                                    mask for dilation")  # type: ignore

    def test_keep_alive_masks_simple_ToyModel7(self):
        """Test keep alive mask on ToyModel7 network"""
        net = ToyModel7()
        pit_net = PIT(net, input_example=torch.rand((3, 15)))
        # conv1 has a filter size of 7 and 10 output channels
        ka_alpha = pit_net._inner_model.conv1.out_channel_masker._keep_alive  # type: ignore
        exp_ka_alpha = torch.tensor([1.0] + [0.0] * 9, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_alpha,  # type: ignore
                                    exp_ka_alpha), "Wrong keep-alive \
                                                    mask for channels")  # type: ignore
        ka_beta = pit_net._inner_model.conv1.timestep_masker._keep_alive  # type: ignore
        exp_ka_beta = torch.tensor([1.0] + [0.0] * 6, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_beta,  # type: ignore
                                    exp_ka_beta), "Wrong keep-alive \
                                                  mask for rf")  # type: ignore
        ka_gamma = pit_net._inner_model.conv1.dilation_masker._keep_alive  # type: ignore
        exp_ka_gamma = torch.tensor([1.0] + [0.0] * 2, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_gamma,  # type: ignore
                                    exp_ka_gamma), "Wrong keep-alive \
                                                    mask for dilation")  # type: ignore

    # def test_keep_alive_masks_simple_ToyModel2(self):
    #     """Test keep alive mask on ToyModel2 network"""
    #     net = ToyModel2()
    #     pit_net = PIT(net, input_example=torch.rand((3, 60)))
    #     # conv1 has a filter size of 3 and 40 output channels
    #     ka_alpha = pit_net._inner_model.conv1.out_channel_masker._keep_alive  # type: ignore
    #     exp_ka_alpha = torch.tensor([1.0] + [0.0] * 39, dtype=torch.float32)
    #     self.assertTrue(torch.equal(ka_alpha,  # type: ignore
    #                                 exp_ka_alpha), "Wrong keep-alive \
    #                                                 mask for channels")  # type: ignore
    #     ka_beta = pit_net._inner_model.conv1.timestep_masker._keep_alive  # type: ignore
    #     exp_ka_beta = torch.tensor([1.0] + [0.0] * 2, dtype=torch.float32)
    #     self.assertTrue(torch.equal(ka_beta,   # type: ignore
    #                                 exp_ka_beta), "Wrong keep-alive \
    #                                               mask for rf")  # type: ignore
    #     ka_gamma = pit_net._inner_model.conv1.dilation_masker._keep_alive  # type: ignore
    #     exp_ka_gamma = torch.tensor([1.0] + [0.0] * 1, dtype=torch.float32)
    #     self.assertTrue(torch.equal(ka_gamma,   # type: ignore
    #                                 exp_ka_gamma), "Wrong keep-alive \
    #                                                 mask for dilation")  # type: ignore

    def test_c_matrices_SimpleNN(self):
        """Test c_beta and c_gamma matrices on SimpleNN network"""
        net = SimpleNN()
        pit_net = PIT(net, input_example=torch.rand((3, 40)))
        # conv1 has a filter size of 5 and 57 output channels
        c_beta = pit_net._inner_model.conv1.timestep_masker._c_beta  # type: ignore
        exp_c_beta = torch.tensor([
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ], dtype=torch.float32)
        self.assertTrue(torch.equal(c_beta, exp_c_beta), "Wrong C beta matrix")  # type: ignore
        c_gamma = pit_net._inner_model.conv1.dilation_masker._c_gamma  # type: ignore
        exp_c_gamma = torch.tensor([
            [1, 1, 1],
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ], dtype=torch.float32)
        self.assertTrue(torch.equal(c_gamma, exp_c_gamma), "Wrong C gamma matrix")  # type: ignore

    def test_c_matrices_ToyModel7(self):
        """Test c_beta and c_gamma matrices on ToyModel7 network"""
        net = ToyModel7()
        pit_net = PIT(net, input_example=torch.rand((3, 15)))
        # conv1 has a filter size of 7 and 10 output channels
        c_beta = pit_net._inner_model.conv1.timestep_masker._c_beta  # type: ignore
        exp_c_beta = torch.Tensor([[1., 1., 1., 1., 1., 1., 1.],
                                   [0., 1., 1., 1., 1., 1., 1.],
                                   [0., 0., 1., 1., 1., 1., 1.],
                                   [0., 0., 0., 1., 1., 1., 1.],
                                   [0., 0., 0., 0., 1., 1., 1.],
                                   [0., 0., 0., 0., 0., 1., 1.],
                                   [0., 0., 0., 0., 0., 0., 1.]])
        self.assertTrue(torch.equal(c_beta, exp_c_beta), "Wrong C beta matrix")  # type: ignore
        c_gamma = pit_net._inner_model.conv1.dilation_masker._c_gamma  # type: ignore
        exp_c_gamma = torch.Tensor([[1., 1., 1.],
                                    [0., 0., 1.],
                                    [0., 1., 1.],
                                    [0., 0., 1.],
                                    [1., 1., 1.],
                                    [0., 0., 1.],
                                    [0., 1., 1.]])
        self.assertTrue(torch.equal(c_gamma, exp_c_gamma), "Wrong C gamma matrix")  # type: ignore

    def test_initial_inference(self):
        """ check that a PITModel just created returns the same output as its inner model"""
        net = SimpleNN()
        x = torch.rand(tuple(net.input_shape[1:]))
        pit_net = PIT(net, input_example=x)
        x = torch.stack([x] * 32, 0)
        net.eval()
        pit_net.eval()
        y = net(x)
        pit_y = pit_net(x)
        assert torch.all(torch.eq(y, pit_y))

    def test_regularization_loss_get_size_macs_ToyModel6(self):
        """Test the regularization loss computation on ToyModel6"""
        net = ToyModel6()
        pit_net = PIT(net, input_example=torch.rand((3, 15)))
        # Check the number of weights for a single conv layer
        conv1_size = pit_net._inner_model.conv1.get_size().item()  # type: ignore
        input_features = pit_net._inner_model.conv1\
                                             .input_features_calculator.features  # type: ignore
        output_channels = pit_net._inner_model.conv1.out_channels_eff  # type: ignore
        k_eff = pit_net._inner_model.conv1.k_eff  # type: ignore
        self.assertEqual(conv1_size,
                         input_features * output_channels * k_eff,  # type: ignore
                         "Wrong layer size computed")  # type: ignore
        # Check the number of MACs for a single conv layer
        conv1_macs = pit_net._inner_model.conv1.get_macs()  # type: ignore
        out_length = pit_net._inner_model.conv1.out_length  # type: ignore
        self.assertEqual(conv1_macs,
                         input_features * output_channels * k_eff * out_length,  # type: ignore
                         "Wrong layer MACs computed")  # type: ignore
        # Check the number of weights for the whole net
        self.assertEqual(pit_net.get_size().item(),
                         # conv0        conv1          conv2
                         (3 * 10 * 3) + (3 * 10 * 3) + (20 * 4 * 9),
                         "Wrong net size computed")  # type: ignore
        # Check the number of weights for the whole net
        self.assertEqual(pit_net.get_macs().item(),
                         # conv0             conv1               conv2
                         (3 * 10 * 3 * 15) + (3 * 10 * 3 * 15) + (20 * 4 * 9 * 15),
                         "Wrong MACs size computed")  # type: ignore

    def test_regularization_loss_get_size_macs_ToyModel7(self):
        """Test the regularization loss computation on ToyModel7"""
        net = ToyModel7()
        pit_net = PIT(net, input_example=torch.rand((3, 15)))
        # Check the number of weights for a single conv layer
        conv2_size = pit_net._inner_model.conv2.get_size().item()  # type: ignore
        input_features = pit_net._inner_model.conv2\
                                             .input_features_calculator.features  # type: ignore
        output_channels = pit_net._inner_model.conv2.out_channels_eff  # type: ignore
        k_eff = pit_net._inner_model.conv2.k_eff  # type: ignore
        self.assertEqual(conv2_size,
                         input_features * output_channels * k_eff,  # type: ignore
                         "Wrong layer size computed")  # type: ignore
        # Check the number of MACs for a single conv layer
        conv2_macs = pit_net._inner_model.conv2.get_macs()  # type: ignore
        out_length = pit_net._inner_model.conv2.out_length  # type: ignore
        self.assertEqual(conv2_macs,
                         input_features * output_channels * k_eff * out_length,  # type: ignore
                         "Wrong layer MACs computed")  # type: ignore
        # Check the number of weights for the whole net
        self.assertEqual(pit_net.get_size().item(),
                         # conv0         conv1         conv2
                         (3 * 10 * 7) + (3 * 10 * 7) + (20 * 4 * 9),
                         "Wrong net size computed")  # type: ignore
        # Check the number of weights for the whole net
        self.assertEqual(pit_net.get_macs().item(),
                         # conv0             conv1               conv2
                         (3 * 10 * 7 * 15) + (3 * 10 * 7 * 15) + (20 * 4 * 9 * 15),
                         "Wrong MACs size computed")  # type: ignore

    def test_regularization_loss_forward_backward_ToyModel4(self):
        """Test the regularization loss after forward and backward steps on ToyModel4"""
        nn_ut = ToyAdd()
        x = torch.rand(tuple(nn_ut.input_shape[1:]))  # type: ignore
        pit_net = PIT(nn_ut, input_example=x)
        optimizer = optim.Adam(pit_net.parameters())
        pit_net.eval()
        inputs = []
        for i in range(8):
            inputs.append(torch.rand((32,) + tuple(nn_ut.input_shape[1:])))  # type: ignore
        prev_loss = 0
        for ix, el in enumerate(inputs):
            pit_net(el)
            loss = pit_net.get_regularization_loss()
            if ix > 0:
                flag_loss = loss < prev_loss
                self.assertTrue(flag_loss, "The loss value is not descending")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prev_loss = loss

    def test_regularization_network_weights_ToyModel3(self):
        """Check the weights remain equal using the regularization loss on ToyModel3"""
        nn_ut = ToyModel3()
        x = torch.rand(tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x)
        optimizer = optim.Adam(pit_net.parameters())
        pit_net.eval()
        inputs = []
        for i in range(8):
            inputs.append(torch.rand((32,) + tuple(nn_ut.input_shape[1:])))
        prev_conv0_weights = 0
        prev_conv2_weights = 0
        prev_conv6_weights = 0
        for ix, el in enumerate(inputs):
            pit_net(el)
            loss = pit_net.get_regularization_loss()
            conv0_weights = pit_net._inner_model.conv0.weight  # type: ignore
            conv0_weights = conv0_weights.detach().numpy()  # type: ignore
            conv0_weights = np.array(conv0_weights, dtype=float)
            conv2_weights = pit_net._inner_model.conv2.weight  # type: ignore
            conv2_weights = conv2_weights.detach().numpy()  # type: ignore
            conv2_weights = np.array(conv2_weights, dtype=float)
            conv6_weights = pit_net._inner_model.conv6.weight  # type: ignore
            conv6_weights = conv6_weights.detach().numpy()  # type: ignore
            conv6_weights = np.array(conv6_weights, dtype=float)
            # Using only the regularization loss the weights of the network should not change
            if ix > 0:
                self.assertTrue(np.isclose(prev_conv0_weights,
                                           conv0_weights, atol=1e-25).all())  # type: ignore
                self.assertTrue(np.isclose(prev_conv2_weights,
                                           conv2_weights, atol=1e-25).all())  # type: ignore
                self.assertTrue(np.isclose(prev_conv6_weights,
                                           conv6_weights, atol=1e-25).all())  # type: ignore
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prev_conv0_weights = conv0_weights
            prev_conv2_weights = conv2_weights
            prev_conv6_weights = conv6_weights

    def test_regularization_network_weights_ToyModel4(self):
        """Check the value of the weights using the regularization loss on ToyModel4"""
        nn_ut = ToyAdd()
        x = torch.rand(tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x)
        optimizer = optim.Adam(pit_net.parameters())
        pit_net.eval()
        inputs = []
        for i in range(8):
            inputs.append(torch.rand((32,) + tuple(nn_ut.input_shape[1:])))
        prev_conv0_weights = 0
        prev_conv1_weights = 0
        prev_conv2_weights = 0
        for ix, el in enumerate(inputs):
            pit_net(el)
            loss = pit_net.get_regularization_loss()
            conv0_weights = pit_net._inner_model.conv0.weight  # type: ignore
            conv0_weights = conv0_weights.detach().numpy()  # type: ignore
            conv0_weights = np.array(conv0_weights, dtype=float)
            conv1_weights = pit_net._inner_model.conv1.weight  # type: ignore
            conv1_weights = conv1_weights.detach().numpy()  # type: ignore
            conv1_weights = np.array(conv1_weights, dtype=float)
            conv2_weights = pit_net._inner_model.conv2.weight  # type: ignore
            conv2_weights = conv2_weights.detach().numpy()  # type: ignore
            conv2_weights = np.array(conv2_weights, dtype=float)
            # Using only the regularization loss the weights of the network should not change
            if ix > 0:
                self.assertTrue(np.isclose(prev_conv0_weights,
                                           conv0_weights, atol=1e-25).all())  # type: ignore
                self.assertTrue(np.isclose(prev_conv1_weights,
                                           conv1_weights, atol=1e-25).all())  # type: ignore
                self.assertTrue(np.isclose(prev_conv2_weights,
                                           conv2_weights, atol=1e-25).all())  # type: ignore
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prev_conv0_weights = conv0_weights
            prev_conv1_weights = conv1_weights
            prev_conv2_weights = conv2_weights

    def test_regularization_loss_forward_backward_ToyModel2(self):
        """Test the regularization loss after forward and backward steps on ToyModel2"""
        nn_ut = ToyModel2()
        x = torch.rand(tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x)
        optimizer = optim.Adam(pit_net.parameters())
        pit_net.eval()
        inputs = []
        for i in range(20):
            inputs.append(torch.rand((32,) + tuple(nn_ut.input_shape[1:])))
        prev_loss = 0
        for ix, el in enumerate(inputs):
            pit_net(el)
            loss = pit_net.get_regularization_loss()
            if ix > 0:
                flag_loss = loss < prev_loss
                self.assertTrue(flag_loss, "The loss value is not descending")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prev_loss = loss

    def test_regularization_loss_masks_ToyModel1(self):
        """The ToyModel1 alpha/beta/gamma masks should go to 0 using only the regularization loss"""
        nn_ut = MultiPath1()
        x = torch.rand(tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x)
        optimizer = optim.Adam(pit_net.parameters())
        pit_net.eval()
        inputs = []
        for i in range(800):
            inputs.append(torch.rand((32,) + tuple(nn_ut.input_shape[1:])))
        for el in inputs:
            pit_net(el)
            loss = pit_net.get_regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        conv3_out_channels = pit_net._inner_model.conv3.out_channels - 1  # type: ignore
        exp_conv3_alpha = torch.tensor([1.0] + [0.0] * conv3_out_channels, dtype=torch.float32)
        conv3_alpha = pit_net._inner_model.conv3.out_channel_masker.alpha  # type: ignore
        conv3_alpha = Parameter(PITBinarizer.apply(conv3_alpha, 0.5))
        self.assertTrue(torch.equal(conv3_alpha,  # type: ignore
                                    exp_conv3_alpha), "The channel mask values should decrease \
                                                     under the threshold target")  # type: ignore
        conv3_rf = pit_net._inner_model.conv3.timestep_masker.rf - 1   # type: ignore
        exp_conv3_beta = torch.tensor([1.0] + [0.0] * conv3_rf, dtype=torch.float32)
        conv3_beta = pit_net._inner_model.conv3.timestep_masker.beta  # type: ignore
        conv3_beta = Parameter(PITBinarizer.apply(conv3_beta, 0.5))
        self.assertTrue(torch.equal(conv3_beta,  # type: ignore
                                    exp_conv3_beta), "The kernel mask values should decrease \
                                                     under the threshold target")  # type: ignore
        exp_conv3_gamma = torch.tensor([1.0] + [0.0], dtype=torch.float32)
        conv3_gamma = pit_net._inner_model.conv3.dilation_masker.gamma  # type: ignore
        conv3_gamma = Parameter(PITBinarizer.apply(conv3_gamma, 0.5))
        self.assertTrue(torch.equal(conv3_gamma,  # type: ignore
                                    exp_conv3_gamma), "The dilation mask values should decrease \
                                                     under the threshold target")  # type: ignore

    def test_regularization_loss_masks_ToyModel7(self):
        """The ToyModel7 alpha/beta/gamma masks should go to 0 using only the regularization loss"""
        nn_ut = ToyModel7()
        x = torch.rand(tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x)
        optimizer = optim.Adam(pit_net.parameters())
        pit_net.eval()
        inputs = []
        for i in range(1100):
            inputs.append(torch.rand((32,) + tuple(nn_ut.input_shape[1:])))
        for el in inputs:
            pit_net(el)
            loss = pit_net.get_regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        conv2_out_channels = pit_net._inner_model.conv2.out_channels - 1  # type: ignore
        exp_conv2_alpha = torch.tensor([1.0] + [0.0] * conv2_out_channels, dtype=torch.float32)
        conv2_alpha = pit_net._inner_model.conv2.out_channel_masker.alpha  # type: ignore
        conv2_alpha = Parameter(PITBinarizer.apply(conv2_alpha, 0.5))
        self.assertTrue(torch.equal(conv2_alpha,  # type: ignore
                                    exp_conv2_alpha), "The channel mask values should decrease \
                                                     under the threshold target")  # type: ignore
        conv2_rf = pit_net._inner_model.conv2.timestep_masker.rf - 1   # type: ignore
        exp_conv2_beta = torch.tensor([1.0] + [0.0] * conv2_rf, dtype=torch.float32)
        conv2_beta = pit_net._inner_model.conv2.timestep_masker.beta  # type: ignore
        conv2_beta = Parameter(PITBinarizer.apply(conv2_beta, 0.5))
        self.assertTrue(torch.equal(conv2_beta,  # type: ignore
                                    exp_conv2_beta), "The kernel mask values should decrease \
                                                     under the threshold target")  # type: ignore
        exp_conv2_gamma = torch.tensor([1.0] + [0.0] * 3, dtype=torch.float32)
        conv2_gamma = pit_net._inner_model.conv2.dilation_masker.gamma  # type: ignore
        conv2_gamma = Parameter(PITBinarizer.apply(conv2_gamma, 0.5))
        self.assertTrue(torch.equal(conv2_gamma,  # type: ignore
                                    exp_conv2_gamma), "The dilation mask values should decrease \
                                                     under the threshold target")  # type: ignore

    def test_layer_optimizations_ToyModel4(self):
        """The ToyModel4 alpha masks should remain fixed to 1 with train_channels=False"""
        nn_ut = ToyAdd()
        x = torch.rand(tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x, train_channels=False)
        optimizer = optim.Adam(pit_net.parameters())
        pit_net.eval()
        inputs = []
        for i in range(600):
            inputs.append(torch.rand((32,) + tuple(nn_ut.input_shape[1:])))
        for el in inputs:
            pit_net(el)
            loss = pit_net.get_regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        conv1_out_channels = pit_net._inner_model.conv1.out_channels - 1  # type: ignore
        exp_conv1_alpha = torch.tensor([1.0] + [1.0] * conv1_out_channels, dtype=torch.float32)
        conv1_alpha = pit_net._inner_model.conv1.out_channel_masker.alpha  # type: ignore
        conv1_alpha = Parameter(PITBinarizer.apply(conv1_alpha, 0.5))
        self.assertTrue(torch.equal(conv1_alpha,  # type: ignore
                                    exp_conv1_alpha), "The channel mask values should remain \
                                                      fixed to 1 with \
                                                      train_channels=False")  # type: ignore
        conv1_rf = pit_net._inner_model.conv1.timestep_masker.rf - 1   # type: ignore
        exp_conv1_beta = torch.tensor([1.0] + [0.0] * conv1_rf, dtype=torch.float32)
        conv1_beta = pit_net._inner_model.conv1.timestep_masker.beta  # type: ignore
        conv1_beta = Parameter(PITBinarizer.apply(conv1_beta, 0.5))
        self.assertTrue(torch.equal(conv1_beta,  # type: ignore
                                    exp_conv1_beta), "The kernel mask values should decrease \
                                                     under the threshold target")  # type: ignore
        exp_conv1_gamma = torch.tensor([1.0] + [0.0], dtype=torch.float32)
        conv1_gamma = pit_net._inner_model.conv1.dilation_masker.gamma  # type: ignore
        conv1_gamma = Parameter(PITBinarizer.apply(conv1_gamma, 0.5))
        self.assertTrue(torch.equal(conv1_gamma,  # type: ignore
                                    exp_conv1_gamma), "The dilation mask values should decrease \
                                                     under the threshold target")  # type: ignore

    def test_layer_optimizations_ToyModel3(self):
        """The ToyModel4 beta masks should remain fixed to 1 with train_rf=False"""
        nn_ut = ToyModel3()
        x = torch.rand(tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x, train_rf=False)
        optimizer = optim.Adam(pit_net.parameters())
        pit_net.eval()
        inputs = []
        for i in range(600):
            inputs.append(torch.rand((32,) + tuple(nn_ut.input_shape[1:])))
        for el in inputs:
            pit_net(el)
            loss = pit_net.get_regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        conv6_out_channels = pit_net._inner_model.conv6.out_channels - 1  # type: ignore
        exp_conv6_alpha = torch.tensor([1.0] + [0.0] * conv6_out_channels, dtype=torch.float32)
        conv6_alpha = pit_net._inner_model.conv6.out_channel_masker.alpha  # type: ignore
        conv6_alpha = Parameter(PITBinarizer.apply(conv6_alpha, 0.5))
        self.assertTrue(torch.equal(conv6_alpha,  # type: ignore
                                    exp_conv6_alpha), "The channel mask values should decrease \
                                                     under the threshold target")  # type: ignore
        conv6_rf = pit_net._inner_model.conv6.timestep_masker.rf - 1   # type: ignore
        exp_conv6_beta = torch.tensor([1.0] + [1.0] * conv6_rf, dtype=torch.float32)
        conv6_beta = pit_net._inner_model.conv6.timestep_masker.beta  # type: ignore
        conv6_beta = Parameter(PITBinarizer.apply(conv6_beta, 0.5))
        self.assertTrue(torch.equal(conv6_beta,  # type: ignore
                                    exp_conv6_beta), "The kernel mask values should remain \
                                                      fixed to 1 with \
                                                      train_rf=False")  # type: ignore
        exp_conv6_gamma = torch.tensor([1.0] + [0.0], dtype=torch.float32)
        conv6_gamma = pit_net._inner_model.conv6.dilation_masker.gamma  # type: ignore
        conv6_gamma = Parameter(PITBinarizer.apply(conv6_gamma, 0.5))
        self.assertTrue(torch.equal(conv6_gamma,  # type: ignore
                                    exp_conv6_gamma), "The dilation mask values should decrease \
                                                     under the threshold target")  # type: ignore

    def test_layer_optimizations_ToyModel6(self):
        """The ToyModel6 gamma masks should remain fixed to 1 with train_dilation=False"""
        nn_ut = ToyModel6()
        x = torch.rand(tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x, train_dilation=False)
        optimizer = optim.Adam(pit_net.parameters())
        pit_net.eval()
        inputs = []
        for i in range(900):
            inputs.append(torch.rand((32,) + tuple(nn_ut.input_shape[1:])))
        for el in inputs:
            pit_net(el)
            loss = pit_net.get_regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        conv2_out_channels = pit_net._inner_model.conv2.out_channels - 1  # type: ignore
        exp_conv2_alpha = torch.tensor([1.0] + [0.0] * conv2_out_channels, dtype=torch.float32)
        conv2_alpha = pit_net._inner_model.conv2.out_channel_masker.alpha  # type: ignore
        conv2_alpha = Parameter(PITBinarizer.apply(conv2_alpha, 0.5))
        self.assertTrue(torch.equal(conv2_alpha,  # type: ignore
                                    exp_conv2_alpha), "The channel mask values should decrease \
                                                     under the threshold target")  # type: ignore
        conv2_rf = pit_net._inner_model.conv2.timestep_masker.rf - 1   # type: ignore
        exp_conv2_beta = torch.tensor([1.0] + [0.0] * conv2_rf, dtype=torch.float32)
        conv2_beta = pit_net._inner_model.conv2.timestep_masker.beta  # type: ignore
        conv2_beta = Parameter(PITBinarizer.apply(conv2_beta, 0.5))
        self.assertTrue(torch.equal(conv2_beta,  # type: ignore
                                    exp_conv2_beta), "The dilation mask values should decrease \
                                                     under the threshold target")  # type: ignore
        exp_conv2_gamma = torch.tensor([1.0] + [1.0] * 3, dtype=torch.float32)
        conv2_gamma = pit_net._inner_model.conv2.dilation_masker.gamma  # type: ignore
        conv2_gamma = Parameter(PITBinarizer.apply(conv2_gamma, 0.5))
        self.assertTrue(torch.equal(conv2_gamma,  # type: ignore
                                    exp_conv2_gamma), "The dilation mask values should remain \
                                                      fixed to 1 with \
                                                      train_dilation=False")  # type: ignore

    def test_combined_loss_ToyModel8(self):
        """Check the changes in the weights using the combined loss"""
        nn_ut = ToyModel8()
        batch_size = 5
        x = torch.rand(tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x)
        x = torch.stack([x] * 32, 0)
        nn_ut.eval()
        pit_net.eval()
        y = nn_ut(x)
        pit_y = pit_net(x)
        assert torch.all(torch.eq(y, pit_y))
        optimizer = optim.Adam(pit_net.parameters())
        lambda_param = 0.0005
        inputs = []
        for i in range(4):
            if torch.rand(1) < 0.5:
                inputs.append((torch.rand((batch_size,) + tuple(nn_ut.input_shape[1:])),
                               torch.zeros(batch_size, dtype=torch.long)))
            else:
                inputs.append((torch.rand((batch_size,) + tuple(nn_ut.input_shape[1:])),
                               torch.ones(batch_size, dtype=torch.long)))
        prev_conv1_weights = 0
        for i in range(50):
            for ix, el in enumerate(inputs):
                input, target = el[0], el[1]
                output = pit_net(input)
                task_loss = nn.CrossEntropyLoss()(output, target)
                nas_loss = lambda_param * pit_net.get_regularization_loss()
                total_loss = task_loss + nas_loss
                conv1_weights = pit_net._inner_model.conv1.weight  # type: ignore
                conv1_weights = conv1_weights.detach().numpy()  # type: ignore
                conv1_weights = np.array(conv1_weights, dtype=float)
                # Using the crossentropy loss combined with the regularization loss
                # the weights of the network should change during the training.
                if ix > 0:
                    self.assertFalse(np.isclose(prev_conv1_weights,
                                     conv1_weights, atol=1e-25).all())  # type: ignore
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                prev_conv1_weights = conv1_weights
            # print(pit_net._inner_model.fc.weight)  # type: ignore

    def test_combined_loss_ones_labels_CE_ToyModel8(self):
        """Check the output of the combined loss with all labels equal to 1"""
        nn_ut = ToyModel8()
        batch_size = 5
        x = torch.rand(tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x)
        x = torch.stack([x] * 32, 0)
        pit_net.eval()
        optimizer = optim.Adam(pit_net.parameters())
        lambda_param = 0.0005
        inputs = []
        for i in range(80):
            inputs.append((torch.full((batch_size,) + tuple(nn_ut.input_shape[1:]), 10,
                           dtype=torch.float32),
                           torch.ones(batch_size, dtype=torch.long)))
        output_check = 0
        for i in range(5):
            for ix, el in enumerate(inputs):
                input, target = el[0], el[1]
                output = pit_net(input)
                task_loss = nn.CrossEntropyLoss()(output, target)
                nas_loss = lambda_param * pit_net.get_regularization_loss()
                total_loss = task_loss + nas_loss
                conv1_weights = pit_net._inner_model.conv1.weight  # type: ignore
                conv1_weights = conv1_weights.detach().numpy()  # type: ignore
                conv1_weights = np.array(conv1_weights, dtype=float)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                output_check = output
        self.assertTrue(torch.sum(torch.argmax(output_check, dim=-1)) == batch_size,  # type: ignore
                        "The network should output only 1 with all labels 1")  # type: ignore

    def test_combined_loss_zero_labels_CE_ToyModel8(self):
        """Check the output of the combined loss with all labels equal to 0"""
        nn_ut = ToyModel8()
        batch_size = 5
        x = torch.rand(tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x)
        x = torch.stack([x] * 32, 0)
        pit_net.eval()
        optimizer = optim.Adam(pit_net.parameters())
        lambda_param = 0.0005
        inputs = []
        for i in range(80):
            inputs.append((torch.full((batch_size,) + tuple(nn_ut.input_shape[1:]), 40,
                           dtype=torch.float32),
                           torch.zeros(batch_size, dtype=torch.long)))
        output_check = 0
        for i in range(5):
            for ix, el in enumerate(inputs):
                input, target = el[0], el[1]
                output = pit_net(input)
                task_loss = nn.CrossEntropyLoss()(output, target)
                nas_loss = lambda_param * pit_net.get_regularization_loss()
                total_loss = task_loss + nas_loss
                conv1_weights = pit_net._inner_model.conv1.weight  # type: ignore
                conv1_weights = conv1_weights.detach().numpy()  # type: ignore
                conv1_weights = np.array(conv1_weights, dtype=float)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                output_check = output
        self.assertTrue(torch.sum(torch.argmax(output_check, dim=-1)) == 0,  # type: ignore
                        "The network should output only 0 with all labels 0")  # type: ignore

    def test_combined_loss_MSE_ToyModel8(self):
        """Check the output of the combined loss with all labels equal to 0"""
        nn_ut = ToyModel8()
        batch_size = 5
        x = torch.rand(tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x)
        x = torch.stack([x] * 32, 0)
        pit_net.eval()
        optimizer = optim.Adam(pit_net.parameters())
        lambda_param = 0.0005
        inputs = []
        for i in range(80):
            inputs.append((torch.full((batch_size,) + tuple(nn_ut.input_shape[1:]), 40,
                           dtype=torch.float32),
                           torch.full((batch_size,) + tuple((2,)), 3,
                           dtype=torch.float32)))
        output_check = 0
        target_check = 0
        for i in range(50):
            for ix, el in enumerate(inputs):
                input, target = el[0], el[1]
                output = pit_net(input)
                task_loss = nn.MSELoss()(output, target)
                nas_loss = lambda_param * pit_net.get_regularization_loss()
                total_loss = task_loss + nas_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                output_check = output
                target_check = target
        output_check = output_check.detach().numpy()  # type: ignore
        output_check = np.array(output_check, dtype=float)
        target_check = target_check.detach().numpy()  # type: ignore
        target_check = np.array(target_check, dtype=float)
        self.assertTrue(np.isclose(output_check,
                                   target_check, atol=1e-1).all())  # type: ignore

    def _check_output_equal(self, nn: nn.Module, pit_nn: PIT, input_shape: Tuple[int, ...],
                            iterations=10):
        for i in range(iterations):
            # add batch size in front
            x = torch.rand((32,) + input_shape)
            nn.eval()
            pit_nn.eval()
            y = nn(x)
            pit_y = pit_nn(x)
            self.assertTrue(torch.all(torch.eq(y, pit_y)), "Wrong output of PIT model")

    def _check_channel_mask_init(self, nn: PIT, check_layers: Tuple[str, ...]):
        """Check if the channel masks are initialized correctly"""
        converted_layer_names = dict(nn._inner_model.named_modules())
        for layer_name in check_layers:
            layer = converted_layer_names[layer_name]
            if isinstance(layer, PITConv1d):
                alpha = layer.out_channel_masker.alpha
                check = torch.ones((layer.out_channels,))
                self.assertTrue(torch.all(alpha == check), "Wrong alpha values")
                self.assertEqual(torch.sum(alpha), layer.out_channels, "Wrong alpha sum")
                self.assertEqual(torch.sum(alpha), layer.out_channels_eff, "Wrong channels eff")
                self.assertEqual(torch.sum(alpha), layer.out_channels_opt, "Wrong channels opt")

    def _check_rf_mask_init(self, nn: PIT, check_layers: Tuple[str, ...]):
        """Check if the RF masks are initialized correctly"""
        converted_layer_names = dict(nn._inner_model.named_modules())
        for layer_name in check_layers:
            layer = converted_layer_names[layer_name]
            if isinstance(layer, PITConv1d):
                kernel_size = layer.kernel_size[0]
                beta = layer.timestep_masker.beta
                check = torch.ones((kernel_size,))
                self.assertTrue(torch.all(beta == check), "Wrong beta values")
                c_check = []
                for i in range(kernel_size):
                    c_check.append([0] * i + [1] * (kernel_size - i))
                c_check = torch.tensor(c_check)
                c_beta = layer.timestep_masker._c_beta
                self.assertTrue(torch.all(c_beta == c_check), "Wrong C beta matrix")
                gamma_beta = layer.timestep_masker()
                gamma_check = torch.tensor(list(range(1, kernel_size + 1))[::-1])
                self.assertTrue(torch.all(gamma_beta == gamma_check), "Wrong theta beta array")

    def _check_dilation_mask_init(self, nn: PIT, check_layers: Tuple[str, ...]):
        """Check if the dilation masks are initialized correctly"""
        converted_layer_names = dict(nn._inner_model.named_modules())
        for layer_name in check_layers:
            layer = converted_layer_names[layer_name]
            if isinstance(layer, PITConv1d):
                kernel_size = layer.kernel_size[0]
                rf_seed = (kernel_size - 1) * layer.dilation[0] + 1
                gamma = layer.dilation_masker.gamma
                len_gamma_exp = math.ceil(math.log2(rf_seed))
                check = torch.ones((len_gamma_exp,))
                self.assertTrue(torch.all(gamma == check), "Wrong gamma values")

                max_dilation = 2**(len_gamma_exp - 1)
                c_check_tr = []
                for i in range(len_gamma_exp):
                    row = []
                    j = 0
                    while j < kernel_size:
                        row = row + [1]
                        j += 1
                        num_zeros = max(min(max_dilation - 1, kernel_size - j - 1), 0)
                        row = row + ([0] * num_zeros)
                        j += num_zeros
                    c_check_tr.append(row)
                    max_dilation //= 2
                c_check = torch.transpose(torch.tensor(c_check_tr), 1, 0)
                c_gamma = layer.dilation_masker._c_gamma
                self.assertTrue(torch.all(c_check == c_gamma), "Wrong C gamma matrix")

                c_theta = []
                for i in range(rf_seed):
                    k_i = sum([1 - int(i % (2**p) == 0) for p in range(1, len_gamma_exp)])
                    val = sum([check[len_gamma_exp - j] for j in range(1, len_gamma_exp - k_i + 1)])
                    c_theta.append(val)
                c_theta = torch.tensor(c_theta)
                theta_gamma = layer.dilation_masker()
                self.assertTrue(torch.all(c_theta == theta_gamma), "Wrong theta gamma array")


    def _write_channel_mask(self, nn: PIT, layer_name: str, mask: torch.Tensor):
        """Force a given value on the output channels mask"""
        converted_layer_names = dict(nn._inner_model.named_modules())
        layer = converted_layer_names[layer_name]
        layer.out_channel_masker.alpha = Parameter(mask)  # type: ignore

    def _write_rf_mask(self, nn: PIT, layer_name: str, mask: torch.Tensor):
        """Force a given value on the rf mask"""
        converted_layer_names = dict(nn._inner_model.named_modules())
        layer = converted_layer_names[layer_name]
        layer.timestep_masker.beta = Parameter(mask)  # type: ignore

    def _write_dilation_mask(self, nn: PIT, layer_name: str, mask: torch.Tensor):
        """Force a given value on the dilation mask"""
        converted_layer_names = dict(nn._inner_model.named_modules())
        layer = converted_layer_names[layer_name]
        layer.dilation_masker.gamma = Parameter(mask)  # type: ignore

    def _read_channel_mask(self, nn: PIT, layer_name: str) -> torch.Tensor:
        """Read a value from the output channels mask of a layer"""
        converted_layer_names = dict(nn._inner_model.named_modules())
        layer = converted_layer_names[layer_name]
        return layer.out_channel_masker.alpha  # type: ignore

    def _read_rf_mask(self, nn: PIT, layer_name: str) -> torch.Tensor:
        """Read a value from the rf mask of a layer"""
        converted_layer_names = dict(nn._inner_model.named_modules())
        layer = converted_layer_names[layer_name]
        return layer.timestep_masker.beta  # type: ignore

    def _read_dilation_mask(self, nn: PIT, layer_name: str) -> torch.Tensor:
        """Read a value from the dilation mask of a layer"""
        converted_layer_names = dict(nn._inner_model.named_modules())
        layer = converted_layer_names[layer_name]
        return layer.dilation_masker.gamma  # type: ignore

    def _check_input_features(self, new_nn: PIT, input_features_dict: Dict[str, int]):
        """Check if the number of input features of each layer in a NAS-able model is as expected.

        input_features_dict is a dictionary containing: {layer_name, expected_input_features}
        """
        # TODO: avoid duplicate code from test_pit_convert.
        converted_layer_names = dict(new_nn._inner_model.named_modules())
        for name, exp in input_features_dict.items():
            layer = converted_layer_names[name]
            in_features = layer.input_features_calculator.features  # type: ignore
            self.assertEqual(in_features, exp,
                             f"Layer {name} has {in_features} input features, expected {exp}")





if __name__ == '__main__':
    unittest.main(verbosity=2)
