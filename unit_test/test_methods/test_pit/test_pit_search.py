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
from typing import Tuple, cast
import unittest
import math
import torch
import torch.nn as nn
from plinio.cost import params, ops
from plinio.methods import PIT
from plinio.methods.pit.nn import PITConv1d, PITConv2d
from unit_test.models import SimpleNN
from unit_test.models import TCResNet14
from unit_test.models import ToyAdd, ToyFlatten, ToyRegression, ToyAdd_2D
import torch.optim as optim


class TestPITSearch(unittest.TestCase):
    """Test search functionality in PIT"""

    def setUp(self):
        self.tc_resnet_config = {
            "input_channels": 6,
            "output_size": 12,
            "num_channels": [24, 36, 36, 48, 48, 72, 72],
            "kernel_size": 9,
            "dropout": 0.5,
            "grad_clip": -1,
            "use_bias": True,
            "use_dilation": False,
            "avg_pool": True,
        }

    def test_converted_output_simple(self):
        """Test that the output of a model converted to PIT is the same as the original nn.Module,
        before any training.
        """
        nn_ut = SimpleNN()
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        self._check_output_equal(nn_ut, pit_net, nn_ut.input_shape)

    def test_converted_output_advanced(self):
        """Test that the output of a model converted to PIT is the same as the original nn.Module,
        before any training.
        """
        input_shape = (6, 50)
        nn_ut = TCResNet14(self.tc_resnet_config)
        pit_net = PIT(nn_ut, input_shape=input_shape)
        self._check_output_equal(nn_ut, pit_net, input_shape)

    def test_regularization_loss_init(self):
        """Test the regularization loss computation on an initialized model"""
        # we use ToyAdd to make sure that mask sharing does not create errors on regloss
        # computation
        net = ToyAdd()

        # Check the number of params for the whole net
        input_shape = net.input_shape
        pit_net = PIT(net, input_shape=input_shape, cost={'params': params, 'ops': ops})
        # conv0 and conv1 have Cin=3, Cout=10, K=3
        # conv2 has Cin=10, Cout=20, K=5
        # the final FC has 140 input features and 2 output features
        exp_size_conv_01 = 3 * 10 * 3
        exp_size_conv_2 = 10 * 20 * 5
        exp_size_fc = 140 * 2
        exp_size_net = 2 * exp_size_conv_01 + exp_size_conv_2 + exp_size_fc
        self.assertEqual(pit_net.get_cost('params'), exp_size_net, "Wrong net size ")

        # Check the number of MACs for the whole net
        # conv2 has half the output length due to pooling
        exp_macs_net = 2 * (exp_size_conv_01 * input_shape[1]) + \
            (exp_size_conv_2 * (input_shape[1] // 2))
        # for FC, macs and size are equal
        exp_macs_net = exp_macs_net + exp_size_fc
        self.assertEqual(pit_net.get_cost('ops'), exp_macs_net, "Wrong net MACs")

    def test_regularization_loss_discrete_continuous_1d(self):
        """Test the regularization loss computation on a model with 1D convolutions with random
        masks, for both continuous and discrete cost estimates."""
        net = ToyAdd()

        # Check the number of params and ops for the whole net
        input_shape = net.input_shape
        pit_net = PIT(net, input_shape=input_shape,
                      cost={'params': params, 'ops': ops}, discrete_cost=True)
        # conv0 and conv1 have Cin=3, Cout=10, K=3
        # conv2 has Cin=10, Cout=20, K=5
        # the final FC has 140 input features and 2 output features
        exp_size_conv_01 = 3 * 10 * 3
        exp_size_conv_2 = 10 * 20 * 5
        exp_size_fc = 140 * 2
        exp_size_net = 2 * exp_size_conv_01 + exp_size_conv_2 + exp_size_fc
        # for the OPs, conv2 has half the output length due to pooling
        exp_ops_conv_01 = exp_size_conv_01 * input_shape[1]
        exp_ops_conv_2 = exp_size_conv_2 * (input_shape[1] // 2)
        exp_ops_net = 2 * exp_ops_conv_01 + exp_ops_conv_2 + exp_size_fc

        self.assertEqual(pit_net.get_cost('params'), exp_size_net, "Wrong initial net size")
        self.assertEqual(pit_net.get_cost('ops'), exp_ops_net, "Wrong initial net ops")

        # Now let us some of the output channels and reduce K to 2
        # We set continuous mask values, which should however be binarized again, since we are
        # using discrete cost evaluation
        conv1 = cast(PITConv1d, pit_net.seed.conv1)
        alpha_mask = torch.Tensor([0.7, 0.2, 0.6, 0.4, 0.9, 0.1, 0.3, 0.3, 0.95, 1.0])
        beta_mask = torch.Tensor([0.2, 0.6, 0.7])
        conv1.out_features_masker.alpha = nn.Parameter(alpha_mask)
        conv1.timestep_masker.beta = nn.Parameter(beta_mask)
        # the expected size with discrete sampling corresponds to having a number of channels
        # equal to the alpha_mask values > 0.5 (5, note that we kept the last channel, always
        # kept alive at 1.0), and a kernel size equal to the number of beta_mask values > 0.5 (2)
        exp_size_conv_1_dsc = 3 * 5 * 2
        # Moreover, since conv1 and conv0 share the channel mask, also the size of conv0 is
        # expected to change (only the channels)
        exp_size_conv_0_dsc = 3 * 5 * 3
        # lastly, conv_2 also changes because of the different number of input channels
        exp_size_conv_2_dsc = 5 * 20 * 5
        # So this is the new expected size
        exp_size_dsc = exp_size_conv_1_dsc + exp_size_conv_0_dsc + exp_size_conv_2_dsc + exp_size_fc
        # Update the OPs too
        exp_ops_conv_0_dsc = exp_size_conv_0_dsc * input_shape[1]
        exp_ops_conv_1_dsc = exp_size_conv_1_dsc * input_shape[1]
        exp_ops_conv_2_dsc = exp_size_conv_2_dsc * (input_shape[1] // 2)
        exp_ops_dsc = exp_ops_conv_1_dsc + exp_ops_conv_0_dsc + exp_ops_conv_2_dsc + exp_size_fc

        # try changing the cost specification on the fly
        pit_net.cost_specification = params
        self.assertEqual(pit_net.cost, exp_size_dsc, "Wrong discrete net size")
        pit_net.cost_specification = ops
        self.assertEqual(pit_net.cost, exp_ops_dsc, "Wrong discrete net ops")

        # with a continuous cost estimate, the cost will be different and depend on the mask values
        exp_size_conv_0_cnt = 3 * torch.sum(alpha_mask) * 3
        exp_size_conv_1_cnt = 3 * torch.sum(alpha_mask) * \
            torch.sum(torch.mul(
                torch.mul(conv1.timestep_masker.theta, conv1._beta_norm),
                torch.mul(conv1.dilation_masker.theta, conv1._gamma_norm)
            ))
        exp_size_conv_2_cnt = torch.sum(alpha_mask) * 20 * 5
        exp_ops_conv_0_cnt = exp_size_conv_0_cnt * input_shape[1]
        exp_ops_conv_1_cnt = exp_size_conv_1_cnt * input_shape[1]
        exp_ops_conv_2_cnt = exp_size_conv_2_cnt * (input_shape[1] // 2)
        exp_size_cnt = exp_size_conv_1_cnt + exp_size_conv_0_cnt + exp_size_conv_2_cnt + exp_size_fc
        exp_ops_cnt = exp_ops_conv_1_cnt + exp_ops_conv_0_cnt + exp_ops_conv_2_cnt + exp_size_fc

        # switch to continuous evaluation
        pit_net.discrete_cost = False
        pit_net.cost_specification = {'params': params, 'ops': ops}
        self.assertEqual(pit_net.get_cost('params'), exp_size_cnt, "Wrong continuous net size")
        self.assertEqual(pit_net.get_cost('ops'), exp_ops_cnt, "Wrong continuous net ops")

        # Lastly, force a dilation=2 in conv2, this means that the actual kernel
        # will become K=3 to maintain same receptive-field
        gamma_mask = torch.Tensor([0.1, 0.9, 0.85])
        conv2 = cast(PITConv1d, pit_net.seed.conv2)
        conv2.dilation_masker.gamma = nn.Parameter(gamma_mask)
        exp_size_conv_2_dsc = 5 * 20 * 3
        exp_size_dsc = exp_size_conv_1_dsc + exp_size_conv_0_dsc + exp_size_conv_2_dsc + exp_size_fc
        exp_ops_conv_2_dsc = exp_size_conv_2_dsc * (input_shape[1] // 2)
        exp_ops_dsc = exp_ops_conv_1_dsc + exp_ops_conv_0_dsc + exp_ops_conv_2_dsc + exp_size_fc

        # switch back to discrete evaluation, and this time we do ops before params, just in case
        pit_net.discrete_cost = True
        self.assertEqual(pit_net.get_cost('ops'), exp_ops_dsc, "Wrong discrete net ops")
        self.assertEqual(pit_net.get_cost('params'), exp_size_dsc, "Wrong discrete net size")

        # lastly, try these conditions in the continuous case
        exp_size_conv_2_cnt = torch.sum(alpha_mask) * 20 * \
            torch.sum(torch.mul(
                torch.mul(conv2.timestep_masker.theta, conv2._beta_norm),
                torch.mul(conv2.dilation_masker.theta, conv2._gamma_norm)
            ))
        exp_ops_conv_2_cnt = exp_size_conv_2_cnt * (input_shape[1] // 2)
        exp_size_cnt = exp_size_conv_1_cnt + exp_size_conv_0_cnt + exp_size_conv_2_cnt + exp_size_fc
        exp_ops_cnt = exp_ops_conv_1_cnt + exp_ops_conv_0_cnt + exp_ops_conv_2_cnt + exp_size_fc

        pit_net.discrete_cost = False
        self.assertEqual(pit_net.get_cost('params'), exp_size_cnt, "Wrong continuous net size")
        pit_net.cost_specification = ops
        # voluntary repetition
        pit_net.cost_specification = ops
        self.assertEqual(pit_net.cost, exp_ops_cnt, "Wrong continuous net ops")

    def test_regularization_loss_discrete_continuous_2d(self):
        """Test the regularization loss computation on a model with 1D convolutions with random
        masks, for both continuous and discrete cost estimates."""
        net = ToyAdd_2D()

        # Check the number of params and ops for the whole net
        input_shape = net.input_shape
        pit_net = PIT(net, input_shape=input_shape,
                      cost={'params': params, 'ops': ops}, discrete_cost=True)
        # conv0 and conv1 have Cin=3, Cout=10, K=(3,3)
        # conv2 has Cin=10, Cout=20, K=(5,5)
        # the final FC has 980 input features and 2 output features
        exp_size_conv_01 = 3 * 10 * 3 * 3
        exp_size_conv_2 = 10 * 20 * 5 * 5
        exp_size_fc = 980 * 2
        exp_size_net = 2 * exp_size_conv_01 + exp_size_conv_2 + exp_size_fc
        # for the OPs, conv2 has half the feature size due to pooling
        exp_ops_conv_01 = exp_size_conv_01 * input_shape[1] * input_shape[2]
        exp_ops_conv_2 = exp_size_conv_2 * (input_shape[1] // 2) * (input_shape[1] // 2)
        exp_ops_net = 2 * exp_ops_conv_01 + exp_ops_conv_2 + exp_size_fc

        self.assertEqual(pit_net.get_cost('params'), exp_size_net, "Wrong initial net size")
        self.assertEqual(pit_net.get_cost('ops'), exp_ops_net, "Wrong initial net ops")

        input_shape = net.input_shape
        # Now let us some of the output channels
        # We set continuous mask values, which should however be binarized again, since we are
        # using discrete cost evaluation
        conv1 = cast(PITConv2d, pit_net.seed.conv1)
        alpha_mask = torch.Tensor([0.7, 0.2, 0.6, 0.4, 0.9, 0.1, 0.3, 0.3, 0.95, 1.0])
        conv1.out_features_masker.alpha = nn.Parameter(alpha_mask)
        # the expected size with discrete sampling corresponds to having a number of channels
        # equal to the alpha_mask values > 0.5 (5, note that we kept the last channel, always
        # kept alive at 1.0). This affects both conv1 and conv0 due to their shared mask
        exp_size_conv_01_dsc = 3 * 5 * 3 * 3
        # lastly, conv_2 also changes because of the different number of input channels
        exp_size_conv_2_dsc = 5 * 20 * 5 * 5
        # So this is the new expected size
        exp_size_dsc = 2 * exp_size_conv_01_dsc + exp_size_conv_2_dsc + exp_size_fc
        # Update the OPs too
        exp_ops_conv_01_dsc = exp_size_conv_01_dsc * input_shape[1] * input_shape[2]
        exp_ops_conv_2_dsc = exp_size_conv_2_dsc * (input_shape[1] // 2) * (input_shape[1] // 2)
        exp_ops_dsc = 2 * exp_ops_conv_01_dsc + exp_ops_conv_2_dsc + exp_size_fc

        # change cost specification on the fly
        pit_net.cost_specification = params
        self.assertEqual(pit_net.cost, exp_size_dsc, "Wrong discrete net size")
        pit_net.cost_specification = ops
        self.assertEqual(pit_net.cost, exp_ops_dsc, "Wrong discrete net ops")

        # with a continuous cost estimate, the cost will be different and depend on the mask values
        exp_size_conv_01_cnt = 3 * torch.sum(alpha_mask) * 3 * 3
        exp_size_conv_2_cnt = torch.sum(alpha_mask) * 20 * 5 * 5
        exp_ops_conv_01_cnt = exp_size_conv_01_cnt * input_shape[1] * input_shape[1]
        exp_ops_conv_2_cnt = exp_size_conv_2_cnt * (input_shape[1] // 2) * (input_shape[1] // 2)
        exp_size_cnt = 2 * exp_size_conv_01_cnt + exp_size_conv_2_cnt + exp_size_fc
        exp_ops_cnt = 2 * exp_ops_conv_01_cnt + exp_ops_conv_2_cnt + exp_size_fc

        # switch to continuous evaluation
        pit_net.discrete_cost = False
        pit_net.cost_specification = params
        self.assertEqual(pit_net.cost, exp_size_cnt, "Wrong continuous net size")
        pit_net.cost_specification = ops
        self.assertEqual(pit_net.cost, exp_ops_cnt, "Wrong continuous net ops")

        # Lastly, mask some channels also in conv2, which also affects the FC layer input
        # will become K=3 to maintain same receptive-field
        alpha_mask2 = torch.Tensor([0.1, 0.25, 0.6, 0.8] + [1] * 16)
        conv2 = cast(PITConv2d, pit_net.seed.conv2)
        conv2.out_features_masker.alpha = nn.Parameter(alpha_mask2)
        exp_size_conv_2_dsc = 5 * 18 * 5 * 5
        # each feature map after conv2 is 7*7, flattened to 49 features
        exp_size_fc_dsc = 882 * 2
        exp_size_dsc = 2 * exp_size_conv_01_dsc + exp_size_conv_2_dsc + exp_size_fc_dsc
        exp_ops_conv_2_dsc = exp_size_conv_2_dsc * (input_shape[1] // 2) * (input_shape[1] // 2)
        exp_ops_dsc = 2 * exp_ops_conv_01_dsc + exp_ops_conv_2_dsc + exp_size_fc_dsc

        # switch back to discrete evaluation, and this time we do ops before params, just in case
        pit_net.discrete_cost = True
        self.assertEqual(pit_net.cost, exp_ops_dsc, "Wrong discrete net ops")
        pit_net.cost_specification = {'params': params, 'ops': ops}
        self.assertEqual(pit_net.get_cost('params'), exp_size_dsc, "Wrong discrete net size")

        # lastly, try these conditions in the continuous case
        exp_size_conv_2_cnt = torch.sum(alpha_mask) * torch.sum(alpha_mask2) * 5 * 5
        # each feature map after conv2 is 7*7, flattened to 49 features
        exp_size_fc_cnt = 49 * torch.sum(alpha_mask2) * 2
        exp_ops_conv_2_cnt = exp_size_conv_2_cnt * (input_shape[1] // 2) * (input_shape[1] // 2)
        exp_size_cnt = 2 * exp_size_conv_01_cnt + exp_size_conv_2_cnt + exp_size_fc_cnt
        exp_ops_cnt = 2 * exp_ops_conv_01_cnt + exp_ops_conv_2_cnt + exp_size_fc_cnt

        pit_net.discrete_cost = False
        self.assertEqual(pit_net.get_cost('params'), exp_size_cnt, "Wrong continuous net size")
        pit_net.cost_specification = ops
        # voluntary repetition
        pit_net.cost_specification = ops
        self.assertEqual(pit_net.cost, exp_ops_cnt, "Wrong continuous net ops")

    def test_regularization_loss_descent(self):
        """Test that the regularization loss decreases after a few forward and backward steps,
        without other loss components"""
        # we use ToyAdd to make sure that mask sharing does not create errors
        nn_ut = ToyAdd()
        batch_size = 8
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        optimizer = optim.Adam(pit_net.parameters())
        n_steps = 10
        prev_cost = pit_net.cost
        print("Initial Reg. Loss:", prev_cost.item())
        for _ in range(n_steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            cost = pit_net.cost
            print("Reg. Loss:", cost.item())
            self.assertLessEqual(cost.item(), prev_cost.item(), "The loss value is not descending")
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            prev_cost = cost

    def test_regularization_loss_weights(self):
        """Check that the weights remain equal using only the regularization loss"""
        # we use ToyAdd to verify that mask sharing does not create problems
        nn_ut = ToyAdd()
        batch_size = 8
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        optimizer = optim.Adam(pit_net.parameters())
        n_steps = 10
        conv0 = cast(PITConv1d, pit_net.seed.conv0)
        conv1 = cast(PITConv1d, pit_net.seed.conv1)
        conv2 = cast(PITConv1d, pit_net.seed.conv2)
        init_conv0_weights = conv0.weight.clone().detach()
        init_conv1_weights = conv1.weight.clone().detach()
        init_conv2_weights = conv2.weight.clone().detach()
        for _ in range(n_steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            loss = pit_net.get_cost()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            conv0_weights = conv0.weight.clone().detach()
            conv1_weights = conv1.weight.clone().detach()
            conv2_weights = conv2.weight.clone().detach()
            self.assertTrue(torch.all(init_conv0_weights == conv0_weights),
                            "Conv0 weights changing")
            self.assertTrue(torch.all(init_conv1_weights == conv1_weights),
                            "Conv1 weights changing")
            self.assertTrue(torch.all(init_conv2_weights == conv2_weights),
                            "Conv2 weights changing")

    def test_regularization_loss_theta_descent(self):
        """Check that theta masks eventually go to 0 (except kept-alive values) using only the
        regularization loss"""
        # as usual, we use ToyAdd to verify mask sharing is not problematic
        nn_ut = ToyAdd()
        batch_size = 1
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        # we must use SGD to be sure we only consider gradients
        optimizer = optim.SGD(pit_net.parameters(), lr=0.001)
        pit_net.eval()
        # required number of steps varies with lr, but 6000 gives a sufficient margin
        max_steps = 6000
        conv0 = cast(PITConv1d, pit_net.seed.conv0)
        conv1 = cast(PITConv1d, pit_net.seed.conv1)
        conv2 = cast(PITConv1d, pit_net.seed.conv2)
        convs = [conv0, conv1, conv2]
        max_dils = [2, 2, 4]
        ths = [layer.binarization_threshold for layer in convs]
        for i in range(max_steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            loss = pit_net.get_cost()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # check that the final masks are all smaller than the binarization threshold
        # (except the kept-alive elements)
        theta_masks = []
        for layer in convs:
            theta_masks.append({
                'cout': layer.out_features_masker.theta.clone().detach(),
                'rf': layer.timestep_masker.theta.clone().detach(),
                'dil': layer.dilation_masker.theta.clone().detach(),
            })

        for mask_set, th, max_dil in zip(theta_masks, ths, max_dils):
            for type in ('cout', 'rf'):
                self.assertTrue(torch.all(mask_set[type][:-1] <= th),
                                f"Mask value for {type} not smaller than {th}")
            # for dilation, the check is a bit more complex, cause we need to esclude one every
            # max_dil elements
            for i in range(len(mask_set['dil'])):
                if i % max_dil != 0:
                    self.assertLess(mask_set['dil'][i], th,
                                    f"Mask value for dil not smaller than {th}")
        for layer, dil in zip(convs, max_dils):
            self.assertEqual(layer.out_features_opt, 1, "Wrong channels_opt")
            self.assertEqual(layer.kernel_size_opt[0], 1, "Wrong kernel_size_opt")
            self.assertEqual(layer.dilation_opt[0], dil, "Wrong dilation_opt")

    def test_no_train_features(self):
        """Test that alpha masks remain fixed (and others change) with train_features=False"""
        nn_ut = ToyAdd()
        batch_size = 1
        steps = 6000
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape, train_features=False)
        optimizer = optim.Adam(pit_net.parameters())
        conv0 = cast(PITConv1d, pit_net.seed.conv0)
        conv1 = cast(PITConv1d, pit_net.seed.conv1)
        conv2 = cast(PITConv1d, pit_net.seed.conv2)
        convs = [conv0, conv1, conv2]
        initial_masks = [layer.out_features_masker.theta.clone().detach() for layer in convs]
        max_dils = [2, 2, 4]
        ths = [layer.binarization_threshold for layer in convs]
        for i in range(steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            loss = pit_net.get_cost()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for layer, max_dil, th, initial in zip(convs, max_dils, ths, initial_masks):
            # theta alpha should be fixed at 1.
            self.assertTrue(torch.all(layer.out_features_masker.theta == initial),
                            "Channels mask changed unexpectedly")
            # theta beta and theta gamma should be below threshold (except kept-alive elements)
            self.assertTrue(torch.all(layer.timestep_masker.theta[:-1] < th),
                            "RF mask above threshold unexpectedly")
            # for dilation, the check is a bit more complex, cause we need to esclude one every
            # max_dil elements
            theta_gamma = layer.dilation_masker.theta
            for i in range(len(theta_gamma)):
                if i % max_dil != 0:
                    self.assertLess(float(theta_gamma[i]), th,
                                    "Dilation mask above threshold unexpectedly")
            self.assertEqual(layer.out_features_opt, layer.out_channels, "Wrong features_opt")
            self.assertEqual(layer.out_features_eff, layer.out_channels, "Wrong features_eff")
            self.assertEqual(layer.kernel_size_opt[0], 1, "Wrong kernel_size_opt")
            self.assertEqual(layer.dilation_opt[0], max_dil, "Wrong dilation_opt")

    def test_no_train_rf(self):
        """Test that beta masks remain fixed (and others change) with train_rf=False"""
        nn_ut = ToyAdd()
        batch_size = 1
        steps = 6000
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape, train_rf=False)
        optimizer = optim.Adam(pit_net.parameters())
        conv0 = cast(PITConv1d, pit_net.seed.conv0)
        conv1 = cast(PITConv1d, pit_net.seed.conv1)
        conv2 = cast(PITConv1d, pit_net.seed.conv2)
        convs = [conv0, conv1, conv2]
        initial_masks = [layer.timestep_masker.theta.clone().detach() for layer in convs]
        max_dils = [2, 2, 4]
        ths = [layer.binarization_threshold for layer in convs]
        for i in range(steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            loss = pit_net.cost
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for layer, max_dil, th, initial in zip(convs, max_dils, ths, initial_masks):
            # theta beta should be fixed at init value
            self.assertTrue(torch.all(layer.timestep_masker.theta == initial),
                            "RF mask changed unexpectedly")
            # theta alpha and theta gamma should be below threshold (except kept-alive elements)
            self.assertTrue(torch.all(layer.out_features_masker.theta[:-1] < th),
                            "channels mask above threshold unexpectedly")
            # for dilation, the check is a bit more complex, cause we need to esclude one every
            # max_dil elements
            theta_gamma = layer.dilation_masker.theta
            for i in range(len(theta_gamma)):
                if i % max_dil != 0:
                    self.assertLess(float(theta_gamma[i]), th,
                                    "Dilation mask above threshold unexpectedly")
            exp_ks_opt = math.ceil(layer.rf / max_dil)
            self.assertEqual(layer.out_features_opt, 1, "Wrong features")
            self.assertEqual(layer.kernel_size_opt[0], exp_ks_opt, "Wrong kernel_size_opt")
            self.assertEqual(layer.dilation_opt[0], max_dil, "Wrong dilation_opt")

    def test_no_train_dilation(self):
        """Test that gamma masks remain fixed at 1 (and others change) with train_rf=False"""
        nn_ut = ToyAdd()
        batch_size = 1
        steps = 6000
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape, train_dilation=False)
        optimizer = optim.Adam(pit_net.parameters())
        conv0 = cast(PITConv1d, pit_net.seed.conv0)
        conv1 = cast(PITConv1d, pit_net.seed.conv1)
        conv2 = cast(PITConv1d, pit_net.seed.conv2)
        convs = [conv0, conv1, conv2]
        initial_masks = [layer.dilation_masker.theta.clone().detach() for layer in convs]
        max_dils = [2, 2, 4]
        ths = [layer.binarization_threshold for layer in convs]
        for i in range(steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            loss = pit_net.get_cost()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for layer, max_dil, th, initial in zip(convs, max_dils, ths, initial_masks):
            # theta gamma should be fixed at init value
            self.assertTrue(torch.all(layer.dilation_masker.theta == initial),
                            "Dilation mask changed unexpectedly")
            self.assertEqual(layer.dilation_opt[0], 1, "Wrong dilation opt")
            # theta alpha and theta beta should be below threshold (except kept-alive elements)
            self.assertTrue(torch.all(layer.out_features_masker.theta[:-1] < th),
                            "Channels mask above threshold unexpectedly")
            self.assertTrue(torch.all(layer.timestep_masker.theta[:-1] < th),
                            "RF mask above threshold unexpectedly")
            self.assertEqual(layer.out_features_opt, 1, "Wrong out_features_opt")
            self.assertEqual(layer.kernel_size_opt[0], 1, "Wrong kernel_size_opt")
            self.assertEqual(layer.dilation_opt[0], 1, "Wrong dilation_opt")

    def test_combined_loss(self):
        """Check that the network weights are changing with a combined loss"""
        nn_ut = ToyFlatten()
        batch_size = 5
        lambda_param = 0.0005
        n_steps = 50
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(pit_net.parameters())
        conv1 = cast(PITConv1d, pit_net.seed.conv1)
        prev_conv1_weight = torch.zeros_like(conv1.weight)
        for _ in range(n_steps):
            input = torch.stack([torch.rand(nn_ut.input_shape)] * batch_size)
            target = torch.randint(0, 2, (batch_size,))
            output = pit_net(input)
            task_loss = criterion(output, target)
            nas_loss = lambda_param * pit_net.get_cost()
            total_loss = task_loss + nas_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            conv1_weight = conv1.weight.clone().detach()
            print("Cout=0, Cin=0 weights:", conv1_weight[0, 0])
            self.assertFalse(torch.all(torch.isclose(prev_conv1_weight, conv1_weight)))
            prev_conv1_weight = conv1_weight

    def test_combined_loss_const_labels(self):
        """Check that a trivial training with constant labels converges when using a combined
        loss"""
        nn_ut = SimpleNN()
        batch_size = 32
        lambda_param = 0.0005
        n_steps = 50
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(pit_net.parameters())
        for _ in range(n_steps):
            input = torch.stack([torch.rand(nn_ut.input_shape)] * batch_size)
            target = torch.ones((batch_size,), dtype=torch.long)
            output = pit_net(input)
            task_loss = criterion(output, target)
            nas_loss = lambda_param * pit_net.get_cost()
            total_loss = task_loss + nas_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        output = pit_net(torch.stack([torch.rand(nn_ut.input_shape)] * batch_size))
        self.assertTrue(torch.sum(torch.argmax(output, dim=-1)) == batch_size,
                        "The network should output only 1s with all labels equal to 1")

    def test_combined_loss_regression(self):
        """Check the output of the combined loss with a trivial regression problem"""
        nn_ut = ToyRegression()
        lambda_param = 0.5  # lambda very large on purpose
        batch_size = 32
        n_steps = 1000
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(pit_net.parameters())
        running_err = 0
        for i in range(n_steps):
            # generate 10 random numbers
            input = torch.rand((batch_size,) + nn_ut.input_shape)
            # the desired output is their sum
            target = torch.sum(input, dim=2).reshape((batch_size, -1))
            output = pit_net(input)
            task_loss = criterion(output, target)
            total_loss = task_loss + lambda_param*pit_net.cost
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            err = float(torch.abs(target - output).detach().numpy().mean())
            running_err = ((running_err * i) + err) / (i + 1)
        self.assertLess(running_err, 1, "Error not decreasing")
        conv0 = cast(PITConv1d, pit_net.seed.conv0)
        self.assertLess(conv0.out_features_opt, conv0.out_channels, "Channels not decreasing")

    def _check_output_equal(self, nn: nn.Module, pit_nn: PIT, input_shape: Tuple[int, ...],
                            iterations=10):
        """Verify that a model and a PIT model produce the same output given the same input"""
        # TODO: avoid duplicated definition with PIT masking
        for i in range(iterations):
            # add batch size in front
            x = torch.rand((32,) + input_shape)
            nn.eval()
            pit_nn.eval()
            y = nn(x)
            pit_y = pit_nn(x)
            self.assertTrue(torch.allclose(y, pit_y, atol=1e-7), "Wrong output of PIT model")


if __name__ == '__main__':
    unittest.main(verbosity=2)
