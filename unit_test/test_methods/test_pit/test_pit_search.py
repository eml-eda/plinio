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
from flexnas.methods import PIT
from flexnas.methods.pit import PITConv1d
from unit_test.models import SimpleNN
from unit_test.models import TCResNet14
from unit_test.models import ToyAdd, ToyFlatten, ToyRegression
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

        input_shape = net.input_shape
        pit_net = PIT(net, input_shape=input_shape, regularizer='macs')
        # Check the number of weights for a single conv layer
        # conv1 has Cin=3, Cout=10, K=3
        exp_size_conv1 = 3 * 10 * 3
        conv1 = cast(PITConv1d, pit_net.inner_model.conv1)
        self.assertEqual(conv1.get_size(), exp_size_conv1, "Wrong layer size")
        # Check the number of MACs for a single conv layer
        # all convs have "same" padding
        exp_macs_conv1 = exp_size_conv1 * input_shape[1]
        self.assertEqual(conv1.get_macs(), exp_macs_conv1, "Wrong layer MACs")

        # Check the number of weights for the whole net
        # conv0 is identical to conv1, and conv2 has Cin=10, Cout=20, K=5
        # the final FC has 140 input features and 2 output features
        exp_size_conv2 = (10 * 20 * 5)
        exp_size_fc = (140 * 2)
        exp_size_net = 2 * exp_size_conv1 + exp_size_conv2 + exp_size_fc
        self.assertEqual(pit_net.get_size(), exp_size_net, "Wrong net size ")
        # Check the number of MACs for the whole net
        # conv2 has half the output length due to pooling
        exp_macs_net = 2 * exp_macs_conv1 + (exp_size_conv2 * (input_shape[1] // 2))
        # for FC, macs and size are equal
        exp_macs_net = exp_macs_net + exp_size_fc
        self.assertEqual(pit_net.get_macs(), exp_macs_net, "Wrong net MACs")

        # Check that get_regularization_loss and get_macs give the same result
        # since we specified the macs regularizer
        self.assertEqual(pit_net.get_regularization_loss(), pit_net.get_macs())

        # change the regularizer and check again
        pit_net.regularizer = 'size'
        self.assertEqual(pit_net.get_regularization_loss(), pit_net.get_size())

    def test_regularization_loss_descent(self):
        """Test that the regularization loss decreases after a few forward and backward steps,
        without other loss components"""
        # we use ToyAdd to make sure that mask sharing does not create errors
        nn_ut = ToyAdd()
        batch_size = 8
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        optimizer = optim.Adam(pit_net.parameters())
        n_steps = 10
        prev_loss = pit_net.get_regularization_loss()
        print("Initial Reg. Loss:", prev_loss.item())
        for i in range(n_steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            loss = pit_net.get_regularization_loss()
            print("Reg. Loss:", loss.item())
            self.assertLessEqual(loss.item(), prev_loss.item(), "The loss value is not descending")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prev_loss = loss

    def test_regularization_loss_weights(self):
        """Check that the weights remain equal using only the regularization loss"""
        # we use ToyAdd to verify that mask sharing does not create problems
        nn_ut = ToyAdd()
        batch_size = 8
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        optimizer = optim.Adam(pit_net.parameters())
        n_steps = 10
        conv0 = cast(PITConv1d, pit_net.inner_model.conv0)
        conv1 = cast(PITConv1d, pit_net.inner_model.conv1)
        conv2 = cast(PITConv1d, pit_net.inner_model.conv2)
        init_conv0_weights = conv0.weight.clone().detach()
        init_conv1_weights = conv1.weight.clone().detach()
        init_conv2_weights = conv2.weight.clone().detach()
        for i in range(n_steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            loss = pit_net.get_regularization_loss()
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

    def test_regularization_loss_theta_descent_progressive(self):
        """Check that theta masks always decrease using only the regularization loss"""
        # TODO: understand why this is not verified
        # as usual, we use ToyAdd to verify mask sharing is not problematic
        nn_ut = ToyAdd()
        batch_size = 8
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        # we must use SGD to be sure we only consider gradients
        optimizer = optim.SGD(pit_net.parameters(), lr=0.001)
        pit_net.eval()
        max_steps = 1000
        conv0 = cast(PITConv1d, pit_net.inner_model.conv0)
        conv1 = cast(PITConv1d, pit_net.inner_model.conv1)
        conv2 = cast(PITConv1d, pit_net.inner_model.conv2)
        convs = [conv0, conv1, conv2]
        prev_theta_masks = []
        for layer in convs:
            prev_theta_masks.append({
                'cout': layer.out_features_masker().clone().detach(),
                'rf': layer.timestep_masker().clone().detach(),
                'dil': layer.dilation_masker().clone().detach(),
            })
        print()
        for i in range(max_steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            loss = pit_net.get_regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            theta_masks = []
            for layer in convs:
                theta_masks.append({
                    'cout': layer.out_features_masker().clone().detach(),
                    'rf': layer.timestep_masker().clone().detach(),
                    'dil': layer.dilation_masker().clone().detach(),
                })
            for mask_set1, mask_set2 in zip(theta_masks, prev_theta_masks):
                for type in ('cout', 'rf', 'dil'):
                    # check that masks are always decreasing or equal
                    self.assertTrue(torch.all(mask_set1[type] <= mask_set2[type]),
                                    f"Mask value for {type} not decreasing")
            prev_theta_masks = theta_masks

    def test_regularization_loss_theta_descent_final(self):
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
        conv0 = cast(PITConv1d, pit_net.inner_model.conv0)
        conv1 = cast(PITConv1d, pit_net.inner_model.conv1)
        conv2 = cast(PITConv1d, pit_net.inner_model.conv2)
        convs = [conv0, conv1, conv2]
        max_dils = [2, 2, 4]
        ths = [layer._binarization_threshold for layer in convs]
        for i in range(max_steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            loss = pit_net.get_regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # check that the final masks are all smaller than the binarization threshold
        # (except the kept-alive elements)
        theta_masks = []
        for layer in convs:
            theta_masks.append({
                'cout': layer.out_features_masker().clone().detach(),
                'rf': layer.timestep_masker().clone().detach(),
                'dil': layer.dilation_masker().clone().detach(),
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
        conv0 = cast(PITConv1d, pit_net.inner_model.conv0)
        conv1 = cast(PITConv1d, pit_net.inner_model.conv1)
        conv2 = cast(PITConv1d, pit_net.inner_model.conv2)
        convs = [conv0, conv1, conv2]
        initial_masks = [layer.out_features_masker().clone().detach() for layer in convs]
        max_dils = [2, 2, 4]
        ths = [layer._binarization_threshold for layer in convs]
        for i in range(steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            loss = pit_net.get_regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for layer, max_dil, th, initial in zip(convs, max_dils, ths, initial_masks):
            # theta alpha should be fixed at 1.
            self.assertTrue(torch.all(layer.out_features_masker() == initial),
                            "Channels mask changed unexpectedly")
            # theta beta and theta gamma should be below threshold (except kept-alive elements)
            self.assertTrue(torch.all(layer.timestep_masker()[:-1] < th),
                            "RF mask above threshold unexpectedly")
            # for dilation, the check is a bit more complex, cause we need to esclude one every
            # max_dil elements
            theta_gamma = layer.dilation_masker()
            for i in range(len(theta_gamma)):
                if i % max_dil != 0:
                    self.assertLess(theta_gamma[i], th,
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
        conv0 = cast(PITConv1d, pit_net.inner_model.conv0)
        conv1 = cast(PITConv1d, pit_net.inner_model.conv1)
        conv2 = cast(PITConv1d, pit_net.inner_model.conv2)
        convs = [conv0, conv1, conv2]
        initial_masks = [layer.timestep_masker().clone().detach() for layer in convs]
        max_dils = [2, 2, 4]
        ths = [layer._binarization_threshold for layer in convs]
        for i in range(steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            loss = pit_net.get_regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for layer, max_dil, th, initial in zip(convs, max_dils, ths, initial_masks):
            # theta beta should be fixed at init value
            self.assertTrue(torch.all(layer.timestep_masker() == initial),
                            "RF mask changed unexpectedly")
            # theta alpha and theta gamma should be below threshold (except kept-alive elements)
            self.assertTrue(torch.all(layer.out_features_masker()[:-1] < th),
                            "channels mask above threshold unexpectedly")
            # for dilation, the check is a bit more complex, cause we need to esclude one every
            # max_dil elements
            theta_gamma = layer.dilation_masker()
            for i in range(len(theta_gamma)):
                if i % max_dil != 0:
                    self.assertLess(theta_gamma[i], th,
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
        conv0 = cast(PITConv1d, pit_net.inner_model.conv0)
        conv1 = cast(PITConv1d, pit_net.inner_model.conv1)
        conv2 = cast(PITConv1d, pit_net.inner_model.conv2)
        convs = [conv0, conv1, conv2]
        initial_masks = [layer.dilation_masker().clone().detach() for layer in convs]
        max_dils = [2, 2, 4]
        ths = [layer._binarization_threshold for layer in convs]
        for i in range(steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            pit_net(x)
            loss = pit_net.get_regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for layer, max_dil, th, initial in zip(convs, max_dils, ths, initial_masks):
            # theta gamma should be fixed at init value
            self.assertTrue(torch.all(layer.dilation_masker() == initial),
                            "Dilation mask changed unexpectedly")
            self.assertEqual(layer.dilation_opt[0], 1, "Wrong dilation opt")
            # theta alpha and theta beta should be below threshold (except kept-alive elements)
            self.assertTrue(torch.all(layer.out_features_masker()[:-1] < th),
                            "Channels mask above threshold unexpectedly")
            self.assertTrue(torch.all(layer.timestep_masker()[:-1] < th),
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
        conv1 = cast(PITConv1d, pit_net.inner_model.conv1)
        prev_conv1_weight = torch.zeros_like(conv1.weight)
        for i in range(n_steps):
            input = torch.stack([torch.rand(nn_ut.input_shape)] * batch_size)
            target = torch.randint(0, 2, (batch_size,))
            output = pit_net(input)
            task_loss = criterion(output, target)
            nas_loss = lambda_param * pit_net.get_regularization_loss()
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
        for i in range(n_steps):
            input = torch.stack([torch.rand(nn_ut.input_shape)] * batch_size)
            target = torch.ones((batch_size,), dtype=torch.long)
            output = pit_net(input)
            task_loss = criterion(output, target)
            nas_loss = lambda_param * pit_net.get_regularization_loss()
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
            nas_loss = lambda_param * pit_net.get_regularization_loss()
            total_loss = task_loss + nas_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            err = float(torch.abs(target - output).detach().numpy().mean())
            running_err = ((running_err * i) + err) / (i + 1)
        self.assertLess(running_err, 1, "Error not decreasing")
        conv0 = cast(PITConv1d, pit_net.inner_model.conv0)
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
            self.assertTrue(torch.all(torch.eq(y, pit_y)), "Wrong output of PIT model")


if __name__ == '__main__':
    unittest.main(verbosity=2)
