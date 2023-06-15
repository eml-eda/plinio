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
# * Author: Matteo Risso <matteo.risso@polito.it>                              *
# *----------------------------------------------------------------------------*

from typing import cast
import unittest
import torch
from plinio.methods import MixPrec
from plinio.methods.mixprec.nn import MixPrec_Conv2d, MixPrecType
from unit_test.models import ToyAdd_2D, SimpleNN2D, ToyRegression_2D


class TestMixPrecSearch(unittest.TestCase):
    """Test search functionality in MixPrec"""

    def test_regularization_loss_init(self):
        """Test the regularization loss computation on an initialized model"""
        # we use ToyAdd_2D to make sure that mask sharing does not create errors on regloss
        # computation
        net = ToyAdd_2D()
        a_prec = (2, 4, 8)
        w_prec = (2, 4, 8)

        input_shape = net.input_shape
        mixprec_net = MixPrec(net,
                              input_shape=input_shape,
                              regularizer='macs',
                              activation_precisions=a_prec,
                              weight_precisions=w_prec)
        # Check the number of weights for a single conv layer
        # conv1 has Cin=3, Cout=10, K=(3, 3)
        eff_w_prec = sum(list(w_prec)) / len(w_prec)
        exp_size_conv1 = 3 * 10 * 3 * 3 * eff_w_prec
        conv1 = cast(MixPrec_Conv2d, mixprec_net.seed.conv1)
        self.assertAlmostEqual(float(conv1.get_size()), exp_size_conv1, 0, "Wrong layer size")
        # Check the number of MACs for a single conv layer
        # all convs have "same" padding
        eff_a_prec = sum(list(a_prec)) / len(a_prec)
        exp_macs_conv1 = exp_size_conv1 * input_shape[1] * input_shape[2] * eff_a_prec
        self.assertAlmostEqual(float(conv1.get_macs()), exp_macs_conv1, 0, "Wrong layer MACs")

        # Check the number of weights for the whole net
        # conv0 is identical to conv1, and conv2 has Cin=10, Cout=20, K=(5, 5)
        # the final FC has 980 input features and 2 output features
        exp_size_conv2 = (10 * 20 * 5 * 5) * eff_w_prec
        exp_size_fc = (980 * 2) * eff_w_prec
        exp_size_net = 2 * exp_size_conv1 + exp_size_conv2 + exp_size_fc
        self.assertAlmostEqual(float(mixprec_net.get_size()), exp_size_net, 0, "Wrong net size ")
        # Check the number of MACs for the whole net
        # conv2 has half the output length due to pooling
        exp_macs_net = 2 * exp_macs_conv1 + \
            (exp_size_conv2 * eff_a_prec * ((input_shape[1] // 2) * (input_shape[2] // 2)))
        # for FC, macs and size are equal
        exp_macs_net = exp_macs_net + exp_size_fc * eff_a_prec
        self.assertAlmostEqual(float(mixprec_net.get_macs()), exp_macs_net, None, "Wrong net MACs",
                               delta=1)

        # Check that get_regularization_loss and get_macs give the same result
        # since we specified the macs regularizer
        self.assertEqual(mixprec_net.get_regularization_loss(), mixprec_net.get_macs())

        # change the regularizer and check again
        mixprec_net.regularizer = 'size'
        self.assertEqual(mixprec_net.get_regularization_loss(), mixprec_net.get_size())

    def test_regularization_loss_descent_layer(self):
        """Test that the regularization loss decreases after a few forward and backward steps,
        without other loss components with PER_LAYER weight mixed-precision (default)"""
        # we use ToyAdd_2D to make sure that mask sharing does not create errors
        nn_ut = ToyAdd_2D()
        batch_size = 8
        mixprec_net = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        optimizer = torch.optim.Adam(mixprec_net.parameters(), lr=1e-2)
        n_steps = 10
        with torch.no_grad():
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            mixprec_net(x)
        prev_loss = mixprec_net.get_regularization_loss()
        print("Initial Reg. Loss:", prev_loss.item())
        for i in range(n_steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            mixprec_net(x)
            loss = mixprec_net.get_regularization_loss()
            print("Reg. Loss:", loss.item())
            self.assertLessEqual(loss.item(), prev_loss.item(), "The loss value is not descending")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prev_loss = loss

    def test_regularization_loss_descent_channel(self):
        """Test that the regularization loss decreases after a few forward and backward steps,
        without other loss components with PER_CHANNEL weight mixed-precision"""
        # we use ToyAdd_2D to make sure that mask sharing does not create errors
        nn_ut = ToyAdd_2D()
        batch_size = 8
        mixprec_net = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                              w_mixprec_type=MixPrecType.PER_CHANNEL)
        optimizer = torch.optim.Adam(mixprec_net.parameters())
        n_steps = 10
        # prev_loss = mixprec_net.get_regularization_loss()
        prev_loss = torch.tensor(float('inf'))
        print("Initial Reg. Loss:", prev_loss.item())
        for i in range(n_steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            mixprec_net(x)
            loss = mixprec_net.get_regularization_loss()
            print("Reg. Loss:", loss.item())
            self.assertLessEqual(loss.item(), prev_loss.item(), "The loss value is not descending")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prev_loss = loss

    def test_regularization_loss_weights_layer(self):
        """Check that the weights remain equal using only the regularization loss
        with PER_LAYER weight mixed-precision (default)"""
        # we use ToyAdd_2D to verify that mask sharing does not create problems
        nn_ut = ToyAdd_2D()
        batch_size = 8
        pit_net = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        optimizer = torch.optim.Adam(pit_net.parameters())
        n_steps = 10
        conv0 = cast(MixPrec_Conv2d, pit_net.seed.conv0)
        conv1 = cast(MixPrec_Conv2d, pit_net.seed.conv1)
        conv2 = cast(MixPrec_Conv2d, pit_net.seed.conv2)
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

    def test_regularization_loss_weights_channel(self):
        """Check that the weights remain equal using only the regularization los
        with PER_CHANNEL weight mixed-precision"""
        # we use ToyAdd_2D to verify that mask sharing does not create problems
        nn_ut = ToyAdd_2D()
        batch_size = 8
        pit_net = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                          w_mixprec_type=MixPrecType.PER_CHANNEL)
        optimizer = torch.optim.Adam(pit_net.parameters())
        n_steps = 10
        conv0 = cast(MixPrec_Conv2d, pit_net.seed.conv0)
        conv1 = cast(MixPrec_Conv2d, pit_net.seed.conv1)
        conv2 = cast(MixPrec_Conv2d, pit_net.seed.conv2)
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

    def test_regularization_loss_alpha_descent_layer(self):
        """Check that alpha converge to the config corresponding to min prec using only the
        regularization loss with PER_LAYER weight mixed-precision (default)"""
        # as usual, we use ToyAdd to verify mask sharing is not problematic
        nn_ut = ToyAdd_2D()
        batch_size = 1
        a_prec = (2, 4, 8)
        w_prec = (8, 4, 2)
        mixprec_net = MixPrec(nn_ut,
                              input_shape=nn_ut.input_shape,
                              regularizer='macs',
                              activation_precisions=a_prec,
                              weight_precisions=w_prec)
        # we must use SGD to be sure we only consider gradients
        optimizer = torch.optim.SGD(mixprec_net.parameters(), lr=0.001)
        mixprec_net.eval()
        # required number of steps varies with lr, but 100 gives a sufficient margin
        max_steps = 100
        conv0 = cast(MixPrec_Conv2d, mixprec_net.seed.conv0)
        conv1 = cast(MixPrec_Conv2d, mixprec_net.seed.conv1)
        conv2 = cast(MixPrec_Conv2d, mixprec_net.seed.conv2)
        convs = [conv0, conv1, conv2]
        for i in range(max_steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            mixprec_net(x)
            loss = mixprec_net.get_regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # check that in the alpha nas params the greatest value is the one corresponding
        # to the first idx (lowest prec)
        alpha_masks = []
        for layer in convs:
            alpha_masks.append({
                'alpha_a': layer.mixprec_a_quantizer.alpha_prec.clone().detach(),
                'alpha_w': layer.mixprec_w_quantizer.alpha_prec.clone().detach(),
            })

        for alpha_set in alpha_masks:
            selected_a_prec = a_prec[int(alpha_set['alpha_a'].argmax())]
            self.assertTrue(selected_a_prec == 2,
                            f"Selected act prec is {selected_a_prec} instead of 2.")
            selected_w_prec = w_prec[int(alpha_set['alpha_w'].argmax())]
            self.assertTrue(selected_w_prec == 2,
                            f"Selected w prec is {selected_w_prec} instead of 2.")

    def test_regularization_loss_alpha_descent_channel(self):
        """Check that alpha converge to the config corresponding to min prec using only the
        regularization loss with PER_CHANNEL weight mixed-precision"""
        # as usual, we use ToyAdd to verify mask sharing is not problematic
        nn_ut = ToyAdd_2D()
        batch_size = 1
        a_prec = (2, 4, 8)
        w_prec = (8, 4, 2)
        mixprec_net = MixPrec(nn_ut,
                              input_shape=nn_ut.input_shape,
                              regularizer='macs',
                              w_mixprec_type=MixPrecType.PER_CHANNEL,
                              activation_precisions=a_prec,
                              weight_precisions=w_prec)
        # we must use SGD to be sure we only consider gradients
        optimizer = torch.optim.SGD(mixprec_net.parameters(), lr=0.001)
        mixprec_net.eval()
        # required number of steps varies with lr, but 100 gives a sufficient margin
        max_steps = 100
        conv0 = cast(MixPrec_Conv2d, mixprec_net.seed.conv0)
        conv1 = cast(MixPrec_Conv2d, mixprec_net.seed.conv1)
        conv2 = cast(MixPrec_Conv2d, mixprec_net.seed.conv2)
        convs = [conv0, conv1, conv2]
        for i in range(max_steps):
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            mixprec_net(x)
            loss = mixprec_net.get_regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # check that in the alpha nas params the greatest value is the one corresponding
        # to the first idx (lowest prec)
        alpha_masks = []
        for layer in convs:
            alpha_masks.append({
                'alpha_a': layer.mixprec_a_quantizer.alpha_prec.clone().detach(),
                'alpha_w': layer.mixprec_w_quantizer.alpha_prec.clone().detach(),
            })

        for alpha_set in alpha_masks:
            selected_a_prec = a_prec[int(alpha_set['alpha_a'].argmax())]
            self.assertTrue(selected_a_prec == 2,
                            f"Selected act prec is {selected_a_prec} instead of 2.")
            selected_w_prec = [w_prec[int(i)] for i in alpha_set['alpha_w'].argmax(dim=0)]
            self.assertTrue(selected_w_prec == [2] * alpha_set['alpha_w'].shape[-1],
                            f"Selected w prec is {selected_w_prec} instead of 2.")

    def test_combined_loss_layer(self):
        """Check that the network weights are changing with a combined loss
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = ToyAdd_2D()
        batch_size = 5
        lambda_param = 0.0005
        n_steps = 10
        mixprec_net = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(mixprec_net.parameters())
        conv1 = cast(MixPrec_Conv2d, mixprec_net.seed.conv1)
        prev_conv1_weight = torch.zeros_like(conv1.weight)
        for i in range(n_steps):
            input = torch.stack([torch.rand(nn_ut.input_shape)] * batch_size)
            target = torch.randint(0, 2, (batch_size,))
            output = mixprec_net(input)
            task_loss = criterion(output, target)
            nas_loss = lambda_param * mixprec_net.get_regularization_loss()
            total_loss = task_loss + nas_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            conv1_weight = conv1.weight.clone().detach()
            print("Cout=0, Cin=0 weights:", conv1_weight[0, 0])
            self.assertFalse(torch.all(torch.isclose(prev_conv1_weight, conv1_weight)))
            prev_conv1_weight = conv1_weight

    def test_combined_loss_channel(self):
        """Check that the network weights are changing with a combined loss
        with PER_CHANNEL weight mixed-precision"""
        nn_ut = ToyAdd_2D()
        batch_size = 5
        lambda_param = 0.0005
        n_steps = 10
        mixprec_net = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                              w_mixprec_type=MixPrecType.PER_CHANNEL)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(mixprec_net.parameters())
        conv1 = cast(MixPrec_Conv2d, mixprec_net.seed.conv1)
        prev_conv1_weight = torch.zeros_like(conv1.weight)
        for i in range(n_steps):
            input = torch.stack([torch.rand(nn_ut.input_shape)] * batch_size)
            target = torch.randint(0, 2, (batch_size,))
            output = mixprec_net(input)
            task_loss = criterion(output, target)
            nas_loss = lambda_param * mixprec_net.get_regularization_loss()
            total_loss = task_loss + nas_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            conv1_weight = conv1.weight.clone().detach()
            print("Cout=0, Cin=0 weights:", conv1_weight[0, 0])
            self.assertFalse(torch.all(torch.isclose(prev_conv1_weight, conv1_weight)))
            prev_conv1_weight = conv1_weight

    def test_combined_loss_const_labels_layer(self):
        """Check that a trivial training with constant labels converges when using a combined
        loss with PER_LAYER weight mixed-precision (default)"""
        nn_ut = SimpleNN2D()
        batch_size = 32
        lambda_param = 0.0005
        n_steps = 10
        mixprec_net = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(mixprec_net.parameters())
        with torch.no_grad():
            x = torch.rand((batch_size,) + nn_ut.input_shape)
            mixprec_net(x)
        for i in range(n_steps):
            input = torch.stack([torch.rand(nn_ut.input_shape)] * batch_size)
            target = torch.ones((batch_size,), dtype=torch.long)
            output = mixprec_net(input)
            task_loss = criterion(output, target)
            nas_loss = lambda_param * mixprec_net.get_regularization_loss()
            total_loss = task_loss + nas_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        output = mixprec_net(torch.stack([torch.rand(nn_ut.input_shape)] * batch_size))
        self.assertTrue(torch.sum(torch.argmax(output, dim=-1)) == batch_size,
                        "The network should output only 1s with all labels equal to 1")

    def test_combined_loss_const_labels_channel(self):
        """Check that a trivial training with constant labels converges when using a combined
        loss with PER_CHANNEL weight mixed-precision"""
        nn_ut = SimpleNN2D()
        batch_size = 32
        lambda_param = 0.0005
        n_steps = 10
        mixprec_net = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                              w_mixprec_type=MixPrecType.PER_CHANNEL)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(mixprec_net.parameters())
        for i in range(n_steps):
            input = torch.stack([torch.rand(nn_ut.input_shape)] * batch_size)
            target = torch.ones((batch_size,), dtype=torch.long)
            output = mixprec_net(input)
            task_loss = criterion(output, target)
            nas_loss = lambda_param * mixprec_net.get_regularization_loss()
            total_loss = task_loss + nas_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        output = mixprec_net(torch.stack([torch.rand(nn_ut.input_shape)] * batch_size))
        self.assertTrue(torch.sum(torch.argmax(output, dim=-1)) == batch_size,
                        "The network should output only 1s with all labels equal to 1")

    def test_combined_loss_regression_layer(self):
        """Check the output of the combined loss with a trivial regression problem
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = ToyRegression_2D()
        lambda_param = .5  # lambda very large on purpose
        batch_size = 32
        n_steps = 500
        mixprec_net = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(mixprec_net.parameters(), lr=1e-2)
        running_err = 0
        for i in range(n_steps):
            # generate 10 random numbers
            input = torch.rand((batch_size,) + nn_ut.input_shape)
            # the desired output is their sum
            target = torch.sum(input, dim=(2, 3)).reshape((batch_size, -1))
            output = mixprec_net(input)
            task_loss = criterion(output, target)
            nas_loss = lambda_param * mixprec_net.get_regularization_loss()
            total_loss = task_loss + nas_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            err = float(torch.abs(target - output).detach().numpy().mean())
            running_err = ((running_err * i) + err) / (i + 1)
        self.assertLess(running_err, 1, "Error not decreasing")

    def test_combined_loss_regression_channel(self):
        """Check the output of the combined loss with a trivial regression problem
        with PER_CHANNEL weight mixed-precision"""
        nn_ut = ToyRegression_2D()
        lambda_param = 0.5  # lambda very large on purpose
        batch_size = 32
        n_steps = 500
        mixprec_net = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                              w_mixprec_type=MixPrecType.PER_CHANNEL)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(mixprec_net.parameters(), lr=1e-2)
        running_err = 0
        for i in range(n_steps):
            # generate 10 random numbers
            input = torch.rand((batch_size,) + nn_ut.input_shape)
            # the desired output is their sum
            target = torch.sum(input, dim=(2, 3)).reshape((batch_size, -1))
            output = mixprec_net(input)
            task_loss = criterion(output, target)
            nas_loss = lambda_param * mixprec_net.get_regularization_loss()
            total_loss = task_loss + nas_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            err = float(torch.abs(target - output).detach().numpy().mean())
            running_err = ((running_err * i) + err) / (i + 1)
        self.assertLess(running_err, 1, "Error not decreasing")


if __name__ == '__main__':
    unittest.main(verbosity=2)
