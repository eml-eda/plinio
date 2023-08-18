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
from typing import cast
import unittest
import torch
from plinio.methods import PIT
from plinio.methods.pit.nn import PITConv1d
from plinio.methods.pit.nn.binarizer import PITBinarizer
from unit_test.models import SimpleNN, TCResNet14
from unit_test.models import ToyAdd, ToyChannelsCat
from unit_test.models.toy_models import ToySequentialConv1d, ToySequentialSeparated
from unit_test.test_methods.test_pit.utils import check_channel_mask_init, write_channel_mask, \
        read_channel_mask, rand_binary_channel_mask, check_input_features, check_rf_mask_init, \
        write_rf_mask, check_dilation_mask_init, write_dilation_mask


class TestPITMasking(unittest.TestCase):
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
            "use_dilation": False,
            "avg_pool": True,
        }

    def test_channel_mask_init(self):
        """Test initialization of channel masks"""
        nn_ut = ToySequentialConv1d()
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        # check that the original channel mask is set with all 1
        check_channel_mask_init(self, pit_net, ('conv0', 'conv1'))

    def test_channel_mask_sharing(self):
        """Test that channel masks sharing works correctly"""
        nn_ut = ToyAdd()
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        # check that the original channel mask is set with all 1
        check_channel_mask_init(self, pit_net, ('conv0', 'conv1'))
        mask0 = torch.Tensor([1, 1, 1, 1, 1, 1, 0, 0, 1, 1])
        write_channel_mask(pit_net, 'conv0', mask0)
        # since conv0 and conv1 share their maskers, we should see it also on conv1
        mask1 = read_channel_mask(pit_net, 'conv1')
        self.assertTrue(torch.all(mask0 == mask1), "Masks not correctly shared")
        # after a forward step, they should remain identical
        _ = pit_net(torch.rand((32,) + nn_ut.input_shape))
        mask0 = read_channel_mask(pit_net, 'conv0')
        mask1 = read_channel_mask(pit_net, 'conv1')
        self.assertTrue(torch.all(mask0 == mask1), "Masks no longer equal after forward")

    def test_channel_mask_sharing_advanced(self):
        """Test that channel masks sharing works correctly for a more advanced model"""
        config = self.tc_resnet_config
        nn_ut = TCResNet14(config)
        pit_net = PIT(nn_ut, input_shape=(6, 50))
        # check that the original channel mask is set with all 1
        check_channel_mask_init(self, pit_net, ('tcn.network.0.tcn1', 'tcn.network.0.downsample'))
        mask0, _ = rand_binary_channel_mask(config['num_channels'][1])
        write_channel_mask(pit_net, 'tcn.network.0.tcn1', mask0)
        # since the two layers share their maskers, we should see it also on the second one
        mask1 = read_channel_mask(pit_net, 'tcn.network.0.downsample')
        self.assertTrue(torch.all(mask0 == mask1), "Masks not correctly shared")
        # after a forward step, they should remain identical
        _ = pit_net(torch.rand((32, 6, 50)))
        mask0 = read_channel_mask(pit_net, 'tcn.network.0.tcn1')
        mask1 = read_channel_mask(pit_net, 'tcn.network.0.downsample')
        self.assertTrue(torch.all(mask0 == mask1), "Masks no longer equal after forward")

    def test_channel_mask_cat(self):
        """Test that layers fed into a cat operation over the channels axis have correct input
        features"""
        nn_ut = ToyChannelsCat()
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        mask0 = (torch.rand((10,)) > 0.5).float()
        write_channel_mask(pit_net, 'conv0', mask0)
        mask1 = (torch.rand((15,)) > 0.5).float()  # float but only 0 and 1
        write_channel_mask(pit_net, 'conv1', mask1)
        # execute model to propagate input features
        _ = pit_net(torch.stack([torch.rand(nn_ut.input_shape)] * 32, 0))
        exp_features = int(torch.sum(mask0)) + int(torch.sum(mask1))
        # first channels are always alive
        exp_features += 1 if mask0[-1] == 0 else 0
        exp_features += 1 if mask1[-1] == 0 else 0
        check_input_features(self, pit_net, {'conv2': exp_features})

    def test_rf_mask_init(self):
        """Test a pit layer receptive field masks at initialization"""
        nn_ut = ToySequentialConv1d()
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        check_rf_mask_init(self, pit_net, ('conv0', 'conv1'))

    def test_rf_mask_forced(self):
        """Test a pit layer receptive field masks forcing some beta values"""
        nn_ut = ToySequentialConv1d()
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        # first try
        write_rf_mask(pit_net, 'conv0', torch.tensor([0.25, 0.4, 0.4]))
        pit_net(torch.rand(nn_ut.input_shape))
        conv0 = cast(PITConv1d, pit_net.seed.conv0)
        theta_beta = conv0.timestep_masker.theta
        bin_theta_beta = PITBinarizer.apply(theta_beta, 0.5)
        # note: first beta element is converted to 1 regardless of value due to "keep-alive"
        theta_beta_exp = torch.tensor([0.25, 0.4 + 0.25, 1 + 0.4 + 0.25])
        bin_theta_beta_exp = torch.tensor([0, 1, 1])
        self.assertTrue(torch.all(theta_beta == theta_beta_exp))
        self.assertTrue(torch.all(bin_theta_beta == bin_theta_beta_exp))
        # second try
        write_rf_mask(pit_net, 'conv0', torch.tensor([0, 0.1, 0.4]))
        pit_net(torch.rand(nn_ut.input_shape))
        theta_beta = conv0.timestep_masker.theta
        bin_theta_beta = PITBinarizer.apply(theta_beta, 0.5)
        theta_beta_exp = torch.tensor([0, 0.1, 1 + 0.1])
        bin_theta_beta_exp = torch.tensor([0, 0, 1])
        self.assertTrue(torch.all(theta_beta == theta_beta_exp))
        self.assertTrue(torch.all(bin_theta_beta == bin_theta_beta_exp))

    def test_dilation_mask_init(self):
        """Test a pit layer dilation masks"""
        nn_ut = ToySequentialConv1d()
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        check_dilation_mask_init(self, pit_net, ('conv0', 'conv1'))

    def test_dilation_mask_forced(self):
        """Test a pit layer receptive field masks forcing some beta values"""
        nn_ut = ToySequentialConv1d()
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        # first try
        write_dilation_mask(pit_net, 'conv0', torch.tensor([0.2, 0.5]))
        pit_net(torch.rand(nn_ut.input_shape))
        conv0 = cast(PITConv1d, pit_net.seed.conv0)
        theta_gamma = conv0.dilation_masker.theta
        bin_theta_gamma = PITBinarizer.apply(theta_gamma, 0.5)
        # note: first gamma elm is converted to 1 regardless of value due to "keep-alive"
        theta_gamma_exp = torch.tensor([1 + 0.2, 0.2, 1 + 0.2])
        bin_theta_gamma_exp = torch.tensor([1, 0, 1])
        self.assertTrue(torch.all(theta_gamma == theta_gamma_exp))
        self.assertTrue(torch.all(bin_theta_gamma == bin_theta_gamma_exp))
        # second try
        write_dilation_mask(pit_net, 'conv0', torch.tensor([0.6, 0.4]))
        pit_net(torch.rand(nn_ut.input_shape))
        theta_beta = conv0.dilation_masker.theta
        bin_theta_beta = PITBinarizer.apply(theta_beta, 0.5)
        theta_beta_exp = torch.tensor([1 + 0.6, 0.6, 1 + 0.6])
        bin_theta_beta_exp = torch.tensor([1, 1, 1])
        self.assertTrue(torch.all(theta_beta == theta_beta_exp))
        self.assertTrue(torch.all(bin_theta_beta == bin_theta_beta_exp))

    def test_keep_alive_masks(self):
        """Test the correctness of keep alive masks"""
        nn_ut = SimpleNN()
        pit_net = PIT(nn_ut, input_shape=nn_ut.input_shape)
        # conv1 has a filter size of 5 and 57 output channels
        conv1 = cast(PITConv1d, pit_net.seed.conv1)
        ka_alpha = cast(torch.Tensor, conv1.out_features_masker._keep_alive)
        exp_ka_alpha = torch.tensor([0.0] * 56 + [1.0], dtype=torch.float32)
        self.assertTrue(torch.equal(ka_alpha, exp_ka_alpha),
                        "Wrong keep-alive mask for channels")
        ka_beta = cast(torch.Tensor, conv1.timestep_masker._keep_alive)
        exp_ka_beta = torch.tensor([0.0] * 4 + [1.0], dtype=torch.float32)
        self.assertTrue(torch.equal(ka_beta, exp_ka_beta),
                        "Wrong keep-alive mask for rf")
        ka_gamma = cast(torch.Tensor, conv1.dilation_masker._keep_alive)
        exp_ka_gamma = torch.tensor([0.0] * 2 + [1.0], dtype=torch.float32)
        self.assertTrue(torch.equal(ka_gamma, exp_ka_gamma),
                        "Wrong keep-alive mask for dilation")

    def test_keep_alive_application(self):
        """Test the correctness of keep alive masks application"""
        net = SimpleNN()
        batch_size = 8
        input_shape = net.input_shape
        pit_net = PIT(net, input_shape=input_shape)
        x = torch.stack([torch.rand(input_shape)] * batch_size)
        # conv1 has a filter size of 5 and 57 output channels
        conv1 = cast(PITConv1d, pit_net.seed.conv1)
        write_channel_mask(pit_net, 'conv1', torch.tensor([0.05] * 57))
        write_rf_mask(pit_net, 'conv1', torch.tensor([0.05] * 5))
        write_dilation_mask(pit_net, 'conv1', torch.tensor([0.05] * 3))
        pit_net(x)
        chan_mask = conv1.out_features_masker.theta
        rf_mask = conv1.timestep_masker.theta
        dil_mask = conv1.dilation_masker.theta
        # even if we set all mask values to 0.05, the first gamma element is always > 1
        self.assertGreaterEqual(float(chan_mask[-1]), 1.0)
        self.assertGreaterEqual(float(rf_mask[-1]), 1.0)
        self.assertGreaterEqual(float(dil_mask[-1]), 1.0)

    def test_input_features_sequential(self):
        """Test that layers read correctly their input features when channel masks are applied"""
        net = ToySequentialSeparated()
        pit_net = PIT(net, input_shape=net.input_shape)
        alpha, cout = rand_binary_channel_mask(10)
        write_channel_mask(pit_net, 'conv0', alpha)
        # run an inference to update channels_eff
        pit_net(torch.rand((32,) + net.input_shape))
        conv1 = cast(PITConv1d, pit_net.seed.conv1)
        summ = conv1.summary()
        self.assertEqual(int(conv1.input_features_calculator.features), cout,
                         "Wrong number of input features retured by calculator")
        self.assertEqual(conv1.in_features_opt, cout,
                         "Wrong number of opt input channels retured by layer")
        self.assertEqual(summ['in_features'], cout,
                         "Wrong number of opt input channels retured by summary")

    def test_input_features_add(self):
        """Test that layers read correctly their input features when channel masks are applied,
        with sharing"""
        net = ToyAdd()
        pit_net = PIT(net, input_shape=net.input_shape)
        alpha, cout = rand_binary_channel_mask(10)
        write_channel_mask(pit_net, 'conv0', alpha)
        # run an inference to update channels_eff
        pit_net(torch.rand((32,) + net.input_shape))
        conv2 = cast(PITConv1d, pit_net.seed.conv2)
        summ = conv2.summary()
        self.assertEqual(int(conv2.input_features_calculator.features), cout,
                         "Wrong number of input features retured by calculator")
        self.assertEqual(conv2.in_features_opt, cout,
                         "Wrong number of opt input channels retured by layer")
        self.assertEqual(summ['in_features'], cout,
                         "Wrong number of opt input channels retured by summary")

    def test_input_features_cat(self):
        """Test that layers read correctly their input features when channel masks are applied,
        with concatenation"""
        net = ToyChannelsCat()
        pit_net = PIT(net, input_shape=net.input_shape)
        alpha0, cout0 = rand_binary_channel_mask(10)
        write_channel_mask(pit_net, 'conv0', alpha0)
        alpha1, cout1 = rand_binary_channel_mask(15)
        write_channel_mask(pit_net, 'conv1', alpha1)
        # run an inference to update channels_eff
        pit_net(torch.rand((32,) + net.input_shape))
        conv2 = cast(PITConv1d, pit_net.seed.conv2)
        summ = conv2.summary()
        self.assertEqual(int(conv2.input_features_calculator.features), cout0 + cout1,
                         "Wrong number of input features retured by calculator")
        self.assertEqual(conv2.in_features_opt, cout0 + cout1,
                         "Wrong number of opt input channels retured by layer")
        self.assertEqual(summ['in_features'], cout0 + cout1,
                         "Wrong number of opt input channels retured by summary")

    def test_input_features_advanced(self):
        """Test that layers read correctly their input features when channel masks are applied,
        with a complex network"""
        config = self.tc_resnet_config
        nn_ut = TCResNet14(config)
        pit_net = PIT(nn_ut, input_shape=(6, 50))
        alpha0, cout0 = rand_binary_channel_mask(config['num_channels'][0])
        write_channel_mask(pit_net, 'conv0', alpha0)
        converted_layer_names = dict(pit_net.seed.named_modules())
        # run an inference to update channels_eff
        pit_net(torch.rand((32, 6, 50)))
        tcn0 = cast(PITConv1d, converted_layer_names['tcn.network.0.tcn0'])
        summ = tcn0.summary()
        self.assertEqual(int(tcn0.input_features_calculator.features), cout0,
                         "Wrong number of input features retured by calculator")
        self.assertEqual(tcn0.in_features_opt, cout0,
                         "Wrong number of opt input channels retured by layer")
        self.assertEqual(summ['in_features'], cout0,
                         "Wrong number of opt input channels retured by summary")

        alpha1, cout1 = rand_binary_channel_mask(config['num_channels'][1])
        write_channel_mask(pit_net, 'tcn.network.0.tcn1', alpha1)
        converted_layer_names = dict(pit_net.seed.named_modules())
        # run an inference to update channels_eff
        pit_net(torch.rand((32, 6, 50)))
        tcn1 = cast(PITConv1d, converted_layer_names['tcn.network.1.tcn0'])
        summ = tcn1.summary()
        self.assertEqual(int(tcn1.input_features_calculator.features), cout1,
                         "Wrong number of input features retured by calculator")
        self.assertEqual(tcn1.in_features_opt, cout1,
                         "Wrong number of opt input channels retured by layer")
        self.assertEqual(summ['in_features'], cout1,
                         "Wrong number of opt input channels retured by summary")


if __name__ == '__main__':
    unittest.main(verbosity=2)
