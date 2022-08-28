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
from typing import Tuple, Dict, cast
import unittest
import math
import torch
from flexnas.methods import PIT
from flexnas.methods.pit import PITConv1d
from flexnas.methods.pit.pit_binarizer import PITBinarizer
from unit_test.models import SimpleNN
from unit_test.models import ToyAdd, ToyChannelsCat
from torch.nn.parameter import Parameter

from unit_test.models.toy_models import ToySequentialConv1d


class TestPITMasking(unittest.TestCase):
    """Test masking operations in PIT"""

    def test_channel_mask_init(self):
        """Test initialization of channel masks"""
        nn_ut = ToySequentialConv1d()
        input_shape = (3, 15)
        pit_net = PIT(nn_ut, input_shape=input_shape)
        # check that the original channel mask is set with all 1
        self._check_channel_mask_init(pit_net, ('conv0', 'conv1'))

    def test_channel_mask_sharing(self):
        """Test that channel masks sharing works correctly"""
        nn_ut = ToyAdd()
        input_shape = (3, 15)
        pit_net = PIT(nn_ut, input_shape=input_shape)
        # check that the original channel mask is set with all 1
        self._check_channel_mask_init(pit_net, ('conv0', 'conv1'))
        mask0 = torch.Tensor([1, 1, 1, 1, 1, 1, 0, 0, 1, 1])
        self._write_channel_mask(pit_net, 'conv0', mask0)
        # since conv0 and conv1 share their maskers, we should see it also on conv1
        mask1 = self._read_channel_mask(pit_net, 'conv1')
        self.assertTrue(torch.all(mask0 == mask1), "Masks not correctly shared")
        # after a forward step, they should remain identical
        _ = pit_net(torch.rand(input_shape))
        mask0 = self._read_channel_mask(pit_net, 'conv0')
        mask1 = self._read_channel_mask(pit_net, 'conv1')
        self.assertTrue(torch.all(mask0 == mask1), "Masks no longer equal after forward")

    def test_channel_mask_cat(self):
        """Test that layers fed into a cat operation over the channels axis have correct input
        features"""
        nn_ut = ToyChannelsCat()
        input_shape = (3, 15)
        pit_net = PIT(nn_ut, input_shape=input_shape)
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
        pit_net = PIT(nn_ut, input_shape=input_shape)
        self._check_rf_mask_init(pit_net, ('conv0', 'conv1'))

    def test_rf_mask_forced(self):
        """Test a pit layer receptive field masks forcing some beta values"""
        nn_ut = ToySequentialConv1d()
        input_shape = (3, 15)
        pit_net = PIT(nn_ut, input_shape=input_shape)
        # first try
        self._write_rf_mask(pit_net, 'conv0', torch.tensor([0.4, 0.4, 0.25]))
        pit_net(torch.rand(input_shape))
        conv0 = cast(PITConv1d, pit_net._inner_model.conv0)
        theta_beta = conv0.timestep_masker()
        bin_theta_beta = PITBinarizer.apply(theta_beta, 0.5)
        # note: first beta element is converted to 1 regardless of value due to "keep-alive"
        theta_beta_exp = torch.tensor([1 + 0.4 + 0.25, 0.4 + 0.25, 0.25])
        bin_theta_beta_exp = torch.tensor([1, 1, 0])
        self.assertTrue(torch.all(theta_beta == theta_beta_exp))
        self.assertTrue(torch.all(bin_theta_beta == bin_theta_beta_exp))
        k_eff = conv0.k_eff
        # obtained from the paper formulas
        exp_norm_factors = torch.tensor([(1 / 3), (1 / 2), 1])
        k_eff_exp = torch.sum(torch.mul(theta_beta_exp, exp_norm_factors))
        self.assertAlmostEqual(float(k_eff), k_eff_exp)  # type: ignore
        # second try
        self._write_rf_mask(pit_net, 'conv0', torch.tensor([0.4, 0.1, 0]))
        pit_net(torch.rand(input_shape))
        theta_beta = conv0.timestep_masker()
        bin_theta_beta = PITBinarizer.apply(theta_beta, 0.5)
        theta_beta_exp = torch.tensor([1 + 0.1, 0.1, 0])
        bin_theta_beta_exp = torch.tensor([1, 0, 0])
        self.assertTrue(torch.all(theta_beta == theta_beta_exp))
        self.assertTrue(torch.all(bin_theta_beta == bin_theta_beta_exp))
        k_eff = conv0.k_eff
        k_eff_exp = torch.sum(torch.mul(theta_beta_exp, exp_norm_factors))
        self.assertAlmostEqual(float(k_eff), k_eff_exp)  # type: ignore

    def test_dilation_mask_init(self):
        """Test a pit layer dilation masks"""
        nn_ut = ToySequentialConv1d()
        input_shape = (3, 15)
        pit_net = PIT(nn_ut, input_shape=input_shape)
        self._check_dilation_mask_init(pit_net, ('conv0', 'conv1'))

    def test_dilation_mask_forced(self):
        """Test a pit layer receptive field masks forcing some beta values"""
        nn_ut = ToySequentialConv1d()
        input_shape = (3, 15)
        pit_net = PIT(nn_ut, input_shape=input_shape)
        # first try
        self._write_dilation_mask(pit_net, 'conv0', torch.tensor([0.5, 0.2]))
        pit_net(torch.rand(input_shape))
        conv0 = cast(PITConv1d, pit_net._inner_model.conv0)
        theta_gamma = conv0.dilation_masker()
        bin_theta_gamma = PITBinarizer.apply(theta_gamma, 0.5)
        # note: first gamma elm is converted to 1 regardless of value due to "keep-alive"
        theta_gamma_exp = torch.tensor([1 + 0.2, 0.2, 1 + 0.2])
        bin_theta_gamma_exp = torch.tensor([1, 0, 1])
        self.assertTrue(torch.all(theta_gamma == theta_gamma_exp))
        self.assertTrue(torch.all(bin_theta_gamma == bin_theta_gamma_exp))
        k_eff = conv0.k_eff
        # obtained from the paper formulas
        exp_norm_factors = torch.tensor([(1 / 2), 1, (1 / 2)])
        k_eff_exp = torch.sum(torch.mul(theta_gamma_exp, exp_norm_factors))
        self.assertAlmostEqual(float(k_eff), k_eff_exp)  # type: ignore
        # second try
        self._write_dilation_mask(pit_net, 'conv0', torch.tensor([0.4, 0.6]))
        pit_net(torch.rand(input_shape))
        theta_beta = conv0.dilation_masker()
        bin_theta_beta = PITBinarizer.apply(theta_beta, 0.5)
        theta_beta_exp = torch.tensor([1 + 0.6, 0.6, 1 + 0.6])
        bin_theta_beta_exp = torch.tensor([1, 1, 1])
        self.assertTrue(torch.all(theta_beta == theta_beta_exp))
        self.assertTrue(torch.all(bin_theta_beta == bin_theta_beta_exp))
        k_eff = conv0.k_eff
        k_eff_exp = torch.sum(torch.mul(theta_beta_exp, exp_norm_factors))
        self.assertAlmostEqual(float(k_eff), k_eff_exp)  # type: ignore

    def test_keep_alive_masks(self):
        """Test the correctness of keep alive masks"""
        nn_ut = SimpleNN()
        input_shape = (3, 40)
        pit_net = PIT(nn_ut, input_shape=input_shape)
        # conv1 has a filter size of 5 and 57 output channels
        conv1 = cast(PITConv1d, pit_net._inner_model.conv1)
        ka_alpha = conv1.out_channel_masker._keep_alive
        exp_ka_alpha = torch.tensor([1.0] + [0.0] * 56, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_alpha, exp_ka_alpha),
                        "Wrong keep-alive mask for channels")
        ka_beta = conv1.timestep_masker._keep_alive
        exp_ka_beta = torch.tensor([1.0] + [0.0] * 4, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_beta, exp_ka_beta),
                        "Wrong keep-alive mask for rf")
        ka_gamma = conv1.dilation_masker._keep_alive
        exp_ka_gamma = torch.tensor([1.0] + [0.0] * 2, dtype=torch.float32)
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
        conv1 = cast(PITConv1d, pit_net._inner_model.conv1)
        self._write_channel_mask(pit_net, 'conv1', torch.tensor([0.05] * 57))
        self._write_rf_mask(pit_net, 'conv1', torch.tensor([0.05] * 5))
        self._write_dilation_mask(pit_net, 'conv1', torch.tensor([0.05] * 3))
        pit_net(x)
        chan_mask = conv1.out_channel_masker()
        rf_mask = conv1.timestep_masker()
        dil_mask = conv1.dilation_masker()
        # even if we set all mask values to 0.05, the first gamma element is always > 1
        self.assertGreaterEqual(chan_mask[0], 1.0)
        self.assertGreaterEqual(rf_mask[0], 1.0)
        self.assertGreaterEqual(dil_mask[0], 1.0)

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
                len_gamma_exp = math.ceil(math.log2(rf_seed))
                gamma = layer.dilation_masker.gamma
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
        # TODO: avoid duplicate definition from test_pit_convert.
        converted_layer_names = dict(new_nn._inner_model.named_modules())
        for name, exp in input_features_dict.items():
            layer = converted_layer_names[name]
            in_features = layer.input_features_calculator.features  # type: ignore
            self.assertEqual(in_features, exp,
                             f"Layer {name} has {in_features} input features, expected {exp}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
