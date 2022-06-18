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
from flexnas.methods.pit.pit_binarizer import PITBinarizer
from unit_test.models import SimpleNN
from unit_test.models import TCResNet14
from unit_test.models import SimplePitNN
from unit_test.models import ToyModel1, ToyModel2, ToyModel3
from unit_test.models import ToyModel4, ToyModel5, ToyModel6, ToyModel7, ToyModel8
from torch.nn.parameter import Parameter
# from pytorch_model_summary import summary
import torch.optim as optim
import numpy as np


class TestPIT(unittest.TestCase):
    """PIT NAS testing class.

    TODO: could be separated in more sub-classes, creating a test_pit folder with test_convert/
    test_extract/ etc subfolders.
    """

    def setUp(self):
        self.config = {
            "input_channels": 6,
            "output_size": 12,
            "num_channels": [24, 36, 36, 48, 48, 72, 72],
            "kernel_size": 9,
            "dropout": 0.5,
            "grad_clip": -1,
            "use_bias": True,
            "avg_pool": True,
        }

    def test_prepare_simple_model(self):
        """Test the conversion of a simple sequential model"""
        nn_ut = SimpleNN()
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 40)))
        self._compare_prepared(nn_ut, new_nn._inner_model, nn_ut, new_nn)
        # Number of NAS-able layers check
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 2
        self.assertEqual(exp_tgt, n_tgt,
                         "SimpleNN has {} conv layers, but found {} target layers".format(
                             exp_tgt, n_tgt))
        # Input features check on the NAS-able layers
        conv0_input = new_nn._inner_model.conv0.input_features_calculator.features  # type: ignore
        conv0_exp_input = 3
        self.assertEqual(conv0_exp_input, conv0_input,
                         "Conv0 has {} input features, but found {}".format(
                             conv0_exp_input, conv0_input))
        conv1_input = new_nn._inner_model.conv1\
                            .input_features_calculator.features.item()  # type: ignore
        conv1_exp_input = 32
        self.assertEqual(conv1_exp_input, conv1_input,
                         "Conv1 has {} input features, but found {}".format(
                             conv1_exp_input, conv1_input))

    def test_toy_model1(self):
        """Test PIT fucntionalities on ToyModel1"""
        nn_ut = ToyModel1()
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 15)))

        # Input features check
        conv2_exp_input = 3
        conv2_input = new_nn._inner_model.conv2\
                            .input_features_calculator.features  # type: ignore
        self.assertEqual(conv2_exp_input, conv2_input,
                         "Conv2 has {} input features, but found {}".format(
                             conv2_exp_input, conv2_input))
        conv5_exp_input = 64
        conv5_input = new_nn._inner_model.conv5\
                            .input_features_calculator.features  # type: ignore
        self.assertEqual(conv5_exp_input, conv5_input,
                         "Conv5 has {} input features, but found {}".format(
                             conv5_exp_input, conv5_input))
        conv4_exp_input = 50
        conv4_input = new_nn._inner_model.conv4\
                            .input_features_calculator.features  # type: ignore
        self.assertEqual(conv4_exp_input, conv4_input,
                         "Conv4 has {} input features, but found {}".format(
                             conv4_exp_input, conv4_input))

        # Input shared masker check
        conv5_alpha = new_nn._inner_model.\
            conv5.out_channel_masker.alpha.detach().numpy()  # type: ignore
        conv4_alpha = new_nn._inner_model.\
            conv4.out_channel_masker.alpha.detach().numpy()  # type: ignore
        conv3_alpha = new_nn._inner_model.\
            conv3.out_channel_masker.alpha.detach().numpy()  # type: ignore
        conv2_alpha = new_nn._inner_model.\
            conv2.out_channel_masker.alpha.detach().numpy()  # type: ignore
        conv1_alpha = new_nn._inner_model.\
            conv1.out_channel_masker.alpha.detach().numpy()  # type: ignore
        conv0_alpha = new_nn._inner_model.\
            conv0.out_channel_masker.alpha.detach().numpy()  # type: ignore

        # Two convolutional layers must have the same shared masker before a concat
        masker_alpha_conv_0_1 = np.array_equal(conv0_alpha, conv1_alpha)
        self.assertTrue(masker_alpha_conv_0_1)

        # The convolutional layer after the add operation must have a different one
        masker_alpha_conv_0_5 = np.array_equal(conv0_alpha, conv5_alpha)
        self.assertFalse(masker_alpha_conv_0_5)

        # Two consecutive convolutional layers with different out channels must have
        # different shared masker associated
        masker_alpha_conv_3_4 = np.array_equal(conv3_alpha, conv4_alpha)
        self.assertFalse(masker_alpha_conv_3_4)

        # Three convolutional layers before and add must have the same shared masker
        masker_alpha_conv_2_4 = np.array_equal(conv2_alpha, conv4_alpha)
        masker_alpha_conv_4_5 = np.array_equal(conv4_alpha, conv5_alpha)
        self.assertTrue(masker_alpha_conv_2_4)
        self.assertTrue(masker_alpha_conv_4_5)

        # Exclude types check
        nn_ut = ToyModel1()
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 15)),
                                       exclude_types=(nn.Conv1d,))
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 0
        self.assertEqual(exp_tgt, n_tgt,
                         "ToyModel1 (excluding the nn.Conv1d type) has {} NAS-able layers,\
                          but found {} target layers".format(exp_tgt, n_tgt))

        # Exclude names check
        nn_ut = ToyModel1()
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 15)),
                                       exclude_names=('conv0', 'conv4'))
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 4
        self.assertEqual(exp_tgt, n_tgt,
                         "ToyModel1 (excluding conv0 and conv4) has {} NAS-able layers,\
                          but found {} target layers".format(exp_tgt, n_tgt))
        # I must not find a PITChannelMasker corresponding to the excluded layer
        conv4_masker = True
        try:
            new_nn._inner_model.conv4.out_channel_masker.alpha.detach().numpy()  # type: ignore
        except Exception:
            conv4_masker = False
        self.assertFalse(conv4_masker)

    def test_toy_model2(self):
        """Test PIT functionalities on ToyModel2"""
        nn_ut = ToyModel2()
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 60)))
        # print(summary(nn_ut, torch.rand((1, 3, 60)), show_input=True, show_hierarchical=False))

        # Input features check
        conv2_exp_input = 3
        conv2_input = new_nn._inner_model.conv2\
                            .input_features_calculator.features  # type: ignore
        self.assertEqual(conv2_exp_input, conv2_input,
                         "Conv2 has {} input features, but found {}".format(
                             conv2_exp_input, conv2_input))
        conv4_exp_input = 40
        conv4_input = new_nn._inner_model.conv4\
                            .input_features_calculator.features  # type: ignore
        self.assertEqual(conv4_exp_input, conv4_input,
                         "Conv4 has {} input features, but found {}".format(
                             conv4_exp_input, conv4_input))

        # Input shared masker check
        conv1_alpha = new_nn._inner_model.\
            conv1.out_channel_masker.alpha.detach().numpy()  # type: ignore
        conv0_alpha = new_nn._inner_model.\
            conv0.out_channel_masker.alpha.detach().numpy()  # type: ignore

        # Two convolutional layers must have the same shared masker before a concat
        masker_alpha_conv_0_1 = np.array_equal(conv0_alpha, conv1_alpha)
        self.assertTrue(masker_alpha_conv_0_1)

        # Exclude names check
        nn_ut = ToyModel2()
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 60)),
                                       exclude_names=('conv0', 'conv4'))
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 3
        self.assertEqual(exp_tgt, n_tgt,
                         "ToyModel2 (excluding conv0 and conv4) has {} NAS-able layers,\
                          but found {} target layers".format(exp_tgt, n_tgt))

        # I must not find a PITChannelMasker corresponding to the excluded layer
        conv4_masker = True
        try:
            new_nn._inner_model.conv4.out_channel_masker.alpha.detach().numpy()  # type: ignore
        except Exception:
            conv4_masker = False
        self.assertFalse(conv4_masker)

        # Test autoconvert set to False
        nn_ut = ToyModel2()
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 60)),
                                       autoconvert_layers=False)
        self._compare_prepared(nn_ut, new_nn._inner_model, nn_ut, new_nn, autoconvert_layers=False)
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 0
        self.assertEqual(exp_tgt, n_tgt,
                         "SimpleNN (excluding the nn.Conv1d type) has {} NAS-able layers,\
                          but found {} target layers".format(exp_tgt, n_tgt))

    def test_exclude_names(self):
        """Test the exclude_names functionality"""
        nn_ut = SimpleNN()
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 40)),
                                       exclude_names=['conv0'])
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 1
        self.assertEqual(exp_tgt, n_tgt,
                         "SimpleNN (excluding conv0) has {} NAS-able layers , \
                             but found {} target layers".format(exp_tgt, n_tgt))
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 40)),
                                       exclude_names=['conv0', 'conv1'])
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 0
        self.assertEqual(exp_tgt, n_tgt,
                         "SimpleNN (excluding conv0 and conv1) has {} NAS-able layers, \
                          but found {} target layers".format(exp_tgt, n_tgt))
        nn_ut = TCResNet14(self.config)
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 6, 50)),
                                       exclude_names=['conv0', 'tcn_network_5_tcn1',
                                       'tcn_network_3_tcn0', 'tcn_network_2_batchnorm1'])
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 16
        self.assertEqual(exp_tgt, n_tgt,
                         "TCResNet14 (excluding 3 conv layers) has {} NAS-able layers, \
                          but found {} target layers".format(exp_tgt, n_tgt))

    def test_exclude_types(self):
        """Test the exclude_types functionality"""
        nn_ut = SimpleNN()
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 40)),
                                       exclude_types=(nn.Conv1d,))
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 0
        self.assertEqual(exp_tgt, n_tgt,
                         "SimpleNN (excluding the nn.Conv1d type) has {} NAS-able layers,\
                          but found {} target layers".format(exp_tgt, n_tgt))

        nn_ut = TCResNet14(self.config)
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 6, 50)),
                                       exclude_types=(nn.Conv1d,))
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 0
        self.assertEqual(exp_tgt, n_tgt,
                         "SimpleNN (excluding the nn.Conv1d type) has {} NAS-able layers,\
                          but found {} target layers".format(exp_tgt, n_tgt))

    def test_prepare_tc_resnet_14(self):
        """Test the conversion of a ResNet-like model"""
        nn_ut = TCResNet14(self.config)
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 6, 50)))
        self._compare_prepared(nn_ut, new_nn._inner_model, nn_ut, new_nn)

        # Number of NAS-able layers check
        n_tgt = len(new_nn._target_layers)
        exp_tgt = 3 * len(self.config['num_channels'][1:]) + 1
        self.assertEqual(exp_tgt, n_tgt,
                         "TCResNet14 has {} conv layers, but found {} target layers".format(
                             exp_tgt, n_tgt))

        converted_layers_name = dict(new_nn._inner_model.named_modules())
        # print(converted_layers_name.keys())

        # Input features check on the NAS-able layers
        conv0_input = new_nn._inner_model\
                            .conv0.input_features_calculator.features  # type: ignore
        conv0_exp_input = 6
        self.assertEqual(conv0_exp_input, conv0_input,
                         "Conv0 has {} input features, but found {}".format(
                             conv0_exp_input, conv0_input))

        tcn_network_0_tcn0_exp_input = 24
        tcn_network_0_tcn0_input = \
            converted_layers_name['tcn.network.0.tcn0'].input_features_calculator\
                                                       .features.item()  # type: ignore
        self.assertEqual(tcn_network_0_tcn0_exp_input, tcn_network_0_tcn0_input,
                         "tcn.network.0.tcn0 has {} input features, but found {}".format(
                             tcn_network_0_tcn0_exp_input, tcn_network_0_tcn0_input))

        tcn_network_2_downsample_exp_input = 36
        tcn_network_2_downsample_input = \
            converted_layers_name['tcn.network.2.downsample'].input_features_calculator\
                                                             .features  # type: ignore
        self.assertEqual(tcn_network_2_downsample_exp_input, tcn_network_2_downsample_input,
                         "tcn.network.2.downsample has {} input features, but found {}".format(
                             tcn_network_2_downsample_exp_input, tcn_network_2_downsample_input))

        tcn_network_5_tcn1_exp_input = 72
        tcn_network_5_tcn1_input = \
            converted_layers_name['tcn.network.5.tcn1'].input_features_calculator\
                                                       .features  # type: ignore
        self.assertEqual(tcn_network_5_tcn1_exp_input, tcn_network_5_tcn1_input,
                         "tcn.network.5.tcn1 has {} input features, but found {}".format(
                             tcn_network_5_tcn1_exp_input, tcn_network_5_tcn1_input))

    def test_prepare_simple_pit_model(self):
        """Test the conversion of a simple sequential model already containing a pit layer"""
        nn_ut = SimplePitNN()
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 40)))
        self._compare_prepared(nn_ut, new_nn._inner_model, nn_ut, new_nn)
        # Check with autoconvert disabled
        new_nn = self._execute_prepare(nn_ut, input_example=torch.rand((1, 3, 40)),
                                       autoconvert_layers=False)
        self._compare_prepared(nn_ut, new_nn._inner_model, nn_ut, new_nn, autoconvert_layers=False)

    def test_custom_channel_masking_ToyModel4(self):
        """Test a pit layer channels output with a custom mask alpha applied on ToyModel4"""
        nn_ut = ToyModel4()
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1])
        nn_ut.eval()
        pit_net.eval()
        y = nn_ut(x)
        pit_y = pit_net(x)
        assert torch.all(torch.eq(y, pit_y))
        # Check that the original channel mask is set with all 1
        assert torch.sum(pit_net._inner_model
                                .conv0.out_channel_masker.alpha).item() == 10  # type: ignore
        # Define a mask that will be overwritten
        new_mask = torch.Tensor([1, 1, 1, 1, 1, 1, 0, 0, 1, 1])
        conv0_alpha = Parameter(new_mask)
        pit_net._inner_model.conv0.out_channel_masker.alpha = conv0_alpha  # type: ignore
        # Define a custom mask for conv1
        new_mask = torch.Tensor([1, 1, 0, 0, 1, 1, 0, 1, 1, 1])
        conv1_alpha = Parameter(new_mask)
        pit_net._inner_model.conv1.out_channel_masker.alpha = conv1_alpha  # type: ignore
        pit_y = pit_net(x)
        # Before an add operation the two channel mask must be equal, so the new_mask
        # assigned on conv1 must also be present on conv0
        conv0_alpha = pit_net._inner_model.conv0.out_channel_masker.alpha  # type: ignore
        conv1_alpha = pit_net._inner_model.conv1.out_channel_masker.alpha  # type: ignore
        assert torch.all(torch.eq(conv0_alpha, conv1_alpha))
        conv2_input = pit_net._inner_model\
                             .conv2.input_features_calculator.features.item()  # type: ignore
        assert conv2_input == torch.sum(new_mask).item()

    def test_custom_channel_masking_ToyModel5(self):
        """Test a pit layer channels output with a custom mask alpha applied on ToyModel5"""
        nn_ut = ToyModel5()
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1])
        nn_ut.eval()
        pit_net.eval()
        y = nn_ut(x)
        pit_y = pit_net(x)
        assert torch.all(torch.eq(y, pit_y))
        # Check before a cat operation between 2 convolutional layers
        new_mask_0 = torch.Tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
        conv0_alpha = Parameter(new_mask_0)
        pit_net._inner_model.conv0.out_channel_masker.alpha = conv0_alpha  # type: ignore
        pit_y = pit_net(x)
        conv2_input = pit_net._inner_model\
                             .conv2.input_features_calculator.features.item()  # type: ignore
        assert conv2_input == torch.sum(new_mask_0).item() * 2

    def test_custom_channel_masking_ToyModel3(self):
        """Test a pit layer channels output with a custom mask alpha applied on ToyModel3"""
        nn_ut = ToyModel3()
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1])
        nn_ut.eval()
        pit_net.eval()
        y = nn_ut(x)
        pit_y = pit_net(x)
        assert torch.all(torch.eq(y, pit_y))
        new_mask_3 = torch.Tensor([1, 1, 0, 0, 0, 0, 0, 1])
        conv3_alpha = Parameter(new_mask_3)
        pit_net._inner_model.conv3.out_channel_masker.alpha = conv3_alpha  # type: ignore
        pit_y = pit_net(x)
        conv3_input = pit_net._inner_model\
                             .conv3.input_features_calculator.features  # type: ignore
        # Check that after the masking the conv layer takes only as input the alive channels
        assert conv3_input == torch.sum(new_mask_3).item()

    def test_custom_receptive_field_masking_ToyModel4(self):
        """Test a pit layer receptive field output with a custom mask applied"""
        nn_ut = ToyModel4()
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1])
        # Check the correct initialization of c_beta matrix
        assert torch.all(torch.eq(pit_net._inner_model
                                         .conv2.timestep_masker._c_beta,  # type: ignore
                         torch.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]])))  # type: ignore
        # Check the correct initialization of beta tensor
        assert torch.all(torch.eq(pit_net._inner_model
                                         .conv2.timestep_masker.beta,  # type: ignore
                         torch.tensor([1, 1, 1])))  # type: ignore
        # Check the correct initialization of theta beta tensor
        assert torch.all(torch.eq(pit_net._inner_model
                                         .conv2.timestep_masker(),  # type: ignore
                         torch.tensor([3, 2, 1])))  # type: ignore

        # Assign a new beta in order to get a beta binarized of [1, 0, 0].
        # The first channel is always alive.
        conv2_beta_new = Parameter(torch.Tensor([0.4, 0.3, 0.3]))
        conv2_beta_new = Parameter(PITBinarizer.apply(conv2_beta_new, 0.5))
        pit_net._inner_model.conv2.dilation_masker.beta = conv2_beta_new  # type: ignore
        pit_net(x)

        # Assign a new beta in order to get a beta binarized of [1, 0, 0].
        # The first channel is always alive.
        conv2_beta_new = Parameter(torch.Tensor([0.4, 0.3, 0.3]))
        conv2_beta_new = Parameter(PITBinarizer.apply(conv2_beta_new, 0.5))
        pit_net._inner_model.conv2.timestep_masker.beta = conv2_beta_new  # type: ignore
        pit_net(x)
        theta_beta = pit_net._inner_model.conv2.timestep_masker()  # type: ignore
        theta_gamma = pit_net._inner_model.conv2.dilation_masker()  # type: ignore
        # Check the correct value of the new theta beta tensor
        assert torch.all(torch.eq(theta_beta,  # type: ignore
                         torch.tensor([1, 0, 0])))  # type: ignore
        _beta_norm = pit_net._inner_model.conv2._beta_norm  # type: ignore
        _gamma_norm = pit_net._inner_model.conv2._gamma_norm  # type: ignore
        norm_theta_beta = torch.mul(theta_beta, _beta_norm)  # type: ignore
        norm_theta_gamma = torch.mul(theta_gamma, _gamma_norm)  # type: ignore
        assert "{:.4f}".format(torch.sum(torch.mul(norm_theta_beta,
                                                   norm_theta_gamma)).item()) == '0.3333'

        # Assign a new beta in order to get a beta binarized of [2, 1, 1].
        # The first channel is always alive.
        conv2_beta_new = Parameter(torch.Tensor([0.4, 0.3, 0.6]))
        conv2_beta_new = Parameter(PITBinarizer.apply(conv2_beta_new, 0.5))
        pit_net._inner_model.conv2.timestep_masker.beta = conv2_beta_new  # type: ignore
        pit_net(x)
        theta_beta = pit_net._inner_model.conv2.timestep_masker()  # type: ignore
        theta_gamma = pit_net._inner_model.conv2.dilation_masker()  # type: ignore
        # Check the correct value of the new theta beta tensor
        assert torch.all(torch.eq(theta_beta,  # type: ignore
                         torch.tensor([2, 1, 1])))  # type: ignore
        _beta_norm = pit_net._inner_model.conv2._beta_norm  # type: ignore
        _gamma_norm = pit_net._inner_model.conv2._gamma_norm  # type: ignore
        norm_theta_beta = torch.mul(theta_beta, _beta_norm)  # type: ignore
        norm_theta_gamma = torch.mul(theta_gamma, _gamma_norm)  # type: ignore
        assert "{:.4f}".format(torch.sum(torch.mul(norm_theta_beta,
                                                   norm_theta_gamma)).item()) == '2.1667'

    def test_custom_dilation_masking_ToyModel6(self):
        """Test a pit layer receptive field output with a custom mask applied"""
        nn_ut = ToyModel6()
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1])
        c_gamma = pit_net._inner_model.conv2.dilation_masker._c_gamma  # type: ignore
        # Check the correct initialization of c_gamma matrix
        exp_c_gamma = torch.Tensor([[1., 1., 1., 1.],
                                    [0., 0., 0., 1.],
                                    [0., 0., 1., 1.],
                                    [0., 0., 0., 1.],
                                    [0., 1., 1., 1.],
                                    [0., 0., 0., 1.],
                                    [0., 0., 1., 1.],
                                    [0., 0., 0., 1.],
                                    [1., 1., 1., 1.]])
        self.assertTrue(torch.equal(c_gamma, exp_c_gamma), "Wrong C gamma matrix")  # type: ignore
        # Check the correct initialization of the gamma tensor
        assert torch.all(torch.eq(pit_net._inner_model
                                         .conv2.dilation_masker.gamma,  # type: ignore
                         torch.tensor([1, 1, 1, 1])))  # type: ignore
        # Assign a new gamma
        conv2_gamma_new = Parameter(torch.Tensor([0.4, 0.4, 0.4, 0.4]))
        conv2_gamma_new = Parameter(PITBinarizer.apply(conv2_gamma_new, 0.5))
        pit_net._inner_model.conv2.dilation_masker.gamma = conv2_gamma_new  # type: ignore
        pit_net(x)
        theta_beta = pit_net._inner_model.conv2.timestep_masker()  # type: ignore
        theta_gamma = pit_net._inner_model.conv2.dilation_masker()  # type: ignore
        assert torch.all(torch.eq(theta_gamma,  # type: ignore
                         torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 1.])))  # type: ignore
        _beta_norm = pit_net._inner_model.conv2._beta_norm  # type: ignore
        _gamma_norm = pit_net._inner_model.conv2._gamma_norm  # type: ignore
        norm_theta_beta = torch.mul(theta_beta, _beta_norm)  # type: ignore
        norm_theta_gamma = torch.mul(theta_gamma, _gamma_norm)  # type: ignore
        # Check the correct value of the new theta gamma tensor
        assert "{:.4f}".format(torch.sum(torch.mul(norm_theta_beta,
                                                   norm_theta_gamma)).item()) == '0.5000'

        # Assign a new gamma
        conv2_gamma_new = Parameter(torch.Tensor([0.7, 0.4, 0.7, 0.4]))
        conv2_gamma_new = Parameter(PITBinarizer.apply(conv2_gamma_new, 0.5))
        pit_net._inner_model.conv2.dilation_masker.gamma = conv2_gamma_new  # type: ignore
        pit_net(x)
        theta_beta = pit_net._inner_model.conv2.timestep_masker()  # type: ignore
        theta_gamma = pit_net._inner_model.conv2.dilation_masker()  # type: ignore
        assert torch.all(torch.eq(theta_gamma,  # type: ignore
                         torch.tensor([2., 0., 1., 0., 1., 0., 1., 0., 2.])))  # type: ignore
        _beta_norm = pit_net._inner_model.conv2._beta_norm  # type: ignore
        _gamma_norm = pit_net._inner_model.conv2._gamma_norm  # type: ignore
        norm_theta_beta = torch.mul(theta_beta, _beta_norm)  # type: ignore
        norm_theta_gamma = torch.mul(theta_gamma, _gamma_norm)  # type: ignore
        # Check the correct value of the new theta gamma tensor
        assert "{:.4f}".format(torch.sum(torch.mul(norm_theta_beta,
                                                   norm_theta_gamma)).item()) == '2.3333'

    def test_keep_alive_masks_simple_SimpleNN(self):
        """Test keep alive mask on SimpleNN network"""
        net = SimpleNN()
        pit_net = PIT(net, input_example=torch.rand((1, 3, 40)))
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
        pit_net = PIT(net, input_example=torch.rand((1, 3, 15)))
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

    def test_keep_alive_masks_simple_ToyModel2(self):
        """Test keep alive mask on ToyModel2 network"""
        net = ToyModel2()
        pit_net = PIT(net, input_example=torch.rand((1, 3, 60)))
        # conv1 has a filter size of 3 and 40 output channels
        ka_alpha = pit_net._inner_model.conv1.out_channel_masker._keep_alive  # type: ignore
        exp_ka_alpha = torch.tensor([1.0] + [0.0] * 39, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_alpha,  # type: ignore
                                    exp_ka_alpha), "Wrong keep-alive \
                                                    mask for channels")  # type: ignore
        ka_beta = pit_net._inner_model.conv1.timestep_masker._keep_alive  # type: ignore
        exp_ka_beta = torch.tensor([1.0] + [0.0] * 2, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_beta,   # type: ignore
                                    exp_ka_beta), "Wrong keep-alive \
                                                  mask for rf")  # type: ignore
        ka_gamma = pit_net._inner_model.conv1.dilation_masker._keep_alive  # type: ignore
        exp_ka_gamma = torch.tensor([1.0] + [0.0] * 1, dtype=torch.float32)
        self.assertTrue(torch.equal(ka_gamma,   # type: ignore
                                    exp_ka_gamma), "Wrong keep-alive \
                                                    mask for dilation")  # type: ignore

    def test_c_matrices_SimpleNN(self):
        """Test c_beta and c_gamma matrices on SimpleNN network"""
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
        pit_net = PIT(net, input_example=torch.rand((1, 3, 15)))
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
        x = torch.rand((32,) + tuple(net.input_shape[1:]))
        pit_net = PIT(net, input_example=x[0:1])
        net.eval()
        pit_net.eval()
        y = net(x)
        pit_y = pit_net(x)
        assert torch.all(torch.eq(y, pit_y))

    def test_regularization_loss_get_size_macs_ToyModel6(self):
        """Test the regularization loss computation on ToyModel6"""
        net = ToyModel6()
        pit_net = PIT(net, input_example=torch.rand((1, 3, 15)))
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
                         (3 * 10 * 3 * 10) + (3 * 10 * 3 * 10) + (20 * 4 * 9 * 4),
                         "Wrong MACs size computed")  # type: ignore

    def test_regularization_loss_get_size_macs_ToyModel7(self):
        """Test the regularization loss computation on ToyModel7"""
        net = ToyModel7()
        pit_net = PIT(net, input_example=torch.rand((1, 3, 15)))
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
                         (3 * 10 * 7 * 10) + (3 * 10 * 7 * 10) + (20 * 4 * 9 * 4),
                         "Wrong MACs size computed")  # type: ignore

    def test_regularization_loss_forward_backward_ToyModel4(self):
        """Test the regularization loss after forward and backward steps on ToyModel4"""
        nn_ut = ToyModel4()
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1])
        optimizer = optim.Adam(pit_net.parameters())
        pit_net.eval()
        inputs = []
        for i in range(8):
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

    def test_regularization_network_weights_ToyModel3(self):
        """Check the weights remain equal using the regularization loss on ToyModel3"""
        nn_ut = ToyModel3()
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1])
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
        nn_ut = ToyModel4()
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1])
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
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1])
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
        nn_ut = ToyModel1()
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1])
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
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1])
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
        nn_ut = ToyModel4()
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1], train_channels=False)
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
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1], train_rf=False)
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
        x = torch.rand((32,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x[0:1], train_dilation=False)
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
        x = torch.rand((batch_size,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x)
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

    def test_combined_loss_ones_labels_ToyModel8(self):
        """Check the output of the combined loss with all labels equal to 1"""
        nn_ut = ToyModel8()
        batch_size = 5
        x = torch.rand((batch_size,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x)
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

    def test_combined_loss_zero_labels_ToyModel8(self):
        """Check the output of the combined loss with all labels equal to 0"""
        nn_ut = ToyModel8()
        batch_size = 5
        x = torch.rand((batch_size,) + tuple(nn_ut.input_shape[1:]))
        pit_net = PIT(nn_ut, input_example=x)
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

    @staticmethod
    def _execute_prepare(
            nn_ut: nn.Module,
            input_example: torch.Tensor,
            regularizer: str = 'size',
            exclude_names: Iterable[str] = (),
            exclude_types: Tuple[Type[nn.Module], ...] = (),
            autoconvert_layers=True):
        new_nn = PIT(nn_ut, input_example, regularizer, exclude_names=exclude_names,
                     exclude_types=exclude_types, autoconvert_layers=autoconvert_layers)
        return new_nn

    def _compare_prepared(self,
                          old_mod: nn.Module, new_mod: nn.Module,
                          old_top: nn.Module, new_top: DNAS,
                          exclude_names: Iterable[str] = (),
                          exclude_types: Tuple[Type[nn.Module], ...] = (),
                          autoconvert_layers=True):
        for name, child in old_mod.named_children():
            new_child = new_mod._modules[name]
            self._compare_prepared(child, new_child, old_top, new_top,   # type: ignore
                                   exclude_names, exclude_types,
                                   autoconvert_layers)  # type: ignore
            if isinstance(child, nn.Conv1d):
                if (name not in exclude_names) and \
                        (not isinstance(child, exclude_types) and (autoconvert_layers)):
                    assert isinstance(new_child, PITConv1d)
                    assert child.out_channels == new_child.out_channels
                    # TODO: add more checks


if __name__ == '__main__':
    unittest.main(verbosity=2)
