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
from typing import cast, Tuple
import unittest
import warnings

import torch
import torch.nn as nn
from plinio.methods import PIT
from plinio.methods.pit.nn import PITConv1d, PITConv2d
from unit_test.models import SimpleNN
from unit_test.models import TCResNet14
from unit_test.models import SimplePitNN
from unit_test.models import ToyAdd, ToyMultiPath1, ToyMultiPath2, ToyInputConnectedDW
from unit_test.models import ToyBatchNorm, ToyIllegalBN
from unit_test.models import CNN3D
from unit_test.models import DSCNN
from unit_test.models import TCN_IR
from unit_test.test_methods.test_pit.utils import compare_prepared, check_target_layers, \
        check_input_features, check_shared_maskers, check_frozen_maskers, check_layers_exclusion, \
        check_batchnorm_folding, check_batchnorm_unfolding, check_batchnorm_memory, \
        compare_identical


class TestPITConvert(unittest.TestCase):
    """Test conversion operations to/from nn.Module from/to PIT"""

    def setUp(self):
        self.tc_resnet_config = {
            "input_channels": 6,
            "output_size": 12,
            "num_channels": [24, 36, 36, 48, 48, 72, 72],
            "kernel_size": 9,
            "dropout": 0.5,
            "grad_clip": -1,
            "use_bias": True,
            "use_dilation": True,
            "avg_pool": True,
        }

    def test_autoimport_simple(self):
        """Test the conversion of a simple sequential model with layer autoconversion"""
        nn_ut = SimpleNN()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=3)
        check_input_features(self, new_nn, {'conv0': 3, 'conv1': 32, 'fc': 570})

    def test_autoimport_train_status_neutrality(self):
        """Test that the conversion does not change the training status of the original model"""
        # Training status is True
        nn_ut = SimpleNN()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        self.assertTrue(new_nn.training, 'Training status changed after conversion')
        self.assertTrue(new_nn.seed.training, 'Training status changed after conversion within new_nn.seed')

        # Training status is False
        nn_ut = SimpleNN()
        nn_ut.eval()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        self.assertFalse(new_nn.training, 'Training status changed after conversion')
        self.assertFalse(new_nn.seed.training, 'Training status changed after conversion within new_nn.seed')

    def test_autoimport_advanced(self):
        """Test the conversion of a ResNet-like model"""
        config = self.tc_resnet_config
        nn_ut = TCResNet14(config)
        new_nn = PIT(nn_ut, input_shape=(6, 50))
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=3 * len(config['num_channels'][1:]) + 2)
        # check some random layers input features
        fc_in_feats = config['num_channels'][-1] * (3 if config['avg_pool'] else 6)
        expected_features = {
            'conv0': 6,
            'tcn.network.0.tcn0': config['num_channels'][0],
            'tcn.network.2.downsample': config['num_channels'][1],
            'tcn.network.5.tcn1': config['num_channels'][-1],
            'out': fc_in_feats,
        }
        check_input_features(self, new_nn, expected_features)

    def test_raised_err_complex_shape(self):
        """Test that PIT raises TypeError"""
        nn_ut = SimpleNN()
        complex_shape = [nn_ut.input_shape, nn_ut.input_shape]
        with self.assertRaises(TypeError) as context:
            PIT(nn_ut, input_shape=complex_shape)  # type: ignore
        msg = 'A TypeError error must be raised.'
        self.assertTrue(isinstance(context.exception, TypeError), msg)

    def test_raised_warn_on_example_and_shape(self):
        """Test that PIT raises Warning when both example and shape are passed"""
        nn_ut = SimpleNN()
        example = torch.stack([torch.rand(nn_ut.input_shape)] * 1, 0)
        with warnings.catch_warnings(record=True) as context:
            PIT(nn_ut, input_example=example, input_shape=nn_ut.input_shape)
        msg = 'An UserWarning must be raised.'
        self.assertTrue(context[0].category is UserWarning, msg)

    def test_autoimport_depthwise(self):
        """Test the conversion of a model with depthwise convolutions (cin=cout=groups)"""
        nn_ut = DSCNN()
        example = torch.stack([torch.rand(nn_ut.input_shape)] * 3, 0)
        new_nn = PIT(nn_ut, input_example=example, input_shape=nn_ut.input_shape)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=14)
        check_input_features(self, new_nn, {'inputlayer': 1, 'depthwise2': 64,
                                            'pointwise3': 64, 'out': 64})
        shared_masker_rules = (
            ('inputlayer', 'depthwise1', True),
            ('conv1', 'depthwise2', True),
            ('conv2', 'depthwise3', True),
            ('conv3', 'depthwise4', True),
        )
        check_shared_maskers(self, new_nn, shared_masker_rules)

    def test_autoimport_net_with_complex_input(self):
        """Test the conversion of a model with a complex input type
        (e.g., list of tensors)"""
        nn_ut = TCN_IR()
        example = [torch.stack([torch.rand(nn_ut.input_shape)] * 2, 0) for _ in range(3)]
        new_nn = PIT(nn_ut, input_example=example)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=8)
        check_target_layers(self, new_nn, exp_tgt=4, unique=True)

    def test_autoimport_multipath(self):
        """Test the conversion of a toy model with multiple concat and add operations"""
        nn_ut = ToyMultiPath1()
        example = torch.stack([torch.rand(nn_ut.input_shape)] * 1, 0)
        new_nn = PIT(nn_ut, input_example=example)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=7)
        check_input_features(self, new_nn, {'conv2': 3, 'conv4': 50, 'conv5': 64, 'fc': 640})
        shared_masker_rules = (
            ('conv2', 'conv4', True),   # inputs to add must share the masker
            ('conv2', 'conv5', True),   # inputs to add must share the masker
            ('conv0', 'conv1', False),  # inputs to concat over the channels must not share
            ('conv3', 'conv4', False),  # consecutive convs must not share
            ('conv0', 'conv5', False),  # two far aways layers must not share
        )
        check_shared_maskers(self, new_nn, shared_masker_rules)

        nn_ut = ToyMultiPath2()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=7)
        check_input_features(self, new_nn, {'conv2': 3, 'conv4': 40})
        shared_masker_rules = (
            ('conv0', 'conv1', True),   # inputs to add
            ('conv2', 'conv3', False),  # concat over channels
        )
        check_shared_maskers(self, new_nn, shared_masker_rules)

    def test_autoimport_frozen_features(self):
        """Test that input- and output-connected features masks are correctly 'frozen'"""
        nn_ut = ToyInputConnectedDW()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        frozen_masker_rules = (
            ('dw_conv', True),   # input-connected and DW
            ('pw_conv', False),  # normal
            ('fc', True),        # output-connected
        )
        check_frozen_maskers(self, new_nn, frozen_masker_rules)
        nn_ut = ToyAdd()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        frozen_masker_rules = (
            ('conv0', False),   # input-connected but not DW
            ('conv1', False),   # input-connected byt not DW
            ('conv2', False),   # normal
            ('fc', True),       # output-connected
        )
        check_frozen_maskers(self, new_nn, frozen_masker_rules)

    def test_autoimport_3d(self):
        """Test the conversion of a simple 3D CNN with layer autoconversion"""
        nn_ut = CNN3D()
        input_shape = (nn_ut.in_channel, nn_ut.patch_size, nn_ut.patch_size)
        new_nn = PIT(nn_ut, input_shape=input_shape)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=7)
        check_input_features(self, new_nn, {'conv1': 1,
                                            'pool1': nn_ut.num_filter,
                                            'conv2': nn_ut.num_filter,
                                            'conv3': nn_ut.num_filter_2,
                                            'conv4': nn_ut.num_filter_2,
                                            'fc1': nn_ut.features_size})

    def test_exclude_types_simple(self):
        nn_ut = ToyMultiPath1()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape, exclude_types=(nn.Conv1d,))
        # excluding Conv1D, only the final FC should be converted to PIT format
        check_target_layers(self, new_nn, exp_tgt=1)
        excluded = ('conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5')
        check_layers_exclusion(self, new_nn, excluded)

    def test_exclude_names_simple(self):
        nn_ut = ToyMultiPath1()
        excluded = ('conv0', 'conv4')
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape, exclude_names=excluded)
        # excluding conv0 and conv4, there are 5 convertible conv1d and linear layers left
        check_target_layers(self, new_nn, exp_tgt=5)
        check_layers_exclusion(self, new_nn, excluded)

        nn_ut = ToyMultiPath2()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape, exclude_names=excluded)
        # excluding conv0 and conv4, there are 4 convertible conv1d  and linear layers left
        check_target_layers(self, new_nn, exp_tgt=5)
        check_layers_exclusion(self, new_nn, excluded)

    def test_import_simple(self):
        """Test the conversion of a simple sequential model that already contains a PIT layer"""
        nn_ut = SimplePitNN()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        compare_prepared(self, nn_ut, new_nn.seed)
        # convert with autoconvert disabled. This is as if we exclude layers except the one already
        # in PIT form
        excluded = ('conv1')
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape, autoconvert_layers=False)
        compare_prepared(self, nn_ut, new_nn.seed, exclude_names=excluded)

    def test_batchnorm_fusion(self):
        """Test that batchnorms are correctly fused during import and re-generated during export"""
        nn_ut = ToyBatchNorm()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        check_batchnorm_folding(self, nn_ut, new_nn.seed)
        check_batchnorm_memory(self, new_nn.seed, ('dw_conv', 'pw_conv', 'fc1'))
        exported_nn = new_nn.export()
        check_batchnorm_unfolding(self, new_nn.seed, exported_nn)

    def test_batchnorm_fusion_illegal(self):
        """Test that unsupported batchnorm fusions trigger an error"""
        nn_ut = ToyIllegalBN()
        with self.assertRaises(ValueError):
            PIT(nn_ut, input_shape=nn_ut.input_shape)

    def test_exclude_names_advanced(self):
        """Test the exclude_names functionality on a ResNet like model"""
        config = self.tc_resnet_config
        nn_ut = TCResNet14(config)
        excluded = [
                'conv0',
                'tcn.network.5.tcn1',
                'tcn.network.5.batchnorm1',
                'tcn.network.3.tcn0',
                'tcn.network.3.batchnorm0'
                ]
        new_nn = PIT(nn_ut, input_shape=(6, 50), exclude_names=excluded)
        compare_prepared(self, nn_ut, new_nn.seed, exclude_names=excluded)
        n_layers = 3 * len(config['num_channels'][1:]) + 2 - 3
        check_layers_exclusion(self, new_nn, excluded)
        check_target_layers(self, new_nn, exp_tgt=n_layers)

    def test_export_initial_simple(self):
        """Test the export of a simple sequential model, just after import"""
        nn_ut = SimpleNN()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        exported_nn = new_nn.export()
        compare_identical(self, nn_ut, exported_nn)

    def test_export_initial_advanced(self):
        """Test the conversion of a ResNet-like model, just after import"""
        nn_ut = TCResNet14(self.tc_resnet_config)
        new_nn = PIT(nn_ut, input_shape=(6, 50))
        exported_nn = new_nn.export()
        compare_identical(self, nn_ut, exported_nn)

    def test_export_initial_depthwise(self):
        """Test the conversion of a model with depthwise convolutions, just after import"""
        nn_ut = DSCNN()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        exported_nn = new_nn.export()
        compare_identical(self, nn_ut, exported_nn)

    def test_export_with_masks(self):
        """Test the conversion of a simple model after forcing the mask values in some layers"""
        nn_ut = SimpleNN()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)

        conv0 = cast(PITConv1d, new_nn.seed.conv0)
        conv0.out_features_masker.alpha = nn.parameter.Parameter(
            torch.tensor([1, 1, 0, 1] * 8, dtype=torch.float))
        conv0.timestep_masker.beta = nn.parameter.Parameter(
            torch.tensor([0, 1, 1], dtype=torch.float))

        conv1 = cast(PITConv1d, new_nn.seed.conv1)
        conv1.out_features_masker.alpha = nn.parameter.Parameter(
            torch.tensor([1, ] * 55 + [0, 1], dtype=torch.float))
        conv1.dilation_masker.gamma = nn.parameter.Parameter(
            torch.tensor([0, 0, 1], dtype=torch.float))
        exported_nn = new_nn.export()

        for name, child in exported_nn.named_children():
            if name == 'conv0':
                child = cast(nn.Conv1d, child)
                pit_child = cast(PITConv1d, new_nn.seed._modules[name])
                self.assertEqual(child.out_channels, 24, "Wrong output channels exported")
                self.assertEqual(child.in_channels, 3, "Wrong input channels exported")
                self.assertEqual(child.kernel_size, (2,), "Wrong kernel size exported")
                self.assertEqual(child.dilation, (1,), "Wrong dilation exported")
                # check that first two timesteps of channel 0 are identical
                self.assertTrue(torch.all(child.weight[0, :, 0:2] == pit_child.weight[0, :, 1:3]),
                                "Wrong weight values in channel 0")
                # check that PIT's 4th channel weights are now stored in the 3rd channel
                self.assertTrue(torch.all(child.weight[2, :, 0:2] == pit_child.weight[3, :, 1:3]),
                                "Wrong weight values in channel 2")
            if name == 'conv1':
                child = cast(nn.Conv1d, child)
                pit_child = cast(PITConv1d, new_nn.seed._modules[name])
                self.assertEqual(child.out_channels, 56, "Wrong output channels exported")
                self.assertEqual(child.in_channels, 24, "Wrong input channels exported")
                self.assertEqual(child.kernel_size, (2,), "Wrong kernel size exported")
                self.assertEqual(child.dilation, (4,), "Wrong dilation exported")
                # check that weights are correctly saved with dilation. In this case the
                # number of input channels changed, so we can only check one Cin at a time
                self.assertTrue(
                    torch.all(child.weight[0:55, 0, 0:2] == pit_child.weight[0:55, 0, 0:6:4]),
                    "Wrong weight values for Cin=0")
                self.assertTrue(
                    torch.all(child.weight[0:55, 2, 0:2] == pit_child.weight[0:55, 3, 0:6:4]),
                    "Wrong weight values for Cin=2")

    def test_export_with_masks_advanced(self):
        """Test the conversion of a ResNet-like model
        after forcing the mask values in some layers"""
        nn_ut = TCResNet14(self.tc_resnet_config)
        new_nn = PIT(nn_ut, input_shape=(6, 50))

        tcn = cast(nn.Module, new_nn.seed.tcn)

        # block0.tcn1
        block0 = cast(nn.Module, cast(nn.Module, tcn.network)._modules['0'])
        tcn1 = cast(PITConv1d, block0.tcn1)
        # Force masking of channels
        tcn1.out_features_masker.alpha = nn.parameter.Parameter(
            torch.tensor([1, ] * 34 + [0, 1], dtype=torch.float))
        # Force masking of receptive-field
        tcn1.timestep_masker.beta = nn.parameter.Parameter(
            torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float))
        # Force masking of dilation
        tcn1.dilation_masker.gamma = nn.parameter.Parameter(
            torch.tensor([0, 1, 1, 1], dtype=torch.float))
        exported_nn = new_nn.export()

        # block1.tcn1
        block1 = cast(nn.Module, cast(nn.Module, tcn.network)._modules['1'])
        tcn1 = cast(PITConv1d, block1.tcn1)
        # Force masking of channels
        tcn1.out_features_masker.alpha = nn.parameter.Parameter(
            torch.tensor([1, ] * 33 + [0, 0, 1], dtype=torch.float))
        # Force masking of receptive-field
        tcn1.timestep_masker.beta = nn.parameter.Parameter(
            torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float))
        # Force masking of dilation
        tcn1.dilation_masker.gamma = nn.parameter.Parameter(
            torch.tensor([1, 1, 1, 1], dtype=torch.float))
        exported_nn = new_nn.export()

        # Checks on 'tcn.network.0.tcn1'
        _, mod = cast(Tuple, next(
            filter(lambda x: x[0] == 'tcn.network.0.tcn1', exported_nn.named_modules()),
            None))
        mod = cast(nn.Conv1d, mod)
        _, pad_mod = cast(Tuple, next(
            filter(lambda x: x[0] == 'tcn.network.0.pad1', exported_nn.named_modules()),
            None))
        pad_mod = cast(nn.ConstantPad1d, pad_mod)
        self.assertEqual(mod.out_channels, 35, "Wrong output channels exported")
        self.assertEqual(mod.in_channels, 36, "Wrong input channels exported")
        self.assertEqual(mod.kernel_size, (2,), "Wrong kernel size exported")
        self.assertEqual(mod.dilation, (2,), "Wrong dilation exported")
        # check that the output sequence length is the expecyed one
        # N.B., the tcn1 layer of TCResNet14 converges on a node sum of
        # a residual branch with stride=2. To sum toghether the sequences
        # their lenghts must match.
        dummy_tcn1_res_branch_oup = torch.rand((1, 35, 25))  # stride = 2
        dummy_tcn1_inp = torch.rand((1, 36, 25))
        dummy_tcn1_oup = mod(pad_mod(dummy_tcn1_inp))
        self.assertTrue(dummy_tcn1_oup.shape == dummy_tcn1_res_branch_oup.shape,
                        "Output Sequence legnth does not match on res branch")

        # Checks on 'tcn.network.1.tcn1'
        _, mod = cast(Tuple, next(
            filter(lambda x: x[0] == 'tcn.network.1.tcn1', exported_nn.named_modules()),
            None))
        mod = cast(nn.Conv1d, mod)
        _, pad_mod = cast(Tuple, next(
            filter(lambda x: x[0] == 'tcn.network.1.pad1', exported_nn.named_modules()),
            None))
        pad_mod = cast(nn.ConstantPad1d, pad_mod)
        self.assertEqual(mod.out_channels, 34, "Wrong output channels exported")
        self.assertEqual(mod.in_channels, 36, "Wrong input channels exported")
        self.assertEqual(mod.kernel_size, (7,), "Wrong kernel size exported")
        self.assertEqual(mod.dilation, (2,), "Wrong dilation exported")
        # check that the output sequence length is the expecyed one
        # N.B., the tcn1 layer of TCResNet14 converges on a node sum of
        # a residual branch with stride=1. To sum toghether the sequences
        # their lenghts must match.
        dummy_tcn1_res_branch_oup = torch.rand((1, 34, 25))  # stride = 1
        dummy_tcn1_inp = torch.rand((1, 36, 25))
        dummy_tcn1_oup = mod(pad_mod(dummy_tcn1_inp))
        self.assertTrue(dummy_tcn1_oup.shape == dummy_tcn1_res_branch_oup.shape,
                        "Output Sequence legnth does not match on res branch")

    def test_export_with_masks_depthwise(self):
        """Test the conversion of a model with depthwise conv after forcing the
        mask values in some layers"""
        nn_ut = DSCNN()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)

        conv1 = cast(PITConv2d, new_nn.seed.conv1)
        conv1.out_features_masker.alpha = nn.parameter.Parameter(
            torch.tensor([1, 1, 0, 1] * 16, dtype=torch.float))

        exported_nn = new_nn.export()

        for name, child in exported_nn.named_children():
            if name == 'conv1':
                child = cast(nn.Conv2d, child)
                pit_child = cast(PITConv2d, new_nn.seed._modules[name])
                self.assertEqual(child.out_channels, 48, "Wrong output channels exported")
                self.assertEqual(child.in_channels, 64, "Wrong input channels exported")
                # check that PIT's 4th channel weights are now stored in the 3rd channel
                self.assertTrue(torch.all(child.weight[2, :, :, :] == pit_child.weight[3, :, :, :]),
                                "Wrong weight values in channel 2")
            if name == 'depthwise2':
                child = cast(nn.Conv2d, child)
                pit_child = cast(PITConv2d, new_nn.seed._modules[name])
                self.assertEqual(child.out_channels, 48, "Wrong output channels exported")
                self.assertEqual(child.in_channels, 48, "Wrong input channels exported")
                # check that PIT's 4th channel weights are now stored in the 3rd channel
                self.assertTrue(torch.all(child.weight[2, :, :, :] == pit_child.weight[3, :, :, :]),
                                "Wrong weight values in channel 2")

    def test_arch_summary(self):
        """Test the summary report for a simple sequential model"""
        nn_ut = SimpleNN()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        summary = new_nn.summary()
        self.assertEqual(summary['conv0']['in_features'], 3, "Wrong in features summary")
        self.assertEqual(summary['conv0']['out_features'], 32, "Wrong out features summary")
        self.assertEqual(summary['conv0']['kernel_size'], (3,), "Wrong kernel size summary")
        self.assertEqual(summary['conv0']['dilation'], (1,), "Wrong dilation summary")
        self.assertEqual(summary['conv1']['in_features'], 32, "Wrong in features summary")
        self.assertEqual(summary['conv1']['out_features'], 57, "Wrong out features summary")
        self.assertEqual(summary['conv1']['kernel_size'], (5,), "Wrong kernel size summary")
        self.assertEqual(summary['conv1']['dilation'], (1,), "Wrong dilation summary")


if __name__ == '__main__':
    unittest.main(verbosity=2)
