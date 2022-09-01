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
from typing import cast, Iterable, Tuple, Type, Dict
import unittest
import torch
import torch.nn as nn
from flexnas.methods import PIT
from flexnas.methods.pit import PITConv1d
from flexnas.methods.pit import PITLayer
from unit_test.models import SimpleNN
from unit_test.models import TCResNet14
from unit_test.models import SimplePitNN
from unit_test.models import ToyMultiPath1, ToyMultiPath2


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
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=3)
        self._check_input_features(new_nn, {'conv0': 3, 'conv1': 32, 'fc': 570})

    def test_autoimport_advanced(self):
        """Test the conversion of a ResNet-like model"""
        config = self.tc_resnet_config
        nn_ut = TCResNet14(config)
        new_nn = PIT(nn_ut, input_shape=(6, 50))
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=3 * len(config['num_channels'][1:]) + 2)
        # check some random layers input features
        fc_in_feats = config['num_channels'][-1] * (3 if config['avg_pool'] else 6)
        expected_features = {
            'conv0': 6,
            'tcn.network.0.tcn0': config['num_channels'][0],
            'tcn.network.2.downsample': config['num_channels'][1],
            'tcn.network.5.tcn1': config['num_channels'][-1],
            'out': fc_in_feats,
        }
        self._check_input_features(new_nn, expected_features)

    def test_autoimport_multipath(self):
        """Test the conversion of a toy model with multiple concat and add operations"""
        nn_ut = ToyMultiPath1()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=7)
        self._check_input_features(new_nn, {'conv2': 3, 'conv4': 50, 'conv5': 64, 'fc': 640})
        shared_masker_rules = (
            ('conv2', 'conv4', True),   # inputs to add must share the masker
            ('conv2', 'conv5', True),   # inputs to add must share the masker
            ('conv0', 'conv1', False),  # inputs to concat over the channels must not share
            ('conv3', 'conv4', False),  # consecutive convs must not share
            ('conv0', 'conv5', False),  # two far aways layers must not share
        )
        self._check_shared_maskers(new_nn, shared_masker_rules)

        nn_ut = ToyMultiPath2()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=6)
        self._check_input_features(new_nn, {'conv2': 3, 'conv4': 40})
        shared_masker_rules = (
            ('conv0', 'conv1', True),   # inputs to add
            ('conv2', 'conv3', False),  # concat over channels
        )
        self._check_shared_maskers(new_nn, shared_masker_rules)

    def test_exclude_types_simple(self):
        nn_ut = ToyMultiPath1()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape, exclude_types=(nn.Conv1d,))
        # excluding Conv1D, only the final FC should be converted to PIT format
        self._check_target_layers(new_nn, exp_tgt=1)
        excluded = ('conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5')
        self._check_layers_exclusion(new_nn, excluded)

    def test_exclude_names_simple(self):
        nn_ut = ToyMultiPath1()
        excluded = ('conv0', 'conv4')
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape, exclude_names=excluded)
        # excluding conv0 and conv4, there are 5 convertible conv1d and linear layers left
        self._check_target_layers(new_nn, exp_tgt=5)
        self._check_layers_exclusion(new_nn, excluded)

        nn_ut = ToyMultiPath2()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape, exclude_names=excluded)
        # excluding conv0 and conv4, there are 4 convertible conv1d  and linear layers left
        self._check_target_layers(new_nn, exp_tgt=4)
        self._check_layers_exclusion(new_nn, excluded)

    def test_import_simple(self):
        """Test the conversion of a simple sequential model that already contains a PIT layer"""
        nn_ut = SimplePitNN()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        self._compare_prepared(nn_ut, new_nn.seed)
        # convert with autoconvert disabled. This is as if we exclude layers except the one already
        # in PIT form
        excluded = ('conv1')
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape, autoconvert_layers=False)
        self._compare_prepared(nn_ut, new_nn.seed, exclude_names=excluded)

    def test_exclude_names_advanced(self):
        """Test the exclude_names functionality on a ResNet like model"""
        config = self.tc_resnet_config
        nn_ut = TCResNet14(config)
        excluded = ['conv0', 'tcn.network.5.tcn1', 'tcn.network.3.tcn0']
        new_nn = PIT(nn_ut, input_shape=(6, 50), exclude_names=excluded)
        self._compare_prepared(nn_ut, new_nn.seed, exclude_names=excluded)
        n_layers = 3 * len(config['num_channels'][1:]) + 2 - len(excluded)
        self._check_layers_exclusion(new_nn, excluded)
        self._check_target_layers(new_nn, exp_tgt=n_layers)

    def test_export_initial_simple(self):
        """Test the export of a simple sequential model, just after import"""
        nn_ut = SimpleNN()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        exported_nn = new_nn.arch_export()
        self._compare_identical(nn_ut, exported_nn)

    def test_export_initial_advanced(self):
        """Test the conversion of a ResNet-like model, just after import"""
        nn_ut = TCResNet14(self.tc_resnet_config)
        new_nn = PIT(nn_ut, input_shape=(6, 50))
        exported_nn = new_nn.arch_export()
        self._compare_identical(nn_ut, exported_nn)

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
        exported_nn = new_nn.arch_export()

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

    def test_arch_summary(self):
        """Test the summary report for a simple sequential model"""
        nn_ut = SimpleNN()
        new_nn = PIT(nn_ut, input_shape=nn_ut.input_shape)
        summary = new_nn.arch_summary()
        self.assertEqual(summary['conv0']['in_features'], 3, "Wrong in features summary")
        self.assertEqual(summary['conv0']['out_features'], 32, "Wrong out features summary")
        self.assertEqual(summary['conv0']['kernel_size'], (3,), "Wrong kernel size summary")
        self.assertEqual(summary['conv0']['dilation'], (1,), "Wrong dilation summary")
        self.assertEqual(summary['conv1']['in_features'], 32, "Wrong in features summary")
        self.assertEqual(summary['conv1']['out_features'], 57, "Wrong out features summary")
        self.assertEqual(summary['conv1']['kernel_size'], (5,), "Wrong kernel size summary")
        self.assertEqual(summary['conv1']['dilation'], (1,), "Wrong dilation summary")
        print(new_nn)

    def _compare_prepared(self,
                          old_mod: nn.Module, new_mod: nn.Module,
                          base_name: str = "",
                          exclude_names: Iterable[str] = (),
                          exclude_types: Tuple[Type[nn.Module], ...] = ()):
        """Compare a nn.Module and its PIT-converted version"""
        for name, child in old_mod.named_children():
            new_child = cast(nn.Module, new_mod._modules[name])
            self._compare_prepared(child, new_child, base_name + name + ".",
                                   exclude_names, exclude_types)
            if isinstance(child, nn.Conv1d):
                if (base_name + name not in exclude_names) and not isinstance(child, exclude_types):
                    self.assertTrue(isinstance(new_child, PITConv1d), f"Layer {name} not converted")
                    self.assertEqual(child.out_channels, new_child.out_channels,
                                     f"Layer {name} wrong output channels")
                    self.assertEqual(child.kernel_size, new_child.kernel_size,
                                     f"Layer {name} wrong kernel size")
                    self.assertEqual(child.dilation, new_child.dilation,
                                     f"Layer {name} wrong dilation")
                    self.assertEqual(child.padding_mode, new_child.padding_mode,
                                     f"Layer {name} wrong padding mode")
                    self.assertEqual(child.padding, new_child.padding,
                                     f"Layer {name} wrong padding")
                    self.assertEqual(child.stride, new_child.stride,
                                     f"Layer {name} wrong stride")
                    self.assertEqual(child.groups, new_child.groups,
                                     f"Layer {name} wrong groups")
                    self.assertTrue(torch.all(child.weight == new_child.weight),
                                    f"Layer {name} wrong weight values")
                    self.assertTrue(torch.all(child.bias == new_child.bias),
                                    f"Layer {name} wrong bias values")
                    # TODO: add other layers

    def _compare_identical(self, old_mod: nn.Module, new_mod: nn.Module):
        """Compare two nn.Modules, where one has been imported and exported by PIT"""
        for name, child in old_mod.named_children():
            new_child = cast(nn.Module, new_mod._modules[name])
            self._compare_identical(child, new_child)
            if isinstance(child, nn.Conv1d):
                self.assertIsInstance(new_child, nn.Conv1d, "Wrong layer type")
                self.assertTrue(child.in_channels == new_child.in_channels)
                self.assertTrue(child.out_channels == new_child.out_channels)
                self.assertTrue(child.kernel_size == new_child.kernel_size)
                self.assertTrue(child.stride == new_child.stride)
                self.assertTrue(child.padding == new_child.padding)
                self.assertTrue(child.dilation == new_child.dilation)
                self.assertTrue(child.groups == new_child.groups)
                self.assertTrue(torch.all(child.weight == new_child.weight))
                if child.bias is not None:
                    self.assertTrue(torch.all(child.bias == new_child.bias))
                else:
                    self.assertIsNone(new_child.bias)
                self.assertTrue(child.padding_mode == new_child.padding_mode)

    def _check_target_layers(self, new_nn: PIT, exp_tgt: int):
        """Check if number of target layers is as expected"""
        n_tgt = len(new_nn._target_layers)
        self.assertEqual(exp_tgt, n_tgt,
                         "Expected {} target layers, but found {}".format(exp_tgt, n_tgt))

    def _check_input_features(self, new_nn: PIT, input_features_dict: Dict[str, int]):
        """Check if the number of input features of each layer in a NAS-able model is as expected.

        input_features_dict is a dictionary containing: {layer_name, expected_input_features}
        """
        converted_layer_names = dict(new_nn.seed.named_modules())
        for name, exp in input_features_dict.items():
            layer = converted_layer_names[name]
            in_features = layer.input_features_calculator.features  # type: ignore
            self.assertEqual(in_features, exp,
                             f"Layer {name} has {in_features} input features, expected {exp}")

    def _check_shared_maskers(self, new_nn: PIT, check_rules: Iterable[Tuple[str, str, bool]]):
        """Check if shared maskers are set correctly during an autoimport.

        The check_dict contains: {1st_layer: (2nd_layer, shared_flag)} where shared_flag can be
        true or false to specify that 1st_layer and 2nd_layer must/must-not share their maskers
        respectively.
        """
        converted_layer_names = dict(new_nn.seed.named_modules())
        for layer_1, layer_2, shared_flag in check_rules:
            masker_1 = converted_layer_names[layer_1].out_features_masker  # type: ignore
            masker_2 = converted_layer_names[layer_2].out_features_masker  # type: ignore
            if shared_flag:
                msg = f"Layers {layer_1} and {layer_2} are expected to share a masker, but don't"
                self.assertEqual(masker_1, masker_2, msg)
            else:
                msg = f"Layers {layer_1} and {layer_2} are expected to have independent maskers"
                self.assertNotEqual(masker_1, masker_2, msg)

    def _check_layers_exclusion(self, new_nn: PIT, excluded: Iterable[str]):
        """Check that layers in "excluded" have not be converted to PIT form"""
        converted_layer_names = dict(new_nn.seed.named_modules())
        for layer_name in excluded:
            layer = converted_layer_names[layer_name]
            # verify that the layer has not been converted to one of the NAS types
            self.assertNotIsInstance(type(layer), PITLayer,
                                     f"Layer {layer_name} should not be converted")
            # additionally, verify that there is no channel_masker (al PIT layers have it)
            # this is probably redundant
            try:
                layer.__getattr__('out_channel_masker')
            except Exception:
                pass
            else:
                self.fail("Excluded layer has the output_channel_masker set")


if __name__ == '__main__':
    unittest.main(verbosity=2)
