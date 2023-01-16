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

from typing import cast, Iterable, Tuple, Type
import unittest
import torch
import torch.nn as nn
from flexnas.methods import MixPrec
from flexnas.methods.mixprec.nn import MixPrec_Conv2d, MixPrecModule
import flexnas.methods.mixprec.quant.nn as qnn
from unit_test.models import SimpleNN2D, DSCNN, ToyMultiPath1_2D, ToyMultiPath2_2D, \
    SimpleMixPrecNN, SimpleExportedNN2D


class TestMixPrecConvert(unittest.TestCase):
    """Test conversion operations to/from nn.Module from/to MixPrec"""

    def test_autoimport_simple(self):
        """Test the conversion of a simple sequential model with layer autoconversion"""
        nn_ut = SimpleNN2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=3)

    def test_autoimport_depthwise(self):
        """Test the conversion of a model with depthwise convolutions (cin=cout=groups)"""
        nn_ut = DSCNN()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=14)

    def test_autoimport_multipath(self):
        """Test the conversion of a toy model with multiple concat and add operations"""
        nn_ut = ToyMultiPath1_2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=7)
        shared_quantizer_rules = (
            ('conv2', 'conv4', True),   # inputs to add must share the quantizer
            ('conv2', 'conv5', True),   # inputs to add must share the quantizer
            ('conv0', 'conv1', True),  # inputs to concat over the channels must share
            ('conv3', 'conv4', False),  # consecutive convs must not share
            ('conv0', 'conv5', False),  # two far aways layers must not share
        )
        self._check_shared_quantizers(new_nn, shared_quantizer_rules)

        nn_ut = ToyMultiPath2_2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=6)
        shared_quantizer_rules = (
            ('conv0', 'conv1', True),   # inputs to add
            ('conv2', 'conv3', True),  # concat over channels
        )
        self._check_shared_quantizers(new_nn, shared_quantizer_rules)

    def test_exclude_types_simple(self):
        nn_ut = ToyMultiPath1_2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape, exclude_types=(nn.Conv2d,))
        # excluding Conv2D, only the final FC should be converted to MixPrec format
        self._check_target_layers(new_nn, exp_tgt=1)
        excluded = ('conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5')
        self._check_layers_exclusion(new_nn, excluded)

    def test_exclude_names_simple(self):
        nn_ut = ToyMultiPath1_2D()
        excluded = ('conv0', 'conv4')
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape, exclude_names=excluded)
        # excluding conv0 and conv4, there are 5 convertible conv2d and linear layers left
        self._check_target_layers(new_nn, exp_tgt=5)
        self._check_layers_exclusion(new_nn, excluded)

        nn_ut = ToyMultiPath2_2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape, exclude_names=excluded)
        # excluding conv0 and conv4, there are 4 convertible conv1d  and linear layers left
        self._check_target_layers(new_nn, exp_tgt=4)
        self._check_layers_exclusion(new_nn, excluded)

    def test_import_simple(self):
        """Test the conversion of a simple sequential model that already contains a PIT layer"""
        nn_ut = SimpleMixPrecNN()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        self._compare_prepared(nn_ut, new_nn.seed)
        # convert with autoconvert disabled. This is as if we exclude layers except the one already
        # in MixPrec form
        excluded = ('conv1')
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape, autoconvert_layers=False)
        self._compare_prepared(nn_ut, new_nn.seed, exclude_names=excluded)

    def test_export_initial_simple(self):
        """Test the export of a simple sequential model, just after import"""
        nn_ut = SimpleNN2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                         activation_precisions=(8,), weight_precisions=(4, 8))
        exported_nn = new_nn.arch_export()
        expected_exported_nn = SimpleExportedNN2D()
        self._compare_exported(exported_nn, expected_exported_nn)

    def _compare_prepared(self,
                          old_mod: nn.Module, new_mod: nn.Module,
                          base_name: str = "",
                          exclude_names: Iterable[str] = (),
                          exclude_types: Tuple[Type[nn.Module], ...] = ()):
        """Compare a nn.Module and its MixPrec-converted version"""
        for name, child in old_mod.named_children():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # BN cannot be compared due to folding
                continue
            new_child = cast(nn.Module, new_mod._modules[name])
            self._compare_prepared(child, new_child, base_name + name + ".",
                                   exclude_names, exclude_types)
            if isinstance(child, nn.Conv2d):
                if (base_name + name not in exclude_names) and not isinstance(child, exclude_types):
                    self.assertTrue(isinstance(new_child, MixPrec_Conv2d),
                                    f"Layer {name} not converted")
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
                    # TODO: add other layers
                    # TODO: removed checks on weights due to BN folding
                    # self.assertTrue(torch.all(child.weight == new_child.weight),
                    #                 f"Layer {name} wrong weight values")
                    # self.assertTrue(torch.all(child.bias == new_child.bias),
                    #                 f"Layer {name} wrong bias values")

    def _compare_exported(self, exported_mod: nn.Module, expected_mod: nn.Module):
        """Compare two nn.Modules, where one has been imported and exported by MixPrec"""
        for name, child in expected_mod.named_children():
            new_child = cast(nn.Module, exported_mod._modules[name])
            # self._compare_exported(child, new_child)
            if isinstance(child, qnn.Quant_Conv2d):
                self.assertIsInstance(new_child, qnn.Quant_Conv2d, "Wrong layer type")
                # Check layer geometry
                self.assertTrue(child.in_channels == new_child.in_channels)
                self.assertTrue(child.out_channels == new_child.out_channels)
                self.assertTrue(child.kernel_size == new_child.kernel_size)
                self.assertTrue(child.stride == new_child.stride)
                self.assertTrue(child.padding == new_child.padding)
                self.assertTrue(child.dilation == new_child.dilation)
                self.assertTrue(child.groups == new_child.groups)
                self.assertTrue(child.padding_mode == new_child.padding_mode)
                # Check qtz param
                self.assertTrue(child.a_precision == new_child.a_precision)
                self.assertTrue(child.w_precision == new_child.w_precision)
            if isinstance(child, qnn.Quant_Linear):
                self.assertIsInstance(new_child, qnn.Quant_Linear, "Wrong layer type")
                # Check layer geometry
                self.assertTrue(child.in_features == new_child.in_features)
                self.assertTrue(child.out_features == new_child.out_features)
                # if child.bias is not None:
                #     self.assertTrue(torch.all(child.bias == new_child.bias))
                # else:
                #     self.assertIsNone(new_child.bias)
                # Check qtz param
                self.assertTrue(child.a_precision == new_child.a_precision)
                self.assertTrue(child.w_precision == new_child.w_precision)

                # Removed due to BN folding
                # self.assertTrue(torch.all(child.weight == new_child.weight))
                # if child.bias is not None:
                #     self.assertTrue(torch.all(child.bias == new_child.bias))
                # else:
                #     self.assertIsNone(new_child.bias)

    def test_export_with_qparams(self):
        """Test the conversion of a simple model after forcing the nas/quant
        params values in some layers"""
        nn_ut = SimpleNN2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape)

        conv0 = cast(MixPrec_Conv2d, new_nn.seed.conv0)
        conv0.mixprec_a_quantizer.alpha_prec = nn.parameter.Parameter(
            torch.tensor([0.3, 0.8, 0.99], dtype=torch.float))
        conv0.mixprec_a_quantizer.mix_qtz[2].clip_val = nn.parameter.Parameter(
            torch.tensor([5.], dtype=torch.float))
        conv0.mixprec_w_quantizer.alpha_prec = nn.parameter.Parameter(
            torch.tensor([1.5, 0.2, 1], dtype=torch.float))

        conv1 = cast(MixPrec_Conv2d, new_nn.seed.conv1)
        conv1.mixprec_a_quantizer.alpha_prec = nn.parameter.Parameter(
            torch.tensor([0.3, 1.8, 0.99], dtype=torch.float))
        conv1.mixprec_a_quantizer.mix_qtz[1].clip_val = nn.parameter.Parameter(
            torch.tensor([1.], dtype=torch.float))
        conv1.mixprec_w_quantizer.alpha_prec = nn.parameter.Parameter(
            torch.tensor([1.5, 0.2, 1.9], dtype=torch.float))

        exported_nn = new_nn.arch_export()
        # Dummy fwd to fill scale-factors values
        dummy_inp = torch.rand((2,) + nn_ut.input_shape)
        with torch.no_grad():
            exported_nn(dummy_inp)

        for name, child in exported_nn.named_children():
            if name == 'conv0':
                child = cast(qnn.Quant_Conv2d, child)
                # mixprec_child = cast(MixPrec_Conv2d, new_nn.seed._modules[name])
                self.assertEqual(child.a_precision, 8, "Wrong act precision")
                self.assertEqual(child.a_quantizer.clip_val, 5.,  # type: ignore
                                 "Wrong act qtz clip_val")
                self.assertEqual(child.w_precision, 2, "Wrong weight precision")
            if name == 'conv1':
                child = cast(qnn.Quant_Conv2d, child)
                # mixprec_child = cast(MixPrec_Conv2d, new_nn.seed._modules[name])
                self.assertEqual(child.a_precision, 4, "Wrong act precision")
                self.assertEqual(child.a_quantizer.clip_val, 1.,  # type: ignore
                                 "Wrong act qtz clip_val")
                self.assertEqual(child.w_precision, 8, "Wrong weight precision")

    def _check_target_layers(self, new_nn: MixPrec, exp_tgt: int):
        """Check if number of target layers is as expected"""
        n_tgt = len(new_nn._target_layers)
        self.assertEqual(exp_tgt, n_tgt,
                         "Expected {} target layers, but found {}".format(exp_tgt, n_tgt))

    def _check_shared_quantizers(self,
                                 new_nn: MixPrec,
                                 check_rules: Iterable[Tuple[str, str, bool]]):
        """Check if shared quantizers are set correctly during an autoimport.

        The check_dict contains: {1st_layer: (2nd_layer, shared_flag)} where shared_flag can be
        true or false to specify that 1st_layer and 2nd_layer must/must-not share their maskers
        respectively.
        """
        converted_layer_names = dict(new_nn.seed.named_modules())
        for layer_1, layer_2, shared_flag in check_rules:
            quantizer_1 = converted_layer_names[layer_1].mixprec_a_quantizer  # type: ignore
            quantizer_2 = converted_layer_names[layer_2].mixprec_a_quantizer  # type: ignore
            if shared_flag:
                msg = f"Layers {layer_1} and {layer_2} are expected to share a quantizer, but don't"
                self.assertEqual(quantizer_1, quantizer_2, msg)
            else:
                msg = f"Layers {layer_1} and {layer_2} are expected to have independent quantizers"
                self.assertNotEqual(quantizer_1, quantizer_2, msg)

    def _check_layers_exclusion(self, new_nn: MixPrec, excluded: Iterable[str]):
        """Check that layers in "excluded" have not be converted to PIT form"""
        converted_layer_names = dict(new_nn.seed.named_modules())
        for layer_name in excluded:
            layer = converted_layer_names[layer_name]
            # verify that the layer has not been converted to one of the NAS types
            self.assertNotIsInstance(type(layer), MixPrecModule,
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
