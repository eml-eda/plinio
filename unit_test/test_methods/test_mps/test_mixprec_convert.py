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
from plinio.methods import MixPrec
from plinio.methods.mixprec.nn import MixPrec_Conv2d, MixPrecModule, MixPrecType, \
    MixPrec_Linear, MixPrec_Identity
import plinio.methods.mixprec.quant.nn as qnn
from unit_test.models import SimpleNN2D, DSCNN, ToyMultiPath1_2D, ToyMultiPath2_2D, \
    SimpleMixPrecNN, SimpleExportedNN2D, SimpleNN2D_NoBN, SimpleExportedNN2D_NoBias, \
    SimpleExportedNN2D_ch, SimpleExportedNN2D_NoBias_ch, ToyAdd_2D


class TestMixPrecConvert(unittest.TestCase):
    """Test conversion operations to/from nn.Module from/to MixPrec"""

    def test_autoimport_simple_layer(self):
        """Test the conversion of a simple sequential model with layer autoconversion
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = SimpleNN2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=3)

    def test_autoimport_simple_channel(self):
        """Test the conversion of a simple sequential model with layer autoconversion
        with PER_CHANNEL weight mixed-precision"""
        nn_ut = SimpleNN2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                         weight_precisions=(0, 2, 4, 8),
                         w_mixprec_type=MixPrecType.PER_CHANNEL)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=3)

    def test_autoimport_lastfc_zero_removal(self):
        """Test the removal of the 0 precision search for the last fc layer"""
        nn_ut = SimpleNN2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                         weight_precisions=(0, 2, 4, 8),
                         w_mixprec_type=MixPrecType.PER_CHANNEL)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=3)
        fc_prec = cast(MixPrec_Linear, new_nn.seed.fc).mixprec_w_quantizer.precisions
        self.assertTrue(0 not in fc_prec, '0 prec not removed by last fc layer')

    def test_autoimport_inp_quant_insertion(self):
        """Test the insertion of the input quantizer"""
        nn_ut = SimpleNN2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                         weight_precisions=(0, 2, 4, 8),
                         w_mixprec_type=MixPrecType.PER_CHANNEL)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=3)
        self.assertTrue(hasattr(new_nn.seed, 'input_quantizer'), 'inp quantizer not inserted')
        inp_quantizer = getattr(new_nn.seed, 'input_quantizer')
        msg = f'inp quantizer is of type {type(inp_quantizer)} instead of MixPrec_Identity'
        self.assertTrue(isinstance(inp_quantizer, MixPrec_Identity), msg)

    def test_autoimport_depthwise_layer(self):
        """Test the conversion of a model with depthwise convolutions (cin=cout=groups)
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = DSCNN()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=14)

    def test_autoimport_depthwise_channel(self):
        """Test the conversion of a model with depthwise convolutions (cin=cout=groups)
        with PER_CHANNEL weight mixed-precision"""
        nn_ut = DSCNN()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                         weight_precisions=(0, 2, 4, 8),
                         w_mixprec_type=MixPrecType.PER_CHANNEL)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=14)
        # Check that depthwise layer shares quantizers with the previous conv layer
        shared_quantizer_rules = (
            ('depthwise1', 'inputlayer', True),
            ('depthwise2', 'conv1', True),
            ('depthwise3', 'conv2', True),
            ('depthwise4', 'conv3', True),
            ('depthwise2', 'depthwise4', False),  # two far aways layers must not share
        )
        self._check_shared_quantizers(new_nn, shared_quantizer_rules)

    def test_autoimport_multipath_layer(self):
        """Test the conversion of a toy model with multiple concat and add operations
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = ToyMultiPath1_2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=7)
        shared_quantizer_rules = (
            ('conv2', 'conv4', True),  # inputs to add must share the quantizer
            ('conv2', 'conv5', True),  # inputs to add must share the quantizer
            ('conv0', 'conv1', True),  # inputs to concat over the channels must share
            ('conv3', 'conv4', False),  # consecutive convs must not share
            ('conv0', 'conv5', False),  # two far aways layers must not share
        )
        self._check_shared_quantizers(new_nn, shared_quantizer_rules)

        # TODO: cat is not supported
        # nn_ut = ToyMultiPath2_2D()
        # new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        # self._compare_prepared(nn_ut, new_nn.seed)
        # self._check_target_layers(new_nn, exp_tgt=6)
        # shared_quantizer_rules = (
        #     ('conv0', 'conv1', True),   # inputs to add
        #     ('conv2', 'conv3', True),  # concat over channels
        # )
        # self._check_shared_quantizers(new_nn, shared_quantizer_rules)

    def test_autoimport_multipath_channel(self):
        """Test the conversion of a toy model with multiple concat and add operations
        with PER_CHANNEL weight mixed-precision"""
        nn_ut = ToyMultiPath1_2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                         weight_precisions=(0, 2, 4, 8),
                         w_mixprec_type=MixPrecType.PER_CHANNEL)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=7)
        shared_quantizer_rules = (
            ('conv2', 'conv4', True),  # inputs to add must share the quantizer
            ('conv2', 'conv5', True),  # inputs to add must share the quantizer
            ('conv0', 'conv1', True),  # inputs to concat over the channels must share
            ('conv3', 'conv4', False),  # consecutive convs must not share
            ('conv0', 'conv5', False),  # two far aways layers must not share
        )
        self._check_shared_quantizers(new_nn, shared_quantizer_rules)
        add_quant_prop_rules = (
            ('conv0', 'add_[conv0, conv1]_quant',
             True),  # input to sum and sum output must share
            ('conv4', 'add_2_[add_1, conv4]_quant',
             True),  # input to sum and sum output must share
            ('conv0', 'add_2_[add_1, conv4]_quant',
             False),  # two far aways layers must not share
        )
        self._check_add_quant_prop(new_nn, add_quant_prop_rules)

        # TODO: cat is not supported
        # nn_ut = ToyMultiPath2_2D()
        # new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
        #                  w_mixprec_type=MixPrecType.PER_CHANNEL)
        # self._compare_prepared(nn_ut, new_nn.seed)
        # self._check_target_layers(new_nn, exp_tgt=6)
        # shared_quantizer_rules = (
        #     ('conv0', 'conv1', True),   # inputs to add
        #     ('conv2', 'conv3', True),  # concat over channels
        # )
        # self._check_shared_quantizers(new_nn, shared_quantizer_rules)

    def test_autoimport_bias_quantizer_pointer(self):
        """Test that each bias quantizer points to the proper weight and act quantizers.
        The bias quantizer need info about act and weight quantization to proper set its
        scale-factor.
        PER_CHANNEL weight mixed-precision"""
        nn_ut = ToyMultiPath1_2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                         weight_precisions=(0, 2, 4, 8),
                         w_mixprec_type=MixPrecType.PER_CHANNEL)
        self._compare_prepared(nn_ut, new_nn.seed)
        self._check_target_layers(new_nn, exp_tgt=7)
        shared_quantizer_rules_a = (
            ('conv0.mixprec_b_quantizer', 'input_quantizer', True),
            ('conv5.mixprec_b_quantizer', 'add_[conv0, conv1]_quant', True),
            ('fc.mixprec_b_quantizer', 'add_2_[add_1, conv4]_quant', True),
            ('conv0.mixprec_b_quantizer', 'conv5.mixprec_b_quantizer',
             False),  # two far aways layers must not share
        )
        self._check_shared_quantizers(new_nn, shared_quantizer_rules_a, act_or_w='act')
        shared_quantizer_rules_w = (
            ('conv0.mixprec_b_quantizer', 'conv0', True),
            ('conv5.mixprec_b_quantizer', 'conv5', True),
            ('fc.mixprec_b_quantizer', 'fc', True),
            ('conv0.mixprec_b_quantizer', 'conv5.mixprec_b_quantizer',
             False),  # two far aways layers must not share
        )
        self._check_shared_quantizers(new_nn, shared_quantizer_rules_w, act_or_w='w')

    def test_exclude_types_simple_layer(self):
        """Test the conversion of a Toy model while excluding conv2d layers
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = ToyMultiPath1_2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape, exclude_types=(nn.Conv2d,))
        # excluding Conv2D, only the final FC should be converted to MixPrec format
        self._check_target_layers(new_nn, exp_tgt=1)
        excluded = ('conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5')
        self._check_layers_exclusion(new_nn, excluded)

    def test_exclude_types_simple_channel(self):
        """Test the conversion of a Toy model while excluding conv2d layers
        with PER_CHANNEL weight mixed-precision (default)"""
        nn_ut = ToyMultiPath1_2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                         weight_precisions=(0, 2, 4, 8),
                         w_mixprec_type=MixPrecType.PER_CHANNEL, exclude_types=(nn.Conv2d,))
        # excluding Conv2D, only the final FC should be converted to MixPrec format
        self._check_target_layers(new_nn, exp_tgt=1)
        excluded = ('conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5')
        self._check_layers_exclusion(new_nn, excluded)

    # TODO: not supported at the moment
    # def test_exclude_names_simple_layer(self):
    #     """Test the conversion of a Toy model while excluding layers by name
    #     with PER_LAYER weight mixed-precision (default)"""
    #     nn_ut = ToyMultiPath1_2D()
    #     excluded = ('conv0', 'conv4')
    #     new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape, exclude_names=excluded)
    #     # excluding conv0 and conv4, there are 5 convertible conv2d and linear layers left
    #     self._check_target_layers(new_nn, exp_tgt=5)
    #     self._check_layers_exclusion(new_nn, excluded)

    #     # TODO: the following part of the test fails during registration of input
    #     # quantizer in the specific case of this network that shares the input with
    #     # multiple layers.
    #     nn_ut = ToyMultiPath2_2D()
    #     new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape, exclude_names=excluded)
    #     # excluding conv0 and conv4, there are 4 convertible conv1d  and linear layers left
    #     self._check_target_layers(new_nn, exp_tgt=4)
    #     self._check_layers_exclusion(new_nn, excluded)

    # TODO: not supported at the moment
    # def test_exclude_names_simple_channel(self):
    #     """Test the conversion of a Toy model while excluding layers by name
    #     with PER_CHANNEL weight mixed-precision"""
    #     nn_ut = ToyMultiPath1_2D()
    #     excluded = ('conv0', 'conv4')
    #     new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
    #                      w_mixprec_type=MixPrecType.PER_CHANNEL, exclude_names=excluded)
    #     # excluding conv0 and conv4, there are 5 convertible conv2d and linear layers left
    #     self._check_target_layers(new_nn, exp_tgt=5)
    #     self._check_layers_exclusion(new_nn, excluded)

    #     # TODO: the following part of the test fails during registration of input
    #     # quantizer in the specific case of this network that shares the input with
    #     # multiple layers.
    #     nn_ut = ToyMultiPath2_2D()
    #     new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape, exclude_names=excluded)
    #     # excluding conv0 and conv4, there are 4 convertible conv1d  and linear layers left
    #     self._check_target_layers(new_nn, exp_tgt=4)
    #     self._check_layers_exclusion(new_nn, excluded)

    def test_import_simple_layer(self):
        """Test the conversion of a simple sequential model that already contains a MixPrec layer
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = SimpleMixPrecNN()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape)
        self._compare_prepared(nn_ut, new_nn.seed)
        # convert with autoconvert disabled. This is as if we exclude layers except the one already
        # in MixPrec form
        excluded = ('conv1')
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape, autoconvert_layers=False)
        self._compare_prepared(nn_ut, new_nn.seed, exclude_names=excluded)

    def test_import_simple_channel(self):
        """Test the conversion of a simple sequential model that already contains a MixPrec layer
        with PER_CHANNEL weight mixed-precision"""
        nn_ut = SimpleMixPrecNN()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                         w_mixprec_type=MixPrecType.PER_CHANNEL)
        self._compare_prepared(nn_ut, new_nn.seed)
        # convert with autoconvert disabled. This is as if we exclude layers except the one already
        # in MixPrec form
        excluded = ('conv1')
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape, autoconvert_layers=False)
        self._compare_prepared(nn_ut, new_nn.seed, exclude_names=excluded)

    def test_export_initial_simple_layer(self):
        """Test the export of a simple sequential model, just after import
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = SimpleNN2D()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                         activation_precisions=(8,), weight_precisions=(4, 8))
        exported_nn = new_nn.arch_export()
        expected_exported_nn = SimpleExportedNN2D()
        self._compare_exported(exported_nn, expected_exported_nn)

    # TODO: Not supported at the moment
    # def test_export_initial_simple_channel(self):
    #     """Test the export of a simple sequential model, just after import
    #     with PER_CHANNEL weight mixed-precision"""
    #     nn_ut = SimpleNN2D()
    #     new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
    #                      w_mixprec_type=MixPrecType.PER_CHANNEL,
    #                      activation_precisions=(8,), weight_precisions=(2, 4, 8))
    #     # Force selection of different precisions for different channels in the net
    #     new_alpha = [
    #         [1, 0, 0] * 10 + [0, 0],  # 10 ch
    #         [0, 1, 0] * 10 + [0, 0],  # 10 ch
    #         [0, 0, 1] * 10 + [1, 1]  # 12 ch
    #     ]
    #     new_alpha_t = nn.Parameter(torch.Tensor(new_alpha))
    #     conv0 = cast(MixPrec_Conv2d, new_nn.seed.conv0)
    #     conv0.mixprec_w_quantizer.alpha_prec = new_alpha_t
    #     # Force precision selection for the final linear layer
    #     new_alpha = [
    #         [0, 0, 0],  # 0 ch
    #         [0, 1, 0],  # 1 ch
    #         [1, 0, 1]  # 2 ch
    #     ]
    #     new_alpha_t = nn.Parameter(torch.Tensor(new_alpha))
    #     fc = cast(MixPrec_Conv2d, new_nn.seed.fc)
    #     fc.mixprec_w_quantizer.alpha_prec = new_alpha_t
    #     # Export
    #     exported_nn = new_nn.arch_export()
    #     expected_exported_nn = SimpleExportedNN2D_ch()
    #     self._compare_exported(exported_nn, expected_exported_nn)

    def test_export_initial_cuda_layer(self):
        """Test the export of a simple sequential model, just after import using
        GPU (if available) with PER_LAYER weight mixed-precision (default)"""
        # Check CUDA availability
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Training on:", device)
        nn_ut = SimpleNN2D().to(device)
        # Dummy inference
        with torch.no_grad():
            x = torch.rand((1,) + nn_ut.input_shape).to(device)
            nn_ut(x)
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                         activation_precisions=(8,), weight_precisions=(4, 8))
        new_nn = new_nn.to(device)
        # Dummy inference
        with torch.no_grad():
            new_nn(x)
        exported_nn = new_nn.arch_export()
        exported_nn = exported_nn.to(device)
        # Dummy inference
        with torch.no_grad():
            exported_nn(x)
        expected_exported_nn = SimpleExportedNN2D().to(device)
        self._compare_exported(exported_nn, expected_exported_nn)

    # TODO: Not supported at the moment
    # def test_export_initial_cuda_channel(self):
    #     """Test the export of a simple sequential model, just after import using
    #     GPU (if available) with PER_CHANNEL weight mixed-precision"""
    #     # Check CUDA availability
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     print("Training on:", device)
    #     nn_ut = SimpleNN2D().to(device)
    #     # Dummy inference
    #     with torch.no_grad():
    #         x = torch.rand((1,) + nn_ut.input_shape).to(device)
    #         nn_ut(x)
    #     new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
    #                      w_mixprec_type=MixPrecType.PER_CHANNEL,
    #                      activation_precisions=(8,), weight_precisions=(2, 4, 8))
    #     new_nn = new_nn.to(device)
    #     # Dummy inference
    #     with torch.no_grad():
    #         new_nn(x)
    #     # Force selection of different precisions for different channels in the net
    #     new_alpha = [
    #         [1, 0, 0] * 10 + [0, 0],  # 10 ch
    #         [0, 1, 0] * 10 + [0, 0],  # 10 ch
    #         [0, 0, 1] * 10 + [1, 1]  # 12 ch
    #     ]
    #     new_alpha_t = nn.Parameter(torch.tensor(new_alpha, device=device, dtype=torch.float))
    #     conv0 = cast(MixPrec_Conv2d, new_nn.seed.conv0)
    #     conv0.mixprec_w_quantizer.alpha_prec = new_alpha_t
    #     # Force precision selection for the final linear layer
    #     new_alpha = [
    #         [0, 0, 0],  # 0 ch
    #         [0, 1, 0],  # 1 ch
    #         [1, 0, 1]  # 2 ch
    #     ]
    #     new_alpha_t = nn.Parameter(torch.tensor(new_alpha, device=device, dtype=torch.float))
    #     fc = cast(MixPrec_Conv2d, new_nn.seed.fc)
    #     fc.mixprec_w_quantizer.alpha_prec = new_alpha_t
    #     # Export
    #     exported_nn = new_nn.arch_export().to(device)
    #     # Dummy inference
    #     with torch.no_grad():
    #         exported_nn(x)
    #     expected_exported_nn = SimpleExportedNN2D_ch().to(device)
    #     self._compare_exported(exported_nn, expected_exported_nn)

    def test_export_initial_nobn_layer(self):
        """Test the export of a simple sequential model with no bn, just after import
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = SimpleNN2D_NoBN()
        new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
                         activation_precisions=(8,), weight_precisions=(4, 8))
        exported_nn = new_nn.arch_export()
        expected_exported_nn = SimpleExportedNN2D_NoBias()
        self._compare_exported(exported_nn, expected_exported_nn)

    # TODO: Not supported at the moment
    # def test_export_initial_nobn_channel(self):
    #     """Test the export of a simple sequential model with no bn, just after import
    #     with PER_LAYER weight mixed-precision (default)"""
    #     nn_ut = SimpleNN2D_NoBN()
    #     new_nn = MixPrec(nn_ut, input_shape=nn_ut.input_shape,
    #                      activation_precisions=(8,), weight_precisions=(4, 8),
    #                      w_mixprec_type=MixPrecType.PER_CHANNEL)
    #     exported_nn = new_nn.arch_export()
    #     expected_exported_nn = SimpleExportedNN2D_NoBias_ch()
    #     self._compare_exported(exported_nn, expected_exported_nn)

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
                self.assertEqual(child.out_a_quantizer.clip_val, 5.,  # type: ignore
                                 "Wrong act qtz clip_val")
                self.assertEqual(child.w_precision, 2, "Wrong weight precision")
            if name == 'conv1':
                child = cast(qnn.Quant_Conv2d, child)
                # mixprec_child = cast(MixPrec_Conv2d, new_nn.seed._modules[name])
                self.assertEqual(child.a_precision, 4, "Wrong act precision")
                self.assertEqual(child.out_a_quantizer.clip_val, 1.,  # type: ignore
                                 "Wrong act qtz clip_val")
                self.assertEqual(child.w_precision, 8, "Wrong weight precision")

    def _check_target_layers(self, new_nn: MixPrec, exp_tgt: int):
        """Check if number of target layers is as expected"""
        n_tgt = len(new_nn._target_layers)
        self.assertEqual(exp_tgt, n_tgt,
                         "Expected {} target layers, but found {}".format(exp_tgt, n_tgt))

    def _check_add_quant_prop(self,
                              new_nn: MixPrec,
                              check_rules: Iterable[Tuple[str, str, bool]]):
        """Check if add quantizers are correctly propaated during an autoimport.

        The check_dict contains: {1st_layer: (2nd_layer, shared_flag)} where shared_flag can be
        true or false to specify that 1st_layer and 2nd_layer must/must-not share their maskers
        respectively.
        """
        converted_layer_names = dict(new_nn.seed.named_modules())
        for layer_1, layer_2, shared_flag in check_rules:
            quantizer_1_a = converted_layer_names[layer_1].mixprec_a_quantizer  # type: ignore
            quantizer_2_a = converted_layer_names[layer_2].mixprec_a_quantizer  # type: ignore
            if shared_flag:
                msg = f"Layers {layer_1} and {layer_2} are expected to share "
                msg_a = msg + "act quantizer, but don't"
                self.assertEqual(quantizer_1_a, quantizer_2_a, msg_a)
            else:
                msg = f"Layers {layer_1} and {layer_2} are expected to have independent "
                msg_a = msg + "act quantizers"
                self.assertNotEqual(quantizer_1_a, quantizer_2_a, msg_a)

    def _check_shared_quantizers(self,
                                 new_nn: MixPrec,
                                 check_rules: Iterable[Tuple[str, str, bool]],
                                 act_or_w: str = 'both'):
        """Check if shared quantizers are set correctly during an autoimport.

        The check_dict contains: {1st_layer: (2nd_layer, shared_flag)} where shared_flag can be
        true or false to specify that 1st_layer and 2nd_layer must/must-not share their maskers
        respectively.
        """
        converted_layer_names = dict(new_nn.seed.named_modules())
        for layer_1, layer_2, shared_flag in check_rules:
            quantizer_1_a, quantizer_2_a = None, None
            quantizer_1_w, quantizer_2_w = None, None
            if act_or_w == 'act' or act_or_w == 'both':
                quantizer_1_a = converted_layer_names[layer_1].mixprec_a_quantizer  # type: ignore
                quantizer_2_a = converted_layer_names[layer_2].mixprec_a_quantizer  # type: ignore
            if act_or_w == 'w' or act_or_w == 'both':
                quantizer_1_w = converted_layer_names[layer_1].mixprec_w_quantizer  # type: ignore
                quantizer_2_w = converted_layer_names[layer_2].mixprec_w_quantizer  # type: ignore
            if shared_flag:
                msg = f"Layers {layer_1} and {layer_2} are expected to share "
                if act_or_w == 'act' or act_or_w == 'both':
                    msg_a = msg + "act quantizer, but don't"
                    self.assertEqual(quantizer_1_a, quantizer_2_a, msg_a)
                if act_or_w == 'w' or act_or_w == 'both':
                    msg_w = msg + "weight quantizer, but don't"
                    self.assertEqual(quantizer_1_w, quantizer_2_w, msg_w)
            else:
                msg = f"Layers {layer_1} and {layer_2} are expected to have independent "
                if act_or_w == 'act' or act_or_w == 'both':
                    msg_a = msg + "act quantizers"
                    self.assertNotEqual(quantizer_1_a, quantizer_2_a, msg_a)
                if act_or_w == 'w' or act_or_w == 'both':
                    msg_w = msg + "weight quantizers"
                    self.assertNotEqual(quantizer_1_w, quantizer_2_w, msg_w)

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

    def _compare_exported(self, exported_mod: nn.Module, expected_mod: nn.Module):
        """Compare two nn.Modules, where one has been imported and exported by MixPrec"""
        for name, child in expected_mod.named_children():
            new_child = cast(nn.Module, exported_mod._modules[name])
            # self._compare_exported(child, new_child)
            if isinstance(child, qnn.Quant_Conv2d):
                self._check_conv2d(child, new_child)
            if isinstance(child, qnn.Quant_Linear):
                self._check_linear(child, new_child)
            if isinstance(child, qnn.Quant_List):
                self.assertIsInstance(new_child, qnn.Quant_List, "Wrong layer type")
                new_child = cast(qnn.Quant_List, new_child)
                for layer, new_layer in zip(child, new_child):
                    if isinstance(layer, qnn.Quant_Conv2d):
                        self._check_conv2d(layer, new_layer)
                    if isinstance(layer, qnn.Quant_Linear):
                        self._check_linear(layer, new_layer)

    def _check_conv2d(self, child, new_child):
        """Collection of checks on Quant_Conv2d"""
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

    def _check_linear(self, child, new_child):
        """Collection of checks on Quant_Linear"""
        self.assertIsInstance(new_child, qnn.Quant_Linear, "Wrong layer type")
        # Check layer geometry
        self.assertTrue(child.in_features == new_child.in_features)
        self.assertTrue(child.out_features == new_child.out_features)
        # Check qtz param
        if type(new_child.out_a_quantizer) == nn.Identity:
            self.assertTrue(new_child.a_precision == 'float')
        else:
            self.assertTrue(child.a_precision == new_child.a_precision)
        self.assertTrue(child.w_precision == new_child.w_precision)

    def test_repeated_precisions(self):
        """Check that if the weights or the activation precisions used for the model's
        initialization contain duplicates then an exception is raised"""
        net = ToyAdd_2D()
        input_shape = net.input_shape

        prec = (2, 4, 8)
        repeated_prec = (0, 2, 0, 8, 4, 4)

        # case (1): the mixed-precision scheme for the weigths is PER_CHANNEL
        with self.assertRaises(ValueError):
            MixPrec(net,
                    input_shape=input_shape,
                    regularizer='macs',
                    activation_precisions=prec,
                    weight_precisions=repeated_prec,
                    w_mixprec_type=MixPrecType.PER_CHANNEL)

        with self.assertRaises(ValueError):
            MixPrec(net,
                    input_shape=input_shape,
                    regularizer='macs',
                    activation_precisions=repeated_prec,
                    weight_precisions=prec,
                    w_mixprec_type=MixPrecType.PER_CHANNEL)

        # case (2): the mixed-precision scheme for the weigths is PER_LAYER
        with self.assertRaises(ValueError):
            MixPrec(net,
                    input_shape=input_shape,
                    regularizer='macs',
                    activation_precisions=prec,
                    weight_precisions=repeated_prec,
                    w_mixprec_type=MixPrecType.PER_LAYER)

        with self.assertRaises(ValueError):
            MixPrec(net,
                    input_shape=input_shape,
                    regularizer='macs',
                    activation_precisions=repeated_prec,
                    weight_precisions=prec,
                    w_mixprec_type=MixPrecType.PER_LAYER)

    def test_out_features_eff(self):
        """Check whether out_features_eff returns the correct number of not pruned channels"""
        net = ToyAdd_2D()
        input_shape = net.input_shape
        a_prec = (2, 4, 8)
        alpha_prec = torch.zeros(4, 10)
        alpha_prec[torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 3]),
                   torch.tensor([3, 5, 9, 0, 1, 2, 7, 8, 6, 4])] = 1
        x = torch.rand(input_shape).unsqueeze(0)
        # Use the following alpha_prec matrix to check the sanity of out_features_eff
        # for one specific layer
        # [[0., 0., 0., 1., 0., 1., 0., 0., 0., 1.],
        #  [1., 1., 1., 0., 0., 0., 0., 1., 1., 0.],
        #  [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        #  [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])

        # case (1): zero_index = 0
        w_prec = (0, 2, 4, 8)
        mixprec_net = MixPrec(net,
                              input_shape=input_shape,
                              activation_precisions=a_prec,
                              weight_precisions=w_prec,
                              w_mixprec_type=MixPrecType.PER_CHANNEL,
                              hard_softmax=True)
        for layer in mixprec_net._target_layers:  # force sampling of 8-bit precision
            if isinstance(layer, MixPrec_Conv2d) or isinstance(layer, MixPrec_Linear):
                alpha_no_0bit = torch.zeros(layer.mixprec_w_quantizer.alpha_prec.shape)
                alpha_no_0bit[-1, :] = 1
                layer.mixprec_w_quantizer.alpha_prec.data = alpha_no_0bit
        conv1 = cast(MixPrec_Conv2d, mixprec_net.seed.conv1)
        conv1.mixprec_w_quantizer.alpha_prec.data = alpha_prec  # update conv1 layer's alpha_prec
        mixprec_net(x)  # perform a forward pass to update the out_features_eff values
        self.assertEqual(conv1.out_features_eff.item(), 7)

        # case (2): zero_index = 2
        w_prec = (2, 4, 0, 8)
        mixprec_net = MixPrec(net,
                              input_shape=input_shape,
                              activation_precisions=a_prec,
                              weight_precisions=w_prec,
                              w_mixprec_type=MixPrecType.PER_CHANNEL,
                              hard_softmax=True)
        for layer in mixprec_net._target_layers:  # force sampling of 8-bit precision
            if isinstance(layer, MixPrec_Conv2d) or isinstance(layer, MixPrec_Linear):
                alpha_no_0bit = torch.zeros(layer.mixprec_w_quantizer.alpha_prec.shape)
                alpha_no_0bit[-1, :] = 1
                layer.mixprec_w_quantizer.alpha_prec.data = alpha_no_0bit
        conv1 = cast(MixPrec_Conv2d, mixprec_net.seed.conv1)
        conv1.mixprec_w_quantizer.alpha_prec.data = alpha_prec  # update conv1 layer's alpha_prec
        mixprec_net(x)  # perform a forward pass to update the out_features_eff values
        self.assertEqual(conv1.out_features_eff.item(), 9)

    def test_input_features_calculator(self):
        """Check whether input_features_calculator returns the correct number of input channels,
        both in the case where 0-bit precision is allowed and in the case where it is not"""
        net = SimpleNN2D()
        input_shape = net.input_shape
        x = torch.rand(input_shape).unsqueeze(0)
        a_prec = (2, 4, 8)

        # case (1): no 0-bit precision, i.e. no pruning allowed
        w_prec = (2, 4, 8)
        mixprec_net = MixPrec(net,
                              input_shape=input_shape,
                              activation_precisions=a_prec,
                              weight_precisions=w_prec,
                              w_mixprec_type=MixPrecType.PER_CHANNEL,
                              hard_softmax=True)
        mixprec_net(x)

        conv0 = cast(MixPrec_Conv2d, mixprec_net.seed.conv0)
        self.assertEqual(conv0.input_features_calculator.features.item(), input_shape[0])

        conv1 = cast(MixPrec_Conv2d, mixprec_net.seed.conv1)
        self.assertEqual(conv1.input_features_calculator.features.item(),
                         conv0.out_features_eff.item())

        fc = cast(MixPrec_Linear, mixprec_net.seed.fc)
        self.assertEqual(fc.input_features_calculator.features.item(),
                         conv1.out_features_eff.item() * 10 * 10)

        # case(2): 0-bit precision with some channels pruned
        w_prec = (0, 2, 4, 8)
        mixprec_net = MixPrec(net,
                              input_shape=input_shape,
                              activation_precisions=a_prec,
                              weight_precisions=w_prec,
                              w_mixprec_type=MixPrecType.PER_CHANNEL,
                              hard_softmax=True)

        for layer in mixprec_net._target_layers:  # force sampling of 8-bit precision
            if isinstance(layer, MixPrec_Conv2d) or isinstance(layer, MixPrec_Linear):
                alpha_no_0bit = torch.zeros(layer.mixprec_w_quantizer.alpha_prec.shape)
                alpha_no_0bit[-1, :] = 1
                layer.mixprec_w_quantizer.alpha_prec.data = alpha_no_0bit
        # prune one channel of conv1 layer
        conv1 = cast(MixPrec_Conv2d, mixprec_net.seed.conv1)
        conv1.mixprec_w_quantizer.alpha_prec.data[0, 2] = 1
        conv1.mixprec_w_quantizer.alpha_prec.data[-1, 2] = 0
        mixprec_net(x)

        fc = cast(MixPrec_Linear, mixprec_net.seed.fc)
        self.assertEqual(fc.input_features_calculator.features.item(),
                         conv1.out_features_eff.item() * 10 * 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
