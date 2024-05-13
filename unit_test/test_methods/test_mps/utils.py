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
from typing import cast, Iterable, Tuple, Type
import unittest
import torch.nn as nn
from plinio.methods import MPS
from plinio.methods.mps.nn import MPSConv1d, MPSConv2d, MPSLinear, MPSModule
from plinio.methods.mps.quant.quantizers import DummyQuantizer
import plinio.methods.mps.quant.nn as qnn


def compare_prepared(test: unittest.TestCase,
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
        compare_prepared(test, child, new_child, base_name + name + ".",
                         exclude_names, exclude_types)
        if isinstance(child, nn.Conv2d):
            if (base_name + name not in exclude_names) and not isinstance(child, exclude_types):
                test.assertTrue(isinstance(new_child, MPSConv2d),
                                f"Layer {name} not converted")
                test.assertEqual(child.out_channels, new_child.out_channels,
                                 f"Layer {name} wrong output channels")
                test.assertEqual(child.kernel_size, new_child.kernel_size,
                                 f"Layer {name} wrong kernel size")
                test.assertEqual(child.dilation, new_child.dilation,
                                 f"Layer {name} wrong dilation")
                test.assertEqual(child.padding_mode, new_child.padding_mode,
                                 f"Layer {name} wrong padding mode")
                test.assertEqual(child.padding, new_child.padding,
                                 f"Layer {name} wrong padding")
                test.assertEqual(child.stride, new_child.stride,
                                 f"Layer {name} wrong stride")
                test.assertEqual(child.groups, new_child.groups,
                                 f"Layer {name} wrong groups")
        if isinstance(child, nn.Conv1d):
            if (base_name + name not in exclude_names) and not isinstance(child, exclude_types):
                test.assertTrue(isinstance(new_child, MPSConv1d),
                                f"Layer {name} not converted")
                test.assertEqual(child.out_channels, new_child.out_channels,
                                 f"Layer {name} wrong output channels")
                test.assertEqual(child.kernel_size, new_child.kernel_size,
                                 f"Layer {name} wrong kernel size")
                test.assertEqual(child.dilation, new_child.dilation,
                                 f"Layer {name} wrong dilation")
                test.assertEqual(child.padding_mode, new_child.padding_mode,
                                 f"Layer {name} wrong padding mode")
                test.assertEqual(child.padding, new_child.padding,
                                 f"Layer {name} wrong padding")
                test.assertEqual(child.stride, new_child.stride,
                                 f"Layer {name} wrong stride")
                test.assertEqual(child.groups, new_child.groups,
                                 f"Layer {name} wrong groups")
        if isinstance(child, nn.Linear):
            if (base_name + name not in exclude_names) and not isinstance(child, exclude_types):
                test.assertTrue(isinstance(new_child, MPSLinear),
                                f"Layer {name} not converted")
                test.assertEqual(child.out_features, new_child.out_features,
                                 f"Layer {name} wrong output features")
                # TODO: add other layers


def check_target_layers(test: unittest.TestCase, new_nn: MPS, exp_tgt: int, unique: bool = False):
    """Check if number of target layers is as expected"""
    if unique:
        n_tgt = len(
            [_ for _ in new_nn._unique_leaf_modules if isinstance(_[2], MPSModule)])
    else:
        n_tgt = len(
            [_ for _ in new_nn._leaf_modules if isinstance(_[2], MPSModule)])
    test.assertEqual(exp_tgt, n_tgt,
                     "Expected {} target layers, but found {}".format(exp_tgt, n_tgt))


def check_add_quant_prop(test: unittest.TestCase,
                         new_nn: MPS,
                         check_rules: Iterable[Tuple[str, str, bool]]):
    """Check if add quantizers are correctly propagated during an autoimport.

    The check_rules contains: {1st_layer: (2nd_layer, shared_flag)} where shared_flag can be
    true or false to specify that 1st_layer and 2nd_layer must/must-not share their maskers
    respectively.
    """
    converted_layer_names = dict(new_nn.seed.named_modules())
    for layer_1, layer_2, shared_flag in check_rules:
        quantizer_1_a = converted_layer_names[layer_1].out_mps_quantizer
        quantizer_2_a = converted_layer_names[layer_2].out_mps_quantizer
        if shared_flag:
            msg = f"Layers {layer_1} and {layer_2} are expected to share " + \
                "act quantizer, but don't"
            test.assertEqual(quantizer_1_a, quantizer_2_a, msg)
        else:
            msg = f"Layers {layer_1} and {layer_2} are expected to have independent " + \
                "act quantizers"
            test.assertNotEqual(quantizer_1_a, quantizer_2_a, msg)


def check_shared_quantizers(test: unittest.TestCase,
                            new_nn: MPS,
                            check_rules: Iterable[Tuple[str, str, bool]],
                            act_or_w: str = 'both'):
    """Check if shared quantizers are set correctly during an autoimport.

    The check_rules contains: {1st_layer: (2nd_layer, shared_flag)} where shared_flag can be
    true or false to specify that 1st_layer and 2nd_layer must/must-not share their
    quantizers respectively.
    """
    converted_layer_names = dict(new_nn.seed.named_modules())
    for layer_1, layer_2, shared_flag in check_rules:
        quantizer_1_a, quantizer_2_a = None, None
        quantizer_1_w, quantizer_2_w = None, None
        if act_or_w == 'act' or act_or_w == 'both':
            quantizer_1_a = converted_layer_names[layer_1].out_mps_quantizer
            quantizer_2_a = converted_layer_names[layer_2].out_mps_quantizer
        if act_or_w == 'w' or act_or_w == 'both':
            quantizer_1_w = converted_layer_names[layer_1].w_mps_quantizer
            quantizer_2_w = converted_layer_names[layer_2].w_mps_quantizer
        if shared_flag:
            msg = f"Layers {layer_1} and {layer_2} are expected to share "
            if act_or_w == 'act' or act_or_w == 'both':
                msg_a = msg + "act quantizer, but don't"
                test.assertEqual(quantizer_1_a, quantizer_2_a, msg_a)
            if act_or_w == 'w' or act_or_w == 'both':
                msg_w = msg + "weight quantizer, but don't"
                test.assertEqual(quantizer_1_w, quantizer_2_w, msg_w)
        else:
            msg = f"Layers {layer_1} and {layer_2} are expected to have independent "
            if act_or_w == 'act' or act_or_w == 'both':
                msg_a = msg + "act quantizers"
                test.assertNotEqual(quantizer_1_a, quantizer_2_a, msg_a)
            if act_or_w == 'w' or act_or_w == 'both':
                msg_w = msg + "weight quantizers"
                test.assertNotEqual(quantizer_1_w, quantizer_2_w, msg_w)


def check_layers_exclusion(test: unittest.TestCase, new_nn: MPS, excluded: Iterable[str]):
    """Check that layers in "excluded" have not be converted to PIT form"""
    converted_layer_names = dict(new_nn.seed.named_modules())
    for layer_name in excluded:
        layer = converted_layer_names[layer_name]
        # verify that the layer has not been converted to one of the NAS types
        test.assertNotIsInstance(type(layer), MPSModule,
                                 f"Layer {layer_name} should not be converted")


def compare_exported(test: unittest.TestCase, exported_mod: nn.Module, expected_mod: nn.Module):
    """Compare two nn.Modules, where one has been imported and exported by MPS"""
    for name, child in expected_mod.named_children():
        new_child = cast(nn.Module, exported_mod._modules[name])
        # test._compare_exported(child, new_child)
        if isinstance(child, qnn.QuantConv2d):
            check_conv2d(test, child, new_child)
        if isinstance(child, qnn.QuantLinear):
            check_linear(test, child, new_child)
        if isinstance(child, qnn.QuantList):
            test.assertIsInstance(new_child, qnn.QuantList, "Wrong layer type")
            new_child = cast(qnn.QuantList, new_child)
            for layer, new_layer in zip(child, new_child):
                if isinstance(layer, qnn.QuantConv2d):
                    check_conv2d(test, layer, new_layer)
                if isinstance(layer, qnn.QuantLinear):
                    check_linear(test, layer, new_layer)


def check_conv2d(test: unittest.TestCase, child, new_child):
    """Collection of checks on QuantConv2d"""
    test.assertIsInstance(new_child, qnn.QuantConv2d, "Wrong layer type")
    # Check layer geometry
    test.assertTrue(child.in_channels == new_child.in_channels)
    test.assertTrue(child.out_channels == new_child.out_channels)
    test.assertTrue(child.kernel_size == new_child.kernel_size)
    test.assertTrue(child.stride == new_child.stride)
    test.assertTrue(child.padding == new_child.padding)
    test.assertTrue(child.dilation == new_child.dilation)
    test.assertTrue(child.groups == new_child.groups)
    test.assertTrue(child.padding_mode == new_child.padding_mode)
    # Check qtz param
    test.assertTrue(child.out_quantizer.precision ==
                    new_child.out_quantizer.precision)
    test.assertTrue(child.w_quantizer.precision ==
                    new_child.w_quantizer.precision)


def check_linear(test: unittest.TestCase, child, new_child):
    """Collection of checks on QuantLinear"""
    test.assertIsInstance(new_child, qnn.QuantLinear, "Wrong layer type")
    # Check layer geometry
    test.assertTrue(child.in_features == new_child.in_features)
    test.assertTrue(child.out_features == new_child.out_features)
    # Check qtz param
    if type(new_child.out_quantizer) != DummyQuantizer:
        test.assertTrue(child.out_quantizer.precision ==
                        new_child.out_quantizer.precision)
    test.assertTrue(child.w_quantizer.precision ==
                    new_child.w_quantizer.precision)
