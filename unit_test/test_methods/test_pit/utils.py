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
from typing import Tuple, Iterable, cast, Type, Dict
import unittest
import math
import random
import torch
import torch.nn as nn
from plinio.methods import PIT
from plinio.methods.pit.nn import PITModule, PITConv1d, PITConv2d, PITConv3d, PITLinear
from plinio.methods.pit.nn.features_masker import PITFrozenFeaturesMasker


def check_output_equal(test: unittest.TestCase, orig_nn: nn.Module, pit_nn: PIT,
                       input_shape: Tuple[int, ...], iterations=10):
    """Verify that a model and a PIT model produce the same output given the same input"""
    orig_nn.eval()
    pit_nn.eval()
    for _ in range(iterations):
        # add batch size in front
        x = torch.rand((32,) + input_shape)
        y = orig_nn(x)
        pit_y = pit_nn(x)
        test.assertTrue(torch.allclose(y, pit_y, atol=1e-7),
                        "Wrong output of PIT model")


def check_batchnorm_folding(test: unittest.TestCase, original_mod: nn.Module, pit_seed: nn.Module):
    """Compare two nn.Modules, where one has been imported and exported by PIT
    to verify batchnorm folding"""
    for name, child in original_mod.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            test.assertTrue(name not in pit_seed._modules,
                            f"BatchNorm {name} not folder")
    for name, child in pit_seed.named_children():
        test.assertFalse(isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)),
                         f"Found BatchNorm {name} in converted module")


def check_batchnorm_memory(test: unittest.TestCase, pit_seed: nn.Module, layers: Iterable[str]):
    """Check that, in a PIT converted model, PIT layers that were originally followed
    by BatchNorm have saved internally the BN information for restoring it later"""
    for name, child in pit_seed.named_children():
        if isinstance(child, PITModule) and name in layers:
            test.assertTrue(child.bn is not None)


def check_batchnorm_unfolding(test: unittest.TestCase, pit_seed: nn.Module,
                              exported_mod: nn.Module):
    """Check that, in a PIT converted model, PIT layers that were originally followed
    by BatchNorm have saved internally the BN information for restoring it later"""
    for name, child in pit_seed.named_children():
        if isinstance(child, PITModule) and child.bn is not None:
            bn_name = name + "_exported_bn"
            test.assertTrue(bn_name in exported_mod._modules)
            new_child = cast(nn.Module, exported_mod._modules[bn_name])
            if isinstance(child, (PITConv1d, PITLinear)):
                test.assertTrue(isinstance(new_child, nn.BatchNorm1d))
            if isinstance(child, (PITConv2d)):
                test.assertTrue(isinstance(new_child, nn.BatchNorm2d))


def compare_prepared(test: unittest.TestCase,
                     old_mod: nn.Module, new_mod: nn.Module,
                     base_name: str = "",
                     exclude_names: Iterable[str] = (),
                     exclude_types: Tuple[Type[nn.Module], ...] = ()):
    """Compare a nn.Module and its PIT-converted version"""
    for name, child in old_mod.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm1d, nn.InstanceNorm2d)):
            # BN and Instance Norm cannot be compared due to folding
            continue
        new_child = cast(nn.Module, new_mod._modules[name])
        compare_prepared(test, child, new_child, base_name + name + ".", exclude_names,
                         exclude_types)
        if isinstance(child, nn.Conv1d):
            if (base_name + name not in exclude_names) and not isinstance(child, exclude_types):
                test.assertTrue(isinstance(new_child, PITConv1d),
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
        if isinstance(child, nn.Conv2d):
            if (base_name + name not in exclude_names) and not isinstance(child, exclude_types):
                test.assertTrue(isinstance(new_child, PITConv2d),
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
        if isinstance(child, nn.Conv3d):
            if (base_name + name not in exclude_names) and not isinstance(child, exclude_types):
                test.assertTrue(isinstance(new_child, PITConv3d),
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
                # TODO: add other layers


def compare_identical(test: unittest.TestCase, old_mod: nn.Module, new_mod: nn.Module):
    """Compare two nn.Modules, where one has been imported and exported by PIT"""
    for name, child in old_mod.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # BN cannot be compared due to folding
            continue
        new_child = cast(nn.Module, new_mod._modules[name])
        compare_identical(test, child, new_child)
        if isinstance(child, nn.Conv1d):
            test.assertIsInstance(new_child, nn.Conv1d, "Wrong layer type")
            test.assertTrue(child.in_channels == new_child.in_channels)
            test.assertTrue(child.out_channels == new_child.out_channels)
            test.assertTrue(child.kernel_size == new_child.kernel_size)
            test.assertTrue(child.stride == new_child.stride)
            test.assertTrue(child.padding == new_child.padding)
            test.assertTrue(child.dilation == new_child.dilation)
            test.assertTrue(child.groups == new_child.groups)
            test.assertTrue(child.padding_mode == new_child.padding_mode)
            # TODO: add other layers


def check_target_layers(test: unittest.TestCase, new_nn: PIT, exp_tgt: int,
                        unique: bool = False):
    """Check if number of converted layers is as expected"""
    if unique:
        n_tgt = len(
            [_ for _ in new_nn._unique_leaf_modules if isinstance(_[2], PITModule)])
    else:
        n_tgt = len(
            [_ for _ in new_nn._leaf_modules if isinstance(_[2], PITModule)])
    test.assertEqual(exp_tgt, n_tgt,
                     "Expected {} target layers, but found {}".format(exp_tgt, n_tgt))


def check_shared_maskers(test: unittest.TestCase, new_nn: PIT,
                         check_rules: Iterable[Tuple[str, str, bool]]):
    """Check if shared maskers are set correctly during an autoimport.

    check_rules contains: (1st_layer, 2nd_layer, shared_flag) where shared_flag can be
    true or false to specify that 1st_layer and 2nd_layer must/must-not share their maskers
    respectively.
    """
    converted_layer_names = dict(new_nn.seed.named_modules())
    for layer_1, layer_2, shared_flag in check_rules:
        # type: ignore
        masker_1 = converted_layer_names[layer_1].out_features_masker
        # type: ignore
        masker_2 = converted_layer_names[layer_2].out_features_masker
        if shared_flag:
            msg = f"Layers {layer_1} and {layer_2} are expected to share a masker, but don't"
            test.assertEqual(masker_1, masker_2, msg)
        else:
            msg = f"Layers {layer_1} and {layer_2} are expected to have independent maskers"
            test.assertNotEqual(masker_1, masker_2, msg)


def check_frozen_maskers(test: unittest.TestCase, new_nn: PIT,
                         check_rules: Iterable[Tuple[str, bool]]):
    """Check if frozen maskers are set correctly during an autoimport.

    check_rules contains: (layer_name, frozen_flag) where frozen_flag can be true or false to
    specify that the features masker for layer_name must/must-not be frozen
    """
    converted_layer_names = dict(new_nn.seed.named_modules())
    for layer, frozen_flag in check_rules:
        # type: ignore
        masker = converted_layer_names[layer].out_features_masker
        if frozen_flag:
            msg = f"Layers {layer} is expected to have a frozen channel masker, but hasn't"
            test.assertTrue(isinstance(masker, PITFrozenFeaturesMasker), msg)
        else:
            msg = f"Layers {layer} is expected to have an unfrozen features masker, but hasn't"
            test.assertFalse(isinstance(masker, PITFrozenFeaturesMasker), msg)


def check_layers_exclusion(test: unittest.TestCase, new_nn: PIT, excluded: Iterable[str]):
    """Check that layers in "excluded" have not be converted to PIT form"""
    converted_layer_names = dict(new_nn.seed.named_modules())
    for layer_name in excluded:
        layer = converted_layer_names[layer_name]
        # verify that the layer has not been converted to one of the NAS types
        test.assertNotIsInstance(type(layer), PITModule,
                                 f"Layer {layer_name} should not be converted")
        # additionally, verify that there is no channel_masker (al PIT layers have it)
        # this is probably redundant
        try:
            layer.__getattr__('out_channel_masker')
        except Exception:
            pass
        else:
            test.fail("Excluded layer has the output_channel_masker set")


def check_channel_mask_init(test: unittest.TestCase, pit_nn: PIT, check_layers: Tuple[str, ...]):
    """Check if the channel masks are initialized correctly"""
    converted_layer_names = dict(pit_nn.seed.named_modules())
    for layer_name in check_layers:
        layer = converted_layer_names[layer_name]
        if isinstance(layer, PITConv1d):
            alpha = layer.out_features_masker.alpha
            check = torch.ones((layer.out_channels,))
            test.assertTrue(torch.all(alpha == check), "Wrong alpha values")
            test.assertEqual(torch.sum(alpha), layer.out_channels, "Wrong alpha sum")
            test.assertEqual(torch.sum(alpha), layer.out_features_eff, "Wrong channels eff")
            test.assertEqual(torch.sum(alpha), layer.out_features_opt, "Wrong channels opt")


def check_rf_mask_init(test: unittest.TestCase, pit_nn: PIT, check_layers: Tuple[str, ...]):
    """Check if the RF masks are initialized correctly"""
    converted_layer_names = dict(pit_nn.seed.named_modules())
    for layer_name in check_layers:
        layer = converted_layer_names[layer_name]
        if isinstance(layer, PITConv1d):
            kernel_size = layer.kernel_size[0]
            beta = layer.timestep_masker.beta
            check = torch.ones((kernel_size,))
            test.assertTrue(torch.all(beta == check), "Wrong beta values")
            c_check = []
            for i in range(kernel_size):
                c_check.append([1] * (i + 1) + [0] * (kernel_size - i - 1))
            c_check = torch.tensor(c_check)
            c_beta = cast(torch.Tensor, layer.timestep_masker._c_beta)
            test.assertTrue(torch.all(c_beta == c_check), "Wrong C beta matrix")
            theta_beta = layer.timestep_masker.theta
            theta_check = torch.tensor(list(range(1, kernel_size + 1)))
            test.assertTrue(torch.all(theta_beta == theta_check), "Wrong theta beta array")


def check_dilation_mask_init(test: unittest.TestCase, pit_nn: PIT, check_layers: Tuple[str, ...]):
    """Check if the dilation masks are initialized correctly"""
    converted_layer_names = dict(pit_nn.seed.named_modules())
    for layer_name in check_layers:
        layer = converted_layer_names[layer_name]
        if isinstance(layer, PITConv1d):
            kernel_size = layer.kernel_size[0]
            rf_seed = (kernel_size - 1) * layer.dilation[0] + 1
            len_gamma_exp = math.ceil(math.log2(rf_seed))
            gamma = layer.dilation_masker.gamma
            check = torch.ones((len_gamma_exp,))
            test.assertTrue(torch.all(gamma == check), "Wrong gamma values")

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
            c_check = torch.fliplr(c_check)
            c_gamma = layer.dilation_masker._c_gamma
            test.assertTrue(torch.all(c_check == c_gamma), "Wrong C gamma matrix")

            c_theta = []
            for i in range(rf_seed):
                k_i = sum([1 - int(i % (2**p) == 0) for p in range(1, len_gamma_exp)])
                val = sum([check[len_gamma_exp - j] for j in range(1, len_gamma_exp - k_i + 1)])
                c_theta.append(val)
            c_theta = torch.tensor(c_theta)
            theta_gamma = layer.dilation_masker.theta
            test.assertTrue(torch.all(c_theta == theta_gamma), "Wrong theta gamma array")


def write_channel_mask(pit_nn: PIT, layer_name: str, mask: torch.Tensor):
    """Force a given value on the output channels mask"""
    converted_layer_names = dict(pit_nn.seed.named_modules())
    layer = converted_layer_names[layer_name]
    layer.out_features_masker.alpha = nn.Parameter(mask)  # type: ignore


def write_rf_mask(pit_nn: PIT, layer_name: str, mask: torch.Tensor):
    """Force a given value on the rf mask"""
    converted_layer_names = dict(pit_nn.seed.named_modules())
    layer = converted_layer_names[layer_name]
    layer.timestep_masker.beta = nn.Parameter(mask)  # type: ignore


def write_dilation_mask(pit_nn: PIT, layer_name: str, mask: torch.Tensor):
    """Force a given value on the dilation mask"""
    converted_layer_names = dict(pit_nn.seed.named_modules())
    layer = converted_layer_names[layer_name]
    layer.dilation_masker.gamma = nn.Parameter(mask)  # type: ignore


def read_channel_mask(pit_nn: PIT, layer_name: str) -> torch.Tensor:
    """Read a value from the output channels mask of a layer"""
    converted_layer_names = dict(pit_nn.seed.named_modules())
    layer = converted_layer_names[layer_name]
    return layer.out_features_masker.alpha  # type: ignore


def read_rf_mask(pit_nn: PIT, layer_name: str) -> torch.Tensor:
    """Read a value from the rf mask of a layer"""
    converted_layer_names = dict(pit_nn.seed.named_modules())
    layer = converted_layer_names[layer_name]
    return layer.timestep_masker.beta  # type: ignore


def read_dilation_mask(pit_nn: PIT, layer_name: str) -> torch.Tensor:
    """Read a value from the dilation mask of a layer"""
    converted_layer_names = dict(pit_nn.seed.named_modules())
    layer = converted_layer_names[layer_name]
    return layer.dilation_masker.gamma  # type: ignore


def check_input_features(test: unittest.TestCase, new_nn: PIT, input_features_dict: Dict[str, int]):
    """Check if the number of input features of each layer in a NAS-able model is as expected.

    input_features_dict is a dictionary containing: {layer_name, expected_input_features}
    """
    converted_layer_names = dict(new_nn.seed.named_modules())
    for name, exp in input_features_dict.items():
        layer = converted_layer_names[name]
        in_features = layer.input_features_calculator.features  # type: ignore
        test.assertEqual(in_features, exp,
                         f"Layer {name} has {in_features} input features, expected {exp}")


def rand_binary_channel_mask(max_n_channels: int) -> Tuple[torch.Tensor, int]:
    """Generate a random binary (0.0 or 1.0) mask of channels"""
    # randomize activation of N-1 channels (the last one is always kept-alive)
    cout = random.randint(1, max_n_channels - 1)
    alpha = torch.tensor([1.0] * cout + [0.0] * (max_n_channels - 1 - cout))
    alpha = alpha[torch.randperm(max_n_channels - 1)]
    alpha = torch.cat((alpha, torch.ones((1,))))
    return alpha, cout + 1
