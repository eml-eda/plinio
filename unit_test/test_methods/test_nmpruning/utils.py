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
# * Author:  Francesco Daghero <francesco.daghero@polito.it>                *
# *----------------------------------------------------------------------------*
from typing import cast, Iterable, Tuple, Type
import unittest
import torch.nn as nn
from plinio.methods import NMPruning
from plinio.methods.nm_pruning.nn import (
    NMPruningConv2d,
    NMPruningLinear,
    NMPruningModule,
)
import torch


def check_target_layers(
    test: unittest.TestCase, new_nn: NMPruning, exp_tgt: int, unique: bool = False
):
    """Check if number of target layers is as expected"""
    if unique:
        n_tgt = len(
            [
                _
                for _ in new_nn._unique_leaf_modules
                if isinstance(_[2], NMPruningModule)
            ]
        )
    else:
        n_tgt = len(
            [_ for _ in new_nn._leaf_modules if isinstance(_[2], NMPruningModule)]
        )
    test.assertEqual(
        exp_tgt, n_tgt, "Expected {} target layers, but found {}".format(exp_tgt, n_tgt)
    )


def check_layers_exclusion(
    test: unittest.TestCase, new_nn: NMPruning, excluded: Iterable[str]
):
    """Check that layers in "excluded" have not be converted to PIT form"""
    converted_layer_names = dict(new_nn.seed.named_modules())
    for layer_name in excluded:
        layer = converted_layer_names[layer_name]
        # verify that the layer has not been converted to one of the NAS types
        test.assertNotIsInstance(
            type(layer), NMPruningModule, f"Layer {layer_name} should not be converted"
        )


def compare_exported(
    test: unittest.TestCase, exported_mod: nn.Module, expected_mod: nn.Module
):
    """Compare two nn.Modules, where one has been imported and exported by NMPruning"""
    for name, child in expected_mod.named_children():
        new_child = cast(nn.Module, exported_mod._modules[name])
        # test._compare_exported(child, new_child)
        if isinstance(child, nn.Conv2d):
            check_conv2d(test, child, new_child)
        if isinstance(child, nn.Linear):
            check_linear(test, child, new_child)
        if isinstance(child, nn.ModuleList):
            test.assertIsInstance(new_child, nn.ModuleList, "Wrong layer type")
            new_child = cast(nn.ModuleList, new_child)
            for layer, new_layer in zip(child, new_child):
                if isinstance(layer, nn.Conv2d):
                    check_conv2d(test, layer, new_layer)
                if isinstance(layer, nn.Linear):
                    check_linear(test, layer, new_layer)


def check_conv2d(test: unittest.TestCase, child, new_child):
    """Collection of checks on Conv2d"""
    test.assertIsInstance(new_child, nn.Conv2d, "Wrong layer type")
    # Check layer geometry
    test.assertTrue(child.in_channels == new_child.in_channels)
    test.assertTrue(child.out_channels == new_child.out_channels)
    test.assertTrue(child.kernel_size == new_child.kernel_size)
    test.assertTrue(child.stride == new_child.stride)
    test.assertTrue(child.padding == new_child.padding)
    test.assertTrue(child.dilation == new_child.dilation)
    test.assertTrue(child.groups == new_child.groups)
    test.assertTrue(child.padding_mode == new_child.padding_mode)


def check_linear(test: unittest.TestCase, child, new_child):
    """Collection of checks on QuantLinear"""
    test.assertIsInstance(new_child, nn.Linear, "Wrong layer type")
    # Check layer geometry
    test.assertTrue(child.in_features == new_child.in_features)
    test.assertTrue(child.out_features == new_child.out_features)


def check_sparsity_conv2d(
    test: unittest.TestCase, exp_mod, n: int, m: int
):
    """Check sparsity of all Conv2d layers"""
    for name, child in exp_mod.named_children():
        print(isinstance(child, nn.Conv2d),child)
        if isinstance(child, nn.Conv2d) and NMPruningConv2d.is_prunable(child, n, m):
            print("Checking sparsity of", name)
            w = child.weight.permute(0, 2, 3, 1) # NCHW -> NHWC
            # Assert that the N:M sparsity is correct
            k, _, _, c = w.shape
            rw = w.reshape(k, -1, m).reshape(-1, m)
            n_nonzero = torch.count_nonzero(rw, dim=-1)
            count = n * torch.ones_like(n_nonzero)
            test.assertTrue(
                torch.equal(count, n_nonzero), "Wrong number of non-zero parameters"
            )

def check_sparsity_linear(
        test: unittest.TestCase, exp_mod, n: int, m: int
):
    """Check sparsity of all Linear layers"""
    for name, child in exp_mod.named_children():
        if isinstance(child, nn.Linear) and NMPruningLinear.is_prunable(child, n, m):
            w = child.weight.permute(1, 0)  # (in, out) -> (out, in)

            fout, fin = w.shape
            rw = w.reshape(fout, -1, m).reshape(-1, m)
            n_nonzero = torch.count_nonzero(rw, dim=-1)
            count = n * torch.ones_like(n_nonzero)
            test.assertTrue(
                torch.equal(count, n_nonzero), "Wrong number of non-zero parameters"
            )

