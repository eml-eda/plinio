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

from typing import cast
import unittest
import torch
import torch.nn as nn
from plinio.methods import MPS
from plinio.methods.mps import get_default_qinfo
from plinio.methods.mps.nn import MPSConv2d, MPSType, MPSLinear, MPSIdentity
from plinio.methods.mps.nn.qtz import MPSBaseQtz
import plinio.methods.mps.quant.nn as qnn
from plinio.methods.mps.quant.quantizers import FQWeight, MinMaxWeight, DummyQuantizer, PACTAct
from unit_test.models import SimpleNN2D, DSCNN, ToyMultiPath1_2D, ToyAdd_2D, \
    SimpleMPSNN, SimpleExportedNN2D, SimpleExportedNN2D_ch, SimpleNN2D_NoBN
from unit_test.test_methods.test_mps.utils import compare_prepared, \
        check_target_layers, check_shared_quantizers, check_add_quant_prop, \
        check_layers_exclusion, compare_exported


class TestMPSConvert(unittest.TestCase):
    """Test conversion operations to/from nn.Module from/to MPS"""

    def test_autoimport_simple_layer(self):
        """Test the conversion of a simple sequential model with layer autoconversion
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = SimpleNN2D()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=4)

    def test_autoimport_train_status_neutrality(self):
        """Test that the conversion does not change the training status of the original model"""
        # Training status is True
        nn_ut = SimpleNN2D()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape)
        self.assertTrue(new_nn.training, 'Training status changed after conversion')
        self.assertTrue(new_nn.seed.training, 'Training status changed after conversion within new_nn.seed')

        # Training status is False
        nn_ut = SimpleNN2D()
        nn_ut.eval()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape)
        self.assertFalse(new_nn.training, 'Training status changed after conversion')
        self.assertFalse(new_nn.seed.training, 'Training status changed after conversion within new_nn.seed')

    def test_autoimport_simple_channel(self):
        """Test the conversion of a simple sequential model with layer autoconversion
        with PER_CHANNEL weight mixed-precision"""
        nn_ut = SimpleNN2D()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     qinfo=get_default_qinfo(w_precision=(0, 2, 4, 8)),
                     w_search_type=MPSType.PER_CHANNEL)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=4)

    def test_autoimport_lastfc_zero_removal(self):
        """Test the removal of the 0 precision search for the last fc layer"""
        nn_ut = SimpleNN2D()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     qinfo=get_default_qinfo(w_precision=(0, 2, 4, 8)),
                     w_search_type=MPSType.PER_CHANNEL)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=4)
        fc_prec = cast(MPSLinear, new_nn.seed.fc).w_mps_quantizer.precision
        self.assertTrue(0 not in fc_prec, '0 prec not removed by last fc layer')

    def test_autoimport_inp_quant_insertion(self):
        """Test the insertion of the input quantizer"""
        nn_ut = SimpleNN2D()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     qinfo=get_default_qinfo(w_precision=(0, 2, 4, 8)),
                     w_search_type=MPSType.PER_CHANNEL)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=4)
        self.assertTrue(hasattr(new_nn.seed, 'x_input_quantizer'), 'inp quantizer not inserted')
        inp_quantizer = getattr(new_nn.seed, 'x_input_quantizer')
        msg = f'inp quantizer is of type {type(inp_quantizer)} instead of MPSIdentity'
        self.assertTrue(isinstance(inp_quantizer, MPSIdentity), msg)

    def test_autoimport_depthwise_layer(self):
        """Test the conversion of a model with depthwise convolutions (cin=cout=groups)
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = DSCNN()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=15)

    def test_autoimport_depthwise_channel(self):
        """Test the conversion of a model with depthwise convolutions (cin=cout=groups)
        with PER_CHANNEL weight mixed-precision"""
        nn_ut = DSCNN()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     qinfo=get_default_qinfo(w_precision=(0, 2, 4, 8)),
                     w_search_type=MPSType.PER_CHANNEL)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=15)
        # Check that depthwise layer shares quantizers with the previous conv layer
        shared_quantizer_rules = (
            ('depthwise1', 'inputlayer', True),
            ('depthwise2', 'conv1', True),
            ('depthwise3', 'conv2', True),
            ('depthwise4', 'conv3', True),
            ('depthwise2', 'depthwise4', False),  # two far aways layers must not share
        )
        check_shared_quantizers(self, new_nn, shared_quantizer_rules)

    def test_autoimport_multipath_layer(self):
        """Test the conversion of a toy model with multiple concat and add operations
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = ToyMultiPath1_2D()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=11)
        shared_quantizer_rules = (
            ('conv2', 'conv4', True),  # inputs to add must share the quantizer
            ('conv2', 'conv5', True),  # inputs to add must share the quantizer
            ('conv0', 'conv1', True),  # inputs to concat over the channels must share
            ('conv3', 'conv4', False),  # consecutive convs must not share
            ('conv0', 'conv5', False),  # two far aways layers must not share
        )
        check_shared_quantizers(self, new_nn, shared_quantizer_rules)

    def test_autoimport_multipath_channel(self):
        """Test the conversion of a toy model with multiple concat and add operations
        with PER_CHANNEL weight mixed-precision"""
        nn_ut = ToyMultiPath1_2D()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     qinfo=get_default_qinfo(w_precision=(0, 2, 4, 8)),
                     w_search_type=MPSType.PER_CHANNEL)
        compare_prepared(self, nn_ut, new_nn.seed)
        check_target_layers(self, new_nn, exp_tgt=11)
        shared_quantizer_rules = (
            ('conv2', 'conv4', True),  # inputs to add must share the quantizer
            ('conv2', 'conv5', True),  # inputs to add must share the quantizer
            ('conv0', 'conv1', True),  # inputs to concat over the channels must share
            ('conv3', 'conv4', False),  # consecutive convs must not share
            ('conv0', 'conv5', False),  # two far aways layers must not share
        )
        check_shared_quantizers(self, new_nn, shared_quantizer_rules)
        add_quant_prop_rules = (
            ('conv0', 'add_conv0_conv1_quant',
             True),  # input to sum and sum output must share
            ('conv4', 'add_add_1_conv4_quant',
             True),  # inputo s and sum output must share
            ('conv0', 'add_add_1_conv4_quant',
             False),  # two far aways layers must not share
        )
        check_add_quant_prop(self, new_nn, add_quant_prop_rules)

    def test_exclude_types_simple_layer(self):
        """Test the conversion of a Toy model while excluding conv2d layers
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = ToyMultiPath1_2D()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape, exclude_types=(nn.Conv2d,))
        # excluding Conv2D, we are left with: input quantizer (MPSIdentity), 3 MPSAdd, and one
        # MPSLinear
        check_target_layers(self, new_nn, exp_tgt=5)
        excluded = ('conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5')
        check_layers_exclusion(self, new_nn, excluded)

    def test_exclude_types_simple_channel(self):
        """Test the conversion of a Toy model while excluding conv2d layers
        with PER_CHANNEL weight mixed-precision (default)"""
        nn_ut = ToyMultiPath1_2D()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     qinfo=get_default_qinfo(w_precision=(0, 2, 4, 8)),
                     w_search_type=MPSType.PER_CHANNEL, exclude_types=(nn.Conv2d,))
        # excluding Conv2D, we are left with: input quantizer (MPSIdentity), 3 MPSAdd, and one
        # MPSLinear
        check_target_layers(self, new_nn, exp_tgt=5)
        excluded = ('conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5')
        check_layers_exclusion(self, new_nn, excluded)

    def test_exclude_names_simple_layer(self):
        """Test the conversion of a Toy model while excluding layers by name
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = ToyMultiPath1_2D()
        excluded = ('conv0', 'conv4')
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape, exclude_names=excluded)
        # excluding conv0 and conv4, there are 4 convertible conv2d layers left,
        # 1 MPSLinear, 3 MPSAdd, and the input quantizer
        check_target_layers(self, new_nn, exp_tgt=9)
        check_layers_exclusion(self, new_nn, excluded)

    def test_exclude_names_simple_channel(self):
        """Test the conversion of a Toy model while excluding layers by name
        with PER_CHANNEL weight mixed-precision"""
        nn_ut = ToyMultiPath1_2D()
        excluded = ('conv0', 'conv4')
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     w_search_type=MPSType.PER_CHANNEL, exclude_names=excluded)
        # excluding conv0 and conv4, there are 4 convertible conv2d layers left,
        # 1 MPSLinear, 3 MPSAdd, and the input quantizer
        check_target_layers(self, new_nn, exp_tgt=9)
        check_layers_exclusion(self, new_nn, excluded)

    def test_import_simple_layer(self):
        """Test the conversion of a simple sequential model that already contains a MPS layer
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = SimpleMPSNN()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape)
        compare_prepared(self, nn_ut, new_nn.seed)
        # convert with autoimport disabled. This is as if we exclude layers except the one already
        # in MPS form
        excluded = ('conv1')
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape, autoconvert_layers=False)
        compare_prepared(self, nn_ut, new_nn.seed, exclude_names=excluded)

    def test_import_simple_channel(self):
        """Test the conversion of a simple sequential model that already contains a MPS layer
        with PER_CHANNEL weight mixed-precision"""
        nn_ut = SimpleMPSNN()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     w_search_type=MPSType.PER_CHANNEL)
        compare_prepared(self, nn_ut, new_nn.seed)
        # convert with autoimport disabled. This is as if we exclude layers except the one already
        # in MPS form
        excluded = ('conv1')
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape, autoconvert_layers=False)
        compare_prepared(self, nn_ut, new_nn.seed, exclude_names=excluded)

    def test_export_initial_simple_layer(self):
        """Test the export of a simple sequential model, just after import
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = SimpleNN2D()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     qinfo=get_default_qinfo(a_precision=(8,), w_precision=(4, 8)))
        exported_nn = new_nn.export()
        expected_exported_nn = SimpleExportedNN2D()
        compare_exported(self, exported_nn, expected_exported_nn)

    def test_export_initial_simple_channel(self):
        """test the export of a simple sequential model, just after import
        with per_channel weight mixed-precision"""
        nn_ut = SimpleNN2D()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     qinfo=get_default_qinfo(a_precision=(8,), w_precision=(2, 4, 8)),
                     w_search_type=MPSType.PER_CHANNEL)
        # force selection of different precision for different channels in the net
        new_alpha = [
            [1, 0, 0] * 10 + [0, 0],  # 10 ch
            [0, 1, 0] * 10 + [0, 0],  # 10 ch
            [0, 0, 1] * 10 + [1, 1]  # 12 ch
        ]
        new_alpha_t = nn.Parameter(torch.tensor(new_alpha, dtype=torch.float))
        conv0 = cast(MPSConv2d, new_nn.seed.conv0)
        conv0.w_mps_quantizer.alpha = new_alpha_t
        # force precision selection for the final linear layer
        new_alpha = [
            [0, 0, 0],  # 0 ch
            [0, 1, 0],  # 1 ch
            [1, 0, 1]  # 2 ch
        ]
        new_alpha_t = nn.Parameter(torch.tensor(new_alpha, dtype=torch.float))
        fc = cast(MPSLinear, new_nn.seed.fc)
        fc.w_mps_quantizer.alpha = new_alpha_t
        exported_nn = new_nn.export()
        expected_exported_nn = SimpleExportedNN2D_ch(bias=False)
        compare_exported(self, exported_nn, expected_exported_nn)

    def test_export_initial_cuda_layer(self):
        """Test the export of a simple sequential model, just after import using
        GPU (if available) with PER_LAYER weight mixed-precision (default)"""
        # Check CUDA availability
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nn_ut = SimpleNN2D().to(device)
        # Dummy inference
        with torch.no_grad():
            x = torch.rand((1,) + nn_ut.input_shape).to(device)
            nn_ut(x)
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     qinfo=get_default_qinfo(a_precision=(8,), w_precision=(4, 8)))
        new_nn = new_nn.to(device)
        # Dummy inference
        with torch.no_grad():
            new_nn(x)
        exported_nn = new_nn.export()
        exported_nn = exported_nn.to(device)
        # Dummy inference
        with torch.no_grad():
            exported_nn(x)
        expected_exported_nn = SimpleExportedNN2D().to(device)
        compare_exported(self, exported_nn, expected_exported_nn)

    def test_export_initial_cuda_channel(self):
        """Test the export of a simple sequential model, just after import using
        GPU (if available) with PER_CHANNEL weight mixed-precision"""
        # Check CUDA availability
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Training on:", device)
        nn_ut = SimpleNN2D().to(device)
        x = torch.rand((1,) + nn_ut.input_shape).to(device)
        # dummy inference
        with torch.no_grad():
            nn_ut(x)
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     qinfo=get_default_qinfo(a_precision=(8,), w_precision=(2, 4, 8)),
                     w_search_type=MPSType.PER_CHANNEL)
        new_nn = new_nn.to(device)
        # dummy inference
        with torch.no_grad():
            new_nn(x)
        # force selection of different precision for different channels in the net
        new_alpha = [
            [1, 0, 0] * 10 + [0, 0],  # 10 ch
            [0, 1, 0] * 10 + [0, 0],  # 10 ch
            [0, 0, 1] * 10 + [1, 1]  # 12 ch
        ]
        new_alpha_t = nn.Parameter(torch.tensor(new_alpha, dtype=torch.float, device=device))
        conv0 = cast(MPSConv2d, new_nn.seed.conv0)
        conv0.w_mps_quantizer.alpha = new_alpha_t
        # force precision selection for the final linear layer
        new_alpha = [
            [0, 0, 0],  # 0 ch
            [0, 1, 0],  # 1 ch
            [1, 0, 1]  # 2 ch
        ]
        new_alpha_t = nn.Parameter(torch.tensor(new_alpha, dtype=torch.float, device=device))
        fc = cast(MPSLinear, new_nn.seed.fc)
        fc.w_mps_quantizer.alpha = new_alpha_t
        exported_nn = new_nn.export().to(device)
        # dummy inference
        with torch.no_grad():
            exported_nn(x)
        expected_exported_nn = SimpleExportedNN2D_ch().to(device)
        compare_exported(self, exported_nn, expected_exported_nn)

    def test_export_initial_nobn_layer(self):
        """Test the export of a simple sequential model with no bn, just after import
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = SimpleNN2D_NoBN()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     qinfo=get_default_qinfo(a_precision=(8,), w_precision=(4, 8)))
        exported_nn = new_nn.export()
        expected_exported_nn = SimpleExportedNN2D(bias=False)
        compare_exported(self, exported_nn, expected_exported_nn)

    def test_export_initial_nobn_channel(self):
        """Test the export of a simple sequential model with no bn, just after import
        with PER_LAYER weight mixed-precision (default)"""
        nn_ut = SimpleNN2D_NoBN()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape,
                     qinfo=get_default_qinfo(a_precision=(8,), w_precision=(2, 4, 8)),
                     w_search_type=MPSType.PER_CHANNEL)
        # force selection of different precision for different channels in the net
        new_alpha = [
            [1, 0, 0] * 10 + [0, 0],  # 10 ch
            [0, 1, 0] * 10 + [0, 0],  # 10 ch
            [0, 0, 1] * 10 + [1, 1]  # 12 ch
        ]
        new_alpha_t = nn.Parameter(torch.tensor(new_alpha, dtype=torch.float))
        conv0 = cast(MPSConv2d, new_nn.seed.conv0)
        conv0.w_mps_quantizer.alpha = new_alpha_t
        # force precision selection for the final linear layer
        new_alpha = [
            [0, 0, 0],  # 0 ch
            [0, 1, 0],  # 1 ch
            [1, 0, 1]  # 2 ch
        ]
        new_alpha_t = nn.Parameter(torch.tensor(new_alpha, dtype=torch.float))
        fc = cast(MPSLinear, new_nn.seed.fc)
        fc.w_mps_quantizer.alpha = new_alpha_t
        exported_nn = new_nn.export()
        expected_exported_nn = SimpleExportedNN2D_ch(bias=False)
        compare_exported(self, exported_nn, expected_exported_nn)

    def test_export_with_alpha(self):
        """Test the conversion of a simple model after forcing the nas/quant
        params values in some layers"""
        nn_ut = SimpleNN2D()
        new_nn = MPS(nn_ut, input_shape=nn_ut.input_shape)

        conv0 = cast(MPSConv2d, new_nn.seed.conv0)
        conv0.out_mps_quantizer.alpha = nn.parameter.Parameter(
            torch.tensor([0.3, 0.8, 0.99], dtype=torch.float))
        conv0.out_mps_quantizer.qtz_funcs[2].clip_val = nn.parameter.Parameter(
            torch.tensor([5.], dtype=torch.float))
        conv0.w_mps_quantizer.alpha = nn.parameter.Parameter(
            torch.tensor([1.5, 0.2, 1], dtype=torch.float))

        conv1 = cast(MPSConv2d, new_nn.seed.conv1)
        conv1.out_mps_quantizer.alpha = nn.parameter.Parameter(
            torch.tensor([0.3, 1.8, 0.99], dtype=torch.float))
        conv1.out_mps_quantizer.qtz_funcs[1].clip_val = nn.parameter.Parameter(
            torch.tensor([1.], dtype=torch.float))
        conv1.w_mps_quantizer.alpha = nn.parameter.Parameter(
            torch.tensor([1.5, 0.2, 1.9], dtype=torch.float))

        exported_nn = new_nn.export()
        # Dummy fwd to fill scale-factors values
        dummy_inp = torch.rand((2,) + nn_ut.input_shape)
        with torch.no_grad():
            exported_nn(dummy_inp)

        for name, child in exported_nn.named_children():
            if name == 'conv0':
                child = cast(qnn.QuantConv2d, child)
                self.assertEqual(child.out_quantizer.precision, 8, "Wrong act precision")
                self.assertEqual(child.out_quantizer.clip_val, 5., "Wrong act qtz clip_val")
                self.assertEqual(child.w_quantizer.precision, 2, "Wrong weight precision")
            if name == 'conv1':
                child = cast(qnn.QuantConv2d, child)
                self.assertEqual(child.out_quantizer.precision, 4, "Wrong act precision")
                self.assertEqual(child.out_quantizer.clip_val, 1.,  # type: ignore
                                 "Wrong act qtz clip_val")
                self.assertEqual(child.w_quantizer.precision, 8, "Wrong weight precision")

    def test_repeated_precision(self):
        """Check that if the weights or the activation precision used for the model's
        initialization contain duplicates then an exception is raised"""
        net = ToyAdd_2D()
        input_shape = net.input_shape

        prec = (2, 4, 8)
        repeated_prec = (0, 2, 0, 8, 4, 4)

        # case (1): the mixed-precision scheme for the weigths is PER_CHANNEL
        with self.assertRaises(ValueError):
            _ = MPS(net,
                    input_shape=input_shape,
                    qinfo=get_default_qinfo(a_precision=prec, w_precision=repeated_prec),
                    w_search_type=MPSType.PER_CHANNEL)

        with self.assertRaises(ValueError):
            MPS(net,
                input_shape=input_shape,
                qinfo=get_default_qinfo(a_precision=repeated_prec, w_precision=prec),
                w_search_type=MPSType.PER_CHANNEL)

        # case (2): the mixed-precision scheme for the weigths is PER_LAYER
        with self.assertRaises(ValueError):
            MPS(net,
                input_shape=input_shape,
                qinfo=get_default_qinfo(a_precision=prec, w_precision=repeated_prec),
                w_search_type=MPSType.PER_LAYER)

        with self.assertRaises(ValueError):
            MPS(net,
                input_shape=input_shape,
                qinfo=get_default_qinfo(a_precision=repeated_prec, w_precision=prec),
                w_search_type=MPSType.PER_LAYER)

    def test_out_features_eff(self):
        """Check whether out_features_eff returns the correct number of not pruned channels"""
        net = ToyAdd_2D()
        input_shape = net.input_shape
        a_prec = (2, 4, 8)
        alpha = torch.zeros(4, 10)
        alpha[torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 3]),
              torch.tensor([3, 5, 9, 0, 1, 2, 7, 8, 6, 4])] = 1
        x = torch.rand(input_shape).unsqueeze(0)
        # Use the following alpha matrix to check the sanity of out_features_eff
        # for one specific layer
        # [[0., 0., 0., 1., 0., 1., 0., 0., 0., 1.],
        #  [1., 1., 1., 0., 0., 0., 0., 1., 1., 0.],
        #  [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        #  [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])

        # case (1): zero_index = 0
        w_prec = (0, 2, 4, 8)
        mixprec_net = MPS(net,
                          input_shape=input_shape,
                          qinfo=get_default_qinfo(a_precision=a_prec, w_precision=w_prec),
                          w_search_type=MPSType.PER_CHANNEL,
                          hard_softmax=True)
        for layer in mixprec_net.modules():  # force sampling of 8-bit precision
            if isinstance(layer, MPSConv2d) or isinstance(layer, MPSLinear):
                alpha_no_0bit = torch.zeros(layer.w_mps_quantizer.alpha.shape)
                alpha_no_0bit[-1, :] = 1
                layer.w_mps_quantizer.alpha.data = alpha_no_0bit
        conv1 = cast(MPSConv2d, mixprec_net.seed.conv1)
        conv1.w_mps_quantizer.alpha.data = alpha  # update conv1 layer's alpha
        mixprec_net(x)  # perform a forward pass to update the out_features_eff values
        self.assertEqual(conv1.out_features_eff.item(), 7)

        # case (2): zero_index = 2
        w_prec = (2, 4, 0, 8)
        mixprec_net = MPS(net,
                          input_shape=input_shape,
                          qinfo=get_default_qinfo(a_precision=a_prec, w_precision=w_prec),
                          w_search_type=MPSType.PER_CHANNEL,
                          hard_softmax=True)
        for layer in mixprec_net.modules():
            if isinstance(layer, MPSConv2d) or isinstance(layer, MPSLinear):
                alpha_no_0bit = torch.zeros(layer.w_mps_quantizer.alpha.shape)
                alpha_no_0bit[-1, :] = 1
                layer.w_mps_quantizer.alpha.data = alpha_no_0bit
        conv1 = cast(MPSConv2d, mixprec_net.seed.conv1)
        conv1.w_mps_quantizer.alpha.data = alpha  # update conv1 layer's alpha
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
        mixprec_net = MPS(net,
                          input_shape=input_shape,
                          qinfo=get_default_qinfo(a_precision=a_prec, w_precision=w_prec),
                          w_search_type=MPSType.PER_CHANNEL,
                          hard_softmax=True)
        mixprec_net(x)

        conv0 = cast(MPSConv2d, mixprec_net.seed.conv0)
        self.assertEqual(conv0.input_features_calculator.features.item(), input_shape[0])

        conv1 = cast(MPSConv2d, mixprec_net.seed.conv1)
        self.assertEqual(conv1.input_features_calculator.features.item(),
                         conv0.out_features_eff.item())

        fc = cast(MPSLinear, mixprec_net.seed.fc)
        self.assertEqual(fc.input_features_calculator.features.item(),
                         conv1.out_features_eff.item() * 10 * 10)

        # case(2): 0-bit precision with some channels pruned
        w_prec = (0, 2, 4, 8)
        mixprec_net = MPS(net,
                          input_shape=input_shape,
                          qinfo=get_default_qinfo(a_precision=a_prec, w_precision=w_prec),
                          w_search_type=MPSType.PER_CHANNEL,
                          hard_softmax=True)

        for layer in mixprec_net.modules():  # force sampling of 8-bit precision
            if isinstance(layer, MPSConv2d) or isinstance(layer, MPSLinear):
                alpha_no_0bit = torch.zeros(layer.w_mps_quantizer.alpha.shape)
                alpha_no_0bit[-1, :] = 1
                layer.w_mps_quantizer.alpha.data = alpha_no_0bit
        # prune one channel of conv1 layer
        conv1 = cast(MPSConv2d, mixprec_net.seed.conv1)
        conv1.w_mps_quantizer.alpha.data[0, 2] = 1
        conv1.w_mps_quantizer.alpha.data[-1, 2] = 0
        mixprec_net(x)

        fc = cast(MPSLinear, mixprec_net.seed.fc)
        self.assertEqual(fc.input_features_calculator.features.item(),
                         conv1.out_features_eff.item() * 10 * 10)


    def test_qinfo_layer(self):
        nn_ut = SimpleNN2D()
        my_qinfo = get_default_qinfo()
        my_qinfo['conv0'] = my_qinfo['layer_default'].copy()
        my_qinfo['conv0']['weight'] = my_qinfo['layer_default']['weight'].copy()
        my_qinfo['conv0']['weight']['quantizer'] = FQWeight
        my_qinfo['conv1'] = my_qinfo['layer_default'].copy()
        my_qinfo['conv1']['output'] = my_qinfo['layer_default']['output'].copy()
        my_qinfo['conv1']['output']['quantizer'] = DummyQuantizer
        my_qinfo['layer_default']['weight']['quantizer'] = MinMaxWeight
        my_qinfo['layer_default']['output']['quantizer'] = PACTAct
        mixprec_net = MPS(nn_ut,
                          input_shape=nn_ut.input_shape,
                          w_search_type=MPSType.PER_CHANNEL,
                          qinfo=my_qinfo)
        # verify that the specified quantizer changed
        conv0_w_qtz = cast(MPSBaseQtz, cast(MPSConv2d, mixprec_net.seed.conv0).w_mps_quantizer).quantizer
        self.assertIs(conv0_w_qtz, FQWeight, "Wrong weight quantizer type")
        # verify that the other quantizers remained at default
        conv1_w_qtz = cast(MPSBaseQtz, cast(MPSConv2d, mixprec_net.seed.conv1).w_mps_quantizer).quantizer
        self.assertIs(conv1_w_qtz, MinMaxWeight, "Wrong weight quantizer type")
        fc_w_qtz = cast(MPSBaseQtz, cast(MPSLinear, mixprec_net.seed.fc).w_mps_quantizer).quantizer
        self.assertIs(fc_w_qtz, MinMaxWeight, "Wrong weight quantizer type")
        # verify that the specified quantizer changed
        conv1_a_qtz = cast(MPSBaseQtz, cast(MPSConv2d, mixprec_net.seed.conv1).out_mps_quantizer).quantizer
        self.assertIs(conv1_a_qtz, DummyQuantizer, "Wrong activation quantizer type")
        # verify that the other quantizers remained at default (final linear has no activation quantization so it's not checked
        conv0_a_qtz = cast(MPSBaseQtz, cast(MPSConv2d, mixprec_net.seed.conv0).out_mps_quantizer).quantizer
        self.assertIs(conv0_a_qtz, PACTAct, "Wrong weight quantizer type")


if __name__ == '__main__':
    unittest.main(verbosity=2)
