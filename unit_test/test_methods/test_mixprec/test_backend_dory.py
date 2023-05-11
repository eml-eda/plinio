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

from pathlib import Path
import unittest

import torch
from plinio.methods import MixPrec
from plinio.methods.mixprec.nn import MixPrecType
from plinio.methods.mixprec.quant.backends import Backend, integerize_arch
from plinio.methods.mixprec.quant.backends.dory import DORYExporter
from unit_test.models import ToySequentialFullyConv2d, ToySequentialConv2d, TutorialModel


class TestMixPrecConvert(unittest.TestCase):
    """Test conversion operations from nn.Module to nn.DORY passing through
       nn.MixPrec and nn.Quant.
    """

    def test_autoimport_fullyconv_layer(self):
        """Test the conversion of a simple fully convolutional sequential model
        with layer autoconversion with PER_LAYER weight mixed-precision (default)"""
        # Instantiate toy model
        nn_ut = ToySequentialFullyConv2d()

        # Convert to mixprec searchable model
        mixprec_nn = MixPrec(nn_ut,
                             input_shape=nn_ut.input_shape,
                             activation_precisions=(8,),
                             weight_precisions=(8,),
                             w_mixprec_type=MixPrecType.PER_LAYER
                             )
        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape)
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.arch_export()
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(torch.all(out_mixprec == out_quant),
                        "Mismatch between mixprec and fake-quantized outputs")

        # Convert to integer DORY-compliant model
        integer_nn = integerize_arch(quantized_nn, Backend.DORY)
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(out_quant.argmax() == out_int.argmax(),
                        "Mismatch between fake-quantized and integer outputs")

        # Convert to onnx
        exporter = DORYExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path('.'))

    def test_autoimport_simple_layer(self):
        """Test the conversion of a simple convolutional and linear sequential model
        with layer autoconversion with PER_LAYER weight mixed-precision (default)"""
        # Instantiate toy model
        nn_ut = ToySequentialConv2d()

        # Convert to mixprec searchable model
        mixprec_nn = MixPrec(nn_ut,
                             input_shape=nn_ut.input_shape,
                             activation_precisions=(8,),
                             weight_precisions=(8,),
                             w_mixprec_type=MixPrecType.PER_LAYER
                             )
        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape)
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.arch_export()
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(torch.all(out_mixprec == out_quant),
                        "Mismatch between mixprec and fake-quantized outputs")

        # Convert to integer DORY-compliant model
        integer_nn = integerize_arch(quantized_nn, Backend.DORY)
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(out_quant.argmax() == out_int.argmax(),
                        "Mismatch between fake-quantized and integer outputs")

        # Convert to onnx
        exporter = DORYExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path('.'))

    def test_autoimport_sequential(self):
        """Test the conversion of a more complex convolutional and linear sequential model
        with conv5x5, conv3x3 and depthwise-separable conv
        with layer autoconversion with PER_LAYER weight mixed-precision (default)"""
        # Instantiate tutorial model
        nn_ut = TutorialModel()

        # Convert to mixprec searchable model
        mixprec_nn = MixPrec(nn_ut,
                             input_shape=nn_ut.input_shape,
                             activation_precisions=(8,),
                             weight_precisions=(8,),
                             w_mixprec_type=MixPrecType.PER_LAYER
                             )
        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape)
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.arch_export()
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(torch.all(out_mixprec == out_quant),
                        "Mismatch between mixprec and fake-quantized outputs")

        # Convert to integer DORY-compliant model
        integer_nn = integerize_arch(quantized_nn, Backend.DORY)
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(out_quant.argmax() == out_int.argmax(),
                        "Mismatch between fake-quantized and integer outputs")

        # Convert to onnx
        exporter = DORYExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path('.'))
