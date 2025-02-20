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

import copy
from pathlib import Path
import unittest

import torch
from plinio.methods.mps import MPS, get_default_qinfo, MPSType
from plinio.methods.mps.quant.backends import Backend, integerize_arch
from plinio.methods.mps.quant.backends.match import MATCHExporter
from unit_test.models import (
    ToySequentialFullyConv2d,
    ToySequentialConv2d,
    TutorialModel,
    ToySequentialFullyConv2dDil,
    ToyMultiPath1_2D,
    ToyResNet,
    ToyResNet_inp_conn,
    ToyResNet_inp_conn_add_out,
)
from .utils import capture_outputs, compare_outputs


class TestBackendMATCH(unittest.TestCase):
    """Test conversion operations from nn.Module to nn.MATCH passing through
    nn.MPS and nn.Quant.
    """

    def test_autoimport_fullyconv_layer(self):
        """Test the conversion of a simple fully convolutional sequential model
        with layer autoconversion with PER_LAYER weight mixed-precision (default)"""
        # Instantiate toy model
        nn_ut = ToySequentialFullyConv2d()

        # Convert to mixprec searchable model
        mixprec_nn = MPS(
            nn_ut,
            input_shape=nn_ut.input_shape,
            qinfo=get_default_qinfo(w_precision=(8,), a_precision=(8,)),
            w_search_type=MPSType.PER_LAYER,
        )
        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape)
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.export()
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(
            torch.all(out_mixprec == out_quant),
            "Mismatch between mixprec and fake-quantized outputs",
        )

        # Convert to integer MATCH-compliant model
        integer_nn = integerize_arch(quantized_nn, Backend.MATCH)
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(
            out_quant.argmax() == out_int.argmax(),
            "Mismatch between fake-quantized and integer outputs",
        )

        # Convert to onnx
        exporter = MATCHExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path("."))
        Path(f"./{integer_nn.__class__.__name__}.onnx").unlink()

    def test_autoimport_fullyconv_w_dil_layer(self):
        """Test the conversion of a simple fully convolutional sequential model
        with dilation and
        with layer autoconversion with PER_LAYER weight mixed-precision (default)"""
        # Instantiate toy model
        nn_ut = ToySequentialFullyConv2dDil()

        # Convert to mixprec searchable model
        mixprec_nn = MPS(
            nn_ut,
            input_shape=nn_ut.input_shape,
            qinfo=get_default_qinfo(w_precision=(8,), a_precision=(8,)),
            w_search_type=MPSType.PER_LAYER,
        )
        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape)
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.export()
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(
            torch.all(out_mixprec == out_quant),
            "Mismatch between mixprec and fake-quantized outputs",
        )

        # Convert to integer MATCH-compliant model
        integer_nn = integerize_arch(quantized_nn, Backend.MATCH)
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(
            out_quant.argmax() == out_int.argmax(),
            "Mismatch between fake-quantized and integer outputs",
        )

        # Convert to onnx
        exporter = MATCHExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path("."))
        Path(f"./{integer_nn.__class__.__name__}.onnx").unlink()

    def test_autoimport_simple_layer(self):
        """Test the conversion of a simple convolutional and linear sequential model
        with layer autoconversion with PER_LAYER weight mixed-precision (default)"""
        # Instantiate toy model
        nn_ut = ToySequentialConv2d()

        # Convert to mixprec searchable model
        mixprec_nn = MPS(
            nn_ut,
            input_shape=nn_ut.input_shape,
            qinfo=get_default_qinfo(w_precision=(8,), a_precision=(8,)),
            w_search_type=MPSType.PER_LAYER,
        )
        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape)
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.export()
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(
            torch.all(out_mixprec == out_quant),
            "Mismatch between mixprec and fake-quantized outputs",
        )

        # Convert to integer MATCH-compliant model
        integer_nn = integerize_arch(quantized_nn, Backend.MATCH)
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(
            out_quant.argmax() == out_int.argmax(),
            "Mismatch between fake-quantized and integer outputs",
        )

        # Convert to onnx
        exporter = MATCHExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path("."))
        Path(f"./{integer_nn.__class__.__name__}.onnx").unlink()

    def test_autoimport_sequential(self):
        """Test the conversion of a more complex convolutional and linear sequential model
        with conv5x5, conv3x3 and depthwise-separable conv
        with layer autoconversion with PER_LAYER weight mixed-precision (default)"""
        # Instantiate tutorial model
        nn_ut = TutorialModel()

        # Convert to mixprec searchable model
        mixprec_nn = MPS(
            nn_ut,
            input_shape=nn_ut.input_shape,
            qinfo=get_default_qinfo(w_precision=(8,), a_precision=(8,)),
            w_search_type=MPSType.PER_LAYER,
        )
        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape)
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.export()
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(
            torch.all(out_mixprec == out_quant),
            "Mismatch between mixprec and fake-quantized outputs",
        )

        # Convert to integer MATCH-compliant model
        integer_nn = integerize_arch(quantized_nn, Backend.MATCH)
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        # #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(
            out_quant.argmax() == out_int.argmax(),
            "Mismatch between fake-quantized and integer outputs",
        )

        # Convert to onnx
        exporter = MATCHExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path("."))
        Path(f"./{integer_nn.__class__.__name__}.onnx").unlink()

    def test_autoimport_sequential_cuda(self):
        """Test the conversion of a more complex convolutional and linear sequential model
        with conv5x5, conv3x3 and depthwise-separable conv
        with layer autoconversion with PER_LAYER weight mixed-precision (default)
        *Usign GPU*
        """
        # Check CUDA availability
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Instantiate tutorial model
        nn_ut = TutorialModel().to(device)

        # Convert to mixprec searchable model
        mixprec_nn = MPS(
            nn_ut,
            input_shape=nn_ut.input_shape,
            qinfo=get_default_qinfo(w_precision=(8,), a_precision=(8,)),
            w_search_type=MPSType.PER_LAYER,
        )
        mixprec_nn = mixprec_nn.to(device)
        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape, device=device)
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.export()
        quantized_nn = quantized_nn.to(device)
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(
            torch.all(out_mixprec == out_quant),
            "Mismatch between mixprec and fake-quantized outputs",
        )

        # Convert to integer MATCH-compliant model
        integer_nn = integerize_arch(quantized_nn, Backend.MATCH)
        integer_nn = integer_nn.to(device)
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        # #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(
            out_quant.argmax() == out_int.argmax(),
            "Mismatch between fake-quantized and integer outputs",
        )

        # Convert to onnx
        exporter = MATCHExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path("."))
        Path(f"./{integer_nn.__class__.__name__}.onnx").unlink()

    def test_autoimport_sequential_specific_scale_bits(self):
        """Test the conversion of a more complex convolutional and linear sequential model
        with conv5x5, conv3x3 and depthwise-separable conv
        with layer autoconversion with PER_LAYER weight mixed-precision (default).
        In this case we also test using a specific value for the scale_bits number.
        """
        # Instantiate tutorial model
        nn_ut = TutorialModel()

        # Convert to mixprec searchable model
        mixprec_nn = MPS(
            nn_ut,
            input_shape=nn_ut.input_shape,
            qinfo=get_default_qinfo(w_precision=(8,), a_precision=(8,)),
            w_search_type=MPSType.PER_LAYER,
        )
        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape)
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.export()
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(
            torch.all(out_mixprec == out_quant),
            "Mismatch between mixprec and fake-quantized outputs",
        )

        # Convert to integer MATCH-compliant model
        integer_nn = integerize_arch(
            quantized_nn, Backend.MATCH, backend_kwargs={"scale_bit": 16}
        )
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(
            out_quant.argmax() == out_int.argmax(),
            "Mismatch between fake-quantized and integer outputs",
        )

        # Convert to onnx
        exporter = MATCHExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path("."))
        Path(f"./{integer_nn.__class__.__name__}.onnx").unlink()

    def test_autoimport_multipath(self):
        """Test the conversion of a more complex model with multiple paths
        with layer autoconversion with PER_LAYER weight mixed-precision (default)"""
        # Instantiate toy model
        nn_ut = ToyMultiPath1_2D()

        # Convert to mixprec searchable model
        mixprec_nn = MPS(
            nn_ut,
            input_shape=nn_ut.input_shape,
            qinfo=get_default_qinfo(w_precision=(8,), a_precision=(8,)),
            w_search_type=MPSType.PER_LAYER,
        )
        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape)
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.export()
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(
            torch.all(out_mixprec == out_quant),
            "Mismatch between mixprec and fake-quantized outputs",
        )

        # Convert to integer MATCH-compliant model
        integer_nn = integerize_arch(quantized_nn, Backend.MATCH)
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(
            out_quant.argmax() == out_int.argmax(),
            "Mismatch between fake-quantized and integer outputs",
        )

        # Convert to onnx
        exporter = MATCHExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path("."))
        Path(f"./{integer_nn.__class__.__name__}.onnx").unlink()

    def test_autoimport_resnet(self):
        """Test the conversion of a more complex model with resnet-like arch
        with layer autoconversion with PER_LAYER weight mixed-precision (default)"""
        nn_ut = ToyResNet()

        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape)
        with torch.no_grad():
            nn_ut(dummy_inp)

        # Convert to mixprec searchable model
        mixprec_nn = MPS(
            nn_ut,
            input_shape=nn_ut.input_shape,
            qinfo=get_default_qinfo(w_precision=(8,), a_precision=(8,)),
            w_search_type=MPSType.PER_LAYER,
        )
        # Dummy inference
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.export()
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(
            torch.all(out_mixprec == out_quant),
            "Mismatch between mixprec and fake-quantized outputs",
        )

        # Convert to integer MATCH-compliant model
        integer_nn = integerize_arch(quantized_nn, Backend.MATCH)
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(
            out_quant.argmax() == out_int.argmax(),
            "Mismatch between fake-quantized and integer outputs",
        )

        # Convert to onnx
        exporter = MATCHExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path("."))
        Path(f"./{integer_nn.__class__.__name__}.onnx").unlink()

    def test_autoimport_resnet_inp_conn(self):
        """Test the conversion of a more complex model with resnet-like arch
        where the input is connected to an add node with a skip connection
        with layer autoconversion with PER_LAYER weight mixed-precision (default)"""
        nn_ut = ToyResNet_inp_conn()

        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape)
        with torch.no_grad():
            nn_ut(dummy_inp)

        # Convert to mixprec searchable model
        qinfo = get_default_qinfo(w_precision=(8,), a_precision=(8,))
        qinfo["layer_default"]["output"]["kwargs"]["init_clip_val"] = 1
        mixprec_nn = MPS(
            nn_ut,
            input_shape=nn_ut.input_shape,
            qinfo=qinfo,
            w_search_type=MPSType.PER_LAYER,
        )
        # Dummy inference
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.export()
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(
            torch.all(out_mixprec == out_quant),
            "Mismatch between mixprec and fake-quantized outputs",
        )

        # Convert to integer MATCH-compliant model
        integer_nn = integerize_arch(copy.deepcopy(quantized_nn), Backend.MATCH)
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(
            out_quant.argmax() == out_int.argmax(),
            "Mismatch between fake-quantized and integer outputs",
        )
        # with torch.no_grad():
        #     all_out_mps = capture_outputs(mixprec_nn.seed, dummy_inp, dequantize=False)
        #     all_out_int = capture_outputs(integer_nn, dummy_inp)
        #     compare_outputs(all_out_mps, all_out_int)

        # Convert to onnx
        exporter = MATCHExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path("."))
        Path(f"./{integer_nn.__class__.__name__}.onnx").unlink()

    def test_autoimport_resnet_output_add(self):
        """Test the conversion of a more complex model with resnet-like arch
        where the output is generated by an add node
        with layer autoconversion with PER_LAYER weight mixed-precision (default)"""
        nn_ut = ToyResNet_inp_conn_add_out()

        # Dummy inference
        dummy_inp = torch.rand((1,) + nn_ut.input_shape)
        with torch.no_grad():
            nn_ut(dummy_inp)

        # Convert to mixprec searchable model
        qinfo = get_default_qinfo(w_precision=(8,), a_precision=(8,))
        qinfo["layer_default"]["output"]["kwargs"]["init_clip_val"] = 1
        mixprec_nn = MPS(
            nn_ut,
            input_shape=nn_ut.input_shape,
            qinfo=qinfo,
            w_search_type=MPSType.PER_LAYER,
            quantize_output=True,
        )
        # Dummy inference
        with torch.no_grad():
            out_mixprec = mixprec_nn(dummy_inp)

        # Convert to (fake) quantized model
        quantized_nn = mixprec_nn.export()
        # Dummy inference
        with torch.no_grad():
            out_quant = quantized_nn(dummy_inp)
        self.assertTrue(
            torch.all(out_mixprec == out_quant),
            "Mismatch between mixprec and fake-quantized outputs",
        )

        # Convert to integer MATCH-compliant model
        integer_nn = integerize_arch(quantized_nn, Backend.MATCH)
        # Dummy inference
        with torch.no_grad():
            out_int = integer_nn(dummy_inp)
        # self.assertTrue(torch.all((100 * abs(out_quant - out_int) / out_quant) < 0.01),
        #                 "Mismatch between fake-quantized and integer outputs")
        self.assertTrue(
            out_quant.argmax() == out_int.argmax(),
            "Mismatch between fake-quantized and integer outputs",
        )

        # Convert to onnx
        exporter = MATCHExporter()
        exporter.export(integer_nn, dummy_inp.shape, Path("."))
        Path(f"./{integer_nn.__class__.__name__}.onnx").unlink()
