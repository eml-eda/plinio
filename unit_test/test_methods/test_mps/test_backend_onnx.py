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
# * Author: Francesco Daghero <francesco.daghero@polito.it>                              *
# *----------------------------------------------------------------------------*

from pathlib import Path
import unittest

import torch
torch.manual_seed(0)
from plinio.methods.mps import MPS, get_default_qinfo, MPSType
from plinio.methods.mps.quant.quantizers import PACTAct
from plinio.methods.mps.quant.backends import Backend, integerize_arch
from plinio.methods.mps.quant.backends.onnx import ONNXExporter
from unit_test.models import (
    ToySequentialFullyConv2d,
    ToySequentialConv2d,
    TutorialModel,
    ToySequentialFullyConv2dDil,
    ToySequentialConv2d_v2,
)
import numpy as np
from unit_test.models.miniresnet import MiniResNet


class TestBackendONNX(unittest.TestCase):
    """Test conversion operations from nn.Module to nn.ONNX passing through
    nn.MPS and nn.Quant.
    Exports multiple models to ONNX, runs them with onnxruntime and compares
    the results with the original PyTorch model.
    """

    def test_autoimport_fullyconv_layer(self):
        import onnxruntime as ort
        for i, model in enumerate([
                        MiniResNet
                        , ToySequentialFullyConv2d,
                        ToySequentialFullyConv2dDil,
                        ToySequentialConv2d,
                        TutorialModel,
                        ToySequentialConv2d_v2,
                      ]):
            for signed in [True, False]:
                print()
                print(f"Test {i} - {model.__name__} - signed: {signed}", end ="\n")
                # Instantiate toy model
                bits = 8
                nn_ut = model()
                input_shape = nn_ut.input_shape if hasattr(nn_ut, "input_shape") else (3, 32, 32)

                # Convert to mixprec searchable model
                mixprec_nn = MPS(
                    nn_ut,
                    input_shape=input_shape,
                    qinfo=get_default_qinfo(w_precision=(bits,), a_precision=(bits,)),
                    w_search_type=MPSType.PER_LAYER,
                )
                # Dummy inference
                dummy_inp = torch.rand([1,] + list(input_shape))
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

                integer_nn = integerize_arch(
                    quantized_nn,
                    Backend.ONNX,
                    backend_kwargs={"signed": signed},
                    remove_input_quantizer=True,
                )

                inp_quant = PACTAct(bits, init_clip_val=1, dequantize=False)
                # Dummy inference
                with torch.no_grad():
                    dummy_inp = inp_quant(dummy_inp)
                    if signed:
                        dummy_inp = dummy_inp - 128

                    out_int = integer_nn(dummy_inp)

                self.assertTrue(
                    out_quant.argmax() == out_int.argmax(),
                    "Mismatch between fake-quantized and integer outputs",
                )

                # Convert to onnx
                exporter = ONNXExporter()
                exporter.export(
                    integer_nn, dummy_inp.shape, Path("."), input_bits=bits, input_signed=signed, integerize_onnx=False
                )

                session = ort.InferenceSession(f"./{integer_nn.__class__.__name__}.onnx", providers=["CPUExecutionProvider"])
                input_name = session.get_inputs()[0].name
                input_data = dummy_inp.numpy()
                output_ort = session.run(None, {input_name: input_data})

                self.assertTrue(
                    torch.allclose(
                        out_int,
                        torch.tensor(np.array(output_ort)).reshape_as(out_int),
                        rtol=0.0001,
                    ),
                    "Mismatch between integer and ONNX outputs",
                )
                Path(f'./{integer_nn.__class__.__name__}.onnx').unlink()
