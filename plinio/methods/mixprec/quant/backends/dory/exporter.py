# *----------------------------------------------------------------------------*
# * Copyright (C) 2023 Politecnico di Torino, Italy                            *
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
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *----------------------------------------------------------------------------*

# Code strongly inspired by:
# https://github.com/pulp-platform/quantlib/blob/main/backends/dory/onnxexporter.py

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from plinio.methods.mixprec.quant.backends.base import remove_inp_quantizer, remove_relu
from .annotator import DORYAnnotator


class DORYExporter:
    """Class to export DORY-compliant integer nn.Module to DORY-compliant onnx model"""

    def __init__(self):
        self._annotator = DORYAnnotator()
        self._onnxname = None
        self._onnxfilepath = None

    def export(self,
               network: nn.Module,
               input_shape: torch.Size,
               path: Path,
               name: Optional[str] = None,
               opset_version: int = 10
               ):
        onnxname = name if name is not None else network.__class__.__name__
        self._onnxname = onnxname
        onnxfilename = onnxname + '.onnx'
        onnxfilepath = path / onnxfilename
        self._onnxfilepath = onnxfilepath

        # Remove input quantizer from `network`
        network = remove_inp_quantizer(network)
        # Remove calls to functional relu from `network`
        network = remove_relu(network)

        # Export network to onnx file
        torch.onnx.export(network,
                          torch.randn(input_shape),
                          onnxfilepath,
                          export_params=True,
                          do_constant_folding=True,
                          opset_version=opset_version)
        # Annotate the onnx file with backend-specific information
        self._annotator.annotate(network, onnxfilepath)
