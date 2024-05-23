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

from functools import partial
import json
from pathlib import Path
from typing import Optional, cast, Union, NamedTuple, List

import torch
import torch.nn as nn

from plinio.methods.mps.quant.backends.base import (remove_inp_quantizer,
                                                    remove_relu, get_map,
                                                    )
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
        # Remove every relu from `network`
        network = remove_relu(network)

        # Export network to onnx file
        torch.onnx.export(network,
                          torch.randn(input_shape),
                          str(onnxfilepath),
                          export_params=True,
                          do_constant_folding=True,
                          opset_version=opset_version,
                          )
        # Annotate the onnx file with backend-specific information
        self._annotator.annotate(network, onnxfilepath)

    def export_json_config(self,
                           code_size: int = 160000,
                           nb_inputs: int = 1,
                           input_bits: int = 8,
                           input_signed: bool = False
                           ):
        self._onnxfilepath = cast(Path, self._onnxfilepath)
        onnx_file = self._onnxfilepath.name
        onnx_name = self._onnxfilepath.stem

        cnn_dory_config = {
            'BNRelu_bits': 32,
            'onnx_file': onnx_file,
            'code reserved space': code_size,
            'n_inputs': nb_inputs,
            'input_bits': input_bits,
            'input_signed': input_signed
        }
        jsonfilepath = self._onnxfilepath.parent
        with open(jsonfilepath.joinpath(f'config_{onnx_name}.json'), 'w') as fp:
            json.dump(cnn_dory_config, fp, indent=4)

    @staticmethod
    def dump_features(network: nn.Module,
                      x: torch.Tensor,
                      path: Union[Path, str]) -> None:
        """Given a network, export the features associated with a given input.
        To verify the correctness of an ONNX export, DORY requires text files
        containing the values of the features for each layer in the target
        network. The format of these text files is exemplified here:
        https://github.com/pulp-platform/dory_examples/tree/master/examples/Quantlab_examples .
        """

        class Features(NamedTuple):
            module_name: str
            features: torch.Tensor

        def export_to_txt(module_name: str, filename: str, path: Path, t: torch.Tensor):
            try:  # for the output, this step is not applicable
                # PyTorch's `nn.Conv2d` layers output CHW arrays, but DORY expects HWC arrays
                t = t.squeeze().permute(1, 2, 0)
            except RuntimeError:
                pass  # I won't permute the features of this module

            filepath = path / f"{filename}.txt"
            with open(str(filepath), 'w') as fp:
                fp.write(f"# {module_name} (shape {list(t.shape)}),\n")
                for c in t.flatten():
                    fp.write(f"{int(c)},\n")

        if type(path) == str:
            path = Path(path)
        path = cast(Path, path)

        # Since PyTorch uses dynamic graphs, we don't have symbolic handles
        # over the inner array. Therefore, we use PyTorch hooks to dump the
        # outputs of Requant layers.
        features: List[Features] = []

        def hook_fn(self, in_: torch.Tensor, out_: torch.Tensor, module_name: str):
            # DORY wants HWC tensors
            features.append(Features(module_name=module_name, features=out_.squeeze(0)))

        # The core dump functionality starts here
        # Get supported DORY layers
        dory_layers = get_map()['dory']
        # 1. set up hooks to intercept features
        for n, m in network.named_modules():
            if isinstance(m, tuple(dory_layers.values()) + (nn.MaxPool2d,)):
                hook = partial(hook_fn, module_name=n)
                m.register_forward_hook(hook)

        # 2. propagate the supplied input through the network; the hooks will capture the features
        x = x.clone()
        y = network(x.to(dtype=torch.float32))

        # 3. export input, features, and output to text files
        export_to_txt('input', 'input', path, x)
        for i, (module_name, f) in enumerate(features):
            export_to_txt(module_name, f"out_layer{i}", path, f)
        export_to_txt('output', f"out_layer{len(features)}", path, y)
        export_to_txt('output', 'output', path, y)
