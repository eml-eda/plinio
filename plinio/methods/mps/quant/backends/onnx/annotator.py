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
# * Author:  Francesco Daghero <francesco.daghero@polito.it>                             *
# *----------------------------------------------------------------------------*

from pathlib import Path

import numpy as np
import onnx
from onnx import helper as onnx_helper
import torch.nn as nn
import itertools
from typing import Optional


class ONNXAnnotator:
    """Class to annotate ONNX-compliant onnx model"""

    def __init__(self, requantization_bits: int = 32):
        self._requantization_bits = requantization_bits

    def annotate(self,
                 network: nn.Module,
                 onnxfilepath: Path,
                 input_bits: int,
                 input_signed: bool):
        """This part will be common to every backend, while _annotate is the actual
        backend-specific annotation.
        """
        # Load onnx
        onnxproto = onnx.load(str(onnxfilepath))

        # Backend-specific annotation
        self._annotate(network, onnxproto, input_bits, input_signed)

        # Save backend-specific annotated onnx
        onnx.save(onnxproto, str(onnxfilepath))

    def _annotate(self,
                  network: nn.Module,
                  onnxproto: onnx.ModelProto,
                  input_bits: int,
                  input_signed: bool):
        """Backend-specific annotation function"""

        def get_onnxnode_attr_by_name(node: onnx.NodeProto,
                                      name: str) -> onnx.AttributeProto:
            return next(iter(filter(lambda a: (a.name == name), node.attribute)))

        def find_pytorch_module(module, node, initializers_names) -> Optional[nn.Module]:
            # Find the layer in the pytorch model
            # It may be either the layer operation or a part of it
            # We care about only nodes with weights :)
            for idx, tensor_name in enumerate(node.input):
                # Convolution
                if "Conv" in node.name and tensor_name in initializers_names:
                    #op_name = layer_name.split("/")[1].rsplit('.', 1)[0]
                    full_name = ".".join((tensor_name.split(".")[:-1]))
                    return module.get_submodule(full_name)
                # IF it is a matmul and the tensor is the weight tensor (an initializer)
                elif "MatMul" in node.name and tensor_name in initializers_names:
                    op_name = layer_name.split("/")[1]
                    return module.get_submodule(op_name)

            return None

        # Define backend-specific supported ONNX nodes. The nodes belonging to
        # different node classes will be annotated using a class-specific logic.
        PLINIO_QNODES = {'Conv', 'Gemm', 'MatMul'}

        # Graph input
        inp_node = onnxproto.graph.input[0].name
        metadata = {}

        # Add the input specifically
        # TODO: Multi-input graphs?
        metadata[inp_node] = {}
        metadata[inp_node]["precision"] = input_bits
        metadata[inp_node]["signed"] = input_signed

        layer_input_precision = input_bits
        layer_input_signed = input_signed

        # Iterate over the nodes and associate them to the corresponding layer
        # There is not beautiful way to do this, as the ONNX names with Sequential
        # blocks are different from the Pytorch names
        # Generally weights have the correct name


        initializers_names = [init.name for init in onnxproto.graph.initializer]
        # Now all the other tensors
        for node in onnxproto.graph.node:
            layer_name = node.name or node.output[0]
            op_type = node.op_type

            pytorch_module =find_pytorch_module(network, node, initializers_names)

            # Iterate over inputs and outputs
            for idx, tensor_name in enumerate(list(node.input)+ list(node.output)):
                tensor_type = "output" if idx >= len(node.input) else "input"
                if tensor_name in metadata:
                    # All input activation in theory end up here,
                    # As we added them already as output of the previous layer
                    # The exception is the input tensor, which we added manually
                    layer_input_precision = metadata[tensor_name]["precision"]
                    layer_input_signed = metadata[tensor_name]["signed"]
                    continue

                onnx_op = node.op_type
                metadata[tensor_name] = {}
                tns_meta=metadata[tensor_name]

                # Weights/Inputs buffer are treated differently
                if tensor_name in initializers_names and pytorch_module is not None:
                    if hasattr(pytorch_module, "w_quantizer"):
                        tns_meta['precision'] = pytorch_module.w_quantizer.precision
                        tns_meta['signed'] = pytorch_module.signed
                    else:
                        tns_meta['precision'] = None
                        tns_meta['signed'] = None
                    if hasattr(pytorch_module, "b_quantizer"):
                        tns_meta['precision'] = pytorch_module.b_quantizer.precision
                        tns_meta['signed'] = True
                    else:
                        tns_meta['precision'] = None
                        tns_meta['signed'] = None

                # If it is not a weight or bias, we assume it is an activation buffer
                elif tensor_type == 'output':
                    # Clip output changes the precision
                    if node.op_type == 'Clip':
                        clip_lo = get_onnxnode_attr_by_name(node, 'min').f
                        clip_hi = get_onnxnode_attr_by_name(node, 'max').f
                        assert np.log2(clip_hi + 1.0) % 1.0 < 1e-6
                        n_levels = clip_hi - clip_lo + 1.0
                        tns_meta['precision'] = int(np.round(np.log2(n_levels)))
                        tns_meta['signed'] = clip_lo < 0
                        # THis is generally the last node of the block so the following is for safety
                        layer_input_precision = tns_meta['precision']
                        layer_input_signed = tns_meta['signed']
                    elif onnx_op in PLINIO_QNODES and pytorch_module is not None:
                        if (hasattr(pytorch_module, 'out_quantizer')):
                            # A quantized node
                            tns_meta['precision'] = self._requantization_bits
                            tns_meta['signed'] = True
                            layer_input_precision = tns_meta['precision']
                            layer_input_signed = tns_meta['signed']
                    else:
                        # Precision does not change w.r.t. the input
                        tns_meta["precision"] = layer_input_precision
                        tns_meta["signed"] = layer_input_signed

        for tensor_name, annotation in metadata.items():
            entry = onnxproto.metadata_props.add()
            entry.key = tensor_name
            entry.value = str(annotation)


    # This function is rhigt now not used but was helpful for exporting onnx with torch 1.11
    def _rename_node_io(self, node):
        for idx, inp in enumerate(node.input):
            if 'onnx' in inp:
                integer_idx = inp.split('_')[-1]
                # assert type(integer_idx) == int
                node.input[idx] = integer_idx
        for idx, oup in enumerate(node.output):
            if 'onnx' in oup:
                integer_idx = oup.split('_')[-1]
                # assert type(integer_idx) == int
                node.output[idx] = integer_idx
