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
# https://github.com/pulp-platform/quantlib/blob/main/backends/dory/onnxannotator.py

from pathlib import Path

import numpy as np
import onnx
from onnx import helper as onnx_helper
import torch.nn as nn


class DORYAnnotator:
    """Class to annotate DORY-compliant onnx model"""

    def __init__(self, requantization_bits: int = 32):
        self._requantization_bits = requantization_bits

    def annotate(self,
                 network: nn.Module,
                 onnxfilepath: Path):
        """This part will be common to every backend, while _annotate is the actual
        backend-specific annotation.
        TODO: build abstract and generic classes for annotator and exporter
        """
        # Load onnx
        onnxproto = onnx.load(str(onnxfilepath))

        # Backend-specific annotation
        self._annotate(network, onnxproto)

        # Save backend-specific annotated onnx
        onnx.save(onnxproto, str(onnxfilepath))

    def _annotate(self,
                  network: nn.Module,
                  onnxproto: onnx.ModelProto):
        """Backend-specific annotation function"""

        def get_onnxnode_attr_by_name(node: onnx.NodeProto,
                                      name: str) -> onnx.AttributeProto:
            return next(iter(filter(lambda a: (a.name == name), node.attribute)))

        # Define backend-specific supported ONNX nodes. The nodes belonging to
        # different node classes will be annotated using a class-specific logic.
        dory_onnxnode_op_types = {
            'linear': {'Conv', 'Gemm'},
            'mul': {'Mul'},
            'add': {'Add'},
            'clip': {'Clip'},
        }

        # Rename input and output nodes.
        # DORY expects a single integer as input/output node name instead of
        # the default onnx::{operator}_{integer}
        inp_node = onnxproto.graph.input[0].name
        if 'onnx' in inp_node:
            integer_idx = inp_node.split('_')[-1]
            onnxproto.graph.input[0].name = integer_idx

        for n in onnxproto.graph.node:

            # Rename input and output nodes.
            # DORY expects a single integer as input/output node name instead of
            # the default onnx::{operator}_{integer}
            self._rename_node_io(n)

            op_type = n.op_type
            annotations = []

            if op_type in dory_onnxnode_op_types['linear']:
                op_name = n.input[1].rsplit('.', 1)[0]
                pytorch_module = network.get_submodule(op_name)
                if isinstance(pytorch_module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    weight_bits = pytorch_module.w_precision
                    bias_bits = pytorch_module.b_precision
                    annotations.append(onnx_helper.make_attribute(key='weight_bits',
                                                                  value=weight_bits))
                    annotations.append(onnx_helper.make_attribute(key='bias_bits',
                                                                  value=bias_bits))

            elif op_type in dory_onnxnode_op_types['mul']:
                mul_bits = self._requantization_bits
                annotations.append(onnx_helper.make_attribute(key='mult_bits',
                                                              value=mul_bits))

            elif op_type in dory_onnxnode_op_types['add']:
                is_requant_add = all(i.isnumeric() for i in n.input)
                if is_requant_add:
                    add_bits = self._requantization_bits
                else:
                    add_bits = 32  # TODO: where to gather this info?
                annotations.append(onnx_helper.make_attribute(key='add_bits',
                                                              value=add_bits))

            elif op_type in dory_onnxnode_op_types['clip']:
                clip_lo = get_onnxnode_attr_by_name(n, 'min').f
                clip_hi = get_onnxnode_attr_by_name(n, 'max').f
                assert np.log2(clip_hi + 1.0) % 1.0 < 1e-6  # TODO: document this choice
                n_levels = clip_hi - clip_lo + 1.0
                output_bits = int(np.round(np.log2(n_levels)))
                annotations.append(onnx_helper.make_attribute(key='out_bits',
                                                              value=output_bits))

            else:  # the backend does not require special handling for this node type
                pass

            # flush attributes to the ONNX node
            n.attribute.extend(annotations)

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
