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

    def annotate(
        self,
        network: nn.Module,
        onnxfilepath: Path,
        input_bits: int,
        input_signed: bool,
        output_bits: int = 32,
        output_signed: bool = True,
        integerize_onnx: bool = False
    ):
        """
        Annotate an ONNX model with the metadata required by the MPS framework.
        """
        # Load onnx
        onnxproto = onnx.load(str(onnxfilepath))

        # Backend-specific annotation
        onnxproto = self._annotate_metadata(network, onnxproto, input_bits, input_signed, output_bits, output_signed)
        if integerize_onnx:
            onnxproto = self._convert_to_int_onnx(network, onnxproto, input_bits, input_signed, output_bits, output_signed)

        # Save backend-specific annotated onnx
        onnx.save(onnxproto, str(onnxfilepath))

    def _annotate_metadata(
        self,
        network: nn.Module,
        onnxproto: onnx.ModelProto,
        input_bits: int,
        input_signed: bool,
        output_bits: int,
        output_signed: bool,
    ) -> onnx.ModelProto:
        """
        Backend-specific annotation function,
        changes only the metadata of the onnxproto, with no destructive operations.
        It updates the metadata_props field of the onnxproto.
        """

        def get_onnxnode_attr_by_name(
            node: onnx.NodeProto, name: str
        ) -> onnx.AttributeProto:
            return next(iter(filter(lambda a: (a.name == name), node.attribute)))

        def find_pytorch_module(
            module, node, initializers_names
        ) -> Optional[nn.Module]:
            # Find the layer in the pytorch model
            # It may be either the layer operation or a part of it
            # We care about only nodes with weights :)
            for idx, tensor_name in enumerate(node.input):
                # Convolution
                if "Conv" in node.name and tensor_name in initializers_names:
                    # op_name = layer_name.split("/")[1].rsplit('.', 1)[0]
                    full_name = ".".join((tensor_name.split(".")[:-1]))
                    return module.get_submodule(full_name)
                # IF it is a matmul and the tensor is the weight tensor (an initializer)
                elif "MatMul" in node.name and tensor_name in initializers_names:
                    op_name = layer_name.split("/")[1]
                    return module.get_submodule(op_name)

            return None

        # Define backend-specific supported ONNX nodes. The nodes belonging to
        # different node classes will be annotated using a class-specific logic.
        PLINIO_QNODES = {"Conv", "Gemm", "MatMul"}

        # Graph input
        inp_node = onnxproto.graph.input[0].name
        out_node = onnxproto.graph.output[0].name
        metadata = {}

        # Add the input specifically
        # TODO: Multi-input graphs?
        metadata[inp_node] = {}
        metadata[inp_node]["precision"] = input_bits
        metadata[inp_node]["signed"] = input_signed

        metadata[out_node] = {}
        metadata[out_node]["precision"] = output_bits
        metadata[out_node]["signed"] = output_signed

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
            pytorch_module = find_pytorch_module(network, node, initializers_names)

            # Iterate over inputs and outputs
            for idx, tensor_name in enumerate(list(node.input) + list(node.output)):
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
                tns_meta = metadata[tensor_name]

                # Weights/Inputs buffer are treated differently
                # Here we handle the precision of the weights
                if tensor_name in initializers_names and pytorch_module is not None:
                    # Matmul has a different naming convention, we check for it
                    if "weight" in tensor_name or "MatMul" in tensor_name:
                        if hasattr(pytorch_module, "w_quantizer"):
                            tns_meta["precision"] = pytorch_module.w_quantizer.precision
                            tns_meta["signed"] = pytorch_module.signed
                        else:
                            tns_meta["precision"] = None
                            tns_meta["signed"] = None
                    if "bias" in tensor_name:
                        if hasattr(pytorch_module, "b_quantizer"):
                            tns_meta["precision"] = pytorch_module.b_quantizer.precision
                            tns_meta["signed"] = True
                        else:
                            tns_meta["precision"] = None
                            tns_meta["signed"] = None

                # If it is not a weight or bias, we assume it is an activation buffer
                elif tensor_type == "output":
                    # Clip output changes the precision
                    if node.op_type == "Clip":
                        clip_lo = get_onnxnode_attr_by_name(node, "min").f
                        clip_hi = get_onnxnode_attr_by_name(node, "max").f
                        assert np.log2(clip_hi + 1.0) % 1.0 < 1e-6
                        n_levels = clip_hi - clip_lo + 1.0
                        tns_meta["precision"] = int(np.round(np.log2(n_levels)))
                        tns_meta["signed"] = clip_lo < 0
                        # THis is generally the last node of the block so the following is for safety
                        layer_input_precision = tns_meta["precision"]
                        layer_input_signed = tns_meta["signed"]
                    elif onnx_op in PLINIO_QNODES and pytorch_module is not None:
                        if hasattr(pytorch_module, "out_quantizer"):
                            # A quantized node
                            tns_meta["precision"] = self._requantization_bits
                            tns_meta["signed"] = True
                            layer_input_precision = tns_meta["precision"]
                            layer_input_signed = tns_meta["signed"]
                    else:
                        # Precision does not change w.r.t. the input
                        tns_meta["precision"] = layer_input_precision
                        tns_meta["signed"] = layer_input_signed

        # Write the metadata to the onnxproto
        for tensor_name, annotation in metadata.items():
            entry = onnxproto.metadata_props.add()
            entry.key = tensor_name
            entry.value = str(annotation)

        return onnxproto

    def _convert_to_int_onnx(
        self,
        network: nn.Module,
        onnxproto: onnx.ModelProto,
        input_bits: int,
        input_signed: bool,
        output_bits: int = 32,
        output_signed: bool = True,
    ):
        """
        A risky annotation function, as it changes the ONNX model and forces integers
        everywhere. It is necessary to support the quantized operations and avoids
        the opset 11 issues with the padding value.
        NOTE: This may fail and can be disabled from the exporter.
        """

        from onnx import shape_inference, checker, numpy_helper

        # Perform a shape inference, after casting the input to int8 or uint8 if necessary
        input_dtype = self._bound_to_onnx(input_bits, input_signed)
        output_dtype = self._bound_to_onnx(output_bits, output_signed)
        onnxproto.graph.input[0].type.tensor_type.elem_type = input_dtype
        onnxproto.graph.output[0].type.tensor_type.elem_type = output_dtype
        onnxproto = shape_inference.infer_shapes(onnxproto)

        # Extremely risky, but we need it to support the padding value
        # NOTE: This is a hack, as direct export with pytorch and opset 11 seems broken
        # NOTE: Do it AFTER shape inference
        # Bump the opset to 11, this breaks the shape inference!
        onnxproto.opset_import[0].version = 13

        # Load the metadata of the _annotate_metadata pass
        metadata = {entry.key: eval(entry.value) for entry in onnxproto.metadata_props}

        # Update all initializers to the correct dtype
        # TODO: This may be skipped? Isn't it already done by the shape inference?
        initializers = [*onnxproto.graph.initializer]
        for initializer in initializers:
            buffer_precision = metadata[initializer.name]["precision"]
            is_signed = metadata[initializer.name]["signed"]
            if buffer_precision is not None:
                new_dtype = self._bound_to_np(buffer_precision, is_signed)
                data = numpy_helper.to_array(initializer).astype(new_dtype)
                new_initializer = numpy_helper.from_array(
                    data, name=initializer.name
                )
                onnxproto.graph.initializer.remove(initializer)
                onnxproto.graph.initializer.extend([new_initializer])


        nodes_to_prune = []
        connection_pruned_nodes_map = {}
        cast_to_add = []
        cast_to_add_map = {}

        for idx, node in enumerate(onnxproto.graph.node):
            # Iterate over the nodes and change the dtypes of the tensors
            if node.op_type == "Constant":
                # Constant buffers are updated to the correct dtype
                attr = node.attribute[0]
                # Rewrite the attribute
                buffer_precision = metadata[node.output[0]]["precision"]
                is_signed = metadata[node.output[0]]["signed"]
                # Only if the precision is known
                #TODO : This may break with sparse tensors
                if buffer_precision is not None:
                    new_dtype = self._bound_to_np(buffer_precision, is_signed)
                    # Change the type of the constant
                    new_tensor = numpy_helper.from_array(
                        numpy_helper.to_array(attr.t).astype(new_dtype), name=node.output[0]
                    )
                    new_attr = onnx.helper.make_attribute(
                        attr.name, new_tensor, attr.doc_string
                    )
                    node.ClearField("attribute")
                    node.attribute.extend([new_attr])

            elif node.op_type in ["Conv", "MatMul"]:
                # Nodes supporting an int operation are converted to their integer counterpart
                new_op_type = (
                    "ConvInteger" if node.op_type == "Conv" else "MatMulInteger"
                )
                new_node = onnx_helper.make_node(
                    new_op_type,
                    inputs=node.input,
                    outputs=node.output,
                    name = node.name.replace("Conv", "ConvInteger").replace("MatMul", "MatMulInteger"),
                )
                new_node.attribute.extend(node.attribute)
                onnxproto.graph.node.remove(node)
                onnxproto.graph.node.insert(idx, new_node)

            elif node.op_type in ["Pad"] and node.output[0] in metadata:
                # Pad nodes should have the correct value for padding,
                # i.e. an int, but only if the following convolution is a quantized one
                # NOTE: This operator changes with opset 11. We HAVE TO rewrite it

                val = 0
                mode = "constant"
                pads = [0, 0, 0, 0]
                for attr in node.attribute:
                    if attr.name == "value":
                        val = int(attr.f)
                    elif attr.name == "mode":
                        mode = attr.s.decode("utf-8")
                    elif attr.name == "pads":
                        pads = list(attr.ints)
                meta_node = metadata[node.output[0]]
                precision = meta_node["precision"]
                signed = meta_node["signed"]
                dtype = self._bound_to_onnx(precision, signed)
                value_tensor_proto = onnx.helper.make_tensor(
                    name=node.name + "_value",
                    data_type=dtype,
                    dims=[1],
                    vals=[val],
                )
                pads_tensor_proto = onnx.helper.make_tensor(
                    name=node.name + "_pads",
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(pads)],
                    vals=np.array(pads, dtype=np.int64),
                )

                new_node = onnx_helper.make_node(
                    "Pad",
                    inputs=[
                        node.input[0],
                        pads_tensor_proto.name,
                        value_tensor_proto.name,
                    ],
                    outputs=node.output,
                    mode=node.attribute[0].s.decode("utf-8"),
                    name=node.name,
                )
                onnxproto.graph.initializer.extend(
                    [value_tensor_proto, pads_tensor_proto]
                )
                onnxproto.graph.node.remove(node)
                onnxproto.graph.node.insert(idx, new_node)

            elif node.op_type in ["Clip"]:
                # Another node changing with opset 11
                prec = metadata[node.output[0]]["precision"]
                if prec is None:
                    # Only quantized nodes are supported
                    continue
                signed = metadata[node.input[0]]["signed"]
                dtype = self._bound_to_onnx(32, True)
                min_tens = onnx.helper.make_tensor(
                    name=node.name + "_min",
                    data_type=dtype,
                    dims=[],
                    vals=[int(node.attribute[1].f)],
                )
                max_tens = onnx.helper.make_tensor(
                    name=node.name + "_max",
                    data_type=dtype,
                    dims=[],
                    vals=[int(node.attribute[0].f)],
                )
                new_node = onnx_helper.make_node(
                    "Clip",
                    inputs=[node.input[0], min_tens.name, max_tens.name],
                    # This should be a int32
                    outputs=node.output,
                    name=node.name,
                )
                new_cast_output = node.output[0].replace("Clip", "Cast").replace("clip", "cast")
                prec_out = metadata[node.output[0]]["precision"]
                signed_out = metadata[node.output[0]]["signed"]
                new_cast_node = onnx_helper.make_node(
                    "Cast",
                    inputs=[node.output[0]],
                    outputs=[new_cast_output],
                    to=self._bound_to_onnx(prec_out, signed_out),
                    name=node.name.replace("Clip", "Cast").replace("clip", "cast"),
                )
                cast_to_add_map[node.output[0]] = new_cast_output
                cast_to_add.append(new_cast_node)

                onnxproto.graph.initializer.extend([min_tens, max_tens])
                onnxproto.graph.node.remove(node)
                onnxproto.graph.node.insert(idx, new_node)
            elif node.op_type in ["Floor", "Ceil"]:
                if node.input[0] in metadata and metadata[node.input[0]]["precision"] is not None:
                    # Not needed anymore for integerized models
                    connection_pruned_nodes_map[node.output[0]] = node.input[0]
                    nodes_to_prune.append(node)

        # Pruned the nodes that are not needed anymore
        for node in onnxproto.graph.node:
            node.input[:] = [connection_pruned_nodes_map.get(inp, inp) for inp in node.input]
            node.input[:] = [cast_to_add_map.get(inp, inp) for inp in node.input]
        for node in nodes_to_prune:
            onnxproto.graph.node.remove(node)
        onnxproto.graph.node.extend(cast_to_add)
        onnxproto = self._topological_sort(onnxproto)
        # Clear the value_info
        onnxproto.graph.ClearField("value_info")
        # Re-infer shapes and dtypes
        onnxproto = shape_inference.infer_shapes(onnxproto)
        # Return the modified onnxproto
        return onnxproto

    def _bound_to_np(self, bits : int, is_signed:bool):
        if is_signed and bits <= 8:
            dtype = np.int8
        elif not is_signed and bits <= 8:
            dtype = np.uint8
        elif is_signed:
            dtype = np.int32
        else:
            dtype = np.uint32
        return dtype

    def _bound_to_onnx(self, bits : int, is_signed:bool):
        if is_signed and bits <= 8:
            dtype = onnx.TensorProto.INT8
        elif not is_signed and bits <= 8:
            dtype = onnx.TensorProto.UINT8
        elif is_signed:
            dtype = onnx.TensorProto.INT32
        else:
            dtype = onnx.TensorProto.UINT32
        return dtype

    def _topological_sort(self, onnxproto : onnx.ModelProto):
        """
        Perform a topological sort of the ONNX graph
        """
        from collections import defaultdict, deque
        graph = onnxproto.graph
        adj_list = defaultdict(list)
        in_degree = defaultdict(int)
        node_map = {node.name: node for node in graph.node}


        # Adjacency list
        for node in graph.node:
            for output in node.output:
                for next_node in graph.node:
                    if output in next_node.input:
                        adj_list[node.name].append(next_node.name)
                        in_degree[next_node.name] += 1

        queue = deque([node.name for node in graph.node if in_degree[node.name] == 0])
        print(queue)
        print(adj_list)
        print(in_degree)

        sorted_nodes = []
        while queue:
            node_name = queue.popleft()
            sorted_nodes.append(node_map[node_name])

            for neighbour in adj_list[node_name]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if len(sorted_nodes) != len(graph.node):
            print(len(sorted_nodes), len(graph.node))
            raise ValueError("Graph contains a cycle, something broke")

        graph.ClearField("node")
        graph.node.extend(sorted_nodes)
        return onnxproto

