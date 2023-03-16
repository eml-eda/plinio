# *----------------------------------------------------------------------------*
# * Copyright (C) 2021 Politecnico di Torino, Italy                            *
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
# * Author:  Daniele Jahier Pagliari <daniele.jahier@polito.it>                *
# *----------------------------------------------------------------------------*
from typing import List, Callable
import math
import torch.fx as fx
from .features_calculation import FlattenFeaturesCalculator, ConcatFeaturesCalculator, \
    ConstFeaturesCalculator
from .utils import try_get_args
from .inspection import is_features_propagating_op, is_features_defining_op, \
    is_shared_input_features_op, is_flatten, is_squeeze, is_features_concatenate, \
    is_untouchable_op, is_zero_or_one_input_op, get_graph_inputs, all_output_nodes


def add_node_properties(mod: fx.GraphModule):
    """Adds properties to a dict of each node in the graph by calling ad-hoc function

    :param mod: module
    :type mod: fx.GraphModule
    """
    g = mod.graph
    queue = get_graph_inputs(g)
    visited = []
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue

        add_single_node_properties(n, mod)

        for succ in all_output_nodes(n):
            queue.append(succ)
        visited.append(n)


def add_single_node_properties(n: fx.Node, mod: fx.GraphModule):
    n.meta['features_propagating'] = is_features_propagating_op(n, mod)
    n.meta['features_defining'] = is_features_defining_op(n, mod)
    n.meta['shared_input_features'] = is_shared_input_features_op(n, mod)
    n.meta['flatten'] = is_flatten(n, mod)
    n.meta['squeeze'] = is_squeeze(n, mod)
    n.meta['features_concatenate'] = is_features_concatenate(n, mod)
    n.meta['untouchable'] = is_untouchable_op(n)
    n.meta['zero_or_one_input'] = is_zero_or_one_input_op(n)


def add_features_calculator(mod: fx.GraphModule, extra_rules: List[Callable] = []):
    """Adds a different feature calculator object in the 'meta' dict of each node of the graph
    depending on the node properties

    :param mod: module
    :type mod: fx.GraphModule
    :param extra_rules: list of callable returning NAS specific features calculator
    :type extra_rules: List[Callable]
    :raises ValueError: for unsupported nodes
    """
    g = mod.graph
    queue = get_graph_inputs(g)
    visited = []
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        # skip nodes for which predecessors have not yet been processed completely
        if any(['features_calculator' not in _.meta for _ in n.all_input_nodes]):
            continue

        # handle extra rules
        fc = None
        for rule in extra_rules:
            fc = rule(n, mod)
            if fc:
                break
        if fc:
            n.meta['features_calculator'] = fc
        # handle default rules
        elif n.meta['flatten']:
            # For flatten ops, the output features are computed as: input_features * spatial_size
            # note that this is NOT simply equal to the output shape if the preceding layer is a
            # NAS-able one, for which some features # could be de-activated
            ifc = n.all_input_nodes[0].meta['features_calculator']
            input_shape = n.all_input_nodes[0].meta['tensor_meta'].shape
            start_dim = try_get_args(n, 1, 'start_dim', 0)
            end_dim = try_get_args(n, 2, 'end_dim', -1)
            assert start_dim != 0 and len(input_shape) - start_dim != 0, \
                "Flattening the batch not supported"
            # if flatten includes the channels
            if start_dim == 1 or len(input_shape) - start_dim == 1:
                flattened_size = math.prod(input_shape[2:end_dim if end_dim != -1 else None])
                n.meta['features_calculator'] = FlattenFeaturesCalculator(ifc, int(flattened_size))
            else:
                n.meta['features_calculator'] = ifc  # just propagate the features
        elif n.meta['squeeze']:
            # Squeeze is similar to flatten but the pytorch operation is slightly different
            ifc = n.all_input_nodes[0].meta['features_calculator']
            input_shape = n.all_input_nodes[0].meta['tensor_meta'].shape
            dim = try_get_args(n, 1, 'dim', None)
            # TODO: add support for no dim by looking at which dimensions are 1
            if dim is None:
                raise ValueError("Squeeze without dim not supported")
            assert dim != 0 and len(input_shape) - dim != 0, \
                "Squeezing the batch is not supported"
            if dim == 1 or len(input_shape) - dim == 1:
                flattened_size = input_shape[2]
                n.meta['features_calculator'] = FlattenFeaturesCalculator(ifc, flattened_size)
            else:
                n.meta['features_calculator'] = ifc  # just propagate the features
        elif n.meta['features_concatenate']:
            # for concatenation over the features axis the number of output features is the sum
            # of the output features of preceding layers as for flatten, this is NOT equal to the
            # input shape of this layer, when one or more predecessors are NAS-able
            ifc = ConcatFeaturesCalculator(
                [prev.meta['features_calculator'] for prev in n.all_input_nodes]
            )
            n.meta['features_calculator'] = ifc
        elif n.meta['shared_input_features']:
            # for nodes that require identical number of features in all their inputs (e.g., add)
            # we simply assume that we can take any of the output features calculators from
            # predecessors
            # this is enforced for NAS-able layers by the use of shared maskers
            n.meta['features_calculator'] = n.all_input_nodes[0].meta['features_calculator']
        elif n.meta['features_defining']:
            # these are "static" (i.e., non NAS-able) nodes that alter the number of output
            # features, and hence the number of input features of subsequent layers
            n.meta['features_calculator'] = ConstFeaturesCalculator(n.meta['tensor_meta'].shape[1])
        elif n.meta['features_propagating']:
            # these are nodes that have a single input and n. output features == n. input features
            # so, we just propagate forward the features calculator of the input
            # this also includes PITBatchNorm1d and PITBatchNorm2d
            n.meta['features_calculator'] = n.all_input_nodes[0].meta['features_calculator']
        else:
            raise ValueError("Unsupported node {} (op: {}, target: {})".format(n, n.op, n.target))

        for succ in all_output_nodes(n):
            queue.append(succ)
        visited.append(n)


def associate_input_features(mod: fx.GraphModule):
    """Associates to each node a reference to the node that sets its features

    :param mod: module
    :type mod: fx.GraphModule
    :raises ValueError: for unsupported nodes
    """
    g = mod.graph
    queue = get_graph_inputs(g)
    visited = []
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        # skip nodes for which predecessors have not yet been processed completely
        if any(['input_features_set_by' not in _.meta for _ in n.all_input_nodes]):
            continue

        prev = n if len(n.all_input_nodes) == 0 else n.all_input_nodes[0]

        if len(n.all_input_nodes) == 0:  # input node
            n.meta['input_features_set_by'] = n
        elif n.meta['features_concatenate']:
            n.meta['input_features_set_by'] = n.all_input_nodes
        elif prev.meta['flatten']:
            input_shape = prev.all_input_nodes[0].meta['tensor_meta'].shape
            start_dim = try_get_args(prev, 1, 'start_dim', 0)
            assert start_dim != 0 and len(input_shape) - start_dim != 0, \
                "Flattening the batch not supported"
            # if flatten includes the channels
            if start_dim == 1 or len(input_shape) - start_dim == 1:
                n.meta['input_features_set_by'] = prev
            else:
                n.meta['input_features_set_by'] = prev.meta['input_features_set_by']
        elif prev.meta['squeeze']:
            input_shape = prev.all_input_nodes[0].meta['tensor_meta'].shape
            dim = try_get_args(prev, 1, 'dim', None)
            if dim is None:
                raise ValueError("Squeeze without dim not supported")
            assert dim != 0 and len(input_shape) - dim != 0, \
                "Squeezing the batch is not supported"
            if dim == 1 or len(input_shape) - dim == 1:
                n.meta['input_features_set_by'] = prev
            else:
                n.meta['input_features_set_by'] = prev.meta['input_features_set_by']
        elif prev.meta['features_concatenate']:
            n.meta['input_features_set_by'] = prev
        elif prev.meta['features_defining']:
            n.meta['input_features_set_by'] = prev
        elif prev.meta['features_propagating']:
            n.meta['input_features_set_by'] = prev.meta['input_features_set_by']
        else:
            raise ValueError("Unsupported node {} (op: {}, target: {})"
                             .format(n, n.op, n.target))

        for succ in all_output_nodes(n):
            queue.append(succ)
        visited.append(n)
