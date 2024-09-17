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
# * Author:  Beatrice Alessandra Motetti <beatrice.motetti@polito.it>          *
# *----------------------------------------------------------------------------*

import copy
from typing import Dict, cast

import torch

from plinio.cost.cost_spec import CostFn, CostSpec
from plinio.graph.inspection import shapes_dict
from plinio.methods.mps.mps import MPS
from plinio.methods.mps.nn.identity import MPSIdentity
from plinio.methods.mps.nn.module import MPSModule
from plinio.methods.mps.nn.qtz import MPSPerChannelQtz, MPSPerLayerQtz


def optimize_prec_assignment(model: MPS,
                             name: str = "ne16"):
    """Reassign to a higher precision the channels of the MPS layers of the model,
    if it decreases the cost. Currently, only the NE16 cost model is supported.

    :param model: the MPS model
    :type model: MPS
    :param name: the cost model name, defaults to "ne16"
    :type name: str, optional
    :return: the model with the reassigned precisions
    :rtype: MPS
    """
    assert name.lower() == "ne16", "This method currently only supports the NE16 cost model"

    # Modify the sampling strategy to be argmax before the precisions reassignment.
    # Perform a dummy forward pass to ensure the theta alpha values are updated.
    model.update_softmax_options(hard=True)
    model(model._input_example)

    with torch.no_grad():
        # Retrieve the cost function and the cost specification
        assert isinstance(model._cost_specification, dict), "Cost specification not found"
        cost_spec = cast(CostSpec, model._cost_specification[name])
        cost_fn_map = cast(Dict[str, CostFn], model._cost_fn_map[name])

        # Initialize
        base_model_cost = torch.tensor(0, dtype=torch.float32)
        best_model_cost = torch.tensor(0, dtype=torch.float32)

        # Iterate over the leaf modules and reassign the precisions in each layer, if needed
        target_list = model._unique_leaf_modules if cost_spec.shared else model._leaf_modules
        for lname, node, layer in target_list:
            if isinstance(layer, MPSModule):
                # Compute the cost of the layer in the original configuration
                l_cost = layer.get_cost(cost_fn_map[lname], shapes_dict(node))
                base_cost = model._cost_reduction_fn(l_cost)
                base_model_cost += base_cost.detach()

                if isinstance(layer, MPSIdentity) or isinstance(layer.w_mps_quantizer, MPSPerLayerQtz):
                    continue

                if isinstance(layer.w_mps_quantizer, MPSPerChannelQtz):
                    w_theta_alpha_array = layer.w_mps_quantizer.theta_alpha.mean(dim=1)
                else:
                    raise ValueError("Unsupported quantizer type")

                best_cost = copy.deepcopy(base_cost)
                best_cost_w_theta_alpha_array = copy.deepcopy(w_theta_alpha_array)
                config_cost = _compute_cost(model, layer, w_theta_alpha_array, cost_fn_map, lname, node)
                assert config_cost == base_cost, "The cost of the layer is not consistent with the original configuration"

                sorted_indexes = torch.argsort(layer.w_mps_quantizer.precision)
                sorted_precisions = [layer.w_mps_quantizer.precision[i] for i in sorted_indexes]

                # Case 1: assign a channel at a time to a higher precision. Save the configuration if the cost decreases
                w_theta_alpha_array_tmp = [copy.deepcopy(w_theta_alpha_array)[i] for i in sorted_indexes]

                for i in range(len(sorted_precisions)):
                    if sorted_precisions[i] == 0:
                        continue
                    for j in range(i + 1, len(sorted_precisions)):
                        w_theta_alpha_array_tmp = [copy.deepcopy(w_theta_alpha_array)[i] for i in sorted_indexes]
                        while w_theta_alpha_array_tmp[i] > 0:
                            w_theta_alpha_array_tmp[i] -= (1. / layer.w_mps_quantizer.theta_alpha.shape[1])
                            w_theta_alpha_array_tmp[j] += (1. / layer.w_mps_quantizer.theta_alpha.shape[1])
                            cost_tmp = _compute_cost(model, layer, w_theta_alpha_array_tmp, cost_fn_map, lname, node)
                            if cost_tmp < best_cost:
                                best_cost = cost_tmp
                                best_cost_w_theta_alpha_array = copy.deepcopy(w_theta_alpha_array_tmp) # TODO: check sorting!!!
                                print("* Layer '{}' cost decreased from {} to {} with the following channels counts for each precision:"
                                      "\n\tprecisions: {}"
                                      "\n\toriginal:   {}"
                                      "\n\tnew:        {}".format(
                                          lname, base_cost, best_cost,
                                          torch.stack(sorted_precisions).tolist(),
                                          torch.mul(w_theta_alpha_array, layer.w_mps_quantizer.alpha.shape[1]).tolist(),
                                          torch.mul(torch.stack(best_cost_w_theta_alpha_array), layer.w_mps_quantizer.alpha.shape[1]).tolist()))


                # Case 2: assign a channel at a time to a higher precision, iteratively for each precision.
                # Save the configuration if the cost decreases
                w_theta_alpha_array_tmp = [copy.deepcopy(w_theta_alpha_array)[i] for i in sorted_indexes]

                for i in range(len(sorted_precisions)):
                    if sorted_precisions[i] == 0:
                        continue
                    for j in range(i + 1, len(sorted_precisions)):
                        while w_theta_alpha_array_tmp[i] > 0:
                            w_theta_alpha_array_tmp[i] -= (1. / layer.w_mps_quantizer.theta_alpha.shape[1])
                            w_theta_alpha_array_tmp[j] += (1. / layer.w_mps_quantizer.theta_alpha.shape[1])
                            cost_tmp = _compute_cost(model, layer, w_theta_alpha_array_tmp, cost_fn_map, lname, node)
                            if cost_tmp < best_cost:
                                best_cost = cost_tmp
                                best_cost_w_theta_alpha_array = copy.deepcopy(w_theta_alpha_array_tmp)
                                print("* Layer '{}' cost decreased from {} to {} with the following channels counts for each precision:"
                                      "\n\tprecisions: {}"
                                      "\n\toriginal:   {}"
                                      "\n\tnew:        {}".format(
                                          lname, base_cost, best_cost,
                                          torch.stack(sorted_precisions).tolist(),
                                          torch.mul(w_theta_alpha_array, layer.w_mps_quantizer.alpha.shape[1]).tolist(),
                                          torch.mul(torch.stack(best_cost_w_theta_alpha_array), layer.w_mps_quantizer.alpha.shape[1]).tolist()))

                best_model_cost += best_cost

                # Sort the best configuration according to the original order of the precisions
                best_theta_alpha_array = torch.tensor([best_cost_w_theta_alpha_array[i] for i in sorted_indexes])
                best_theta_alpha_array = torch.mul(best_theta_alpha_array, layer.w_mps_quantizer.theta_alpha.shape[1])

                # Update the layer with the best configuration.
                # Modify only the alpha parameter of each layer, and not the theta_alpha, to avoid
                # any conflict in the case of parallel branches.
                new_alpha = _reassign_precisions(best_theta_alpha_array, layer.w_mps_quantizer.alpha)
                layer.w_mps_quantizer.alpha.data = new_alpha.clone()

            elif model.full_cost:
                # TODO: this is constant and can be pre-computed for efficiency
                # TODO: should we add default bitwidth and format for non-MPS layers or not?
                v = vars(layer)
                v.update(shapes_dict(node))
                base_model_cost = base_model_cost + cost_fn_map[lname](v)

    # Update the theta_alpha parameters with a dummy forward pass
    model(model._input_example)
    print("Model cost decreased from {} to {}".format(base_model_cost.item(), best_model_cost.item()))

    return model


def _compute_cost(model, layer, w_theta_alpha_array, cost_fn_map, lname, node):
    """Compute the cost of the layer with the given configuration. The cost is computed
    as the product of the theta alpha values of the input and weight quantizers and the
    cost function of the layer.
    """
    cost = torch.zeros((len(layer.in_mps_quantizer.precision),
                        len(layer.w_mps_quantizer.precision)))
    for i, (in_prec, in_theta_alpha) in enumerate(zip(layer.in_mps_quantizer.precision,
                                                    layer.in_mps_quantizer.theta_alpha)):
        for j, (w_prec, w_theta_alpha) in enumerate(zip(layer.w_mps_quantizer.precision,
                                                        w_theta_alpha_array)):
            spec = layer.get_modified_vars()
            spec['in_format'] = int
            spec['w_format'] = int
            spec['in_precision'] = in_prec
            spec['w_precision'] = w_prec
            spec['w_theta_alpha'] = w_theta_alpha
            spec.update(shapes_dict(node))
            cost[i][j] = in_theta_alpha * w_theta_alpha * cost_fn_map[lname](spec)
    config_cost = model._cost_reduction_fn(cost)
    return config_cost


def _reassign_precisions(best, scores):
    """Reassign the precisions of the channels based on the given alpha values.
    The reassignment algorithm is greedy, and tries to assign to each precision the
    channels which have the highest alpha value for that precision.
    """
    num_precisions, num_channels = scores.size()

    # Extract the current assignments (precision with the highest alpha for each channel).
    # Then, sort the channels for each precision by their alpha values
    current_assignment = torch.argmax(scores, dim=0)
    sorted_indices = torch.argsort(scores, dim=1, descending=True)
    new_assignment = current_assignment.clone()  # Empty list to store new assignments

    # Enforce the new cardinality
    for prec in range(num_precisions):
        # Get the number of channels that should be assigned to this precision
        target_count = int(best[prec].item())

        # If no channels must have this precision, reassign all the channels at the
        # current precision.
        if target_count == 0:
            prec_indices = (current_assignment == prec).nonzero(as_tuple=True)[0]
            new_assignment[prec_indices] = -1  # Temporarily mark as unassigned
            continue

        # If at least one channel should have the current precision, assign the top
        # 'target_count' channels to this precision.
        # First, get the indices of channels currently assigned to this precision and
        # the top 'target_count' channels for this precision based on the alpha values
        prec_indices = (current_assignment == prec).nonzero(as_tuple=True)[0]
        top_indices = sorted_indices[prec][:target_count]

        # Assign those top channels to this precision
        new_assignment[top_indices] = prec

        # Reassign the remaining channels
        excess_channels = prec_indices[target_count:]
        if len(excess_channels) > 0:
            new_assignment[excess_channels] = -1  # Temporarily mark as unassigned

    # Reassign channels marked as unassigned to precisions that need more channels
    for prec in range(num_precisions):
        target_count = int(best[prec].item())
        current_count = (new_assignment == prec).sum().item()

        # If there are not enough channels assigned to this precision, use the unassigned channels
        if current_count < target_count:
            unassigned_channels = (new_assignment == -1).nonzero(as_tuple=True)[0]
            channels_needed = target_count - current_count

            # Get the top 'channels_needed' channels for this precision and reassign them
            top_unassigned = sorted_indices[prec][torch.isin(sorted_indices[prec], unassigned_channels)][:channels_needed]
            new_assignment[top_unassigned] = prec

    # Create the binary assignment matrix, that will replace the original alpha matrix
    binary_matrix = torch.zeros_like(scores)

    for channel in range(num_channels):
        assigned_prec = new_assignment[channel]
        if assigned_prec != -1:  # Only assign if the channel has been reassigned
            binary_matrix[assigned_prec, channel] = 1
    return binary_matrix