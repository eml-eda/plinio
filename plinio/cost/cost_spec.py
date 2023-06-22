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
# * Author:  Daniele Jahier Pagliari <daniele.jahier@polito.it>                *
# *----------------------------------------------------------------------------*
from typing import Callable, Tuple
from collections import UserDict
import torch
from .pattern import Constraint, Pattern, PatternSpec


CostFn = Callable[[PatternSpec], torch.Tensor]


def cost_spec_zero_fn(_):
    return torch.tensor([0.0])


def cost_spec_fail_fn(x):
    raise KeyError(f"Cannot find cost model for pattern {x}")


class CostSpec(UserDict):
    """Class to wrap a PLiNIO Cost Specification

    The cost specification is basically a container class which includes:
        * a dictionary mapping a layer (for now, later it will be a pattern) to a differentiable
        function producing a scalar cost
        * a series of additional configuration parameters
            - shared: True if the cost of a layer should be evaluated once for the whole NN
            (e.g. Params), False if the cost model should be evaluated once for each time the
            layer is invoked during a forward pass (e.g. MACs)

    Documentation TBD!
    """
    def __init__(
            self,
            shared: bool = True,
            default_behavior: str = 'zero'
    ):
        super(CostSpec, self).__init__()
        self.shared = shared
        if default_behavior == 'zero':
            self.default = cost_spec_zero_fn
        elif default_behavior == 'fail':
            self.default = cost_spec_fail_fn
        else:
            raise ValueError(f"Unknown default behavior {default_behavior}")

    def __setitem__(self,
                    key: Tuple[Pattern, Constraint],
                    cost_fn: CostFn):
        """Associates a cost function to a pattern + constraint pair"""
        if key[0] not in self.data:
            self.data[key[0]] = []
        self.data[key[0]].append((key[1], cost_fn))

    def __getitem__(self, key: Tuple[Pattern, PatternSpec]):
        """Finds the most accurate cost function for a given pattern + spec"""
        best_match = self.default
        best_constr = None
        if key[0] in self.data:
            for constr, cost_fn in self.data[key[0]]:
                if constr is None or constr(key[1]):
                    if best_constr is None:
                        best_match = cost_fn
                        best_constr = constr
                    else:
                        # fail if we have two incompatible models, e.g., one for 3x3 convs
                        # and one for DWConvs, and we are processing a DW3x3Conv.
                        raise KeyError("Found two conflicting cost models! Terminating")
        return best_match
