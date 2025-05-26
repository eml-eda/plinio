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
from typing import Any, Callable, Dict, Optional, Type
import torch.nn as nn

# example = {'in_channels': 32, 'out_channels': 64}
# the keys include:
#  - all keys in vars(c), i.e., _parameters, in_channels, out_channels, etc.
#  - 'output_shapes' and 'input_shapes' which contain two lists of effective input/output shapes
#  - 'input_dtype', 'output_dtype', 'parameters_dtype', which contain the dtype (float or int)
# of each tensor
#  - 'input_bits', 'output_bits', 'parameters_bits', which contain the effective bitwidth of each
# tensor
PatternSpec = Dict[str, Any]

# later can become a real pattern, for now a single layer
Pattern = Type[nn.Module]


Constraint = Optional[Callable[[PatternSpec], bool]]


# examples of constraint: depthwise separable convolution
def conv_dw_constraint(spec: PatternSpec):
    return spec['in_channels'] == spec['groups'] and spec['out_channels'] == spec['groups']


# another example: conv with 3 or 3x3 filter (useless at the moment)
def conv_3_constraint(spec: PatternSpec):
    k = spec['kernel_size']
    return all([ki == 3 for ki in k])


# Patterns/constraints pairs definition
Conv1dGeneric = (nn.Conv1d, None)
Conv2dGeneric = (nn.Conv2d, None)
Conv3dGeneric = (nn.Conv3d, None)
LinearGeneric = (nn.Linear, None)
Conv1dDW = (nn.Conv1d, conv_dw_constraint)
Conv2dDW = (nn.Conv2d, conv_dw_constraint)
Conv1d3 = (nn.Conv1d, conv_3_constraint)
Conv2d3x3 = (nn.Conv2d, conv_3_constraint)
