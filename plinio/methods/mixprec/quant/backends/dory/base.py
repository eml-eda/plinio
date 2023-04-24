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
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *----------------------------------------------------------------------------*

from typing import Type, Dict

import torch.nn as nn

import quant.nn as qnn
import dory.nn as dory_nn

# add new supported layers here:
dory_layer_map: Dict[Type[nn.Module], Type[dory_nn.DORYModule]] = {
    qnn.Quant_Conv2d: dory_nn.DORYConv2d,
    qnn.Quant_Linear: dory_nn.DORYLinear,
    qnn.Quant_Identity: dory_nn.DORYIdentity,
}
