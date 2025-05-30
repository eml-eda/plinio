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

import plinio.methods.mps.quant.nn as qnn
import plinio.methods.mps.quant.backends.match.nn as match_nn

# add new supported layers here:
match_layer_map: Dict[Type[nn.Module], Type[match_nn.MATCHModule]] = {
    qnn.QuantConv1d: match_nn.MATCHConv1d,
    qnn.QuantConv2d: match_nn.MATCHConv2d,
    qnn.QuantConv3d: match_nn.MATCHConv3d,
    qnn.QuantLinear: match_nn.MATCHLinear,
    qnn.QuantAdd: match_nn.MATCHAdd,
}
