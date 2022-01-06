#*----------------------------------------------------------------------------*
#* Copyright (C) 2022 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Daniele Jahier Pagliari <daniele.jahier@polito.it>                *
#*----------------------------------------------------------------------------*

from typing import Iterable, List
import torch
import torch.nn as nn
from flexnas.methods import DNAS
from flexnas.layers.pit import *

class PIT(DNAS):

    replacement_dict = {
        nn.Conv1d : PITConv1d,
    }

    def __init__(self, config = None):
        super(PIT, self).__init__(config)
        
    def optimizable_layers(self) -> List[nn.Module]:
        return list(PIT.replacement_dict.keys())

    def _replacement_layer(self, layer: nn.Module, net: nn.Module) -> nn.Module:
        raise NotImplementedError

    def get_regularization_loss(self, net: nn.Module) -> torch.Tensor:
        raise NotImplementedError