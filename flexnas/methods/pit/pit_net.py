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

from typing import Tuple, Dict, Iterable, Any
import torch
import torch.nn as nn
from flexnas.methods.dnas_base import DNASNet
from .pit_conv1d import PITConv1d

class PITNet(DNASNet):

    replacement_dict = {
        nn.Conv1d : PITConv1d,
    }

    def __init__(self, model: nn.Module, config = None, exclude_names: Iterable[str] = (), exclude_types: Iterable[nn.Module] = ()):
        super(PITNet, self).__init__(model, config, exclude_names, exclude_types)
        
    def optimizable_layers(self) -> Tuple[nn.Module]:
        return tuple(PITNet.replacement_dict.keys())

    def replacement_layer(self, name: str, layer: nn.Module, model: nn.Module) -> nn.Module:
        for OldClass, NewClass in PITNet.replacement_dict.items():
            if isinstance(layer, OldClass):
                return NewClass(layer, self.config)
        return None

    def get_regularization_loss(self, net: nn.Module) -> torch.Tensor:
        raise NotImplementedError