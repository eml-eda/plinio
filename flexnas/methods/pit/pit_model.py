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

from typing import Tuple, Type, Iterable, Optional
import torch
import torch.nn as nn
from flexnas.methods.dnas_base import DNASModel
from .pit_conv1d import PITConv1d


class PITModel(DNASModel):

    replacement_dict = {nn.Conv1d: PITConv1d}
    regularizers = ('size', 'flops')

    def __init__(
            self,
            model: nn.Module,
            regularizer: Optional[str] = 'size',
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            train_channels=True,
            train_rf=True,
            train_dilation=True):
        super(PITModel, self).__init__(model, regularizer, exclude_names, exclude_types)
        self.train_channels = train_channels
        self.train_rf = train_rf
        self.train_dilation = train_dilation

    def supported_regularizers(self) -> Tuple[str, ...]:
        return PITModel.regularizers

    def optimizable_layers(self) -> Tuple[Type[nn.Module], ...]:
        return tuple(PITModel.replacement_dict.keys())

    def replacement_layer(self, name: str, layer: nn.Module, model: nn.Module) -> nn.Module:
        for OldClass, NewClass in PITModel.replacement_dict.items():
            if isinstance(layer, OldClass):
                return NewClass(layer)
        raise ValueError("Replacement Layer not found")

    def get_regularization_loss(self) -> torch.Tensor:
        raise NotImplementedError
