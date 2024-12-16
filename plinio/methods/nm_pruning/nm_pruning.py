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
# * Author:  Francesco Daghero <francesco.daghero@polito.it>                   *
# *----------------------------------------------------------------------------*

from typing import Any, Tuple, Type, Iterable, Dict, Optional
import torch
import torch.nn as nn

from plinio.methods.dnas_base import DNAS
from .graph import convert


class NMPruning(DNAS):
    """A class that wraps a nn.Module for pruning

    :param model: the inner nn.Module to be pruned
    :type model: nn.Module
    :param input_example: an example input tensor, required for symbolic tracing
    :type input_example: torch.Tensor
    :param input_shape: the shape of an input tensor, without batch size, required for symbolic
    tracing
    :type input_shape: Tuple[int, ...]
    :param autoconvert_layers: should the constructor try to autoconvert prunable layers,
    defaults to True
    :type autoconvert_layers: bool, optional
    :param n: number of non-zero parameters in a group
    :type n: int
    :param m: group of weights considered for pruning
    :type m: int
    :param pruning_decay: the decay factor for the pruning mask
    :type pruning_decay: float
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS,
    defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS,
    defaults to ()
    :type exclude_types: Iterable[Type[nn.Module]], optional
    """

    def __init__(
        self,
        model: nn.Module,
        input_example: Optional[Any] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        autoconvert_layers: bool = True,
        n: int = 1,
        m: int = 1,
        pruning_decay: float = 0.0002,
        exclude_names: Iterable[str] = (),
        exclude_types: Iterable[Type[nn.Module]] = (),
    ):
        super(NMPruning, self).__init__(model, {}, input_example, input_shape)
        self.is_training = model.training
        self.seed, self._leaf_modules, self._unique_leaf_modules = convert(
            model,
            self._input_example,
            "autoimport" if autoconvert_layers else "import",
            n,
            m,
            pruning_decay,
            exclude_names,
            exclude_types,
        )
        self.n = n
        self.m = m
        self.pruning_decay = pruning_decay

        # Restore training status after forced `eval()` in convert
        if self.is_training:
            self.train()
            self.seed.train()
        else:
            self.eval()
            self.seed.eval()

    def forward(self, *args: Any) -> torch.Tensor:
        """Forward function for the model.
        Simply invokes the inner model's forward

        :return: the output tensor
        :rtype: torch.Tensor
        """
        return self.seed.forward(*args)

    def export(self):
        """Export the architecture as a torch.nn.Module

        :return: the precision-assignement found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        mod, _, _ = convert(
            self.seed, self._input_example, "export", self.n, self.m, self.pruning_decay
        )
        return mod

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the network hyperparameters

        :return: a dictionary containing the layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            "n": self.n,
            "m": self.m,
            "pruning_decay": self.pruning_decay,
        }
