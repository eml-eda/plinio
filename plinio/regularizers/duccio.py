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

from typing import Tuple, Optional, Dict
import torch
from plinio.methods import DNAS

class DUCCIO():
    def __init__(self,
                 targets: Dict[str, torch.Tensor],
                 task_loss: Optional[torch.Tensor] = None,
                 final_strengths: Optional[Tuple[torch.Tensor,...]] = None):
        """An implementation of the DUCCIO regularization method proposed in:
            https://arxiv.org/abs/2206.00302
        The regularization loss term computed by DUCCIO is:
        sum(max(0, (cost_i - target_i)) * strength) for each target cost metric i.

        :param targets: the target values for each cost metric, specified as a {name: value}
        dictionary, where "name" is the name of a cost metric in the target model, and "value"
        is a scalar tensor
        :type targets: Dict[str, torch.Tensor]
        :param task_loss: the task loss value at the beginning of the search, used to compute the
        final regularization strength value for each cost metric. Defaults to 1e-3.
        :type task_loss: torch.Tensor, optional
        :param final_strengths: the final regularization strength values for each cost metric,
        as a tuple of tensors. If specified, this tuple must have the same number of elements
        as there are values in targets. If not specified, the final strength values are derived
        from the task_loss
        :type final_strengths: Optional[Tuple[torch.Tensor,...]], optional
        """
        if final_strengths is not None and len(targets.values()) != len(final_strengths):
            raise ValueError("The lengths of targets and final strengths must match")
        if task_loss is None and final_strengths is None:
            raise ValueError("Either task_loss or final_strengths must be specified")
        if task_loss is not None and final_strengths is not None:
            raise ValueError("Either task_loss or final_strengths must be None")
        self.targets = targets
        self.final_strengths = final_strengths
        self.task_loss = task_loss

    def __call__(self, model: DNAS, epoch: int = 1, n_epochs: int = 1) -> torch.Tensor:
        """Computes the regularization loss term, using the annealing method described in the paper.

        :param model: the model to regularize
        :type model: DNAS
        :param epoch: the current epoch, used to anneal the regularization strength, defaults to 1.
        :type epoch: int, optional
        :param n_epochs: the total number of epochs, used to anneal the regularization strength,
        defaults to 1.
        strength is not annealed and the final value is used
        :type n_epochs: int, optional
        :return: the regularization loss value
        :rtype: torch.Tensor
        """
        # initialize final strengths on first call, if not done explicitly at construction
        with torch.no_grad():
            if self.final_strengths is None:
                self.final_strengths = tuple(torch.maximum(torch.tensor(0.0), self.task_loss / (model.get_cost(n) - t)) for n, t in self.targets.items())

        cost = torch.tensor(0.0)
        for (cost_name, target), strength in zip(self.targets.items(), self.final_strengths):
            eff_strength = torch.min(strength/100 + epoch * (strength*99/100) / (n_epochs / 2), strength)
            cost = cost + (eff_strength * torch.maximum(torch.tensor(0.0), model.get_cost(cost_name) - target))
        return cost
