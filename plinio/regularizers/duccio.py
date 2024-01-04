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

from typing import Tuple
import torch
from plinio.methods import DNAS

class DUCCIO():
    def __init__(self,
                 targets: Tuple[float,...],
                 cost_names: Tuple[str,...] = ('params', 'ops'),
                 final_strengths: Tuple[float,...] = (1e-3, 1e-3)):
        """An implementation of the DUCCIO regularization method proposed in:
            https://arxiv.org/abs/2206.00302
        The regularization loss term computed by DUCCIO is:
        sum(max(0, (cost_i - target_i)) * strength) for each target cost metric i.

        :param targets: the target values for each cost metric. The length of this tuple must match
        the length of `cost_names` and `final_strengths`
        :type targets: Tuple[float,...]
        :param cost_names: the names of the cost metrics to use, defaults to ('params', 'ops').
        The length of this tuple must match the length of `targets` and `final_strengths`
        :type cost_names: Tuple[str,...], optional
        :param final_strengths: the final regularization strength values for each cost metric. Defaults to (1e-3, 1e-3).
        The length of this tuple must match the length of `targets` and `cost_names`
        """
        if not (len(targets) == len(cost_names) == len(final_strengths)):
            raise ValueError("The lengths of targets, cost_names and final_strengths must match")
        self.targets = targets
        self.cost_names = cost_names
        self.final_strengths = final_strengths

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
        cost = torch.tensor(0.0)
        for cost_name, target, strength in zip(self.cost_names, self.targets, self.final_strengths):
            eff_strength = min(strength/100 + epoch * (strength*99/100) / (n_epochs / 2), strength)
            cost += (eff_strength * torch.max(torch.tensor(0.0), model.get_cost(cost_name) - target))
        return cost


