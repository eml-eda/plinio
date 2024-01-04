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

import torch
from plinio.methods import DNAS

class BaseRegularizer():
    def __init__(self, cost_name: str = 'params', strength: float = 1e-3):
        """A simple regularizer that computes a loss term that depends on a single cost metric, 
        multiplied by a fixed strength factor.

        :param cost_name: the name of the cost metric to use, defaults to 'params'
        :type cost_name: str, optional
        :param strength: the strength factor, defaults to 1e-3
        :type strength: float, optional
        """
        self.cost_name = cost_name
        self.strength = strength

    def __call__(self, model: DNAS) -> torch.Tensor:
        """Computes the regularization loss term.

        :param model: the model to regularize
        :type model: DNAS
        :return: the regularization loss value
        :rtype: torch.Tensor
        """
        return model.get_cost(self.cost_name) * self.strength

