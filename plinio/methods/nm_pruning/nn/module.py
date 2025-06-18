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
# * Author:  Francesco Daghero <francesco.daghero@polito.it>                             *
# *----------------------------------------------------------------------------*

from abc import abstractmethod
from typing import Dict, Any
import torch.fx as fx


class NMPruningModule:
    """An abstract class representing the interface that all MPS layers should implement
    """
    @abstractmethod
    def __init__(self):
        pass


    @staticmethod
    @abstractmethod
    def autoimport(node: fx.Node,
                   mod: fx.GraphModule,
                   n: int,
                   m: int,
                   pruning_decay: float):
        """Create a new fx.Node relative to a NMPruning layer, starting from the fx.Node
        of a nn.Module layer, and replace it into the parent fx.GraphModule

        :param node: a fx.Node corresponding to the module to be converted
        :type node: fx.Node
        :param mod: the parent fx.GraphModule
        :type mod: fx.GraphModule
        :param n: number of non-zero parameters
        :type n: int
        :param m: group of weights considered for pruning
        :type m: int
        :param pruning_decay: the decay factor for the pruning mask
        :type pruning_decay: float
        :raises TypeError: if the input fx.Node is not of the correct type
        """
        raise NotImplementedError("Trying to import layer using the base abstract class")

    @staticmethod
    @abstractmethod
    def export(node: fx.Node, mod: fx.GraphModule):
        """Replaces a fx.Node corresponding to a MPSModule, with a standard nn.Module layer
        within a fx.GraphModule

        :param node: the node to be rewritten
        :type node: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        raise NotImplementedError("Trying to export layer using the base abstract class")

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer bitwidth

        :return: a dictionary containing the optimized layer bitwidth values
        :rtype: Dict[str, Any]
        """
        return {}