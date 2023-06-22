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

from abc import abstractmethod
from typing import Any, Dict, Iterable,  Iterator, List, Tuple, Type, Union
from plinio.cost import CostSpec
import torch
import torch.nn as nn


class DNAS(nn.Module):
    """Abstract class to wrap a nn.Module with a DNAS functionality

    :param model: the inner nn.Module instance optimized by the NAS
    :type model: nn.Module
    :param cost: the cost model(s) used by the NAS
    :type cost: Union[CostSpec, List[CostSpec]]
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS,
    defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that shuould be ignored by the NAS,
    defaults to ()
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :raises ValueError: when called with an unsupported regularizer
    """
    @abstractmethod
    def __init__(
            self,
            cost: Union[CostSpec, Dict[str, CostSpec]],
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = ()):
        super(DNAS, self).__init__()
        self._cost_specification = cost
        self.exclude_names = exclude_names
        self.exclude_types = tuple(exclude_types)

    @abstractmethod
    def forward(self, *args: Any) -> torch.Tensor:
        """Forward function for the DNAS model.

        :raises NotImplementedError: on the base DNAS class
        :return: the output tensor
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @property
    def cost_specification(self) -> Union[CostSpec, Dict[str, CostSpec]]:
        return self._cost_specification

    @cost_specification.setter
    def cost_specification(self, _: Union[CostSpec, List[CostSpec]]):
        raise NotImplementedError(
                "Cannot set the cost specification(s) on the base DNAS class")

    @property
    @abstractmethod
    def cost(self) -> torch.Tensor:
        """Returns the value of the default cost metric

        :raises NotImplementedError: on the base DNAS class
        :return: a scalar tensor with the cost value
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    # use method instead of property in case of multiple costs
    def get_cost(self, name: str) -> torch.Tensor:
        """Returns the value of the model cost metric named "name"

        :raises NotImplementedError: on the base DNAS class
        :return: a scalar tensor with the cost value
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of the NAS, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API, but PITLayers never have sub-layers
        :type recurse: bool
        :return: an iterator over the architectural parameters of the NAS
        :rtype: Iterator[nn.Parameter]
        """
        raise NotImplementedError("Calling arch_parameters on base abstract DNAS class")

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        """Returns an iterator over the architectural parameters of the NAS

        :param recurse: kept for uniformity with pytorch API, but PITLayers never have sub-layers
        :type recurse: bool
        :return: an iterator over the architectural parameters of the NAS
        :rtype: Iterator[nn.Parameter]
        """
        for name, param in self.named_nas_parameters(recurse=recurse):
            yield param

    @abstractmethod
    def named_net_parameters(
            self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the inner network parameters, EXCEPT the NAS architectural
        parameters, yielding both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API, not actually used
        :type recurse: bool
        :return: an iterator over the inner network parameters
        :rtype: Iterator[nn.Parameter]
        """
        raise NotImplementedError("Calling arch_parameters on base abstract DNAS class")

    def net_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        """Returns an iterator over the inner network parameters, EXCEPT the NAS architectural
        parameters

        :param recurse: kept for uniformity with pytorch API, not actually used
        :type recurse: bool
        :return: an iterator over the architectural parameters (masks) of the NAS
        :rtype: Iterator[nn.Parameter]
        """
        for name, param in self.named_net_parameters(recurse=recurse):
            yield param
