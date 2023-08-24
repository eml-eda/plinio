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
from typing import Any, Dict, Iterator, Optional, Tuple, Union, cast
from plinio.cost import CostSpec, CostFn
import torch
import torch.nn as nn
from warnings import warn


class DNAS(nn.Module):
    """Abstract class to wrap a nn.Module with a DNAS functionality

    :param model: the inner nn.Module instance optimized by the NAS
    :type model: nn.Module
    :param cost: the cost model(s) used by the NAS
    :type cost: Union[CostSpec, List[CostSpec]]
    :param input_example: an input with the same shape and type of the seed's input, used
    for symbolic tracing (default: None)
    :type input_example: Optional[Any]
    :param input_shape: the shape of an input tensor, without batch size, used as an
    alternative to input_example to generate a random input for symbolic tracing (default: None)
    :type input_shape: Optional[Tuple[int, ...]]
    """
    @abstractmethod
    def __init__(
            self,
            model: nn.Module,
            cost: Union[CostSpec, Dict[str, CostSpec]],
            input_example: Optional[Any] = None,
            input_shape: Optional[Tuple[int, ...]] = None):
        super(DNAS, self).__init__()
        self._device = next(model.parameters()).device
        self._cost_specification = cost
        self._input_example = self._resolve_input_example(input_example, input_shape)
        self._cost_fn_map = {}

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
    def cost_specification(self, cs: Union[CostSpec, Dict[str, CostSpec]]):
        self._cost_specification = cs

    @property
    def cost(self) -> torch.Tensor:
        """Returns the value of the default cost metric

        :return: a scalar tensor with the cost value
        :rtype: torch.Tensor
        """
        return self.get_cost(None)

    @abstractmethod
    def export(self) -> nn.Module:
        """Export the architecture found by the NAS as a 'nn.Module'

        :raises NotImplementedError: on the base DNAS class
        :return: the architecture found by the NAS
        :rtype: nn.Module
        """
        raise NotImplementedError("Trying to export optimized model on the base DNAS class")

    @abstractmethod
    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Generates a dictionary representation of the architecture found by the NAS.
        Only optimized layers are reported

        :raises NotImplementedError: on the base DNAS class
        :return: a dictionary representation of the architecture found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        raise NotImplementedError("Trying to get model summary on the base DNAS class")

    def get_cost(self, name: Optional[str] = None) -> torch.Tensor:
        """Returns the value of the model cost metric named "name".
        Only allowed alternative in case of multiple cost metrics.

        :return: a scalar tensor with the cost value
        :rtype: torch.Tensor
        """
        if name is None:
            assert isinstance(self._cost_specification, CostSpec), \
                "Multiple model cost metrics provided, you must use .get_cost('cost_name')"
            cost_spec = self._cost_specification
            cost_fn_map = self._cost_fn_map
        else:
            assert isinstance(self._cost_specification, dict), \
                "The provided cost specification is not a dictionary."
            cost_spec = self._cost_specification[name]
            cost_fn_map = self._cost_fn_map[name]
        cost_spec = cast(CostSpec, cost_spec)
        cost_fn_map = cast(Dict[str, CostFn], cost_fn_map)
        return self._get_single_cost(cost_spec, cost_fn_map)

    @abstractmethod
    def _get_single_cost(self, cost_spec: CostSpec,
                         cost_fn_map: Dict[str, CostFn]) -> torch.Tensor:
        """NAS-specific method to compute a single cost value

        :raises NotImplementedError: on the base DNAS class
        :return: a scalar tensor with the cost value
        :rtype: torch.Tensor
        """
        raise NotImplementedError("Trying to compute cost on base DNAS class")

    def _create_cost_fn_map(self) -> Union[Dict[str, CostFn], Dict[str, Dict[str, CostFn]]]:
        """Private method to define a map from layers to cost value(s)"""
        if isinstance(self._cost_specification, dict):
            cost_fn_maps = {}
            for n, c in self._cost_specification.items():
                cost_fn_maps[n] = self._single_cost_fn_map(c)
        else:
            cost_fn_maps = self._single_cost_fn_map(self._cost_specification)
        return cost_fn_maps

    @abstractmethod
    def _single_cost_fn_map(self, c: CostSpec) -> Dict[str, CostFn]:
        """NAS-specific creator of {layertype, cost_fn} maps based on a CostSpec.

        :raises NotImplementedError: on the base DNAS class
        :return: a scalar tensor with the cost value
        :rtype: torch.Tensor
        """
        raise NotImplementedError("Trying to create cost function map on base DNAS class")

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
        for _, param in self.named_nas_parameters(recurse=recurse):
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
        for _, param in self.named_net_parameters(recurse=recurse):
            yield param

    def _resolve_input_example(self, example, shape):
        """Selects between using input_example and input_shape, with sanity checks"""
        if example is None and shape is None:
            msg = 'One of `input_example` and `input_shape` must be different from None'
            raise ValueError(msg)
        if example is not None and shape is not None:
            msg = ('Warning: you specified both `input_example` and `input_shape`.'
                   'The first will be considered for shape propagation')
            warn(msg)
        if example is not None:
            return example
        if shape is not None:
            try:
                # create a "fake" minibatch of 1 input for shape prop
                example = torch.stack([torch.rand(shape)] * 1, 0).to(self._device)
                return example
            except TypeError:
                msg = ('If the provided `input_shape` is not a simple tuple '
                       'the user should pass instead an `input_example`.')
                raise TypeError(msg)
