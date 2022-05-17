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
from typing import Any, Optional, Iterable, Tuple, Type
import copy
import torch
import torch.nn as nn


class DNASModel(nn.Module):

    @abstractmethod
    def __init__(
            self,
            model: nn.Module,
            regularizer: Optional[str] = None,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = ()):
        """Abstract DNAS model constructor

        :param model: the inner nn.Module instance optimized by the NAS
        :type model: nn.Module
        :param regularizer: the name of the model cost regularizer used by the NAS 
        :type regularizer: Optional[str], optional
        :param exclude_names: the names of `model` submodules that should be ignored by the NAS, defaults to ()
        :type exclude_names: Iterable[str], optional
        :param exclude_types: the types of `model` submodules that shuould be ignored by the NAS, defaults to ()
        :type exclude_types: Iterable[Type[nn.Module]], optional
        :raises ValueError: when called with an unsupported regularizer
        """
        super(DNASModel, self).__init__()
        if regularizer not in self.supported_regularizers():
            raise ValueError("Unsupported regularizer {}".format(regularizer))
        self.regularizer = regularizer
        self.exclude_names = exclude_names
        self.exclude_types = tuple(exclude_types)
        self._inner_model = copy.deepcopy(model)

    def forward(self, *args: Any) -> torch.Tensor:
        """Forward function for the DNAS model. Simply invokes the inner model's forward

        :return: the output tensor
        :rtype: torch.Tensor
        """
        return self._inner_model.forward(*args)

    @abstractmethod
    def supported_regularizers(self) -> Tuple[str, ...]:
        """Returns a tuple of strings with the names of the supported cost regularizers

        :raises NotImplementedError: on the base DNAS class
        :return: a tuple of strings with the names of the supported cost regularizers
        :rtype: Tuple[str, ...]
        """
        raise NotImplementedError

    @abstractmethod
    def get_regularization_loss(self) -> torch.Tensor:
        """Returns the value of the model cost regularization loss

        :raises NotImplementedError: on the base DNAS class
        :return: a scalar tensor with the loss value
        :rtype: torch.Tensor
        """
        raise NotImplementedError
