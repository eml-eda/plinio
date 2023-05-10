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
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *----------------------------------------------------------------------------*

from enum import Enum, auto
import torch.nn as nn

from . import dory


class Backend(Enum):
    ONNX = auto()
    DORY = auto()
    DIANA = auto()
    # Add new backends here


def backend_solver(layer: nn.Module, backend: Backend) -> nn.Module:
    """Depending on the specific `layer` and specified `backend` returns
    the appropriate backend-specific layer implementation.

    :param layer: the layer to be converted
    :type layer: nn.Module
    :param backend: the backend to be used
    :type backend: Backend
    :param backend: the specific backend to be used
    :type backend: Backend
    :return: the backend specific layer implementation
    :rtype: nn.Module
    """
    raise NotImplementedError
