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

from typing import Dict, Optional
import torch.fx as fx
from ..quantizers import Quantizer
from ..backends import Backend, backend_factory
from .identity import QuantIdentity


class QuantAdd(QuantIdentity):
    """A nn.Module implementing a quantized Identity layer

    :param quantizer: output activations quantizer
    :type quantizer: Type[Quantizer]
    """

    def __init__(self, quantizer: Quantizer):
        super(QuantAdd, self).__init__(quantizer)

    @staticmethod
    def autoimport() -> Optional[Quantizer]:
        raise NotImplementedError

    @staticmethod
    def export(
        n: fx.Node,
        mod: fx.GraphModule,
        backend: Backend,
        backend_kwargs: Dict = {},
    ):
        """Replaces a fx.Node corresponding to a Quant_Identity layer,
        with a backend-specific quantize Identity layer within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        :param backend: the specific backend to be used
        :type backend: Backend
        """
        submodule = mod.get_submodule(str(n.target))
        if type(submodule) != QuantAdd:
            raise TypeError(f"Trying to export a layer of type {type(submodule)}")
        integer_add = backend_factory(submodule, backend)
        new_submodule = integer_add(submodule.out_quantizer, **backend_kwargs)
        mod.add_submodule(str(n.target), new_submodule)
