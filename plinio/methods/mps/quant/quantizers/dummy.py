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

from typing import Dict, Any, Optional, Iterator, Tuple
import torch
import torch.fx as fx
import torch.nn as nn
from .quantizer import Quantizer


class DummyQuantizer(Quantizer):
    """A nn.Module implementing a dummy quantizer, used as place-holder for the
    input quantizer when creating MPSModule layers, then replaced when processing
    the DNN graph. Implements an identity operation
    """
    def __init__(self, precision: int):
        # the precision parameter is required by the super-class, but will be
        # ignored in this sub-class.
        super(DummyQuantizer, self).__init__(precision, True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The (dummy) forward function

        :param input: the input float weights tensor
        :type input: torch.Tensor
        :return: the output fake-quantized weights tensor
        :rtype: torch.Tensor
        """
        return input

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule, backend: Optional[str]):
        """Replaces a fx.Node corresponding to a Quantizer, with a "backend-aware" layer
        within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        :param backend: an optional string specifying the target backend
        :type backend: Optional[str]
        """
        raise NotImplementedError("Cannot export a Dummy Quantizer")

    @property
    def scale(self) -> torch.Tensor:
        """Return a (dummy) scale factor, always equal to 1

        :return: the scale factor
        :rtype: torch.Tensor
        """
        return torch.tensor(1.0)

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer quantization hyperparameters

        :return: a dictionary containing the optimized layer quantization hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'scale_factor': self.scale.detach().item(),
        }

    def named_quant_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the quantization parameters of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API,
        but Quantizer never have sub-layers TODO: check if true
        :type recurse: bool
        :return: an iterator over the quantization parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        # this is just to circumvent a type warning on the return type
        empty_dict = {}
        for k, v in empty_dict:
            yield k, v

    def __repr__(self):
        msg = (
            f'{self.__class__.__name__}'
            f'(precision=None (dummy quantizer), '
            f'scale_factor={self.scale})'
        )
        return msg
