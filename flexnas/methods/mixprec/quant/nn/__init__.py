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

from .quant_module import QuantModule
from .quant_identity import Quant_Identity
from .quant_relu import Quant_ReLU
from .quant_linear import Quant_Linear
from .quant_conv2d import Quant_Conv2d
from .quant_list import Quant_List

__all__ = [
    'QuantModule', 'Quant_Identity', 'Quant_ReLU',
    'Quant_Linear', 'Quant_Conv2d', 'Quant_List',
]
