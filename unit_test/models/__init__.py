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
from .tc_resnet_14 import TCResNet14
from .simple_nn import SimpleNN
from .simple_nn_pit import SimplePitNN
from .toy_models import ToyModel1, ToyModel2, ToyModel3, ToyModel4
from .toy_models import ToyModel5, ToyModel6, ToyModel7, ToyModel8

__all__ = ['TCResNet14', 'SimpleNN', 'SimplePitNN', 'ToyModel1',
           'ToyModel2', 'ToyModel3', 'ToyModel4', 'ToyModel5', 'ToyModel6',
           'ToyModel7', 'ToyModel8']
