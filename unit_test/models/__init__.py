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
from .toy_models import ToySequentialConv1d, ToyAdd, ToyTimeCat, ToyChannelsCat
from .toy_models import ToyFlatten, ToyMultiPath1, ToyMultiPath2, ToyRegression

__all__ = ['TCResNet14', 'SimpleNN', 'SimplePitNN', 'ToySequentialConv1d', 'ToyAdd',
           'ToyTimeCat', 'ToyChannelsCat', 'ToyFlatten', 'ToyMultiPath1', 'ToyMultiPath2',
           'ToyRegression']
