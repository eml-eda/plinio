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
from .simple_nn import SimpleNN, SimpleNN2D, SimpleNN2D_NoBN
from .simple_nn_pit import SimplePitNN
from .simple_nn_mps import (
    SimpleMPSNN,
    SimpleExportedNN1D,
    SimpleExportedNN2D,
    SimpleExportedNN2D_ch,
)
from .dscnn import DSCNN
from .toy_models import (
    ToySequentialConv1d,
    ToySequentialConv2d,
    ToySequentialFullyConv2d,
    ToySequentialSeparated,
    ToyAdd,
    ToySequentialConv2d_v2,
)
from .toy_models import ToyTimeCat, ToyChannelsCat
from .toy_models import (
    ToyFlatten,
    ToyMultiPath1,
    ToyMultiPath2,
    ToyRegression,
    ToyMultiPath1_2D,
    ToyMultiPath2_2D,
    ToyAdd_2D,
    ToyRegression_2D,
    ToyInputConnectedDW,
    ToyBatchNorm,
    ToyIllegalBN,
    ToySequentialFullyConv2dDil,
    ToyResNet,
    ToyResNet_1D,
)
from .phd_course_model import TutorialModel, TutorialModel_NoDW
from .tcn_infrared import TCN_IR

__all__ = [
    "TCResNet14",
    "SimpleNN",
    "SimpleNN2D",
    "SimplePitNN",
    "DSCNN",
    "ToySequentialConv1d",
    "ToySequentialConv2d",
    "ToySequentialFullyConv2d",
    "ToyMultiPath1_2D",
    "ToyMultiPath2_2D",
    "ToySequentialSeparated",
    "ToyAdd",
    "ToyTimeCat",
    "ToyChannelsCat",
    "ToyFlatten",
    "ToyMultiPath1",
    "ToyMultiPath2",
    "ToyRegression",
    "ToyInputConnectedDW",
    "SimpleMPSNN",
    "SimpleExportedNN1D",
    "SimpleExportedNN2D",
    "ToyAdd_2D",
    "ToyRegression_2D",
    "SimpleNN2D_NoBN",
    "SimpleExportedNN2D_ch",
    "ToyBatchNorm",
    "ToyIllegalBN",
    "ToySequentialConv2d_v2",
    "TutorialModel",
    "TutorialModel_NoDW",
    "TCN_IR",
    "ToySequentialFullyConv2dDil",
    "ToyResNet",
    "ToyResNet_1D",
]
