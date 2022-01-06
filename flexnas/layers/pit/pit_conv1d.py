#*----------------------------------------------------------------------------*
#* Copyright (C) 2021 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Daniele Jahier Pagliari <daniele.jahier@polito.it>                *
#*----------------------------------------------------------------------------*

import torch
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import copy

### NOTE: made dummy for now, to be replaced with Matteo's implementation later

class PITConv1d(Conv1d):

    def __init__(self, custom_param, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(PITConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
        self.custom_param = custom_param
   
    def forward(self, x):
        # ...PIT CODE...
        return super(PITConv1d, self).forward(x)
    