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
import torch.nn as nn
import torch.nn.functional as F


class MySimpleNN(nn.Module):
    def __init__(self, input_shape=(3, 40), num_classes=3):
        super(MySimpleNN, self).__init__()
        self.input_shape = input_shape
        self.conv0 = nn.Conv1d(3, 32, 3, padding='same')
        self.bn0 = nn.BatchNorm1d(32)
        self.pool0 = nn.AvgPool1d(2)
        self.conv1 = nn.Conv1d(32, 57, 5, padding='same')
        self.bn1 = nn.BatchNorm1d(57)
        self.pool1 = nn.AvgPool1d(2)
        self.dpout = nn.Dropout(0.5)
        self.fc = nn.Linear(input_shape[-1] // 2 // 2, num_classes)
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = F.relu6(self.pool0(self.bn0(self.conv0(x))))
        x = F.relu6(self.pool1(self.bn1(self.conv1(x))))
        x = self.dpout(x)
        res = self.fc(x)
        return res
