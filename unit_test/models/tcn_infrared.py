# *----------------------------------------------------------------------------*
# * Copyright (C) 2023 Politecnico di Torino, Italy                            *
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

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCN_IR(nn.Module):
    def __init__(self):
        super(TCN_IR, self).__init__()
        self.input_shape = (1, 8, 8)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.tcn = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.fc1 = nn.Linear(32, 3)

    def forward(self, x1, x2, x3):
        feature_list = []

        x = F.relu(self.bn1(self.conv1(x1)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        out_i = torch.flatten(x, 1)
        out_to_cat = out_i.unsqueeze(-1)
        feature_list.append(out_to_cat)

        x = F.relu(self.bn1(self.conv1(x2)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        out_i = torch.flatten(x, 1)
        out_to_cat = out_i.unsqueeze(-1)
        feature_list.append(out_to_cat)

        x = F.relu(self.bn1(self.conv1(x3)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        out_i = torch.flatten(x, 1)
        out_to_cat = out_i.unsqueeze(-1)
        feature_list.append(out_to_cat)

        feature_out = torch.cat(feature_list, dim=-1)

        x = self.bn3(self.tcn(feature_out))
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        return x
