#*----------------------------------------------------------------------------*
#* Copyright (C) 2022 Politecnico di Torino, Italy                            *
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
import os
import sys
import pprint
sys.path.append(os.getcwd())

import argparse
import json
import torch
from torchinfo import summary
from dataloaders import GSCDataLoader
from models import TCResNet14

def run(config_file, in_dir, out_dir):
    print("[GSC Trainer] Config File:", config_file)
    print("[GSC Trainer] Input Directory:", in_dir)
    print("[GSC Trainer] Output Directory:", out_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[GSC Trainer] Training on:", device)

    with open(config_file, 'r') as f:
        config = json.load(f)
    print("[GSC Trainer] Configuration:")
    pprint.pprint(config)

    train = GSCDataLoader(in_dir, config['data']['batch_size'], set_= 'train')
    val = GSCDataLoader(in_dir, config['data']['batch_size'], set_= 'valid')
    test = GSCDataLoader(in_dir, config['data']['batch_size'], set_= 'test')

    input_size = list(next(iter(train))['data'].shape)
    input_size[0] = 1 # remove batch size
    print("[GSC Trainer] Input size is:", input_size)

    net = TCResNet14(config['model'])
    net.to(device)
    summary(net, input_size, depth=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', default=None, type=str, help='config file path')
    parser.add_argument('-i', '--in_dir', default=None, type=str, help='input data directory path')
    parser.add_argument('-o', '--out_dir', default=None, type=str, help='output data directory path')
    args = vars(parser.parse_args())
    run(**args)