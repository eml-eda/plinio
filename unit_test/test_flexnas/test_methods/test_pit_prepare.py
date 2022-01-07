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
from typing import Iterable, Tuple, Type
import unittest
import torch.nn as nn
from flexnas.methods import PITModel
from flexnas.methods.dnas_base.dnas_model import DNASModel
from models import TCResNet14
from .utils import MySimpleNN


class TestPITPrepare(unittest.TestCase):

    def test_simple_model(self):
        nn_ut = MySimpleNN()
        self._execute_prepare(nn_ut)

    def test_tc_resnet_14(self):
        config = {
            "input_size": 40,
            "output_size": 12,
            "num_channels": [24, 36, 36, 48, 48, 72, 72],
            "kernel_size": 9,
            "dropout": 0.5,
            "grad_clip": -1,
            "use_bias": True,
            "avg_pool": True,
        }
        nn_ut = TCResNet14(config)
        self._execute_prepare(nn_ut)

    def _execute_prepare(
            self,
            nn_ut: nn.Module,
            exclude_names: Iterable[str] = (),
            exclude_types: Tuple[Type[nn.Module], ...] = ()):
        new_nn = PITModel(nn_ut)
        inner = new_nn._inner_model
        self._compare_prepared(nn_ut, inner, nn_ut, new_nn, exclude_names, exclude_types)

    def _compare_prepared(self,
                          old_mod: nn.Module, new_mod: nn.Module,
                          old_top: nn.Module, new_top: DNASModel,
                          exclude_names: Iterable[str], exclude_types: Tuple[Type[nn.Module]]):
        for name, child in old_mod.named_children():
            new_child = new_mod._modules[name]
            self._compare_prepared(child, new_child, old_top, new_top, exclude_names, exclude_types)
            if isinstance(child, new_top.optimizable_layers()):
                if (name not in exclude_names) and (not isinstance(child, exclude_types)):
                    repl = new_top.replacement_layer(name, child, old_top)
                    print(type(new_child))
                    print(type(repl))
                    assert isinstance(new_child, type(repl))


if __name__ == '__main__':
    unittest.main()
