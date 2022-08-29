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

from typing import Any, Tuple, Type, Iterable, Dict, cast
import torch
import torch.nn as nn
from flexnas.methods.dnas_base import DNAS
from .pit_converter import convert
from .pit_layer import PITLayer


class PIT(DNAS):
    """A class that wraps a nn.Module with the functionality of the PIT NAS tool

    :param model: the inner nn.Module instance optimized by the NAS
    :type model: nn.Module
    :param input_shape: the shape of an input tensor, without batch size, required for symbolic
    tracing
    :type input_shape: Tuple[int, ...]
    :param regularizer: a string defining the type of cost regularizer, defaults to 'size'
    :type regularizer: Optional[str], optional
    :param autoconvert_layers: should the constructor try to autoconvert NAS-able layers,
    defaults to True
    :type autoconvert_layers: bool, optional
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    when auto-converting defaults to ()
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :param train_channels: flag to control whether output channels are optimized by PIT or not,
    defaults to True
    :type train_channels: bool, optional
    :param train_rf: flag to control whether receptive field is optimized by PIT or not, defaults
    to True
    :type train_rf: bool, optional
    :param train_dilation: flag to control whether dilation is optimized by PIT or not, defaults
    to True
    :type train_dilation: bool, optional
    """
    def __init__(
            self,
            model: nn.Module,
            input_shape: Tuple[int, ...],
            regularizer: str = 'size',
            autoconvert_layers: bool = True,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            train_channels: bool = True,
            train_rf: bool = True,
            train_dilation: bool = True):
        super(PIT, self).__init__(regularizer, exclude_names, exclude_types)
        self._input_shape = input_shape
        self._inner_model, self._target_layers = convert(
            model,
            input_shape,
            'autoimport' if autoconvert_layers else 'import',
            exclude_names,
            exclude_types
        )
        # after conversion to make sure they are applied to all layers
        self.train_channels = train_channels
        self.train_rf = train_rf
        self.train_dilation = train_dilation
        self._regularizer = regularizer

    def forward(self, *args: Any) -> torch.Tensor:
        """Forward function for the DNAS model. Simply invokes the inner model's forward

        :return: the output tensor
        :rtype: torch.Tensor
        """
        return self._inner_model.forward(*args)

    def supported_regularizers(self) -> Tuple[str, ...]:
        """Returns a list of names of supported regularizers

        :return: a tuple of strings with the name of supported regularizers
        :rtype: Tuple[str, ...]
        """
        return ('size', 'macs')

    def get_size(self) -> torch.Tensor:
        """Computes the total number of parameters of all NAS-able layers

        :return: the total number of parameters
        :rtype: torch.Tensor
        """
        size = torch.tensor(0, dtype=torch.float32)
        # size = torch.tensor(0)
        for layer in self._target_layers:
            # size += layer.get_size()
            size = size + layer.get_size()
        return size

    def get_macs(self) -> torch.Tensor:
        """Computes the total number of MACs in all NAS-able layers

        :return: the total number of MACs
        :rtype: torch.Tensor
        """
        macs = torch.tensor(0, dtype=torch.float32)
        # macs = torch.tensor(0)
        for layer in self._target_layers:
            # macs += layer.get_macs()
            macs = macs + layer.get_macs()
        return macs

    @property
    def regularizer(self) -> str:
        """Returns the regularizer type

        :raises ValueError: for unsupported conversion types
        :return: the string identifying the regularizer type
        :rtype: str
        """
        return self._regularizer

    @regularizer.setter
    def regularizer(self, value: str):
        if value == 'size':
            self.get_regularization_loss = self.get_size
        elif value == 'macs':
            self.get_regularization_loss = self.get_macs
        else:
            raise ValueError(f"Invalid regularizer {value}")
        self._regularizer = value

    @property
    def train_channels(self) -> bool:
        """Returns True if PIT is training the output channels masks

        :return: True if PIT is training the output channels masks
        :rtype: bool
        """
        return self._train_channels

    @train_channels.setter
    def train_channels(self, value: bool):
        """Set to True to let PIT train the output channels masks

        :param value: set to True to let PIT train the output channels masks
        :type value: bool
        """
        for layer in self._target_layers:
            if hasattr(layer, 'train_channels'):
                layer.train_channels = value
            else:
                print(f"Warning: layer {layer} does not support channels optimization")
        self._train_channels = value

    @property
    def train_rf(self) -> bool:
        """Returns True if PIT is training the filters receptive fields masks

        :return: True if PIT is training the filters receptive fields masks
        :rtype: bool
        """
        return self._train_rf

    @train_rf.setter
    def train_rf(self, value: bool):
        """Set to True to let PIT train the filters receptive fields masks

        :param value: set to True to let PIT train the filters receptive fields masks
        :type value: bool
        """
        for layer in self._target_layers:
            if hasattr(layer, 'train_rf'):
                layer.train_rf = value
            else:
                print(f"Warning: layer {layer} does not support receptive field optimization")
        self._train_rf = value

    @property
    def train_dilation(self):
        """Returns True if PIT is training the filters dilation masks

        :return: True if PIT is training the filters dilation masks
        :rtype: bool
        """
        return self._train_dilation

    @train_dilation.setter
    def train_dilation(self, value: bool):
        """Set to True to let PIT train the filters dilation masks

        :param value: set to True to let PIT train the filters dilation masks
        :type value: bool
        """
        for layer in self._target_layers:
            if hasattr(layer, 'train_dilation'):
                layer.train_dilation = value
            else:
                print(f"Warning: layer {layer} does not support dilation optimization")
        self._train_dilation = value

    def arch_export(self):
        """Export the architecture found by the NAS as a `nn.Module`

        The returned model will have the trained weights found during the search filled in, but
        should be fine-tuned for optimal results.

        :return: the architecture found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        mod, _ = convert(self._inner_model, self._input_shape, 'export')

        return mod

    def arch_summary(self) -> Dict[str, Dict[str, Any]]:
        """Generates a dictionary representation of the architecture found by the NAS.
        Only optimized layers are reported

        :return: a dictionary representation of the architecture found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        arch = {}
        for name, layer in self._inner_model.named_modules():
            if layer in self._target_layers:
                layer = cast(PITLayer, layer)
                arch[name] = layer.summary()
                arch[name]['type'] = layer.__class__.__name__
        return arch

    def __str__(self):
        """Prints the architecture found by the NAS to screen

        :return: a str representation of the current architecture
        :rtype: str
        """
        arch = self.arch_summary()
        s = ""
        for name, layer in arch.items():
            s += "{} ({}): Cin = {}, Cout = {}, K = {}, D = {}".format(
                layer['type'],
                name,
                layer['in_channels'],
                layer['out_channels'],
                layer['kernel_size'],
                layer['dilation']
            ) + "\n"
        return s
