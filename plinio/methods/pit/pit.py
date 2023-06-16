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
from typing import Any, Callable, List, Union, Tuple, Type, Iterable, Dict, cast, Iterator, Optional
from warnings import warn

import torch
import torch.nn as nn
import torch.fx as fx

from plinio.methods.dnas_base import DNAS
from plinio.cost import CostSpec, PatternSpec, params
from plinio.graph.inspection import shapes_dict
from .graph import convert, pit_layer_map
from .nn.module import PITModule


class PIT(DNAS):
    """A class that wraps a nn.Module with the functionality of the PIT NAS tool

    :param model: the inner nn.Module instance optimized by the NAS
    :type model: nn.Module
    :param input_example: an input example, it has same type of the expected
    `model`'s input, used for symbolic tracing (default: None)
    :type input_example: Optional[Any]
    :param input_shape: the shape of an input tensor, without batch size, used as an
    alternative to input_example to generate a random input for symbolic tracing (default: None)
    :type input_shape: Optional[Tuple[int, ...]]
    :param cost: the cost models(s) used by the NAS, defaults to the number of params.
    :type cost: Union[CostSpec, List[CostSpec]]
    :param autoconvert_layers: should the constructor try to autoconvert NAS-able layers,
    defaults to True
    :type autoconvert_layers: bool, optional
    :param discrete_cost: True is the cost model should be computed on a discrete sample,
    rather than a continuous relaxation, defaults to False
    :type discrete_cost: bool, optional
    :param full_cost: True is the cost model should be applied to the entire network, rather
    than just to the NAS-able layers, defaults to False
    :type discrete_sampling: bool, optional
    :param exclude_names: the names of `model` submodules that should be ignored by the NAS
    when auto-converting layers, defaults to ()
    :type exclude_names: Iterable[str], optional
    :param exclude_types: the types of `model` submodules that should be ignored by the NAS
    when auto-converting defaults to ()
    :type exclude_types: Iterable[Type[nn.Module]], optional
    :param train_features: flag to control whether output features are optimized by PIT or not,
    defaults to True
    :type train_features: bool, optional
    :param train_rf: flag to control whether receptive field is optimized by PIT or not, defaults
    to True
    :type train_rf: bool, optional
    :param train_dilation: flag to control whether dilation is optimized by PIT or not, defaults
    to True
    :type train_dilation: bool, optional

    :raises UserWarning: when both `input_example` and `input_shape` are NOT None,
    a warning is raised and `input_example` will be used.
    :raises ValueError: when both `input_example` and `input_shape` are None
    """
    def __init__(
            self,
            model: nn.Module,
            input_example: Optional[Any] = None,
            input_shape: Optional[Tuple[int, ...]] = None,
            cost: Union[CostSpec, List[CostSpec]] = params,
            autoconvert_layers: bool = True,
            discrete_cost: bool = False,
            full_cost: bool = False,
            exclude_names: Iterable[str] = (),
            exclude_types: Iterable[Type[nn.Module]] = (),
            train_features: bool = True,
            train_rf: bool = True,
            train_dilation: bool = True):
        super(PIT, self).__init__(cost, exclude_names, exclude_types)
        self._device = next(model.parameters()).device
        self._input_example = self._resolve_input_example(input_example, input_shape)
        # self._input_shape = input_shape
        self.seed, self._leaf_modules = convert(
            model,
            self._input_example,
            'autoimport' if autoconvert_layers else 'import',
            exclude_names,
            exclude_types
        )
        # Get unique leaf modules, needed to correctly compute cost metrics that should not
        # be counted twice when a layer is used twice in a single fwd pass
        # (e.g. true for params, false for ops)
        self._unique_leaf_modules = self._uniquify_leaf_modules()
        self.cost_fn_maps = self._create_cost_fn_maps()
        # after conversion to make sure they are applied to all layers
        self.train_features = train_features
        self.train_rf = train_rf
        self.train_dilation = train_dilation
        self.discrete_cost = discrete_cost
        self.full_cost = full_cost

    def forward(self, *args: Any) -> torch.Tensor:
        """Forward function for the DNAS model. Simply invokes the inner model's forward

        :return: the output tensor
        :rtype: torch.Tensor
        """
        return self.seed.forward(*args)

    @property
    def cost(self) -> torch.Tensor:
        if isinstance(self._cost_specification, list):
            return self._evaluate_ith_cost_metric(0, self._cost_specification[0])
        else:
            return self._evaluate_ith_cost_metric(0, self._cost_specification)

    @property
    def cost_specification(self) -> Union[CostSpec, List[CostSpec]]:
        return self._cost_specification

    @cost_specification.setter
    def cost_specification(self, cs: Union[CostSpec, List[CostSpec]]):
        self._cost_specification = cs
        self.cost_fn_maps = self._create_cost_fn_maps()

    # use method instead of property in case of multiple costs
    def get_cost(self, i: int = 0) -> torch.Tensor:
        assert \
            i == 0 or \
            (isinstance(self._cost_specification, list) and len(self._cost_specification) > i), \
            f"Cannot access {i}-th cost specification"
        if isinstance(self._cost_specification, list):
            return self._evaluate_ith_cost_metric(i, self._cost_specification[i])
        else:
            return self._evaluate_ith_cost_metric(0, self._cost_specification)

    def _evaluate_ith_cost_metric(self, cost_idx: int, cost_spec: CostSpec) -> torch.Tensor:
        cost = torch.tensor(0, dtype=torch.float32)
        target_list = self._unique_leaf_modules if cost_spec.shared else self._leaf_modules
        for name, node, layer in target_list:
            if isinstance(layer, PITModule):
                v = layer.get_modified_vars()
                v.update(shapes_dict(node))
                cost = cost + self.cost_fn_maps[cost_idx][name](v)
            elif self.full_cost:
                v = vars(layer)
                v.update(shapes_dict(node))
                cost = cost + self.cost_fn_maps[cost_idx][name](vars(layer))
        return cost

    @property
    def discrete_cost(self) -> bool:
        return self._discrete_cost

    @discrete_cost.setter
    def discrete_cost(self, value: bool):
        for _, _, layer in self._unique_leaf_modules:
            if hasattr(layer, 'discrete_cost'):
                layer.discrete_cost = value
        self._discrete_cost = value

    @property
    def train_features(self) -> bool:
        """Returns True if PIT is training the output features masks

        :return: True if PIT is training the output features masks
        :rtype: bool
        """
        return self._train_features

    @train_features.setter
    def train_features(self, value: bool):
        """Set to True to let PIT train the output features masks

        :param value: set to True to let PIT train the output features masks
        :type value: bool
        """
        for _, _, layer in self._leaf_modules:
            if hasattr(layer, 'train_features'):
                layer.train_features = value
        self._train_features = value

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
        for _, _, layer in self._leaf_modules:
            if hasattr(layer, 'train_rf'):
                layer.train_rf = value
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
        for _, _, layer in self._leaf_modules:
            if hasattr(layer, 'train_dilation'):
                layer.train_dilation = value
        self._train_dilation = value

    def arch_export(self, add_bn=True):
        """Export the architecture found by the NAS as a `nn.Module`

        The returned model will have the trained weights found during the search filled in, but
        should be fine-tuned for optimal results.

        :param add_bn: determines if BatchNorm layers that have been fused with PITLayers
        in order to make the channel masking work are re-added to the exported model. If set to
        True, the model MUST be fine-tuned.
        :type add_bn: bool
        :return: the architecture found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        if not add_bn:
            for _, _, layer in self._leaf_modules:
                if isinstance(layer, PITModule) and hasattr(layer, 'following_bn_args'):
                    layer.following_bn_args = None

        mod, _ = convert(self.seed, self._input_example, 'export')

        return mod

    def arch_summary(self) -> Dict[str, Dict[str, Any]]:
        """Generates a dictionary representation of the architecture found by the NAS.
        Only optimized layers are reported

        :return: a dictionary representation of the architecture found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        arch = {}
        for name, layer in self.seed.named_modules():
            if isinstance(layer, PITModule):
                arch[name] = layer.summary()
                arch[name]['type'] = layer.__class__.__name__
        return arch

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of the NAS, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API, but PITLayers never have sub-layers
        :type recurse: bool
        :return: an iterator over the architectural parameters of the NAS
        :rtype: Iterator[nn.Parameter]
        """
        included = set()
        for lname, _, layer in self._unique_leaf_modules:
            if isinstance(layer, PITModule):
                layer = cast(PITModule, layer)
                prfx = prefix
                prfx += "." if len(prefix) > 0 else ""
                prfx += lname
                for name, param in layer.named_nas_parameters(prefix=lname, recurse=recurse):
                    # avoid duplicates (e.g. shared channels masks)
                    if param not in included:
                        included.add(param)
                        yield name, param

    def named_net_parameters(
            self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the inner network parameters, EXCEPT the NAS architectural
        parameters, yielding both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API, not actually used
        :type recurse: bool
        :return: an iterator over the inner network parameters
        :rtype: Iterator[nn.Parameter]
        """
        exclude = set(_[0] for _ in self.named_nas_parameters())
        for name, param in self.named_parameters():
            if name not in exclude:
                yield name, param

    def _uniquify_leaf_modules(self) -> List[Tuple[str, fx.Node, nn.Module]]:
        names = set()
        unique_modules = []
        for name, node, layer in self._leaf_modules:
            if name not in names:
                names.add(name)
                unique_modules.append((name, node, layer))
        return unique_modules

    # TODO: this could be made general for all DNAS?
    def _create_cost_fn_maps(self) -> List[Dict[str, Callable[[PatternSpec], torch.Tensor]]]:
        cost_fn_maps = []
        if isinstance(self._cost_specification, list):
            for c in self._cost_specification:
                cost_fn_maps.append(self._single_cost_fn_map(c))
        else:
            cost_fn_maps.append(self._single_cost_fn_map(self._cost_specification))
        return cost_fn_maps

    def _single_cost_fn_map(self, c: CostSpec) -> Dict[str, Callable[[PatternSpec], torch.Tensor]]:
        cost_fn_map = {}
        for name, layer in self.seed.named_modules():
            if isinstance(layer, PITModule):
                # get original layer type from PITModule type
                # TODO: make this more readable
                t = list(pit_layer_map.keys())[list(pit_layer_map.values()).index(type(layer))]
                # equally unreadable alternative
                # t = layer.__class__.__bases__[0]
            else:
                t = type(layer)
            cost_fn_map[name] = c[(t, vars(layer))]
        return cost_fn_map

    def _resolve_input_example(self, example, shape):
        """Document here"""
        if example is None and shape is None:
            msg = 'One of `input_example` and `input_shape` must be different than None'
            raise ValueError(msg)
        if example is not None and shape is not None:
            msg = ('Warning: you specified both `input_example` and `input_shape`.'
                   'The first will be considered for shape propagation')
            warn(msg)
        if example is not None:
            return example
        if shape is not None:
            try:
                # create a "fake" minibatch of 1 input for shape prop
                example = torch.stack([torch.rand(shape)] * 1, 0).to(self._device)
                return example
            except TypeError:
                msg = ('If the provided `input_shape` is not a simple tuple '
                       'the user should pass instead an `input_example`.')
                raise TypeError(msg)

    def __str__(self):
        """Prints the architecture found by the NAS to screen

        :return: a str representation of the current architecture
        :rtype: str
        """
        arch = self.arch_summary()
        return str(arch)
