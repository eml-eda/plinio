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
# * Author:  Beatrice Alessandra Motetti <beatrice.motetti@polito.it>          *
# *----------------------------------------------------------------------------*
import torch

from . import CostSpec
from .mpic_latency import _mpic_latency_conv1d_generic, _mpic_latency_conv2d_generic, \
        _mpic_latency_conv1d_dw, _mpic_latency_conv2d_dw, _mpic_latency_linear
from .pattern import Conv1dGeneric, Conv2dGeneric, LinearGeneric, \
        Conv1dDW, Conv2dDW


def _energy_from_cycles_mpic(cycles):
    """Compute the energy consumption according to the MPIC model.

    Parameters
    ----------
    - cycles: number of cycles

    Output
    ------
    - energy consumption in J"""

    _FREQUENCY = 250 * 1e+6
    _MEAN_POWER = torch.tensor([5.30, 5.39, 5.46, 5.38]).mean() * 1e-3

    return (cycles / _FREQUENCY) * _MEAN_POWER


def _mpic_energy_conv1d_generic(spec):
    mpic_latency = _mpic_latency_conv1d_generic(spec)
    cost = _energy_from_cycles_mpic(mpic_latency)
    return cost


def _mpic_energy_conv2d_generic(spec):
    mpic_latency = _mpic_latency_conv2d_generic(spec)
    cost = _energy_from_cycles_mpic(mpic_latency)
    return cost


def _mpic_energy_conv1d_dw(spec):
    mpic_latency = _mpic_latency_conv1d_dw(spec)
    cost = _energy_from_cycles_mpic(mpic_latency)
    return cost


def _mpic_energy_conv2d_dw(spec):
    mpic_latency = _mpic_latency_conv2d_dw(spec)
    cost = _energy_from_cycles_mpic(mpic_latency)
    return cost


def _mpic_energy_linear(spec):
    mpic_latency = _mpic_latency_linear(spec)
    cost = _energy_from_cycles_mpic(mpic_latency)
    return cost


mpic_energy = CostSpec(shared=False, default_behavior='zero')
mpic_energy[Conv1dGeneric] = _mpic_energy_conv1d_generic
mpic_energy[Conv2dGeneric] = _mpic_energy_conv2d_generic
mpic_energy[Conv1dDW] = _mpic_energy_conv1d_dw
mpic_energy[Conv2dDW] = _mpic_energy_conv2d_dw
mpic_energy[LinearGeneric] = _mpic_energy_linear
