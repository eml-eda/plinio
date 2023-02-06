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

from enum import Enum, auto
from typing import Dict, Tuple, Type, cast
import torch
import torch.nn as nn
from ..quant.quantizers import Quantizer


class MixPrecType(Enum):
    PER_CHANNEL = auto()
    PER_LAYER = auto()


class MixPrec_Qtz_Channel(nn.Module):
    """A nn.Module implementing a generic mixed-precision quantization searchable
    operation of each channel of tensor provided as input.
    This module includes trainable NAS parameters.

    :param precisions: different bitwitdth alternatives among which perform search
    :type precisions: Tuple[int, ...]
    :param cout: number of output channels
    :type cout: int
    :param quantizer: input quantizer
    :type quantizer: Quantizer
    :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
    :type quantizer_kwargs: Dict, optional
    """
    def __init__(self,
                 precisions: Tuple[int, ...],
                 cout: int,
                 quantizer: Type[Quantizer],
                 quantizer_kwargs: Dict = {}):
        super(MixPrec_Qtz_Channel, self).__init__()
        self.precisions = precisions
        self.cout = cout
        self.quantizer = quantizer
        self.quantizer_kwargs = quantizer_kwargs
        # NAS parameters
        self.alpha_prec = nn.Parameter(torch.Tensor(len(precisions), cout))
        self.alpha_prec.data.fill_(1.)  # Initially each precision is equiprobable
        # Mixed-Precision Quantizers
        self.mix_qtz = nn.ModuleList()
        for p in precisions:
            qtz = quantizer(p, **quantizer_kwargs)  # type: ignore
            qtz = cast(nn.Module, qtz)
            self.mix_qtz.append(qtz)
        # Init temperature to std value
        self.temperature = 1.

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the searchable mixed-precision layer.

        In a nutshell, computes the different quantized representations of `mix_qtz`
        and combine them weighting the different terms channel-wise by means of
        softmax-ed `alpha_prec` trainable parameters.

        :param input: the input float tensor
        :type input: torch.Tensor
        :return: the output fake-quantized with searchable precision tensor
        :rtype: torch.Tensor
        """
        soft_alpha = nn.functional.softmax(self.alpha_prec / self.temperature,
                                           dim=0)
        y = []
        for i, quantizer in enumerate(self.mix_qtz):
            soft_alpha_i = soft_alpha[i].view((self.cout,) + (1,) * len(input.shape[1:]))
            y.append(soft_alpha_i * quantizer(input))
        y = torch.stack(y, dim=0).sum(dim=0)
        return y

    @property
    def effective_precision(self) -> torch.Tensor:
        """Return each channel effective precision as the average precision weighted by
        softmax-ed `alpha_prec` parameters

        :return: the effective precision
        :rtype: torch.Tensor
        """
        soft_alpha = nn.functional.softmax(self.alpha_prec / self.temperature,
                                           dim=0)
        device = self.alpha_prec.device
        p_tensor = torch.Tensor(self.precisions, device=device)
        eff_prec = (soft_alpha.sum(dim=1) * p_tensor).sum() / self.cout  # TODO: Check
        return eff_prec

    @property
    def temperature(self) -> float:
        """Returns the actual softmax temperature for this layer.

        :return: the actual softmax temperature for this layer.
        :rtype: float
        """
        return self._temperature

    @temperature.setter
    def temperature(self, tau: float):
        """Set the softmax temperature for this layer.

        :param tau: the softmax temperature to be setted
        :type tau: float
        """
        self._temperature = tau


class MixPrec_Qtz_Layer(nn.Module):
    """A nn.Module implementing a generic mixed-precision quantization searchable
    operation of the tensor provided as input.
    This module includes trainable NAS parameters.

    :param precisions: different bitwitdth alternatives among which perform search
    :type precisions: Tuple[int, ...]
    :param quantizer: input quantizer
    :type quantizer: Quantizer
    :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
    :type quantizer_kwargs: Dict, optional
    """
    def __init__(self,
                 precisions: Tuple[int, ...],
                 quantizer: Type[Quantizer],
                 quantizer_kwargs: Dict = {}):
        super(MixPrec_Qtz_Layer, self).__init__()
        self.precisions = precisions
        self.quantizer = quantizer
        self.quantizer_kwargs = quantizer_kwargs
        # NAS parameters
        self.alpha_prec = nn.Parameter(torch.Tensor(len(precisions)))
        self.alpha_prec.data.fill_(1.)  # Initially each precision is equiprobable
        # Mixed-Precision Quantizers
        self.mix_qtz = nn.ModuleList()
        for p in precisions:
            qtz = quantizer(p, **quantizer_kwargs)  # type: ignore
            qtz = cast(nn.Module, qtz)
            self.mix_qtz.append(qtz)
        # Init temperature to std value
        self.temperature = 1.

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the searchable mixed-precision layer.

        In a nutshell, computes the different quantized representations of `mix_qtz`
        and combine them weighting the different terms by means of softmax-ed
        `alpha_prec` trainable parameters.

        :param input: the input float tensor
        :type input: torch.Tensor
        :return: the output fake-quantized with searchable precision tensor
        :rtype: torch.Tensor
        """
        soft_alpha = nn.functional.softmax(self.alpha_prec / self.temperature,
                                           dim=0)
        y = []
        for i, quantizer in enumerate(self.mix_qtz):
            y.append(soft_alpha[i] * quantizer(input))
        y = torch.stack(y, dim=0).sum(dim=0)
        return y

    @property
    def effective_precision(self) -> torch.Tensor:
        """Return the effective precision as the average precision weighted by
        softmax-ed `alpha_prec` parameters

        :return: the effective precision
        :rtype: torch.Tensor
        """
        soft_alpha = nn.functional.softmax(self.alpha_prec / self.temperature,
                                           dim=0)
        device = self.alpha_prec.device
        p_tensor = torch.tensor(self.precisions, device=device)
        eff_prec = (soft_alpha * p_tensor).sum()
        return eff_prec

    @property
    def temperature(self) -> float:
        """Returns the actual softmax temperature for this layer.

        :return: the actual softmax temperature for this layer.
        :rtype: float
        """
        return self._temperature

    @temperature.setter
    def temperature(self, tau: float):
        """Set the softmax temperature for this layer.

        :param tau: the softmax temperature to be setted
        :type tau: float
        """
        self._temperature = tau


class MixPrec_Qtz_Layer_Bias(nn.Module):
    """A nn.Module implementing mixed-precision quantization searchable
    operation of the bias vector provided as input.
    This module includes trainable NAS parameters which are shared with the
    `mixprec_w_quantizer` in order to select the proper corresponding quantizer.

    :param quantizer: input quantizer
    :type quantizer: Quantizer
    :param cout: number of output channels
    :type cout: int
    :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
    :type quantizer_kwargs: Dict, optional
    """
    def __init__(self,
                 quantizer: Type[Quantizer],
                 cout: int,
                 mixprec_a_quantizer: MixPrec_Qtz_Layer,
                 mixprec_w_quantizer: MixPrec_Qtz_Layer,
                 quantizer_kwargs: Dict = {}):
        super(MixPrec_Qtz_Layer_Bias, self).__init__()
        self.quantizer = quantizer
        self.cout = cout
        self.mixprec_a_quantizer = mixprec_a_quantizer
        self.mixprec_w_quantizer = mixprec_w_quantizer
        self.quantizer_kwargs = quantizer_kwargs

        # Build bias quantizer
        self.quantizer_kwargs['cout'] = cout
        self.qtz_func = quantizer(**self.quantizer_kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the searchable mixed-precision layer.

        In a nutshell, computes the different quantized representations of `mix_qtz`
        and combine them weighting the different terms by means of softmax-ed
        `alpha_prec` trainable parameters.

        :param input: the input float tensor
        :type input: torch.Tensor
        :return: the output fake-quantized with searchable precision tensor
        :rtype: torch.Tensor
        """
        self.qtz_func = cast(nn.Module, self.qtz_func)
        y = self.qtz_func(input, self.s_a, self.s_w)
        return y

    @property
    def s_a(self) -> torch.Tensor:
        """Return the aggregated act scale factor
        :return: the scale factor
        :rtype: torch.Tensor
        """
        s_a = torch.tensor(0, dtype=torch.float32)
        temp = self.mixprec_a_quantizer.temperature
        alpha_prec = self.mixprec_a_quantizer.alpha_prec
        sm_alpha = nn.functional.softmax(alpha_prec / temp, dim=0)
        for i, qtz in enumerate(self.mixprec_a_quantizer.mix_qtz):
            s_a = s_a + (sm_alpha[i] * qtz.s_a)
        return s_a

    @property
    def s_w(self) -> torch.Tensor:
        """Return the aggregated weight scale factor
        :return: the scale factor
        :rtype: torch.Tensor
        """
        s_w = torch.tensor(0, dtype=torch.float32)
        temp = self.mixprec_w_quantizer.temperature
        alpha_prec = self.mixprec_w_quantizer.alpha_prec
        sm_alpha = nn.functional.softmax(alpha_prec / temp, dim=0)
        for i, qtz in enumerate(self.mixprec_w_quantizer.mix_qtz):
            s_w = s_w + (sm_alpha[i] * qtz.s_w)
        return s_w


class MixPrec_Qtz_Channel_Bias(nn.Module):
    """A nn.Module implementing mixed-precision quantization searchable
    operation of each channel of bias vector provided as input.
    This module includes trainable NAS parameters.

    :param cout: number of output channels
    :type cout: int
    :param quantizer: input quantizer
    :type quantizer: Quantizer
    :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
    :type quantizer_kwargs: Dict, optional
    """
    def __init__(self,
                 quantizer: Type[Quantizer],
                 cout: int,
                 mixprec_a_quantizer: MixPrec_Qtz_Layer,
                 mixprec_w_quantizer: MixPrec_Qtz_Channel,
                 quantizer_kwargs: Dict = {}):
        super(MixPrec_Qtz_Channel_Bias, self).__init__()
        self.quantizer = quantizer
        self.cout = cout
        self.mixprec_a_quantizer = mixprec_a_quantizer
        self.mixprec_w_quantizer = mixprec_w_quantizer
        self.quantizer_kwargs = quantizer_kwargs

        # Build bias quantizer
        self.quantizer_kwargs['cout'] = cout
        self.qtz_func = quantizer(**self.quantizer_kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the searchable mixed-precision layer.

        In a nutshell, computes the different quantized representations of `mix_qtz`
        and combine them weighting the different terms by means of softmax-ed
        `alpha_prec` trainable parameters.

        :param input: the input float tensor
        :type input: torch.Tensor
        :return: the output fake-quantized with searchable precision tensor
        :rtype: torch.Tensor
        """
        self.qtz_func = cast(nn.Module, self.qtz_func)
        y = self.qtz_func(input, self.s_a, self.s_w)
        return y

    @property
    def s_a(self) -> torch.Tensor:
        """Return the aggregated act scale factor
        :return: the scale factor
        :rtype: torch.Tensor
        """
        s_a = torch.tensor(0, dtype=torch.float32)
        temp = self.mixprec_a_quantizer.temperature
        alpha_prec = self.mixprec_a_quantizer.alpha_prec
        sm_alpha = nn.functional.softmax(alpha_prec / temp, dim=0)
        for i, qtz in enumerate(self.mixprec_a_quantizer.mix_qtz):
            s_a = s_a + (sm_alpha[i] * qtz.s_a)
        return s_a

    @property
    def s_w(self) -> torch.Tensor:
        """Return the aggregated weight scale factor
        :return: the scale factor
        :rtype: torch.Tensor
        """
        s_w = torch.tensor(0, dtype=torch.float32)
        temp = self.mixprec_w_quantizer.temperature
        alpha_prec = self.mixprec_w_quantizer.alpha_prec
        sm_alpha = nn.functional.softmax(alpha_prec / temp, dim=0)
        for i, qtz in enumerate(self.mixprec_w_quantizer.mix_qtz):
            s_w = s_w + (sm_alpha[i] * qtz.s_w)
        return s_w
