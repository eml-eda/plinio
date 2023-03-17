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
from typing import Dict, Tuple, Type, cast, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quant.quantizers import Quantizer
from .argmaxer import STEArgmax


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
    :param gumbel_softmax: use Gumbel SoftMax for sampling, instead of a normal SoftMax
    :type gumbel_softmax: bool, optional
    :param hard_softmax: use hard Gumbel SoftMax sampling (only applies when gumbel_softmax = True)
    :type hard_softmax: bool, optional
    :param disable_sampling: whether to disable the sampling of the architectural coefficients
    during the forward pass
    :type disable_sampling: bool, optional
    """
    def __init__(self,
                 precisions: Tuple[int, ...],
                 cout: int,
                 quantizer: Type[Quantizer],
                 quantizer_kwargs: Dict = {},
                 gumbel_softmax: bool = False,
                 hard_softmax: bool = False,
                 disable_sampling: bool = False):
        super(MixPrec_Qtz_Channel, self).__init__()
        if len(precisions) != len(set(precisions)):
            raise ValueError("Precisions cannot be repeated")
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
        self.register_buffer('temperature', torch.tensor(1., dtype=torch.float32))
        self.temperature = cast(torch.Tensor, self.temperature)
        self.update_softmax_options(gumbel_softmax, hard_softmax, disable_sampling)
        self.register_buffer('theta_alpha', torch.tensor(len(precisions), dtype=torch.float32))
        with torch.no_grad():
            self.theta_alpha.data = self.alpha_prec
            self.sample_alpha()
        self.register_buffer('out_features_eff', torch.tensor(self.cout, dtype=torch.float32))
        self.zero_index = None
        for i, p in enumerate(self.precisions):
            if p == 0:
                self.zero_index = i
                break

    @property
    def features_mask(self) -> torch.Tensor:
        """Return the binarized mask that specifies which output features (channels) are kept by
        the NAS

        :return: the binarized mask over the features axis
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            if self.zero_index is not None:
                # extract the row corresponding to 0-bit precision from the one-hot argmax matrix
                # of size (n_prec, cout), and do a 1s complement
                return 1 - STEArgmax.apply(self.theta_alpha)[self.zero_index, :]
            else:
                return torch.ones((self.cout,))

    def sample_alpha_sm(self):
        """
        Samples the alpha coefficients using a standard SoftMax (with temperature).
        The corresponding normalized parameters (summing to 1) are stored in the theta_alpha buffer.
        """
        self.theta_alpha = F.softmax(self.alpha_prec / self.temperature.item(), dim=0)
        if self.hard_softmax:
            self.theta_alpha = STEArgmax.apply(self.theta_alpha)

    def sample_alpha_gs(self):
        """
        Samples the alpha architectural coefficients using a Gumbel SoftMax (with temperature).
        The corresponding normalized parameters (summing to 1) are stored in the theta_alpha buffer.
        """
        if self.training:
            self.theta_alpha = F.gumbel_softmax(
                logits=self.alpha_prec,
                tau=self.temperature.item(),
                hard=self.hard_softmax,
                dim=0)
        else:
            self.sample_alpha_sm()

    def sample_alpha_none(self):
        """Sample the previous alpha architectural coefficients. Used to change the alpha
        coefficients at each iteration"""
        return

    def update_softmax_options(self, gumbel_softmax, hard_softmax, disable_sampling):
        """Set the flags to choose between the softmax, the hard and soft Gumbel-softmax
        and the sampling disabling of the architectural coefficients in the quantizers

        :param gumbel_softmax: whether to use the Gumbel-softmax instead of the softmax
        :type gumbel_softmax: bool
        :param hard_softmax: whether to use the hard version of the Gumbel-softmax
        (param gumbel_softmax must be equal to True)
        :type gumbel_softmax: bool
        :param disable_sampling: whether to disable the sampling of the architectural
        coefficients in the forward pass
        :type disable_sampling: bool
        """
        self.hard_softmax = hard_softmax
        if disable_sampling:
            self.sample_alpha = self.sample_alpha_none
        elif gumbel_softmax:
            self.sample_alpha = self.sample_alpha_gs
        else:
            self.sample_alpha = self.sample_alpha_sm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the searchable mixed-precision layer.

        In a nutshell, it computes the different quantized representations of `mix_qtz`
        and combines them weighting the different terms channel-wise by means of
        softmax-ed `alpha_prec` trainable parameters.

        :param input: the input float tensor
        :type input: torch.Tensor
        :return: the output fake-quantized with searchable precision tensor
        :rtype: torch.Tensor
        """
        self.sample_alpha()

        y = []
        for i, quantizer in enumerate(self.mix_qtz):
            theta_alpha_i = self.theta_alpha[i].view((self.cout,) + (1,) * len(input.shape[1:]))
            y.append(theta_alpha_i * quantizer(input))
        y = torch.stack(y, dim=0).sum(dim=0)

        # compute number of not pruned channels
        if self.zero_index is not None:
            self.out_features_eff = self.cout - torch.sum(self.theta_alpha[self.zero_index, :])
        else:
            self.out_features_eff = torch.tensor(self.cout, dtype=torch.float32)

        return y

    @property
    def effective_precision(self) -> torch.Tensor:
        """Return each channel effective precision as the average precision weighted by
        softmax-ed `alpha_prec` parameters

        :return: the effective precision
        :rtype: torch.Tensor
        """
        device = self.theta_alpha.device
        p_tensor = torch.tensor(self.precisions, device=device)
        eff_prec = (self.theta_alpha.sum(dim=1) * p_tensor).sum() / self.cout  # TODO: Check
        return eff_prec


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
    :param gumbel_softmax: use Gumbel SoftMax for sampling, instead of a normal SoftMax
    :type gumbel_softmax: bool, optional
    :param hard_softmax: use hard Gumbel SoftMax sampling (only applies when gumbel_softmax = True)
    :type hard_softmax: bool, optional
    :param disable_sampling: whether to disable the sampling of the architectural coefficients
    during the forward pass
    :type disable_sampling: bool, optional
    """
    def __init__(self,
                 precisions: Tuple[int, ...],
                 quantizer: Type[Quantizer],
                 quantizer_kwargs: Dict = {},
                 gumbel_softmax: bool = False,
                 hard_softmax: bool = False,
                 disable_sampling: bool = False):
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
        self.register_buffer('temperature', torch.tensor(1.))
        self.temperature = cast(torch.Tensor, self.temperature)
        self.update_softmax_options(gumbel_softmax, hard_softmax, disable_sampling)
        self.register_buffer('theta_alpha', torch.tensor(len(precisions), dtype=torch.float32))
        with torch.no_grad():
            self.theta_alpha.data = self.alpha_prec
            self.sample_alpha()

    def sample_alpha_sm(self):
        """
        Samples the alpha coefficients using a standard SoftMax (with temperature).
        The corresponding normalized parameters (summing to 1) are stored in the theta_alpha buffer.
        """
        self.theta_alpha = F.softmax(self.alpha_prec / self.temperature.item(), dim=0)
        if self.hard_softmax:
            self.theta_alpha = STEArgmax.apply(self.theta_alpha)

    def sample_alpha_gs(self):
        """
        Samples the alpha architectural coefficients using a Gumbel SoftMax (with temperature).
        The corresponding normalized parameters (summing to 1) are stored in the theta_alpha buffer.
        """
        if self.training:
            self.theta_alpha = F.gumbel_softmax(
                logits=self.alpha_prec,
                tau=self.temperature.item(),
                hard=self.hard_softmax,
                dim=0)
        else:
            self.sample_alpha_sm()

    def sample_alpha_none(self):
        """Sample the previous alpha architectural coefficients. Used to change the alpha
        coefficients at each iteration"""
        return

    def update_softmax_options(self, gumbel_softmax, hard_softmax, disable_sampling):
        """Set the flags to choose between the softmax, the hard and soft Gumbel-softmax
        and the sampling disabling of the architectural coefficients in the quantizers

        :param gumbel_softmax: whether to use the Gumbel-softmax instead of the softmax
        :type gumbel_softmax: bool
        :param hard_softmax: whether to use the hard version of the Gumbel-softmax
        (param gumbel_softmax must be equal to True)
        :type gumbel_softmax: bool
        :param disable_sampling: whether to disable the sampling of the architectural
        coefficients in the forward pass
        :type disable_sampling: bool
        """
        self.hard_softmax = hard_softmax
        if disable_sampling:
            self.sample_alpha = self.sample_alpha_none
        elif gumbel_softmax:
            self.sample_alpha = self.sample_alpha_gs
        else:
            self.sample_alpha = self.sample_alpha_sm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the searchable mixed-precision layer.

        In a nutshell, it computes the different quantized representations of `mix_qtz`
        and combines them weighting the different terms by means of softmax-ed
        `alpha_prec` trainable parameters.

        :param input: the input float tensor
        :type input: torch.Tensor
        :return: the output fake-quantized with searchable precision tensor
        :rtype: torch.Tensor
        """
        self.sample_alpha()

        y = []
        for i, quantizer in enumerate(self.mix_qtz):
            y.append(self.theta_alpha[i] * quantizer(input))
        y = torch.stack(y, dim=0).sum(dim=0)
        return y

    @property
    def effective_precision(self) -> torch.Tensor:
        """Return the effective precision as the average precision weighted by
        softmax-ed `alpha_prec` parameters

        :return: the effective precision
        :rtype: torch.Tensor
        """
        device = self.theta_alpha.device
        p_tensor = torch.tensor(self.precisions, device=device)
        eff_prec = (self.theta_alpha * p_tensor).sum()
        return eff_prec


class MixPrec_Qtz_Layer_Bias(nn.Module):
    """A nn.Module implementing mixed-precision quantization searchable
    operation of the bias vector provided as input.
    This module includes trainable NAS parameters which are shared with the
    `mixprec_w_quantizer` in order to select the proper corresponding quantizer.

    :param quantizer: bias quantizer
    :type quantizer: Quantizer
    :param cout: number of output channels
    :type cout: int
    :param mixprec_w_quantizer: mixprec weight quantizer, it is used to gather info
    about weight scale factor
    :type mixprec_w_quantizer: MixPrec_Qtz_Layer
    :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
    :type quantizer_kwargs: Dict, optional
    :param mixprec_a_quantizer: mixprec activation quantizer, it is used to gather info
    about act scale factor. Optional argument, is used only if the user defines
    the network placing MixPrec modules manually.
    :type mixprec_a_quantizer: Optional[MixPrec_Qtz_Layer]
    """
    def __init__(self,
                 quantizer: Type[Quantizer],
                 cout: int,
                 mixprec_w_quantizer: MixPrec_Qtz_Layer,
                 quantizer_kwargs: Dict = {},
                 mixprec_a_quantizer: Optional[MixPrec_Qtz_Layer] =
                 cast(MixPrec_Qtz_Layer, nn.Identity())):
        super(MixPrec_Qtz_Layer_Bias, self).__init__()
        self.quantizer = quantizer
        self.cout = cout
        # This will be eventually overwritten later when we process the model graph
        self.mixprec_a_quantizer = cast(MixPrec_Qtz_Layer, mixprec_a_quantizer)
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
        sm_alpha = self.mixprec_a_quantizer.theta_alpha
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
        sm_alpha = self.mixprec_w_quantizer.theta_alpha
        for i, qtz in enumerate(self.mixprec_w_quantizer.mix_qtz):
            s_w = s_w + (sm_alpha[i] * qtz.s_w)
        return s_w


class MixPrec_Qtz_Channel_Bias(nn.Module):
    """A nn.Module implementing mixed-precision quantization searchable
    operation of each channel of bias vector provided as input.
    This module includes trainable NAS parameters.

    :param quantizer: bias quantizer
    :type quantizer: Quantizer
    :param cout: number of output channels
    :type cout: int
    :param mixprec_w_quantizer: mixprec weight quantizer, it is used to gather info
    about weight scale factor
    :type mixprec_w_quantizer: MixPrec_Qtz_Channel
    :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
    :type quantizer_kwargs: Dict, optional
    :param mixprec_a_quantizer: mixprec activation quantizer, it is used to gather info
    about act scale factor. Optional argument, is used only if the user defines
    the network placing MixPrec modules manually.
    :type mixprec_a_quantizer: Optional[MixPrec_Qtz_Layer]
    """
    def __init__(self,
                 quantizer: Type[Quantizer],
                 cout: int,
                 mixprec_w_quantizer: MixPrec_Qtz_Channel,
                 quantizer_kwargs: Dict = {},
                 mixprec_a_quantizer: Optional[MixPrec_Qtz_Layer] =
                 cast(MixPrec_Qtz_Layer, nn.Identity())):
        super(MixPrec_Qtz_Channel_Bias, self).__init__()
        self.quantizer = quantizer
        self.cout = cout
        # This will be overwritten later when we process the model graph
        self.mixprec_a_quantizer = cast(MixPrec_Qtz_Layer, mixprec_a_quantizer)
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
        sm_alpha = self.mixprec_a_quantizer.theta_alpha
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
        sm_alpha = self.mixprec_w_quantizer.theta_alpha
        for i, qtz in enumerate(self.mixprec_w_quantizer.mix_qtz):
            s_w = s_w + (sm_alpha[i] * qtz.s_w)
        return s_w
