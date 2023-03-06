from typing import Iterable
import torch
import torch.nn as nn
from .combiner import PITSuperNetCombiner


class PITSuperNetModule(nn.Module):
    """A nn.Module containing some different layer alternatives.
    One of these layers will be selected by the PITSuperNet NAS tool for the current layer.
    This module is for the most part just a placeholder and its logic is contained inside
    the PITSuperNetCombiner of which instance is defined in the constructor.

    :param input_layers: iterable of possible alternative layers to be selected
    :type input_layers: Iterable[nn.Module]
    """
    def __init__(self,
                 input_layers: Iterable[nn.Module],
                 gumbel_softmax: bool = False,
                 hard_softmax: bool = False):
        super(PITSuperNetModule, self).__init__()
        self.sn_input_layers = nn.ModuleList(list(input_layers))
        self.sn_combiner = PITSuperNetCombiner(self.sn_input_layers, gumbel_softmax, hard_softmax)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward function for the PITSuperNetModule that returns a weighted
        sum of all the outputs of the different input layers.
        It computes all possible layer outputs and passes the list to the sn_combiner
        which computes the weighted sum.

        :param input: the input tensor
        :type input: torch.Tensor
        :return: the output tensor (weighted sum of all layers output)
        :rtype: torch.Tensor
        """
        layers_outputs = [layer(input) for layer in self.sn_input_layers]
        return self.sn_combiner(layers_outputs)

    def __getitem__(self, pos: int) -> nn.Module:
        """Get the layer at position pos in the list of all the possible
        layers for the PITSuperNetModule

        :param pos: position of the required module in the list input_layers
        :type pos: int
        :return: module at postion pos in the list input_layers
        :rtype: nn.Module
        """
        return self.sn_input_layers[pos]
