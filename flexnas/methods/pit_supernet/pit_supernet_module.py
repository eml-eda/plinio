from typing import Iterable
import torch
import torch.nn as nn
from .pit_supernet_combiner import PITSuperNetCombiner


class PITSuperNetModule(nn.Module):

    def __init__(self, input_layers: Iterable[nn.Module]):
        super(PITSuperNetModule, self).__init__()

        self._input_layers = nn.ModuleList(list(input_layers))
        self.input_shape = None
        self.combiner = PITSuperNetCombiner(self._input_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        layers_outputs = [layer(input) for layer in self._input_layers]
        return self.combiner(layers_outputs)

    def __getitem__(self, pos: int) -> nn.Module:
        """Get the layer at position pos in the list of all the possible
        layers for the SuperNetModule

        :param pos: position of the required layer in the list input_layers
        :type pos: int
        :return: layer at postion pos in the list input_layers
        :rtype: nn.Module
        """
        return self._input_layers[pos]
